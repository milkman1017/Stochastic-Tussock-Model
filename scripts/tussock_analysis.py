import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Globals / constants (match your C++ model)
# ============================================================

SLA_CM2_PER_G = 98.0
ROOT_TISSUE_DENSITY_G_CM3 = 0.21

ROOT_LENGTH_CM = 50.0
MM_TO_CM = 0.1

LEAF_NECRO_REMAIN_FRAC = 0.75  # dead_leaf_area *= 0.75 each year in model
G_TO_KG = 1e-3


def per_root_cone_volume_cm3(diam_mm: float) -> float:
    """Matches Tiller::perRootConeVolumeCm3 in the C++ model."""
    if np.isnan(diam_mm) or diam_mm <= 0:
        return 0.0
    r_cm = (diam_mm * MM_TO_CM) * 0.5
    return (1.0 / 3.0) * np.pi * (r_cm ** 2) * ROOT_LENGTH_CM


def live_root_volume_cm3(num_roots: float, diam_mm: float) -> float:
    if np.isnan(num_roots) or np.isnan(diam_mm):
        return float("nan")
    if num_roots <= 0:
        return 0.0
    return float(num_roots) * per_root_cone_volume_cm3(float(diam_mm))


# ============================================================
# Config / CLI
# ============================================================

@dataclass(frozen=True)
class Config:
    input_dir: Path
    output_dir: Path
    n_timepoints: int
    seed: int
    pattern: str
    use_only_alive_for_radius: bool
    radius_stat: str  # "max" or "p95"


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Sample timesteps from sim CSVs and generate plots (PNGs only).")
    p.add_argument("-i", "--input-dir", required=True, type=Path)
    p.add_argument("-o", "--output-dir", required=True, type=Path)
    p.add_argument("--n-timepoints", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pattern", type=str, default="*.csv")
    p.add_argument("--include-dead", action="store_true",
                   help="If set, compute tussock radius using all tillers, not just alive.")
    p.add_argument("--radius-stat", choices=["max", "p95"], default="max")

    a = p.parse_args()
    if not a.input_dir.exists() or not a.input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found or not a directory: {a.input_dir}")
    a.output_dir.mkdir(parents=True, exist_ok=True)

    return Config(
        input_dir=a.input_dir,
        output_dir=a.output_dir,
        n_timepoints=a.n_timepoints,
        seed=a.seed,
        pattern=a.pattern,
        use_only_alive_for_radius=not a.include_dead,
        radius_stat=a.radius_stat,
    )


# ============================================================
# IO / Sampling
# ============================================================

def list_sim_csvs(input_dir: Path, pattern: str) -> list[Path]:
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern '{pattern}' in {input_dir}")
    return files


def read_sim_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def sample_timesteps(timesteps: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    timesteps = np.array(timesteps, dtype=int)
    timesteps = np.unique(timesteps)
    if timesteps.size == 0:
        return np.array([], dtype=int)
    k = min(n, timesteps.size)
    sampled = rng.choice(timesteps, size=k, replace=False)
    sampled.sort()
    return sampled


# ============================================================
# Radius (coords are cm)
# ============================================================

def radial_distance_xy_cm(sub: pd.DataFrame) -> np.ndarray:
    if "X" in sub.columns and "Y" in sub.columns:
        x = sub["X"].to_numpy(dtype=float)
        y = sub["Y"].to_numpy(dtype=float)
        return np.sqrt(x * x + y * y)
    if "X" in sub.columns and "Z" in sub.columns:
        x = sub["X"].to_numpy(dtype=float)
        z = sub["Z"].to_numpy(dtype=float)
        return np.sqrt(x * x + z * z)
    raise ValueError("Missing coordinate columns: need (X,Y) or (X,Z).")


def summarize_radius_cm(r_cm: np.ndarray, stat: str) -> float:
    if r_cm.size == 0:
        return float("nan")
    if stat == "max":
        return float(np.nanmax(r_cm))
    if stat == "p95":
        return float(np.nanpercentile(r_cm, 95))
    raise ValueError(f"Unknown stat: {stat}")


# ============================================================
# Derived columns (in-memory only)
# ============================================================

def add_live_root_mass_from_cone(df: pd.DataFrame) -> pd.DataFrame:
    """Add RootVolLive_cm3 and RootMassLive_g using the cone geometry from your C++ model."""
    df = df.copy()
    if {"NumRoots", "RootDiamMM"}.issubset(df.columns):
        vols = [
            live_root_volume_cm3(nr, dmm)
            for nr, dmm in zip(df["NumRoots"].astype(float), df["RootDiamMM"].astype(float))
        ]
        df["RootVolLive_cm3"] = np.array(vols, dtype=float)
        df["RootMassLive_g"] = df["RootVolLive_cm3"] * ROOT_TISSUE_DENSITY_G_CM3
    else:
        df["RootVolLive_cm3"] = np.nan
        df["RootMassLive_g"] = np.nan
    return df


# ============================================================
# Timestep summary + productivity + decomposition proxies
# ============================================================

def _pick_id_column(df: pd.DataFrame) -> str | None:
    """
    Try to find a stable tiller identifier column for dead-tiller (count) dynamics.
    Adjust/extend this list to match your sim output.
    """
    candidates = ["TillerID", "TillerId", "ID", "Id", "uid", "UID", "tiller_id", "tillerID"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build_timestep_summary(df: pd.DataFrame, use_only_alive_for_radius: bool, radius_stat: str) -> pd.DataFrame:
    required = {"TimeStep", "Status", "RootNecroMass", "RootNecroMassCum", "DeadLeafMass", "LeafArea"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    alive_counts = df.groupby("TimeStep")["Status"].apply(lambda s: int((s == 1).sum())).astype(float)

    radii = {}
    for t, sub in df.groupby("TimeStep"):
        sub_for_radius = sub[sub["Status"] == 1] if use_only_alive_for_radius else sub
        r_cm = radial_distance_xy_cm(sub_for_radius)
        radii[int(t)] = summarize_radius_cm(r_cm, radius_stat)

    total_root_necro_g = df.groupby("TimeStep")["RootNecroMass"].sum().astype(float)
    total_root_necro_cum_g = df.groupby("TimeStep")["RootNecroMassCum"].sum().astype(float)
    total_dead_leaf_mass_g = df.groupby("TimeStep")["DeadLeafMass"].sum().astype(float)

    # Total live leaf mass (g): sum LeafArea then divide by SLA
    total_leaf_mass_g = df.groupby("TimeStep")["LeafArea"].sum().astype(float) / SLA_CM2_PER_G

    if "RootMassLive_g" in df.columns:
        total_live_root_mass_g = df.groupby("TimeStep")["RootMassLive_g"].sum().astype(float)
    else:
        total_live_root_mass_g = pd.Series(index=alive_counts.index, data=np.nan, dtype=float)

    out = pd.DataFrame({
        "TimeStep": alive_counts.index.astype(int),
        "alive_tillers": alive_counts.values,
        "tussock_radius_cm": pd.Series(radii).reindex(alive_counts.index).values,
        "total_root_necromass_g": total_root_necro_g.reindex(alive_counts.index).values,
        "total_root_necromass_kg": (total_root_necro_g.reindex(alive_counts.index).values * G_TO_KG),
        "total_root_necromasscum_g": total_root_necro_cum_g.reindex(alive_counts.index).values,
        "total_dead_leaf_mass_g": total_dead_leaf_mass_g.reindex(alive_counts.index).values,
        "total_leaf_mass_g": total_leaf_mass_g.reindex(alive_counts.index).values,
        "total_live_root_mass_g": total_live_root_mass_g.reindex(alive_counts.index).values,
    }).sort_values("TimeStep")

    out["live_root_mass_per_tiller_g"] = out["total_live_root_mass_g"] / out["alive_tillers"].replace(0, np.nan)
    return out


def add_root_productivity(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Turnover-aware productivity proxy (g per tiller per year):
      RootProd(t) =
          max(0, LiveRoot_per_tiller(t) - LiveRoot_per_tiller(t-1))
          + NecromassProduced_per_tiller(t)
    with NecromassProduced(t) = diff(total RootNecroMassCum).
    """
    s = summary.copy()
    necro_prod_g = s["total_root_necromasscum_g"].diff().clip(lower=0).fillna(0.0)
    necro_prod_per_tiller_g = necro_prod_g / s["alive_tillers"].replace(0, np.nan)
    d_live_pos = s["live_root_mass_per_tiller_g"].diff().clip(lower=0)
    s["root_productivity_g_per_tiller_per_yr"] = d_live_pos + necro_prod_per_tiller_g
    return s


def add_decomposition_metrics(summary: pd.DataFrame, df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Adds metrics in BOTH flux form (g/yr, tillers/yr) and first-order rate-constant form (yr^-1).

    ROOT necromass:
      M(t) = standing pool (g)
      add(t) = newly produced necromass (g) inferred from cumulative
      loss_flux(t) = M(t-1) + add(t) - M(t)  (clipped >= 0)
      k_root(t) = loss_flux(t) / M(t-1)

    DEAD TILLERS (preferred; requires stable tiller ID):
      ND(t) = count of dead tillers
      newDead(t) = count of alive->dead transitions between t-1 and t (needs ID)
      removed(t) = ND(t-1) + newDead(t) - ND(t)  (clipped >= 0)
      k_dead(t) = removed(t) / ND(t-1)

    LEAF LITTER (fallback if no ID):
      Uses your remain fraction; k_leaf is constant = -ln(remain_frac).
    """
    s = summary.copy()

    # ---- ROOT ----
    M = s["total_root_necromass_g"].astype(float)
    add = s["total_root_necromasscum_g"].diff().clip(lower=0).fillna(0.0)

    root_loss_flux_g = (M.shift(1) + add - M).clip(lower=0)
    s["root_decomp_flux_g_per_yr"] = root_loss_flux_g

    denom_M = M.shift(1)
    s["k_root_per_yr"] = (root_loss_flux_g / denom_M).where(denom_M > 0)

    # ---- DEAD TILLERS (count pool) if possible ----
    id_col = _pick_id_column(df_raw)
    if id_col is not None:
        # Build per-timestep alive/dead sets using ID
        t_vals = np.array(sorted(df_raw["TimeStep"].dropna().astype(int).unique()))
        dead_counts = {}
        new_dead = {}

        for i, t in enumerate(t_vals):
            sub = df_raw[df_raw["TimeStep"].astype(int) == int(t)]
            dead_ids = set(sub.loc[sub["Status"] != 1, id_col].dropna().astype(str).tolist())
            alive_ids = set(sub.loc[sub["Status"] == 1, id_col].dropna().astype(str).tolist())
            dead_counts[int(t)] = float(len(dead_ids))

            if i == 0:
                new_dead[int(t)] = 0.0
            else:
                t_prev = int(t_vals[i - 1])
                sub_prev = df_raw[df_raw["TimeStep"].astype(int) == t_prev]
                alive_prev = set(sub_prev.loc[sub_prev["Status"] == 1, id_col].dropna().astype(str).tolist())

                # Alive -> Dead transitions
                new_dead[int(t)] = float(len(alive_prev.intersection(dead_ids)))

        ND = pd.Series(dead_counts).reindex(s["TimeStep"].astype(int)).astype(float)
        newD = pd.Series(new_dead).reindex(s["TimeStep"].astype(int)).astype(float).fillna(0.0)

        s["dead_tillers"] = ND.values
        s["new_dead_tillers"] = newD.values

        removed_dead = (ND.shift(1) + newD - ND).clip(lower=0)
        s["dead_tiller_removed_per_yr"] = removed_dead

        denom_ND = ND.shift(1)
        s["k_dead_tiller_per_yr"] = (removed_dead / denom_ND).where(denom_ND > 0)

        # For backward compatibility with your plotting variable name:
        # "tiller decomposition rate density" will now mean k_dead_tiller_per_yr (yr^-1)
        s["tiller_decomp_rate_for_density"] = s["k_dead_tiller_per_yr"]
        s["tiller_decomp_density_xlabel"] = "yr$^{-1}$ (dead tiller removal)"
        s["tiller_decomp_density_title"] = "Dead tiller removal rate constant density"

    else:
        # ---- Fallback: leaf litter first-order k implied by remain fraction ----
        # Note: This is NOT the same as paper's dead-tiller removal k_D, but at least it is comparable in units/form.
        k_leaf = float(-np.log(LEAF_NECRO_REMAIN_FRAC))
        s["k_leaf_per_yr"] = k_leaf

        # Put into the plotting slot as a constant series so the density panel remains.
        s["tiller_decomp_rate_for_density"] = k_leaf
        s["tiller_decomp_density_xlabel"] = "yr$^{-1}$ (leaf litter, implied)"
        s["tiller_decomp_density_title"] = "Leaf litter decay rate constant density (fallback)"

    return s


def root_necromass_bulk_density_g_cm3(total_root_necro_g: float, tussock_radius_cm: float) -> float:
    """
    Proxy bulk density: total root necromass (g) / cylinder volume (cm^3) with depth ROOT_LENGTH_CM.
    """
    if not np.isfinite(total_root_necro_g) or not np.isfinite(tussock_radius_cm):
        return float("nan")
    if tussock_radius_cm <= 0:
        return float("nan")
    vol_cm3 = np.pi * (tussock_radius_cm ** 2) * ROOT_LENGTH_CM
    return float(total_root_necro_g) / vol_cm3


# ============================================================
# Plot helpers
# ============================================================

def save_scatter(x, y, xlabel, ylabel, title, outpath: Path):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]

    plt.figure()
    plt.scatter(x, y, s=20)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def save_scatter_with_regression(x, y, xlabel, ylabel, title, outpath: Path):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]

    plt.figure()
    plt.scatter(x, y, s=20)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if x.size >= 2:
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ys = m * xs + b
        plt.plot(xs, ys, linestyle="--")

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def save_hist_density(x, xlabel, ylabel, title, outpath: Path, bins: int = 30):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    plt.figure()
    plt.hist(x, bins=bins, density=True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def save_hist_panel_one_plot(datasets, titles, xlabels, outpath: Path, bins: int = 30):
    """
    Put ALL density histograms (except tussock radius density) on one multi-panel figure.
    """
    n = len(datasets)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.5 * nrows))
    axes = np.array(axes).ravel()

    for i in range(nrows * ncols):
        ax = axes[i]
        if i >= n:
            ax.axis("off")
            continue

        data = np.asarray(datasets[i], dtype=float)
        data = data[np.isfinite(data)]
        ax.hist(data, bins=bins, density=True)
        ax.set_title(titles[i])
        ax.set_xlabel(xlabels[i])
        ax.set_ylabel("Density")

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


# ============================================================
# Driver
# ============================================================

def main():
    cfg = parse_args()
    rng = np.random.default_rng(cfg.seed)

    sim_paths = list_sim_csvs(cfg.input_dir, cfg.pattern)

    # Scatter plots (sampled points across sims)
    alive_x, alive_y = [], []
    necro_x, necro_y = [], []
    radius_vals = []
    prod_x, prod_y = [], []

    # Density datasets (sampled)
    # CHANGED: now store *rate constants* (yr^-1) for decomposition to match paper form
    k_root_rates = []
    k_tiller_rates = []

    tiller_radii = []
    leaf_sizes = []
    root_necro_bulk_dens = []

    # These let us label the "tiller decomposition" panel appropriately depending on ID availability
    tiller_decomp_xlabel = None
    tiller_decomp_title = None

    for fp in sim_paths:
        df = read_sim_csv(fp)
        df = add_live_root_mass_from_cone(df)

        summary = build_timestep_summary(df, cfg.use_only_alive_for_radius, cfg.radius_stat)
        summary = add_root_productivity(summary)
        summary = add_decomposition_metrics(summary, df)

        # Capture labels once (they'll be consistent unless some files have IDs and some don't)
        if tiller_decomp_xlabel is None and "tiller_decomp_density_xlabel" in summary.columns:
            tiller_decomp_xlabel = str(summary["tiller_decomp_density_xlabel"].iloc[0])
        if tiller_decomp_title is None and "tiller_decomp_density_title" in summary.columns:
            tiller_decomp_title = str(summary["tiller_decomp_density_title"].iloc[0])

        sampled_ts = sample_timesteps(summary["TimeStep"].values, cfg.n_timepoints, rng)
        if sampled_ts.size == 0:
            continue

        ss = summary[summary["TimeStep"].isin(sampled_ts)]

        # --- scatter plots ---
        alive_x.append(ss["tussock_radius_cm"].to_numpy(float))
        alive_y.append(ss["alive_tillers"].to_numpy(float))

        # switched axes: x = necromass (kg), y = radius (cm)
        necro_x.append(ss["total_root_necromass_kg"].to_numpy(float))
        necro_y.append(ss["tussock_radius_cm"].to_numpy(float))

        radius_vals.append(ss["tussock_radius_cm"].to_numpy(float))

        ss_prod = ss.dropna(subset=["root_productivity_g_per_tiller_per_yr", "tussock_radius_cm"])
        prod_x.append(ss_prod["root_productivity_g_per_tiller_per_yr"].to_numpy(float))
        prod_y.append(ss_prod["tussock_radius_cm"].to_numpy(float))

        # --- density datasets ---
        # ROOT decomposition: use k_root_per_yr (yr^-1)
        if "k_root_per_yr" in ss.columns:
            k_root_rates.append(ss["k_root_per_yr"].to_numpy(float))

        # "Tiller decomposition": use either k_dead_tiller_per_yr (preferred) or fallback k_leaf_per_yr
        if "tiller_decomp_rate_for_density" in ss.columns:
            k_tiller_rates.append(ss["tiller_decomp_rate_for_density"].to_numpy(float))

        sub = df[df["TimeStep"].isin(sampled_ts)]
        sub = sub[sub["Status"] == 1] if cfg.use_only_alive_for_radius else sub

        if "Radius" in sub.columns:
            tiller_radii.append(sub["Radius"].astype(float).to_numpy())
        if "LeafArea" in sub.columns:
            leaf_sizes.append(sub["LeafArea"].astype(float).to_numpy())

        for _, row in ss.iterrows():
            bd = root_necromass_bulk_density_g_cm3(
                total_root_necro_g=float(row["total_root_necromass_g"]),
                tussock_radius_cm=float(row["tussock_radius_cm"]),
            )
            root_necro_bulk_dens.append(bd)

    def cat(arrs):
        return np.concatenate(arrs) if arrs else np.array([], dtype=float)

    x1, y1 = cat(alive_x), cat(alive_y)
    x2, y2 = cat(necro_x), cat(necro_y)
    rdist = cat(radius_vals)
    x4, y4 = cat(prod_x), cat(prod_y)

    h_k_root = cat(k_root_rates)
    h_k_tiller = cat(k_tiller_rates)
    h_tiller_radius = cat(tiller_radii)
    h_leaf_size = cat(leaf_sizes)
    h_bulk_density = np.asarray(root_necro_bulk_dens, dtype=float)
    h_bulk_density = h_bulk_density[np.isfinite(h_bulk_density)]

    if x1.size == 0:
        raise ValueError("No points generated (check required columns and timesteps).")

    # One PNG per main plot (UNCHANGED)
    save_scatter(
        x1, y1,
        xlabel="Tussock radius (cm)",
        ylabel="Alive tillers (Status == 1)",
        title="Alive tillers vs tussock radius (20 random timesteps per sim)",
        outpath=cfg.output_dir / "alive_vs_tussock_radius.png",
    )

    save_scatter(
        x2, y2,
        xlabel="Total root necromass (kg)",
        ylabel="Tussock radius (cm)",
        title="Standing root necromass pool vs tussock radius (axes switched)",
        outpath=cfg.output_dir / "root_necromass_vs_tussock_radius.png",
    )

    save_hist_density(
        rdist,
        xlabel="Tussock radius (cm)",
        ylabel="Density",
        title="Distribution of tussock radii (cm) (20 random timesteps per sim)",
        outpath=cfg.output_dir / "tussock_radius_density_cm.png",
        bins=30,
    )

    save_scatter_with_regression(
        x4, y4,
        xlabel="Tiller root productivity (g per tiller yr$^{-1}$)",
        ylabel="Tussock radius (cm)",
        title="Root productivity vs tussock radius (20 random timesteps per sim)",
        outpath=cfg.output_dir / "root_productivity_vs_tussock_radius.png",
    )

    # All other density histograms in ONE panel plot
    # CHANGED: first two panels now plot yr^-1 rate constants (k), not g yr^-1 fluxes
    if tiller_decomp_xlabel is None:
        tiller_decomp_xlabel = "yr$^{-1}$"
    if tiller_decomp_title is None:
        tiller_decomp_title = "Tiller decomposition rate constant density"

    save_hist_panel_one_plot(
        datasets=[
            h_k_root,
            h_k_tiller,
            h_tiller_radius,
            h_leaf_size,
            h_bulk_density,
        ],
        titles=[
            "Root decomposition rate constant density",
            tiller_decomp_title,
            "Tiller radius density",
            "Leaf size density",
            "Root necromass bulk density",
        ],
        xlabels=[
            "yr$^{-1}$",
            tiller_decomp_xlabel,
            "cm",
            "cm$^2$",
            "g cm$^{-3}$ (proxy)",
        ],
        outpath=cfg.output_dir / "density_panels.png",
        bins=30,
    )

    print(f"Wrote PNGs to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
