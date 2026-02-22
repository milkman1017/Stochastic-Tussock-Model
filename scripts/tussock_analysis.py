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

LEAF_NECRO_REMAIN_FRAC = 0.75  # dead_leaf_area = 0.75*dead_leaf_area + prev_leaf_area each year
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
# Timestep summary + productivity + decomposition metrics
# ============================================================

def compute_packing_metrics_by_timestep(
    df: pd.DataFrame,
    use_only_alive_for_radius: bool,
    radius_stat: str,
) -> pd.DataFrame:
    """
    Computes (per timestep):
      - tussock_radius_cm (from coords)
      - alive_tillers
      - packing_density_tillers_per_cm2 = N_alive / (pi*R^2)
      - packing_fraction_area = sum(pi*r_i^2) / (pi*R^2)  (uses per-tiller Radius)
      - mean_leaf_area_alive_cm2
      - mean_tiller_radius_alive_cm
      - newborn_alive_tillers (proxy = Status==1 & Age==1)
    """
    if "TimeStep" not in df.columns or "Status" not in df.columns:
        raise ValueError("Missing required columns for packing metrics: TimeStep, Status")

    out_rows = []
    for t, sub in df.groupby("TimeStep"):
        t_int = int(t)

        sub_alive = sub[sub["Status"] == 1]
        n_alive = int(sub_alive.shape[0])

        # newborn proxy: Age == 1 among alive
        if "Age" in sub_alive.columns:
            newborn = int((sub_alive["Age"].astype(float) == 1.0).sum())
        else:
            newborn = np.nan

        # tussock radius from coordinates
        sub_for_radius = sub_alive if use_only_alive_for_radius else sub
        r_cm = radial_distance_xy_cm(sub_for_radius) if sub_for_radius.shape[0] > 0 else np.array([], dtype=float)
        R = summarize_radius_cm(r_cm, radius_stat) if r_cm.size > 0 else float("nan")

        area = np.pi * (R ** 2) if (np.isfinite(R) and R > 0) else float("nan")

        # count density
        if np.isfinite(area) and area > 0:
            count_density = n_alive / area
        else:
            count_density = float("nan")

        # packing fraction by area of circles
        if "Radius" in sub_alive.columns and np.isfinite(area) and area > 0:
            ri = sub_alive["Radius"].astype(float).to_numpy()
            ri = ri[np.isfinite(ri) & (ri > 0)]
            sum_area = float(np.sum(np.pi * ri * ri)) if ri.size > 0 else 0.0
            packing_frac = sum_area / area
            mean_ri = float(np.mean(ri)) if ri.size > 0 else float("nan")
        else:
            packing_frac = float("nan")
            mean_ri = float("nan")

        # mean leaf area among alive
        if "LeafArea" in sub_alive.columns:
            la = sub_alive["LeafArea"].astype(float).to_numpy()
            la = la[np.isfinite(la)]
            mean_la = float(np.mean(la)) if la.size > 0 else float("nan")
        else:
            mean_la = float("nan")

        out_rows.append({
            "TimeStep": t_int,
            "tussock_radius_cm": R,
            "alive_tillers": float(n_alive),
            "newborn_alive_tillers": float(newborn) if np.isfinite(newborn) else np.nan,
            "packing_density_tillers_per_cm2": float(count_density),
            "packing_fraction_area": float(packing_frac),
            "mean_leaf_area_alive_cm2": float(mean_la),
            "mean_tiller_radius_alive_cm": float(mean_ri),
        })

    return pd.DataFrame(out_rows).sort_values("TimeStep")


def build_timestep_summary(df: pd.DataFrame, use_only_alive_for_radius: bool, radius_stat: str) -> pd.DataFrame:
    required = {"TimeStep", "Status", "RootNecroMass", "RootNecroMassCum", "DeadLeafMass", "LeafArea"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # base metrics that were already in your analysis
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

    # merge in packing metrics (count density + packing fraction + newborn proxy, etc.)
    pm = compute_packing_metrics_by_timestep(df, use_only_alive_for_radius, radius_stat)
    out = out.merge(pm, on=["TimeStep", "tussock_radius_cm", "alive_tillers"], how="left")

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


def add_decomposition_metrics_optionA(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Option A: match the paper's "rate constant" form (yr^-1) by computing first-order k
    from pool balance for BOTH:
      - root necromass pool (M, g): uses RootNecroMassCum diff as input
      - leaf necromass pool (D, g): uses last year's live leaf mass as input (litterfall proxy)

    ROOT:
      add_root(t) = diff(M_cum) (>=0)
      loss_root(t) = M(t-1) + add_root(t) - M(t)
      k_root(t) = loss_root(t) / M(t-1)

    LEAF NECROMASS:
      add_leaf(t) ≈ total_leaf_mass_g(t-1)
      loss_leaf(t) = D(t-1) + add_leaf(t) - D(t)
      k_leaf(t) = loss_leaf(t) / D(t-1)
    """
    s = summary.copy()

    # ---- ROOT ----
    M = s["total_root_necromass_g"].astype(float)
    add_root = s["total_root_necromasscum_g"].diff().clip(lower=0).fillna(0.0)
    loss_root = (M.shift(1) + add_root - M).clip(lower=0)

    s["root_decomp_flux_g_per_yr"] = loss_root
    denom_M = M.shift(1)
    s["k_root_per_yr"] = (loss_root / denom_M).where(denom_M > 0)

    # ---- LEAF NECROMASS ----
    D = s["total_dead_leaf_mass_g"].astype(float)
    add_leaf = s["total_leaf_mass_g"].shift(1).fillna(0.0)
    loss_leaf = (D.shift(1) + add_leaf - D).clip(lower=0)

    s["leaf_decomp_flux_g_per_yr"] = loss_leaf
    denom_D = D.shift(1)
    s["k_leaf_per_yr"] = (loss_leaf / denom_D).where(denom_D > 0)

    # sanity reference (constant implied by remain fraction)
    s["k_leaf_implied_from_remainfrac"] = float(-np.log(LEAF_NECRO_REMAIN_FRAC))

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


def save_density_dependence_panels(
    packing_metric: np.ndarray,
    yseries: list[np.ndarray],
    titles: list[str],
    ylabels: list[str],
    xlabel: str,
    outpath: Path,
):
    """
    3-panel scatterplots of putatively density-dependent responses vs a packing metric.
    """
    x = np.asarray(packing_metric, dtype=float)
    okx = np.isfinite(x)
    x = x[okx]

    n = len(yseries)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.8 * nrows))
    axes = np.array(axes).ravel()

    for i in range(nrows * ncols):
        ax = axes[i]
        if i >= n:
            ax.axis("off")
            continue

        y = np.asarray(yseries[i], dtype=float)
        y = y[okx]
        ok = np.isfinite(y)
        xx = x[ok]
        yy = y[ok]

        ax.scatter(xx, yy, s=18)
        ax.set_title(titles[i])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabels[i])

        if xx.size >= 2:
            m, b = np.polyfit(xx, yy, 1)
            xs = np.linspace(xx.min(), xx.max(), 100)
            ys = m * xs + b
            ax.plot(xs, ys, linestyle="--")

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def save_spatial_heterogeneity_plot(
    bin_centers: np.ndarray,
    mean_annulus_density: np.ndarray,
    mean_annulus_packing: np.ndarray,
    mean_leaf_area: np.ndarray,
    outpath: Path,
):
    """
    Multi-panel plot of spatial heterogeneity vs normalized distance from tussock center.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))
    axes = np.array(axes).ravel()

    x = np.asarray(bin_centers, dtype=float)

    axes[0].plot(x, mean_annulus_density)
    axes[0].set_xlabel("Normalized distance from center (r / R)")
    axes[0].set_ylabel("Alive tiller density in annulus (tillers cm$^{-2}$)")
    axes[0].set_title("Annulus tiller density vs distance")

    axes[1].plot(x, mean_annulus_packing)
    axes[1].set_xlabel("Normalized distance from center (r / R)")
    axes[1].set_ylabel("Packing fraction in annulus (area/area)")
    axes[1].set_title("Annulus packing fraction vs distance")

    axes[2].plot(x, mean_leaf_area)
    axes[2].set_xlabel("Normalized distance from center (r / R)")
    axes[2].set_ylabel("Mean leaf area (cm$^2$)")
    axes[2].set_title("Mean leaf area vs distance")

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


# ============================================================
# Spatial heterogeneity aggregation
# ============================================================

def accumulate_annulus_stats(
    df: pd.DataFrame,
    timesteps: np.ndarray,
    use_only_alive_for_radius: bool,
    radius_stat: str,
    nbins: int = 12,
):
    """
    For each (sim, timestep) sample:
      - take alive tillers
      - compute r = sqrt(x^2+y^2), R = tussock radius
      - bin by normalized r/R into nbins
      - per annulus compute:
          density = count / annulus_area
          packing = sum(pi*ri^2) / annulus_area
          mean leaf area
    Aggregate across all samples by averaging (simple mean) within each bin.
    """
    # store lists per bin
    dens_bins = [[] for _ in range(nbins)]
    pack_bins = [[] for _ in range(nbins)]
    leaf_bins = [[] for _ in range(nbins)]

    edges = np.linspace(0.0, 1.0, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    for t in timesteps:
        sub = df[df["TimeStep"].astype(int) == int(t)]
        if sub.shape[0] == 0:
            continue

        alive = sub[sub["Status"] == 1]
        if alive.shape[0] == 0:
            continue

        # tussock radius
        sub_for_radius = alive if use_only_alive_for_radius else sub
        r_cm_all = radial_distance_xy_cm(sub_for_radius)
        if r_cm_all.size == 0:
            continue
        R = summarize_radius_cm(r_cm_all, radius_stat)
        if not np.isfinite(R) or R <= 0:
            continue

        # alive radii from center
        r_cm = radial_distance_xy_cm(alive)
        rn = r_cm / R
        ok = np.isfinite(rn) & (rn >= 0) & (rn <= 1.0)
        rn = rn[ok]
        if rn.size == 0:
            continue

        # per-tiller radius for packing fraction
        if "Radius" in alive.columns:
            ri = alive["Radius"].astype(float).to_numpy()[ok]
            ri = np.where(np.isfinite(ri) & (ri > 0), ri, 0.0)
        else:
            ri = np.zeros_like(rn)

        # leaf area
        if "LeafArea" in alive.columns:
            la = alive["LeafArea"].astype(float).to_numpy()[ok]
        else:
            la = np.full_like(rn, np.nan)

        # assign bins
        bidx = np.searchsorted(edges, rn, side="right") - 1
        bidx = np.clip(bidx, 0, nbins - 1)

        # compute per annulus stats
        for b in range(nbins):
            mask = (bidx == b)
            if not np.any(mask):
                continue

            r0n = edges[b]
            r1n = edges[b + 1]
            # annulus area in cm^2
            ann_area = np.pi * ((r1n * R) ** 2 - (r0n * R) ** 2)
            if not np.isfinite(ann_area) or ann_area <= 0:
                continue

            cnt = int(np.sum(mask))
            dens = cnt / ann_area

            pack = float(np.sum(np.pi * (ri[mask] ** 2))) / ann_area

            la_m = la[mask]
            la_m = la_m[np.isfinite(la_m)]
            mean_la = float(np.mean(la_m)) if la_m.size > 0 else float("nan")

            dens_bins[b].append(dens)
            pack_bins[b].append(pack)
            leaf_bins[b].append(mean_la)

    def mean_or_nan(xs):
        xs = np.asarray(xs, dtype=float)
        xs = xs[np.isfinite(xs)]
        return float(np.mean(xs)) if xs.size > 0 else float("nan")

    mean_dens = np.array([mean_or_nan(dens_bins[b]) for b in range(nbins)], dtype=float)
    mean_pack = np.array([mean_or_nan(pack_bins[b]) for b in range(nbins)], dtype=float)
    mean_leaf = np.array([mean_or_nan(leaf_bins[b]) for b in range(nbins)], dtype=float)

    return centers, mean_dens, mean_pack, mean_leaf


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
    k_root_rates = []
    k_leaf_rates = []
    tiller_radii = []
    leaf_sizes = []
    root_necro_bulk_dens = []

    # Density dependence data (sampled points across sims)
    dd_x_packdens = []
    dd_x_packfrac = []
    dd_newborn = []
    dd_alive = []
    dd_mean_leaf_area = []
    dd_root_prod = []

    # Spatial heterogeneity aggregation across sims/timesteps
    # We'll aggregate using the sampled timesteps per sim
    hetero_centers_accum = None
    hetero_dens_accum = []
    hetero_pack_accum = []
    hetero_leaf_accum = []

    for fp in sim_paths:
        df = read_sim_csv(fp)
        df = add_live_root_mass_from_cone(df)

        summary = build_timestep_summary(df, cfg.use_only_alive_for_radius, cfg.radius_stat)
        summary = add_root_productivity(summary)
        summary = add_decomposition_metrics_optionA(summary)

        sampled_ts = sample_timesteps(summary["TimeStep"].values, cfg.n_timepoints, rng)
        if sampled_ts.size == 0:
            continue

        ss = summary[summary["TimeStep"].isin(sampled_ts)]

        # --- original scatter plots ---
        alive_x.append(ss["tussock_radius_cm"].to_numpy(float))
        alive_y.append(ss["alive_tillers"].to_numpy(float))

        necro_x.append(ss["total_root_necromass_kg"].to_numpy(float))
        necro_y.append(ss["tussock_radius_cm"].to_numpy(float))

        radius_vals.append(ss["tussock_radius_cm"].to_numpy(float))

        ss_prod = ss.dropna(subset=["root_productivity_g_per_tiller_per_yr", "tussock_radius_cm"])
        prod_x.append(ss_prod["root_productivity_g_per_tiller_per_yr"].to_numpy(float))
        prod_y.append(ss_prod["tussock_radius_cm"].to_numpy(float))

        # --- decomposition densities (k, yr^-1) ---
        k_root_rates.append(ss["k_root_per_yr"].to_numpy(float))
        k_leaf_rates.append(ss["k_leaf_per_yr"].to_numpy(float))

        # --- density datasets from raw per-tiller values (as before) ---
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

        # --- density dependence: packing metrics vs responses ---
        # use packing fraction (area-based) AND count density as x metrics (both reflect crowding differently)
        dd_x_packdens.append(ss["packing_density_tillers_per_cm2"].to_numpy(float))
        dd_x_packfrac.append(ss["packing_fraction_area"].to_numpy(float))
        dd_newborn.append(ss["newborn_alive_tillers"].to_numpy(float))
        dd_alive.append(ss["alive_tillers"].to_numpy(float))
        dd_mean_leaf_area.append(ss["mean_leaf_area_alive_cm2"].to_numpy(float))
        dd_root_prod.append(ss["root_productivity_g_per_tiller_per_yr"].to_numpy(float))

        # --- spatial heterogeneity: annulus stats vs distance from center ---
        centers, mdens, mpack, mleaf = accumulate_annulus_stats(
            df=df,
            timesteps=sampled_ts,
            use_only_alive_for_radius=cfg.use_only_alive_for_radius,
            radius_stat=cfg.radius_stat,
            nbins=12,
        )
        hetero_centers_accum = centers
        hetero_dens_accum.append(mdens)
        hetero_pack_accum.append(mpack)
        hetero_leaf_accum.append(mleaf)

    def cat(arrs):
        return np.concatenate(arrs) if arrs else np.array([], dtype=float)

    x1, y1 = cat(alive_x), cat(alive_y)
    x2, y2 = cat(necro_x), cat(necro_y)
    rdist = cat(radius_vals)
    x4, y4 = cat(prod_x), cat(prod_y)

    h_k_root = cat(k_root_rates)
    h_k_leaf = cat(k_leaf_rates)
    h_tiller_radius = cat(tiller_radii)
    h_leaf_size = cat(leaf_sizes)
    h_bulk_density = np.asarray(root_necro_bulk_dens, dtype=float)
    h_bulk_density = h_bulk_density[np.isfinite(h_bulk_density)]

    if x1.size == 0:
        raise ValueError("No points generated (check required columns and timesteps).")

    # -------------------------
    # Original plots (unchanged filenames)
    # -------------------------
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

    # Density panels (CHANGED only in first 2 datasets: now k_root and k_leaf, both yr^-1)
    save_hist_panel_one_plot(
        datasets=[
            h_k_root,
            h_k_leaf,
            h_tiller_radius,
            h_leaf_size,
            h_bulk_density,
        ],
        titles=[
            "Root decomposition rate constant density",
            "Leaf tissue decomposition rate constant density",
            "Tiller radius density",
            "Leaf size density",
            "Root necromass bulk density",
        ],
        xlabels=[
            "yr$^{-1}$",
            "yr$^{-1}$",
            "cm",
            "cm$^2$",
            "g cm$^{-3}$ (proxy)",
        ],
        outpath=cfg.output_dir / "density_panels.png",
        bins=30,
    )

    # -------------------------
    # NEW plot 1: Density dependence panels
    # -------------------------
    dd_packdens = cat(dd_x_packdens)
    dd_packfrac = cat(dd_x_packfrac)
    dd_new = cat(dd_newborn)
    dd_alv = cat(dd_alive)
    dd_meanLA = cat(dd_mean_leaf_area)
    dd_rootP = cat(dd_root_prod)

    # Panel set using packing fraction (more sensitive to variable tiller radii)
    save_density_dependence_panels(
        packing_metric=dd_packfrac,
        yseries=[dd_new, dd_alv, dd_meanLA],
        titles=[
            "Newborn alive tillers vs packing fraction",
            "Alive tillers vs packing fraction",
            "Mean leaf area vs packing fraction",
        ],
        ylabels=[
            "Newborn alive tillers (Age==1)",
            "Alive tillers",
            "Mean leaf area (cm$^2$)",
        ],
        xlabel="Packing fraction (sum $\\pi r_i^2$ / $\\pi R^2$)",
        outpath=cfg.output_dir / "density_dependence_packing_fraction_panels.png",
    )

    # Optional second density-dependence panel set using count density
    save_density_dependence_panels(
        packing_metric=dd_packdens,
        yseries=[dd_new, dd_rootP, dd_meanLA],
        titles=[
            "Newborn alive tillers vs count density",
            "Root productivity vs count density",
            "Mean leaf area vs count density",
        ],
        ylabels=[
            "Newborn alive tillers (Age==1)",
            "Root productivity (g per tiller yr$^{-1}$)",
            "Mean leaf area (cm$^2$)",
        ],
        xlabel="Alive tiller density (tillers cm$^{-2}$) using $\\pi R^2$",
        outpath=cfg.output_dir / "density_dependence_count_density_panels.png",
    )

    # -------------------------
    # NEW plot 2: Spatial heterogeneity vs distance from center
    # -------------------------
    if hetero_centers_accum is not None and hetero_dens_accum:
        dens_mat = np.vstack(hetero_dens_accum)
        pack_mat = np.vstack(hetero_pack_accum)
        leaf_mat = np.vstack(hetero_leaf_accum)

        # simple mean across sims (ignores NaNs)
        mean_dens = np.nanmean(dens_mat, axis=0)
        mean_pack = np.nanmean(pack_mat, axis=0)
        mean_leaf = np.nanmean(leaf_mat, axis=0)

        save_spatial_heterogeneity_plot(
            bin_centers=hetero_centers_accum,
            mean_annulus_density=mean_dens,
            mean_annulus_packing=mean_pack,
            mean_leaf_area=mean_leaf,
            outpath=cfg.output_dir / "spatial_heterogeneity_vs_distance.png",
        )

    print(f"Wrote PNGs to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
