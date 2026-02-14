#!/usr/bin/env python3

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_tussock_radius(df_t: pd.DataFrame) -> float:
    """
    Tussock radius at a timestep using ALL tillers (alive + dead):
      max( sqrt(X^2 + Y^2) + tiller Radius )
    """
    if len(df_t) == 0:
        return np.nan

    r_xy = np.sqrt(df_t["X"].to_numpy() ** 2 + df_t["Y"].to_numpy() ** 2)
    r_tiller = df_t["Radius"].to_numpy()
    return float(np.max(r_xy + r_tiller))


def compute_eta_capacity(tussock_radius: float, mean_diameter: float) -> float:
    """
    Hexagonal packing capacity from Curasi et al.:
      eta(t) = pi * r(t)^2 / (theta^2 * sqrt(12))
    where theta is mean tiller diameter.

    Returns NaN if inputs are invalid.
    """
    if not np.isfinite(tussock_radius) or tussock_radius <= 0:
        return np.nan
    if not np.isfinite(mean_diameter) or mean_diameter <= 0:
        return np.nan
    return float(np.pi * (tussock_radius ** 2) / ((mean_diameter ** 2) * np.sqrt(12.0)))


def summarize_by_timestep(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a per-timestep summary with columns:
      TimeStep, tussock_radius, n_alive, n_total, density, prop_new,
      mean_diameter, eta_capacity, packing_index_total, packing_index_alive

    - tussock_radius computed from ALL tillers (alive+dead)
    - n_alive/density/prop_new computed from alive tillers (Status==1)
    - n_total = total tillers (alive + dead)
    - density = n_alive / (pi * tussock_radius^2)
    - prop_new = (# alive with Age==1) / n_alive

    Packing index additions (per Curasi et al.):
      eta_capacity = pi * r^2 / (theta^2 * sqrt(12))
      packing_index_total = n_total / eta_capacity
      packing_index_alive = n_alive / eta_capacity
    """
    required = ["TimeStep", "Age", "Radius", "X", "Y", "Status"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out_rows = []
    for t, df_t in df.groupby("TimeStep", sort=True):
        tuss_r = compute_tussock_radius(df_t)

        alive = df_t[df_t["Status"] == 1]
        n_alive = int(len(alive))
        n_total = int(len(df_t))

        area = np.pi * (tuss_r ** 2) if np.isfinite(tuss_r) and tuss_r > 0 else np.nan
        density = (n_alive / area) if (np.isfinite(area) and area > 0) else np.nan

        n_new = int(len(alive[alive["Age"] == 1])) if n_alive > 0 else 0
        prop_new = (n_new / n_alive) if n_alive > 0 else np.nan

        # mean tiller diameter theta from the per-tiller Radius column
        # (assumes Radius is the tiller radius at that timestep)
        rad = pd.to_numeric(df_t["Radius"], errors="coerce").to_numpy(dtype=float)
        mean_rad = float(np.nanmean(rad)) if np.isfinite(rad).any() else np.nan
        mean_diam = 2.0 * mean_rad if np.isfinite(mean_rad) else np.nan

        eta = compute_eta_capacity(tuss_r, mean_diam)

        packing_total = (n_total / eta) if (np.isfinite(eta) and eta > 0) else np.nan
        packing_alive = (n_alive / eta) if (np.isfinite(eta) and eta > 0) else np.nan

        out_rows.append(
            {
                "TimeStep": int(t),
                "tussock_radius": tuss_r,
                "n_alive": n_alive,
                "n_total": n_total,
                "density": density,
                "prop_new": prop_new,
                "mean_diameter": mean_diam,
                "eta_capacity": eta,
                "packing_index_total": packing_total,
                "packing_index_alive": packing_alive,
            }
        )

    return pd.DataFrame(out_rows).sort_values("TimeStep").reset_index(drop=True)


def pick_random_timesteps(summary: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(summary) == 0:
        return summary
    if len(summary) <= n:
        return summary.copy()
    idx = rng.choice(summary.index.to_numpy(), size=n, replace=False)
    return summary.loc[np.sort(idx)].copy()


def ecdf(values: np.ndarray):
    """Return x (sorted) and y (ECDF values) for finite values."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.array([]), np.array([])
    x = np.sort(v)
    y = np.arange(1, x.size + 1) / x.size
    return x, y


def plot_radius_vs_time(all_summaries: dict, outdir: Path):
    all_ts = sorted(set(np.concatenate([s["TimeStep"].to_numpy() for s in all_summaries.values()])))
    aligned = []

    plt.figure(figsize=(10, 6))

    for s in all_summaries.values():
        ser = pd.Series(s["tussock_radius"].to_numpy(), index=s["TimeStep"].to_numpy())
        ser = ser.reindex(all_ts)
        aligned.append(ser.to_numpy())
        plt.plot(all_ts, ser.to_numpy(), linewidth=1.0, alpha=0.2)

    mean_r = np.nanmean(np.vstack(aligned), axis=0)
    plt.plot(all_ts, mean_r, linewidth=3.0)

    plt.xlabel("TimeStep")
    plt.ylabel("Tussock radius (all tillers)")
    plt.title("Tussock radius vs TimeStep (per sim + mean)")
    plt.tight_layout()
    plt.savefig(outdir / "tussock_radius_vs_time.png", dpi=300)
    plt.close()


def plot_alive_vs_radius_random(all_summaries: dict, outdir: Path, n_points: int, rng: np.random.Generator):
    xs, ys = [], []
    for s in all_summaries.values():
        sub = pick_random_timesteps(s, n_points, rng)
        xs.append(sub["tussock_radius"].to_numpy())
        ys.append(sub["n_alive"].to_numpy())

    x = np.concatenate(xs) if xs else np.array([])
    y = np.concatenate(ys) if ys else np.array([])

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, s=20, alpha=0.6)
    plt.xlabel("Tussock radius (all tillers)")
    plt.ylabel("Number of alive tillers")
    plt.title(f"Alive tillers vs radius (random {n_points}/sim)")
    plt.tight_layout()
    plt.savefig(outdir / "alive_tillers_vs_radius_scatter.png", dpi=300)
    plt.close()


def plot_total_vs_radius_random(all_summaries: dict, outdir: Path, n_points: int, rng: np.random.Generator):
    xs, ys = [], []
    for s in all_summaries.values():
        sub = pick_random_timesteps(s, n_points, rng)
        xs.append(sub["tussock_radius"].to_numpy())
        ys.append(sub["n_total"].to_numpy())

    x = np.concatenate(xs) if xs else np.array([])
    y = np.concatenate(ys) if ys else np.array([])

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, s=20, alpha=0.6)
    plt.xlabel("Tussock radius (all tillers)")
    plt.ylabel("Total number of tillers (alive + dead)")
    plt.title(f"Total tillers vs radius (random {n_points}/sim)")
    plt.tight_layout()
    plt.savefig(outdir / "total_tillers_vs_radius_scatter.png", dpi=300)
    plt.close()


def plot_density_vs_radius_scatter(all_summaries: dict, outdir: Path, n_points: int, rng: np.random.Generator):
    xs, ys = [], []
    for s in all_summaries.values():
        sub = pick_random_timesteps(s, n_points, rng)
        good = np.isfinite(sub["tussock_radius"]) & np.isfinite(sub["density"])
        sub = sub.loc[good]
        xs.append(sub["tussock_radius"].to_numpy())
        ys.append(sub["density"].to_numpy())

    x = np.concatenate(xs) if xs else np.array([])
    y = np.concatenate(ys) if ys else np.array([])

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, s=20, alpha=0.6)
    plt.xlabel("Tussock radius (all tillers)")
    plt.ylabel("Tiller density (alive / area)")
    plt.title(f"Tiller density vs radius (random {n_points}/sim)")
    plt.tight_layout()
    plt.savefig(outdir / "density_vs_radius_scatter.png", dpi=300)
    plt.close()


def plot_prop_new_vs_radius_random(all_summaries: dict, outdir: Path, n_points: int, rng: np.random.Generator):
    xs, ys = [], []
    for s in all_summaries.values():
        sub = pick_random_timesteps(s, n_points, rng)
        good = np.isfinite(sub["tussock_radius"]) & np.isfinite(sub["prop_new"])
        sub = sub.loc[good]
        xs.append(sub["tussock_radius"].to_numpy())
        ys.append(sub["prop_new"].to_numpy())

    x = np.concatenate(xs) if xs else np.array([])
    y = np.concatenate(ys) if ys else np.array([])

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, s=20, alpha=0.6)
    plt.xlabel("Tussock radius (all tillers)")
    plt.ylabel("Proportion new daughters (Age==1 among alive)")
    plt.title(f"New daughter proportion vs radius (random {n_points}/sim)")
    plt.tight_layout()
    plt.savefig(outdir / "prop_new_daughters_vs_radius_scatter.png", dpi=300)
    plt.close()


def _plot_age_hist(ax, ages: np.ndarray, title: str):
    ages = ages[np.isfinite(ages)]
    if len(ages) == 0:
        ax.set_title(title + " (no data)")
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")
        return

    a_min = int(np.floor(np.min(ages)))
    a_max = int(np.ceil(np.max(ages)))
    bins = np.arange(a_min - 0.5, a_max + 1.5, 1.0)

    ax.hist(ages, bins=bins)
    ax.set_title(title)
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")

    mean_age = float(np.mean(ages))
    ax.axvline(mean_age, color="red", linestyle="--", linewidth=2, label=f"mean: {mean_age:.2f}")
    ax.legend()


def plot_age_distributions_alive_dead(final_alive_ages: np.ndarray, final_dead_ages: np.ndarray, outdir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    _plot_age_hist(axes[0], final_alive_ages, "Age distribution (alive tillers; final timestep per sim)")
    _plot_age_hist(axes[1], final_dead_ages, "Age distribution (dead tillers; final timestep per sim)")

    fig.suptitle("Tiller age distributions")
    fig.tight_layout()
    fig.savefig(outdir / "age_distributions_alive_vs_dead.png", dpi=300)
    plt.close(fig)


# -----------------------
# New plots: packing index
# -----------------------

def plot_packing_index_vs_time(all_summaries: dict, outdir: Path):
    all_ts = sorted(set(np.concatenate([s["TimeStep"].to_numpy() for s in all_summaries.values()])))
    aligned = []

    plt.figure(figsize=(10, 6))
    for s in all_summaries.values():
        ser = pd.Series(s["packing_index_total"].to_numpy(), index=s["TimeStep"].to_numpy()).reindex(all_ts)
        aligned.append(ser.to_numpy())
        plt.plot(all_ts, ser.to_numpy(), linewidth=1.0, alpha=0.2)

    mean_pi = np.nanmean(np.vstack(aligned), axis=0)
    plt.plot(all_ts, mean_pi, linewidth=3.0)

    plt.axhline(1.0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    plt.xlabel("TimeStep")
    plt.ylabel("Packing index (N_total / eta)")
    plt.title("Packing index vs TimeStep (per sim + mean)")
    plt.tight_layout()
    plt.savefig(outdir / "packing_index_vs_time.png", dpi=300)
    plt.close()


def plot_packing_index_vs_radius_random(all_summaries: dict, outdir: Path, n_points: int, rng: np.random.Generator):
    xs, ys = [], []
    for s in all_summaries.values():
        sub = pick_random_timesteps(s, n_points, rng)
        good = np.isfinite(sub["tussock_radius"]) & np.isfinite(sub["packing_index_total"])
        sub = sub.loc[good]
        xs.append(sub["tussock_radius"].to_numpy())
        ys.append(sub["packing_index_total"].to_numpy())

    x = np.concatenate(xs) if xs else np.array([])
    y = np.concatenate(ys) if ys else np.array([])

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, s=20, alpha=0.6)
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    plt.xlabel("Tussock radius (all tillers)")
    plt.ylabel("Packing index (N_total / eta)")
    plt.title(f"Packing index vs radius (random {n_points}/sim)")
    plt.tight_layout()
    plt.savefig(outdir / "packing_index_vs_radius_scatter.png", dpi=300)
    plt.close()


def plot_eta_vs_total_tillers(all_summaries: dict, outdir: Path):
    """
    Simple diagnostic: for each sim, plot eta_capacity(t) and n_total(t) over time.
    This shows whether the system approaches the geometric capacity.
    """
    plt.figure(figsize=(10, 6))
    for name, s in all_summaries.items():
        t = s["TimeStep"].to_numpy()
        eta = s["eta_capacity"].to_numpy()
        nt = s["n_total"].to_numpy()
        # plot faint to avoid clutter
        plt.plot(t, eta, linewidth=1.0, alpha=0.15)
        plt.plot(t, nt, linewidth=1.0, alpha=0.15)

    plt.xlabel("TimeStep")
    plt.ylabel("Count")
    plt.title("Eta capacity and total tillers vs TimeStep (all sims; faint)\n(each sim contributes two lines: eta and N_total)")
    plt.tight_layout()
    plt.savefig(outdir / "eta_capacity_vs_total_tillers.png", dpi=300)
    plt.close()


# -----------------------
# New plots: ECDFs
# -----------------------

def plot_final_radius_ecdf(final_radii: np.ndarray, outdir: Path):
    x, y = ecdf(final_radii)
    plt.figure(figsize=(8, 6))
    if x.size > 0:
        plt.plot(x, y, linewidth=2.0)
    plt.xlabel("Final tussock radius")
    plt.ylabel("ECDF")
    plt.title("ECDF of final tussock radius across simulations")
    plt.tight_layout()
    plt.savefig(outdir / "final_radius_ecdf.png", dpi=300)
    plt.close()


def plot_final_packing_index_ecdf(final_pi: np.ndarray, outdir: Path):
    x, y = ecdf(final_pi)
    plt.figure(figsize=(8, 6))
    if x.size > 0:
        plt.plot(x, y, linewidth=2.0)
        plt.axvline(1.0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    plt.xlabel("Final packing index (N_total / eta)")
    plt.ylabel("ECDF")
    plt.title("ECDF of final packing index across simulations")
    plt.tight_layout()
    plt.savefig(outdir / "final_packing_index_ecdf.png", dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Make summary plots from a directory of individual-tiller simulation CSVs.")
    ap.add_argument("--in_dir", required=True, type=str, help="Directory containing simulation CSV files")
    ap.add_argument("--out_dir", required=True, type=str, help="Directory to save plots into")
    ap.add_argument("--n_random", default=20, type=int, help="Random timesteps to sample per sim for scatter plots")
    ap.add_argument("--seed", default=123, type=int, help="RNG seed for reproducible random sampling")
    ap.add_argument("--pattern", default="*.csv", type=str, help="Glob pattern for sim files (default: *.csv)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    files = sorted(in_dir.glob(args.pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"No files matched {args.pattern} in {in_dir}")

    all_summaries = {}
    final_alive_ages = []
    final_dead_ages = []
    final_radii = []
    final_packing = []

    for f in files:
        df = pd.read_csv(f)

        # Ensure TimeStep is int
        df["TimeStep"] = pd.to_numeric(df["TimeStep"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["TimeStep"]).copy()
        df["TimeStep"] = df["TimeStep"].astype(int)

        # Make per-timestep summary (radius/area/packing derived)
        summary = summarize_by_timestep(df)
        all_summaries[f.stem] = summary

        # Age distributions taken at FINAL timestep for each sim (avoids double-counting across time)
        t_final = int(df["TimeStep"].max())
        df_final = df[df["TimeStep"] == t_final].copy()
        df_final["Age"] = pd.to_numeric(df_final["Age"], errors="coerce")

        alive_final = df_final[df_final["Status"] == 1]
        dead_final = df_final[df_final["Status"] != 1]

        final_alive_ages.append(alive_final["Age"].to_numpy(dtype=float))
        final_dead_ages.append(dead_final["Age"].to_numpy(dtype=float))

        # Final radius / packing index from the timestep summary
        if len(summary) > 0:
            final_radii.append(float(summary["tussock_radius"].iloc[-1]))
            final_packing.append(float(summary["packing_index_total"].iloc[-1]))

    # Existing plots
    plot_radius_vs_time(all_summaries, out_dir)
    plot_alive_vs_radius_random(all_summaries, out_dir, args.n_random, rng)
    plot_total_vs_radius_random(all_summaries, out_dir, args.n_random, rng)
    plot_density_vs_radius_scatter(all_summaries, out_dir, args.n_random, rng)
    plot_prop_new_vs_radius_random(all_summaries, out_dir, args.n_random, rng)

    alive_ages = np.concatenate(final_alive_ages) if final_alive_ages else np.array([])
    dead_ages = np.concatenate(final_dead_ages) if final_dead_ages else np.array([])
    plot_age_distributions_alive_dead(alive_ages, dead_ages, out_dir)

    # New packing index plots
    plot_packing_index_vs_time(all_summaries, out_dir)
    plot_packing_index_vs_radius_random(all_summaries, out_dir, args.n_random, rng)
    plot_eta_vs_total_tillers(all_summaries, out_dir)

    # New ECDF plots (final timestep across sims)
    plot_final_radius_ecdf(np.array(final_radii, dtype=float), out_dir)
    plot_final_packing_index_ecdf(np.array(final_packing, dtype=float), out_dir)

    print(f"Saved plots to: {out_dir.resolve()}")
    print("Files written:")
    for p in [
        # existing
        "tussock_radius_vs_time.png",
        "alive_tillers_vs_radius_scatter.png",
        "total_tillers_vs_radius_scatter.png",
        "density_vs_radius_scatter.png",
        "prop_new_daughters_vs_radius_scatter.png",
        "age_distributions_alive_vs_dead.png",
        # new
        "packing_index_vs_time.png",
        "packing_index_vs_radius_scatter.png",
        "eta_capacity_vs_total_tillers.png",
        "final_radius_ecdf.png",
        "final_packing_index_ecdf.png",
    ]:
        print("  -", p)


if __name__ == "__main__":
    main()
