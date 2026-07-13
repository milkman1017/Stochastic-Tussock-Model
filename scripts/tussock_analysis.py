#!/usr/bin/env python3

"""
Streaming deep tussock diagnostics.

Scans a hypothesis directory for model outputs like:

    <hypothesis_dir>/<question>/<site>/set_###/final_sims/

or more generally:

    <hypothesis_dir>/.../set_###/final_sims/

For each set, it tries to read:

    final_sims/summaries/summary_*.csv
    final_sims/yearly_summaries/yearly_summary_*.csv
    full per-tiller simulation CSVs inside final_sims/

It writes/appends:

Overall hypothesis-level outputs:

    <hypothesis_dir>/compiled_deep_metrics_per_sim.csv
    <hypothesis_dir>/compiled_deep_metrics_per_set.csv
    <hypothesis_dir>/compiled_deep_metrics_per_question_summary.csv
    <hypothesis_dir>/compiled_deep_metrics_progress_log.csv
    <hypothesis_dir>/compiled_deep_metrics_error_log.txt

Per-question outputs:

    <hypothesis_dir>/<question>/deep_metrics_per_sim.csv
    <hypothesis_dir>/<question>/deep_metrics_per_set.csv
    <hypothesis_dir>/<question>/deep_metrics_per_question_summary.csv
    <hypothesis_dir>/<question>/deep_metrics_progress_log.csv
    <hypothesis_dir>/<question>/deep_metrics_error_log.txt

Important:
    append_df_csv() is schema-safe. If later sets produce extra metric columns,
    it unions old and new columns and rewrites the CSV rather than raw-appending
    malformed rows.
"""

import argparse
import math
import re
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------
# General helpers
# -----------------------------

def to_numeric_inplace(
    df,
    skip_cols=("_source_file", "full_file", "question", "site", "set", "set_dir"),
):
    for c in df.columns:
        if c not in skip_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def cv(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    mu = np.mean(x)

    if not np.isfinite(mu) or mu == 0:
        return np.nan

    return float(np.std(x) / mu)


def safe_slope(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan

    if np.nanmax(x) == np.nanmin(x):
        return np.nan

    return float(np.polyfit(x, y, 1)[0])


def wasserstein_1d(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) == 0 or len(y) == 0:
        return np.nan

    x = np.sort(x)
    y = np.sort(y)

    i = 0
    j = 0
    n = len(x)
    m = len(y)

    cdf_x = 0.0
    cdf_y = 0.0
    prev = min(x[0], y[0])
    w = 0.0

    while i < n or j < m:
        nx = x[i] if i < n else np.inf
        ny = y[j] if j < m else np.inf
        nxt = min(nx, ny)

        w += abs(cdf_x - cdf_y) * (nxt - prev)

        if nx == nxt:
            val = nxt
            while i < n and x[i] == val:
                i += 1
            cdf_x = i / n

        if ny == nxt:
            val = nxt
            while j < m and y[j] == val:
                j += 1
            cdf_y = j / m

        prev = nxt

    return float(w)


def nearest_neighbor_mean(x, y, max_n=1000):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    pts = np.column_stack([x[mask], y[mask]])

    n = len(pts)

    if n < 2:
        return np.nan

    if n > max_n:
        rng = np.random.default_rng(1)
        keep = rng.choice(n, size=max_n, replace=False)
        pts = pts[keep]
        n = len(pts)

    dmins = []

    for i in range(n):
        d = np.sqrt(np.sum((pts - pts[i]) ** 2, axis=1))
        d[i] = np.inf
        dmins.append(np.min(d))

    return float(np.mean(dmins))


# -----------------------------
# Schema-safe CSV writing
# -----------------------------

def append_df_csv(df, path):
    """
    Append a dataframe to CSV while allowing columns to differ across appends.

    This fixes the ParserError where one set writes 126 columns and the next
    set writes 186 columns. Instead of raw-appending, this function reads the
    existing CSV, takes the union of old and new columns, aligns both tables,
    and rewrites the file.

    This is slower than raw append, but it is much safer for exploratory
    model diagnostics where available metrics can differ among sets.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if df is None or df.empty:
        return

    df = df.copy()

    if not path.exists():
        df.to_csv(path, index=False)
        return

    try:
        old = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        df.to_csv(path, index=False)
        return
    except pd.errors.ParserError as e:
        raise RuntimeError(
            f"Existing CSV is malformed and cannot be safely appended: {path}\n"
            f"Delete this file or rerun with --fresh.\n"
            f"Original pandas error: {e}"
        )

    all_cols = list(old.columns)

    for c in df.columns:
        if c not in all_cols:
            all_cols.append(c)

    old = old.reindex(columns=all_cols)
    df = df.reindex(columns=all_cols)

    out = pd.concat([old, df], ignore_index=True)
    out.to_csv(path, index=False)


def append_row_csv(row, path):
    append_df_csv(pd.DataFrame([row]), path)


# -----------------------------
# Directory discovery
# -----------------------------

def find_set_dirs(hypothesis_dir):
    hypothesis_dir = Path(hypothesis_dir).resolve()
    pattern = re.compile(r"^set_\d+$")

    set_dirs = []

    for p in hypothesis_dir.rglob("set_*"):
        if p.is_dir() and pattern.match(p.name) and (p / "final_sims").exists():
            set_dirs.append(p)

    return sorted(set_dirs)


def question_from_set_dir(hypothesis_dir, set_dir):
    hypothesis_dir = Path(hypothesis_dir).resolve()
    set_dir = Path(set_dir).resolve()

    rel = set_dir.relative_to(hypothesis_dir)
    parts = rel.parts

    if len(parts) >= 2:
        return parts[0]

    return "UNKNOWN"


def site_from_set_dir(hypothesis_dir, set_dir):
    hypothesis_dir = Path(hypothesis_dir).resolve()
    set_dir = Path(set_dir).resolve()

    rel = set_dir.relative_to(hypothesis_dir)
    parts = rel.parts

    # Typical:
    #   hypothesis_dir / question / site / set_001
    if len(parts) >= 3:
        return parts[-2]

    return "UNKNOWN"


# -----------------------------
# CSV readers
# -----------------------------

def read_many_csvs(paths):
    dfs = []

    for p in paths:
        try:
            df = pd.read_csv(p)
            df["_source_file"] = str(p)
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] Could not read {p}: {e}", flush=True)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def read_summary_tables(final_sims_dir):
    final_sims_dir = Path(final_sims_dir)

    summary_paths = sorted((final_sims_dir / "summaries").glob("summary_*.csv"))
    yearly_paths = sorted((final_sims_dir / "yearly_summaries").glob("yearly_summary_*.csv"))

    summaries = read_many_csvs(summary_paths)
    yearly = read_many_csvs(yearly_paths)

    return summaries, yearly


def find_full_tiller_files(final_sims_dir):
    final_sims_dir = Path(final_sims_dir)
    all_csvs = sorted(final_sims_dir.rglob("*.csv"))

    out = []

    for p in all_csvs:
        parts = set(p.parts)
        name = p.name.lower()

        if "summaries" in parts:
            continue

        if "yearly_summaries" in parts:
            continue

        if "summary" in name:
            continue

        if "population_results" in name:
            continue

        if "optimization_results" in name:
            continue

        out.append(p)

    return out


# -----------------------------
# Final summary metrics
# -----------------------------

def metrics_from_summaries(summaries):
    if summaries.empty:
        return pd.DataFrame()

    df = summaries.copy()
    df = to_numeric_inplace(df)

    if "sim_id" not in df.columns:
        return pd.DataFrame()

    rows = []

    for sim_id, g in df.groupby("sim_id"):
        r = g.iloc[-1]

        row = {
            "sim_id": int(sim_id),
        }

        for c in [
            "final_t",
            "final_diameter",
            "alive_y",
            "rmax_y",
            "overflow_t",
            "extinct_t",
            "missing_year",
            "alive_final",
            "LeafArea",
        ]:
            if c in r.index:
                row[c] = r[c]

        if "alive_final" in row:
            row["is_alive_final"] = int(row["alive_final"] > 0)
            row["is_extinct_final"] = int(row["alive_final"] <= 0)

        if "overflow_t" in row:
            row["has_overflow"] = int(row["overflow_t"] >= 0)

        if "extinct_t" in row:
            row["went_extinct_before_end"] = int(row["extinct_t"] >= 0)

        if "missing_year" in row:
            row["has_missing_year"] = int(row["missing_year"] == 1)

        rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------
# Yearly trajectory metrics
# -----------------------------

def metrics_from_yearly(yearly):
    if yearly.empty:
        return pd.DataFrame()

    df = yearly.copy()
    df = to_numeric_inplace(df)

    if "sim_id" not in df.columns or "time_step" not in df.columns:
        return pd.DataFrame()

    rows = []

    for sim_id, g in df.groupby("sim_id"):
        g = g.sort_values("time_step").copy()

        if g.empty:
            continue

        tmax = g["time_step"].max()

        if np.isfinite(tmax):
            last_half = g[g["time_step"] >= 0.5 * tmax]
            last_quarter = g[g["time_step"] >= 0.75 * tmax]
        else:
            last_half = g
            last_quarter = g

        row = {
            "sim_id": int(sim_id),
            "n_years_recorded": int(len(g)),
            "final_time_step_yearly": float(g["time_step"].iloc[-1]),
        }

        if "n_total" in g.columns:
            row["final_n_total"] = float(g["n_total"].iloc[-1])
            row["max_n_total"] = float(g["n_total"].max())
            row["mean_n_total"] = float(g["n_total"].mean())
            row["cv_n_total"] = cv(g["n_total"])
            row["slope_n_total_last_half"] = safe_slope(
                last_half["time_step"],
                last_half["n_total"],
            )

        if "n_alive" in g.columns:
            row["final_n_alive"] = float(g["n_alive"].iloc[-1])
            row["max_n_alive"] = float(g["n_alive"].max())
            row["mean_n_alive"] = float(g["n_alive"].mean())
            row["median_n_alive"] = float(g["n_alive"].median())
            row["cv_n_alive"] = cv(g["n_alive"])
            row["slope_n_alive_all"] = safe_slope(g["time_step"], g["n_alive"])
            row["slope_n_alive_last_half"] = safe_slope(
                last_half["time_step"],
                last_half["n_alive"],
            )

            if np.isfinite(row["max_n_alive"]) and row["max_n_alive"] > 0:
                row["final_alive_over_peak_alive"] = row["final_n_alive"] / row["max_n_alive"]
            else:
                row["final_alive_over_peak_alive"] = np.nan

        if "n_dead" in g.columns:
            row["final_n_dead"] = float(g["n_dead"].iloc[-1])
            row["max_n_dead"] = float(g["n_dead"].max())
            row["mean_n_dead"] = float(g["n_dead"].mean())
            row["slope_n_dead_last_half"] = safe_slope(
                last_half["time_step"],
                last_half["n_dead"],
            )

        if "n_alive" in g.columns and "n_dead" in g.columns:
            denom = row.get("final_n_alive", np.nan)

            if np.isfinite(denom) and denom > 0:
                row["final_dead_to_alive_ratio"] = row.get("final_n_dead", np.nan) / denom
            else:
                row["final_dead_to_alive_ratio"] = np.nan

        if "n_newborn" in g.columns:
            row["total_newborns"] = float(g["n_newborn"].sum())
            row["late_newborns"] = float(last_quarter["n_newborn"].sum())
            row["mean_newborns_per_year"] = float(g["n_newborn"].mean())
            row["max_newborns_one_year"] = float(g["n_newborn"].max())
            row["slope_newborns_last_half"] = safe_slope(
                last_half["time_step"],
                last_half["n_newborn"],
            )

        if "diameter" in g.columns:
            row["final_diameter_yearly"] = float(g["diameter"].iloc[-1])
            row["max_diameter_yearly"] = float(g["diameter"].max())
            row["mean_diameter_yearly"] = float(g["diameter"].mean())
            row["cv_diameter"] = cv(g["diameter"])
            row["slope_diameter_all"] = safe_slope(g["time_step"], g["diameter"])
            row["slope_diameter_last_half"] = safe_slope(
                last_half["time_step"],
                last_half["diameter"],
            )

            if np.isfinite(row["max_diameter_yearly"]) and row["max_diameter_yearly"] > 0:
                row["final_diameter_over_peak_diameter"] = (
                    row["final_diameter_yearly"] / row["max_diameter_yearly"]
                )
            else:
                row["final_diameter_over_peak_diameter"] = np.nan

        if "radius" in g.columns:
            row["final_radius_yearly"] = float(g["radius"].iloc[-1])
            row["max_radius_yearly"] = float(g["radius"].max())
            row["slope_radius_last_half"] = safe_slope(
                last_half["time_step"],
                last_half["radius"],
            )

        if "leaf_area_mean" in g.columns:
            row["final_leaf_area_mean_yearly"] = float(g["leaf_area_mean"].iloc[-1])
            row["mean_leaf_area_mean_yearly"] = float(g["leaf_area_mean"].mean())
            row["slope_leaf_area_mean_last_half"] = safe_slope(
                last_half["time_step"],
                last_half["leaf_area_mean"],
            )

        if "overflow" in g.columns:
            row["any_overflow_yearly"] = int(np.nanmax(g["overflow"]) > 0)
            row["overflow_year_fraction"] = float(np.nanmean(g["overflow"] > 0))

        row["late_absolute_alive_slope"] = abs(row.get("slope_n_alive_last_half", np.nan))
        row["late_absolute_diameter_slope"] = abs(row.get("slope_diameter_last_half", np.nan))

        rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------
# Full per-tiller metrics
# -----------------------------

def infer_sim_id_from_file_or_df(path, df, fallback_sim_id):
    for c in ["sim_id", "SimID", "simulation_id", "SimulationID"]:
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce").dropna()

            if len(vals):
                return int(vals.iloc[0])

    nums = re.findall(r"\d+", Path(path).stem)

    if nums:
        return int(nums[-1])

    return int(fallback_sim_id)


def compute_lineage_depths(parent_map):
    depths = {}

    def depth(tid, stack=None):
        if stack is None:
            stack = set()

        if tid in depths:
            return depths[tid]

        if tid in stack:
            depths[tid] = np.nan
            return np.nan

        stack.add(tid)
        parent = parent_map.get(tid, -1)

        if parent < 0 or parent not in parent_map:
            depths[tid] = 0
        else:
            d_parent = depth(parent, stack)
            depths[tid] = np.nan if not np.isfinite(d_parent) else d_parent + 1

        stack.remove(tid)
        return depths[tid]

    for tid in parent_map:
        depth(tid)

    return depths


def metrics_from_full_tiller_file(path, fallback_sim_id):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Could not read full tiller file {path}: {e}", flush=True)
        return None

    required = {"TimeStep", "TillerID", "Radius", "X", "Y", "Status"}

    if not required.issubset(df.columns):
        return None

    df = to_numeric_inplace(df)
    sim_id = infer_sim_id_from_file_or_df(path, df, fallback_sim_id)

    if df.empty:
        return None

    tmax = df["TimeStep"].max()

    if not np.isfinite(tmax):
        return None

    final = df[df["TimeStep"] == tmax].copy()
    alive = final[final["Status"] == 1].copy()
    dead = final[final["Status"] == 0].copy()

    row = {
        "sim_id": sim_id,
        "full_file": str(path),
        "final_time_step_full": float(tmax),
        "final_total_tillers_full": int(len(final)),
        "final_live_tillers_full": int(len(alive)),
        "final_dead_tillers_full": int(len(dead)),
        "live_fraction_full": float(len(alive) / len(final)) if len(final) else np.nan,
    }

    if len(alive):
        cx = alive["X"].mean()
        cy = alive["Y"].mean()
        radial = np.sqrt((alive["X"] - cx) ** 2 + (alive["Y"] - cy) ** 2)

        row["live_radial_distance_mean"] = float(radial.mean())
        row["live_radial_distance_sd"] = float(radial.std())
        row["live_radial_distance_max"] = float(radial.max())
        row["live_nearest_neighbor_mean"] = nearest_neighbor_mean(alive["X"], alive["Y"])

        area = math.pi * max(float(radial.max()), 1e-12) ** 2
        row["live_tiller_density_xy"] = float(len(alive) / area)

        row["live_radius_mean"] = float(alive["Radius"].mean())
        row["live_radius_median"] = float(alive["Radius"].median())
        row["live_radius_sd"] = float(alive["Radius"].std())

        if "Age" in alive.columns:
            row["live_age_mean"] = float(alive["Age"].mean())
            row["live_age_median"] = float(alive["Age"].median())
            row["live_age_sd"] = float(alive["Age"].std())
            row["live_age_max"] = float(alive["Age"].max())
            row["live_frac_age_le_1"] = float(np.mean(alive["Age"] <= 1))
            row["live_frac_age_le_2"] = float(np.mean(alive["Age"] <= 2))
            row["live_frac_age_ge_5"] = float(np.mean(alive["Age"] >= 5))
            row["live_frac_age_ge_10"] = float(np.mean(alive["Age"] >= 10))

        if "LeafArea" in alive.columns:
            row["live_leaf_area_mean"] = float(alive["LeafArea"].mean())
            row["live_leaf_area_median"] = float(alive["LeafArea"].median())
            row["total_live_leaf_area_final"] = float(alive["LeafArea"].sum())

        if "NumRoots" in alive.columns:
            row["live_num_roots_mean"] = float(alive["NumRoots"].mean())
            row["live_num_roots_median"] = float(alive["NumRoots"].median())
            row["live_frac_zero_roots"] = float(np.mean(alive["NumRoots"] <= 0))

        if "RootDiamMM" in alive.columns:
            row["live_root_diam_mean"] = float(alive["RootDiamMM"].mean())
            row["live_root_diam_median"] = float(alive["RootDiamMM"].median())

    if len(final):
        live_leaf_area = (
            final.loc[final["Status"] == 1, "LeafArea"].sum()
            if "LeafArea" in final.columns
            else np.nan
        )

        for c in [
            "DeadLeafArea",
            "DeadLeafMass",
            "RootNecroVol",
            "RootNecroVolCum",
            "RootNecroMass",
            "RootNecroMassCum",
        ]:
            if c in final.columns:
                row[f"total_{c}_final"] = float(final[c].sum())

        if np.isfinite(live_leaf_area) and live_leaf_area > 0:
            if "DeadLeafMass" in final.columns:
                row["dead_leaf_mass_per_live_leaf_area"] = float(
                    final["DeadLeafMass"].sum() / live_leaf_area
                )

            if "DeadLeafArea" in final.columns:
                row["dead_leaf_area_per_live_leaf_area"] = float(
                    final["DeadLeafArea"].sum() / live_leaf_area
                )

            if "RootNecroMassCum" in final.columns:
                row["root_necro_mass_cum_per_live_leaf_area"] = float(
                    final["RootNecroMassCum"].sum() / live_leaf_area
                )

            if "RootNecroVolCum" in final.columns:
                row["root_necro_vol_cum_per_live_leaf_area"] = float(
                    final["RootNecroVolCum"].sum() / live_leaf_area
                )

    if "TillerID" in df.columns:
        row["n_unique_tillers_ever"] = int(df["TillerID"].nunique())

    if "ParentTillerID" in df.columns and "TillerID" in df.columns:
        ever = df.drop_duplicates("TillerID").copy()
        parent_counts = ever[ever["ParentTillerID"] >= 0]["ParentTillerID"].value_counts()

        row["n_reproducing_parents_ever"] = int(len(parent_counts))
        row["mean_offspring_per_reproducing_parent"] = (
            float(parent_counts.mean()) if len(parent_counts) else 0.0
        )
        row["median_offspring_per_reproducing_parent"] = (
            float(parent_counts.median()) if len(parent_counts) else 0.0
        )
        row["max_offspring_one_parent"] = (
            float(parent_counts.max()) if len(parent_counts) else 0.0
        )

        if len(ever):
            row["frac_tillers_that_reproduced"] = float(
                ever["TillerID"].isin(parent_counts.index).mean()
            )

        parent_map = {
            int(r["TillerID"]): int(r["ParentTillerID"])
            for _, r in ever[["TillerID", "ParentTillerID"]].dropna().iterrows()
        }

        depths = compute_lineage_depths(parent_map)

        if depths:
            depth_values = np.array(list(depths.values()), dtype=float)
            depth_values = depth_values[np.isfinite(depth_values)]

            row["lineage_depth_mean_ever"] = (
                float(np.mean(depth_values)) if len(depth_values) else np.nan
            )
            row["lineage_depth_max_ever"] = (
                float(np.max(depth_values)) if len(depth_values) else np.nan
            )

            if len(alive):
                live_ids = alive["TillerID"].dropna().astype(int).tolist()
                live_depths = np.array([depths.get(tid, np.nan) for tid in live_ids], dtype=float)
                live_depths = live_depths[np.isfinite(live_depths)]

                row["lineage_depth_mean_live"] = (
                    float(np.mean(live_depths)) if len(live_depths) else np.nan
                )
                row["lineage_depth_max_live"] = (
                    float(np.max(live_depths)) if len(live_depths) else np.nan
                )

    return row


def metrics_from_full_tiller_files(final_sims_dir):
    paths = find_full_tiller_files(final_sims_dir)
    rows = []

    for i, p in enumerate(paths):
        row = metrics_from_full_tiller_file(p, fallback_sim_id=i)

        if row is not None:
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# -----------------------------
# Set-level aggregation
# -----------------------------

def aggregate_sim_to_set(sim_metrics):
    row = {
        "n_sims": int(sim_metrics["sim_id"].nunique())
        if "sim_id" in sim_metrics.columns
        else len(sim_metrics),
    }

    if "is_alive_final" in sim_metrics.columns:
        vals = pd.to_numeric(sim_metrics["is_alive_final"], errors="coerce")
        row["alive_frac_final"] = float(vals.mean())
        row["extinct_frac_final"] = float(1.0 - vals.mean())

    if "has_overflow" in sim_metrics.columns:
        vals = pd.to_numeric(sim_metrics["has_overflow"], errors="coerce")
        row["overflow_frac"] = float(vals.mean())

    if "any_overflow_yearly" in sim_metrics.columns:
        vals = pd.to_numeric(sim_metrics["any_overflow_yearly"], errors="coerce")
        row["overflow_frac_yearly"] = float(vals.mean())

    if "has_missing_year" in sim_metrics.columns:
        vals = pd.to_numeric(sim_metrics["has_missing_year"], errors="coerce")
        row["missing_year_frac"] = float(vals.mean())

    diam_col = None

    for candidate in ["final_diameter", "final_diameter_yearly"]:
        if candidate in sim_metrics.columns:
            diam_col = candidate
            break

    if diam_col is not None:
        diam = pd.to_numeric(sim_metrics[diam_col], errors="coerce").dropna().to_numpy(dtype=float)

        row["sim_diameter_mean"] = float(np.mean(diam)) if len(diam) else np.nan
        row["sim_diameter_median"] = float(np.median(diam)) if len(diam) else np.nan
        row["sim_diameter_sd"] = float(np.std(diam)) if len(diam) else np.nan
        row["sim_diameter_cv"] = cv(diam)

    summarize_cols = [
        "final_n_total",
        "max_n_total",
        "cv_n_total",
        "slope_n_total_last_half",
        "final_n_alive",
        "max_n_alive",
        "mean_n_alive",
        "cv_n_alive",
        "slope_n_alive_all",
        "slope_n_alive_last_half",
        "late_absolute_alive_slope",
        "final_alive_over_peak_alive",
        "final_n_dead",
        "final_dead_to_alive_ratio",
        "total_newborns",
        "late_newborns",
        "mean_newborns_per_year",
        "max_newborns_one_year",
        "slope_newborns_last_half",
        "final_diameter_yearly",
        "max_diameter_yearly",
        "cv_diameter",
        "slope_diameter_all",
        "slope_diameter_last_half",
        "late_absolute_diameter_slope",
        "final_diameter_over_peak_diameter",
        "final_total_tillers_full",
        "final_live_tillers_full",
        "final_dead_tillers_full",
        "live_fraction_full",
        "live_radial_distance_mean",
        "live_radial_distance_sd",
        "live_radial_distance_max",
        "live_nearest_neighbor_mean",
        "live_tiller_density_xy",
        "live_radius_mean",
        "live_age_median",
        "live_frac_age_le_1",
        "live_frac_age_ge_5",
        "live_leaf_area_mean",
        "total_live_leaf_area_final",
        "live_num_roots_mean",
        "live_frac_zero_roots",
        "live_root_diam_mean",
        "dead_leaf_mass_per_live_leaf_area",
        "dead_leaf_area_per_live_leaf_area",
        "root_necro_mass_cum_per_live_leaf_area",
        "root_necro_vol_cum_per_live_leaf_area",
        "n_unique_tillers_ever",
        "n_reproducing_parents_ever",
        "mean_offspring_per_reproducing_parent",
        "max_offspring_one_parent",
        "frac_tillers_that_reproduced",
        "lineage_depth_mean_ever",
        "lineage_depth_max_ever",
        "lineage_depth_mean_live",
        "lineage_depth_max_live",
    ]

    for c in summarize_cols:
        if c not in sim_metrics.columns:
            continue

        vals = pd.to_numeric(sim_metrics[c], errors="coerce")

        row[f"{c}_mean"] = float(vals.mean())
        row[f"{c}_median"] = float(vals.median())
        row[f"{c}_sd"] = float(vals.std())

    return row


def add_training_fit_metrics_to_set_row(set_row, sim_metrics, training_diameters):
    if training_diameters is None or len(training_diameters) == 0:
        return set_row

    obs = np.asarray(training_diameters, dtype=float)
    obs = obs[np.isfinite(obs)]

    if len(obs) == 0:
        return set_row

    obs_sd = float(np.std(obs)) if len(obs) > 1 else np.nan
    obs_mean = float(np.mean(obs))
    obs_median = float(np.median(obs))

    set_row["obs_diameter_mean"] = obs_mean
    set_row["obs_diameter_median"] = obs_median
    set_row["obs_diameter_sd"] = obs_sd
    set_row["n_obs_diameters"] = int(len(obs))

    diam_col = None

    for candidate in ["final_diameter", "final_diameter_yearly"]:
        if candidate in sim_metrics.columns:
            diam_col = candidate
            break

    if diam_col is None:
        return set_row

    diam = pd.to_numeric(sim_metrics[diam_col], errors="coerce").dropna().to_numpy(dtype=float)

    set_row["diameter_wasserstein_to_training"] = wasserstein_1d(obs, diam)

    sim_sd = float(np.std(diam)) if len(diam) > 1 else np.nan

    if np.isfinite(obs_sd) and obs_sd > 0 and np.isfinite(sim_sd):
        set_row["diameter_sd_ratio_sim_obs"] = float(sim_sd / obs_sd)
        set_row["diameter_abs_sd_ratio_error"] = float(abs((sim_sd / obs_sd) - 1.0))
    else:
        set_row["diameter_sd_ratio_sim_obs"] = np.nan
        set_row["diameter_abs_sd_ratio_error"] = np.nan

    if len(diam):
        set_row["diameter_mean_error_sim_minus_obs"] = float(np.mean(diam) - obs_mean)
        set_row["diameter_median_error_sim_minus_obs"] = float(np.median(diam) - obs_median)

    return set_row


def rough_plausibility_score_for_row(row):
    """
    Quick monitoring score. Lower is generally less weird.

    This is intentionally rough, not a final objective function.
    """
    terms = []

    def add(name, weight=1.0, default=0.0):
        val = row.get(name, np.nan)

        if pd.isna(val) or not np.isfinite(val):
            val = default

        terms.append(weight * float(val))

    add("extinct_frac_final", weight=3.0)
    add("overflow_frac", weight=3.0)
    add("missing_year_frac", weight=2.0)

    add("diameter_abs_sd_ratio_error", weight=1.0)
    add("late_absolute_alive_slope_mean", weight=0.05)
    add("late_absolute_diameter_slope_mean", weight=0.05)
    add("cv_n_alive_mean", weight=1.0)

    if not terms:
        return np.nan

    return float(np.sum(terms))


# -----------------------------
# Question summary
# -----------------------------

def rewrite_question_summary(per_set_path, summary_path):
    per_set_path = Path(per_set_path)
    summary_path = Path(summary_path)

    if not per_set_path.exists():
        return

    try:
        df = pd.read_csv(per_set_path)
    except pd.errors.EmptyDataError:
        return
    except pd.errors.ParserError as e:
        print(
            f"[WARN] Could not rewrite summary because {per_set_path} is malformed: {e}",
            flush=True,
        )
        return

    if df.empty or "question" not in df.columns:
        return

    numeric_cols = [
        c for c in df.columns
        if c not in ["question", "site", "set", "set_dir"]
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    rows = []

    for question, g in df.groupby("question"):
        row = {
            "question": question,
            "n_sets_completed": int(len(g)),
        }

        for c in numeric_cols:
            vals = pd.to_numeric(g[c], errors="coerce")

            row[f"{c}_mean"] = float(vals.mean())
            row[f"{c}_median"] = float(vals.median())
            row[f"{c}_sd"] = float(vals.std())
            row[f"{c}_min"] = float(vals.min())
            row[f"{c}_max"] = float(vals.max())

        rows.append(row)

    out = pd.DataFrame(rows)

    if "rough_plausibility_score_mean" in out.columns:
        out = out.sort_values("rough_plausibility_score_mean", ascending=True)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(summary_path, index=False)


# -----------------------------
# Per-set processing
# -----------------------------

def process_one_set(hypothesis_dir, set_dir, training_diameters):
    final_sims_dir = Path(set_dir) / "final_sims"

    question = question_from_set_dir(hypothesis_dir, set_dir)
    site = site_from_set_dir(hypothesis_dir, set_dir)
    set_name = Path(set_dir).name

    summaries, yearly = read_summary_tables(final_sims_dir)

    m_summary = metrics_from_summaries(summaries)
    m_yearly = metrics_from_yearly(yearly)
    m_full = metrics_from_full_tiller_files(final_sims_dir)

    dfs = [d for d in [m_summary, m_yearly, m_full] if not d.empty]

    if not dfs:
        return None, None

    sim_metrics = dfs[0]

    for d in dfs[1:]:
        sim_metrics = sim_metrics.merge(d, on="sim_id", how="outer")

    sim_metrics.insert(0, "question", question)
    sim_metrics.insert(1, "site", site)
    sim_metrics.insert(2, "set", set_name)
    sim_metrics.insert(3, "set_dir", str(set_dir))

    set_row = aggregate_sim_to_set(sim_metrics)

    set_row["question"] = question
    set_row["site"] = site
    set_row["set"] = set_name
    set_row["set_dir"] = str(set_dir)

    set_row = add_training_fit_metrics_to_set_row(
        set_row=set_row,
        sim_metrics=sim_metrics,
        training_diameters=training_diameters,
    )

    set_row["rough_plausibility_score"] = rough_plausibility_score_for_row(set_row)

    id_cols = ["question", "site", "set", "set_dir"]

    set_df = pd.DataFrame([set_row])
    set_df = set_df[id_cols + [c for c in set_df.columns if c not in id_cols]]

    return sim_metrics, set_df


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Append deep tussock diagnostics per question and across a full hypothesis directory."
    )

    parser.add_argument(
        "hypothesis_dir",
        help="Top-level hypothesis directory containing question folders and set_### outputs.",
    )

    parser.add_argument(
        "--training-csv",
        default=None,
        help="Optional observed training CSV used for diameter Wasserstein and SD-ratio metrics.",
    )

    parser.add_argument(
        "--diam-col",
        default="diam",
        help="Observed diameter column in training CSV. Default: diam",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip set directories already present in compiled_deep_metrics_per_set.csv.",
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete previous compiled and per-question metric CSVs before starting.",
    )

    args = parser.parse_args()

    hypothesis_dir = Path(args.hypothesis_dir).resolve()

    if not hypothesis_dir.exists():
        raise FileNotFoundError(f"Hypothesis directory not found: {hypothesis_dir}")

    overall_per_sim_path = hypothesis_dir / "compiled_deep_metrics_per_sim.csv"
    overall_per_set_path = hypothesis_dir / "compiled_deep_metrics_per_set.csv"
    overall_per_question_path = hypothesis_dir / "compiled_deep_metrics_per_question_summary.csv"
    overall_progress_path = hypothesis_dir / "compiled_deep_metrics_progress_log.csv"
    overall_error_path = hypothesis_dir / "compiled_deep_metrics_error_log.txt"

    if args.fresh:
        for p in [
            overall_per_sim_path,
            overall_per_set_path,
            overall_per_question_path,
            overall_progress_path,
            overall_error_path,
        ]:
            if p.exists():
                p.unlink()

        for question_dir in hypothesis_dir.iterdir():
            if not question_dir.is_dir():
                continue

            for p in [
                question_dir / "deep_metrics_per_sim.csv",
                question_dir / "deep_metrics_per_set.csv",
                question_dir / "deep_metrics_per_question_summary.csv",
                question_dir / "deep_metrics_progress_log.csv",
                question_dir / "deep_metrics_error_log.txt",
            ]:
                if p.exists():
                    p.unlink()

    training_diameters = None

    if args.training_csv:
        training_csv = Path(args.training_csv).resolve()

        if not training_csv.exists():
            raise FileNotFoundError(f"Training CSV not found: {training_csv}")

        obs = pd.read_csv(training_csv)

        if args.diam_col not in obs.columns:
            raise ValueError(
                f"{args.diam_col} not found in training CSV. "
                f"Available columns: {list(obs.columns)}"
            )

        training_diameters = (
            pd.to_numeric(obs[args.diam_col], errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
        )

        print(f"[INFO] Loaded {len(training_diameters)} observed diameters.", flush=True)

    set_dirs = find_set_dirs(hypothesis_dir)

    if not set_dirs:
        raise RuntimeError(f"No set_### directories with final_sims found under: {hypothesis_dir}")

    print(f"[INFO] Found {len(set_dirs)} set dirs.", flush=True)

    completed = set()

    if args.resume and overall_per_set_path.exists():
        try:
            old = pd.read_csv(overall_per_set_path)
            if "set_dir" in old.columns:
                completed = set(old["set_dir"].astype(str))
            print(f"[INFO] Resume mode: found {len(completed)} completed set dirs.", flush=True)
        except pd.errors.ParserError as e:
            raise RuntimeError(
                f"Cannot resume because compiled output is malformed: {overall_per_set_path}\n"
                f"Rerun with --fresh after patching the script.\n"
                f"Original pandas error: {e}"
            )

    for i, set_dir in enumerate(set_dirs, start=1):
        set_dir = Path(set_dir).resolve()
        set_dir_str = str(set_dir)

        question = question_from_set_dir(hypothesis_dir, set_dir)
        site = site_from_set_dir(hypothesis_dir, set_dir)
        set_name = set_dir.name

        question_dir = hypothesis_dir / question

        question_per_sim_path = question_dir / "deep_metrics_per_sim.csv"
        question_per_set_path = question_dir / "deep_metrics_per_set.csv"
        question_summary_path = question_dir / "deep_metrics_per_question_summary.csv"
        question_progress_path = question_dir / "deep_metrics_progress_log.csv"
        question_error_path = question_dir / "deep_metrics_error_log.txt"

        if args.resume and set_dir_str in completed:
            print(f"[SKIP] {i}/{len(set_dirs)} already done: {set_dir}", flush=True)
            continue

        print(
            f"[RUN] {i}/{len(set_dirs)} question={question} site={site} set={set_name}",
            flush=True,
        )

        try:
            sim_metrics, set_df = process_one_set(
                hypothesis_dir=hypothesis_dir,
                set_dir=set_dir,
                training_diameters=training_diameters,
            )

            if sim_metrics is None or set_df is None:
                progress_row = {
                    "set_dir": set_dir_str,
                    "question": question,
                    "site": site,
                    "set": set_name,
                    "status": "no_usable_metrics",
                }

                append_row_csv(progress_row, question_progress_path)
                append_row_csv(progress_row, overall_progress_path)

                print(f"[WARN] No usable metrics: {set_dir}", flush=True)
                continue

            append_df_csv(sim_metrics, question_per_sim_path)
            append_df_csv(set_df, question_per_set_path)
            rewrite_question_summary(question_per_set_path, question_summary_path)

            append_df_csv(sim_metrics, overall_per_sim_path)
            append_df_csv(set_df, overall_per_set_path)
            rewrite_question_summary(overall_per_set_path, overall_per_question_path)

            score = (
                set_df["rough_plausibility_score"].iloc[0]
                if "rough_plausibility_score" in set_df.columns
                else np.nan
            )

            alive = (
                set_df["alive_frac_final"].iloc[0]
                if "alive_frac_final" in set_df.columns
                else np.nan
            )

            wdist = (
                set_df["diameter_wasserstein_to_training"].iloc[0]
                if "diameter_wasserstein_to_training" in set_df.columns
                else np.nan
            )

            progress_row = {
                "set_dir": set_dir_str,
                "question": question,
                "site": site,
                "set": set_name,
                "status": "done",
                "rough_plausibility_score": score,
                "alive_frac_final": alive,
                "diameter_wasserstein_to_training": wdist,
            }

            append_row_csv(progress_row, question_progress_path)
            append_row_csv(progress_row, overall_progress_path)

            print(
                f"[DONE] {question}/{site}/{set_name} "
                f"score={score:.4g} alive={alive:.4g} wdist={wdist:.4g}",
                flush=True,
            )

        except Exception as e:
            msg = traceback.format_exc()

            for error_path in [question_error_path, overall_error_path]:
                with open(error_path, "a") as f:
                    f.write("\n" + "=" * 80 + "\n")
                    f.write(f"ERROR in {set_dir}\n")
                    f.write(msg)

            progress_row = {
                "set_dir": set_dir_str,
                "question": question,
                "site": site,
                "set": set_name,
                "status": "error",
                "error": str(e),
            }

            append_row_csv(progress_row, question_progress_path)
            append_row_csv(progress_row, overall_progress_path)

            print(f"[ERROR] {set_dir}: {e}", flush=True)
            continue

    rewrite_question_summary(overall_per_set_path, overall_per_question_path)

    print("[DONE] Streaming analysis complete.", flush=True)
    print(f"Overall per-sim: {overall_per_sim_path}", flush=True)
    print(f"Overall per-set: {overall_per_set_path}", flush=True)
    print(f"Overall per-question summary: {overall_per_question_path}", flush=True)
    print(f"Overall progress log: {overall_progress_path}", flush=True)


if __name__ == "__main__":
    main()