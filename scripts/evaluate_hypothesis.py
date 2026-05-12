#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SUMMARY_REQUIRED_COLUMNS = [
    "sim_id",
    "final_t",
    "final_diameter",
    "alive_y",
    "rmax_y",
    "overflow_t",
    "extinct_t",
    "missing_year",
    "alive_final",
    "LeafArea",
]


RAW_SIM_REQUIRED_COLUMNS = [
    "TimeStep",
    "TillerID",
    "Radius",
    "LeafArea",
    "X",
    "Y",
    "Status",
]


FINAL_POP_REQUIRED_COLUMNS = [
    "iteration",
    "alive_tussocks_final",
    "extinct_tussocks_final",
    "overgrown_tussocks",
    "overflow_tussocks",
    "avg_tussock_diameter",
]


AUDIT_OUTPUT_FILENAMES = {
    "all_final_sim_summaries.csv",
    "model_family_summary.csv",
    "final_parameters_by_set.csv",
    "best_optimization_rows.csv",
    "all_optimization_rows.csv",
    "parameter_stability_summary.csv",
    "raw_final_timestep_population_stats.csv",
    "final_population_results_compiled.csv",
}

MODEL_PARAMETER_COLUMNS = [
    "ks",
    "kr",
    "ke",
    "bs",
    "br",
    "be",
    "c_space_survival",
    "c_space_reproduction",
    "k_crowd_survival",
    "k_crowd_reproduction",
    "k_crowd_establishment",
    "leaf_offset",
]

WEIGHTED_LOSS_COMPONENT_COLUMNS = [
    "fit_loss_weighted",
    "diameter_sd_loss_weighted",
    "live_tiller_radius_prior_weighted",
    "extinct_loss_weighted",
    "overflow_loss_weighted",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Audit final tussock simulations for viability, plausibility, field fit, "
            "parameter stability, final population outcomes, and plots."
        )
    )

    p.add_argument(
        "--hypothesis-dirs",
        nargs="*",
        required=True,
        help="Directories to evaluate/compare. Can be the whole output/subdir directory, a site directory, or one mechanism directory.",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Default: <hypothesis-dir>/audit_summary",
    )
    p.add_argument(
        "--observed-csv",
        default=None,
        help="Optional field data CSV containing observed tussock diameters.",
    )
    p.add_argument(
        "--obs-diam-col",
        default="diam",
        help="Observed diameter column name in --observed-csv. Default: diam",
    )
    p.add_argument(
        "--obs-site-col",
        default="site",
        help="Observed site column name, used when grouping by site if present. Default: site",
    )
    p.add_argument(
        "--min-alive-tillers",
        type=int,
        default=25,
        help="Minimum alive tillers at constraint year to count as viable. Default: 25",
    )
    p.add_argument(
        "--max-overflow-frac",
        type=float,
        default=0.20,
        help="Family-level failure threshold for overflow fraction. Default: 0.20",
    )
    p.add_argument(
        "--max-invalid-frac",
        type=float,
        default=0.20,
        help="Family-level failure threshold for invalid fraction. Default: 0.20",
    )
    p.add_argument(
        "--max-low-alive-frac",
        type=float,
        default=0.50,
        help="Family-level failure threshold for low alive fraction. Default: 0.50",
    )
    p.add_argument(
        "--max-leaf-bad-frac",
        type=float,
        default=0.25,
        help="Family-level warning/failure threshold for bad leaf area fraction. Default: 0.25",
    )
    p.add_argument(
        "--diam-q-low",
        type=float,
        default=0.05,
        help="Lower observed diameter quantile for rough diameter match. Default: 0.05",
    )
    p.add_argument(
        "--diam-q-high",
        type=float,
        default=0.95,
        help="Upper observed diameter quantile for rough diameter match. Default: 0.95",
    )
    p.add_argument(
        "--diam-buffer",
        type=float,
        default=0.0,
        help="Extra buffer added below/above observed diameter quantile range. Default: 0.0",
    )
    p.add_argument(
        "--max-diameter-wasserstein",
        type=float,
        default=None,
        help="Optional family-level cutoff for VIABLE_BAD_FIELD_FIT classification.",
    )
    p.add_argument(
        "--leaf-min",
        type=float,
        default=0.0,
        help="LeafArea values <= this are counted as bad. Default: 0.0",
    )
    p.add_argument(
        "--leaf-max",
        type=float,
        default=2000.0,
        help="LeafArea values >= this are counted as bad. Default: 2000.0",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plot generation.",
    )
    p.add_argument(
        "--plot-dpi",
        type=int,
        default=250,
        help="DPI for saved plots. Default: 250.",
    )

    return p.parse_args()


def wasserstein_distance_1d(x: Iterable[float], y: Iterable[float]) -> float:
    """Small dependency-free 1D Wasserstein distance."""
    x = np.asarray(list(x), dtype=float)
    y = np.asarray(list(y), dtype=float)

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if x.size == 0 or y.size == 0:
        return np.nan

    x_sorted = np.sort(x)
    y_sorted = np.sort(y)

    n = x_sorted.size
    m = y_sorted.size

    i = 0
    j = 0
    cdfx = 0.0
    cdfy = 0.0
    prev = min(x_sorted[0], y_sorted[0])
    w1 = 0.0

    while i < n or j < m:
        next_x = x_sorted[i] if i < n else np.inf
        next_y = y_sorted[j] if j < m else np.inf
        nxt = next_x if next_x < next_y else next_y

        w1 += abs(cdfx - cdfy) * (nxt - prev)

        if next_x == nxt:
            val = nxt
            while i < n and x_sorted[i] == val:
                i += 1
            cdfx = i / n

        if next_y == nxt:
            val = nxt
            while j < m and y_sorted[j] == val:
                j += 1
            cdfy = j / m

        prev = nxt

    return float(w1)


def is_set_dir_name(name: str) -> bool:
    return re.fullmatch(r"set_\d+", str(name)) is not None


def infer_set_id(path: Path) -> str:
    for part in path.parts:
        if is_set_dir_name(part):
            return part
    return ""


def infer_site_from_final_sims(final_sims_dir: Path) -> str:
    """
    Supports both layouts:

    Old:
      .../set_001/<site>/final_sims

    New:
      .../<site>/set_001/final_sims
    """
    parent = final_sims_dir.parent

    if is_set_dir_name(parent.name):
        # New layout: site/set_001/final_sims
        return parent.parent.name

    if parent.name and parent.name != "final_sims":
        # Old layout: set_001/site/final_sims
        return parent.name

    return ""


def infer_run_dir_from_final_sims(final_sims_dir: Path) -> Path:
    """
    Returns the directory containing parameters.txt, optimization_results.csv,
    final_population_results.csv, and final_sims.

    In both supported layouts this is final_sims_dir.parent.
    """
    return final_sims_dir.parent


def infer_model_family(hypothesis_dir: Path, final_sims_dir: Path) -> str:
    """
    Infer model/mechanism family while ignoring site/set/final_sims.

    Supports:

    Old:
      hypothesis/mechanism/set_001/site/final_sims
      -> mechanism

    New:
      hypothesis/mechanism/site/set_001/final_sims
      -> mechanism

    If there is no extra mechanism directory, returns hypothesis_dir.name.
    """
    try:
        rel = final_sims_dir.relative_to(hypothesis_dir)
    except ValueError:
        return hypothesis_dir.name

    parts = list(rel.parts)

    cleaned = []
    for i, part in enumerate(parts):
        if is_set_dir_name(part):
            # Old layout: mechanism/set/site/final_sims
            if i > 0:
                prev = parts[i - 1]
                # New layout has site just before set; remove it from model family.
                # Old layout has model family just before set, so keep previous cleaned parts.
                # We handle this below by checking whether final_sims follows set directly.
                pass
            break
        cleaned.append(part)

    # If new layout, relative path likely mechanism/site/set_001/final_sims.
    # The piece immediately before set_### is the site, not model family.
    set_idx = None
    for i, part in enumerate(parts):
        if is_set_dir_name(part):
            set_idx = i
            break

    if set_idx is not None:
        if set_idx + 1 < len(parts) and parts[set_idx + 1] == "final_sims":
            # New layout: .../<site>/set_001/final_sims
            model_parts = parts[:max(0, set_idx - 1)]
        else:
            # Old layout: .../set_001/<site>/final_sims
            model_parts = parts[:set_idx]

        if model_parts:
            return "/".join(model_parts)

    return hypothesis_dir.name


def infer_model_family_from_run_dir(hypothesis_dir: Path, run_dir: Path) -> str:
    """
    Infer model family from a run directory containing optimization_results.csv,
    parameters.txt, or final_population_results.csv.

    Supports:
      old: mechanism/set_001/site
      new: mechanism/site/set_001
    """
    try:
        rel = run_dir.relative_to(hypothesis_dir)
    except ValueError:
        return hypothesis_dir.name

    parts = list(rel.parts)
    set_idx = None

    for i, part in enumerate(parts):
        if is_set_dir_name(part):
            set_idx = i
            break

    if set_idx is None:
        return "/".join(parts[:-1]) if len(parts) > 1 else hypothesis_dir.name

    # New layout: mechanism/site/set_001
    # Old layout: mechanism/set_001/site
    if set_idx == len(parts) - 1:
        model_parts = parts[:max(0, set_idx - 1)]
    else:
        model_parts = parts[:set_idx]

    if model_parts:
        return "/".join(model_parts)

    return hypothesis_dir.name


def infer_site_from_run_dir(run_dir: Path) -> str:
    """
    Supports:
      old: .../set_001/site
      new: .../site/set_001
    """
    if is_set_dir_name(run_dir.name):
        return run_dir.parent.name

    return run_dir.name


def read_summary_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    for col in SUMMARY_REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    for col in SUMMARY_REQUIRED_COLUMNS:
        if col != "sim_id":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "sim_id" in df.columns:
        df["sim_id"] = pd.to_numeric(df["sim_id"], errors="coerce").astype("Int64")

    return df[SUMMARY_REQUIRED_COLUMNS].copy()


def find_final_summary_dirs(hypothesis_dir: Path) -> List[Path]:
    dirs = []

    for p in hypothesis_dir.rglob("final_sims"):
        summary_dir = p / "summaries"

        if summary_dir.is_dir() and any(summary_dir.glob("summary_*.csv")):
            dirs.append(p)

    return sorted(dirs)


def load_all_final_summaries(
    hypothesis_dir: Path,
    min_alive_tillers: int,
    leaf_min: float,
    leaf_max: float,
) -> pd.DataFrame:
    rows = []

    for final_sims_dir in find_final_summary_dirs(hypothesis_dir):
        summary_dir = final_sims_dir / "summaries"
        set_id = infer_set_id(final_sims_dir)
        site = infer_site_from_final_sims(final_sims_dir)
        model_family = infer_model_family(hypothesis_dir, final_sims_dir)

        for summary_file in sorted(summary_dir.glob("summary_*.csv")):
            df = read_summary_file(summary_file)

            df["hypothesis_dir"] = str(hypothesis_dir)
            df["model_family"] = model_family
            df["set_id"] = set_id
            df["site"] = site
            df["final_sims_dir"] = str(final_sims_dir)
            df["summary_file"] = str(summary_file)

            df["invalid"] = (
                (df["missing_year"].fillna(1).astype(int) == 1)
                | (~np.isfinite(df["final_diameter"].astype(float)))
            )

            df["overflow"] = df["overflow_t"].fillna(-1).astype(int) >= 0

            df["extinct_by_constraint"] = (
                (df["missing_year"].fillna(1).astype(int) == 0)
                & (df["alive_y"].fillna(0).astype(float) <= 0)
            )

            df["extinct_final"] = df["alive_final"].fillna(0).astype(float) <= 0

            df["low_alive"] = (
                df["alive_y"].fillna(0).astype(float) < float(min_alive_tillers)
            )

            leaf = df["LeafArea"].astype(float)
            df["leaf_bad"] = np.isfinite(leaf) & (
                (leaf <= leaf_min) | (leaf >= leaf_max)
            )

            df["viability_pass"] = (
                (~df["invalid"])
                & (~df["overflow"])
                & (~df["low_alive"])
            )

            rows.append(df)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def load_observed_diameters(
    observed_csv: Optional[str],
    obs_diam_col: str,
    obs_site_col: str,
) -> Tuple[pd.DataFrame, np.ndarray]:
    if observed_csv is None:
        return pd.DataFrame(), np.array([], dtype=float)

    obs_path = Path(observed_csv)

    if not obs_path.exists():
        raise FileNotFoundError(f"Observed CSV not found: {obs_path}")

    obs = pd.read_csv(obs_path)

    if obs_diam_col not in obs.columns:
        raise ValueError(
            f"Observed diameter column '{obs_diam_col}' not found in {obs_path}"
        )

    obs["_obs_diam"] = pd.to_numeric(obs[obs_diam_col], errors="coerce")
    obs = obs[np.isfinite(obs["_obs_diam"])].copy()

    if obs_site_col not in obs.columns:
        obs[obs_site_col] = "ALL"

    all_diams = obs["_obs_diam"].to_numpy(dtype=float)

    return obs, all_diams


def observed_for_group(
    obs: pd.DataFrame,
    all_obs_diams: np.ndarray,
    site: str,
    obs_site_col: str,
) -> np.ndarray:
    if obs.empty:
        return np.array([], dtype=float)

    if site and site != "ALL" and obs_site_col in obs.columns:
        site_obs = obs.loc[
            obs[obs_site_col].astype(str) == str(site),
            "_obs_diam",
        ].to_numpy(dtype=float)

        if site_obs.size > 0:
            return site_obs

    return all_obs_diams


def add_diameter_match_flags(
    sim_df: pd.DataFrame,
    obs: pd.DataFrame,
    all_obs_diams: np.ndarray,
    obs_site_col: str,
    q_low: float,
    q_high: float,
    buffer: float,
) -> pd.DataFrame:
    sim_df = sim_df.copy()

    sim_df["diameter_match"] = False
    sim_df["diameter_lower"] = np.nan
    sim_df["diameter_upper"] = np.nan

    if sim_df.empty or all_obs_diams.size == 0:
        return sim_df

    group_cols = ["model_family", "set_id", "site", "final_sims_dir"]

    for keys, idx in sim_df.groupby(group_cols, dropna=False).groups.items():
        site = keys[2]

        obs_diams = observed_for_group(obs, all_obs_diams, site, obs_site_col)
        obs_diams = obs_diams[np.isfinite(obs_diams)]

        if obs_diams.size == 0:
            continue

        lo = float(np.quantile(obs_diams, q_low) - buffer)
        hi = float(np.quantile(obs_diams, q_high) + buffer)

        diam = sim_df.loc[idx, "final_diameter"].astype(float)
        viable = sim_df.loc[idx, "viability_pass"].astype(bool)

        sim_df.loc[idx, "diameter_lower"] = lo
        sim_df.loc[idx, "diameter_upper"] = hi
        sim_df.loc[idx, "diameter_match"] = viable & (diam >= lo) & (diam <= hi)

    return sim_df


def classify_family(
    row: pd.Series,
    max_invalid: float,
    max_overflow: float,
    max_low_alive: float,
    max_leaf_bad: float,
    max_wass: Optional[float],
) -> str:
    if row["invalid_frac"] > max_invalid:
        return "FAIL_INVALID"

    if row["overflow_frac"] > max_overflow:
        return "FAIL_OVERFLOW"

    if row["low_alive_frac"] > max_low_alive:
        return "FAIL_LOW_ALIVE"

    if row["viability_pass_frac"] <= 0:
        return "FAIL_NO_VIABLE_SIMS"

    if row["leaf_bad_frac"] > max_leaf_bad:
        return "VIABLE_BUT_LEAF_WEIRD"

    if np.isfinite(row.get("diameter_wasserstein", np.nan)):
        if max_wass is not None and row["diameter_wasserstein"] > max_wass:
            return "VIABLE_BAD_FIELD_FIT"

        if row.get("diameter_match_frac", 0.0) <= 0:
            return "VIABLE_NO_DIAMETER_MATCH"

        return "PLAUSIBLE_BY_SCREEN"

    return "VIABLE_NO_FIELD_DATA"


def summarize_families(
    sim_df: pd.DataFrame,
    obs: pd.DataFrame,
    all_obs_diams: np.ndarray,
    obs_site_col: str,
    args: argparse.Namespace,
) -> pd.DataFrame:
    if sim_df.empty:
        return pd.DataFrame()

    rows = []
    group_cols = ["model_family", "site"]

    for (model_family, site), g in sim_df.groupby(group_cols, dropna=False):
        n = len(g)
        viable = g["viability_pass"].astype(bool)
        viable_g = g.loc[viable].copy()

        obs_diams = observed_for_group(obs, all_obs_diams, str(site), obs_site_col)
        obs_diams = obs_diams[np.isfinite(obs_diams)]

        sim_diam_viable = viable_g["final_diameter"].astype(float).to_numpy()
        sim_diam_viable = sim_diam_viable[np.isfinite(sim_diam_viable)]

        if obs_diams.size > 0 and sim_diam_viable.size > 0:
            wass = wasserstein_distance_1d(obs_diams, sim_diam_viable)
            obs_mean = float(np.mean(obs_diams))
            obs_sd = float(np.std(obs_diams, ddof=1)) if obs_diams.size > 1 else 0.0
        else:
            wass = np.nan
            obs_mean = np.nan
            obs_sd = np.nan

        row = {
            "model_family": model_family,
            "site": site,
            "n_sims": int(n),
            "n_sets": int(g["set_id"].nunique()),
            "invalid_frac": float(g["invalid"].mean()),
            "overflow_frac": float(g["overflow"].mean()),
            "extinct_by_constraint_frac": float(g["extinct_by_constraint"].mean()),
            "extinct_final_frac": float(g["extinct_final"].mean()),
            "low_alive_frac": float(g["low_alive"].mean()),
            "leaf_bad_frac": float(g["leaf_bad"].mean()),
            "viability_pass_frac": float(g["viability_pass"].mean()),
            "diameter_match_frac": (
                float(g["diameter_match"].mean())
                if "diameter_match" in g
                else np.nan
            ),
            "diameter_match_frac_among_viable": (
                float(viable_g["diameter_match"].mean())
                if "diameter_match" in viable_g and len(viable_g) > 0
                else np.nan
            ),
            "diameter_wasserstein": wass,
            "sim_final_diameter_mean_viable": (
                float(np.mean(sim_diam_viable))
                if sim_diam_viable.size > 0
                else np.nan
            ),
            "sim_final_diameter_sd_viable": (
                float(np.std(sim_diam_viable, ddof=1))
                if sim_diam_viable.size > 1
                else np.nan
            ),
            "obs_diameter_mean": obs_mean,
            "obs_diameter_sd": obs_sd,
            "alive_y_mean": float(g["alive_y"].mean()),
            "alive_final_mean": float(g["alive_final"].mean()),
            "final_diameter_mean_all": float(g["final_diameter"].mean()),
            "final_diameter_sd_all": float(g["final_diameter"].std()),
        }

        row["classification"] = classify_family(
            pd.Series(row),
            max_invalid=args.max_invalid_frac,
            max_overflow=args.max_overflow_frac,
            max_low_alive=args.max_low_alive_frac,
            max_leaf_bad=args.max_leaf_bad_frac,
            max_wass=args.max_diameter_wasserstein,
        )

        rows.append(row)

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values(
            by=[
                "classification",
                "viability_pass_frac",
                "diameter_match_frac_among_viable",
                "diameter_wasserstein",
            ],
            ascending=[True, False, False, True],
            na_position="last",
        )

    return out


def read_parameter_file(path: Path) -> Dict[str, float]:
    params: Dict[str, float] = {}

    if not path.exists():
        return params

    with path.open("r") as f:
        for raw in f:
            line = raw.strip()

            if not line or line.startswith("#") or line.startswith(";") or "=" not in line:
                continue

            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()

            try:
                params[key] = float(val)
            except ValueError:
                continue

    return params


def collect_final_parameters(hypothesis_dir: Path) -> pd.DataFrame:
    rows = []

    for final_sims_dir in find_final_summary_dirs(hypothesis_dir):
        run_dir = infer_run_dir_from_final_sims(final_sims_dir)
        param_file = run_dir / "parameters.txt"

        if not param_file.exists():
            continue

        params = read_parameter_file(param_file)

        if not params:
            continue

        row = {
            "model_family": infer_model_family(hypothesis_dir, final_sims_dir),
            "set_id": infer_set_id(final_sims_dir),
            "site": infer_site_from_final_sims(final_sims_dir),
            "final_sims_dir": str(final_sims_dir),
            "parameter_file": str(param_file),
        }

        row.update(params)
        rows.append(row)

    return pd.DataFrame(rows)


def collect_best_optimization_rows(hypothesis_dir: Path) -> pd.DataFrame:
    rows = []

    for opt_file in sorted(hypothesis_dir.rglob("optimization_results.csv")):
        df = pd.read_csv(opt_file)

        if df.empty or "loss" not in df.columns:
            continue

        df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
        df = df[np.isfinite(df["loss"])]

        if df.empty:
            continue

        best = df.loc[df["loss"].idxmin()].to_dict()

        run_dir = opt_file.parent

        row = {
            "model_family": infer_model_family_from_run_dir(hypothesis_dir, run_dir),
            "set_id": infer_set_id(opt_file),
            "site": infer_site_from_run_dir(run_dir),
            "optimization_file": str(opt_file),
        }

        row.update(best)
        rows.append(row)

    return pd.DataFrame(rows)


def collect_all_optimization_rows(hypothesis_dir: Path) -> pd.DataFrame:
    rows = []

    for opt_file in sorted(hypothesis_dir.rglob("optimization_results.csv")):
        df = pd.read_csv(opt_file)

        if df.empty:
            continue

        run_dir = opt_file.parent

        df = df.copy()
        df["model_family"] = infer_model_family_from_run_dir(hypothesis_dir, run_dir)
        df["set_id"] = infer_set_id(opt_file)
        df["site"] = infer_site_from_run_dir(run_dir)
        df["optimization_file"] = str(opt_file)

        possible_iter_cols = [
            "iteration",
            "iter",
            "generation",
            "gen",
            "step",
            "trial",
            "n_iter",
        ]

        iteration_col = None

        for col in possible_iter_cols:
            if col in df.columns:
                iteration_col = col
                break

        if iteration_col is None:
            df["iteration"] = np.arange(len(df), dtype=int)
        elif iteration_col != "iteration":
            df["iteration"] = pd.to_numeric(df[iteration_col], errors="coerce")
            missing = ~np.isfinite(df["iteration"])
            df.loc[missing, "iteration"] = np.arange(missing.sum(), dtype=int)

        rows.append(df)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)

    if "loss" in out.columns:
        out["loss"] = pd.to_numeric(out["loss"], errors="coerce")

    out["iteration"] = pd.to_numeric(out["iteration"], errors="coerce")

    return out


def summarize_parameter_stability(params_df: pd.DataFrame) -> pd.DataFrame:
    if params_df.empty:
        return pd.DataFrame()

    params_df = params_df.copy()

    meta_cols = {
        "model_family",
        "set_id",
        "site",
        "final_sims_dir",
        "parameter_file",
    }

    param_cols = [c for c in params_df.columns if c not in meta_cols]

    numeric_cols = []

    for c in param_cols:
        vals = pd.to_numeric(params_df[c], errors="coerce")

        if np.isfinite(vals).any():
            params_df[c] = vals
            numeric_cols.append(c)

    rows = []

    for (model_family, site), g in params_df.groupby(["model_family", "site"], dropna=False):
        for param in numeric_cols:
            vals = pd.to_numeric(g[param], errors="coerce").dropna()

            if vals.empty:
                continue

            mean = float(vals.mean())
            sd = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            cv = float(sd / abs(mean)) if mean != 0 else np.nan

            rows.append({
                "model_family": model_family,
                "site": site,
                "parameter": param,
                "n": int(len(vals)),
                "mean": mean,
                "sd": sd,
                "cv_abs": cv,
                "min": float(vals.min()),
                "median": float(vals.median()),
                "max": float(vals.max()),
            })

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values(["model_family", "site", "parameter"])

    return out


def read_raw_sim_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    for col in RAW_SIM_REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    for col in RAW_SIM_REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def find_raw_final_sim_files(final_sims_dir: Path) -> List[Path]:
    """
    Find raw per-tiller simulation CSV files inside a final_sims directory.

    Excludes summary files and audit output files.
    """
    files = []

    for p in sorted(final_sims_dir.rglob("*.csv")):
        name = p.name.lower()
        parts = {part.lower() for part in p.parts}

        if "summaries" in parts:
            continue

        if name.startswith("summary_"):
            continue

        if name in AUDIT_OUTPUT_FILENAMES:
            continue

        files.append(p)

    return files


def load_raw_final_timestep_population_stats(hypothesis_dir: Path) -> pd.DataFrame:
    """
    Load raw per-tiller final simulation CSVs and summarize only the last timestep
    of each simulation.

    Important:
      - Radius in the raw CSV is tiller radius, not tussock radius.
      - Tussock radius is estimated from final alive tiller X/Y positions.
      - Final alive tillers are counted from Status == 1 at the last timestep.
    """
    rows = []

    for final_sims_dir in find_final_summary_dirs(hypothesis_dir):
        set_id = infer_set_id(final_sims_dir)
        site = infer_site_from_final_sims(final_sims_dir)
        model_family = infer_model_family(hypothesis_dir, final_sims_dir)

        raw_files = find_raw_final_sim_files(final_sims_dir)

        for sim_file in raw_files:
            df = read_raw_sim_file(sim_file)

            if df.empty or "TimeStep" not in df.columns:
                continue

            time_vals = df["TimeStep"].to_numpy(dtype=float)
            time_vals = time_vals[np.isfinite(time_vals)]

            if time_vals.size == 0:
                continue

            final_t = np.max(time_vals)
            final_df = df.loc[df["TimeStep"] == final_t].copy()

            if final_df.empty:
                continue

            alive_df = final_df.loc[final_df["Status"] == 1].copy()
            n_alive_final = int(len(alive_df))

            x = alive_df["X"].to_numpy(dtype=float)
            y = alive_df["Y"].to_numpy(dtype=float)

            keep_xy = np.isfinite(x) & np.isfinite(y)
            x = x[keep_xy]
            y = y[keep_xy]

            if x.size > 0:
                dist_from_origin = np.sqrt(x**2 + y**2)
                tussock_radius = float(np.max(dist_from_origin))
                tussock_diameter = float(2.0 * tussock_radius)

                x_span = float(np.max(x) - np.min(x))
                y_span = float(np.max(y) - np.min(y))
                spatial_diameter = float(max(x_span, y_span))
            else:
                tussock_radius = np.nan
                tussock_diameter = np.nan
                spatial_diameter = np.nan

            leaf = alive_df["LeafArea"].to_numpy(dtype=float)
            leaf = leaf[np.isfinite(leaf)]

            tiller_radius = alive_df["Radius"].to_numpy(dtype=float)
            tiller_radius = tiller_radius[np.isfinite(tiller_radius)]

            row = {
                "model_family": model_family,
                "set_id": set_id,
                "site": site,
                "final_sims_dir": str(final_sims_dir),
                "sim_file": str(sim_file),
                "final_timestep": float(final_t),
                "alive_final_from_raw": n_alive_final,
                "tussock_radius_from_xy": tussock_radius,
                "tussock_diameter_from_xy": tussock_diameter,
                "spatial_diameter_xy_span": spatial_diameter,
                "mean_leaf_area_alive_final": (
                    float(np.mean(leaf)) if leaf.size > 0 else np.nan
                ),
                "median_leaf_area_alive_final": (
                    float(np.median(leaf)) if leaf.size > 0 else np.nan
                ),
                "mean_tiller_radius_alive_final": (
                    float(np.mean(tiller_radius)) if tiller_radius.size > 0 else np.nan
                ),
                "median_tiller_radius_alive_final": (
                    float(np.median(tiller_radius)) if tiller_radius.size > 0 else np.nan
                ),
            }

            rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def find_final_population_result_files(hypothesis_dir: Path) -> List[Path]:
    files = []

    for p in sorted(hypothesis_dir.rglob("*.csv")):
        name = p.name.lower()

        if name.startswith("final_population_results"):
            files.append(p)

    return files


def load_final_population_results(hypothesis_dir: Path) -> pd.DataFrame:
    rows = []

    for result_file in find_final_population_result_files(hypothesis_dir):
        df = pd.read_csv(result_file)

        for col in FINAL_POP_REQUIRED_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan

        for col in FINAL_POP_REQUIRED_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        run_dir = result_file.parent

        df["model_family"] = infer_model_family_from_run_dir(hypothesis_dir, run_dir)
        df["set_id"] = infer_set_id(result_file)
        df["site"] = infer_site_from_run_dir(run_dir)
        df["final_population_result_file"] = str(result_file)

        rows.append(df)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)

    count_cols = [
        "alive_tussocks_final",
        "extinct_tussocks_final",
        "overgrown_tussocks",
        "overflow_tussocks",
    ]

    out["total_tussocks_classified"] = out[count_cols].sum(axis=1)

    out["prop_alive"] = np.where(
        out["total_tussocks_classified"] > 0,
        out["alive_tussocks_final"] / out["total_tussocks_classified"],
        np.nan,
    )
    out["prop_extinct"] = np.where(
        out["total_tussocks_classified"] > 0,
        out["extinct_tussocks_final"] / out["total_tussocks_classified"],
        np.nan,
    )
    out["prop_overgrown"] = np.where(
        out["total_tussocks_classified"] > 0,
        out["overgrown_tussocks"] / out["total_tussocks_classified"],
        np.nan,
    )
    out["prop_overflow"] = np.where(
        out["total_tussocks_classified"] > 0,
        out["overflow_tussocks"] / out["total_tussocks_classified"],
        np.nan,
    )

    return out


def numeric_optimized_parameter_columns(opt_df: pd.DataFrame) -> List[str]:
    """
    Return only actual optimized model parameter columns from optimization_results.csv.

    This intentionally excludes total loss, raw loss terms, weighted loss terms,
    diagnostics, and stored loss weights.
    """
    if opt_df.empty:
        return []

    cols = []

    for col in MODEL_PARAMETER_COLUMNS:
        if col not in opt_df.columns:
            continue

        vals = pd.to_numeric(opt_df[col], errors="coerce")

        if np.isfinite(vals).any():
            cols.append(col)

    return cols


def final_parameter_columns(final_params: pd.DataFrame) -> List[str]:
    if final_params.empty:
        return []

    meta_cols = {
        "model_family",
        "set_id",
        "site",
        "final_sims_dir",
        "parameter_file",
    }

    cols = []

    for col in final_params.columns:
        if col in meta_cols:
            continue

        vals = pd.to_numeric(final_params[col], errors="coerce")

        if np.isfinite(vals).any():
            cols.append(col)

    return cols


def choose_subplot_grid(n: int) -> Tuple[int, int]:
    if n <= 0:
        return 0, 0

    ncols = int(math.ceil(math.sqrt(n)))
    nrows = int(math.ceil(n / ncols))

    return nrows, ncols


def save_empty_plot(path: Path, title: str, message: str, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.set_title(title)
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=ax.transAxes,
        wrap=True,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_optimization_parameter_traces(
    opt_df: pd.DataFrame,
    out_path: Path,
    dpi: int,
) -> None:
    """
    Plot total loss plus optimized model parameters.

    One subplot per variable. One line per set/site/model_family optimization run.
    This plot intentionally excludes raw loss components, weighted loss components,
    diagnostics, and stored weight columns.
    """
    if opt_df.empty:
        save_empty_plot(
            out_path,
            "Optimization traces: loss and parameters",
            "No optimization_results.csv files were found.",
            dpi,
        )
        return

    plot_cols = []

    if "loss" in opt_df.columns:
        loss_vals = pd.to_numeric(opt_df["loss"], errors="coerce")
        if np.isfinite(loss_vals).any():
            plot_cols.append("loss")

    plot_cols.extend(numeric_optimized_parameter_columns(opt_df))
    plot_cols = list(dict.fromkeys(plot_cols))

    if not plot_cols:
        save_empty_plot(
            out_path,
            "Optimization traces: loss and parameters",
            "No total loss or optimized parameter columns were found.",
            dpi,
        )
        return

    nrows, ncols = choose_subplot_grid(len(plot_cols))
    fig_width = max(10, 4.2 * ncols)
    fig_height = max(6, 3.2 * nrows)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )

    axes_flat = axes.ravel()
    group_cols = ["model_family", "site", "set_id", "optimization_file"]

    for ax, col in zip(axes_flat, plot_cols):
        for _, g in opt_df.groupby(group_cols, dropna=False):
            x = pd.to_numeric(g["iteration"], errors="coerce")
            y = pd.to_numeric(g[col], errors="coerce")

            keep = np.isfinite(x) & np.isfinite(y)

            if keep.sum() == 0:
                continue

            gg = pd.DataFrame({"x": x[keep], "y": y[keep]}).sort_values("x")

            ax.plot(
                gg["x"].to_numpy(),
                gg["y"].to_numpy(),
                linewidth=0.9,
                alpha=0.65,
            )

        ax.set_title(col)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.25)

    for ax in axes_flat[len(plot_cols):]:
        ax.axis("off")

    fig.suptitle("Optimization traces: total loss and optimized parameters", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_optimization_weighted_loss_traces(
    opt_df: pd.DataFrame,
    out_path: Path,
    dpi: int,
) -> None:
    """
    Plot total loss plus weighted individual loss components.

    This intentionally includes only:
      - loss
      - selected *_weighted component columns

    It excludes raw terms, diagnostic terms, weights, and parameters.
    """
    if opt_df.empty:
        save_empty_plot(
            out_path,
            "Optimization traces: weighted loss components",
            "No optimization_results.csv files were found.",
            dpi,
        )
        return

    plot_cols = []

    if "loss" in opt_df.columns:
        loss_vals = pd.to_numeric(opt_df["loss"], errors="coerce")
        if np.isfinite(loss_vals).any():
            plot_cols.append("loss")

    for col in WEIGHTED_LOSS_COMPONENT_COLUMNS:
        if col not in opt_df.columns:
            continue

        vals = pd.to_numeric(opt_df[col], errors="coerce")
        if np.isfinite(vals).any():
            plot_cols.append(col)

    plot_cols = list(dict.fromkeys(plot_cols))

    if not plot_cols:
        save_empty_plot(
            out_path,
            "Optimization traces: weighted loss components",
            "No total loss or weighted loss component columns were found.",
            dpi,
        )
        return

    nrows, ncols = choose_subplot_grid(len(plot_cols))
    fig_width = max(10, 4.2 * ncols)
    fig_height = max(6, 3.2 * nrows)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )

    axes_flat = axes.ravel()
    group_cols = ["model_family", "site", "set_id", "optimization_file"]

    for ax, col in zip(axes_flat, plot_cols):
        for _, g in opt_df.groupby(group_cols, dropna=False):
            x = pd.to_numeric(g["iteration"], errors="coerce")
            y = pd.to_numeric(g[col], errors="coerce")

            keep = np.isfinite(x) & np.isfinite(y)

            if keep.sum() == 0:
                continue

            gg = pd.DataFrame({"x": x[keep], "y": y[keep]}).sort_values("x")

            ax.plot(
                gg["x"].to_numpy(),
                gg["y"].to_numpy(),
                linewidth=0.9,
                alpha=0.65,
            )

        ax.set_title(col)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.25)

    for ax in axes_flat[len(plot_cols):]:
        ax.axis("off")

    fig.suptitle("Optimization traces: total loss and weighted loss components", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_optimization_traces(
    opt_df: pd.DataFrame,
    out_path: Path,
    dpi: int,
) -> None:
    """
    Backward-compatible wrapper. New scripts should call the two split trace plots.
    """
    plot_optimization_parameter_traces(opt_df=opt_df, out_path=out_path, dpi=dpi)


def plot_optimized_parameter_distributions(
    final_params: pd.DataFrame,
    opt_df: pd.DataFrame,
    out_path: Path,
    dpi: int,
) -> None:
    if final_params.empty:
        save_empty_plot(
            out_path,
            "Optimized parameter distributions",
            "No final parameter files were found.",
            dpi,
        )
        return

    optimized_cols = numeric_optimized_parameter_columns(opt_df)
    available_final_cols = set(final_parameter_columns(final_params))
    param_cols = [c for c in optimized_cols if c in available_final_cols]

    if not param_cols:
        save_empty_plot(
            out_path,
            "Optimized parameter distributions",
            "No optimized parameter columns were found in both optimization traces and final parameter files.",
            dpi,
        )
        return

    data_scaled = []
    point_x = []
    point_y = []

    for i, param in enumerate(param_cols, start=1):
        vals = pd.to_numeric(final_params[param], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]

        if vals.size == 0:
            data_scaled.append(np.array([], dtype=float))
            continue

        if vals.size > 1:
            sd = np.std(vals, ddof=1)
        else:
            sd = 0.0

        if sd > 0:
            scaled = (vals - np.mean(vals)) / sd
        else:
            scaled = vals - np.mean(vals)

        data_scaled.append(scaled)

        if scaled.size == 1:
            offsets = np.array([0.0])
        else:
            offsets = np.linspace(-0.08, 0.08, scaled.size)

        point_x.extend(i + offsets)
        point_y.extend(scaled)

    fig_width = max(10, 0.55 * len(param_cols))
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    ax.scatter(
        point_x,
        point_y,
        s=12,
        alpha=0.35,
        edgecolors="none",
        zorder=1,
    )

    nonempty_positions = []
    nonempty_data = []

    for i, vals in enumerate(data_scaled, start=1):
        if vals.size > 0:
            nonempty_positions.append(i)
            nonempty_data.append(vals)

    if nonempty_data:
        parts = ax.violinplot(
            nonempty_data,
            positions=nonempty_positions,
            showmeans=False,
            showmedians=True,
            showextrema=False,
        )

        for body in parts["bodies"]:
            body.set_alpha(0.65)
            body.set_zorder(2)

        if "cmedians" in parts:
            parts["cmedians"].set_linewidth(1.5)
            parts["cmedians"].set_zorder(3)

    ax.axhline(0, linewidth=1, alpha=0.35)
    ax.set_xticks(np.arange(1, len(param_cols) + 1))
    ax.set_xticklabels(param_cols, rotation=60, ha="right")
    ax.set_ylabel("Scaled optimized parameter value\nz-score within each parameter")
    ax.set_title("Optimized parameter distributions across final parameter sets")
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_final_population_stats(
    raw_pop_df: pd.DataFrame,
    out_path: Path,
    dpi: int,
) -> None:
    if raw_pop_df.empty:
        save_empty_plot(
            out_path,
            "Final tussock population stats",
            "No raw final simulation CSVs were found.",
            dpi,
        )
        return

    df = raw_pop_df.copy()

    for col in [
        "tussock_diameter_from_xy",
        "alive_final_from_raw",
        "tussock_radius_from_xy",
        "mean_leaf_area_alive_final",
        "mean_tiller_radius_alive_final",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 9))
    ax1, ax2, ax3, ax4 = axes.ravel()

    diam = df["tussock_diameter_from_xy"].to_numpy(dtype=float)
    diam = diam[np.isfinite(diam)]

    if diam.size > 0:
        ax1.hist(diam, bins=40, alpha=0.8)
        ax1.axvline(np.mean(diam), linewidth=1.5, alpha=0.8)

    ax1.set_title("Final tussock diameter from X/Y")
    ax1.set_xlabel("Final tussock diameter")
    ax1.set_ylabel("Number of final sims")
    ax1.grid(True, alpha=0.25)

    alive_final = df["alive_final_from_raw"].to_numpy(dtype=float)
    alive_final = alive_final[np.isfinite(alive_final)]

    if alive_final.size > 0:
        ax2.hist(alive_final, bins=40, alpha=0.8)
        ax2.axvline(np.mean(alive_final), linewidth=1.5, alpha=0.8)

    ax2.set_title("Alive tillers at final timestep")
    ax2.set_xlabel("Final alive tillers")
    ax2.set_ylabel("Number of final sims")
    ax2.grid(True, alpha=0.25)

    radius = df["tussock_radius_from_xy"].to_numpy(dtype=float)
    alive_final_for_scatter = df["alive_final_from_raw"].to_numpy(dtype=float)

    keep = np.isfinite(radius) & np.isfinite(alive_final_for_scatter)

    if keep.sum() > 0:
        ax3.scatter(
            radius[keep],
            alive_final_for_scatter[keep],
            s=13,
            alpha=0.35,
            edgecolors="none",
        )

    ax3.set_title("Final alive tillers vs tussock radius")
    ax3.set_xlabel("Tussock radius from X/Y")
    ax3.set_ylabel("Final alive tillers")
    ax3.grid(True, alpha=0.25)

    leaf = df["mean_leaf_area_alive_final"].to_numpy(dtype=float)
    leaf = leaf[np.isfinite(leaf)]

    if leaf.size > 0:
        ax4.hist(leaf, bins=40, alpha=0.8)
        ax4.axvline(np.mean(leaf), linewidth=1.5, alpha=0.8)

    ax4.set_title("Mean LeafArea of alive tillers at final timestep")
    ax4.set_xlabel("Mean final alive-tiller LeafArea")
    ax4.set_ylabel("Number of final sims")
    ax4.grid(True, alpha=0.25)

    fig.suptitle("Final tussock population stats: raw final timestep across all sets", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_final_population_outcomes(
    final_pop_df: pd.DataFrame,
    out_path: Path,
    dpi: int,
) -> None:
    if final_pop_df.empty:
        save_empty_plot(
            out_path,
            "Final population outcomes",
            "No final_population_results*.csv files were found.",
            dpi,
        )
        return

    df = final_pop_df.copy()

    for col in [
        "iteration",
        "prop_alive",
        "prop_extinct",
        "prop_overgrown",
        "prop_overflow",
        "avg_tussock_diameter",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 9))
    ax1, ax2, ax3, ax4 = axes.ravel()

    group_cols = ["model_family", "site", "set_id", "final_population_result_file"]

    outcome_specs = [
        ("prop_alive", "Alive tussocks", ax1),
        ("prop_extinct", "Extinct tussocks", ax2),
        ("prop_overgrown", "Overgrown tussocks", ax3),
        ("prop_overflow", "Overflow tussocks", ax4),
    ]

    for prop_col, title, ax in outcome_specs:
        if prop_col not in df.columns:
            ax.set_title(title)
            ax.text(
                0.5,
                0.5,
                f"{prop_col} not found",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        for _, g in df.groupby(group_cols, dropna=False):
            x = pd.to_numeric(g["iteration"], errors="coerce")
            y = pd.to_numeric(g[prop_col], errors="coerce")

            keep = np.isfinite(x) & np.isfinite(y)

            if keep.sum() == 0:
                continue

            gg = pd.DataFrame({"x": x[keep], "y": y[keep]}).sort_values("x")

            ax.plot(
                gg["x"].to_numpy(),
                gg["y"].to_numpy(),
                linewidth=1.0,
                alpha=0.65,
            )

        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Proportion")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25)

    fig.suptitle("Final population outcomes across mechanisms/sets", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_final_population_outcome_summary(
    final_pop_df: pd.DataFrame,
    out_path: Path,
    dpi: int,
) -> None:
    if final_pop_df.empty:
        save_empty_plot(
            out_path,
            "Final population outcome summary",
            "No final_population_results*.csv files were found.",
            dpi,
        )
        return

    df = final_pop_df.copy()

    for col in ["iteration", "prop_alive", "avg_tussock_diameter"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    group_cols = ["model_family", "site", "set_id", "final_population_result_file"]

    last_rows = []

    for _, g in df.groupby(group_cols, dropna=False):
        g = g[np.isfinite(g["iteration"])].copy()

        if g.empty:
            continue

        g = g.sort_values("iteration")
        last_rows.append(g.iloc[-1])

    if not last_rows:
        save_empty_plot(
            out_path,
            "Final population outcome summary",
            "No valid final iterations were found.",
            dpi,
        )
        return

    last = pd.DataFrame(last_rows)

    x = pd.to_numeric(last["prop_alive"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(last["avg_tussock_diameter"], errors="coerce").to_numpy(dtype=float)

    keep = np.isfinite(x) & np.isfinite(y)

    fig, ax = plt.subplots(figsize=(7, 6))

    if keep.sum() > 0:
        ax.scatter(
            x[keep],
            y[keep],
            s=28,
            alpha=0.6,
            edgecolors="none",
        )

    ax.set_title("Mechanism screen: viability vs tussock size")
    ax.set_xlabel("Final proportion alive")
    ax.set_ylabel("Final average tussock diameter")
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_final_population_counts(
    final_pop_df: pd.DataFrame,
    out_path: Path,
    dpi: int,
    hypothesis_label: str,
) -> None:
    if final_pop_df.empty:
        save_empty_plot(
            out_path,
            "Final population counts",
            "No final population data found.",
            dpi,
        )
        return

    df = final_pop_df.copy()

    grouped = pd.DataFrame([{
        'hypothesis': hypothesis_label,
        'alive_tussocks_final': float(df['alive_tussocks_final'].mean(skipna=True)),
        'extinct_tussocks_final': float(df['extinct_tussocks_final'].mean(skipna=True)),
        'overflow_tussocks': float(df['overflow_tussocks'].mean(skipna=True)),
    }])

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(grouped))
    width = 0.25

    ax.bar(x - width, grouped['alive_tussocks_final'], width, label='Alive', alpha=0.8)
    ax.bar(x, grouped['extinct_tussocks_final'], width, label='Extinct', alpha=0.8)
    ax.bar(x + width, grouped['overflow_tussocks'], width, label='Overflow', alpha=0.8)

    ax.set_xlabel('Hypothesis')
    ax.set_ylabel('Average Count')
    ax.set_title('Final Population Counts by Hypothesis')
    ax.set_xticks(x)
    ax.set_xticklabels(grouped['hypothesis'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_time_series(
    final_pop_df: pd.DataFrame,
    out_path: Path,
    dpi: int,
    hypothesis_label: str,
) -> None:
    if final_pop_df.empty:
        save_empty_plot(
            out_path,
            "Time series of population outcomes",
            "No final population data found.",
            dpi,
        )
        return

    df = final_pop_df.copy()

    for col in [
        "iteration",
        "prop_alive",
        "prop_extinct",
        "prop_overgrown",
        "prop_overflow",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 9))
    ax1, ax2, ax3, ax4 = axes.ravel()

    group_cols = ["model_family", "site", "set_id", "final_population_result_file"]

    outcome_specs = [
        ("prop_alive", "Alive tussocks", ax1),
        ("prop_extinct", "Extinct tussocks", ax2),
        ("prop_overgrown", "Overgrown tussocks", ax3),
        ("prop_overflow", "Overflow tussocks", ax4),
    ]

    for prop_col, title, ax in outcome_specs:
        if prop_col not in df.columns:
            ax.set_title(title)
            ax.text(
                0.5,
                0.5,
                f"{prop_col} not found",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        for _, g in df.groupby(group_cols, dropna=False):
            x = pd.to_numeric(g["iteration"], errors="coerce")
            y = pd.to_numeric(g[prop_col], errors="coerce")

            keep = np.isfinite(x) & np.isfinite(y)

            if keep.sum() == 0:
                continue

            gg = pd.DataFrame({"x": x[keep], "y": y[keep]}).sort_values("x")

            ax.plot(
                gg["x"].to_numpy(),
                gg["y"].to_numpy(),
                linewidth=1.0,
                alpha=0.65,
            )

        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Proportion")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25)

    fig.suptitle(f"Time series: {hypothesis_label}", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_all_plots(
    raw_pop_df: pd.DataFrame,
    final_params: pd.DataFrame,
    opt_df: pd.DataFrame,
    final_pop_df: pd.DataFrame,
    hypothesis_dir: Path,
    out_dir: Path,
    dpi: int,
) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_optimization_parameter_traces(
        opt_df=opt_df,
        out_path=plot_dir / "optimization_trace_loss_and_parameters.png",
        dpi=dpi,
    )

    plot_optimization_weighted_loss_traces(
        opt_df=opt_df,
        out_path=plot_dir / "optimization_trace_weighted_loss_components.png",
        dpi=dpi,
    )

    plot_optimized_parameter_distributions(
        final_params=final_params,
        opt_df=opt_df,
        out_path=plot_dir / "optimized_parameter_distributions.png",
        dpi=dpi,
    )

    plot_final_population_stats(
        raw_pop_df=raw_pop_df,
        out_path=plot_dir / "final_population_stats.png",
        dpi=dpi,
    )

    plot_final_population_outcome_summary(
        final_pop_df=final_pop_df,
        out_path=plot_dir / "final_population_outcome_summary.png",
        dpi=dpi,
    )

    plot_final_population_counts(
        final_pop_df=final_pop_df,
        out_path=plot_dir / "final_population_counts.png",
        dpi=dpi,
        hypothesis_label=hypothesis_dir.name,
    )

    plot_time_series(
        final_pop_df=final_pop_df,
        out_path=plot_dir / "time_series.png",
        dpi=dpi,
        hypothesis_label=hypothesis_dir.name,
    )



def main() -> None:
    args = parse_args()

    hypothesis_dirs = [Path(d).resolve() for d in args.hypothesis_dirs]

    for d in hypothesis_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Hypothesis directory does not exist: {d}")

    if len(hypothesis_dirs) == 1:
        # Single hypothesis evaluation
        hypothesis_dir = hypothesis_dirs[0]
        out_dir = (
            Path(args.out_dir).resolve()
            if args.out_dir
            else hypothesis_dir / "audit_summary"
        )
        evaluate_single_hypothesis(hypothesis_dir, out_dir, args)
    else:
        # Comparison
        out_dir = (
            Path(args.out_dir).resolve()
            if args.out_dir
            else Path("comparison")
        )
        # First, evaluate each hypothesis individually in subdirs
        for d in hypothesis_dirs:
            single_out = out_dir / d.name
            evaluate_single_hypothesis(d, single_out, args)
        # Then, compare
        compare_hypotheses(hypothesis_dirs, out_dir, args)

def evaluate_single_hypothesis(hypothesis_dir: Path, out_dir: Path, args) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    obs, all_obs_diams = load_observed_diameters(
        observed_csv=args.observed_csv,
        obs_diam_col=args.obs_diam_col,
        obs_site_col=args.obs_site_col,
    )

    sim_df = load_all_final_summaries(
        hypothesis_dir=hypothesis_dir,
        min_alive_tillers=args.min_alive_tillers,
        leaf_min=args.leaf_min,
        leaf_max=args.leaf_max,
    )

    if sim_df.empty:
        raise RuntimeError(
            f"No final_sims/summaries/summary_*.csv files found under {hypothesis_dir}"
        )

    sim_df = add_diameter_match_flags(
        sim_df=sim_df,
        obs=obs,
        all_obs_diams=all_obs_diams,
        obs_site_col=args.obs_site_col,
        q_low=args.diam_q_low,
        q_high=args.diam_q_high,
        buffer=args.diam_buffer,
    )

    family_summary = summarize_families(
        sim_df=sim_df,
        obs=obs,
        all_obs_diams=all_obs_diams,
        obs_site_col=args.obs_site_col,
        args=args,
    )

    final_params = collect_final_parameters(hypothesis_dir)
    best_opt = collect_best_optimization_rows(hypothesis_dir)
    all_opt = collect_all_optimization_rows(hypothesis_dir)
    param_stability = summarize_parameter_stability(final_params)
    raw_pop_df = load_raw_final_timestep_population_stats(hypothesis_dir)
    final_pop_df = load_final_population_results(hypothesis_dir)

    sim_df.to_csv(out_dir / "all_final_sim_summaries.csv", index=False)
    family_summary.to_csv(out_dir / "model_family_summary.csv", index=False)
    final_params.to_csv(out_dir / "final_parameters_by_set.csv", index=False)
    best_opt.to_csv(out_dir / "best_optimization_rows.csv", index=False)
    all_opt.to_csv(out_dir / "all_optimization_rows.csv", index=False)
    param_stability.to_csv(out_dir / "parameter_stability_summary.csv", index=False)
    raw_pop_df.to_csv(out_dir / "raw_final_timestep_population_stats.csv", index=False)
    final_pop_df.to_csv(out_dir / "final_population_results_compiled.csv", index=False)

    if not args.no_plots:
        make_all_plots(
            raw_pop_df=raw_pop_df,
            final_params=final_params,
            opt_df=all_opt,
            final_pop_df=final_pop_df,
            hypothesis_dir=hypothesis_dir,
            out_dir=out_dir,
            dpi=args.plot_dpi,
        )

    print(f"Wrote audit outputs to: {out_dir}")
    print("")
    print("Key files:")
    print(f"  {out_dir / 'all_final_sim_summaries.csv'}")
    print(f"  {out_dir / 'model_family_summary.csv'}")
    print(f"  {out_dir / 'final_parameters_by_set.csv'}")
    print(f"  {out_dir / 'best_optimization_rows.csv'}")
    print(f"  {out_dir / 'all_optimization_rows.csv'}")
    print(f"  {out_dir / 'parameter_stability_summary.csv'}")
    print(f"  {out_dir / 'raw_final_timestep_population_stats.csv'}")
    print(f"  {out_dir / 'final_population_results_compiled.csv'}")

    if not args.no_plots:
        print("")
        print("Plots:")
        print(f"  {out_dir / 'plots' / 'optimization_trace_loss_and_parameters.png'}")
        print(f"  {out_dir / 'plots' / 'optimization_trace_weighted_loss_components.png'}")
        print(f"  {out_dir / 'plots' / 'optimized_parameter_distributions.png'}")
        print(f"  {out_dir / 'plots' / 'final_population_stats.png'}")
        print(f"  {out_dir / 'plots' / 'final_population_outcome_summary.png'}")
        print(f"  {out_dir / 'plots' / 'final_population_counts.png'}")
        print(f"  {out_dir / 'plots' / 'time_series.png'}")

    print("")
    print("Top-level summary:")
    print(family_summary.to_string(index=False))


def find_final_population_results_compiled(hypothesis_dir: Path) -> Optional[Path]:
    candidate = hypothesis_dir / "final_population_results_compiled.csv"
    if candidate.exists():
        return candidate

    candidate = hypothesis_dir / "audit_summary" / "final_population_results_compiled.csv"
    if candidate.exists():
        return candidate

    candidates = list(hypothesis_dir.rglob("final_population_results_compiled.csv"))
    if not candidates:
        return None

    return min(candidates, key=lambda p: len(p.relative_to(hypothesis_dir).parts))


def collect_hypothesis_stats(hypothesis_dir: Path) -> Optional[pd.Series]:
    final_pop_path = find_final_population_results_compiled(hypothesis_dir)
    if final_pop_path is not None:
        df = pd.read_csv(final_pop_path)
        if not df.empty and {
            'alive_tussocks_final', 'extinct_tussocks_final', 'overflow_tussocks', 'prop_alive', 'avg_tussock_diameter'
        }.issubset(df.columns):
            return pd.Series({
                'alive_mean': float(df['alive_tussocks_final'].mean()),
                'alive_std': float(df['alive_tussocks_final'].std(ddof=1)) if len(df) > 1 else 0.0,
                'extinct_mean': float(df['extinct_tussocks_final'].mean()),
                'extinct_std': float(df['extinct_tussocks_final'].std(ddof=1)) if len(df) > 1 else 0.0,
                'overflow_mean': float(df['overflow_tussocks'].mean()),
                'overflow_std': float(df['overflow_tussocks'].std(ddof=1)) if len(df) > 1 else 0.0,
                'prop_alive_mean': float(df['prop_alive'].mean()),
                'prop_alive_std': float(df['prop_alive'].std(ddof=1)) if len(df) > 1 else 0.0,
                'prop_extinct_mean': float(df['prop_extinct'].mean()),
                'prop_extinct_std': float(df['prop_extinct'].std(ddof=1)) if len(df) > 1 else 0.0,
                'prop_overgrown_mean': float(df['prop_overgrown'].mean()),
                'prop_overgrown_std': float(df['prop_overgrown'].std(ddof=1)) if len(df) > 1 else 0.0,
                'prop_overflow_mean': float(df['prop_overflow'].mean()),
                'prop_overflow_std': float(df['prop_overflow'].std(ddof=1)) if len(df) > 1 else 0.0,
                'avg_diam_mean': float(df['avg_tussock_diameter'].mean()),
                'avg_diam_std': float(df['avg_tussock_diameter'].std(ddof=1)) if len(df) > 1 else 0.0,
            })

    files = list(hypothesis_dir.rglob("final_population_results.csv"))
    if not files:
        return None

    dfs = []
    for p in files:
        df = pd.read_csv(p)
        if not df.empty and {
            'alive_tussocks_final', 'extinct_tussocks_final', 'overflow_tussocks', 'prop_alive', 'avg_tussock_diameter'
        }.issubset(df.columns):
            dfs.append(df[['alive_tussocks_final', 'extinct_tussocks_final', 'overflow_tussocks', 'prop_alive', 'prop_extinct', 'prop_overgrown', 'prop_overflow', 'avg_tussock_diameter']])

    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True)
    return pd.Series({
        'alive_mean': float(df['alive_tussocks_final'].mean()),
        'alive_std': float(df['alive_tussocks_final'].std(ddof=1)) if len(df) > 1 else 0.0,
        'extinct_mean': float(df['extinct_tussocks_final'].mean()),
        'extinct_std': float(df['extinct_tussocks_final'].std(ddof=1)) if len(df) > 1 else 0.0,
        'overflow_mean': float(df['overflow_tussocks'].mean()),
        'overflow_std': float(df['overflow_tussocks'].std(ddof=1)) if len(df) > 1 else 0.0,
        'prop_alive_mean': float(df['prop_alive'].mean()),
        'prop_alive_std': float(df['prop_alive'].std(ddof=1)) if len(df) > 1 else 0.0,
        'prop_extinct_mean': float(df['prop_extinct'].mean()),
        'prop_extinct_std': float(df['prop_extinct'].std(ddof=1)) if len(df) > 1 else 0.0,
        'prop_overgrown_mean': float(df['prop_overgrown'].mean()),
        'prop_overgrown_std': float(df['prop_overgrown'].std(ddof=1)) if len(df) > 1 else 0.0,
        'prop_overflow_mean': float(df['prop_overflow'].mean()),
        'prop_overflow_std': float(df['prop_overflow'].std(ddof=1)) if len(df) > 1 else 0.0,
        'avg_diam_mean': float(df['avg_tussock_diameter'].mean()),
        'avg_diam_std': float(df['avg_tussock_diameter'].std(ddof=1)) if len(df) > 1 else 0.0,
    })


def compare_hypotheses(hypothesis_dirs: list[Path], out_dir: Path, args) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    
    all_data = []
    labels = []
    
    for d in hypothesis_dirs:
        stats = collect_hypothesis_stats(d)
        if stats is None:
            print(f"Warning: no final population stats found for {d}")
            continue

        all_data.append(stats)
        labels.append(d.name)
    
    if not all_data:
        print("No data found in provided directories.")
        return
    
    df_stats = pd.DataFrame(all_data, index=labels)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(df_stats))
    width = 0.25
    
    alive = df_stats['alive_mean'].to_numpy()
    alive_err = df_stats['alive_std'].to_numpy()
    extinct = df_stats['extinct_mean'].to_numpy()
    extinct_err = df_stats['extinct_std'].to_numpy()
    overflow = df_stats['overflow_mean'].to_numpy()
    overflow_err = df_stats['overflow_std'].to_numpy()
    
    ax.bar(x - width, alive, width, yerr=alive_err, capsize=4, label='Alive', alpha=0.8)
    ax.bar(x, extinct, width, yerr=extinct_err, capsize=4, label='Extinct', alpha=0.8)
    ax.bar(x + width, overflow, width, yerr=overflow_err, capsize=4, label='Overflow', alpha=0.8)
    
    ax.set_xlabel('Hypothesis')
    ax.set_ylabel('Average Count')
    ax.set_title('Final Population Counts Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.25)
    
    fig.tight_layout()
    plt.savefig(out_dir / "hypothesis_comparison_counts.png", dpi=args.plot_dpi)
    plt.close()
    
    # Plot 2: Viability Summary (prop_alive)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    prop_alive = df_stats['prop_alive_mean'].to_numpy()
    prop_alive_err = df_stats['prop_alive_std'].to_numpy()
    
    ax.bar(x, prop_alive, width=0.5, yerr=prop_alive_err, capsize=4, alpha=0.8, color='green')
    
    ax.set_xlabel('Hypothesis')
    ax.set_ylabel('Proportion Alive')
    ax.set_title('Viability Summary: Proportion of Alive Tussocks')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.25)
    
    fig.tight_layout()
    plt.savefig(out_dir / "hypothesis_comparison_viability.png", dpi=args.plot_dpi)
    plt.close()
    
    # Plot 3: Tussock Size Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    diam = df_stats['avg_diam_mean'].to_numpy()
    diam_err = df_stats['avg_diam_std'].to_numpy()
    
    ax.bar(x, diam, width=0.5, yerr=diam_err, capsize=4, alpha=0.8, color='blue')
    
    ax.set_xlabel('Hypothesis')
    ax.set_ylabel('Average Tussock Diameter')
    ax.set_title('Tussock Size Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.25)
    
    fig.tight_layout()
    plt.savefig(out_dir / "hypothesis_comparison_size.png", dpi=args.plot_dpi)
    plt.close()
    
    # Plot 4: Population Composition Stacked Bars
    fig, ax = plt.subplots(figsize=(12, 6))
    
    prop_alive_vals = df_stats['prop_alive_mean'].to_numpy()
    prop_extinct_vals = df_stats['prop_extinct_mean'].to_numpy()
    prop_overgrown_vals = df_stats['prop_overgrown_mean'].to_numpy()
    prop_overflow_vals = df_stats['prop_overflow_mean'].to_numpy()
    
    ax.bar(x, prop_alive_vals, width=0.5, label='Alive', alpha=0.8, color='green')
    ax.bar(x, prop_extinct_vals, width=0.5, bottom=prop_alive_vals, label='Extinct', alpha=0.8, color='red')
    ax.bar(x, prop_overgrown_vals, width=0.5, bottom=prop_alive_vals + prop_extinct_vals, label='Overgrown', alpha=0.8, color='orange')
    ax.bar(x, prop_overflow_vals, width=0.5, bottom=prop_alive_vals + prop_extinct_vals + prop_overgrown_vals, label='Overflow', alpha=0.8, color='purple')
    
    ax.set_xlabel('Hypothesis')
    ax.set_ylabel('Proportion')
    ax.set_title('Population Composition')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.25)
    
    fig.tight_layout()
    plt.savefig(out_dir / "hypothesis_comparison_composition.png", dpi=args.plot_dpi)
    plt.close()
    
    # Plot 5: Composite Score Ranking (simple: prop_alive * avg_diam / 100)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    score = prop_alive * diam / 100  # arbitrary composite score
    
    ax.bar(x, score, width=0.5, alpha=0.8, color='purple')
    
    ax.set_xlabel('Hypothesis')
    ax.set_ylabel('Composite Score')
    ax.set_title('Composite Score Ranking (Viability × Size)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.25)
    
    fig.tight_layout()
    plt.savefig(out_dir / "hypothesis_comparison_ranking.png", dpi=args.plot_dpi)
    plt.close()
    
    print(f"Comparison plots saved in {out_dir}")


if __name__ == "__main__":
    main()