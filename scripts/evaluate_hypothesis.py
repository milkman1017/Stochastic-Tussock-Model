#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Audit final tussock simulations for viability, plausibility, field fit, and parameter stability."
    )
    p.add_argument(
        "--hypothesis-dir",
        required=True,
        help="Directory to crawl. Can be a whole hypothesis dir or a subdir like h1/repro.",
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


def infer_set_id(path: Path) -> str:
    for part in path.parts:
        if re.fullmatch(r"set_\d+", part):
            return part
    return ""


def infer_site_from_final_sims(final_sims_dir: Path) -> str:
    # Expected: .../set_001/<site>/final_sims
    parent = final_sims_dir.parent
    if parent.name and parent.name != "final_sims":
        return parent.name
    return ""


def infer_model_family(hypothesis_dir: Path, final_sims_dir: Path) -> str:
    # Path relative to hypothesis dir, excluding set_xxx/site/final_sims and deeper.
    try:
        rel = final_sims_dir.relative_to(hypothesis_dir)
    except ValueError:
        return hypothesis_dir.name

    parts = list(rel.parts)
    trimmed = []
    for part in parts:
        if re.fullmatch(r"set_\d+", part):
            break
        trimmed.append(part)

    if trimmed:
        return "/".join(trimmed)
    return hypothesis_dir.name


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

            # Row-level flags.
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
            df["low_alive"] = df["alive_y"].fillna(0).astype(float) < float(min_alive_tillers)

            leaf = df["LeafArea"].astype(float)
            df["leaf_bad"] = np.isfinite(leaf) & ((leaf <= leaf_min) | (leaf >= leaf_max))

            df["viability_pass"] = (
                (~df["invalid"])
                & (~df["overflow"])
                & (~df["low_alive"])
            )

            rows.append(df)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    return out


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
        raise ValueError(f"Observed diameter column '{obs_diam_col}' not found in {obs_path}")

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

    # If site-specific data are available, use them. Otherwise use all observed data.
    if site and site != "ALL" and obs_site_col in obs.columns:
        site_obs = obs.loc[obs[obs_site_col].astype(str) == str(site), "_obs_diam"].to_numpy(dtype=float)
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


def classify_family(row: pd.Series, max_invalid: float, max_overflow: float, max_low_alive: float,
                    max_leaf_bad: float, max_wass: Optional[float]) -> str:
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
            "diameter_match_frac": float(g["diameter_match"].mean()) if "diameter_match" in g else np.nan,
            "diameter_match_frac_among_viable": (
                float(viable_g["diameter_match"].mean())
                if "diameter_match" in viable_g and len(viable_g) > 0 else np.nan
            ),
            "diameter_wasserstein": wass,
            "sim_final_diameter_mean_viable": (
                float(np.mean(sim_diam_viable)) if sim_diam_viable.size > 0 else np.nan
            ),
            "sim_final_diameter_sd_viable": (
                float(np.std(sim_diam_viable, ddof=1)) if sim_diam_viable.size > 1 else np.nan
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
        sort_cols = [
            "classification",
            "viability_pass_frac",
            "diameter_match_frac_among_viable",
            "diameter_wasserstein",
        ]
        out = out.sort_values(
            by=sort_cols,
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
        site_dir = final_sims_dir.parent
        param_file = site_dir / "parameters.txt"
        if not param_file.exists():
            # Fallback: sometimes a set-level parameter file exists.
            set_id = infer_set_id(final_sims_dir)
            for parent in final_sims_dir.parents:
                if parent.name == set_id:
                    candidate = parent / "parameters.txt"
                    if candidate.exists():
                        param_file = candidate
                    break

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
        try:
            df = pd.read_csv(opt_file)
        except Exception:
            continue

        if df.empty or "loss" not in df.columns:
            continue

        df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
        df = df[np.isfinite(df["loss"])]
        if df.empty:
            continue

        best = df.loc[df["loss"].idxmin()].to_dict()

        parent = opt_file.parent
        set_id = infer_set_id(opt_file)
        site = parent.name
        model_family = infer_model_family(hypothesis_dir, parent / "final_sims")

        row = {
            "model_family": model_family,
            "set_id": set_id,
            "site": site,
            "optimization_file": str(opt_file),
        }
        row.update(best)
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_parameter_stability(params_df: pd.DataFrame) -> pd.DataFrame:
    if params_df.empty:
        return pd.DataFrame()

    meta_cols = {"model_family", "set_id", "site", "final_sims_dir", "parameter_file"}
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


def main() -> None:
    args = parse_args()
    hypothesis_dir = Path(args.hypothesis_dir).resolve()
    if not hypothesis_dir.exists():
        raise FileNotFoundError(f"Hypothesis directory does not exist: {hypothesis_dir}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else hypothesis_dir / "audit_summary"
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
        raise RuntimeError(f"No final_sims/summaries/summary_*.csv files found under {hypothesis_dir}")

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
    param_stability = summarize_parameter_stability(final_params)

    sim_df.to_csv(out_dir / "all_final_sim_summaries.csv", index=False)
    family_summary.to_csv(out_dir / "model_family_summary.csv", index=False)
    final_params.to_csv(out_dir / "final_parameters_by_set.csv", index=False)
    best_opt.to_csv(out_dir / "best_optimization_rows.csv", index=False)
    param_stability.to_csv(out_dir / "parameter_stability_summary.csv", index=False)

    print(f"Wrote audit outputs to: {out_dir}")
    print("")
    print("Key files:")
    print(f"  {out_dir / 'all_final_sim_summaries.csv'}")
    print(f"  {out_dir / 'model_family_summary.csv'}")
    print(f"  {out_dir / 'final_parameters_by_set.csv'}")
    print(f"  {out_dir / 'best_optimization_rows.csv'}")
    print(f"  {out_dir / 'parameter_stability_summary.csv'}")
    print("")
    print("Top-level summary:")
    print(family_summary.to_string(index=False))


if __name__ == "__main__":
    main()
