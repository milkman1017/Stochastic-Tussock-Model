#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


import configparser

FINAL_POP_REQUIRED_COLUMNS = [
    "iteration",
    "alive_tussocks_final",
    "extinct_tussocks_final",
    "overgrown_tussocks",
    "overflow_tussocks",
    "avg_tussock_diameter",
]


OUTCOME_COUNT_COLUMNS = [
    "alive_tussocks_final",
    "extinct_tussocks_final",
    "overgrown_tussocks",
    "overflow_tussocks",
]


OUTCOME_PROP_COLUMNS = [
    "prop_alive",
    "prop_extinct",
    "prop_overgrown",
    "prop_overflow",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare higher-level hypothesis/question outputs from one h-dir. "
            "Expected layout: <h_dir>/<question>/resampled_fits/<ecotype>/set_001/final_population_results.csv"
        )
    )

    p.add_argument(
        "--h-dir",
        required=True,
        help=(
            "Top hypothesis directory, e.g. "
            "/home/lucentlab/wmahler/Stochastic-Tussock-Model/h1-5"
        ),
    )

    p.add_argument(
        "--ecotype",
        default="",
        help=(
            "Optional ecotype/site to restrict to, e.g. TL. "
            "If empty, all ecotypes under resampled_fits/* are combined."
        ),
    )

    p.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Output directory. Default: <h-dir>/higher_level_comparison "
            "or <h-dir>/higher_level_comparison_<ecotype> if --ecotype is given."
        ),
    )

    p.add_argument(
        "--plot-dpi",
        type=int,
        default=250,
        help="DPI for saved plots. Default: 250.",
    )

    p.add_argument(
        "--title",
        default=None,
        help="Optional title prefix for plots. Default is the h-dir name.",
    )

    p.add_argument(
        "--include-nonstandard-files",
        action="store_true",
        help=(
            "If set, include files whose names start with final_population_results. "
            "By default, only exact final_population_results.csv files are used."
        ),
    )

    return p.parse_args()


def is_set_dir_name(name: str) -> bool:
    return re.fullmatch(r"set_\d+", str(name)) is not None


def infer_set_id(path: Path) -> str:
    for part in path.parts:
        if is_set_dir_name(part):
            return part
    return ""


def safe_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def save_empty_plot(path: Path, title: str, message: str, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
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

def wasserstein_distance_1d(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

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

def get_overgrown_radius_threshold_for_set(set_dir: Path) -> float:
    config_file = set_dir / "config_snapshot.ini"
    default_threshold = 2.5

    if not config_file.exists():
        return default_threshold

    cp = configparser.ConfigParser()
    cp.read(config_file)

    return cp.getfloat(
        "Constraints",
        "overgrown_radius_threshold",
        fallback=default_threshold,
    )


def read_final_sim_diameters_for_set(set_dir: Path) -> tuple[np.ndarray, dict[str, int]]:
    summary_dir = set_dir / "final_sims" / "summaries"

    empty_counts = {
        "n_sim_diameters_total_for_wasserstein": 0,
        "n_sim_diameters_eligible_for_wasserstein": 0,
        "n_sim_diameters_extinct_excluded": 0,
        "n_sim_diameters_overgrown_excluded": 0,
        "n_sim_diameters_missing_filter_data": 0,
    }

    if not summary_dir.is_dir():
        return np.array([], dtype=float), empty_counts

    dfs = []

    for summary_file in sorted(summary_dir.glob("summary_*.csv")):
        try:
            df = pd.read_csv(summary_file)
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return np.array([], dtype=float), empty_counts

    df = pd.concat(dfs, ignore_index=True)

    for col in ["final_diameter", "alive_final", "rmax_y"]:
        if col not in df.columns:
            df[col] = np.nan

    df["final_diameter"] = pd.to_numeric(df["final_diameter"], errors="coerce")
    df["alive_final"] = pd.to_numeric(df["alive_final"], errors="coerce")
    df["rmax_y"] = pd.to_numeric(df["rmax_y"], errors="coerce")

    overgrown_radius_threshold = get_overgrown_radius_threshold_for_set(set_dir)

    has_diameter = np.isfinite(df["final_diameter"])
    has_filter_data = (
        np.isfinite(df["alive_final"])
        & np.isfinite(df["rmax_y"])
    )

    extinct = has_filter_data & (df["alive_final"] <= 0)
    overgrown = has_filter_data & (df["rmax_y"] > float(overgrown_radius_threshold))

    eligible = (
        has_diameter
        & has_filter_data
        & (df["alive_final"] > 0)
        & (df["rmax_y"] <= float(overgrown_radius_threshold))
    )

    counts = {
        "n_sim_diameters_total_for_wasserstein": int(has_diameter.sum()),
        "n_sim_diameters_eligible_for_wasserstein": int(eligible.sum()),
        "n_sim_diameters_extinct_excluded": int((has_diameter & extinct).sum()),
        "n_sim_diameters_overgrown_excluded": int((has_diameter & overgrown).sum()),
        "n_sim_diameters_missing_filter_data": int((has_diameter & ~has_filter_data).sum()),
    }

    sim_diameters = df.loc[eligible, "final_diameter"].to_numpy(dtype=float)
    sim_diameters = sim_diameters[np.isfinite(sim_diameters)]

    return sim_diameters, counts


def read_training_diameters_for_set(set_dir: Path, ecotype: str) -> np.ndarray:
    training_file = set_dir / "sampled_training_data.csv"

    if not training_file.exists():
        return np.array([], dtype=float)

    try:
        df = pd.read_csv(training_file)
    except Exception:
        return np.array([], dtype=float)

    if "diam" not in df.columns:
        return np.array([], dtype=float)

    if ecotype and ecotype != "ALL" and "site" in df.columns:
        df = df[df["site"].astype(str) == str(ecotype)].copy()

    diam = pd.to_numeric(df["diam"], errors="coerce").to_numpy(dtype=float)
    return diam[np.isfinite(diam)]


def compute_set_diameter_wasserstein(result_file: Path, ecotype: str) -> dict[str, float]:
    set_dir = result_file.parent

    sim_diameters, sim_counts = read_final_sim_diameters_for_set(set_dir)
    training_diameters = read_training_diameters_for_set(set_dir, ecotype)

    w1 = wasserstein_distance_1d(training_diameters, sim_diameters)

    out = {
        "diameter_wasserstein_1d": float(w1) if np.isfinite(w1) else np.nan,
        "n_training_diameters_for_wasserstein": int(training_diameters.size),
        "n_sim_diameters_for_wasserstein": int(sim_diameters.size),
    }

    out.update(sim_counts)

    return out


def find_final_population_files(
    h_dir: Path,
    ecotype: str,
    include_nonstandard_files: bool,
) -> list[Path]:
    files: list[Path] = []

    for question_dir in sorted([p for p in h_dir.iterdir() if p.is_dir()]):
        resampled_dir = question_dir / "resampled_fits"

        if not resampled_dir.is_dir():
            continue

        if ecotype:
            ecotype_dirs = [resampled_dir / ecotype]
        else:
            ecotype_dirs = sorted([p for p in resampled_dir.iterdir() if p.is_dir()])

        for ecotype_dir in ecotype_dirs:
            if not ecotype_dir.is_dir():
                continue

            for set_dir in sorted([p for p in ecotype_dir.iterdir() if p.is_dir() and is_set_dir_name(p.name)]):
                if include_nonstandard_files:
                    candidates = sorted(set_dir.glob("final_population_results*.csv"))
                else:
                    candidates = [set_dir / "final_population_results.csv"]

                for candidate in candidates:
                    if candidate.exists():
                        files.append(candidate)

    return files


def parse_metadata_from_file(h_dir: Path, result_file: Path) -> dict[str, str]:
    rel = result_file.relative_to(h_dir)
    parts = rel.parts

    question = parts[0] if len(parts) > 0 else ""
    ecotype = ""
    set_id = infer_set_id(result_file)

    if "resampled_fits" in parts:
        idx = parts.index("resampled_fits")
        if idx + 1 < len(parts):
            ecotype = parts[idx + 1]

    return {
        "h_dir": h_dir.name,
        "question": question,
        "ecotype": ecotype,
        "set_id": set_id,
        "final_population_result_file": str(result_file),
    }

def load_one_final_population_file(h_dir: Path, result_file: Path) -> pd.DataFrame:
    df = pd.read_csv(result_file)

    for col in FINAL_POP_REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    for col in FINAL_POP_REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    meta = parse_metadata_from_file(h_dir=h_dir, result_file=result_file)

    for key, value in meta.items():
        df[key] = value

    df["total_tussocks_classified"] = df[OUTCOME_COUNT_COLUMNS].sum(axis=1)

    df["prop_alive"] = np.where(
        df["total_tussocks_classified"] > 0,
        df["alive_tussocks_final"] / df["total_tussocks_classified"],
        np.nan,
    )
    df["prop_extinct"] = np.where(
        df["total_tussocks_classified"] > 0,
        df["extinct_tussocks_final"] / df["total_tussocks_classified"],
        np.nan,
    )
    df["prop_overgrown"] = np.where(
        df["total_tussocks_classified"] > 0,
        df["overgrown_tussocks"] / df["total_tussocks_classified"],
        np.nan,
    )
    df["prop_overflow"] = np.where(
        df["total_tussocks_classified"] > 0,
        df["overflow_tussocks"] / df["total_tussocks_classified"],
        np.nan,
    )

    wdist_info = compute_set_diameter_wasserstein(
        result_file=result_file,
        ecotype=str(meta.get("ecotype", "")),
    )

    for key, value in wdist_info.items():
        df[key] = value

    return df


def load_all_final_population_results(
    h_dir: Path,
    ecotype: str,
    include_nonstandard_files: bool,
) -> pd.DataFrame:
    files = find_final_population_files(
        h_dir=h_dir,
        ecotype=ecotype,
        include_nonstandard_files=include_nonstandard_files,
    )

    rows = []

    for result_file in files:
        try:
            rows.append(load_one_final_population_file(h_dir=h_dir, result_file=result_file))
        except Exception as exc:
            print(f"WARNING: could not read {result_file}: {exc}")

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def get_last_iteration_rows(final_pop_df: pd.DataFrame) -> pd.DataFrame:
    if final_pop_df.empty:
        return pd.DataFrame()

    df = final_pop_df.copy()
    df["iteration"] = pd.to_numeric(df["iteration"], errors="coerce")

    group_cols = [
        "question",
        "ecotype",
        "set_id",
        "final_population_result_file",
    ]

    last_rows = []

    for _, g in df.groupby(group_cols, dropna=False):
        g = g[np.isfinite(g["iteration"])].copy()

        if g.empty:
            continue

        g = g.sort_values("iteration")
        last_rows.append(g.iloc[-1])

    if not last_rows:
        return pd.DataFrame()

    return pd.DataFrame(last_rows).reset_index(drop=True)


def summarize_by_question(last_df: pd.DataFrame) -> pd.DataFrame:
    if last_df.empty:
        return pd.DataFrame()

    rows = []

    metric_cols = (
        OUTCOME_COUNT_COLUMNS
        + OUTCOME_PROP_COLUMNS
        + [
            "avg_tussock_diameter",
            "diameter_wasserstein_1d",
            "n_training_diameters_for_wasserstein",
            "n_sim_diameters_for_wasserstein",
            "n_sim_diameters_total_for_wasserstein",
            "n_sim_diameters_eligible_for_wasserstein",
            "n_sim_diameters_extinct_excluded",
            "n_sim_diameters_overgrown_excluded",
            "n_sim_diameters_missing_filter_data",
        ]
    )

    for question, g in last_df.groupby("question", dropna=False):
        row = {
            "question": question,
            "n_final_population_files": int(g["final_population_result_file"].nunique()),
            "n_sets": int(g["set_id"].nunique()),
            "n_ecotypes": int(g["ecotype"].nunique()),
            "ecotypes": ",".join(sorted(g["ecotype"].dropna().astype(str).unique())),
        }

        for col in metric_cols:
            if col not in g.columns:
                continue

            vals = pd.to_numeric(g[col], errors="coerce")
            row[f"{col}_mean"] = float(vals.mean(skipna=True))
            row[f"{col}_sd"] = float(vals.std(ddof=1, skipna=True)) if vals.notna().sum() > 1 else 0.0
            row[f"{col}_median"] = float(vals.median(skipna=True))
            row[f"{col}_min"] = float(vals.min(skipna=True))
            row[f"{col}_max"] = float(vals.max(skipna=True))

        rows.append(row)

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values(
            by=[
                "diameter_wasserstein_1d_mean",
                "prop_alive_mean",
                "avg_tussock_diameter_mean",
                "prop_overflow_mean",
            ],
            ascending=[True, False, False, True],
            na_position="last",
        )

    return out

def summarize_by_question_ecotype(last_df: pd.DataFrame) -> pd.DataFrame:
    if last_df.empty:
        return pd.DataFrame()

    rows = []

    metric_cols = (
        OUTCOME_COUNT_COLUMNS
        + OUTCOME_PROP_COLUMNS
        + [
            "avg_tussock_diameter",
            "diameter_wasserstein_1d",
            "n_training_diameters_for_wasserstein",
            "n_sim_diameters_for_wasserstein",
            "n_sim_diameters_total_for_wasserstein",
            "n_sim_diameters_eligible_for_wasserstein",
            "n_sim_diameters_extinct_excluded",
            "n_sim_diameters_overgrown_excluded",
            "n_sim_diameters_missing_filter_data",
        ]
    )

    for (question, ecotype), g in last_df.groupby(["question", "ecotype"], dropna=False):
        row = {
            "question": question,
            "ecotype": ecotype,
            "n_final_population_files": int(g["final_population_result_file"].nunique()),
            "n_sets": int(g["set_id"].nunique()),
        }

        for col in metric_cols:
            if col not in g.columns:
                continue

            vals = pd.to_numeric(g[col], errors="coerce")
            row[f"{col}_mean"] = float(vals.mean(skipna=True))
            row[f"{col}_sd"] = float(vals.std(ddof=1, skipna=True)) if vals.notna().sum() > 1 else 0.0
            row[f"{col}_median"] = float(vals.median(skipna=True))

        rows.append(row)

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values(["question", "ecotype"])

    return out

def summarize_time_series(final_pop_df: pd.DataFrame) -> pd.DataFrame:
    if final_pop_df.empty:
        return pd.DataFrame()

    df = final_pop_df.copy()

    for col in ["iteration"] + OUTCOME_COUNT_COLUMNS + OUTCOME_PROP_COLUMNS + ["avg_tussock_diameter"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    group_cols = ["question", "iteration"]

    summary_rows = []

    for (question, iteration), g in df.groupby(group_cols, dropna=False):
        row = {
            "question": question,
            "iteration": iteration,
            "n_rows": int(len(g)),
            "n_sets": int(g["set_id"].nunique()),
            "n_ecotypes": int(g["ecotype"].nunique()),
        }

        for col in OUTCOME_COUNT_COLUMNS + OUTCOME_PROP_COLUMNS + ["avg_tussock_diameter"]:
            vals = pd.to_numeric(g[col], errors="coerce")
            row[f"{col}_mean"] = float(vals.mean(skipna=True))
            row[f"{col}_sd"] = float(vals.std(ddof=1, skipna=True)) if vals.notna().sum() > 1 else 0.0

        summary_rows.append(row)

    out = pd.DataFrame(summary_rows)

    if not out.empty:
        out = out.sort_values(["question", "iteration"])

    return out

def plot_tussock_diameter_wasserstein(
    last_df: pd.DataFrame,
    out_path: Path,
    title_prefix: str,
    dpi: int,
) -> None:
    if last_df.empty or "diameter_wasserstein_1d" not in last_df.columns:
        save_empty_plot(
            out_path,
            "Tussock diameter distribution deviation",
            "No Wasserstein distance data found.",
            dpi,
        )
        return

    df = last_df.copy()
    df["question"] = df["question"].astype(str)
    df["diameter_wasserstein_1d"] = pd.to_numeric(
        df["diameter_wasserstein_1d"],
        errors="coerce",
    )

    df = df[np.isfinite(df["diameter_wasserstein_1d"])].copy()

    if df.empty:
        save_empty_plot(
            out_path,
            "Tussock diameter distribution deviation",
            "No finite Wasserstein distance values found. This can happen if no sims were alive and non-overgrown.",
            dpi,
        )
        return

    summary_rows = []

    for question, g in df.groupby("question", dropna=False):
        vals = pd.to_numeric(g["diameter_wasserstein_1d"], errors="coerce")
        vals = vals[np.isfinite(vals)]

        if vals.empty:
            continue

        n_sets = int(vals.size)

        if "n_sim_diameters_eligible_for_wasserstein" in g.columns:
            n_eligible_sims = int(
                pd.to_numeric(
                    g["n_sim_diameters_eligible_for_wasserstein"],
                    errors="coerce",
                ).fillna(0).sum()
            )
        else:
            n_eligible_sims = 0

        summary_rows.append({
            "question": question,
            "mean": float(vals.mean()),
            "sd": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
            "n_sets": n_sets,
            "n_eligible_sims": n_eligible_sims,
        })

    summary = pd.DataFrame(summary_rows)

    if summary.empty:
        save_empty_plot(
            out_path,
            "Tussock diameter distribution deviation",
            "No finite Wasserstein distance values found after grouping.",
            dpi,
        )
        return

    # Lowest Wasserstein is best, so sort best-to-worst.
    summary = summary.sort_values("mean", ascending=True).reset_index(drop=True)

    labels = summary["question"].astype(str).to_list()
    x = np.arange(len(summary))

    y = pd.to_numeric(summary["mean"], errors="coerce").to_numpy()
    yerr = pd.to_numeric(summary["sd"], errors="coerce").fillna(0).to_numpy()
    n_sets = pd.to_numeric(summary["n_sets"], errors="coerce").fillna(0).astype(int).to_numpy()
    n_eligible_sims = pd.to_numeric(
        summary["n_eligible_sims"],
        errors="coerce",
    ).fillna(0).astype(int).to_numpy()

    fig_width = max(10, 0.80 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    bars = ax.bar(
        x,
        y,
        yerr=yerr,
        capsize=4,
        zorder=2,
    )

    rng = np.random.default_rng(123)
    question_to_x = {q: i for i, q in enumerate(labels)}

    for question, g in df.groupby("question", dropna=False):
        question = str(question)

        if question not in question_to_x:
            continue

        vals = pd.to_numeric(g["diameter_wasserstein_1d"], errors="coerce")
        vals = vals[np.isfinite(vals)].to_numpy(dtype=float)

        if vals.size == 0:
            continue

        jitter = rng.uniform(low=-0.08, high=0.08, size=vals.size)

        ax.scatter(
            question_to_x[question] + jitter,
            vals,
            s=26,
            alpha=0.75,
            edgecolors="black",
            linewidths=0.25,
            zorder=5,
        )

    ymax = 0.0
    for yi, ei in zip(y, yerr):
        if np.isfinite(yi):
            ymax = max(ymax, yi + (ei if np.isfinite(ei) else 0.0))

    text_offset = max(0.15, 0.02 * ymax) if ymax > 0 else 0.15

    for xi, yi, ei, n_set, n_sim in zip(x, y, yerr, n_sets, n_eligible_sims):
        if not np.isfinite(yi):
            continue

        err = ei if np.isfinite(ei) else 0.0

        ax.text(
            xi,
            yi + err + text_offset,
            f"{yi:.2f}\nsets={n_set}\nsims={n_sim}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            zorder=6,
        )

    ax.set_xlabel("Question / tested subdirectory")
    ax.set_ylabel(
        "1D Wasserstein distance from training diameter distribution\n"
        "using only alive, non-overgrown final sims"
    )
    ax.set_title(f"{title_prefix}: final simulated diameter deviation from training data")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.grid(True, axis="y", alpha=0.25, zorder=0)

    if ymax > 0:
        ax.set_ylim(0, ymax + (4.5 * text_offset))

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

def plot_population_composition(
    last_df: pd.DataFrame,
    out_path: Path,
    title_prefix: str,
    dpi: int,
) -> None:
    if last_df.empty:
        save_empty_plot(
            out_path,
            "Population composition",
            "No final-iteration data found.",
            dpi,
        )
        return

    raw = last_df.copy()

    raw["question"] = raw["question"].astype(str)
    raw["alive_tussocks_final"] = pd.to_numeric(
        raw["alive_tussocks_final"],
        errors="coerce",
    ).fillna(0)
    raw["extinct_tussocks_final"] = pd.to_numeric(
        raw["extinct_tussocks_final"],
        errors="coerce",
    ).fillna(0)
    raw["overflow_tussocks"] = pd.to_numeric(
        raw["overflow_tussocks"],
        errors="coerce",
    ).fillna(0)

    grouped = (
        raw.groupby("question", dropna=False)[
            ["alive_tussocks_final", "extinct_tussocks_final", "overflow_tussocks"]
        ]
        .sum()
        .reset_index()
    )

    grouped["total_classified"] = (
        grouped["alive_tussocks_final"]
        + grouped["extinct_tussocks_final"]
        + grouped["overflow_tussocks"]
    )

    grouped = grouped[grouped["total_classified"] > 0].copy()

    if grouped.empty:
        save_empty_plot(
            out_path,
            "Population composition",
            "No nonzero final-iteration outcome counts found.",
            dpi,
        )
        return

    grouped["prop_alive"] = (
        grouped["alive_tussocks_final"] / grouped["total_classified"]
    )
    grouped["prop_extinct"] = (
        grouped["extinct_tussocks_final"] / grouped["total_classified"]
    )
    grouped["prop_overflow"] = (
        grouped["overflow_tussocks"] / grouped["total_classified"]
    )

    grouped = grouped.sort_values("question").reset_index(drop=True)

    labels = grouped["question"].astype(str).to_list()
    x = np.arange(len(grouped))

    alive = grouped["prop_alive"].to_numpy()
    extinct = grouped["prop_extinct"].to_numpy()
    overflow = grouped["prop_overflow"].to_numpy()

    alive_n = grouped["alive_tussocks_final"].to_numpy()
    extinct_n = grouped["extinct_tussocks_final"].to_numpy()
    overflow_n = grouped["overflow_tussocks"].to_numpy()

    fig_width = max(10, 0.70 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    bars_alive = ax.bar(
        x,
        alive,
        label="Alive",
        zorder=2,
    )

    bars_extinct = ax.bar(
        x,
        extinct,
        bottom=alive,
        label="Extinct",
        zorder=2,
    )

    bars_overflow = ax.bar(
        x,
        overflow,
        bottom=alive + extinct,
        label="Overflow",
        zorder=2,
    )

    def add_segment_labels(bar_container, values, bottoms, counts):
        for bar, val, bottom, count in zip(bar_container, values, bottoms, counts):
            if not np.isfinite(val) or val <= 0:
                continue

            pct = val * 100.0
            y = bottom + (val / 2.0)

            # Skip extremely tiny segments to avoid unreadable text
            if pct < 1.0:
                continue

            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y,
                f"{pct:.1f}%\n(n={int(round(count))})",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="white",
                zorder=5,
            )

    add_segment_labels(
        bars_alive,
        alive,
        np.zeros_like(alive),
        alive_n,
    )
    add_segment_labels(
        bars_extinct,
        extinct,
        alive,
        extinct_n,
    )
    add_segment_labels(
        bars_overflow,
        overflow,
        alive + extinct,
        overflow_n,
    )

    ax.set_xlabel("Question / tested subdirectory")
    ax.set_ylabel("Final proportion across all sims")
    ax.set_title(f"{title_prefix}: final population composition")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right")

    ax.set_ylim(0, 1.05)

    ax.legend()
    ax.grid(True, axis="y", alpha=0.25, zorder=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_population_counts(
    summary_df: pd.DataFrame,
    last_df: pd.DataFrame,
    out_path: Path,
    title_prefix: str,
    dpi: int,
) -> None:
    if summary_df.empty:
        save_empty_plot(
            out_path,
            "Population counts",
            "No summary data found.",
            dpi,
        )
        return

    df = summary_df.sort_values("question").copy()
    labels = df["question"].astype(str).to_list()
    x = np.arange(len(df))
    width = 0.25

    alive = pd.to_numeric(df["alive_tussocks_final_mean"], errors="coerce").to_numpy()
    extinct = pd.to_numeric(df["extinct_tussocks_final_mean"], errors="coerce").to_numpy()
    overflow = pd.to_numeric(df["overflow_tussocks_mean"], errors="coerce").to_numpy()

    alive_err = pd.to_numeric(
        df["alive_tussocks_final_sd"],
        errors="coerce",
    ).fillna(0).to_numpy()
    extinct_err = pd.to_numeric(
        df["extinct_tussocks_final_sd"],
        errors="coerce",
    ).fillna(0).to_numpy()
    overflow_err = pd.to_numeric(
        df["overflow_tussocks_sd"],
        errors="coerce",
    ).fillna(0).to_numpy()

    fig_width = max(11, 0.75 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    bars_alive = ax.bar(
        x - width,
        alive,
        width,
        yerr=alive_err,
        capsize=3,
        label="Alive",
        alpha=0.45,
        zorder=1,
    )
    bars_extinct = ax.bar(
        x,
        extinct,
        width,
        yerr=extinct_err,
        capsize=3,
        label="Extinct",
        alpha=0.45,
        zorder=1,
    )
    bars_overflow = ax.bar(
        x + width,
        overflow,
        width,
        yerr=overflow_err,
        capsize=3,
        label="Overflow",
        alpha=0.45,
        zorder=1,
    )

    # Overlay raw per-file/set points as small transparent dots.
    # Dots are drawn above the bars and use the same color as the matching bar.
    if not last_df.empty:
        raw = last_df.copy()
        raw["question"] = raw["question"].astype(str)

        metric_info = [
            ("alive_tussocks_final", -width, bars_alive),
            ("extinct_tussocks_final", 0.0, bars_extinct),
            ("overflow_tussocks", width, bars_overflow),
        ]

        question_to_x = {label: i for i, label in enumerate(labels)}
        rng = np.random.default_rng(123)

        for metric_col, center_offset, bar_container in metric_info:
            if metric_col not in raw.columns:
                continue

            color = (
                bar_container.patches[0].get_facecolor()
                if len(bar_container.patches) > 0
                else None
            )

            for question, g in raw.groupby("question", dropna=False):
                if question not in question_to_x:
                    continue

                vals = pd.to_numeric(
                    g[metric_col],
                    errors="coerce",
                ).dropna().to_numpy()

                if vals.size == 0:
                    continue

                base_x = question_to_x[question] + center_offset

                jitter = rng.uniform(
                    low=-0.07,
                    high=0.07,
                    size=vals.size,
                )

                ax.scatter(
                    base_x + jitter,
                    vals,
                    s=22,
                    alpha=0.75,
                    color=color,
                    edgecolors="black",
                    linewidths=0.25,
                    zorder=5,
                )

    ax.set_xlabel("Question / tested subdirectory")
    ax.set_ylabel("Mean final count")
    ax.set_title(f"{title_prefix}: final population counts")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25, zorder=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

def plot_viability(
    summary_df: pd.DataFrame,
    out_path: Path,
    title_prefix: str,
    dpi: int,
) -> None:
    if summary_df.empty:
        save_empty_plot(
            out_path,
            "Viability",
            "No summary data found.",
            dpi,
        )
        return

    df = summary_df.sort_values("question").copy()
    labels = df["question"].astype(str).to_list()
    x = np.arange(len(df))

    y = pd.to_numeric(df["prop_alive_mean"], errors="coerce").to_numpy()
    yerr = pd.to_numeric(df["prop_alive_sd"], errors="coerce").fillna(0).to_numpy()

    fig_width = max(10, 0.65 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    ax.bar(x, y, yerr=yerr, capsize=4)

    ax.set_xlabel("Question / tested subdirectory")
    ax.set_ylabel("Mean final proportion alive")
    ax.set_title(f"{title_prefix}: viability comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_tussock_size(
    summary_df: pd.DataFrame,
    out_path: Path,
    title_prefix: str,
    dpi: int,
) -> None:
    if summary_df.empty:
        save_empty_plot(
            out_path,
            "Tussock size",
            "No summary data found.",
            dpi,
        )
        return

    df = summary_df.sort_values("question").copy()
    labels = df["question"].astype(str).to_list()
    x = np.arange(len(df))

    y = pd.to_numeric(df["avg_tussock_diameter_mean"], errors="coerce").to_numpy()
    yerr = pd.to_numeric(df["avg_tussock_diameter_sd"], errors="coerce").fillna(0).to_numpy()

    fig_width = max(10, 0.65 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    ax.bar(x, y, yerr=yerr, capsize=4)

    ax.set_xlabel("Question / tested subdirectory")
    ax.set_ylabel("Mean final average tussock diameter")
    ax.set_title(f"{title_prefix}: final tussock size comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_viability_vs_size(
    summary_df: pd.DataFrame,
    out_path: Path,
    title_prefix: str,
    dpi: int,
) -> None:
    if summary_df.empty:
        save_empty_plot(
            out_path,
            "Viability vs size",
            "No summary data found.",
            dpi,
        )
        return

    df = summary_df.copy()

    x = pd.to_numeric(df["prop_alive_mean"], errors="coerce")
    y = pd.to_numeric(df["avg_tussock_diameter_mean"], errors="coerce")
    labels = df["question"].astype(str)

    keep = np.isfinite(x) & np.isfinite(y)

    fig, ax = plt.subplots(figsize=(8, 7))

    if keep.sum() > 0:
        ax.scatter(x[keep], y[keep], s=45, alpha=0.8)

        for xi, yi, label in zip(x[keep], y[keep], labels[keep]):
            ax.annotate(
                label,
                (xi, yi),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
                alpha=0.85,
            )

    ax.set_xlabel("Mean final proportion alive")
    ax.set_ylabel("Mean final average tussock diameter")
    ax.set_title(f"{title_prefix}: viability vs size")
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_time_series_metric(
    ts_df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    out_path: Path,
    title_prefix: str,
    dpi: int,
) -> None:
    if ts_df.empty or metric_col not in ts_df.columns:
        save_empty_plot(
            out_path,
            ylabel,
            f"No time-series data found for {metric_col}.",
            dpi,
        )
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for question, g in ts_df.groupby("question", dropna=False):
        x = pd.to_numeric(g["iteration"], errors="coerce")
        y = pd.to_numeric(g[metric_col], errors="coerce")

        keep = np.isfinite(x) & np.isfinite(y)

        if keep.sum() == 0:
            continue

        gg = pd.DataFrame({"x": x[keep], "y": y[keep]}).sort_values("x")

        ax.plot(
            gg["x"].to_numpy(),
            gg["y"].to_numpy(),
            linewidth=1.5,
            alpha=0.85,
            label=str(question),
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title_prefix}: {ylabel} over iterations")
    ax.grid(True, alpha=0.25)

    if ts_df["question"].nunique() <= 20:
        ax.legend(fontsize=8, loc="best")
    else:
        ax.legend(fontsize=6, loc="center left", bbox_to_anchor=(1.02, 0.5))

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_all_time_series(
    ts_df: pd.DataFrame,
    plot_dir: Path,
    title_prefix: str,
    dpi: int,
) -> None:
    plot_time_series_metric(
        ts_df=ts_df,
        metric_col="prop_alive_mean",
        ylabel="Mean proportion alive",
        out_path=plot_dir / "time_series_prop_alive.png",
        title_prefix=title_prefix,
        dpi=dpi,
    )

    plot_time_series_metric(
        ts_df=ts_df,
        metric_col="prop_extinct_mean",
        ylabel="Mean proportion extinct",
        out_path=plot_dir / "time_series_prop_extinct.png",
        title_prefix=title_prefix,
        dpi=dpi,
    )

    plot_time_series_metric(
        ts_df=ts_df,
        metric_col="prop_overgrown_mean",
        ylabel="Mean proportion overgrown",
        out_path=plot_dir / "time_series_prop_overgrown.png",
        title_prefix=title_prefix,
        dpi=dpi,
    )

    plot_time_series_metric(
        ts_df=ts_df,
        metric_col="prop_overflow_mean",
        ylabel="Mean proportion overflow",
        out_path=plot_dir / "time_series_prop_overflow.png",
        title_prefix=title_prefix,
        dpi=dpi,
    )

    plot_time_series_metric(
        ts_df=ts_df,
        metric_col="avg_tussock_diameter_mean",
        ylabel="Mean average tussock diameter",
        out_path=plot_dir / "time_series_avg_tussock_diameter.png",
        title_prefix=title_prefix,
        dpi=dpi,
    )

def make_plots(
    summary_df: pd.DataFrame,
    last_df: pd.DataFrame,
    ts_df: pd.DataFrame,
    out_dir: Path,
    title_prefix: str,
    dpi: int,
) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_population_composition(
        last_df=last_df,
        out_path=plot_dir / "question_comparison_population_composition.png",
        title_prefix=title_prefix,
        dpi=dpi,
    )

    plot_population_counts(
        summary_df=summary_df,
        last_df=last_df,
        out_path=plot_dir / "question_comparison_population_counts.png",
        title_prefix=title_prefix,
        dpi=dpi,
    )

    plot_viability(
        summary_df=summary_df,
        out_path=plot_dir / "question_comparison_viability.png",
        title_prefix=title_prefix,
        dpi=dpi,
    )

    plot_tussock_size(
        summary_df=summary_df,
        out_path=plot_dir / "question_comparison_tussock_size.png",
        title_prefix=title_prefix,
        dpi=dpi,
    )

    plot_tussock_diameter_wasserstein(
        last_df=last_df,
        out_path=plot_dir / "question_comparison_tussock_diameter_wasserstein.png",
        title_prefix=title_prefix,
        dpi=dpi,
    )

    plot_viability_vs_size(
        summary_df=summary_df,
        out_path=plot_dir / "question_comparison_viability_vs_size.png",
        title_prefix=title_prefix,
        dpi=dpi,
    )

    plot_all_time_series(
        ts_df=ts_df,
        plot_dir=plot_dir,
        title_prefix=title_prefix,
        dpi=dpi,
    )

def main() -> None:
    args = parse_args()

    h_dir = Path(args.h_dir).resolve()

    if not h_dir.exists():
        raise FileNotFoundError(f"h-dir does not exist: {h_dir}")

    if not h_dir.is_dir():
        raise NotADirectoryError(f"h-dir is not a directory: {h_dir}")

    ecotype = args.ecotype.strip()

    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        if ecotype:
            out_dir = h_dir / f"higher_level_comparison_{ecotype}"
        else:
            out_dir = h_dir / "higher_level_comparison_all_ecotypes"

    out_dir.mkdir(parents=True, exist_ok=True)

    title_prefix = args.title if args.title else h_dir.name

    if ecotype:
        title_prefix = f"{title_prefix} ({ecotype})"
    else:
        title_prefix = f"{title_prefix} (all ecotypes combined)"

    final_pop_df = load_all_final_population_results(
        h_dir=h_dir,
        ecotype=ecotype,
        include_nonstandard_files=args.include_nonstandard_files,
    )

    if final_pop_df.empty:
        raise RuntimeError(
            "No final_population_results.csv files found. Expected layout like:\n"
            f"  {h_dir}/<question>/resampled_fits/<ecotype>/set_001/final_population_results.csv\n"
            f"ecotype filter: {ecotype if ecotype else '[all ecotypes]'}"
        )

    last_df = get_last_iteration_rows(final_pop_df)
    summary_df = summarize_by_question(last_df)
    summary_by_ecotype_df = summarize_by_question_ecotype(last_df)
    ts_df = summarize_time_series(final_pop_df)

    final_pop_df.to_csv(out_dir / "all_final_population_results_compiled.csv", index=False)
    last_df.to_csv(out_dir / "final_iteration_rows_by_set.csv", index=False)
    summary_df.to_csv(out_dir / "question_summary.csv", index=False)
    summary_by_ecotype_df.to_csv(out_dir / "question_summary_by_ecotype.csv", index=False)
    ts_df.to_csv(out_dir / "question_time_series_summary.csv", index=False)

    make_plots(
        summary_df=summary_df,
        last_df=last_df,
        ts_df=ts_df,
        out_dir=out_dir,
        title_prefix=title_prefix,
        dpi=args.plot_dpi,
    )

    print("========================================")
    print(f"h-dir: {h_dir}")
    print(f"ecotype filter: {ecotype if ecotype else '[all ecotypes combined]'}")
    print(f"output directory: {out_dir}")
    print("========================================")
    print(f"compiled rows: {len(final_pop_df)}")
    print(f"final iteration rows: {len(last_df)}")
    print(f"questions found: {final_pop_df['question'].nunique()}")
    print(f"ecotypes found: {final_pop_df['ecotype'].nunique()}")
    print(f"sets found: {final_pop_df['set_id'].nunique()}")
    print("")
    print("CSV outputs:")
    print(f"  {out_dir / 'all_final_population_results_compiled.csv'}")
    print(f"  {out_dir / 'final_iteration_rows_by_set.csv'}")
    print(f"  {out_dir / 'question_summary.csv'}")
    print(f"  {out_dir / 'question_summary_by_ecotype.csv'}")
    print(f"  {out_dir / 'question_time_series_summary.csv'}")
    print("")
    print("Plot outputs:")
    print(f"  {out_dir / 'plots' / 'question_comparison_population_composition.png'}")
    print(f"  {out_dir / 'plots' / 'question_comparison_population_counts.png'}")
    print(f"  {out_dir / 'plots' / 'question_comparison_viability.png'}")
    print(f"  {out_dir / 'plots' / 'question_comparison_tussock_size.png'}")
    print(f"  {out_dir / 'plots' / 'question_comparison_tussock_diameter_wasserstein.png'}")
    print(f"  {out_dir / 'plots' / 'question_comparison_viability_vs_size.png'}")
    print(f"  {out_dir / 'plots' / 'time_series_prop_alive.png'}")
    print(f"  {out_dir / 'plots' / 'time_series_prop_extinct.png'}")
    print(f"  {out_dir / 'plots' / 'time_series_prop_overgrown.png'}")
    print(f"  {out_dir / 'plots' / 'time_series_prop_overflow.png'}")
    print(f"  {out_dir / 'plots' / 'time_series_avg_tussock_diameter.png'}")
    print("")
    print("Question summary:")

    if summary_df.empty:
        print("No summary rows generated.")
    else:
        cols_to_show = [
            "question",
            "n_final_population_files",
            "n_sets",
            "n_ecotypes",
            "prop_alive_mean",
            "prop_extinct_mean",
            "prop_overgrown_mean",
            "prop_overflow_mean",
            "avg_tussock_diameter_mean",
            "diameter_wasserstein_1d_mean",
            "diameter_wasserstein_1d_sd",
            "n_sim_diameters_eligible_for_wasserstein_mean",
            "n_sim_diameters_extinct_excluded_mean",
            "n_sim_diameters_overgrown_excluded_mean",
        ]

        cols_to_show = [c for c in cols_to_show if c in summary_df.columns]
        print(summary_df[cols_to_show].to_string(index=False))


if __name__ == "__main__":
    main()