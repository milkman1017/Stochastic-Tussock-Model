#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Callable, Iterable, Sequence, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOGGER = logging.getLogger("higher_level_comparison")
T = TypeVar("T")
R = TypeVar("R")


FINAL_POP_REQUIRED_COLUMNS = [
    "iteration",
    "alive_tussocks_final",
    "extinct_tussocks_final",
    "overflow_tussocks",
    "avg_tussock_diameter",
]

FINAL_SUMMARY_COLUMNS = [
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
    "cumulative_attempted_daughters",
    "cumulative_established_births",
    "cumulative_deaths",
    "cumulative_tillers_created",
    "cumulative_survival_evaluations",
    "cumulative_survivals",
    "cumulative_reproduction_evaluations",
    "cumulative_establishment_evaluations",
    "mean_survival_probability",
    "mean_reproduction_probability",
    "mean_establishment_probability",
]

YEARLY_METRICS = [
    "n_total",
    "n_alive",
    "n_dead",
    "deaths_this_step",
    "survival_evaluations",
    "survivors_this_step",
    "reproduction_evaluations",
    "attempted_daughters",
    "establishment_evaluations",
    "established_daughters",
    "realized_survival_rate",
    "realized_reproduction_rate",
    "realized_establishment_rate",
    "mean_survival_probability",
    "mean_reproduction_probability",
    "mean_establishment_probability",
    "cumulative_attempted_daughters",
    "cumulative_established_births",
    "cumulative_deaths",
    "cumulative_tillers_created",
    "diameter",
    "radius",
    "leaf_area_mean",
    "overflow",
]

FINAL_DISTRIBUTION_METRICS = [
    "final_diameter",
    "alive_final",
    "LeafArea",
    "cumulative_attempted_daughters",
    "cumulative_established_births",
    "cumulative_deaths",
    "cumulative_tillers_created",
    "mean_survival_probability",
    "mean_reproduction_probability",
    "mean_establishment_probability",
    "establishment_success_rate",
    "turnover_ratio",
    "net_recruitment",
    "births_per_final_alive",
    "realized_survival_rate_overall",
    "realized_reproduction_rate_overall",
    "realized_establishment_rate_overall",
]

OPTIMIZER_HISTORY_METRICS = [
    "prop_alive",
    "prop_extinct",
    "prop_overflow",
    "avg_tussock_diameter",
]

BIOLOGICAL_PLAUSIBILITY_COMPONENTS = [
    "prior_alive_trajectory_penalty",
    "prior_population_growth_penalty",
    "prior_young_radius_growth_penalty",
    "prior_mature_plateau_penalty",
    "prior_turnover_penalty",
    "prior_birth_timing_penalty",
    "prior_death_timing_penalty",
    "prior_demographic_activity_penalty",
]

BIOLOGICAL_PRIOR_REFERENCE = (
    "Broad Eriophorum vaginatum priors informed by Fetcher & Shaver (1982, 1983), "
    "Chandler et al. (2015), and Curasi et al. (2023; doi:10.1111/nph.18751). "
    "They are intended as weak plausibility constraints, not exact observations."
)


# ---------------------------------------------------------------------------
# CLI and generic helpers
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare hypothesis/question outputs from one h-dir. Expected layout: "
            "<h_dir>/<question>/resampled_fits/<ecotype>/set_001/"
        )
    )
    parser.add_argument("--h-dir", required=True, help="Top hypothesis directory.")
    parser.add_argument(
        "--ecotype",
        default="",
        help="Optional ecotype/site restriction. Empty means combine all ecotypes.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults inside --h-dir.",
    )
    parser.add_argument("--plot-dpi", type=int, default=250)
    parser.add_argument("--title", default=None)
    parser.add_argument(
        "--include-nonstandard-files",
        action="store_true",
        help="Also include final_population_results*.csv files.",
    )

    # Optional biological gates. With the defaults, no hard gate is imposed.
    parser.add_argument("--min-alive", type=float, default=None)
    parser.add_argument("--max-extinct", type=float, default=None)
    parser.add_argument("--max-overflow", type=float, default=None)

    # Dimensionless score weights.
    parser.add_argument("--weight-diameter", type=float, default=1.0)
    parser.add_argument("--weight-extinct", type=float, default=1.0)
    parser.add_argument("--weight-overflow", type=float, default=1.0)
    parser.add_argument("--weight-trajectory", type=float, default=1.0)
    parser.add_argument(
        "--weight-plausibility",
        type=float,
        default=2.0,
        help=(
            "Weight for the default biological-plausibility loss. This loss is "
            "calculated from the simulated yearly trajectories even when no "
            "empirical trajectory-target CSV is supplied."
        ),
    )
    parser.add_argument(
        "--disable-plausibility-score",
        action="store_true",
        help="Disable the default biological-plausibility contribution to model ranking.",
    )
    parser.add_argument(
        "--missing-diameter-penalty",
        type=float,
        default=5.0,
        help="Score contribution when diameter Wasserstein cannot be calculated.",
    )
    parser.add_argument(
        "--q90-weight",
        type=float,
        default=0.5,
        help="Question robust score = median set score + this * q90 set score.",
    )

    # Broad biological priors. These are deliberately configurable because the
    # exact founding state, simulation duration, ecotype, and environment vary.
    parser.add_argument(
        "--prior-adult-alive-min",
        type=float,
        default=100.0,
        help="Soft lower bound for live tillers in a mature tussock.",
    )
    parser.add_argument(
        "--prior-adult-alive-max",
        type=float,
        default=600.0,
        help="Soft upper bound for live tillers in a mature tussock.",
    )
    parser.add_argument(
        "--prior-population-growth-rate",
        type=float,
        default=0.20,
        help="Central early per-capita tiller growth rate used for the rough logistic prior.",
    )
    parser.add_argument(
        "--prior-max-population-growth-rate",
        type=float,
        default=0.52,
        help="Soft upper bound on annual log growth of live tiller number.",
    )
    parser.add_argument(
        "--prior-alive-envelope-factor",
        type=float,
        default=4.0,
        help="Multiplicative uncertainty around the rough live-tiller trajectory.",
    )
    parser.add_argument(
        "--prior-mature-years",
        type=float,
        default=50.0,
        help="Approximate time by which growth should be approaching an asymptote.",
    )
    parser.add_argument(
        "--prior-young-radius-growth-min",
        type=float,
        default=0.05,
        help="Soft lower bound for young-tussock radial growth in cm per year.",
    )
    parser.add_argument(
        "--prior-young-radius-growth-max",
        type=float,
        default=0.45,
        help="Soft upper bound for young-tussock radial growth in cm per year.",
    )
    parser.add_argument(
        "--prior-mature-relative-growth-max",
        type=float,
        default=0.05,
        help="Soft upper bound on late annual relative growth of live tiller number.",
    )
    parser.add_argument(
        "--prior-mature-radius-growth-abs-max",
        type=float,
        default=0.10,
        help="Soft bound on absolute late radius growth or decline in cm per year.",
    )
    parser.add_argument(
        "--prior-first-birth-min-year",
        type=float,
        default=1.0,
        help="Soft earliest plausible year of first established daughter production.",
    )
    parser.add_argument(
        "--prior-first-birth-max-year",
        type=float,
        default=10.0,
        help="Soft latest plausible year of first established daughter production.",
    )
    parser.add_argument(
        "--prior-first-death-max-year",
        type=float,
        default=12.0,
        help="Soft latest plausible year by which some tiller mortality should occur.",
    )
    parser.add_argument(
        "--prior-turnover-ratio-min",
        type=float,
        default=0.25,
        help="Soft lower bound on cumulative deaths divided by established births.",
    )
    parser.add_argument(
        "--prior-turnover-ratio-max",
        type=float,
        default=1.75,
        help="Soft upper bound on cumulative deaths divided by established births.",
    )
    parser.add_argument(
        "--prior-min-active-event-year-fraction",
        type=float,
        default=0.10,
        help="Soft minimum fraction of modeled years with births and with deaths.",
    )
    parser.add_argument(
        "--prior-max-single-year-event-fraction",
        type=float,
        default=0.75,
        help="Soft maximum fraction of all births or deaths occurring in one year.",
    )
    parser.add_argument(
        "--max-plausibility-loss",
        type=float,
        default=2.0,
        help=(
            "Default biological gate: parameter sets above this plausibility loss "
            "are marked as failing. Set to a negative value to disable this gate."
        ),
    )

    parser.add_argument(
        "--trajectory-targets-csv",
        default=None,
        help=(
            "Optional CSV with time_step plus target metric columns. Optional "
            "<metric>_scale columns define normalization scales."
        ),
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=1000,
        help="Hierarchical bootstrap replicates for fate confidence intervals.",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=12345)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Reader/analysis worker threads. 0 chooses an automatic value; "
            "1 disables parallel loading."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Log progress after this many completed parallel jobs.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Log path. Defaults to <out-dir>/higher_level_comparison.log.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip plot generation when only tables are needed.",
    )
    parser.add_argument(
        "--skip-large-compiled-csvs",
        action="store_true",
        help=(
            "Skip the two potentially huge row-level CSVs while retaining all "
            "set-, question-, and trajectory-summary outputs."
        ),
    )
    return parser.parse_args()


def resolve_workers(requested: int) -> int:
    if requested < 0:
        raise ValueError("--workers must be >= 0")
    if requested == 0:
        return max(1, min(32, os.cpu_count() or 1))
    return max(1, requested)


def setup_logging(log_file: Path, level_name: str) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.handlers.clear()
    LOGGER.setLevel(getattr(logging, level_name.upper()))
    LOGGER.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(threadName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    LOGGER.addHandler(console)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)


@contextmanager
def logged_stage(name: str):
    start = time.perf_counter()
    LOGGER.info("START %s", name)
    try:
        yield
    except Exception:
        LOGGER.exception("FAILED %s after %.2f s", name, time.perf_counter() - start)
        raise
    else:
        LOGGER.info("DONE  %s in %.2f s", name, time.perf_counter() - start)


def parallel_map_ordered(
    func: Callable[[T], R],
    items: Sequence[T],
    workers: int,
    label: str,
    progress_every: int,
) -> list[R]:
    item_list = list(items)
    total = len(item_list)
    if total == 0:
        LOGGER.info("%s: nothing to process", label)
        return []

    progress_every = max(1, progress_every)
    effective_workers = min(max(1, workers), total)
    LOGGER.info("%s: %d jobs using %d thread(s)", label, total, effective_workers)
    results: list[R | None] = [None] * total
    failures = 0

    if effective_workers == 1:
        for index, item in enumerate(item_list):
            try:
                results[index] = func(item)
            except Exception as exc:
                failures += 1
                LOGGER.warning("%s failed for %s: %s", label, item, exc)
            completed = index + 1
            if completed % progress_every == 0 or completed == total:
                LOGGER.info("%s: %d/%d complete", label, completed, total)
    else:
        with ThreadPoolExecutor(
            max_workers=effective_workers,
            thread_name_prefix="reader",
        ) as executor:
            future_to_index = {
                executor.submit(func, item): index
                for index, item in enumerate(item_list)
            }
            completed = 0
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                item = item_list[index]
                try:
                    results[index] = future.result()
                except Exception as exc:
                    failures += 1
                    LOGGER.warning("%s failed for %s: %s", label, item, exc)
                completed += 1
                if completed % progress_every == 0 or completed == total:
                    LOGGER.info("%s: %d/%d complete", label, completed, total)

    if failures:
        LOGGER.warning("%s: completed with %d failed job(s)", label, failures)
    return [result for result in results if result is not None]


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    start = time.perf_counter()
    LOGGER.info("Writing %s (%d rows, %d columns)", path.name, len(frame), len(frame.columns))
    try:
        frame.to_csv(temp_path, index=False)
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    LOGGER.info("Wrote %s in %.2f s", path.name, time.perf_counter() - start)


def is_set_dir_name(name: str) -> bool:
    return re.fullmatch(r"set_\d+", str(name)) is not None


def infer_set_id(path: Path) -> str:
    for part in path.parts:
        if is_set_dir_name(part):
            return part
    return ""


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    n = numeric(numerator)
    d = numeric(denominator)
    return pd.Series(np.where(np.isfinite(d) & (d > 0), n / d, np.nan), index=n.index)


def finite_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def q(values: Iterable[float], probability: float) -> float:
    arr = finite_array(values)
    return float(np.quantile(arr, probability)) if arr.size else np.nan


def robust_scale(values: Iterable[float]) -> float:
    arr = finite_array(values)
    if arr.size == 0:
        return 1.0
    iqr = float(np.quantile(arr, 0.75) - np.quantile(arr, 0.25))
    if np.isfinite(iqr) and iqr > 1e-12:
        return iqr
    sd = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    if np.isfinite(sd) and sd > 1e-12:
        return sd
    med = abs(float(np.median(arr)))
    return med if med > 1e-12 else 1.0


def capped_square(value: float, cap: float = 25.0) -> float:
    if not np.isfinite(value):
        return cap
    return float(min(cap, max(0.0, value) ** 2))


def interval_penalty(
    value: float,
    lower: float,
    upper: float,
    scale: float | None = None,
) -> float:
    """Zero inside an interval and a smooth squared penalty outside it."""
    if not np.isfinite(value):
        return 4.0
    if lower > upper:
        lower, upper = upper, lower
    if scale is None or not np.isfinite(scale) or scale <= 0:
        scale = max(upper - lower, abs(lower), abs(upper), 1.0)
    if value < lower:
        return capped_square((lower - value) / scale)
    if value > upper:
        return capped_square((value - upper) / scale)
    return 0.0


def upper_penalty(value: float, upper: float, scale: float) -> float:
    if not np.isfinite(value):
        return 4.0
    return capped_square(max(0.0, value - upper) / max(scale, 1e-12))


def lower_penalty(value: float, lower: float, scale: float) -> float:
    if not np.isfinite(value):
        return 4.0
    return capped_square(max(0.0, lower - value) / max(scale, 1e-12))


def linear_slope(x: Iterable[float], y: Iterable[float]) -> float:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    keep = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(keep.sum()) < 2:
        return np.nan
    x_kept = x_arr[keep]
    y_kept = y_arr[keep]
    if np.ptp(x_kept) <= 1e-12:
        return np.nan
    return float(np.polyfit(x_kept, y_kept, 1)[0])


def logistic_population(time_values: np.ndarray, n0: float, carrying_capacity: float, rate: float) -> np.ndarray:
    time_values = np.asarray(time_values, dtype=float)
    n0 = max(float(n0), 1e-6)
    carrying_capacity = max(float(carrying_capacity), n0 + 1e-6)
    rate = max(float(rate), 1e-6)
    ratio = (carrying_capacity - n0) / n0
    return carrying_capacity / (1.0 + ratio * np.exp(-rate * time_values))


def rough_alive_prior_bounds(
    time_values: np.ndarray,
    initial_alive: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a weak logistic center and a multiplicative uncertainty envelope."""
    time_values = np.asarray(time_values, dtype=float)
    shifted_time = np.maximum(0.0, time_values - np.nanmin(time_values))
    adult_center = 0.5 * (args.prior_adult_alive_min + args.prior_adult_alive_max)
    center = logistic_population(
        shifted_time,
        max(initial_alive, 1.0),
        adult_center,
        args.prior_population_growth_rate,
    )
    factor = max(float(args.prior_alive_envelope_factor), 1.01)
    lower = np.maximum(1.0, center / factor)
    upper = np.minimum(
        max(float(args.prior_adult_alive_max), 1.0) * 1.25,
        center * factor,
    )
    return center, lower, upper


def wasserstein_distance_1d(x: Iterable[float], y: Iterable[float]) -> float:
    x_arr = finite_array(x)
    y_arr = finite_array(y)
    if x_arr.size == 0 or y_arr.size == 0:
        return np.nan

    x_sorted = np.sort(x_arr)
    y_sorted = np.sort(y_arr)
    n = x_sorted.size
    m = y_sorted.size
    i = j = 0
    cdf_x = cdf_y = 0.0
    previous = min(x_sorted[0], y_sorted[0])
    distance = 0.0

    while i < n or j < m:
        next_x = x_sorted[i] if i < n else np.inf
        next_y = y_sorted[j] if j < m else np.inf
        current = min(next_x, next_y)
        distance += abs(cdf_x - cdf_y) * (current - previous)

        if next_x == current:
            value = current
            while i < n and x_sorted[i] == value:
                i += 1
            cdf_x = i / n
        if next_y == current:
            value = current
            while j < m and y_sorted[j] == value:
                j += 1
            cdf_y = j / m
        previous = current

    return float(distance)


def save_empty_plot(path: Path, title: str, message: str, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def metadata_from_result_file(h_dir: Path, result_file: Path) -> dict[str, str]:
    relative = result_file.relative_to(h_dir)
    parts = relative.parts
    question = parts[0] if parts else ""
    ecotype = ""
    if "resampled_fits" in parts:
        index = parts.index("resampled_fits")
        if index + 1 < len(parts):
            ecotype = parts[index + 1]
    set_id = infer_set_id(result_file)
    set_key = f"{question}|{ecotype}|{set_id}"
    return {
        "h_dir": h_dir.name,
        "question": question,
        "ecotype": ecotype,
        "set_id": set_id,
        "set_key": set_key,
        "set_dir": str(result_file.parent),
        "final_population_result_file": str(result_file),
    }


# ---------------------------------------------------------------------------
# File discovery and optimizer history
# ---------------------------------------------------------------------------


def find_final_population_files(
    h_dir: Path,
    ecotype: str,
    include_nonstandard_files: bool,
) -> list[Path]:
    files: list[Path] = []
    for question_dir in sorted(path for path in h_dir.iterdir() if path.is_dir()):
        resampled_dir = question_dir / "resampled_fits"
        if not resampled_dir.is_dir():
            continue
        ecotype_dirs = [resampled_dir / ecotype] if ecotype else sorted(
            path for path in resampled_dir.iterdir() if path.is_dir()
        )
        for ecotype_dir in ecotype_dirs:
            if not ecotype_dir.is_dir():
                continue
            set_dirs = sorted(
                path for path in ecotype_dir.iterdir()
                if path.is_dir() and is_set_dir_name(path.name)
            )
            for set_dir in set_dirs:
                candidates = (
                    sorted(set_dir.glob("final_population_results*.csv"))
                    if include_nonstandard_files
                    else [set_dir / "final_population_results.csv"]
                )
                files.extend(path for path in candidates if path.exists())
    return files


def load_optimizer_history(h_dir: Path, result_file: Path) -> pd.DataFrame:
    frame = pd.read_csv(result_file)
    for column in FINAL_POP_REQUIRED_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = numeric(frame[column])

    metadata = metadata_from_result_file(h_dir, result_file)
    for key, value in metadata.items():
        frame[key] = value

    # Overgrown is deprecated and deliberately excluded.
    total = (
        frame["alive_tussocks_final"].fillna(0)
        + frame["extinct_tussocks_final"].fillna(0)
        + frame["overflow_tussocks"].fillna(0)
    )
    frame["total_tussocks_classified"] = total
    frame["prop_alive"] = np.where(total > 0, frame["alive_tussocks_final"] / total, np.nan)
    frame["prop_extinct"] = np.where(total > 0, frame["extinct_tussocks_final"] / total, np.nan)
    frame["prop_overflow"] = np.where(total > 0, frame["overflow_tussocks"] / total, np.nan)
    return frame


def load_all_optimizer_history(
    h_dir: Path,
    ecotype: str,
    include_nonstandard_files: bool,
    workers: int,
    progress_every: int,
) -> tuple[pd.DataFrame, list[Path]]:
    files = find_final_population_files(h_dir, ecotype, include_nonstandard_files)
    loader = partial(load_optimizer_history, h_dir)
    frames = parallel_map_ordered(
        loader,
        files,
        workers,
        "optimizer histories",
        progress_every,
    )
    frames = [frame for frame in frames if not frame.empty]
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(), files)


def get_last_optimizer_rows(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()
    rows: list[pd.Series] = []
    for _, group in history.groupby("set_key", dropna=False):
        group = group[np.isfinite(numeric(group["iteration"]))].copy()
        if not group.empty:
            rows.append(group.sort_values("iteration").iloc[-1])
    return pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame()


def summarize_optimizer_history(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()
    rows: list[dict[str, float | int | str]] = []
    for (question, iteration), group in history.groupby(["question", "iteration"], dropna=False):
        row: dict[str, float | int | str] = {
            "question": str(question),
            "iteration": float(iteration),
            "n_sets": int(group["set_key"].nunique()),
            "n_ecotypes": int(group["ecotype"].nunique()),
        }
        for metric in OPTIMIZER_HISTORY_METRICS:
            values = finite_array(group[metric])
            row[f"{metric}_mean"] = float(np.mean(values)) if values.size else np.nan
            row[f"{metric}_median"] = float(np.median(values)) if values.size else np.nan
            row[f"{metric}_q10"] = q(values, 0.10)
            row[f"{metric}_q90"] = q(values, 0.90)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["question", "iteration"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Final stochastic simulation summaries
# ---------------------------------------------------------------------------


def read_training_diameters(set_dir: Path, ecotype: str) -> np.ndarray:
    path = set_dir / "sampled_training_data.csv"
    if not path.exists():
        return np.array([], dtype=float)
    try:
        frame = pd.read_csv(path)
    except Exception:
        return np.array([], dtype=float)
    if "diam" not in frame.columns:
        return np.array([], dtype=float)
    if ecotype and ecotype != "ALL" and "site" in frame.columns:
        frame = frame[frame["site"].astype(str) == str(ecotype)]
    return finite_array(numeric(frame["diam"]))


def load_final_sim_summaries(h_dir: Path, result_file: Path) -> pd.DataFrame:
    metadata = metadata_from_result_file(h_dir, result_file)
    set_dir = result_file.parent
    summary_dir = set_dir / "final_sims" / "summaries"
    if not summary_dir.is_dir():
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for path in sorted(summary_dir.glob("summary_*.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            LOGGER.warning("Could not read %s: %s", path, exc)
            continue
        for column in FINAL_SUMMARY_COLUMNS:
            if column not in frame.columns:
                frame[column] = np.nan
            frame[column] = numeric(frame[column])
        frame["summary_file"] = str(path)
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    frame = pd.concat(frames, ignore_index=True)
    for key, value in metadata.items():
        frame[key] = value

    # Mutually exclusive final fate. Overflow takes precedence because an
    # overflow run normally terminates while still containing living tillers.
    overflow = np.isfinite(frame["overflow_t"]) & (frame["overflow_t"] >= 0)
    extinct = (~overflow) & (frame["alive_final"].fillna(0) <= 0)
    frame["final_fate"] = np.select(
        [overflow, extinct],
        ["overflow", "extinct"],
        default="alive",
    )

    frame["establishment_success_rate"] = safe_divide(
        frame["cumulative_established_births"],
        frame["cumulative_attempted_daughters"],
    )
    frame["turnover_ratio"] = safe_divide(
        frame["cumulative_deaths"],
        frame["cumulative_established_births"],
    )
    frame["net_recruitment"] = (
        frame["cumulative_established_births"] - frame["cumulative_deaths"]
    )
    frame["births_per_final_alive"] = safe_divide(
        frame["cumulative_established_births"],
        frame["alive_final"],
    )
    frame["realized_survival_rate_overall"] = safe_divide(
        frame["cumulative_survivals"],
        frame["cumulative_survival_evaluations"],
    )
    frame["realized_reproduction_rate_overall"] = safe_divide(
        frame["cumulative_attempted_daughters"],
        frame["cumulative_reproduction_evaluations"],
    )
    frame["realized_establishment_rate_overall"] = safe_divide(
        frame["cumulative_established_births"],
        frame["cumulative_establishment_evaluations"],
    )
    return frame


def load_all_final_sim_summaries(
    h_dir: Path,
    result_files: Sequence[Path],
    workers: int,
    progress_every: int,
) -> pd.DataFrame:
    loader = partial(load_final_sim_summaries, h_dir)
    frames = parallel_map_ordered(
        loader,
        result_files,
        workers,
        "final simulation summaries",
        progress_every,
    )
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Yearly biological trajectories
# ---------------------------------------------------------------------------


def load_yearly_summaries(h_dir: Path, result_file: Path) -> pd.DataFrame:
    metadata = metadata_from_result_file(h_dir, result_file)
    yearly_dir = result_file.parent / "final_sims" / "yearly_summaries"
    if not yearly_dir.is_dir():
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for path in sorted(yearly_dir.glob("yearly_summary_*.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            LOGGER.warning("Could not read %s: %s", path, exc)
            continue
        if "sim_id" not in frame.columns:
            frame["sim_id"] = np.nan
        if "time_step" not in frame.columns:
            continue
        frame["sim_id"] = numeric(frame["sim_id"])
        frame["time_step"] = numeric(frame["time_step"])
        for metric in YEARLY_METRICS:
            if metric not in frame.columns:
                frame[metric] = np.nan
            frame[metric] = numeric(frame[metric])
        frame["yearly_summary_file"] = str(path)
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    frame = pd.concat(frames, ignore_index=True)
    for key, value in metadata.items():
        frame[key] = value
    return frame


def load_all_yearly_summaries(
    h_dir: Path,
    result_files: Sequence[Path],
    workers: int,
    progress_every: int,
) -> pd.DataFrame:
    loader = partial(load_yearly_summaries, h_dir)
    frames = parallel_map_ordered(
        loader,
        result_files,
        workers,
        "yearly simulation summaries",
        progress_every,
    )
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_yearly_by_set(yearly_sim: pd.DataFrame) -> pd.DataFrame:
    if yearly_sim.empty:
        return pd.DataFrame()
    rows: list[dict[str, float | int | str]] = []
    keys = ["question", "ecotype", "set_id", "set_key", "time_step"]
    for group_key, group in yearly_sim.groupby(keys, dropna=False):
        row: dict[str, float | int | str] = dict(zip(keys, group_key))
        row["n_sims"] = int(group["sim_id"].nunique())
        for metric in YEARLY_METRICS:
            values = finite_array(group[metric])
            row[f"{metric}_mean"] = float(np.mean(values)) if values.size else np.nan
            row[f"{metric}_median"] = float(np.median(values)) if values.size else np.nan
            row[f"{metric}_q10"] = q(values, 0.10)
            row[f"{metric}_q90"] = q(values, 0.90)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["question", "ecotype", "set_id", "time_step"])


def summarize_yearly_by_question(yearly_set: pd.DataFrame) -> pd.DataFrame:
    """Aggregate set medians so each fitted parameter set receives equal weight."""
    if yearly_set.empty:
        return pd.DataFrame()
    rows: list[dict[str, float | int | str]] = []
    for (question, time_step), group in yearly_set.groupby(["question", "time_step"], dropna=False):
        row: dict[str, float | int | str] = {
            "question": str(question),
            "time_step": float(time_step),
            "n_sets": int(group["set_key"].nunique()),
            "n_ecotypes": int(group["ecotype"].nunique()),
        }
        for metric in YEARLY_METRICS:
            source = f"{metric}_median"
            values = finite_array(group[source]) if source in group.columns else np.array([])
            row[f"{metric}_mean_across_sets"] = float(np.mean(values)) if values.size else np.nan
            row[f"{metric}_median_across_sets"] = float(np.median(values)) if values.size else np.nan
            row[f"{metric}_q10_across_sets"] = q(values, 0.10)
            row[f"{metric}_q90_across_sets"] = q(values, 0.90)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["question", "time_step"])


# ---------------------------------------------------------------------------
# Optional trajectory targets and per-set scores
# ---------------------------------------------------------------------------


def read_trajectory_targets(path_string: str | None) -> pd.DataFrame:
    if not path_string:
        return pd.DataFrame()
    path = Path(path_string).resolve()
    if not path.exists():
        raise FileNotFoundError(f"trajectory target file does not exist: {path}")
    frame = pd.read_csv(path)
    if "time_step" not in frame.columns:
        raise ValueError("trajectory targets CSV must contain time_step")
    frame["time_step"] = numeric(frame["time_step"])
    return frame


def trajectory_loss_for_set(set_yearly: pd.DataFrame, targets: pd.DataFrame) -> tuple[float, int]:
    if set_yearly.empty or targets.empty:
        return np.nan, 0
    available_metrics = [
        metric for metric in YEARLY_METRICS
        if metric in targets.columns and f"{metric}_median" in set_yearly.columns
    ]
    if not available_metrics:
        return np.nan, 0

    merged = targets.merge(set_yearly, on="time_step", how="inner")
    losses: list[float] = []
    for metric in available_metrics:
        observed = numeric(merged[metric])
        simulated = numeric(merged[f"{metric}_median"])
        keep = np.isfinite(observed) & np.isfinite(simulated)
        if not keep.any():
            continue
        scale_column = f"{metric}_scale"
        if scale_column in merged.columns:
            scales = numeric(merged.loc[keep, scale_column]).to_numpy(dtype=float)
            fallback = robust_scale(observed[keep])
            scales = np.where(np.isfinite(scales) & (scales > 0), scales, fallback)
        else:
            scales = np.full(int(keep.sum()), robust_scale(observed[keep]), dtype=float)
        residual = (simulated[keep].to_numpy(dtype=float) - observed[keep].to_numpy(dtype=float)) / scales
        losses.append(float(np.sqrt(np.mean(residual * residual))))
    return (float(np.mean(losses)), len(losses)) if losses else (np.nan, 0)


def biological_plausibility_for_set(
    set_yearly: pd.DataFrame,
    final_group: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, float | int | str]:
    """Score weak, literature-informed constraints on tussock development.

    A value of zero means all evaluated quantities fall inside broad prior
    envelopes. Values around one indicate a moderate violation. Large values
    identify trajectories that are qualitatively inconsistent with a slowly
    saturating, continuously turning-over tussock population.
    """
    output: dict[str, float | int | str] = {
        component: np.nan for component in BIOLOGICAL_PLAUSIBILITY_COMPONENTS
    }
    output.update(
        {
            "biological_plausibility_loss": np.nan,
            "plausibility_components_evaluated": 0,
            "plausibility_prior_reference": BIOLOGICAL_PRIOR_REFERENCE,
        }
    )
    if set_yearly.empty:
        output["biological_plausibility_loss"] = 8.0
        output["plausibility_missing_data_penalty"] = 8.0
        return output

    frame = set_yearly.sort_values("time_step").copy()
    time_values = numeric(frame["time_step"]).to_numpy(dtype=float)
    keep_time = np.isfinite(time_values)
    frame = frame.loc[keep_time].copy()
    time_values = time_values[keep_time]
    if len(frame) < 2:
        output["biological_plausibility_loss"] = 8.0
        output["plausibility_missing_data_penalty"] = 8.0
        return output

    def metric_values(metric: str, statistic: str = "median") -> np.ndarray:
        column = f"{metric}_{statistic}"
        if column not in frame.columns:
            return np.full(len(frame), np.nan, dtype=float)
        return numeric(frame[column]).to_numpy(dtype=float)

    alive = metric_values("n_alive")
    diameter = metric_values("diameter")
    radius = metric_values("radius")
    if not np.isfinite(radius).any() and np.isfinite(diameter).any():
        radius = diameter / 2.0
    # Means are used for sparse annual events so asynchronous stochastic
    # births/deaths are not erased by a zero median across simulations.
    established = metric_values("established_daughters", "mean")
    deaths = metric_values("deaths_this_step", "mean")
    cumulative_births = metric_values("cumulative_established_births")
    cumulative_deaths = metric_values("cumulative_deaths")

    finite_alive = alive[np.isfinite(alive) & (alive >= 0)]
    initial_alive = float(finite_alive[0]) if finite_alive.size else 1.0
    _, alive_lower, alive_upper = rough_alive_prior_bounds(time_values, initial_alive, args)
    factor_log = np.log(max(args.prior_alive_envelope_factor, 1.01))
    alive_penalties: list[float] = []
    for observed, lower, upper in zip(alive, alive_lower, alive_upper):
        if not np.isfinite(observed) or observed < 0:
            continue
        if observed < lower:
            alive_penalties.append(capped_square(np.log(lower / max(observed, 0.25)) / factor_log))
        elif observed > upper:
            alive_penalties.append(capped_square(np.log(observed / max(upper, 0.25)) / factor_log))
        else:
            alive_penalties.append(0.0)
    output["prior_alive_trajectory_penalty"] = (
        float(np.mean(alive_penalties)) if alive_penalties else 4.0
    )

    growth_penalties: list[float] = []
    for index in range(1, len(alive)):
        dt = time_values[index] - time_values[index - 1]
        previous = alive[index - 1]
        current = alive[index]
        if dt <= 0 or not np.isfinite(previous) or not np.isfinite(current):
            continue
        if previous <= 0 or current <= 0:
            continue
        annual_log_growth = np.log(current / previous) / dt
        growth_penalties.append(
            upper_penalty(
                annual_log_growth,
                args.prior_max_population_growth_rate,
                max(0.15, 0.30 * args.prior_max_population_growth_rate),
            )
        )
    output["prior_population_growth_penalty"] = (
        float(np.mean(growth_penalties)) if growth_penalties else 2.0
    )

    first_time = float(np.nanmin(time_values))
    final_time = float(np.nanmax(time_values))
    elapsed = max(0.0, final_time - first_time)
    young_window_end = first_time + min(10.0, max(3.0, 0.25 * elapsed))
    young_keep = time_values <= young_window_end
    young_radius_slope = linear_slope(time_values[young_keep], radius[young_keep])
    output["prior_young_radius_growth_cm_per_year"] = young_radius_slope
    output["prior_young_radius_growth_penalty"] = interval_penalty(
        young_radius_slope,
        args.prior_young_radius_growth_min,
        args.prior_young_radius_growth_max,
        scale=max(
            args.prior_young_radius_growth_max - args.prior_young_radius_growth_min,
            0.20,
        ),
    )

    mature_penalties: list[float] = []
    late_relative_alive_growth = np.nan
    late_radius_slope = np.nan
    if elapsed >= 0.75 * args.prior_mature_years:
        late_window_start = max(
            first_time + 0.75 * elapsed,
            final_time - max(10.0, 0.20 * elapsed),
        )
        late_keep = time_values >= late_window_start
        alive_late = alive[late_keep]
        time_late = time_values[late_keep]
        alive_slope = linear_slope(time_late, alive_late)
        alive_level = float(np.nanmedian(alive_late)) if np.isfinite(alive_late).any() else np.nan
        late_relative_alive_growth = (
            alive_slope / alive_level
            if np.isfinite(alive_slope) and np.isfinite(alive_level) and alive_level > 0
            else np.nan
        )
        mature_penalties.append(
            upper_penalty(
                abs(late_relative_alive_growth),
                args.prior_mature_relative_growth_max,
                max(args.prior_mature_relative_growth_max, 0.025),
            )
        )
        late_radius_slope = linear_slope(time_late, radius[late_keep])
        mature_penalties.append(
            upper_penalty(
                abs(late_radius_slope),
                args.prior_mature_radius_growth_abs_max,
                max(args.prior_mature_radius_growth_abs_max, 0.05),
            )
        )
    else:
        mature_penalties.append(0.0)
    output["prior_late_relative_alive_growth_per_year"] = late_relative_alive_growth
    output["prior_late_radius_growth_cm_per_year"] = late_radius_slope
    output["prior_mature_plateau_penalty"] = float(np.mean(mature_penalties))

    final_births = (
        float(cumulative_births[np.isfinite(cumulative_births)][-1])
        if np.isfinite(cumulative_births).any()
        else float(np.nanmedian(numeric(final_group["cumulative_established_births"])))
    )
    final_deaths = (
        float(cumulative_deaths[np.isfinite(cumulative_deaths)][-1])
        if np.isfinite(cumulative_deaths).any()
        else float(np.nanmedian(numeric(final_group["cumulative_deaths"])))
    )
    turnover_ratio = final_deaths / final_births if final_births > 0 else np.nan
    output["prior_cumulative_births"] = final_births
    output["prior_cumulative_deaths"] = final_deaths
    output["prior_turnover_ratio"] = turnover_ratio
    turnover_parts = [
        lower_penalty(final_births, 1.0, 5.0),
        lower_penalty(final_deaths, 1.0, 5.0),
        interval_penalty(
            turnover_ratio,
            args.prior_turnover_ratio_min,
            args.prior_turnover_ratio_max,
            scale=max(args.prior_turnover_ratio_max - args.prior_turnover_ratio_min, 0.5),
        ),
    ]
    output["prior_turnover_penalty"] = float(np.mean(turnover_parts))

    def first_positive_time(values: np.ndarray) -> float:
        keep = np.isfinite(values) & (values > 0)
        return float(time_values[np.flatnonzero(keep)[0]]) if keep.any() else np.nan

    first_birth_time = first_positive_time(established)
    first_death_time = first_positive_time(deaths)
    output["prior_first_birth_year"] = first_birth_time
    output["prior_first_death_year"] = first_death_time
    if np.isfinite(first_birth_time):
        output["prior_birth_timing_penalty"] = interval_penalty(
            first_birth_time - first_time,
            args.prior_first_birth_min_year,
            args.prior_first_birth_max_year,
            scale=max(args.prior_first_birth_max_year - args.prior_first_birth_min_year, 3.0),
        )
    else:
        output["prior_birth_timing_penalty"] = 8.0
    if np.isfinite(first_death_time):
        output["prior_death_timing_penalty"] = upper_penalty(
            first_death_time - first_time,
            args.prior_first_death_max_year,
            max(4.0, 0.5 * args.prior_first_death_max_year),
        )
    else:
        output["prior_death_timing_penalty"] = 8.0

    activity_parts: list[float] = []
    active_birth_fraction = float(np.mean(np.isfinite(established) & (established > 0)))
    active_death_fraction = float(np.mean(np.isfinite(deaths) & (deaths > 0)))
    output["prior_active_birth_year_fraction"] = active_birth_fraction
    output["prior_active_death_year_fraction"] = active_death_fraction
    activity_parts.append(
        lower_penalty(
            active_birth_fraction,
            args.prior_min_active_event_year_fraction,
            max(args.prior_min_active_event_year_fraction, 0.05),
        )
    )
    activity_parts.append(
        lower_penalty(
            active_death_fraction,
            args.prior_min_active_event_year_fraction,
            max(args.prior_min_active_event_year_fraction, 0.05),
        )
    )
    for values, name in ((established, "birth"), (deaths, "death")):
        positive = values[np.isfinite(values) & (values > 0)]
        concentration = float(np.max(positive) / np.sum(positive)) if positive.size and np.sum(positive) > 0 else 1.0
        output[f"prior_max_single_year_{name}_fraction"] = concentration
        activity_parts.append(
            upper_penalty(
                concentration,
                args.prior_max_single_year_event_fraction,
                max(0.10, 1.0 - args.prior_max_single_year_event_fraction),
            )
        )
    output["prior_demographic_activity_penalty"] = float(np.mean(activity_parts))

    component_weights = {
        "prior_alive_trajectory_penalty": 2.0,
        "prior_population_growth_penalty": 1.0,
        "prior_young_radius_growth_penalty": 1.0,
        "prior_mature_plateau_penalty": 1.5,
        "prior_turnover_penalty": 1.5,
        "prior_birth_timing_penalty": 0.75,
        "prior_death_timing_penalty": 0.75,
        "prior_demographic_activity_penalty": 1.0,
    }
    weighted_sum = 0.0
    total_weight = 0.0
    evaluated = 0
    for component, weight in component_weights.items():
        value = float(output[component])
        if np.isfinite(value):
            weighted_sum += weight * value
            total_weight += weight
            evaluated += 1
    output["biological_plausibility_loss"] = weighted_sum / total_weight if total_weight else 8.0
    output["plausibility_components_evaluated"] = evaluated
    return output


def biological_prior_configuration(args: argparse.Namespace) -> pd.DataFrame:
    rows = [
        ("adult_alive_min", args.prior_adult_alive_min, "tillers", "soft mature-tussock lower bound"),
        ("adult_alive_max", args.prior_adult_alive_max, "tillers", "adult tussocks commonly reported at several hundred tillers"),
        ("population_growth_rate", args.prior_population_growth_rate, "yr^-1", "center of rough logistic trajectory"),
        ("max_population_growth_rate", args.prior_max_population_growth_rate, "yr^-1", "published broad demographic upper range"),
        ("alive_envelope_factor", args.prior_alive_envelope_factor, "fold", "multiplicative uncertainty around rough trajectory"),
        ("mature_years", args.prior_mature_years, "years", "growth expected to approach an asymptote within several decades"),
        ("young_radius_growth_min", args.prior_young_radius_growth_min, "cm yr^-1", "broad lower young-tussock radius-growth bound"),
        ("young_radius_growth_max", args.prior_young_radius_growth_max, "cm yr^-1", "broad upper young-tussock radius-growth bound"),
        ("mature_relative_growth_max", args.prior_mature_relative_growth_max, "yr^-1", "late live-tiller trajectory should be nearly flat"),
        ("mature_radius_growth_abs_max", args.prior_mature_radius_growth_abs_max, "cm yr^-1", "late radius should be nearly flat"),
        ("first_birth_min_year", args.prior_first_birth_min_year, "years", "weak lower timing bound"),
        ("first_birth_max_year", args.prior_first_birth_max_year, "years", "observed daughter production peaks near tiller age four; broad bound used"),
        ("first_death_max_year", args.prior_first_death_max_year, "years", "individual tillers are relatively short lived"),
        ("turnover_ratio_min", args.prior_turnover_ratio_min, "deaths/birth", "persistent tussocks require mortality and replacement"),
        ("turnover_ratio_max", args.prior_turnover_ratio_max, "deaths/birth", "broad upper turnover bound"),
        ("min_active_event_year_fraction", args.prior_min_active_event_year_fraction, "fraction", "births and deaths should recur rather than occur once"),
        ("max_single_year_event_fraction", args.prior_max_single_year_event_fraction, "fraction", "limits one-year demographic pulses"),
        ("plausibility_score_weight", args.weight_plausibility, "score weight", "contribution to total set score"),
        ("max_plausibility_loss", args.max_plausibility_loss, "loss", "default soft biological gate"),
    ]
    frame = pd.DataFrame(rows, columns=["prior", "value", "units", "interpretation"])
    frame["reference_note"] = BIOLOGICAL_PRIOR_REFERENCE
    return frame


def aggregate_metric_statistics(row: dict[str, float | int | str], prefix: str, values: Iterable[float]) -> None:
    arr = finite_array(values)
    row[f"{prefix}_mean"] = float(np.mean(arr)) if arr.size else np.nan
    row[f"{prefix}_sd"] = float(np.std(arr, ddof=1)) if arr.size > 1 else (0.0 if arr.size else np.nan)
    row[f"{prefix}_median"] = float(np.median(arr)) if arr.size else np.nan
    row[f"{prefix}_q10"] = q(arr, 0.10)
    row[f"{prefix}_q90"] = q(arr, 0.90)
    row[f"{prefix}_min"] = float(np.min(arr)) if arr.size else np.nan
    row[f"{prefix}_max"] = float(np.max(arr)) if arr.size else np.nan


def build_set_metrics(
    final_sim: pd.DataFrame,
    yearly_set: pd.DataFrame,
    result_files: Sequence[Path],
    h_dir: Path,
    args: argparse.Namespace,
    trajectory_targets: pd.DataFrame,
) -> pd.DataFrame:
    if final_sim.empty:
        return pd.DataFrame()

    file_by_set = {
        metadata_from_result_file(h_dir, path)["set_key"]: path
        for path in result_files
    }
    final_groups = [
        (str(set_key), group)
        for set_key, group in final_sim.groupby("set_key", dropna=False, sort=False)
    ]
    yearly_by_set = (
        {
            str(set_key): group
            for set_key, group in yearly_set.groupby("set_key", dropna=False, sort=False)
        }
        if not yearly_set.empty
        else {}
    )

    training_jobs: list[tuple[str, Path, str]] = []
    for set_key, group in final_groups:
        first = group.iloc[0]
        training_jobs.append(
            (set_key, Path(str(first["set_dir"])), str(first["ecotype"]))
        )

    def load_training_job(job: tuple[str, Path, str]) -> tuple[str, np.ndarray]:
        set_key, set_dir, ecotype = job
        return set_key, read_training_diameters(set_dir, ecotype)

    loaded_training = parallel_map_ordered(
        load_training_job,
        training_jobs,
        args.workers,
        "training diameter files",
        args.progress_every,
    )
    training_by_set = dict(loaded_training)

    rows: list[dict[str, float | int | bool | str]] = []
    total_sets = len(final_groups)
    for set_index, (set_key, group) in enumerate(final_groups, start=1):
        first = group.iloc[0]
        n_sim = int(len(group))
        fate_counts = group["final_fate"].value_counts()
        n_alive = int(fate_counts.get("alive", 0))
        n_extinct = int(fate_counts.get("extinct", 0))
        n_overflow = int(fate_counts.get("overflow", 0))

        result_file = file_by_set.get(set_key)
        set_dir = Path(str(first["set_dir"]))
        training_diameters = training_by_set.get(
            set_key,
            np.array([], dtype=float),
        )
        eligible_diameters = finite_array(
            group.loc[group["final_fate"] == "alive", "final_diameter"]
        )
        diameter_w1 = wasserstein_distance_1d(training_diameters, eligible_diameters)
        diameter_scale = robust_scale(training_diameters)
        diameter_w1_normalized = diameter_w1 / diameter_scale if np.isfinite(diameter_w1) else np.nan

        row: dict[str, float | int | bool | str] = {
            "question": str(first["question"]),
            "ecotype": str(first["ecotype"]),
            "set_id": str(first["set_id"]),
            "set_key": set_key,
            "set_dir": str(set_dir),
            "final_population_result_file": str(result_file) if result_file else "",
            "n_sims": n_sim,
            "n_alive": n_alive,
            "n_extinct": n_extinct,
            "n_overflow": n_overflow,
            "prop_alive": n_alive / n_sim if n_sim else np.nan,
            "prop_extinct": n_extinct / n_sim if n_sim else np.nan,
            "prop_overflow": n_overflow / n_sim if n_sim else np.nan,
            "n_training_diameters": int(training_diameters.size),
            "n_alive_diameters": int(eligible_diameters.size),
            "diameter_wasserstein_1d": diameter_w1,
            "diameter_training_scale": diameter_scale,
            "diameter_wasserstein_normalized": diameter_w1_normalized,
        }

        for metric in FINAL_DISTRIBUTION_METRICS:
            aggregate_metric_statistics(row, metric, group[metric])

        set_trajectory = yearly_by_set.get(set_key, pd.DataFrame())
        trajectory_loss, n_target_metrics = trajectory_loss_for_set(set_trajectory, trajectory_targets)
        row["trajectory_loss"] = trajectory_loss
        row["n_trajectory_target_metrics"] = n_target_metrics

        plausibility = biological_plausibility_for_set(
            set_trajectory,
            group,
            args,
        )
        row.update(plausibility)

        passes_gate = True
        if args.min_alive is not None:
            passes_gate &= bool(row["prop_alive"] >= args.min_alive)
        if args.max_extinct is not None:
            passes_gate &= bool(row["prop_extinct"] <= args.max_extinct)
        if args.max_overflow is not None:
            passes_gate &= bool(row["prop_overflow"] <= args.max_overflow)
        if (
            not args.disable_plausibility_score
            and args.max_plausibility_loss >= 0
            and np.isfinite(float(row["biological_plausibility_loss"]))
        ):
            passes_gate &= bool(
                float(row["biological_plausibility_loss"])
                <= args.max_plausibility_loss
            )
        row["passes_biological_gate"] = passes_gate

        diameter_component = (
            float(diameter_w1_normalized)
            if np.isfinite(diameter_w1_normalized)
            else float(args.missing_diameter_penalty)
        )
        trajectory_component = float(trajectory_loss) if np.isfinite(trajectory_loss) else 0.0
        plausibility_component = (
            0.0
            if args.disable_plausibility_score
            else float(row["biological_plausibility_loss"])
        )
        row["score_diameter_component"] = args.weight_diameter * diameter_component
        row["score_extinction_component"] = args.weight_extinct * float(row["prop_extinct"])
        row["score_overflow_component"] = args.weight_overflow * float(row["prop_overflow"])
        row["score_trajectory_component"] = args.weight_trajectory * trajectory_component
        row["score_plausibility_component"] = args.weight_plausibility * plausibility_component
        row["set_total_score"] = (
            row["score_diameter_component"]
            + row["score_extinction_component"]
            + row["score_overflow_component"]
            + row["score_trajectory_component"]
            + row["score_plausibility_component"]
        )
        rows.append(row)

        if set_index % max(1, args.progress_every) == 0 or set_index == total_sets:
            LOGGER.info("parameter-set metrics: %d/%d complete", set_index, total_sets)

    return pd.DataFrame(rows).sort_values(["question", "ecotype", "set_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Question-level summaries, bootstrap CIs, and Pareto front
# ---------------------------------------------------------------------------


def hierarchical_fate_bootstrap(
    subset: pd.DataFrame,
    reps: int,
    seed: int,
) -> dict[str, float]:
    if subset.empty or reps <= 0:
        return {}

    fate_order = ("alive", "extinct", "overflow")
    set_sizes: list[int] = []
    set_probabilities: list[np.ndarray] = []
    for _, group in subset.groupby("set_key", dropna=False, sort=False):
        counts = (
            group["final_fate"]
            .value_counts()
            .reindex(fate_order, fill_value=0)
            .to_numpy(dtype=int)
        )
        n = int(counts.sum())
        if n > 0:
            set_sizes.append(n)
            set_probabilities.append(counts / n)

    n_sets = len(set_sizes)
    if n_sets == 0:
        return {}

    rng = np.random.default_rng(seed)
    set_sizes_array = np.asarray(set_sizes, dtype=int)
    set_probabilities_array = np.asarray(set_probabilities, dtype=float)
    bootstrap_values = np.empty((reps, len(fate_order)), dtype=float)
    set_selection_probability = np.full(n_sets, 1.0 / n_sets, dtype=float)

    for replicate in range(reps):
        selected_copies = rng.multinomial(n_sets, set_selection_probability)
        total_counts = np.zeros(len(fate_order), dtype=np.int64)
        total_samples = 0
        for set_index, copies in enumerate(selected_copies):
            if copies == 0:
                continue
            draws = int(copies * set_sizes_array[set_index])
            total_counts += rng.multinomial(
                draws,
                set_probabilities_array[set_index],
            )
            total_samples += draws
        bootstrap_values[replicate] = total_counts / total_samples

    output: dict[str, float] = {}
    for fate_index, fate in enumerate(fate_order):
        output[f"prop_{fate}_bootstrap_low"] = float(
            np.quantile(bootstrap_values[:, fate_index], 0.025)
        )
        output[f"prop_{fate}_bootstrap_high"] = float(
            np.quantile(bootstrap_values[:, fate_index], 0.975)
        )
    return output


def summarize_questions(
    set_metrics: pd.DataFrame,
    final_sim: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    if set_metrics.empty:
        return pd.DataFrame()

    rows: list[dict[str, float | int | bool | str]] = []
    final_by_question = {
        str(question): group
        for question, group in final_sim.groupby("question", dropna=False, sort=False)
    }
    question_groups = list(set_metrics.groupby("question", dropna=False, sort=False))
    for question_index, (question, group) in enumerate(question_groups):
        sim_group = final_by_question.get(str(question), pd.DataFrame())
        fate_counts = sim_group["final_fate"].value_counts()
        n_sims = int(len(sim_group))

        scores = finite_array(group["set_total_score"])
        accepted = group["passes_biological_gate"].astype(bool)
        row: dict[str, float | int | bool | str] = {
            "question": str(question),
            "n_sets": int(group["set_key"].nunique()),
            "n_ecotypes": int(group["ecotype"].nunique()),
            "ecotypes": ",".join(sorted(group["ecotype"].dropna().astype(str).unique())),
            "n_sims_pooled": n_sims,
            "n_alive_pooled": int(fate_counts.get("alive", 0)),
            "n_extinct_pooled": int(fate_counts.get("extinct", 0)),
            "n_overflow_pooled": int(fate_counts.get("overflow", 0)),
            "prop_alive_pooled": float((sim_group["final_fate"] == "alive").mean()) if n_sims else np.nan,
            "prop_extinct_pooled": float((sim_group["final_fate"] == "extinct").mean()) if n_sims else np.nan,
            "prop_overflow_pooled": float((sim_group["final_fate"] == "overflow").mean()) if n_sims else np.nan,
            "accepted_sets": int(accepted.sum()),
            "accepted_set_fraction": float(accepted.mean()) if len(accepted) else np.nan,
            "set_score_median": float(np.median(scores)) if scores.size else np.nan,
            "set_score_q10": q(scores, 0.10),
            "set_score_q90": q(scores, 0.90),
        }
        row["robust_question_score"] = (
            row["set_score_median"] + args.q90_weight * row["set_score_q90"]
            if np.isfinite(row["set_score_median"]) and np.isfinite(row["set_score_q90"])
            else np.nan
        )

        summary_metrics = [
            "prop_alive",
            "prop_extinct",
            "prop_overflow",
            "diameter_wasserstein_1d",
            "diameter_wasserstein_normalized",
            "trajectory_loss",
            "biological_plausibility_loss",
            *BIOLOGICAL_PLAUSIBILITY_COMPONENTS,
            "cumulative_attempted_daughters_median",
            "cumulative_established_births_median",
            "cumulative_deaths_median",
            "cumulative_tillers_created_median",
            "establishment_success_rate_median",
            "turnover_ratio_median",
            "mean_survival_probability_median",
            "mean_reproduction_probability_median",
            "mean_establishment_probability_median",
            "realized_survival_rate_overall_median",
            "realized_reproduction_rate_overall_median",
            "realized_establishment_rate_overall_median",
        ]
        for metric in summary_metrics:
            if metric in group.columns:
                aggregate_metric_statistics(row, metric, group[metric])

        row.update(
            hierarchical_fate_bootstrap(
                sim_group,
                args.bootstrap_reps,
                args.bootstrap_seed + question_index * 1009,
            )
        )
        rows.append(row)
        LOGGER.info(
            "question summaries: %d/%d complete (%s)",
            question_index + 1,
            len(question_groups),
            question,
        )

    output = pd.DataFrame(rows)
    output = add_pareto_flags(output)
    gate_configured = (
        any(
            threshold is not None
            for threshold in (args.min_alive, args.max_extinct, args.max_overflow)
        )
        or (not args.disable_plausibility_score and args.max_plausibility_loss >= 0)
    )
    output["gate_configured"] = gate_configured
    output = output.sort_values(
        ["pareto_optimal", "accepted_set_fraction", "robust_question_score"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    output["rank"] = np.arange(1, len(output) + 1)
    return output


def add_pareto_flags(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    output = summary.copy()
    objectives = [
        "prop_extinct_pooled",
        "prop_overflow_pooled",
        "diameter_wasserstein_normalized_median",
    ]
    if (
        "biological_plausibility_loss_median" in output.columns
        and np.isfinite(numeric(output["biological_plausibility_loss_median"])).any()
    ):
        objectives.append("biological_plausibility_loss_median")
    if "trajectory_loss_median" in output.columns and np.isfinite(numeric(output["trajectory_loss_median"])).any():
        objectives.append("trajectory_loss_median")

    matrix = np.array(
        output[objectives].apply(numeric).to_numpy(dtype=float),
        dtype=float,
        copy=True,
        order="C",
    )
    # Missing objectives are treated as worse than any finite result.
    for column_index in range(matrix.shape[1]):
        finite = matrix[np.isfinite(matrix[:, column_index]), column_index]
        replacement = (float(np.max(finite)) + robust_scale(finite)) if finite.size else 1e9
        matrix[~np.isfinite(matrix[:, column_index]), column_index] = replacement

    pareto = np.ones(len(output), dtype=bool)
    for i in range(len(output)):
        for j in range(len(output)):
            if i == j:
                continue
            no_worse = np.all(matrix[j] <= matrix[i])
            strictly_better = np.any(matrix[j] < matrix[i])
            if no_worse and strictly_better:
                pareto[i] = False
                break
    output["pareto_optimal"] = pareto
    output["pareto_objectives"] = ",".join(objectives)
    return output


def summarize_by_question_ecotype(set_metrics: pd.DataFrame) -> pd.DataFrame:
    if set_metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, float | int | str]] = []
    for (question, ecotype), group in set_metrics.groupby(["question", "ecotype"], dropna=False):
        row: dict[str, float | int | str] = {
            "question": str(question),
            "ecotype": str(ecotype),
            "n_sets": int(group["set_key"].nunique()),
            "accepted_set_fraction": float(group["passes_biological_gate"].astype(bool).mean()),
        }
        for metric in [
            "prop_alive",
            "prop_extinct",
            "prop_overflow",
            "diameter_wasserstein_normalized",
            "trajectory_loss",
            "biological_plausibility_loss",
            "set_total_score",
        ]:
            aggregate_metric_statistics(row, metric, group[metric])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["question", "ecotype"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_population_composition(
    question_summary: pd.DataFrame,
    path: Path,
    title: str,
    dpi: int,
) -> None:
    if question_summary.empty:
        save_empty_plot(path, "Final population composition", "No final simulation summaries found.", dpi)
        return
    frame = question_summary.sort_values("rank").copy()
    labels = frame["question"].astype(str).tolist()
    x = np.arange(len(frame))
    alive = numeric(frame["prop_alive_pooled"]).fillna(0).to_numpy()
    extinct = numeric(frame["prop_extinct_pooled"]).fillna(0).to_numpy()
    overflow = numeric(frame["prop_overflow_pooled"]).fillna(0).to_numpy()

    fig, ax = plt.subplots(figsize=(max(10, 0.75 * len(labels)), 6))
    ax.bar(x, alive, label="Alive")
    ax.bar(x, extinct, bottom=alive, label="Extinct")
    ax.bar(x, overflow, bottom=alive + extinct, label="Overflow")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Pooled final proportion")
    ax.set_xlabel("Question / hypothesis")
    ax.set_title(f"{title}: mutually exclusive final fates")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_ranked_scores(question_summary: pd.DataFrame, path: Path, title: str, dpi: int) -> None:
    if question_summary.empty:
        save_empty_plot(path, "Robust model score", "No question scores found.", dpi)
        return
    frame = question_summary.sort_values("robust_question_score").copy()
    frame = frame[np.isfinite(numeric(frame["robust_question_score"]))]
    if frame.empty:
        save_empty_plot(path, "Robust model score", "No finite scores found.", dpi)
        return
    labels = frame["question"].astype(str).tolist()
    values = numeric(frame["robust_question_score"]).to_numpy()
    fig, ax = plt.subplots(figsize=(max(10, 0.75 * len(labels)), 6))
    bars = ax.bar(np.arange(len(frame)), values)
    for bar, pareto in zip(bars, frame["pareto_optimal"].astype(bool)):
        if pareto:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                "Pareto",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_ylabel("Robust score (lower is better)")
    ax.set_xlabel("Question / hypothesis")
    ax.set_title(f"{title}: robust set-distribution score")
    ax.set_xticks(np.arange(len(frame)))
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_pareto(question_summary: pd.DataFrame, path: Path, title: str, dpi: int) -> None:
    if question_summary.empty:
        save_empty_plot(path, "Pareto comparison", "No question summary found.", dpi)
        return
    x = numeric(question_summary["prop_extinct_pooled"])
    y = numeric(question_summary["diameter_wasserstein_normalized_median"])
    overflow = numeric(question_summary["prop_overflow_pooled"]).fillna(0)
    keep = np.isfinite(x) & np.isfinite(y)
    if not keep.any():
        save_empty_plot(path, "Pareto comparison", "No finite Pareto plotting metrics found.", dpi)
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    sizes = 45 + 500 * overflow[keep].to_numpy(dtype=float)
    ax.scatter(x[keep], y[keep], s=sizes, alpha=0.8)
    for xi, yi, label, pareto in zip(
        x[keep],
        y[keep],
        question_summary.loc[keep, "question"].astype(str),
        question_summary.loc[keep, "pareto_optimal"].astype(bool),
    ):
        ax.annotate(
            f"{label}{' *' if pareto else ''}",
            (xi, yi),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Pooled extinction proportion (lower is better)")
    ax.set_ylabel("Median normalized diameter Wasserstein (lower is better)")
    ax.set_title(f"{title}: fate versus survivor diameter fit\nmarker size reflects overflow")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_set_metric_boxplots(
    set_metrics: pd.DataFrame,
    metric: str,
    ylabel: str,
    path: Path,
    title: str,
    dpi: int,
) -> None:
    if set_metrics.empty or metric not in set_metrics.columns:
        save_empty_plot(path, ylabel, f"No {metric} data found.", dpi)
        return
    questions = sorted(set_metrics["question"].dropna().astype(str).unique())
    data = [finite_array(set_metrics.loc[set_metrics["question"].astype(str) == question, metric]) for question in questions]
    keep = [(question, values) for question, values in zip(questions, data) if values.size]
    if not keep:
        save_empty_plot(path, ylabel, f"No finite {metric} data found.", dpi)
        return
    labels = [item[0] for item in keep]
    arrays = [item[1] for item in keep]
    fig, ax = plt.subplots(figsize=(max(10, 0.75 * len(labels)), 6))
    ax.boxplot(arrays, showfliers=False)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=60, ha="right")
    rng = np.random.default_rng(123)
    for index, values in enumerate(arrays, start=1):
        jitter = rng.uniform(-0.08, 0.08, size=len(values))
        ax.scatter(index + jitter, values, s=20, alpha=0.7)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Question / hypothesis")
    ax.set_title(f"{title}: set-level {ylabel.lower()}")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_plausibility_components(
    set_metrics: pd.DataFrame,
    path: Path,
    title: str,
    dpi: int,
) -> None:
    available = [
        component
        for component in BIOLOGICAL_PLAUSIBILITY_COMPONENTS
        if component in set_metrics.columns
    ]
    if set_metrics.empty or not available:
        save_empty_plot(path, "Biological plausibility components", "No plausibility components found.", dpi)
        return
    questions = sorted(set_metrics["question"].dropna().astype(str).unique())
    matrix = np.full((len(questions), len(available)), np.nan, dtype=float)
    for row_index, question in enumerate(questions):
        subset = set_metrics[set_metrics["question"].astype(str) == question]
        for column_index, component in enumerate(available):
            values = finite_array(subset[component])
            matrix[row_index, column_index] = float(np.median(values)) if values.size else np.nan
    fig, ax = plt.subplots(figsize=(max(11, 1.15 * len(available)), max(5, 0.55 * len(questions))))
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest")
    ax.set_yticks(np.arange(len(questions)))
    ax.set_yticklabels(questions)
    labels = [
        component.removeprefix("prior_").removesuffix("_penalty").replace("_", " ")
        for component in available
    ]
    ax.set_xticks(np.arange(len(available)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            if np.isfinite(value):
                ax.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title(f"{title}: median biological-prior penalty by hypothesis")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Penalty; lower is better")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_alive_with_prior_envelope(
    yearly_question: pd.DataFrame,
    path: Path,
    title: str,
    dpi: int,
    args: argparse.Namespace,
) -> None:
    metric = "n_alive_median_across_sets"
    if yearly_question.empty or metric not in yearly_question.columns:
        save_empty_plot(path, "Live tillers and rough prior", "No yearly live-tiller data found.", dpi)
        return
    all_times = finite_array(yearly_question["time_step"])
    if all_times.size < 2:
        save_empty_plot(path, "Live tillers and rough prior", "Too few time points.", dpi)
        return
    x_prior = np.linspace(float(np.min(all_times)), float(np.max(all_times)), 300)
    center, lower, upper = rough_alive_prior_bounds(x_prior, 1.0, args)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(x_prior, lower, upper, alpha=0.15, label="Broad biological-prior envelope")
    ax.plot(x_prior, center, linestyle="--", linewidth=1.5, label="Rough prior center")
    for question, group in yearly_question.groupby("question", dropna=False):
        group = group.sort_values("time_step")
        x = numeric(group["time_step"]).to_numpy(dtype=float)
        y = numeric(group[metric]).to_numpy(dtype=float)
        keep = np.isfinite(x) & np.isfinite(y)
        if keep.any():
            ax.plot(x[keep], y[keep], linewidth=1.5, label=str(question))
    ax.set_xlabel("Simulation year / time step")
    ax.set_ylabel("Living tillers")
    ax.set_title(f"{title}: live-tiller trajectories versus weak growth prior")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_optimizer_metric(
    optimizer_summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    path: Path,
    title: str,
    dpi: int,
) -> None:
    mean_column = f"{metric}_mean"
    if optimizer_summary.empty or mean_column not in optimizer_summary.columns:
        save_empty_plot(path, ylabel, f"No optimizer-history data for {metric}.", dpi)
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for question, group in optimizer_summary.groupby("question", dropna=False):
        group = group.sort_values("iteration")
        x = numeric(group["iteration"])
        y = numeric(group[mean_column])
        keep = np.isfinite(x) & np.isfinite(y)
        if keep.any():
            ax.plot(x[keep], y[keep], label=str(question), linewidth=1.5)
    ax.set_xlabel("Optimization/resampling iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}: optimizer history — {ylabel}")
    ax.grid(True, alpha=0.25)
    if optimizer_summary["question"].nunique() <= 20:
        ax.legend(fontsize=8)
    else:
        ax.legend(fontsize=6, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_yearly_metric(
    yearly_question: pd.DataFrame,
    metric: str,
    ylabel: str,
    path: Path,
    title: str,
    dpi: int,
) -> None:
    median_column = f"{metric}_median_across_sets"
    low_column = f"{metric}_q10_across_sets"
    high_column = f"{metric}_q90_across_sets"
    if yearly_question.empty or median_column not in yearly_question.columns:
        save_empty_plot(path, ylabel, f"No biological yearly data for {metric}.", dpi)
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    any_line = False
    for question, group in yearly_question.groupby("question", dropna=False):
        group = group.sort_values("time_step")
        x = numeric(group["time_step"]).to_numpy(dtype=float)
        median = numeric(group[median_column]).to_numpy(dtype=float)
        low = numeric(group[low_column]).to_numpy(dtype=float)
        high = numeric(group[high_column]).to_numpy(dtype=float)
        keep = np.isfinite(x) & np.isfinite(median)
        if not keep.any():
            continue
        line = ax.plot(x[keep], median[keep], linewidth=1.5, label=str(question))[0]
        band_keep = keep & np.isfinite(low) & np.isfinite(high)
        if band_keep.any():
            ax.fill_between(
                x[band_keep],
                low[band_keep],
                high[band_keep],
                alpha=0.12,
                color=line.get_color(),
            )
        any_line = True

    if not any_line:
        plt.close(fig)
        save_empty_plot(path, ylabel, f"No finite biological yearly data for {metric}.", dpi)
        return
    ax.set_xlabel("Simulation year / time step")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}: biological trajectory — {ylabel}\nline = median set; band = 10th–90th percentile across sets")
    ax.grid(True, alpha=0.25)
    if yearly_question["question"].nunique() <= 20:
        ax.legend(fontsize=8)
    else:
        ax.legend(fontsize=6, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def make_plots(
    question_summary: pd.DataFrame,
    set_metrics: pd.DataFrame,
    optimizer_summary: pd.DataFrame,
    yearly_question: pd.DataFrame,
    out_dir: Path,
    title: str,
    dpi: int,
    args: argparse.Namespace,
) -> None:
    plot_dir = out_dir / "plots"
    optimizer_dir = plot_dir / "optimizer_history"
    yearly_dir = plot_dir / "biological_trajectories"
    plot_dir.mkdir(parents=True, exist_ok=True)
    optimizer_dir.mkdir(parents=True, exist_ok=True)
    yearly_dir.mkdir(parents=True, exist_ok=True)

    plot_population_composition(question_summary, plot_dir / "population_composition.png", title, dpi)
    plot_ranked_scores(question_summary, plot_dir / "robust_question_scores.png", title, dpi)
    plot_pareto(question_summary, plot_dir / "pareto_fate_vs_diameter.png", title, dpi)
    plot_set_metric_boxplots(
        set_metrics,
        "diameter_wasserstein_normalized",
        "Normalized diameter Wasserstein distance",
        plot_dir / "set_distribution_diameter_wasserstein.png",
        title,
        dpi,
    )
    plot_set_metric_boxplots(
        set_metrics,
        "biological_plausibility_loss",
        "Biological plausibility loss",
        plot_dir / "set_distribution_biological_plausibility.png",
        title,
        dpi,
    )
    plot_plausibility_components(
        set_metrics,
        plot_dir / "biological_plausibility_components.png",
        title,
        dpi,
    )
    plot_alive_with_prior_envelope(
        yearly_question,
        plot_dir / "trajectory_n_alive_with_prior_envelope.png",
        title,
        dpi,
        args,
    )
    plot_set_metric_boxplots(
        set_metrics,
        "set_total_score",
        "Total set score",
        plot_dir / "set_distribution_total_score.png",
        title,
        dpi,
    )
    plot_set_metric_boxplots(
        set_metrics,
        "cumulative_established_births_median",
        "Median cumulative established births",
        plot_dir / "set_distribution_cumulative_births.png",
        title,
        dpi,
    )
    plot_set_metric_boxplots(
        set_metrics,
        "cumulative_deaths_median",
        "Median cumulative deaths",
        plot_dir / "set_distribution_cumulative_deaths.png",
        title,
        dpi,
    )

    optimizer_plot_specs = {
        "prop_alive": "Mean final proportion alive",
        "prop_extinct": "Mean final proportion extinct",
        "prop_overflow": "Mean final proportion overflow",
        "avg_tussock_diameter": "Mean final average tussock diameter",
    }
    for metric, ylabel in optimizer_plot_specs.items():
        plot_optimizer_metric(
            optimizer_summary,
            metric,
            ylabel,
            optimizer_dir / f"optimizer_history_{metric}.png",
            title,
            dpi,
        )

    yearly_plot_specs = {
        "n_alive": "Living tillers",
        "n_dead": "Currently retained dead tillers",
        "diameter": "Tussock diameter",
        "attempted_daughters": "Attempted daughters per year",
        "established_daughters": "Established daughters per year",
        "deaths_this_step": "Deaths per year",
        "cumulative_attempted_daughters": "Cumulative attempted daughters",
        "cumulative_established_births": "Cumulative established births",
        "cumulative_deaths": "Cumulative deaths",
        "cumulative_tillers_created": "Cumulative actual tillers created",
        "mean_survival_probability": "Mean survival probability",
        "mean_reproduction_probability": "Mean reproduction probability",
        "mean_establishment_probability": "Mean establishment probability",
        "realized_survival_rate": "Realized survival rate",
        "realized_reproduction_rate": "Realized reproduction rate",
        "realized_establishment_rate": "Realized establishment rate",
    }
    for metric, ylabel in yearly_plot_specs.items():
        plot_yearly_metric(
            yearly_question,
            metric,
            ylabel,
            yearly_dir / f"trajectory_{metric}.png",
            title,
            dpi,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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
        suffix = ecotype if ecotype else "all_ecotypes"
        out_dir = h_dir / f"higher_level_comparison_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    args.workers = resolve_workers(args.workers)
    args.progress_every = max(1, args.progress_every)
    if args.prior_adult_alive_min <= 0:
        raise ValueError("--prior-adult-alive-min must be > 0")
    if args.prior_adult_alive_max <= args.prior_adult_alive_min:
        raise ValueError("--prior-adult-alive-max must exceed --prior-adult-alive-min")
    if args.prior_alive_envelope_factor <= 1:
        raise ValueError("--prior-alive-envelope-factor must be > 1")
    if args.prior_mature_years <= 0:
        raise ValueError("--prior-mature-years must be > 0")
    log_file = (
        Path(args.log_file).resolve()
        if args.log_file
        else out_dir / "higher_level_comparison.log"
    )
    setup_logging(log_file, args.log_level)

    title = args.title if args.title else h_dir.name
    title = f"{title} ({ecotype})" if ecotype else f"{title} (all ecotypes combined)"

    run_start = time.perf_counter()
    LOGGER.info("h-dir: %s", h_dir)
    LOGGER.info("ecotype filter: %s", ecotype if ecotype else "[all ecotypes combined]")
    LOGGER.info("output directory: %s", out_dir)
    LOGGER.info("log file: %s", log_file)
    LOGGER.info("worker threads: %d", args.workers)
    LOGGER.info("bootstrap replicates: %d", args.bootstrap_reps)
    LOGGER.info(
        "biological plausibility score: %s (weight %.3f; gate %.3f)",
        "disabled" if args.disable_plausibility_score else "enabled",
        args.weight_plausibility,
        args.max_plausibility_loss,
    )
    LOGGER.info("biological prior: %s", BIOLOGICAL_PRIOR_REFERENCE)
    LOGGER.info("all CSV and plot writes are serialized on the main thread")

    with logged_stage("discover and load optimizer histories"):
        optimizer_history, result_files = load_all_optimizer_history(
            h_dir,
            ecotype,
            args.include_nonstandard_files,
            args.workers,
            args.progress_every,
        )
        if not result_files:
            raise RuntimeError(
                "No final_population_results.csv files found. Expected layout:\n"
                f"  {h_dir}/<question>/resampled_fits/<ecotype>/set_001/final_population_results.csv"
            )
        LOGGER.info("discovered %d optimizer-history file(s)", len(result_files))

    with logged_stage("summarize optimizer histories"):
        last_optimizer_rows = get_last_optimizer_rows(optimizer_history)
        optimizer_summary = summarize_optimizer_history(optimizer_history)

    unique_result_file_by_set: dict[str, Path] = {}
    for result_file in result_files:
        set_key = metadata_from_result_file(h_dir, result_file)["set_key"]
        unique_result_file_by_set.setdefault(set_key, result_file)
    unique_result_files = list(unique_result_file_by_set.values())
    LOGGER.info("unique parameter sets to inspect: %d", len(unique_result_files))

    with logged_stage("load final stochastic simulation summaries"):
        final_sim = load_all_final_sim_summaries(
            h_dir,
            unique_result_files,
            args.workers,
            args.progress_every,
        )
        if final_sim.empty:
            raise RuntimeError(
                "No final simulation summary files found under "
                "<set>/final_sims/summaries/summary_*.csv"
            )
        LOGGER.info("loaded %d final simulation row(s)", len(final_sim))

    with logged_stage("load yearly biological summaries"):
        yearly_sim = load_all_yearly_summaries(
            h_dir,
            unique_result_files,
            args.workers,
            args.progress_every,
        )
        LOGGER.info("loaded %d yearly simulation row(s)", len(yearly_sim))

    with logged_stage("aggregate yearly trajectories"):
        yearly_set = summarize_yearly_by_set(yearly_sim)
        yearly_question = summarize_yearly_by_question(yearly_set)
        trajectory_targets = read_trajectory_targets(args.trajectory_targets_csv)
        prior_configuration = biological_prior_configuration(args)

    with logged_stage("calculate parameter-set metrics and scores"):
        set_metrics = build_set_metrics(
            final_sim,
            yearly_set,
            unique_result_files,
            h_dir,
            args,
            trajectory_targets,
        )

    with logged_stage("calculate question summaries and bootstrap intervals"):
        question_summary = summarize_questions(set_metrics, final_sim, args)
        question_ecotype_summary = summarize_by_question_ecotype(set_metrics)

    with logged_stage("write CSV outputs"):
        standard_outputs = [
            (optimizer_history, out_dir / "optimizer_history_all_rows.csv"),
            (last_optimizer_rows, out_dir / "optimizer_history_final_rows_by_set.csv"),
            (optimizer_summary, out_dir / "optimizer_history_question_summary.csv"),
            (set_metrics, out_dir / "parameter_set_metrics_and_scores.csv"),
            (question_summary, out_dir / "question_model_selection_summary.csv"),
            (question_ecotype_summary, out_dir / "question_summary_by_ecotype.csv"),
            (yearly_set, out_dir / "yearly_summary_by_parameter_set.csv"),
            (yearly_question, out_dir / "yearly_summary_by_question.csv"),
            (prior_configuration, out_dir / "biological_plausibility_prior_configuration.csv"),
        ]
        for frame, output_path in standard_outputs:
            write_csv_atomic(frame, output_path)

        if args.skip_large_compiled_csvs:
            LOGGER.info("Skipping large row-level compiled CSVs by request")
        else:
            write_csv_atomic(
                final_sim,
                out_dir / "final_simulation_summaries_compiled.csv",
            )
            write_csv_atomic(
                yearly_sim,
                out_dir / "yearly_simulation_summaries_compiled.csv",
            )

    if args.skip_plots:
        LOGGER.info("Skipping plots by request")
    else:
        with logged_stage("generate plots"):
            make_plots(
                question_summary,
                set_metrics,
                optimizer_summary,
                yearly_question,
                out_dir,
                title,
                args.plot_dpi,
                args,
            )

    elapsed = time.perf_counter() - run_start
    LOGGER.info("========================================")
    LOGGER.info("Deprecated overgrown class: ignored")
    LOGGER.info("questions: %d", set_metrics["question"].nunique())
    LOGGER.info("parameter sets: %d", set_metrics["set_key"].nunique())
    LOGGER.info("final stochastic simulations: %d", len(final_sim))
    LOGGER.info("yearly simulation rows: %d", len(yearly_sim))
    LOGGER.info("total runtime: %.2f s", elapsed)
    LOGGER.info("Primary model-selection output: %s", out_dir / "question_model_selection_summary.csv")
    LOGGER.info("Parameter-set scores: %s", out_dir / "parameter_set_metrics_and_scores.csv")
    LOGGER.info("Yearly question summary: %s", out_dir / "yearly_summary_by_question.csv")
    LOGGER.info("Log: %s", log_file)

    if not question_summary.empty:
        columns = [
            "rank",
            "question",
            "pareto_optimal",
            "n_sets",
            "n_sims_pooled",
            "prop_alive_pooled",
            "prop_extinct_pooled",
            "prop_overflow_pooled",
            "diameter_wasserstein_normalized_median",
            "trajectory_loss_median",
            "biological_plausibility_loss_median",
            "accepted_set_fraction",
            "robust_question_score",
        ]
        columns = [column for column in columns if column in question_summary.columns]
        LOGGER.info("Question ranking:\n%s", question_summary[columns].to_string(index=False))


if __name__ == "__main__":
    main()
