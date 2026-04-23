#!/usr/bin/env python3
import argparse
import configparser
import csv
import math
import os
import random
import subprocess
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

try:
    import seaborn as sns
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False


ALL_MODEL_PARAM_NAMES = [
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


@dataclass
class PathSettings:
    output_dir: str
    training_csv: str
    subdir: str


@dataclass
class OptimizationSettings:
    max_evals: int
    n_init: int
    tol_f: float
    tol_x: float
    init_log10_span: float
    step_frac: float
    step_abs: float
    optimize_log_space: bool
    cma_sigma: float
    cma_popsize: int
    cma_patience: int


@dataclass
class ConstraintSettings:
    extinction_weight: float
    constraint_year: int
    min_alive_tillers: int
    constraint_pass_frac: float
    alive_overflow_threshold: int
    overgrown_radius_threshold: float
    hard_fail_on_overflow: bool
    require_survive_to_end_for_fit: bool


@dataclass
class PlotSettings:
    plot_every: int
    plot_kde: bool
    print_fail_breakdown: bool


@dataclass
class MechanismSettings:
    use_spatial_survival: bool
    use_spatial_reproduction: bool
    use_spatial_establishment: bool
    use_crowding_survival: bool
    use_crowding_reproduction: bool
    use_crowding_establishment: bool
    crowding_radius_cm: float


@dataclass
class ResamplingSettings:
    n_sets: int
    train_percent: float
    sample_with_replacement: bool
    random_seed: int | None


@dataclass
class RunSettings:
    sites: list | None
    config_path: Path
    config_dir: Path
    project_root: Path
    paths: PathSettings
    optimization: OptimizationSettings
    constraints: ConstraintSettings
    plotting: PlotSettings
    mechanisms: MechanismSettings
    resampling: ResamplingSettings
    active_params: list[str]


def parse_args():
    parser = argparse.ArgumentParser(description="Tussock model parameterization with repeated random training subsets")
    parser.add_argument("--config", type=str, required=True, help="Path to the combined ini config file")
    parser.add_argument("--sites", nargs="*", default=None)
    return parser.parse_args()


def read_bool(config, section, key, fallback=False):
    return config.getboolean(section, key, fallback=fallback)


def positive_param(name: str) -> bool:
    return name in {
        "ks", "kr", "ke",
        "k_crowd_survival", "k_crowd_reproduction", "k_crowd_establishment"
    }


def bounded01_param(name: str) -> bool:
    return name in {"c_space_survival", "c_space_reproduction"}


def resolve_path(base_dir: Path, raw_path: str) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def determine_active_params(config: configparser.ConfigParser) -> list[str]:
    if config.has_section("ParameterActivation"):
        active = []
        for name in ALL_MODEL_PARAM_NAMES:
            key = f"optimize_{name}"
            if config.getboolean("ParameterActivation", key, fallback=False):
                active.append(name)
        if active:
            return active

    mech = config["Mechanisms"] if config.has_section("Mechanisms") else {}
    active = ["leaf_offset"]

    if str(mech.get("use_spatial_survival", "false")).lower() == "true":
        active.extend(["ks", "bs", "c_space_survival"])
    if str(mech.get("use_spatial_reproduction", "false")).lower() == "true":
        active.extend(["kr", "br", "c_space_reproduction"])
    if str(mech.get("use_spatial_establishment", "false")).lower() == "true":
        active.extend(["ke", "be"])
    if str(mech.get("use_crowding_survival", "false")).lower() == "true":
        active.append("k_crowd_survival")
    if str(mech.get("use_crowding_reproduction", "false")).lower() == "true":
        active.append("k_crowd_reproduction")
    if str(mech.get("use_crowding_establishment", "false")).lower() == "true":
        active.append("k_crowd_establishment")

    out = []
    seen = set()
    for x in active:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def load_combined_config(config_path: str, cli_sites=None):
    config_path = Path(config_path).resolve()
    config_dir = config_path.parent
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.resolve()

    config = configparser.ConfigParser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    config.read(config_path)

    if "Tussock Model" not in config:
        raise ValueError(f"Missing [Tussock Model] section in {config_path}")

    output_dir_raw = config.get("Paths", "output_dir", fallback="parameterization_outputs")
    training_csv_raw = config.get("Paths", "training_csv", fallback="./input_data/tussock_density_tussock_diam.csv")
    subdir = config.get("Paths", "subdir", fallback="runs")

    output_dir = resolve_path(project_root, output_dir_raw)
    training_csv = resolve_path(project_root, training_csv_raw)

    paths = PathSettings(
        output_dir=str(output_dir),
        training_csv=str(training_csv),
        subdir=str(subdir),
    )

    optimization = OptimizationSettings(
        max_evals=config.getint("Optimization", "max_evals", fallback=200),
        n_init=config.getint("Optimization", "n_init", fallback=5),
        tol_f=config.getfloat("Optimization", "tol_f", fallback=1e-3),
        tol_x=config.getfloat("Optimization", "tol_x", fallback=1e-3),
        init_log10_span=config.getfloat("Optimization", "init_log10_span", fallback=1.0),
        step_frac=config.getfloat("Optimization", "step_frac", fallback=0.2),
        step_abs=config.getfloat("Optimization", "step_abs", fallback=0.3),
        optimize_log_space=read_bool(config, "Optimization", "optimize_log_space", fallback=False),
        cma_sigma=config.getfloat("Optimization", "cma_sigma", fallback=0.5),
        cma_popsize=config.getint("Optimization", "cma_popsize", fallback=12),
        cma_patience=config.getint("Optimization", "cma_patience", fallback=20),
    )

    constraints = ConstraintSettings(
        extinction_weight=config.getfloat("Constraints", "extinction_weight", fallback=0.0),
        constraint_year=config.getint("Constraints", "constraint_year", fallback=25),
        min_alive_tillers=config.getint("Constraints", "min_alive_tillers", fallback=25),
        constraint_pass_frac=config.getfloat("Constraints", "constraint_pass_frac", fallback=0.8),
        alive_overflow_threshold=config.getint("Constraints", "alive_overflow_threshold", fallback=500),
        overgrown_radius_threshold=config.getfloat("Constraints", "overgrown_radius_threshold", fallback=2.5),
        hard_fail_on_overflow=read_bool(config, "Constraints", "hard_fail_on_overflow", fallback=False),
        require_survive_to_end_for_fit=read_bool(config, "Constraints", "require_survive_to_end_for_fit", fallback=False),
    )

    plotting = PlotSettings(
        plot_every=config.getint("Plotting", "plot_every", fallback=10),
        plot_kde=read_bool(config, "Plotting", "plot_kde", fallback=False),
        print_fail_breakdown=read_bool(config, "Plotting", "print_fail_breakdown", fallback=False),
    )

    mechanisms = MechanismSettings(
        use_spatial_survival=read_bool(config, "Mechanisms", "use_spatial_survival", fallback=False),
        use_spatial_reproduction=read_bool(config, "Mechanisms", "use_spatial_reproduction", fallback=False),
        use_spatial_establishment=read_bool(config, "Mechanisms", "use_spatial_establishment", fallback=False),
        use_crowding_survival=read_bool(config, "Mechanisms", "use_crowding_survival", fallback=False),
        use_crowding_reproduction=read_bool(config, "Mechanisms", "use_crowding_reproduction", fallback=False),
        use_crowding_establishment=read_bool(config, "Mechanisms", "use_crowding_establishment", fallback=False),
        crowding_radius_cm=config.getfloat("Mechanisms", "crowding_radius_cm", fallback=2.0),
    )

    resampling = ResamplingSettings(
        n_sets=config.getint("Resampling", "n_sets", fallback=1),
        train_percent=config.getfloat("Resampling", "train_percent", fallback=100.0),
        sample_with_replacement=read_bool(config, "Resampling", "sample_with_replacement", fallback=False),
        random_seed=(config.getint("Resampling", "random_seed", fallback=-1) if config.has_section("Resampling") else -1),
    )
    if resampling.random_seed == -1:
        resampling.random_seed = None

    active_params = determine_active_params(config)
    sites = cli_sites if cli_sites is not None else None

    run_settings = RunSettings(
        sites=sites,
        config_path=config_path,
        config_dir=config_dir,
        project_root=project_root,
        paths=paths,
        optimization=optimization,
        constraints=constraints,
        plotting=plotting,
        mechanisms=mechanisms,
        resampling=resampling,
        active_params=active_params,
    )
    return config, run_settings


def _safe_makedirs(dirpath: str | Path):
    if dirpath:
        Path(dirpath).mkdir(parents=True, exist_ok=True)


def default_parameter_values() -> OrderedDict:
    return OrderedDict([
        ("ks", 1.0),
        ("kr", 1.0),
        ("ke", 1.0),
        ("bs", 0.0),
        ("br", 0.0),
        ("be", 0.0),
        ("c_space_survival", 0.5),
        ("c_space_reproduction", 0.5),
        ("k_crowd_survival", 0.1),
        ("k_crowd_reproduction", 0.1),
        ("k_crowd_establishment", 0.1),
        ("leaf_offset", 0.0),
    ])


def coerce_model_param(name: str, value: float) -> float:
    if not np.isfinite(value):
        raise ValueError(f"Non-finite value for parameter '{name}': {value}")
    if positive_param(name):
        return float(max(0.0, value))
    if bounded01_param(name):
        return float(min(1.0, max(0.0, value)))
    return float(value)


def read_parameter_file(param_file: str | Path) -> OrderedDict:
    params = default_parameter_values()
    param_file = Path(param_file)
    if not param_file.exists():
        raise FileNotFoundError(f"Model parameter file not found: {param_file}")
    with param_file.open("r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            if k not in params:
                raise ValueError(f"Unexpected key in parameter file '{param_file}': '{k}'. Allowed keys: {list(params.keys())}")
            params[k] = coerce_model_param(k, float(v))
    return params


def random_initial_parameters(active_params: list[str]) -> OrderedDict:
    params = default_parameter_values()

    def logu(lo, hi):
        return 10 ** random.uniform(math.log10(lo), math.log10(hi))

    for name in active_params:
        if name in {"ks", "kr", "ke"}:
            params[name] = logu(1e-3, 100.0)
        elif name in {"k_crowd_survival", "k_crowd_reproduction", "k_crowd_establishment"}:
            params[name] = logu(1e-4, 10.0)
        elif name in {"bs", "br", "be"}:
            params[name] = random.uniform(-3.0, 3.0)
        elif name in {"c_space_survival", "c_space_reproduction"}:
            params[name] = random.uniform(0.0, 1.0)
        elif name == "leaf_offset":
            params[name] = random.uniform(-200.0, 200.0)

    for k in params:
        params[k] = coerce_model_param(k, params[k])
    return params


def initialize_random_parameter_file(param_file: str | Path, active_params: list[str]) -> OrderedDict:
    params = random_initial_parameters(active_params)
    write_parameter_file(params, param_file)
    return params


def write_parameter_file(parameters: OrderedDict, param_file: str | Path):
    param_file = Path(param_file)
    _safe_makedirs(param_file.parent)
    with param_file.open("w") as f:
        for k, v in parameters.items():
            f.write(f"{k}={float(v)}\n")


def write_parameter_snapshot(parameters: OrderedDict, site_outdir: str | Path):
    write_parameter_file(parameters, Path(site_outdir) / "parameters.txt")


def write_config_snapshot(src_config: configparser.ConfigParser, out_path: str | Path):
    out_path = Path(out_path)
    _safe_makedirs(out_path.parent)
    with out_path.open("w") as f:
        src_config.write(f)


def params_to_vector(params: OrderedDict, active_params: list[str]) -> np.ndarray:
    return np.array([params[k] for k in active_params], dtype=float)


def vector_to_params(vec: np.ndarray, template: OrderedDict, active_params: list[str]) -> OrderedDict:
    out = OrderedDict(template)
    for k, v in zip(active_params, vec):
        out[k] = coerce_model_param(k, float(v))
    return out


def sample_random_params_around(base_params: OrderedDict, active_params: list[str], log10_span: float) -> OrderedDict:
    out = OrderedDict(base_params)

    def logmul(v, span):
        basep = float(max(1e-12, abs(v)))
        u = random.uniform(-span, span)
        return basep * (10 ** u)

    for k in active_params:
        base = float(base_params[k])
        if k in {"bs", "br", "be"}:
            out[k] = base + random.uniform(-3.0, 3.0)
        elif bounded01_param(k):
            out[k] = base + random.uniform(-0.5, 0.5)
        elif positive_param(k):
            out[k] = logmul(base if base > 0 else 1.0, log10_span)
        elif k == "leaf_offset":
            out[k] = base + random.uniform(-200.0, 200.0)
        else:
            out[k] = base
        out[k] = coerce_model_param(k, out[k])

    return out


def fixed_axis_limits_from_observed(training_diameters: np.ndarray, bins: int = 30) -> dict:
    obs = np.asarray(training_diameters, dtype=float)
    obs = obs[np.isfinite(obs)]
    if obs.size == 0:
        return {"xlim": (0.0, 1.0), "ylim": (0.0, 1.0)}
    xmin = float(np.min(obs))
    xmax = float(np.max(obs))
    span = xmax - xmin
    pad = 0.05 * span if span > 0 else 0.5
    xlim = (xmin - pad, xmax + pad)
    hist, _ = np.histogram(obs, bins=bins, density=True)
    ymax = float(np.max(hist)) if hist.size else 1.0
    if not np.isfinite(ymax) or ymax <= 0:
        ymax = 1.0
    return {"xlim": xlim, "ylim": (0.0, 1.1 * ymax)}


def tussock_model(config_path: str | Path, output_dir: str | Path, output_mode: str, project_root: str | Path):
    cp = configparser.ConfigParser()
    cp.read(config_path)

    num_sims = int(cp.get("Tussock Model", "nsims"))
    num_threads = int(cp.get("Tussock Model", "nthreads"))
    sim_time = int(cp.get("Tussock Model", "nyears"))
    mode_flag = 1 if output_mode.lower().startswith("s") else 0

    cpp_input = f"{sim_time}\n{num_sims}\n{Path(output_dir)}\n{num_threads}\n{mode_flag}\n"

    exe = os.path.abspath(os.path.join("model", "tussock_model"))
    if not os.path.exists(exe):
        raise FileNotFoundError(f"Expected binary not found: {exe}")

    p = subprocess.Popen(
        [exe, "--config", str(Path(config_path).resolve())],
        cwd=str(project_root),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = p.communicate(input=cpp_input)
    if p.returncode != 0:
        raise RuntimeError(f"tussock_model failed (code={p.returncode})\nstdout:\n{out}\nstderr:\n{err}")


def wasserstein_distance_1d(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        return np.inf
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
    return w1


def barrier_loss(fail_stats, constraint_year, min_alive):
    T = float(max(1, constraint_year))
    v_alive = []
    v_over = []
    v_missing = []
    v_early_ext = []
    for s in fail_stats:
        alive_y = int(s.get("alive_y", 0))
        overflow_t = s.get("overflow_t", None)
        extinct_t = s.get("extinct_t", None)
        missing_year = bool(s.get("missing_year", False))
        v_alive.append(max(0.0, (min_alive - alive_y) / max(1.0, float(min_alive))))
        v_over.append(0.0 if overflow_t is None else max(0.0, (T - float(min(constraint_year, overflow_t))) / T))
        v_missing.append(1.0 if missing_year else 0.0)
        v_early_ext.append(0.0 if extinct_t is None else max(0.0, (T - float(min(constraint_year, extinct_t))) / T))
    return 1.0 * float(np.mean(v_alive)) + 5.0 * float(np.mean(v_over)) + 5.0 * float(np.mean(v_missing)) + 0.5 * float(np.mean(v_early_ext))


def read_sim_summaries(sim_outdir: str | Path, num_sims: int) -> pd.DataFrame:
    summary_dir = Path(sim_outdir) / "summaries"
    if not summary_dir.is_dir():
        return pd.DataFrame({
            "sim_id": np.arange(num_sims, dtype=int),
            "final_t": -1,
            "final_diameter": np.nan,
            "alive_y": 0,
            "rmax_y": np.inf,
            "overflow_t": -1,
            "extinct_t": -1,
            "missing_year": 1,
            "alive_final": 0,
            "LeafArea": np.nan
        })

    dfs = []
    for i in range(num_sims):
        fn = summary_dir / f"summary_{i}.csv"
        if fn.exists():
            dfs.append(pd.read_csv(fn))

    if not dfs:
        return pd.DataFrame(columns=[
            "sim_id", "final_t", "final_diameter", "alive_y", "rmax_y",
            "overflow_t", "extinct_t", "missing_year", "alive_final", "LeafArea"
        ])

    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["sim_id"], keep="last").set_index("sim_id").reindex(range(num_sims)).reset_index()

    for c, default in [("LeafArea", np.nan), ("alive_final", 0), ("final_t", -1)]:
        if c not in df.columns:
            df[c] = default

    df["missing_year"] = df["missing_year"].fillna(1).astype(int)
    df["alive_y"] = df["alive_y"].fillna(0).astype(int)
    df["alive_final"] = df["alive_final"].fillna(0).astype(int)
    df["rmax_y"] = df["rmax_y"].fillna(np.inf)
    df["final_diameter"] = pd.to_numeric(df["final_diameter"], errors="coerce")
    df["overflow_t"] = df["overflow_t"].fillna(-1).astype(int)
    df["extinct_t"] = df["extinct_t"].fillna(-1).astype(int)
    df["final_t"] = df["final_t"].fillna(-1).astype(int)
    df["LeafArea"] = pd.to_numeric(df["LeafArea"], errors="coerce")
    return df


def write_population_results(sim_df: pd.DataFrame, iteration_label: int, out_csv_path: str | Path, overgrown_radius_threshold: float):
    out_csv_path = Path(out_csv_path)
    _safe_makedirs(out_csv_path.parent)
    file_exists = out_csv_path.exists()

    alive_final_flag = sim_df["alive_final"].to_numpy(dtype=int)
    final_diameter = sim_df["final_diameter"].to_numpy(dtype=float)
    rmax_y = sim_df["rmax_y"].to_numpy(dtype=float)
    overflow_t = sim_df["overflow_t"].to_numpy(dtype=int)

    row = {
        "iteration": int(iteration_label),
        "alive_tussocks_final": int(np.sum(alive_final_flag > 0)),
        "extinct_tussocks_final": int(np.sum(alive_final_flag <= 0)),
        "overgrown_tussocks": int(np.sum(np.isfinite(rmax_y) & (rmax_y > float(overgrown_radius_threshold)))),
        "overflow_tussocks": int(np.sum(overflow_t >= 0)),
        "avg_tussock_diameter": float(np.nanmean(final_diameter)) if np.any(np.isfinite(final_diameter)) else np.nan,
    }

    with out_csv_path.open("a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def diameter_objective(sim_outdir, num_sims, iteration_label, training_data, frames_dir, axis_limits, constraints: ConstraintSettings, plotting: PlotSettings):
    training_data = training_data.copy()
    training_data["field_davg"] = pd.to_numeric(training_data["diam"], errors="coerce")
    training_diameters = training_data["field_davg"].dropna().values
    if training_diameters.size == 0:
        return float("inf")

    obs_std = float(np.std(training_diameters)) if training_diameters.size > 1 else 1.0
    if not np.isfinite(obs_std) or obs_std <= 0:
        obs_std = 1.0

    df = read_sim_summaries(sim_outdir, num_sims=num_sims)

    missing_mask = (df["missing_year"].to_numpy(dtype=int) == 1)
    overflow_mask = (df["overflow_t"].to_numpy(dtype=int) >= 0)
    alive_y = df["alive_y"].to_numpy(dtype=float)
    alive_final = df["alive_final"].to_numpy(dtype=int)
    missing_frac = float(np.mean(missing_mask)) if num_sims > 0 else 0.0
    over_frac = float(np.mean(overflow_mask)) if num_sims > 0 else 0.0
    extinct_final = int((alive_final == 0).sum())
    extinct_frac = extinct_final / max(1, num_sims)

    if constraints.hard_fail_on_overflow and overflow_mask.any():
        return float((1e6 * obs_std) * (1.0 + over_frac))

    ok = (
        (df["missing_year"].to_numpy(dtype=int) == 0) &
        (df["alive_y"].to_numpy(dtype=int) >= constraints.min_alive_tillers) &
        (df["overflow_t"].to_numpy(dtype=int) < 0)
    )
    if constraints.require_survive_to_end_for_fit:
        ok = ok & (df["alive_final"].to_numpy(dtype=int) > 0)

    pass_count = int(ok.sum())
    pass_frac = pass_count / max(1, num_sims)
    target_pass_frac = constraints.constraint_pass_frac
    pass_shortfall = max(0.0, target_pass_frac - pass_frac) / max(1e-12, target_pass_frac)

    fail_stats = []
    for row in df.itertuples(index=False):
        overflow_t = None if int(row.overflow_t) < 0 else int(row.overflow_t)
        extinct_t = None if int(row.extinct_t) < 0 else int(row.extinct_t)
        missing_year = bool(int(row.missing_year))
        fail_stats.append({
            "alive_y": 0 if missing_year else int(row.alive_y),
            "overflow_t": overflow_t,
            "extinct_t": extinct_t,
            "missing_year": missing_year,
            "alive_final": int(getattr(row, "alive_final", 0)),
            "final_t": int(getattr(row, "final_t", -1)),
        })

    fit_mask = (
        (df["missing_year"].to_numpy(dtype=int) == 0) &
        (df["overflow_t"].to_numpy(dtype=int) < 0)
    )
    if constraints.require_survive_to_end_for_fit:
        fit_mask = fit_mask & (df["alive_final"].to_numpy(dtype=int) > 0)

    sim_diam_fit = df.loc[fit_mask, "final_diameter"].to_numpy(dtype=float)
    sim_diam_fit = sim_diam_fit[np.isfinite(sim_diam_fit)]

    if sim_diam_fit.size == 0:
        fit_loss = 5.0 * obs_std
        sd_loss = 1.0
    else:
        fit_loss = wasserstein_distance_1d(training_diameters, sim_diam_fit)
        obs_sd = float(np.std(training_diameters)) if training_diameters.size > 1 else 1.0
        sim_sd = float(np.std(sim_diam_fit)) if sim_diam_fit.size > 1 else 0.0
        if not np.isfinite(obs_sd) or obs_sd <= 0:
            obs_sd = 1.0
        if not np.isfinite(sim_sd):
            sim_sd = 0.0
        sd_loss = abs(sim_sd - obs_sd) / obs_sd

    bar = barrier_loss(
        fail_stats=fail_stats,
        constraint_year=constraints.constraint_year,
        min_alive=constraints.min_alive_tillers,
    )

    low_alive_frac = float(np.mean(alive_y < constraints.min_alive_tillers)) if num_sims > 0 else 0.0
    ext_term = float(constraints.extinction_weight) * extinct_frac * obs_std

    leaf = df["LeafArea"].to_numpy(dtype=float)
    leaf_finite = np.isfinite(leaf)
    leaf_bad = leaf_finite & ((leaf <= 0.0) | (leaf >= 2000.0))
    denom = max(1, int(leaf_finite.sum()))
    leaf_bad_frac = float(leaf_bad.sum()) / float(denom)

    loss = (
        float(fit_loss)
        + 2.0 * obs_std * float(sd_loss)
        + 5.0 * obs_std * float(bar)
        + 2.0 * obs_std * low_alive_frac
        + 10.0 * obs_std * over_frac
        + 10.0 * obs_std * missing_frac
        + ext_term
        + 10.0 * obs_std * leaf_bad_frac
        + 5.0 * obs_std * pass_shortfall
    )

    do_plot = plotting.plot_every is not None and int(plotting.plot_every) > 0 and (iteration_label % int(plotting.plot_every) == 0)
    if do_plot:
        Path(frames_dir).mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots()
        if plotting.plot_kde and _HAS_SNS:
            sns.kdeplot(training_diameters, label="Observed", linewidth=1, ax=ax)
            if sim_diam_fit.size > 0:
                sns.kdeplot(sim_diam_fit, label="Modeled (fit subset)", linewidth=1, ax=ax)
        else:
            ax.hist(training_diameters, bins=30, density=True, alpha=0.4, label="Observed")
            if sim_diam_fit.size > 0:
                ax.hist(sim_diam_fit, bins=30, density=True, alpha=0.4, label="Modeled (fit subset)")
        ax.set_xlim(*axis_limits["xlim"])
        ax.set_ylim(*axis_limits["ylim"])
        ax.legend()
        ax.set_title(
            f"Iter: {iteration_label} | loss={loss:.3g} | pass={pass_count}/{num_sims} | "
            f"fit={fit_loss:.3g} | sd={sd_loss:.3g} | bar={bar:.3g} | "
            f"overflow={over_frac:.1%} | extinct_final={extinct_final}/{num_sims}"
        )
        ax.set_xlabel("Tussock Diameter")
        plt.savefig(Path(frames_dir) / f"Mean_Tuss_diameter_iteration_{iteration_label}.png", dpi=200)
        plt.close(fig)

    return float(loss)


def animate_fitting(frames_dir, iteration_labels, outfilename):
    frames = []
    frames_dir = Path(frames_dir)
    outfilename = Path(outfilename)

    for lab in iteration_labels:
        fn = frames_dir / f"Mean_Tuss_diameter_iteration_{lab}.png"
        if fn.exists():
            frames.append(Image.open(fn))
    if not frames:
        return

    frames[0].save(outfilename, save_all=True, append_images=frames[1:], duration=75, loop=0)

    for fn in frames_dir.iterdir():
        fn.unlink()
    frames_dir.rmdir()


def write_optimization_results(parameters: OrderedDict, active_params: list[str], loss: float, iteration_label, out_csv_path: str | Path):
    out_csv_path = Path(out_csv_path)
    _safe_makedirs(out_csv_path.parent)
    file_exists = out_csv_path.exists()
    row = {k: parameters[k] for k in active_params}
    row.update({"loss": float(loss), "iteration": iteration_label})

    with out_csv_path.open("a", newline="") as csvfile:
        fieldnames = active_params + ["loss", "iteration"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def cma_es_optimize(f, x0, sigma0, max_evals, tol_f, tol_x, project_fn, popsize=0, patience=20):
    n = x0.size
    if popsize is None or popsize <= 0:
        lmbda = 4 + int(3 * np.log(max(1, n)))
    else:
        lmbda = int(popsize)
    mu = max(1, lmbda // 2)
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = weights / np.sum(weights)
    mueff = 1.0 / np.sum(weights ** 2)
    cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
    cs = (mueff + 2.0) / (n + mueff + 5.0)
    c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
    damps = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (n + 1.0)) - 1.0) + cs
    chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))
    m = project_fn(np.array(x0, dtype=float))
    sigma = float(max(1e-12, sigma0))
    C = np.eye(n)
    B = np.eye(n)
    D = np.ones(n)
    invsqrtC = np.eye(n)
    pc = np.zeros(n)
    ps = np.zeros(n)
    evals = 0
    generation = 0
    best_x = m.copy()
    best_f = float("inf")
    stall = 0

    while evals < max_evals:
        generation += 1
        remaining = max_evals - evals
        cur_lambda = min(lmbda, remaining)
        if cur_lambda <= 0:
            break
        arz = np.random.randn(n, cur_lambda)
        ary = (B @ (D[:, None] * arz)).T
        arx = np.array([project_fn(m + sigma * ary[k]) for k in range(cur_lambda)])
        fvals = np.empty(cur_lambda, dtype=float)
        for k in range(cur_lambda):
            fvals[k] = f(arx[k])
            evals += 1
            if fvals[k] < best_f:
                best_f = float(fvals[k])
                best_x = arx[k].copy()
        order = np.argsort(fvals)
        arx = arx[order]
        ary = ary[order]
        fvals = fvals[order]
        use_mu = min(mu, cur_lambda)
        w = weights[:use_mu]
        w = w / np.sum(w)
        m = np.sum(arx[:use_mu] * w[:, None], axis=0)
        y_w = np.sum(ary[:use_mu] * w[:, None], axis=0)
        ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * (invsqrtC @ y_w)
        norm_ps = np.linalg.norm(ps)
        left = norm_ps / np.sqrt(1.0 - (1.0 - cs) ** (2.0 * generation))
        right = (1.4 + 2.0 / (n + 1.0)) * chi_n
        hsig = 1.0 if left < right else 0.0
        pc = (1.0 - cc) * pc + hsig * np.sqrt(cc * (2.0 - cc) * mueff) * y_w
        rank_mu = np.zeros((n, n))
        for i in range(use_mu):
            rank_mu += w[i] * np.outer(ary[i], ary[i])
        C = ((1.0 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1.0 - hsig) * cc * (2.0 - cc) * C) + cmu * rank_mu)
        sigma = sigma * np.exp((cs / damps) * (norm_ps / chi_n - 1.0))
        sigma = float(max(1e-12, sigma))
        C = 0.5 * (C + C.T)
        eigvals, eigvecs = np.linalg.eigh(C + 1e-12 * np.eye(n))
        eigvals = np.maximum(eigvals, 1e-20)
        D = np.sqrt(eigvals)
        B = eigvecs
        invsqrtC = B @ np.diag(1.0 / D) @ B.T
        f_spread = float(np.max(fvals) - np.min(fvals)) if cur_lambda > 1 else 0.0
        x_spread = float(sigma * np.max(D))
        if fvals[0] <= best_f + 1e-12:
            stall += 1
        else:
            stall = 0
        if f_spread < tol_f and x_spread < tol_x:
            break
        if patience is not None and patience > 0 and stall >= patience:
            break

    return best_x, best_f, evals


def build_projection_functions(opt_settings: OptimizationSettings, active_params: list[str], template_params: OrderedDict):
    bounded_idxs = {name: i for i, name in enumerate(active_params) if bounded01_param(name)}
    positive_idxs = {name: i for i, name in enumerate(active_params) if positive_param(name)}
    leaf_idx = active_params.index("leaf_offset") if "leaf_offset" in active_params else None

    def project_vec(x: np.ndarray) -> np.ndarray:
        x = np.array(x, dtype=float)
        x[~np.isfinite(x)] = 0.0
        if opt_settings.optimize_log_space:
            for idx in bounded_idxs.values():
                x[idx] = float(np.clip(x[idx], -10.0, 10.0))
            for idx in positive_idxs.values():
                x[idx] = float(max(0.0, x[idx]))
            if leaf_idx is not None and not np.isfinite(x[leaf_idx]):
                x[leaf_idx] = 0.0
            return x
        for idx in bounded_idxs.values():
            x[idx] = float(np.clip(x[idx], 0.0, 1.0))
        for idx in positive_idxs.values():
            x[idx] = float(max(0.0, x[idx]))
        if leaf_idx is not None and not np.isfinite(x[leaf_idx]):
            x[leaf_idx] = 0.0
        return x

    def x_to_model_params(x: np.ndarray) -> OrderedDict:
        if opt_settings.optimize_log_space:
            vec = np.array(x, dtype=float).copy()
            for idx in bounded_idxs.values():
                vec[idx] = 1.0 / (1.0 + np.exp(-vec[idx]))
            return vector_to_params(vec, template_params, active_params)
        return vector_to_params(project_vec(x), template_params, active_params)

    def params_to_x(params: OrderedDict) -> np.ndarray:
        vec = params_to_vector(params, active_params)
        if opt_settings.optimize_log_space:
            x = vec.copy()
            for idx in bounded_idxs.values():
                p = float(min(1.0 - 1e-9, max(1e-9, x[idx])))
                x[idx] = np.log(p / (1.0 - p))
            return x
        return vec

    return project_vec, x_to_model_params, params_to_x


def sample_training_subset(df: pd.DataFrame, percent: float, with_replacement: bool, rng: random.Random) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    frac = max(0.0, min(100.0, float(percent))) / 100.0
    if frac <= 0.0:
        raise ValueError("train_percent must be > 0")
    n_total = len(df)
    n_take = max(1, int(math.ceil(frac * n_total)))

    if with_replacement:
        idx = [rng.randrange(n_total) for _ in range(n_take)]
    else:
        if n_take >= n_total:
            idx = list(range(n_total))
            rng.shuffle(idx)
        else:
            idx = rng.sample(range(n_total), n_take)

    subset = df.iloc[idx].copy().reset_index(drop=True)
    return subset


def run_one_fit_set(set_idx: int,
                    base_config: configparser.ConfigParser,
                    run_settings: RunSettings,
                    full_training_df: pd.DataFrame,
                    site_list: list[str],
                    master_rng: random.Random):
    output_root = Path(run_settings.paths.output_dir).resolve()
    run_group_root = output_root / run_settings.paths.subdir
    set_root = run_group_root / f"set_{set_idx:03d}"
    set_root.mkdir(parents=True, exist_ok=True)

    run_param_file = set_root / "parameters.txt"
    run_config_snapshot = set_root / "config_snapshot.ini"
    sampled_training_csv = set_root / "sampled_training_data.csv"
    sampled_training_meta_csv = set_root / "sampled_training_metadata.csv"

    config_for_run = configparser.ConfigParser()
    for sec in base_config.sections():
        config_for_run[sec] = dict(base_config[sec])

    if config_for_run.has_option("Paths", "param_file"):
        config_for_run.remove_option("Paths", "param_file")

    config_for_run.set("Paths", "output_dir", str(set_root))
    config_for_run.set("Paths", "training_csv", str(sampled_training_csv))
    write_config_snapshot(config_for_run, run_config_snapshot)

    set_seed = master_rng.randrange(0, 2**31 - 1)
    set_rng = random.Random(set_seed)

    sampled_training_df = sample_training_subset(
        full_training_df,
        percent=run_settings.resampling.train_percent,
        with_replacement=run_settings.resampling.sample_with_replacement,
        rng=set_rng,
    )
    sampled_training_df.to_csv(sampled_training_csv, index=False)

    pd.DataFrame([{
        "set_idx": set_idx,
        "seed": set_seed,
        "n_rows_sampled": len(sampled_training_df),
        "n_rows_full": len(full_training_df),
        "train_percent": run_settings.resampling.train_percent,
        "sample_with_replacement": run_settings.resampling.sample_with_replacement,
    }]).to_csv(sampled_training_meta_csv, index=False)

    if not run_param_file.exists():
        print(f"[set {set_idx:03d}] creating random init: {run_param_file}")
        template_params = initialize_random_parameter_file(run_param_file, run_settings.active_params)
    else:
        template_params = read_parameter_file(run_param_file)

    nyears = int(base_config.get("Tussock Model", "nyears"))
    if nyears < run_settings.constraints.constraint_year:
        raise ValueError(f"nyears={nyears} < constraint_year={run_settings.constraints.constraint_year}")

    project_vec, x_to_model_params, params_to_x = build_projection_functions(
        run_settings.optimization, run_settings.active_params, template_params
    )

    for site in site_list:
        print("\n====================================")
        print(f"   SET {set_idx:03d} | PARAMETERIZING SITE: {site}")
        print("====================================\n")

        if site == "ALL":
            training_data = sampled_training_df.copy()
            site_tag = "ALL"
        else:
            training_data = sampled_training_df[sampled_training_df["site"] == site].copy()
            site_tag = str(site)

        if training_data.empty:
            print(f"[set {set_idx:03d}] site {site_tag} has no sampled rows, skipping")
            continue

        training_data["field_davg"] = pd.to_numeric(training_data["diam"], errors="coerce")
        training_diameters = training_data["field_davg"].dropna().values
        if training_diameters.size == 0:
            print(f"[set {set_idx:03d}] site {site_tag} has no valid diam values, skipping")
            continue

        axis_limits = fixed_axis_limits_from_observed(training_diameters, bins=30)

        site_outdir = set_root / site_tag
        site_outdir.mkdir(parents=True, exist_ok=True)

        cpp_outdir = site_outdir / "simulation_outputs"
        cpp_outdir.mkdir(parents=True, exist_ok=True)

        frames_dir = site_outdir / "mean_diameter_frames"
        opt_csv_path = site_outdir / "optimization_results.csv"
        pop_csv_path = site_outdir / "population_results.csv"

        for p in (opt_csv_path, pop_csv_path):
            if p.exists():
                p.unlink()

        eval_label = 0
        frame_labels = []
        best_seen = {"loss": float("inf"), "params": None}

        def objective_vec(x: np.ndarray) -> float:
            nonlocal eval_label
            eval_label += 1
            x = project_vec(x)
            params = x_to_model_params(x)

            write_parameter_file(params, run_param_file)
            write_parameter_snapshot(params, site_outdir)

            tussock_model(
                config_path=run_config_snapshot,
                output_dir=cpp_outdir,
                output_mode="summary",
                project_root=run_settings.project_root,
            )

            loss = diameter_objective(
                sim_outdir=cpp_outdir,
                num_sims=int(base_config.get("Tussock Model", "nsims")),
                iteration_label=eval_label,
                training_data=training_data,
                frames_dir=frames_dir,
                axis_limits=axis_limits,
                constraints=run_settings.constraints,
                plotting=run_settings.plotting,
            )

            if run_settings.plotting.plot_every and run_settings.plotting.plot_every > 0 and eval_label % run_settings.plotting.plot_every == 0:
                frame_labels.append(eval_label)

            write_optimization_results(params, run_settings.active_params, loss, eval_label, opt_csv_path)

            sim_df = read_sim_summaries(cpp_outdir, num_sims=int(base_config.get("Tussock Model", "nsims")))
            write_population_results(sim_df, eval_label, pop_csv_path, run_settings.constraints.overgrown_radius_threshold)

            if loss < best_seen["loss"]:
                best_seen["loss"] = loss
                best_seen["params"] = OrderedDict(params)

            return loss

        if len(run_settings.active_params) == 0:
            loss_trial = objective_vec(np.array([], dtype=float))
            best_seen["loss"] = loss_trial
            best_seen["params"] = OrderedDict(template_params)
        else:
            best_init_loss = float("inf")
            best_init_params = OrderedDict(template_params)

            for _ in range(run_settings.optimization.n_init):
                trial_params = sample_random_params_around(
                    template_params,
                    run_settings.active_params,
                    run_settings.optimization.init_log10_span,
                )
                x_trial = project_vec(params_to_x(trial_params))
                loss_trial = objective_vec(x_trial)
                if loss_trial < best_init_loss:
                    best_init_loss = loss_trial
                    best_init_params = OrderedDict(trial_params)

            x0 = project_vec(params_to_x(best_init_params))

            if run_settings.optimization.optimize_log_space:
                sigma0 = run_settings.optimization.cma_sigma
            else:
                step_scales = []
                for i in range(x0.size):
                    mag = abs(x0[i])
                    step_scales.append(
                        run_settings.optimization.step_frac * mag if mag > 1e-12 else run_settings.optimization.step_abs
                    )
                sigma0 = float(np.median(step_scales)) if step_scales else run_settings.optimization.cma_sigma
                sigma0 = max(1e-6, sigma0)

            remaining_budget = max(0, run_settings.optimization.max_evals - eval_label)
            if remaining_budget > 0 and x0.size > 0:
                best_x, _, _ = cma_es_optimize(
                    objective_vec,
                    x0,
                    sigma0,
                    remaining_budget,
                    run_settings.optimization.tol_f,
                    run_settings.optimization.tol_x,
                    project_vec,
                    run_settings.optimization.cma_popsize,
                    run_settings.optimization.cma_patience,
                )
                if best_seen["params"] is None:
                    best_seen["params"] = x_to_model_params(best_x)

        final_params = best_seen["params"] if best_seen["params"] is not None else OrderedDict(template_params)

        print(f"[set {set_idx:03d} | {site_tag}] best loss: {best_seen['loss']:.6g}")
        for k in run_settings.active_params:
            print(f"  {k}={final_params[k]}")

        write_parameter_file(final_params, run_param_file)
        write_parameter_snapshot(final_params, site_outdir)

        final_sims_dir = site_outdir / "final_sims"
        final_sims_dir.mkdir(parents=True, exist_ok=True)

        tussock_model(
            config_path=run_config_snapshot,
            output_dir=final_sims_dir,
            output_mode="full",
            project_root=run_settings.project_root,
        )

        final_summary_df = read_sim_summaries(final_sims_dir, num_sims=int(base_config.get("Tussock Model", "nsims")))
        final_pop_csv_path = site_outdir / "final_population_results.csv"
        if final_pop_csv_path.exists():
            final_pop_csv_path.unlink()
        write_population_results(final_summary_df, eval_label, final_pop_csv_path, run_settings.constraints.overgrown_radius_threshold)

        if run_settings.plotting.plot_every and run_settings.plotting.plot_every > 0:
            animate_fitting(frames_dir, frame_labels, site_outdir / "diameter_dist_fitting.gif")

        print(f"Completed set {set_idx:03d}, site: {site_tag}")


def main():
    args = parse_args()
    combined_config, run_settings = load_combined_config(args.config, cli_sites=args.sites)

    full_training_df = pd.read_csv(run_settings.paths.training_csv)

    if run_settings.sites is None:
        site_list = ["ALL"]
    elif len(run_settings.sites) == 1 and run_settings.sites[0].lower() == "all":
        site_list = sorted(full_training_df["site"].dropna().unique())
    else:
        site_list = run_settings.sites

    output_root = Path(run_settings.paths.output_dir).resolve()
    group_root = output_root / run_settings.paths.subdir
    group_root.mkdir(parents=True, exist_ok=True)

    seed = run_settings.resampling.random_seed
    if seed is None:
        seed = random.randrange(0, 2**31 - 1)
    master_rng = random.Random(seed)

    pd.DataFrame([{
        "master_seed": seed,
        "n_sets": run_settings.resampling.n_sets,
        "train_percent": run_settings.resampling.train_percent,
        "sample_with_replacement": run_settings.resampling.sample_with_replacement,
        "subdir": run_settings.paths.subdir,
        "output_root": str(output_root),
    }]).to_csv(group_root / "run_manifest.csv", index=False)

    for set_idx in range(1, run_settings.resampling.n_sets + 1):
        run_one_fit_set(
            set_idx=set_idx,
            base_config=combined_config,
            run_settings=run_settings,
            full_training_df=full_training_df,
            site_list=site_list,
            master_rng=master_rng,
        )

    print("All sets done.")


if __name__ == "__main__":
    main()