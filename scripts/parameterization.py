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
import cma

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
    "c_space_establishment",
    "k_crowd_survival",
    "k_crowd_reproduction",
    "k_crowd_establishment",
    "leaf_offset",
]


@dataclass
class ParameterConstraint:
    lower: float | None = None
    upper: float | None = None


@dataclass
class PathSettings:
    output_dir: str
    training_csv: str
    tiller_count_diameter_csv: str
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
    overgrown_radius_threshold: float
    hard_fail_on_overflow: bool
    require_survive_to_end_for_fit: bool
    diameter_sd_weight: float
    extinct_frac_weight: float
    overflow_frac_weight: float
    tiller_count_diameter_weight: float


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
    spatial_survival_form: str
    spatial_reproduction_form: str
    spatial_establishment_form: str


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
    param_constraints: dict[str, ParameterConstraint]


def parse_args():
    parser = argparse.ArgumentParser(description="Tussock model parameterization with repeated random training subsets")
    parser.add_argument("--config", type=str, required=True, help="Path to the combined ini config file")
    parser.add_argument("--sites", nargs="*", default=None)
    return parser.parse_args()


def read_bool(config, section, key, fallback=False):
    return config.getboolean(section, key, fallback=fallback)



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
        active.extend(["ke", "be", "c_space_establishment"])
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


def parse_parameter_constraint(raw: str) -> ParameterConstraint:
    """Parse one parameter constraint from the config.

    Supported syntax:
      blank      -> unconstrained
      >0 or >=0  -> lower bound only
      <0 or <=0  -> upper bound only
      0,1        -> lower and upper bounds
      ,1         -> upper bound only
      -5,        -> lower bound only
    """
    raw = str(raw).strip()
    if raw == "":
        return ParameterConstraint()
    if raw.startswith(">="):
        return ParameterConstraint(lower=float(raw[2:].strip()), upper=None)
    if raw.startswith(">"):
        return ParameterConstraint(lower=float(raw[1:].strip()), upper=None)
    if raw.startswith("<="):
        return ParameterConstraint(lower=None, upper=float(raw[2:].strip()))
    if raw.startswith("<"):
        return ParameterConstraint(lower=None, upper=float(raw[1:].strip()))
    if "," in raw:
        lo_raw, hi_raw = raw.split(",", 1)
        lo = float(lo_raw.strip()) if lo_raw.strip() != "" else None
        hi = float(hi_raw.strip()) if hi_raw.strip() != "" else None
        if lo is not None and hi is not None and lo > hi:
            raise ValueError(f"Invalid constraint '{raw}': lower bound is greater than upper bound")
        return ParameterConstraint(lower=lo, upper=hi)
    raise ValueError(f"Could not parse parameter constraint '{raw}'. Use blank, >0, <0, or lower,upper such as 0,1.")


def read_parameter_constraints(config: configparser.ConfigParser) -> dict[str, ParameterConstraint]:
    constraints = {name: ParameterConstraint() for name in ALL_MODEL_PARAM_NAMES}

    # Mixing coefficients are probabilities/weights and must remain in [0, 1].
    for name in {
        "c_space_survival",
        "c_space_reproduction",
        "c_space_establishment",
    }:
        constraints[name] = ParameterConstraint(lower=0.0, upper=1.0)

    if not config.has_section("ParameterConstraints"):
        return constraints

    for name in ALL_MODEL_PARAM_NAMES:
        raw = config.get("ParameterConstraints", name, fallback="").strip()
        if raw:
            constraints[name] = parse_parameter_constraint(raw)

    return constraints


def apply_parameter_constraint(name: str, value: float, param_constraints: dict[str, ParameterConstraint] | None = None) -> float:
    if not np.isfinite(value):
        raise ValueError(f"Non-finite value for parameter '{name}': {value}")
    if param_constraints is None:
        param_constraints = {}
    c = param_constraints.get(name, ParameterConstraint())
    out = float(value)
    if c.lower is not None:
        out = max(float(c.lower), out)
    if c.upper is not None:
        out = min(float(c.upper), out)
    return float(out)


def has_parameter_constraint(name: str, param_constraints: dict[str, ParameterConstraint]) -> bool:
    c = param_constraints.get(name, ParameterConstraint())
    return c.lower is not None or c.upper is not None


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
    tiller_count_diameter_csv_raw = config.get(
        "Paths",
        "tiller_count_diameter_csv",
        fallback="./input_data/eriophorum_tiller_count_diameter.csv",
    )
    subdir = config.get("Paths", "subdir", fallback="runs")

    output_dir = resolve_path(project_root, output_dir_raw)
    training_csv = resolve_path(project_root, training_csv_raw)
    tiller_count_diameter_csv = resolve_path(project_root, tiller_count_diameter_csv_raw)

    paths = PathSettings(
        output_dir=str(output_dir),
        training_csv=str(training_csv),
        tiller_count_diameter_csv=str(tiller_count_diameter_csv),
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
        overgrown_radius_threshold=config.getfloat("Constraints", "overgrown_radius_threshold", fallback=2.5),
        hard_fail_on_overflow=read_bool(config, "Constraints", "hard_fail_on_overflow", fallback=False),
        require_survive_to_end_for_fit=read_bool(config, "Constraints", "require_survive_to_end_for_fit", fallback=False),
        diameter_sd_weight=config.getfloat("Constraints", "diameter_sd_weight", fallback=1.0),
        extinct_frac_weight=config.getfloat("Constraints", "extinct_frac_weight", fallback=5.0),
        overflow_frac_weight=config.getfloat("Constraints", "overflow_frac_weight", fallback=5.0),
        tiller_count_diameter_weight=config.getfloat(
            "Constraints",
            "tiller_count_diameter_weight",
            fallback=5.0,
        ),
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
        spatial_survival_form=config.get("Mechanisms", "spatial_survival_form", fallback="linear"),
        spatial_reproduction_form=config.get("Mechanisms", "spatial_reproduction_form", fallback="linear"),
        spatial_establishment_form=config.get("Mechanisms", "spatial_establishment_form", fallback="linear"),
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
    param_constraints = read_parameter_constraints(config)
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
        param_constraints=param_constraints,
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
        ("c_space_establishment", 0.5),
        ("k_crowd_survival", 0.1),
        ("k_crowd_reproduction", 0.1),
        ("k_crowd_establishment", 0.1),
        ("leaf_offset", 0.0),
    ])


def coerce_model_param(name: str, value: float, param_constraints: dict[str, ParameterConstraint] | None = None) -> float:
    return apply_parameter_constraint(name, value, param_constraints)


def read_parameter_file(param_file: str | Path, param_constraints: dict[str, ParameterConstraint] | None = None) -> OrderedDict:
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
            params[k] = coerce_model_param(k, float(v), param_constraints)
    return params


def random_initial_parameters(active_params: list[str], param_constraints: dict[str, ParameterConstraint] | None = None) -> OrderedDict:
    params = default_parameter_values()
    if param_constraints is None:
        param_constraints = {}

    def logu(lo, hi):
        return 10 ** random.uniform(math.log10(lo), math.log10(hi))

    def maybe_sample_from_finite_bounds(name: str) -> float | None:
        c = param_constraints.get(name, ParameterConstraint())
        if c.lower is not None and c.upper is not None and np.isfinite(c.lower) and np.isfinite(c.upper):
            return random.uniform(float(c.lower), float(c.upper))
        return None

    for name in active_params:
        bounded_sample = maybe_sample_from_finite_bounds(name)
        if bounded_sample is not None:
            params[name] = bounded_sample
        elif name in {"ks", "kr", "ke"}:
            # Slopes can be negative unless constrained by [ParameterConstraints].
            sign = 1.0 if random.random() < 0.5 else -1.0
            params[name] = sign * logu(1e-3, 100.0)
        elif name in {"k_crowd_survival", "k_crowd_reproduction", "k_crowd_establishment"}:
            # Crowding coefficients often make biological sense as positive, but are only enforced if constrained.
            params[name] = logu(1e-4, 10.0)
        elif name in {"bs", "br", "be"}:
            params[name] = random.uniform(-3.0, 3.0)
        elif name in {"c_space_survival", "c_space_reproduction", "c_space_establishment"}:
            params[name] = random.uniform(0.0, 1.0)
        elif name == "leaf_offset":
            params[name] = random.uniform(-200.0, 200.0)

    for k in params:
        params[k] = coerce_model_param(k, params[k], param_constraints)
    return params


def initialize_random_parameter_file(
    param_file: str | Path,
    active_params: list[str],
    param_constraints: dict[str, ParameterConstraint] | None = None,
) -> OrderedDict:
    params = random_initial_parameters(active_params, param_constraints)
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


def vector_to_params(
    vec: np.ndarray,
    template: OrderedDict,
    active_params: list[str],
    param_constraints: dict[str, ParameterConstraint] | None = None,
) -> OrderedDict:
    out = OrderedDict(template)
    for k, v in zip(active_params, vec):
        out[k] = coerce_model_param(k, float(v), param_constraints)
    return out


def sample_random_params_around(
    base_params: OrderedDict,
    active_params: list[str],
    log10_span: float,
    param_constraints: dict[str, ParameterConstraint] | None = None,
) -> OrderedDict:
    out = OrderedDict(base_params)
    if param_constraints is None:
        param_constraints = {}

    def logmul_preserve_sign(v, span):
        basep = float(max(1e-12, abs(v)))
        sign = -1.0 if v < 0 else 1.0
        if abs(v) <= 1e-12 and random.random() < 0.5:
            sign = -1.0
        u = random.uniform(-span, span)
        return sign * basep * (10 ** u)

    for k in active_params:
        base = float(base_params[k])
        if k in {"bs", "br", "be"}:
            out[k] = base + random.uniform(-3.0, 3.0)
        elif k in {"c_space_survival", "c_space_reproduction", "c_space_establishment"}:
            out[k] = base + random.uniform(-0.5, 0.5)
        elif k in {"ks", "kr", "ke", "k_crowd_survival", "k_crowd_reproduction", "k_crowd_establishment"}:
            out[k] = logmul_preserve_sign(base if abs(base) > 1e-12 else 1.0, log10_span)
            if k in {"ks", "kr", "ke"} and not has_parameter_constraint(k, param_constraints) and random.random() < 0.25:
                out[k] = -out[k]
        elif k == "leaf_offset":
            out[k] = base + random.uniform(-200.0, 200.0)
        else:
            out[k] = base
        out[k] = coerce_model_param(k, out[k], param_constraints)

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


def tussock_model(
    config_path: str | Path,
    output_dir: str | Path,
    output_mode: str,
    project_root: str | Path,
    simulation_seed: int,
):
    """Run the C++ simulator with a deterministic base seed.

    The patched C++ model reads TUSSOCK_BASE_SEED and deterministically derives
    one independent seed per sim_id. Reusing the same base seed for every
    objective evaluation gives common random numbers, making candidate losses
    directly comparable instead of letting CMA-ES optimize Monte Carlo noise.
    """
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

    env = os.environ.copy()
    env["TUSSOCK_BASE_SEED"] = str(int(simulation_seed))

    p = subprocess.Popen(
        [exe, "--config", str(Path(config_path).resolve())],
        cwd=str(project_root),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
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


def multivariate_energy_distance(observed: np.ndarray, simulated: np.ndarray) -> float:
    """Energy distance between two multivariate samples.

    Both samples must already be on comparable scales. The result is zero only
    when the two distributions match in the population limit.
    """
    observed = np.asarray(observed, dtype=float)
    simulated = np.asarray(simulated, dtype=float)

    if observed.ndim != 2 or simulated.ndim != 2:
        raise ValueError("Energy-distance inputs must be two-dimensional arrays")
    if observed.shape[1] != simulated.shape[1]:
        raise ValueError("Energy-distance samples must have the same number of columns")
    if observed.shape[0] == 0 or simulated.shape[0] == 0:
        return 5.0

    d_xy = np.linalg.norm(
        observed[:, None, :] - simulated[None, :, :],
        axis=2,
    ).mean()
    d_xx = np.linalg.norm(
        observed[:, None, :] - observed[None, :, :],
        axis=2,
    ).mean()
    d_yy = np.linalg.norm(
        simulated[:, None, :] - simulated[None, :, :],
        axis=2,
    ).mean()

    return float(max(0.0, 2.0 * d_xy - d_xx - d_yy))


def tiller_count_diameter_joint_loss(
    sim_df: pd.DataFrame,
    observed_df: pd.DataFrame,
) -> tuple[float, int]:
    """Match the empirical joint distribution of diameter and living tillers.

    The two coordinates are standardized by the observed standard deviations,
    so diameter and tiller count contribute comparably. Overflowed simulations
    are deliberately retained at the state where they stopped, so explosive
    trajectories remain visible to this empirical loss. Extinct and non-finite
    endpoints are excluded and handled by their own loss components.
    """
    required_obs = {"d", "N_alive"}
    missing_obs = required_obs.difference(observed_df.columns)
    if missing_obs:
        raise ValueError(
            "Tiller count/diameter CSV is missing columns: "
            + ", ".join(sorted(missing_obs))
        )

    obs = observed_df[["d", "N_alive"]].apply(
        pd.to_numeric,
        errors="coerce",
    ).dropna()
    obs = obs[(obs["d"] > 0.0) & (obs["N_alive"] > 0.0)]

    final_diameter = pd.to_numeric(sim_df["final_diameter"], errors="coerce")
    alive_final = pd.to_numeric(sim_df["alive_final"], errors="coerce")
    valid_sim = (
        final_diameter.notna()
        & np.isfinite(final_diameter)
        & (final_diameter > 0.0)
        & alive_final.notna()
        & np.isfinite(alive_final)
        & (alive_final > 0.0)
    )
    sim = sim_df.loc[valid_sim, ["final_diameter", "alive_final"]].copy()
    sim.columns = ["d", "N_alive"]
    sim = sim.apply(pd.to_numeric, errors="coerce").dropna()

    if obs.empty or sim.empty:
        return 5.0, int(len(sim))

    obs_values = obs.to_numpy(dtype=float)
    sim_values = sim.to_numpy(dtype=float)

    center = np.mean(obs_values, axis=0)
    scale = np.std(obs_values, axis=0, ddof=0)
    scale[~np.isfinite(scale) | (scale <= 0.0)] = 1.0

    obs_z = (obs_values - center) / scale
    sim_z = (sim_values - center) / scale

    return multivariate_energy_distance(obs_z, sim_z), int(len(sim_values))


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


def diameter_objective(
    sim_outdir,
    num_sims,
    iteration_label,
    training_data,
    tiller_count_diameter_data,
    frames_dir,
    axis_limits,
    constraints: ConstraintSettings,
    plotting: PlotSettings,
):
    """Evaluate diameter fit and the empirical diameter--living-tiller relation.

    There is no fixed living-tiller target at an arbitrary constraint year.
    Population size is constrained directly by the observed joint distribution
    of terminal tussock diameter and living tiller count. Overflowed simulations
    remain in both empirical comparisons at the state where they stopped, so an
    explosion cannot disappear from the objective merely because the simulator
    terminated it early.
    """
    training_data = training_data.copy()
    training_data["field_davg"] = pd.to_numeric(
        training_data["diam"],
        errors="coerce",
    )
    training_diameters = training_data["field_davg"].dropna().to_numpy(dtype=float)
    training_diameters = training_diameters[np.isfinite(training_diameters)]

    if training_diameters.size == 0:
        components = {
            "loss": float("inf"),
            "fit_loss_raw": float("inf"),
            "fit_loss_weighted": float("inf"),
            "diameter_sd_loss_raw": np.nan,
            "diameter_sd_loss_weighted": np.nan,
            "tiller_count_diameter_loss_raw": np.nan,
            "tiller_count_diameter_loss_weighted": np.nan,
            "extinct_frac_raw": np.nan,
            "extinct_loss_weighted": np.nan,
            "overflow_frac_raw": np.nan,
            "overflow_loss_weighted": np.nan,
            "obs_std": np.nan,
            "n_fit_sims": 0,
            "n_joint_sims": 0,
            "missing_frac": np.nan,
        }
        return float("inf"), components

    obs_std = (
        float(np.std(training_diameters))
        if training_diameters.size > 1
        else 1.0
    )
    if not np.isfinite(obs_std) or obs_std <= 0.0:
        obs_std = 1.0

    df = read_sim_summaries(sim_outdir, num_sims=num_sims)

    final_diameter = pd.to_numeric(
        df["final_diameter"],
        errors="coerce",
    ).to_numpy(dtype=float)
    alive_final = pd.to_numeric(
        df["alive_final"],
        errors="coerce",
    ).fillna(0).to_numpy(dtype=int)
    overflow_mask = (
        pd.to_numeric(df["overflow_t"], errors="coerce")
        .fillna(-1)
        .to_numpy(dtype=int)
        >= 0
    )

    valid_endpoint = np.isfinite(final_diameter) & (final_diameter > 0.0)
    missing_frac = float(np.mean(~valid_endpoint)) if num_sims > 0 else 0.0

    extinct_mask = alive_final <= 0
    extinct_frac = float(np.mean(extinct_mask)) if num_sims > 0 else 0.0
    overflow_frac = float(np.mean(overflow_mask)) if num_sims > 0 else 0.0

    # Overflowed endpoints are retained. Their terminal diameter and living
    # count are the state at which the threshold was crossed and therefore
    # provide useful direction to the optimizer.
    fit_mask = valid_endpoint.copy()
    if constraints.require_survive_to_end_for_fit:
        fit_mask &= ~extinct_mask

    sim_diam_fit = final_diameter[fit_mask]

    if sim_diam_fit.size == 0:
        fit_loss = 5.0 * obs_std
        sd_loss = 1.0
    else:
        fit_loss = wasserstein_distance_1d(
            training_diameters,
            sim_diam_fit,
        )

        simulated_sd = (
            float(np.std(sim_diam_fit))
            if sim_diam_fit.size > 1
            else 0.0
        )
        if not np.isfinite(simulated_sd):
            simulated_sd = 0.0

        sd_loss = abs(simulated_sd - obs_std) / obs_std

    tiller_count_diameter_loss, n_joint_sims = (
        tiller_count_diameter_joint_loss(
            sim_df=df,
            observed_df=tiller_count_diameter_data,
        )
    )

    w_sd = constraints.diameter_sd_weight
    w_count_diameter = constraints.tiller_count_diameter_weight
    w_ext = constraints.extinct_frac_weight
    w_over = constraints.overflow_frac_weight

    fit_loss_weighted = float(fit_loss)
    sd_loss_weighted = w_sd * obs_std * float(sd_loss)
    tiller_count_diameter_loss_weighted = (
        w_count_diameter
        * obs_std
        * float(tiller_count_diameter_loss)
    )
    extinct_loss_weighted = w_ext * obs_std * extinct_frac
    overflow_loss_weighted = w_over * obs_std * overflow_frac

    loss = (
        fit_loss_weighted
        + sd_loss_weighted
        + tiller_count_diameter_loss_weighted
        + extinct_loss_weighted
        + overflow_loss_weighted
    )

    components = {
        "loss": float(loss),

        # Raw terms.
        "fit_loss_raw": float(fit_loss),
        "diameter_sd_loss_raw": float(sd_loss),
        "tiller_count_diameter_loss_raw": float(
            tiller_count_diameter_loss
        ),
        "extinct_frac_raw": float(extinct_frac),
        "overflow_frac_raw": float(overflow_frac),

        # Weighted terms that sum to total loss.
        "fit_loss_weighted": float(fit_loss_weighted),
        "diameter_sd_loss_weighted": float(sd_loss_weighted),
        "tiller_count_diameter_loss_weighted": float(
            tiller_count_diameter_loss_weighted
        ),
        "extinct_loss_weighted": float(extinct_loss_weighted),
        "overflow_loss_weighted": float(overflow_loss_weighted),

        # Diagnostics.
        "obs_std": float(obs_std),
        "n_fit_sims": int(sim_diam_fit.size),
        "n_joint_sims": int(n_joint_sims),
        "missing_frac": float(missing_frac),

        # Actual weights used this iteration.
        "diameter_sd_weight": float(w_sd),
        "tiller_count_diameter_weight": float(w_count_diameter),
        "extinct_frac_weight": float(w_ext),
        "overflow_frac_weight": float(w_over),
    }

    do_plot = (
        plotting.plot_every is not None
        and int(plotting.plot_every) > 0
        and iteration_label % int(plotting.plot_every) == 0
    )

    if do_plot:
        Path(frames_dir).mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots()

        if plotting.plot_kde and _HAS_SNS:
            sns.kdeplot(
                training_diameters,
                label="Observed",
                linewidth=1,
                ax=ax,
            )
            if sim_diam_fit.size > 0:
                sns.kdeplot(
                    sim_diam_fit,
                    label="Modeled",
                    linewidth=1,
                    ax=ax,
                )
        else:
            ax.hist(
                training_diameters,
                bins=30,
                density=True,
                alpha=0.4,
                label="Observed",
            )
            if sim_diam_fit.size > 0:
                ax.hist(
                    sim_diam_fit,
                    bins=30,
                    density=True,
                    alpha=0.4,
                    label="Modeled",
                )

        ax.set_xlim(*axis_limits["xlim"])
        ax.set_ylim(*axis_limits["ylim"])
        ax.legend()
        ax.set_title(
            f"Iter: {iteration_label} | loss={loss:.3g} | "
            f"fit={fit_loss_weighted:.3g} | "
            f"sd={sd_loss_weighted:.3g} | "
            f"countdiam={tiller_count_diameter_loss_weighted:.3g} | "
            f"ext={extinct_loss_weighted:.3g} | "
            f"over={overflow_loss_weighted:.3g}"
        )
        ax.set_xlabel("Tussock Diameter")
        plt.savefig(
            Path(frames_dir)
            / f"Mean_Tuss_diameter_iteration_{iteration_label}.png",
            dpi=200,
        )
        plt.close(fig)

    return float(loss), components

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


def write_optimization_results(
    parameters: OrderedDict,
    active_params: list[str],
    loss: float,
    iteration_label,
    out_csv_path: str | Path,
    loss_components: dict | None = None,
):
    out_csv_path = Path(out_csv_path)
    _safe_makedirs(out_csv_path.parent)

    file_exists = out_csv_path.exists()

    row = {k: parameters[k] for k in active_params}
    row.update({"loss": float(loss), "iteration": iteration_label})

    if loss_components is not None:
        for k, v in loss_components.items():
            if k in {"loss", "iteration"}:
                continue
            row[k] = v

    fieldnames = active_params + ["loss", "iteration"]

    if loss_components is not None:
        extra_cols = [k for k in loss_components.keys() if k not in {"loss", "iteration"}]
        fieldnames.extend(extra_cols)

    with out_csv_path.open("a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)



def pycma_optimize(f, x0, sigma0, max_evals, tol_f, tol_x, project_fn, popsize=0, patience=20):
    """Run CMA-ES using pycma's maintained CMAEvolutionStrategy.

    The objective `f` should accept an already-projected vector. We still call
    project_fn before evaluation because this model uses custom parameter
    constraints/transforms outside of pycma's native bounds system.

    """
    x0 = project_fn(np.array(x0, dtype=float))
    sigma0 = float(max(1e-12, sigma0))
    max_evals = int(max(1, max_evals))

    opts = {
        "maxfevals": max_evals,
        "tolfun": float(tol_f),
        "tolx": float(tol_x),
        "verb_disp": 1,
        "verbose": -9,
    }

    if popsize is not None and int(popsize) > 0:
        opts["popsize"] = int(popsize)

    es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, opts)

    best_x = x0.copy()
    best_f = float("inf")
    evals = 0
    generations_without_improvement = 0

    while not es.stop() and evals < max_evals:
        xs = es.ask()

        # pycma tell() requires at least mu solutions. Do not do a partial
        # generation if the remaining evaluation budget is too small.
        mu = int(es.sp.weights.mu)
        remaining = max_evals - evals

        if remaining < mu:
            break
        if remaining < len(xs):
            break

        xs_projected = []
        fvals = []

        previous_best_f = best_f
        generation_best = float("inf")

        for x in xs:
            x_proj = project_fn(np.array(x, dtype=float))
            fx = float(f(x_proj))

            xs_projected.append(x_proj)
            fvals.append(fx)

            evals += 1

            if fx < best_f:
                best_f = fx
                best_x = x_proj.copy()

            if fx < generation_best:
                generation_best = fx

        if len(fvals) < mu:
            break

        es.tell(xs_projected, fvals)

        if best_f < previous_best_f - 1e-12:
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

        if patience is not None and int(patience) > 0:
            if generations_without_improvement >= int(patience):
                break

    return best_x, best_f, evals


def build_projection_functions(
    opt_settings: OptimizationSettings,
    active_params: list[str],
    template_params: OrderedDict,
    param_constraints: dict[str, ParameterConstraint] | None = None,
):
    if param_constraints is None:
        param_constraints = {}

    constrained_idxs = {
        name: i
        for i, name in enumerate(active_params)
        if has_parameter_constraint(name, param_constraints)
    }

    # Keep the optional logit transform only for exactly [0, 1] bounded parameters.
    logit_idxs = {
        name: i
        for i, name in enumerate(active_params)
        if (
            param_constraints.get(name, ParameterConstraint()).lower == 0.0
            and param_constraints.get(name, ParameterConstraint()).upper == 1.0
        )
    }

    leaf_idx = active_params.index("leaf_offset") if "leaf_offset" in active_params else None

    def project_vec(x: np.ndarray) -> np.ndarray:
        x = np.array(x, dtype=float)
        x[~np.isfinite(x)] = 0.0

        if opt_settings.optimize_log_space:
            # In transformed space, [0,1] parameters are represented on the logit scale.
            # Other constrained parameters are still projected directly.
            for idx in logit_idxs.values():
                x[idx] = float(np.clip(x[idx], -10.0, 10.0))
            for name, idx in constrained_idxs.items():
                if name in logit_idxs:
                    continue
                x[idx] = apply_parameter_constraint(name, x[idx], param_constraints)
            if leaf_idx is not None and not np.isfinite(x[leaf_idx]):
                x[leaf_idx] = 0.0
            return x

        for name, idx in constrained_idxs.items():
            x[idx] = apply_parameter_constraint(name, x[idx], param_constraints)
        if leaf_idx is not None and not np.isfinite(x[leaf_idx]):
            x[leaf_idx] = 0.0
        return x

    def x_to_model_params(x: np.ndarray) -> OrderedDict:
        if opt_settings.optimize_log_space:
            vec = np.array(x, dtype=float).copy()
            for idx in logit_idxs.values():
                vec[idx] = 1.0 / (1.0 + np.exp(-vec[idx]))
            return vector_to_params(vec, template_params, active_params, param_constraints)
        return vector_to_params(project_vec(x), template_params, active_params, param_constraints)

    def params_to_x(params: OrderedDict) -> np.ndarray:
        vec = params_to_vector(params, active_params)
        if opt_settings.optimize_log_space:
            x = vec.copy()
            for idx in logit_idxs.values():
                p = float(min(1.0 - 1e-9, max(1e-9, x[idx])))
                x[idx] = np.log(p / (1.0 - p))
            return project_vec(x)
        return project_vec(vec)

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
    run_group_root.mkdir(parents=True, exist_ok=True)

    # One sampled training subset per set, shared across all sites.
    set_seed = master_rng.randrange(0, 2**31 - 1)
    set_rng = random.Random(set_seed)

    # Every candidate in this fitting set uses exactly the same stochastic
    # realizations. Final full-output simulations use a separate fixed seed so
    # they provide an independent reproducible check of the selected parameters.
    optimization_model_seed = set_rng.randrange(0, 2**31 - 1)
    final_model_seed = set_rng.randrange(0, 2**31 - 1)

    sampled_training_df = sample_training_subset(
        full_training_df,
        percent=run_settings.resampling.train_percent,
        with_replacement=run_settings.resampling.sample_with_replacement,
        rng=set_rng,
    )

    tiller_count_diameter_path = Path(
        run_settings.paths.tiller_count_diameter_csv
    )
    if not tiller_count_diameter_path.exists():
        raise FileNotFoundError(
            f"Tiller count/diameter data not found: {tiller_count_diameter_path}"
        )
    tiller_count_diameter_data = pd.read_csv(tiller_count_diameter_path)
    required_count_columns = {"d", "N_alive"}
    missing_count_columns = required_count_columns.difference(
        tiller_count_diameter_data.columns
    )
    if missing_count_columns:
        raise ValueError(
            "Tiller count/diameter data is missing columns: "
            + ", ".join(sorted(missing_count_columns))
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

        # NEW OUTPUT STRUCTURE:
        #   <output_root>/<subdir>/<site>/set_001/
        site_root = run_group_root / site_tag
        set_root = site_root / f"set_{set_idx:03d}"
        set_root.mkdir(parents=True, exist_ok=True)

        run_param_file = set_root / "parameters.txt"
        run_config_snapshot = set_root / "config_snapshot.ini"
        sampled_training_csv = set_root / "sampled_training_data.csv"
        sampled_training_meta_csv = set_root / "sampled_training_metadata.csv"
        count_diameter_snapshot_csv = set_root / "tiller_count_diameter_data.csv"

        sampled_training_df.to_csv(sampled_training_csv, index=False)
        tiller_count_diameter_data.to_csv(
            count_diameter_snapshot_csv,
            index=False,
        )

        pd.DataFrame([{
            "set_idx": set_idx,
            "site": site_tag,
            "seed": set_seed,
            "optimization_model_seed": optimization_model_seed,
            "final_model_seed": final_model_seed,
            "n_rows_sampled": len(sampled_training_df),
            "n_rows_site_training": len(training_data),
            "n_rows_full": len(full_training_df),
            "train_percent": run_settings.resampling.train_percent,
            "sample_with_replacement": run_settings.resampling.sample_with_replacement,
        }]).to_csv(sampled_training_meta_csv, index=False)

        config_for_run = configparser.ConfigParser()

        for sec in base_config.sections():
            config_for_run[sec] = dict(base_config[sec])

        if config_for_run.has_option("Paths", "param_file"):
            config_for_run.remove_option("Paths", "param_file")

        config_for_run.set("Paths", "output_dir", str(set_root))
        config_for_run.set("Paths", "training_csv", str(sampled_training_csv))
        config_for_run.set(
            "Paths",
            "tiller_count_diameter_csv",
            str(count_diameter_snapshot_csv),
        )
        write_config_snapshot(config_for_run, run_config_snapshot)

        template_params = initialize_random_parameter_file(
            run_param_file,
            run_settings.active_params,
            run_settings.param_constraints,
        )

        axis_limits = fixed_axis_limits_from_observed(training_diameters, bins=30)

        project_vec, x_to_model_params, params_to_x = build_projection_functions(
            run_settings.optimization,
            run_settings.active_params,
            template_params,
            run_settings.param_constraints,
        )

        cpp_outdir = set_root / "simulation_outputs"
        cpp_outdir.mkdir(parents=True, exist_ok=True)

        frames_dir = set_root / "mean_diameter_frames"
        opt_csv_path = set_root / "optimization_results.csv"
        pop_csv_path = set_root / "population_results.csv"

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
            write_parameter_snapshot(params, set_root)

            tussock_model(
                config_path=run_config_snapshot,
                output_dir=cpp_outdir,
                output_mode="summary",
                project_root=run_settings.project_root,
                simulation_seed=optimization_model_seed,
            )

            loss, loss_components = diameter_objective(
                sim_outdir=cpp_outdir,
                num_sims=int(base_config.get("Tussock Model", "nsims")),
                iteration_label=eval_label,
                training_data=training_data,
                tiller_count_diameter_data=tiller_count_diameter_data,
                frames_dir=frames_dir,
                axis_limits=axis_limits,
                constraints=run_settings.constraints,
                plotting=run_settings.plotting,
            )

            if (
                run_settings.plotting.plot_every
                and run_settings.plotting.plot_every > 0
                and eval_label % run_settings.plotting.plot_every == 0
            ):
                frame_labels.append(eval_label)

            write_optimization_results(
                params,
                run_settings.active_params,
                loss,
                eval_label,
                opt_csv_path,
                loss_components=loss_components,
            )

            sim_df = read_sim_summaries(
                cpp_outdir,
                num_sims=int(base_config.get("Tussock Model", "nsims")),
            )

            write_population_results(
                sim_df,
                eval_label,
                pop_csv_path,
                run_settings.constraints.overgrown_radius_threshold,
            )

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
                trial_params = random_initial_parameters(
                    run_settings.active_params,
                    run_settings.param_constraints,
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
                        run_settings.optimization.step_frac * mag
                        if mag > 1e-12
                        else run_settings.optimization.step_abs
                    )

                sigma0 = float(np.median(step_scales)) if step_scales else run_settings.optimization.cma_sigma
                sigma0 = max(1e-6, sigma0)

            remaining_budget = max(0, run_settings.optimization.max_evals - eval_label)

            if remaining_budget > 0 and x0.size > 0:
                best_x, _, _ = pycma_optimize(
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

        final_params = (
            best_seen["params"]
            if best_seen["params"] is not None
            else OrderedDict(template_params)
        )

        print(f"[set {set_idx:03d} | {site_tag}] best loss: {best_seen['loss']:.6g}")

        for k in run_settings.active_params:
            print(f"  {k}={final_params[k]}")

        write_parameter_file(final_params, run_param_file)
        write_parameter_snapshot(final_params, set_root)

        final_sims_dir = set_root / "final_sims"
        final_sims_dir.mkdir(parents=True, exist_ok=True)

        tussock_model(
            config_path=run_config_snapshot,
            output_dir=final_sims_dir,
            output_mode="full",
            project_root=run_settings.project_root,
            simulation_seed=final_model_seed,
        )

        final_summary_df = read_sim_summaries(
            final_sims_dir,
            num_sims=int(base_config.get("Tussock Model", "nsims")),
        )

        final_pop_csv_path = set_root / "final_population_results.csv"

        if final_pop_csv_path.exists():
            final_pop_csv_path.unlink()

        write_population_results(
            final_summary_df,
            eval_label,
            final_pop_csv_path,
            run_settings.constraints.overgrown_radius_threshold,
        )

        if run_settings.plotting.plot_every and run_settings.plotting.plot_every > 0:
            animate_fitting(
                frames_dir,
                frame_labels,
                set_root / "diameter_dist_fitting.gif",
            )

        print(f"Completed site {site_tag}, set {set_idx:03d}")


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
        "common_random_numbers": True,
        "n_sets": run_settings.resampling.n_sets,
        "train_percent": run_settings.resampling.train_percent,
        "sample_with_replacement": run_settings.resampling.sample_with_replacement,
        "subdir": run_settings.paths.subdir,
        "output_root": str(output_root),
        "output_structure": "<output_root>/<subdir>/<site>/set_###",
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