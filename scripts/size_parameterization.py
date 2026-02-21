#!/usr/bin/env python3
# parameterize_tussock.py
#
# Quantile-bin parameterization by "diam" instead of by site.
#
# NEW:
#   - Split training data into diameter quantile bins (based on diam column)
#   - Quantile split is controlled by CLI args:
#       --diam_quantiles N
#       OR
#       --diam_quantile_edges 0,0.25,0.5,0.75,1
#
# ALSO:
#   - Exclude any training rows where diam == 0 (after numeric coercion).
#
# "Only constrain weights" + minimal model-valid constraints to avoid degenerate
# "never reproduces / 1 tiller forever" regimes.
#
# Constrained:
#   - fr, fr_repr, c_repr in [0,1]  (weights/fractions)
# Minimal model-valid (not “biological prior”; required to preserve intended monotonicity / meaning):
#   - ks >= 0   (so survival decreases with distance in bs - ks*distance)
#   - kr >= 0   (so reproduction decreases with distance in br - kr*distance)
#   - c_daughter >= 0 (carbon cost can’t be negative)
#
# Everything else: unconstrained real-valued in linear-space optimization,
# or left linear in log-space optimization (we ONLY logit-code the weights here).
#
# NOTE: This script expects your C++ reads:
#   ks, kr, bs, br, g_offset, c_space, c_repro(or c_repr), fr, fr_repr, c_daughter
#
# If your C++ now uses c_repro as the key, keep the Python key as c_repr
# but write out "c_repr=" and rely on your C++ back-compat mapping,
# OR rename to c_repro throughout. Here we keep c_repr to match your existing file.

import argparse
import configparser
import csv
import math
import os
import random
import subprocess
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

try:
    import seaborn as sns  # optional
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tussock model parameterization (Nelder–Mead; soft constraints + fit)"
    )

    # ---- Quantile-based parameterization (replaces site-based mode) ----
    parser.add_argument(
        "--diam_quantiles",
        type=int,
        default=None,
        help=(
            "Number of quantile bins to split training data by the 'diam' column. "
            "Example: --diam_quantiles 4 (quartiles). "
            "If omitted (and --diam_quantile_edges omitted), runs a single bin (ALL)."
        ),
    )

    parser.add_argument(
        "--diam_quantile_edges",
        type=str,
        default=None,
        help=(
            "Explicit quantile edges in [0,1] as comma-separated list. "
            "Example: --diam_quantile_edges 0,0.25,0.5,0.75,1 "
            "If provided, overrides --diam_quantiles."
        ),
    )

    parser.add_argument("--max_evals", type=int, default=200, help="Max objective evaluations per bin.")
    parser.add_argument("--n_init", type=int, default=25, help="Number of random initial trials per bin.")
    parser.add_argument("--tol_f", type=float, default=1e-3, help="Stop if simplex loss spread < tol_f.")
    parser.add_argument("--tol_x", type=float, default=1e-3, help="Stop if simplex size < tol_x.")

    parser.add_argument(
        "--init_log10_span",
        type=float,
        default=1.0,
        help=(
            "Random init multiplicative span around current values (log10-space) for scale-like params. "
            "Each param is multiplied by 10**U(-span, +span). "
            "Intercept-like params (bs/br) are jittered additively instead."
        ),
    )

    parser.add_argument("--step_frac", type=float, default=0.2)
    parser.add_argument("--step_abs", type=float, default=0.3)

    parser.add_argument("--extinction_weight", type=float, default=0.0)

    parser.add_argument("--constraint_year", type=int, default=25)
    parser.add_argument("--min_alive_tillers", type=int, default=25)
    parser.add_argument("--radius_cap", type=float, default=2.5)
    parser.add_argument("--constraint_pass_frac", type=float, default=0.8)
    parser.add_argument("--alive_overflow_threshold", type=int, default=400)

    # If set: ONLY weights are logit-coded; everything else stays linear.
    # (This is different from your old log-space that exp()'d positive-only params.)
    parser.add_argument("--optimize_log_space", action="store_true")

    parser.add_argument("--print_fail_breakdown", action="store_true")

    parser.add_argument("--plot_every", type=int, default=10)
    parser.add_argument("--plot_kde", action="store_true")

    # Stall expansion inside Nelder–Mead
    parser.add_argument("--stall_expand_after", type=int, default=30)
    parser.add_argument("--stall_expand_factor", type=float, default=2.0)

    return parser.parse_args()


def get_config():
    config = configparser.ConfigParser()
    config.read("parameterization.ini")
    return config


def _safe_makedirs(dirpath: str):
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)


def get_param_file_path(config: configparser.ConfigParser) -> str:
    return config.get("Parameterization", "param_file", fallback=os.path.join("parameters", "parameters.txt"))


def initialize_random_parameters_file(param_file: str, config: configparser.ConfigParser) -> OrderedDict:
    """
    Sensible random init that avoids the degenerate "no reproduction forever" regimes.
    This is NOT imposing heavy priors; it just seeds reasonable magnitudes.

    - ks, kr: positive log-uniform
    - bs, br: uniform(-3, 3)
    - g_offset: uniform(-50, 50)
    - c_space: positive log-uniform (can be >1; C++ clamps to [0,1] currently, but we keep it >=0 here)
    - c_repr, fr, fr_repr: uniform(0,1)
    - c_daughter: log-uniform 1e-3 .. 1
    """
    raw_names = config.get(
        "Parameterization",
        "param_names",
        fallback="ks,kr,bs,br,g_offset,c_space,c_repr,fr,fr_repr,c_daughter",
    )
    names = [s.strip() for s in raw_names.split(",") if s.strip()]
    if not names:
        names = ["ks", "kr", "bs", "br", "g_offset", "c_space", "c_repr", "fr", "fr_repr", "c_daughter"]

    def logu(lo, hi):
        # log-uniform in base10
        return 10 ** random.uniform(math.log10(lo), math.log10(hi))

    params = OrderedDict()
    for k in names:
        if k == "g_offset":
            params[k] = float(random.uniform(-50.0, 50.0))
        elif k in {"fr", "fr_repr", "c_repr"}:
            params[k] = float(random.uniform(0.0, 1.0))
        elif k in {"bs", "br"}:
            params[k] = float(random.uniform(-3.0, 3.0))
        elif k == "ks":
            params[k] = float(logu(1e-3, 100.0))  # ks >= 0
        elif k == "kr":
            params[k] = float(logu(1e-3, 100.0))  # kr >= 0
        elif k == "c_space":
            params[k] = float(logu(1e-3, 200.0))  # >=0 (your C++ clamps to [0,1])
        elif k == "c_daughter":
            params[k] = float(logu(1e-3, 1.0))    # >=0
        else:
            # default: mild positive scale
            params[k] = float(logu(1e-3, 10.0))

    _safe_makedirs(os.path.dirname(param_file))
    with open(param_file, "w") as f:
        f.write("# Auto-generated on first run (random init)\n")
        for k, v in params.items():
            f.write(f"{k}={v}\n")
    return params


def read_parameters_file(param_file: str, config: configparser.ConfigParser) -> OrderedDict:
    if not os.path.exists(param_file):
        print(f"[init] Parameter file not found; creating random init at: {param_file}")
        return initialize_random_parameters_file(param_file, config)

    params = OrderedDict()
    with open(param_file, "r") as f:
        for raw in f:
            line = raw.strip()
            if (not line) or line.startswith("#") or ("=" not in line):
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()

            kl = k.lower()
            if kl in {"gmin", "gmax", "default_min", "default_max"}:
                continue
            if kl.endswith("_min") or kl.endswith("_max"):
                continue

            params[k] = float(v)

    if len(params) == 0:
        raise ValueError(f"No parameters parsed from {param_file}. Expected key=value lines.")
    return params


def _ensure_params_present(params: OrderedDict, config: configparser.ConfigParser) -> OrderedDict:
    # defaults
    if "g_offset" not in params:
        params["g_offset"] = float(config.get("Parameterization", "g_offset_default", fallback="0.0"))
    if "c_space" not in params:
        params["c_space"] = float(config.get("Parameterization", "c_space_default", fallback="1.0"))
    if "c_repr" not in params:
        params["c_repr"] = float(config.get("Parameterization", "c_repr_default", fallback="0.5"))
    if "fr" not in params:
        params["fr"] = float(config.get("Parameterization", "fr_default", fallback="0.5"))
    if "fr_repr" not in params:
        params["fr_repr"] = float(config.get("Parameterization", "fr_repr_default", fallback="0.0"))
    if "c_daughter" not in params:
        params["c_daughter"] = float(config.get("Parameterization", "c_daughter_default", fallback="0.0"))

    for k in ["ks", "kr", "bs", "br"]:
        if k not in params:
            params[k] = float(config.get("Parameterization", f"{k}_default", fallback="1.0"))

    # ONLY enforce weights + minimal model-valid constraints
    params["ks"] = float(max(0.0, params["ks"]))
    params["kr"] = float(max(0.0, params["kr"]))
    params["c_daughter"] = float(max(0.0, params["c_daughter"]))

    for w in ["fr", "fr_repr", "c_repr"]:
        params[w] = float(min(1.0, max(0.0, params[w])))

    wanted = ["ks", "kr", "bs", "br", "g_offset", "c_space", "c_repr", "fr", "fr_repr", "c_daughter"]
    out = OrderedDict()
    for k in wanted:
        if k in params:
            out[k] = params[k]
    for k, v in params.items():
        if k not in out:
            out[k] = v
    return out


def write_parameters_to_paths(parameters: OrderedDict, primary_param_file: str, site_outdir: str):
    _safe_makedirs(os.path.dirname(primary_param_file))
    _safe_makedirs(site_outdir)

    site_copy = os.path.join(site_outdir, "parameters.txt")

    for path in (primary_param_file, site_copy):
        _safe_makedirs(os.path.dirname(path))
        with open(path, "w") as f:
            for k, v in parameters.items():
                f.write(f"{k}={v}\n")


def params_to_vector(params: OrderedDict) -> np.ndarray:
    return np.array(list(params.values()), dtype=float)


def vector_to_params(vec: np.ndarray, param_names) -> OrderedDict:
    return OrderedDict((k, float(v)) for k, v in zip(param_names, vec))


def tussock_model(config: configparser.ConfigParser, output_mode: str):
    num_sims = int(config.get("Tussock Model", "nsims"))
    outdir = config.get("Tussock Model", "filepath")
    num_threads = int(config.get("Tussock Model", "nthreads"))
    sim_time = int(config.get("Tussock Model", "nyears"))

    mode_flag = 1 if output_mode.lower().startswith("s") else 0
    cpp_input = f"{sim_time}\n{num_sims}\n{outdir}\n{num_threads}\n{mode_flag}\n"

    exe = os.path.abspath(os.path.join("model", "tussock_model"))

    if not os.path.exists(exe):
        raise FileNotFoundError(
            f"Expected binary not found: {exe}\n"
            f"Did `cd model && make` produce model/tussock_model?"
        )
    if not os.access(exe, os.X_OK):
        raise PermissionError(f"Binary exists but is not executable: {exe}")

    script_dir = os.path.abspath(os.path.dirname(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    p = subprocess.Popen(
        [exe],
        cwd=project_root,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = p.communicate(input=cpp_input)
    if p.returncode != 0:
        raise RuntimeError(
            f"tussock_model failed (code={p.returncode})\n"
            f"exe: {exe}\n"
            f"stdout:\n{out}\n"
            f"stderr:\n{err}\n"
        )


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


def barrier_loss(fail_stats, constraint_year, min_alive, r_cap, overflow_threshold):
    T = float(max(1, constraint_year))

    v_alive = []
    v_rad = []
    v_over = []
    v_ext = []
    v_missing = []

    v_surv_time = []
    v_alive_end = []

    for s in fail_stats:
        alive_y = int(s.get("alive_y", 0))
        overflow_t = s.get("overflow_t", None)
        extinct_t = s.get("extinct_t", None)
        missing_year = bool(s.get("missing_year", False))

        rmax_y = float(s.get("rmax_y", r_cap))
        if not np.isfinite(rmax_y):
            rmax_y = r_cap

        if alive_y <= 0:
            v_rad.append(0.0)
        else:
            v_rad.append(max(0.0, (rmax_y - r_cap) / max(1e-12, float(r_cap))))

        v_alive.append(max(0.0, (min_alive - alive_y) / max(1.0, float(min_alive))))

        if overflow_t is None:
            v_over.append(0.0)
        else:
            v_over.append(max(0.0, (T - float(min(constraint_year, overflow_t))) / T))

        if extinct_t is None:
            v_ext.append(0.0)
        else:
            v_ext.append(max(0.0, (T - float(min(constraint_year, extinct_t))) / T))

        v_missing.append(1.0 if missing_year else 0.0)

        ft = int(s.get("final_t", -1))
        if ft < 0:
            v_surv_time.append(1.0)
        else:
            v_surv_time.append(max(0.0, (T - min(T, float(ft))) / T))

        alive_end = int(s.get("alive_final", 0))
        v_alive_end.append(0.0 if alive_end > 0 else 1.0)

    w_alive = 1.0
    w_rad = 1.0
    w_over = 2.0
    w_ext = 2.0
    w_missing = 5.0

    w_surv_time = 3.0
    w_alive_end = 3.0

    return (
        w_alive * float(np.mean(v_alive))
        + w_rad * float(np.mean(v_rad))
        + w_over * float(np.mean(v_over))
        + w_ext * float(np.mean(v_ext))
        + w_missing * float(np.mean(v_missing))
        + w_surv_time * float(np.mean(v_surv_time))
        + w_alive_end * float(np.mean(v_alive_end))
    )


def read_sim_summaries(sim_outdir: str, num_sims: int) -> pd.DataFrame:
    summary_dir = os.path.join(sim_outdir, "summaries")
    if not os.path.isdir(summary_dir):
        df = pd.DataFrame(
            {
                "sim_id": np.arange(num_sims, dtype=int),
                "final_t": -1,
                "final_diameter": np.nan,
                "alive_y": 0,
                "rmax_y": np.inf,
                "overflow_t": -1,
                "extinct_t": -1,
                "missing_year": 1,
                "alive_final": 0,
                "LeafArea": np.nan,
            }
        )
        return df

    dfs = []
    for i in range(num_sims):
        fn = os.path.join(summary_dir, f"summary_{i}.csv")
        if os.path.exists(fn):
            dfs.append(pd.read_csv(fn))
    if not dfs:
        return pd.DataFrame(
            columns=[
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
        )

    df = pd.concat(dfs, ignore_index=True)

    df = df.drop_duplicates(subset=["sim_id"], keep="last").set_index("sim_id")
    df = df.reindex(range(num_sims))
    df = df.reset_index()

    if "LeafArea" not in df.columns:
        df["LeafArea"] = np.nan
    if "alive_final" not in df.columns:
        df["alive_final"] = 0
    if "final_t" not in df.columns:
        df["final_t"] = -1

    df["missing_year"] = df["missing_year"].fillna(1).astype(int)
    df["alive_y"] = df["alive_y"].fillna(0).astype(int)
    df["alive_final"] = df["alive_final"].fillna(0).astype(int)
    df["rmax_y"] = df["rmax_y"].fillna(np.inf)
    df["final_diameter"] = df["final_diameter"].astype(float)
    df["overflow_t"] = df["overflow_t"].fillna(-1).astype(int)
    df["extinct_t"] = df["extinct_t"].fillna(-1).astype(int)
    df["final_t"] = df["final_t"].fillna(-1).astype(int)

    df["LeafArea"] = pd.to_numeric(df["LeafArea"], errors="coerce")
    return df


def diameter_objective(
    config,
    iteration_label,
    training_data,
    frames_dir,
    axis_limits,
    extinction_weight,
    constraint_year,
    min_alive_tillers,
    radius_cap,
    constraint_pass_frac,
    alive_overflow_threshold,
    print_fail_breakdown,
    plot_every,
    plot_kde,
):
    num_sims = int(config.get("Tussock Model", "nsims"))
    sim_filepath = config.get("Tussock Model", "filepath")

    training_data = training_data.copy()
    training_data["field_davg"] = pd.to_numeric(training_data["diam"], errors="coerce")
    training_diameters = training_data["field_davg"].dropna().values

    if training_diameters.size == 0:
        return float("inf")

    obs_std = float(np.std(training_diameters)) if training_diameters.size > 1 else 1.0
    if not np.isfinite(obs_std) or obs_std <= 0:
        obs_std = 1.0

    df = read_sim_summaries(sim_filepath, num_sims=num_sims)

    sim_diameters_final = df["final_diameter"].to_numpy(dtype=float)
    sim_diameters_final = sim_diameters_final[np.isfinite(sim_diameters_final)]

    extinct_final = int((df["alive_final"].to_numpy(dtype=int) == 0).sum())

    ok = (
        (df["missing_year"].to_numpy(dtype=int) == 0)
        & (df["alive_y"].to_numpy(dtype=int) >= int(min_alive_tillers))
        & (df["rmax_y"].to_numpy(dtype=float) <= float(radius_cap))
    )
    pass_count = int(ok.sum())
    pass_needed = int(math.ceil(constraint_pass_frac * max(1, num_sims)))

    fail_stats = []
    for row in df.itertuples(index=False):
        overflow_t = None if int(row.overflow_t) < 0 else int(row.overflow_t)
        extinct_t = None if int(row.extinct_t) < 0 else int(row.extinct_t)
        missing_year = bool(int(row.missing_year))

        fail_stats.append(
            {
                "alive_y": 0 if missing_year else int(row.alive_y),
                "rmax_y": float("inf") if missing_year else float(row.rmax_y),
                "overflow_t": overflow_t,
                "extinct_t": extinct_t,
                "missing_year": missing_year,
                "alive_final": int(getattr(row, "alive_final", 0)),
                "final_t": int(getattr(row, "final_t", -1)),
            }
        )

    fit_loss = wasserstein_distance_1d(training_diameters, sim_diameters_final)

    bar = barrier_loss(
        fail_stats=fail_stats,
        constraint_year=constraint_year,
        min_alive=min_alive_tillers,
        r_cap=radius_cap,
        overflow_threshold=alive_overflow_threshold,
    )

    extinct_frac = extinct_final / max(1, num_sims)
    ext_term = float(extinction_weight) * extinct_frac * obs_std

    leaf = df["LeafArea"].to_numpy(dtype=float)
    leaf_finite = np.isfinite(leaf)
    leaf_bad = leaf_finite & ((leaf <= 0.0) | (leaf >= 2000.0))
    denom = max(1, int(leaf_finite.sum()))
    leaf_bad_frac = float(leaf_bad.sum()) / float(denom)

    leaf_penalty_weight = 10.0
    leaf_penalty = leaf_penalty_weight * obs_std * leaf_bad_frac

    if pass_count < pass_needed:
        loss = (1e3 * obs_std) * (1.0 + bar) + ext_term + leaf_penalty
        if print_fail_breakdown:
            missing_n = int((df["missing_year"].to_numpy(dtype=int) == 1).sum())
            min_alive_fail = int(((df["missing_year"] == 0) & (df["alive_y"] < min_alive_tillers)).sum())
            radius_fail = int(((df["missing_year"] == 0) & (df["rmax_y"] > radius_cap)).sum())
            print(
                f"[HARD_REGION] eval={iteration_label} pass={pass_count}/{num_sims} need>={pass_needed} | "
                f"missing_year={missing_n}, min_alive={min_alive_fail}, radius={radius_fail} | bar={bar:.4g} "
                f"| leaf_bad={leaf_bad_frac:.2%}"
            )
    else:
        loss = float(fit_loss) + (10.0 * obs_std) * float(bar) + ext_term + leaf_penalty

    do_plot = (plot_every is not None) and (int(plot_every) > 0) and (iteration_label % int(plot_every) == 0)
    if do_plot:
        os.makedirs(frames_dir, exist_ok=True)
        fig, ax = plt.subplots()

        if plot_kde and _HAS_SNS:
            sns.kdeplot(training_diameters, label="Observed", linewidth=1, ax=ax)
            if sim_diameters_final.size > 0:
                sns.kdeplot(sim_diameters_final, label="Modeled", linewidth=1, ax=ax)
        else:
            ax.hist(training_diameters, bins=30, density=True, alpha=0.4, label="Observed")
            if sim_diameters_final.size > 0:
                ax.hist(sim_diameters_final, bins=30, density=True, alpha=0.4, label="Modeled")

        if axis_limits.get("xlim") is None:
            obs = training_diameters[np.isfinite(training_diameters)]
            xmin, xmax = float(np.min(obs)), float(np.max(obs))
            span = xmax - xmin
            pad = 0.05 * span if span > 0 else 0.5
            axis_limits["xlim"] = (xmin - pad, xmax + pad)
        ax.set_xlim(*axis_limits["xlim"])

        if axis_limits.get("ylim") is None:
            ymin, ymax = ax.get_ylim()
            axis_limits["ylim"] = (0.0, max(1.0, 1.1 * float(ymax)))
        ax.set_ylim(*axis_limits["ylim"])

        ax.legend()
        ax.set_title(
            f"Iter: {iteration_label} | loss={loss:.3g} | pass={pass_count}/{num_sims} "
            f"| bar={bar:.3g} | extinct_final={extinct_final}/{num_sims} | leaf_bad={leaf_bad_frac:.2%}"
        )
        ax.set_xlabel("Tussock Diameter")

        frame_filename = os.path.join(frames_dir, f"Mean_Tuss_diameter_iteration_{iteration_label}.png")
        plt.savefig(frame_filename, dpi=200)
        plt.close(fig)

    return float(loss)


def animate_fitting(frames_dir, iteration_labels, outfilename):
    frames = []
    for lab in iteration_labels:
        fn = os.path.join(frames_dir, f"Mean_Tuss_diameter_iteration_{lab}.png")
        if os.path.exists(fn):
            frames.append(Image.open(fn))

    if not frames:
        return

    frames[0].save(
        outfilename,
        save_all=True,
        append_images=frames[1:],
        duration=75,
        loop=0,
    )

    for fn in os.listdir(frames_dir):
        os.remove(os.path.join(frames_dir, fn))
    os.rmdir(frames_dir)


def write_optimization_results(parameters: OrderedDict, loss: float, iteration_label, out_csv_path: str):
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    file_exists = os.path.exists(out_csv_path)

    with open(out_csv_path, "a", newline="") as csvfile:
        fieldnames = list(parameters.keys()) + ["loss", "iteration"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({**parameters, "loss": float(loss), "iteration": iteration_label})


def nelder_mead(
    f,
    x0,
    step,
    max_evals,
    tol_f,
    tol_x,
    project_fn,
    init_simplex=None,
    stall_expand_after=30,
    stall_expand_factor=2.0,
):
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5

    n = x0.size

    if init_simplex is None:
        simplex = [project_fn(x0)]
        for i in range(n):
            xi = x0.copy()
            xi[i] += step[i]
            simplex.append(project_fn(xi))
    else:
        simplex = [project_fn(np.array(x, dtype=float)) for x in init_simplex]
        if len(simplex) < n + 1:
            base = simplex[0] if simplex else project_fn(x0)
            simplex = [base]
            for i in range(n):
                xi = base.copy()
                xi[i] += step[i]
                simplex.append(project_fn(xi))
        simplex = simplex[: n + 1]

    fvals = []
    evals = 0
    for x in simplex:
        fvals.append(f(x))
        evals += 1
        if evals >= max_evals:
            order = np.argsort(fvals)
            return simplex[order[0]], fvals[order[0]], evals

    best_so_far = min(fvals)
    stall = 0

    while evals < max_evals:
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]

        if fvals[0] < best_so_far - 1e-12:
            best_so_far = fvals[0]
            stall = 0
        else:
            stall += 1

        f_spread = max(fvals) - min(fvals)
        best = simplex[0]
        sizes = [np.linalg.norm(x - best) for x in simplex[1:]]
        x_spread = max(sizes) if sizes else 0.0

        if f_spread < tol_f and x_spread < tol_x:
            break

        if stall_expand_after is not None and stall_expand_after > 0 and stall >= stall_expand_after:
            x_best = simplex[0]
            for i in range(1, n + 1):
                simplex[i] = project_fn(x_best + float(stall_expand_factor) * (simplex[i] - x_best))
                fvals[i] = f(simplex[i])
                evals += 1
                if evals >= max_evals:
                    break
            stall = 0
            continue

        x_bar = np.mean(simplex[:-1], axis=0)
        x_worst = simplex[-1]

        x_r = project_fn(x_bar + alpha * (x_bar - x_worst))
        f_r = f(x_r)
        evals += 1

        if fvals[0] <= f_r < fvals[-2]:
            simplex[-1] = x_r
            fvals[-1] = f_r
            continue

        if f_r < fvals[0]:
            x_e = project_fn(x_bar + gamma * (x_r - x_bar))
            f_e = f(x_e)
            evals += 1

            if f_e < f_r:
                simplex[-1] = x_e
                fvals[-1] = f_e
            else:
                simplex[-1] = x_r
                fvals[-1] = f_r
            continue

        if f_r < fvals[-1]:
            x_c = project_fn(x_bar + rho * (x_r - x_bar))
        else:
            x_c = project_fn(x_bar + rho * (x_worst - x_bar))

        f_c = f(x_c)
        evals += 1

        if f_c < fvals[-1]:
            simplex[-1] = x_c
            fvals[-1] = f_c
            continue

        x_best = simplex[0]
        new_simplex = [x_best]
        new_fvals = [fvals[0]]
        for i in range(1, n + 1):
            x_s = project_fn(x_best + sigma * (simplex[i] - x_best))
            f_s = f(x_s)
            evals += 1
            new_simplex.append(x_s)
            new_fvals.append(f_s)
            if evals >= max_evals:
                break

        simplex = new_simplex
        fvals = new_fvals

    order = np.argsort(fvals)
    return simplex[order[0]], fvals[order[0]], evals


def sample_random_params_around(base_params: OrderedDict, log10_span: float) -> OrderedDict:
    """
    Jitter around template parameters without imposing hard bounds
    (except weights + minimal model-valid constraints).
    """
    def logmul(v, span):
        basep = float(max(1e-12, abs(v)))
        u = random.uniform(-span, span)
        return basep * (10 ** u)

    out = OrderedDict()
    for k, base in base_params.items():
        if k == "g_offset":
            out[k] = float(base + random.uniform(-60.0, 60.0))
            continue

        if k in {"bs", "br"}:
            out[k] = float(base + random.uniform(-3.0, 3.0))
            continue

        # weights/fractions (bounded)
        if k in {"fr", "fr_repr", "c_repr"}:
            out[k] = float(min(1.0, max(0.0, base + random.uniform(-0.5, 0.5))))
            continue

        if k in {"ks", "kr"}:
            out[k] = float(logmul(base if base != 0 else 1.0, log10_span))
            continue

        if k == "c_daughter":
            out[k] = float(logmul(base if base != 0 else 1e-2, log10_span))
            continue

        # everything else: multiplicative jitter, preserving sign
        mag = logmul(base if base != 0 else 1.0, log10_span)
        out[k] = float(math.copysign(mag, base if base != 0 else 1.0))

    # minimal model-valid constraints
    if "ks" in out:
        out["ks"] = float(max(0.0, out["ks"]))
    if "kr" in out:
        out["kr"] = float(max(0.0, out["kr"]))
    if "c_daughter" in out:
        out["c_daughter"] = float(max(0.0, out["c_daughter"]))
    return out


def main():
    args = parse_args()
    config = get_config()

    full_training_df = pd.read_csv("./input_data/tussock_allometry.csv")

    # ---- Quantile bin setup (by diam), excluding diam==0 ----
    df_all = full_training_df.copy()
    df_all["diam"] = pd.to_numeric(df_all["diam"], errors="coerce")

    # Exclude NaNs and any rows with diam == 0
    df_q = df_all.dropna(subset=["diam"]).copy()
    df_q = df_q[df_q["diam"] != 0].copy()

    if df_q.empty:
        raise ValueError("No training rows remain after filtering to numeric diam and excluding diam==0.")

    # Determine quantile edges
    if args.diam_quantile_edges is not None:
        edges = [float(x.strip()) for x in args.diam_quantile_edges.split(",") if x.strip() != ""]
        if len(edges) < 2:
            raise ValueError("--diam_quantile_edges must contain at least two comma-separated values.")
        if edges[0] != 0.0 or edges[-1] != 1.0:
            raise ValueError("--diam_quantile_edges must start with 0 and end with 1.")
        if any((e < 0.0 or e > 1.0) for e in edges):
            raise ValueError("--diam_quantile_edges values must lie in [0,1].")
        if any(edges[i] >= edges[i + 1] for i in range(len(edges) - 1)):
            raise ValueError("--diam_quantile_edges must be strictly increasing.")
        q_edges = edges
    else:
        if args.diam_quantiles is None:
            # Default: single bin (equivalent to old ALL behavior)
            q_edges = [0.0, 1.0]
        else:
            if args.diam_quantiles < 1:
                raise ValueError("--diam_quantiles must be >= 1.")
            q_edges = list(np.linspace(0.0, 1.0, args.diam_quantiles + 1))

    # Build bins using qcut with explicit quantile edges
    # duplicates='drop' prevents crashes when diam has many ties
    bin_codes, bin_edges = pd.qcut(
        df_q["diam"],
        q=q_edges,
        labels=False,
        retbins=True,
        duplicates="drop",
    )

    df_q["_diam_bin"] = bin_codes.astype(int)

    # Construct bin metadata for naming
    n_bins = int(df_q["_diam_bin"].max()) + 1
    group_list = []
    for b in range(n_bins):
        lo = float(bin_edges[b])
        hi = float(bin_edges[b + 1])
        tag = f"Q{b+1:02d}_diam_{lo:.3g}_to_{hi:.3g}"
        group_list.append((b, tag, lo, hi))

    primary_param_file = os.path.abspath(get_param_file_path(config))
    template_params = read_parameters_file(primary_param_file, config)
    template_params = _ensure_params_present(template_params, config)

    param_names = list(template_params.keys())

    weight_names = ["fr", "fr_repr", "c_repr"]
    idx_weights = {k: (param_names.index(k) if k in param_names else None) for k in weight_names}

    idx_ks = param_names.index("ks") if "ks" in param_names else None
    idx_kr = param_names.index("kr") if "kr" in param_names else None
    idx_cd = param_names.index("c_daughter") if "c_daughter" in param_names else None

    nyears = int(config.get("Tussock Model", "nyears"))
    if nyears < args.constraint_year:
        raise ValueError(
            f"nyears={nyears} < constraint_year={args.constraint_year}. Set nyears >= constraint_year."
        )

    def _clip_weights_linear(x: np.ndarray) -> np.ndarray:
        for k, idx in idx_weights.items():
            if idx is not None:
                x[idx] = float(np.clip(x[idx], 0.0, 1.0))
        return x

    def project_vec(x: np.ndarray) -> np.ndarray:
        x = np.array(x, dtype=float)
        x[~np.isfinite(x)] = 0.0

        if args.optimize_log_space:
            # weights are logits; keep them bounded for stability
            for k, idx in idx_weights.items():
                if idx is not None:
                    x[idx] = float(np.clip(x[idx], -10.0, 10.0))

            # minimal model-valid constraints in x-space (linear params)
            if idx_ks is not None:
                x[idx_ks] = float(max(0.0, x[idx_ks]))
            if idx_kr is not None:
                x[idx_kr] = float(max(0.0, x[idx_kr]))
            if idx_cd is not None:
                x[idx_cd] = float(max(0.0, x[idx_cd]))

            return x

        # linear-space: clamp only weights + minimal model-valid
        x = _clip_weights_linear(x)
        if idx_ks is not None:
            x[idx_ks] = float(max(0.0, x[idx_ks]))
        if idx_kr is not None:
            x[idx_kr] = float(max(0.0, x[idx_kr]))
        if idx_cd is not None:
            x[idx_cd] = float(max(0.0, x[idx_cd]))
        return x

    def x_to_model_params(x: np.ndarray) -> OrderedDict:
        x = np.array(x, dtype=float)

        if args.optimize_log_space:
            vec = x.copy()

            # decode weights from logits
            for k, idx in idx_weights.items():
                if idx is not None:
                    vec[idx] = 1.0 / (1.0 + np.exp(-vec[idx]))

            params = vector_to_params(vec, param_names)
        else:
            vec = project_vec(x)
            params = vector_to_params(vec, param_names)

        # enforce minimal model-valid constraints again (insurance)
        if "ks" in params:
            params["ks"] = float(max(0.0, params["ks"]))
        if "kr" in params:
            params["kr"] = float(max(0.0, params["kr"]))
        if "c_daughter" in params:
            params["c_daughter"] = float(max(0.0, params["c_daughter"]))

        for k in weight_names:
            if k in params:
                params[k] = float(min(1.0, max(0.0, params[k])))

        return params

    def params_to_x(params: OrderedDict) -> np.ndarray:
        vec = params_to_vector(params)

        if args.optimize_log_space:
            x = vec.copy()

            # encode weights as logits
            for k, idx in idx_weights.items():
                if idx is not None:
                    p = float(min(1.0, max(0.0, x[idx])))
                    p = min(1.0 - 1e-9, max(1e-9, p))
                    x[idx] = np.log(p / (1.0 - p))

            # everything else stays linear
            return x

        return vec

    for b, group_tag, lo, hi in group_list:
        print("\n====================================")
        print(f"   PARAMETERIZING DIAM QUANTILE BIN: {group_tag}")
        print("====================================\n")

        training_data = df_q[df_q["_diam_bin"] == b].copy()
        site_tag = group_tag

        base_outdir = config.get("Parameterization", "outdir", fallback="parameterization_outputs")
        site_outdir = os.path.join(base_outdir, site_tag)
        os.makedirs(site_outdir, exist_ok=True)

        cpp_outdir = os.path.join(site_outdir, "simulation_outputs")
        os.makedirs(cpp_outdir, exist_ok=True)
        config.set("Tussock Model", "filepath", cpp_outdir)

        frames_dir = os.path.join(site_outdir, "mean_diameter_frames")
        axis_limits = {"xlim": None, "ylim": None}

        opt_csv_path = os.path.join(site_outdir, "optimization_results.csv")
        if os.path.exists(opt_csv_path):
            os.remove(opt_csv_path)

        eval_label = 0
        frame_labels = []

        best_seen = {"loss": float("inf"), "params": None}

        def objective_vec(x: np.ndarray) -> float:
            nonlocal eval_label
            eval_label += 1

            x = project_vec(x)
            params = x_to_model_params(x)

            write_parameters_to_paths(params, primary_param_file, site_outdir)

            tussock_model(config, output_mode="summary")

            loss = diameter_objective(
                config=config,
                iteration_label=eval_label,
                training_data=training_data,
                frames_dir=frames_dir,
                axis_limits=axis_limits,
                extinction_weight=args.extinction_weight,
                constraint_year=args.constraint_year,
                min_alive_tillers=args.min_alive_tillers,
                radius_cap=args.radius_cap,
                constraint_pass_frac=args.constraint_pass_frac,
                alive_overflow_threshold=args.alive_overflow_threshold,
                print_fail_breakdown=args.print_fail_breakdown,
                plot_every=args.plot_every,
                plot_kde=args.plot_kde,
            )

            if args.plot_every and args.plot_every > 0 and (eval_label % args.plot_every == 0):
                frame_labels.append(eval_label)

            write_optimization_results(params, loss, eval_label, opt_csv_path)

            if loss < best_seen["loss"]:
                best_seen["loss"] = loss
                best_seen["params"] = params

            return loss

        print(f"[{site_tag}] random init trials: {args.n_init}")
        print(f"[{site_tag}] training rows: {len(training_data)} | diam range: [{training_data['diam'].min():.6g}, {training_data['diam'].max():.6g}]")

        best_init_loss = float("inf")
        best_init_params = template_params

        best_points = []  # (loss, x_trial)

        for _ in range(args.n_init):
            trial_params = sample_random_params_around(template_params, args.init_log10_span)
            x_trial = project_vec(params_to_x(trial_params))
            loss_trial = objective_vec(x_trial)

            best_points.append((loss_trial, x_trial))
            if loss_trial < best_init_loss:
                best_init_loss = loss_trial
                best_init_params = trial_params

        x0 = project_vec(params_to_x(best_init_params))

        step = np.zeros_like(x0)
        for i in range(x0.size):
            mag = abs(x0[i])
            step[i] = args.step_frac * mag if mag > 1e-12 else args.step_abs

        remaining_budget = max(0, args.max_evals - eval_label)
        print(f"[{site_tag}] starting Nelder–Mead (remaining eval budget: {remaining_budget})")

        if remaining_budget > 0:
            best_points.sort(key=lambda t: t[0])
            n = x0.size
            init_simplex = [x for _, x in best_points[: (n + 1)]]

            best_x, best_f, used = nelder_mead(
                f=objective_vec,
                x0=x0,
                step=step,
                max_evals=args.max_evals,
                tol_f=args.tol_f,
                tol_x=args.tol_x,
                project_fn=project_vec,
                init_simplex=init_simplex,
                stall_expand_after=args.stall_expand_after,
                stall_expand_factor=args.stall_expand_factor,
            )
        else:
            best_x = x0

        final_params = best_seen["params"] if best_seen["params"] is not None else x_to_model_params(best_x)

        print(f"[{site_tag}] best loss: {best_seen['loss']:.6g}")
        print(f"[{site_tag}] best params:")
        for k, v in final_params.items():
            print(f"  {k}={v}")

        write_parameters_to_paths(final_params, primary_param_file, site_outdir)

        final_sims_dir = os.path.join(site_outdir, "final_sims")
        os.makedirs(final_sims_dir, exist_ok=True)
        config.set("Tussock Model", "filepath", final_sims_dir)

        print(f"[{site_tag}] running final full-output sims into: {final_sims_dir}")
        tussock_model(config, output_mode="full")

        if args.plot_every and args.plot_every > 0:
            gif_path = os.path.join(site_outdir, "diameter_dist_fitting.gif")
            animate_fitting(frames_dir, frame_labels, gif_path)

        print(f"Completed bin: {site_tag}")

    print("All done.")


if __name__ == "__main__":
    main()
