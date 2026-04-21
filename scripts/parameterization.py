#!/usr/bin/env python3
# parameterize_tussock.py

import argparse
import configparser
import csv
import math
import os
import random
import subprocess
from collections import OrderedDict
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

try:
	import seaborn as sns
	_HAS_SNS = True
except Exception:
	_HAS_SNS = False


MODEL_PARAM_NAMES = [
	"ks",
	"kr",
	"bs",
	"br",
	"c_space",
	"c_repro",
	"k_crowd",
	"leaf_offset",
]


@dataclass
class PathSettings:
	param_file: str
	output_dir: str
	training_csv: str


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
	radius_cap: float
	constraint_pass_frac: float
	alive_overflow_threshold: int
	hard_fail_on_overflow: bool
	require_survive_to_end_for_fit: bool


@dataclass
class PlotSettings:
	plot_every: int
	plot_kde: bool
	print_fail_breakdown: bool


@dataclass
class RunSettings:
	sites: list | None
	paths: PathSettings
	optimization: OptimizationSettings
	constraints: ConstraintSettings
	plotting: PlotSettings


def parse_args():
	parser = argparse.ArgumentParser(description="Tussock model parameterization")
	parser.add_argument(
		"--config",
		type=str,
		required=True,
		help="Path to the combined ini config file",
	)
	parser.add_argument("--sites", nargs="*", default=None)
	return parser.parse_args()


def read_bool(config, section, key, fallback=False):
	return config.getboolean(section, key, fallback=fallback)


def load_combined_config(config_path: str, cli_sites=None):
	config = configparser.ConfigParser()

	if not os.path.exists(config_path):
		raise FileNotFoundError(f"Config not found: {config_path}")

	config.read(config_path)

	if "Tussock Model" not in config:
		raise ValueError(f"Missing [Tussock Model] section in {config_path}")

	param_file = config.get("Paths", "param_file", fallback=os.path.join("parameters", "parameters.txt"))
	output_dir = config.get("Paths", "output_dir", fallback="parameterization_outputs")
	training_csv = config.get("Paths", "training_csv", fallback="./input_data/tussock_density_tussock_diam.csv")

	paths = PathSettings(
		param_file=os.path.abspath(param_file),
		output_dir=output_dir,
		training_csv=training_csv,
	)

	optimization = OptimizationSettings(
		max_evals=config.getint("Optimization", "max_evals", fallback=200),
		n_init=config.getint("Optimization", "n_init", fallback=25),
		tol_f=config.getfloat("Optimization", "tol_f", fallback=1e-3),
		tol_x=config.getfloat("Optimization", "tol_x", fallback=1e-3),
		init_log10_span=config.getfloat("Optimization", "init_log10_span", fallback=1.0),
		step_frac=config.getfloat("Optimization", "step_frac", fallback=0.2),
		step_abs=config.getfloat("Optimization", "step_abs", fallback=0.3),
		optimize_log_space=read_bool(config, "Optimization", "optimize_log_space", fallback=False),
		cma_sigma=config.getfloat("Optimization", "cma_sigma", fallback=0.5),
		cma_popsize=config.getint("Optimization", "cma_popsize", fallback=0),
		cma_patience=config.getint("Optimization", "cma_patience", fallback=20),
	)

	constraints = ConstraintSettings(
		extinction_weight=config.getfloat("Constraints", "extinction_weight", fallback=0.0),
		constraint_year=config.getint("Constraints", "constraint_year", fallback=25),
		min_alive_tillers=config.getint("Constraints", "min_alive_tillers", fallback=25),
		radius_cap=config.getfloat("Constraints", "radius_cap", fallback=2.5),
		constraint_pass_frac=config.getfloat("Constraints", "constraint_pass_frac", fallback=0.8),
		alive_overflow_threshold=config.getint("Constraints", "alive_overflow_threshold", fallback=400),
		hard_fail_on_overflow=read_bool(config, "Constraints", "hard_fail_on_overflow", fallback=False),
		require_survive_to_end_for_fit=read_bool(config, "Constraints", "require_survive_to_end_for_fit", fallback=False),
	)

	plotting = PlotSettings(
		plot_every=config.getint("Plotting", "plot_every", fallback=10),
		plot_kde=read_bool(config, "Plotting", "plot_kde", fallback=False),
		print_fail_breakdown=read_bool(config, "Plotting", "print_fail_breakdown", fallback=False),
	)

	sites = cli_sites if cli_sites is not None else None

	run_settings = RunSettings(
		sites=sites,
		paths=paths,
		optimization=optimization,
		constraints=constraints,
		plotting=plotting,
	)

	return config, run_settings


def _safe_makedirs(dirpath: str):
	if dirpath:
		os.makedirs(dirpath, exist_ok=True)


def coerce_model_param(name: str, value: float) -> float:
	if not np.isfinite(value):
		raise ValueError(f"Non-finite value for parameter '{name}': {value}")

	if name in {"ks", "kr", "k_crowd"}:
		return float(max(0.0, value))

	if name in {"c_space", "c_repro"}:
		return float(min(1.0, max(0.0, value)))

	return float(value)


def read_parameter_file(param_file: str) -> OrderedDict:
	if not os.path.exists(param_file):
		raise FileNotFoundError(
			f"Model parameter file not found: {param_file}\n"
			f"This file should contain only model parameters used by the simulation."
		)

	params = OrderedDict()
	with open(param_file, "r") as f:
		for raw in f:
			line = raw.strip()
			if not line or line.startswith("#") or "=" not in line:
				continue

			k, v = line.split("=", 1)
			k = k.strip()
			v = v.strip()

			if k not in MODEL_PARAM_NAMES:
				raise ValueError(
					f"Unexpected key in parameter file '{param_file}': '{k}'.\n"
					f"Allowed keys are: {MODEL_PARAM_NAMES}"
				)

			if k in params:
				raise ValueError(f"Duplicate parameter '{k}' in {param_file}")

			params[k] = coerce_model_param(k, float(v))

	missing = [k for k in MODEL_PARAM_NAMES if k not in params]
	if missing:
		raise ValueError(f"Missing parameters in {param_file}: {missing}")

	return OrderedDict((k, params[k]) for k in MODEL_PARAM_NAMES)


def random_initial_parameters() -> OrderedDict:
	def logu(lo, hi):
		return 10 ** random.uniform(math.log10(lo), math.log10(hi))

	params = OrderedDict()
	params["ks"] = float(logu(1e-3, 100.0))
	params["kr"] = float(logu(1e-3, 100.0))
	params["bs"] = float(random.uniform(-3.0, 3.0))
	params["br"] = float(random.uniform(-3.0, 3.0))
	params["c_space"] = float(random.uniform(0.0, 1.0))
	params["c_repro"] = float(random.uniform(0.0, 1.0))
	params["k_crowd"] = float(logu(1e-4, 10.0))
	params["leaf_offset"] = float(random.uniform(-200.0, 200.0))

	for k in MODEL_PARAM_NAMES:
		params[k] = coerce_model_param(k, params[k])

	return params


def initialize_random_parameter_file(param_file: str) -> OrderedDict:
	params = random_initial_parameters()
	write_parameter_file(params, param_file)
	return params


def write_parameter_file(parameters: OrderedDict, param_file: str):
	_safe_makedirs(os.path.dirname(param_file))
	with open(param_file, "w") as f:
		for k in MODEL_PARAM_NAMES:
			f.write(f"{k}={float(parameters[k])}\n")


def write_parameter_snapshot(parameters: OrderedDict, site_outdir: str):
	snapshot_path = os.path.join(site_outdir, "parameters.txt")
	write_parameter_file(parameters, snapshot_path)


def write_model_runtime_ini(
	project_root: str,
	param_file_rel_from_root: str,
	constraint_year: int,
	alive_overflow_threshold: int,
):
	ini_path = os.path.join(project_root, "model_runtime.ini")

	cp = configparser.ConfigParser()
	cp["Parameterization"] = {
		"param_file": str(param_file_rel_from_root),
		"constraint_year": str(int(constraint_year)),
		"alive_overflow_threshold": str(int(alive_overflow_threshold)),
	}

	with open(ini_path, "w") as f:
		cp.write(f)


def params_to_vector(params: OrderedDict) -> np.ndarray:
	return np.array([params[k] for k in MODEL_PARAM_NAMES], dtype=float)


def vector_to_params(vec: np.ndarray) -> OrderedDict:
	return OrderedDict((k, coerce_model_param(k, float(v))) for k, v in zip(MODEL_PARAM_NAMES, vec))


def sample_random_params_around(base_params: OrderedDict, log10_span: float) -> OrderedDict:
	def logmul(v, span):
		basep = float(max(1e-12, abs(v)))
		u = random.uniform(-span, span)
		return basep * (10 ** u)

	out = OrderedDict()

	for k in MODEL_PARAM_NAMES:
		base = float(base_params[k])

		if k in {"bs", "br"}:
			out[k] = float(base + random.uniform(-3.0, 3.0))
		elif k in {"c_space", "c_repro"}:
			out[k] = float(base + random.uniform(-0.5, 0.5))
		elif k in {"ks", "kr", "k_crowd"}:
			ref = base if base > 0 else 1.0
			out[k] = float(logmul(ref, log10_span))
		elif k == "leaf_offset":
			out[k] = float(base + random.uniform(-200.0, 200.0))
		else:
			out[k] = base

	for k in MODEL_PARAM_NAMES:
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
	ylim = (0.0, 1.1 * ymax)

	return {"xlim": xlim, "ylim": ylim}


def tussock_model(config: configparser.ConfigParser, output_mode: str, project_root: str):
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


def barrier_loss(fail_stats, constraint_year, min_alive, r_cap):
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
	w_over = 5.0
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
		return pd.DataFrame(
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
	constraints: ConstraintSettings,
	plotting: PlotSettings,
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

	overflow_mask = df["overflow_t"].to_numpy(dtype=int) >= 0
	over_frac = float(np.mean(overflow_mask)) if num_sims > 0 else 0.0

	extinct_final = int((df["alive_final"].to_numpy(dtype=int) == 0).sum())
	extinct_frac = extinct_final / max(1, num_sims)

	if constraints.hard_fail_on_overflow and overflow_mask.any():
		if plotting.print_fail_breakdown:
			print(f"[OVERFLOW_HARD_FAIL] eval={iteration_label} overflow_frac={over_frac:.2%}")
		return float((1e6 * obs_std) * (1.0 + over_frac))

	ok = (
		(df["missing_year"].to_numpy(dtype=int) == 0)
		& (df["alive_y"].to_numpy(dtype=int) >= constraints.min_alive_tillers)
		& (df["rmax_y"].to_numpy(dtype=float) <= constraints.radius_cap)
		& (df["overflow_t"].to_numpy(dtype=int) < 0)
	)

	if constraints.require_survive_to_end_for_fit:
		ok = ok & (df["alive_final"].to_numpy(dtype=int) > 0)

	pass_count = int(ok.sum())
	pass_needed = int(math.ceil(constraints.constraint_pass_frac * max(1, num_sims)))

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

	sim_diam_ok = df.loc[ok, "final_diameter"].to_numpy(dtype=float)
	sim_diam_ok = sim_diam_ok[np.isfinite(sim_diam_ok)]
	fit_loss = wasserstein_distance_1d(training_diameters, sim_diam_ok)

	bar = barrier_loss(
		fail_stats=fail_stats,
		constraint_year=constraints.constraint_year,
		min_alive=constraints.min_alive_tillers,
		r_cap=constraints.radius_cap,
	)

	ext_term = float(constraints.extinction_weight) * extinct_frac * obs_std

	leaf = df["LeafArea"].to_numpy(dtype=float)
	leaf_finite = np.isfinite(leaf)
	leaf_bad = leaf_finite & ((leaf <= 0.0) | (leaf >= 2000.0))
	denom = max(1, int(leaf_finite.sum()))
	leaf_bad_frac = float(leaf_bad.sum()) / float(denom)
	leaf_penalty_weight = 10.0
	leaf_penalty = leaf_penalty_weight * obs_std * leaf_bad_frac

	overflow_penalty = (1e4 * obs_std) * over_frac

	if pass_count < pass_needed or not np.isfinite(fit_loss):
		pass_deficit = (pass_needed - pass_count) / max(1.0, float(pass_needed))
		loss = (1e3 * obs_std) * (1.0 + bar + 5.0 * pass_deficit) + ext_term + leaf_penalty + overflow_penalty

		if plotting.print_fail_breakdown:
			missing_n = int((df["missing_year"].to_numpy(dtype=int) == 1).sum())
			min_alive_fail = int(((df["missing_year"] == 0) & (df["alive_y"] < constraints.min_alive_tillers)).sum())
			radius_fail = int(((df["missing_year"] == 0) & (df["rmax_y"] > constraints.radius_cap)).sum())
			print(
				f"[HARD_REGION] eval={iteration_label} pass={pass_count}/{num_sims} need>={pass_needed} | "
				f"overflow={over_frac:.2%} extinct_final={extinct_final}/{num_sims} | "
				f"missing_year={missing_n}, min_alive={min_alive_fail}, radius={radius_fail} | "
				f"bar={bar:.4g} | leaf_bad={leaf_bad_frac:.2%}"
			)
	else:
		loss = float(fit_loss) + (10.0 * obs_std) * float(bar) + ext_term + leaf_penalty + overflow_penalty

	do_plot = (
		plotting.plot_every is not None
		and int(plotting.plot_every) > 0
		and (iteration_label % int(plotting.plot_every) == 0)
	)

	if do_plot:
		os.makedirs(frames_dir, exist_ok=True)
		fig, ax = plt.subplots()

		if plotting.plot_kde and _HAS_SNS:
			sns.kdeplot(training_diameters, label="Observed", linewidth=1, ax=ax)
			if sim_diam_ok.size > 0:
				sns.kdeplot(sim_diam_ok, label="Modeled (ok subset)", linewidth=1, ax=ax)
		else:
			ax.hist(training_diameters, bins=30, density=True, alpha=0.4, label="Observed")
			if sim_diam_ok.size > 0:
				ax.hist(sim_diam_ok, bins=30, density=True, alpha=0.4, label="Modeled (ok subset)")

		ax.set_xlim(*axis_limits["xlim"])
		ax.set_ylim(*axis_limits["ylim"])
		ax.legend()
		ax.set_title(
			f"Iter: {iteration_label} | loss={loss:.3g} | pass={pass_count}/{num_sims} "
			f"| bar={bar:.3g} | overflow={over_frac:.1%} | extinct_final={extinct_final}/{num_sims} | leaf_bad={leaf_bad_frac:.2%}"
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
	_safe_makedirs(os.path.dirname(out_csv_path))
	file_exists = os.path.exists(out_csv_path)

	with open(out_csv_path, "a", newline="") as csvfile:
		fieldnames = MODEL_PARAM_NAMES + ["loss", "iteration"]
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		if not file_exists:
			writer.writeheader()
		writer.writerow({**parameters, "loss": float(loss), "iteration": iteration_label})


def cma_es_optimize(
	f,
	x0,
	sigma0,
	max_evals,
	tol_f,
	tol_x,
	project_fn,
	popsize=0,
	patience=20,
):
	n = x0.size

	if popsize is None or popsize <= 0:
		lmbda = 4 + int(3 * np.log(n))
	else:
		lmbda = int(popsize)
	mu = lmbda // 2

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

		m_old = m.copy()

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

		C = (
			(1.0 - c1 - cmu) * C
			+ c1 * (np.outer(pc, pc) + (1.0 - hsig) * cc * (2.0 - cc) * C)
			+ cmu * rank_mu
		)

		sigma = sigma * np.exp((cs / damps) * (norm_ps / chi_n - 1.0))
		sigma = float(max(1e-12, sigma))

		C = 0.5 * (C + C.T)
		try:
			eigvals, eigvecs = np.linalg.eigh(C)
			eigvals = np.maximum(eigvals, 1e-20)
			D = np.sqrt(eigvals)
			B = eigvecs
			invsqrtC = B @ np.diag(1.0 / D) @ B.T
		except np.linalg.LinAlgError:
			C = C + 1e-8 * np.eye(n)
			eigvals, eigvecs = np.linalg.eigh(C)
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


def build_projection_functions(opt_settings: OptimizationSettings):
	idx_weights = {k: MODEL_PARAM_NAMES.index(k) for k in ["c_space", "c_repro"]}
	idx_ks = MODEL_PARAM_NAMES.index("ks")
	idx_kr = MODEL_PARAM_NAMES.index("kr")
	idx_kcrowd = MODEL_PARAM_NAMES.index("k_crowd")
	idx_leaf_offset = MODEL_PARAM_NAMES.index("leaf_offset")

	def project_vec(x: np.ndarray) -> np.ndarray:
		x = np.array(x, dtype=float)
		x[~np.isfinite(x)] = 0.0

		if opt_settings.optimize_log_space:
			for _, idx in idx_weights.items():
				x[idx] = float(np.clip(x[idx], -10.0, 10.0))
			x[idx_ks] = float(max(0.0, x[idx_ks]))
			x[idx_kr] = float(max(0.0, x[idx_kr]))
			x[idx_kcrowd] = float(max(0.0, x[idx_kcrowd]))
			if not np.isfinite(x[idx_leaf_offset]):
				x[idx_leaf_offset] = 0.0
			return x

		for _, idx in idx_weights.items():
			x[idx] = float(np.clip(x[idx], 0.0, 1.0))
		x[idx_ks] = float(max(0.0, x[idx_ks]))
		x[idx_kr] = float(max(0.0, x[idx_kr]))
		x[idx_kcrowd] = float(max(0.0, x[idx_kcrowd]))
		if not np.isfinite(x[idx_leaf_offset]):
			x[idx_leaf_offset] = 0.0
		return x

	def x_to_model_params(x: np.ndarray) -> OrderedDict:
		x = np.array(x, dtype=float)

		if opt_settings.optimize_log_space:
			vec = x.copy()
			for _, idx in idx_weights.items():
				vec[idx] = 1.0 / (1.0 + np.exp(-vec[idx]))
			params = vector_to_params(vec)
		else:
			params = vector_to_params(project_vec(x))

		return params

	def params_to_x(params: OrderedDict) -> np.ndarray:
		vec = params_to_vector(params)

		if opt_settings.optimize_log_space:
			x = vec.copy()
			for _, idx in idx_weights.items():
				p = float(min(1.0 - 1e-9, max(1e-9, x[idx])))
				x[idx] = np.log(p / (1.0 - p))
			return x

		return vec

	return project_vec, x_to_model_params, params_to_x


def main():
	args = parse_args()
	combined_config, run_settings = load_combined_config(args.config, cli_sites=args.sites)

	full_training_df = pd.read_csv(run_settings.paths.training_csv)

	if run_settings.sites is None:
		site_list = ["ALL"]
	elif len(run_settings.sites) == 1 and run_settings.sites[0].lower() == "all":
		site_list = sorted(full_training_df["site"].unique())
	else:
		site_list = run_settings.sites

	script_dir = os.path.abspath(os.path.dirname(__file__))
	project_root = os.path.abspath(os.path.join(script_dir, ".."))

	if not os.path.exists(run_settings.paths.param_file):
		print(f"[init] Parameter file not found; creating random init: {run_settings.paths.param_file}")
		template_params = initialize_random_parameter_file(run_settings.paths.param_file)
	else:
		template_params = read_parameter_file(run_settings.paths.param_file)

	param_file_rel = os.path.relpath(run_settings.paths.param_file, start=project_root)

	nyears = int(combined_config.get("Tussock Model", "nyears"))
	if nyears < run_settings.constraints.constraint_year:
		raise ValueError(
			f"nyears={nyears} < constraint_year={run_settings.constraints.constraint_year}. "
			f"Set nyears >= constraint_year."
		)

	project_vec, x_to_model_params, params_to_x = build_projection_functions(run_settings.optimization)

	for site in site_list:
		print("\n====================================")
		print(f"   PARAMETERIZING SITE: {site}")
		print("====================================\n")

		if site == "ALL":
			training_data = full_training_df.copy()
			site_tag = "ALL"
		else:
			training_data = full_training_df[full_training_df["site"] == site].copy()
			site_tag = str(site)

		training_data = training_data.copy()
		training_data["field_davg"] = pd.to_numeric(training_data["diam"], errors="coerce")
		training_diameters = training_data["field_davg"].dropna().values
		axis_limits = fixed_axis_limits_from_observed(training_diameters, bins=30)

		site_outdir = os.path.join(run_settings.paths.output_dir, site_tag)
		os.makedirs(site_outdir, exist_ok=True)

		cpp_outdir = os.path.join(site_outdir, "simulation_outputs")
		os.makedirs(cpp_outdir, exist_ok=True)
		combined_config.set("Tussock Model", "filepath", cpp_outdir)

		frames_dir = os.path.join(site_outdir, "mean_diameter_frames")

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

			write_parameter_file(params, run_settings.paths.param_file)
			write_parameter_snapshot(params, site_outdir)

			write_model_runtime_ini(
				project_root=project_root,
				param_file_rel_from_root=param_file_rel,
				constraint_year=run_settings.constraints.constraint_year,
				alive_overflow_threshold=run_settings.constraints.alive_overflow_threshold,
			)

			tussock_model(combined_config, output_mode="summary", project_root=project_root)

			loss = diameter_objective(
				config=combined_config,
				iteration_label=eval_label,
				training_data=training_data,
				frames_dir=frames_dir,
				axis_limits=axis_limits,
				constraints=run_settings.constraints,
				plotting=run_settings.plotting,
			)

			if run_settings.plotting.plot_every and run_settings.plotting.plot_every > 0:
				if eval_label % run_settings.plotting.plot_every == 0:
					frame_labels.append(eval_label)

			write_optimization_results(params, loss, eval_label, opt_csv_path)

			if loss < best_seen["loss"]:
				best_seen["loss"] = loss
				best_seen["params"] = params

			return loss

		print(f"[{site_tag}] random init trials: {run_settings.optimization.n_init}")

		best_init_loss = float("inf")
		best_init_params = template_params

		for _ in range(run_settings.optimization.n_init):
			trial_params = sample_random_params_around(
				template_params,
				run_settings.optimization.init_log10_span,
			)
			x_trial = project_vec(params_to_x(trial_params))
			loss_trial = objective_vec(x_trial)

			if loss_trial < best_init_loss:
				best_init_loss = loss_trial
				best_init_params = trial_params

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
		print(f"[{site_tag}] starting CMA-ES (remaining eval budget: {remaining_budget})")

		if remaining_budget > 0:
			best_x, best_f, used = cma_es_optimize(
				f=objective_vec,
				x0=x0,
				sigma0=sigma0,
				max_evals=remaining_budget,
				tol_f=run_settings.optimization.tol_f,
				tol_x=run_settings.optimization.tol_x,
				project_fn=project_vec,
				popsize=run_settings.optimization.cma_popsize,
				patience=run_settings.optimization.cma_patience,
			)
		else:
			best_x = x0

		final_params = best_seen["params"] if best_seen["params"] is not None else x_to_model_params(best_x)

		print(f"[{site_tag}] best loss: {best_seen['loss']:.6g}")
		print(f"[{site_tag}] best params:")
		for k, v in final_params.items():
			print(f"  {k}={v}")

		write_parameter_file(final_params, run_settings.paths.param_file)
		write_parameter_snapshot(final_params, site_outdir)

		write_model_runtime_ini(
			project_root=project_root,
			param_file_rel_from_root=param_file_rel,
			constraint_year=run_settings.constraints.constraint_year,
			alive_overflow_threshold=run_settings.constraints.alive_overflow_threshold,
		)

		final_sims_dir = os.path.join(site_outdir, "final_sims")
		os.makedirs(final_sims_dir, exist_ok=True)
		combined_config.set("Tussock Model", "filepath", final_sims_dir)

		print(f"[{site_tag}] running final full-output sims into: {final_sims_dir}")
		tussock_model(combined_config, output_mode="full", project_root=project_root)

		if run_settings.plotting.plot_every and run_settings.plotting.plot_every > 0:
			gif_path = os.path.join(site_outdir, "diameter_dist_fitting.gif")
			animate_fitting(frames_dir, frame_labels, gif_path)

		print(f"Completed site: {site_tag}")

	print("All done.")


if __name__ == "__main__":
	main()