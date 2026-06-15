#!/usr/bin/env python3
"""
create_IPM_scalarset_degree_sweep_sumcoding.py

Degree × scalar-set sweep with SUM coding + explicit functional-form permutations.

UPDATED for the NEW transition-format IPM CSV (one row per n → n+1 transition):
- Size at time n:          area_n
- Size at time n+1:        area_n_plus_1
- Survival (0/1):          survival        (may be missing for some rows/years)
- Tillering count:         tillering       (may be missing for some rows/years)
- Ecotype grouping:        Src             (or change COL_ECOTYPE below)

Fecundity modeling:
- Binary logistic regression (Binomial GLM with logit link):
    fec_bin = 1 if tillering >= 1 else 0
- We keep fec (count) for reference, but NOT used for modeling.

Functional-form permutations (size transforms):
    1) identity:   z = x
    2) log1p:      z = log(1 + x)

Important: We DO NOT drop rows globally for missing survival/tillering.
Instead, each vital rate uses only rows where its response is available:
- growth_mean uses rows with y_growth > 0
- survival uses rows with surv not null
- fecundity uses rows with fec_bin not null

Outputs:
- IPM/degree_scalarset_model_selection_sumcoding.csv
- IPM/best_model_curves_<vital>_deg_sweep.png
- IPM/AIC_by_degree_<vital>.png
- IPM/pooled_vs_best_<vital>.png
- IPM/ipm_cpp_functions.json

Figures:
- IPM/model_space_heatmap_<vital>.png
- IPM/top_model_weights_<vital>.png
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2
from matplotlib.lines import Line2D


# ----------------------------
# Config (UPDATED)
# ----------------------------
DATA_PATH = "input_data/IPM_data.csv"
OUTDIR = "IPM"

# Transition-format column names
COL_X = "area_n"
COL_Y_GROWTH = "area_n_plus_1"
COL_SURV = "survival"
COL_TILLERS = "tillering"
COL_ECOTYPE = "Src"

# Functional-form permutations (size transforms)
XFORMS = ["identity", "log1p"]  # "identity": z=x; "log1p": z=log(1+x)

DEGREES = [1, 2, 3]
TOPK_PRINT = 10
TOPK_PLOT = 2
N_GRID = 300

SCATTER_S = 10
SCATTER_ALPHA = 0.5

OUT_SELECTION = os.path.join(OUTDIR, "degree_scalarset_model_selection_sumcoding.csv")
OUT_CPP_JSON = os.path.join(OUTDIR, "ipm_cpp_functions.json")

# Added summary outputs for paper-style model-selection panels
OUT_FUNC_ONLY_CSV = os.path.join(OUTDIR, "functional_form_only_best_scores.csv")
OUT_ECOTYPE_ONLY_CSV = os.path.join(OUTDIR, "ecotype_effect_only_best_scores.csv")
OUT_COMBINED_TOP10_CSV = os.path.join(OUTDIR, "combined_model_top10_scores.csv")

# For consistent heatmap axes
SCALARSET_ORDER = [
    "none",
    "offset", "slope", "curvature",
    "offset+slope", "offset+curvature", "slope+curvature",
    "offset+slope+curvature"
]


# ----------------------------
# Data
# ----------------------------
def ensure_outdir():
    os.makedirs(OUTDIR, exist_ok=True)


def load_data(path):
    """
    Load transition-format IPM data.

    Creates standardized working columns:
      x        = area at year n
      log1p_x  = log1p(x)
      ecotype  = Src (or chosen ecotype column)
      y_growth = area at year n+1
      surv     = survival (0/1) if present else NaN
      fec      = tillering count if present else NaN
      fec_bin  = 1 if fec>=1 else 0, but NaN where fec is NaN
    """
    df = pd.read_csv(path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Required for all vitals: x and ecotype
    if COL_X not in df.columns:
        raise ValueError("Missing required column for size at n: '{}'".format(COL_X))
    if COL_ECOTYPE not in df.columns:
        raise ValueError("Missing required ecotype column: '{}'".format(COL_ECOTYPE))

    df = df.copy()

    # x
    df["x"] = pd.to_numeric(df[COL_X], errors="coerce")
    df.dropna(subset=["x"], inplace=True)

    # leaf area shouldn't be negative; drop defensively so log1p is safe
    df = df[df["x"] >= 0].copy()

    # transforms used in candidate functional forms
    df["log1p_x"] = np.log1p(df["x"].astype(float))

    # ecotype
    df["ecotype"] = df[COL_ECOTYPE].astype(str)

    # growth response (may be missing for some rows)
    if COL_Y_GROWTH in df.columns:
        df["y_growth"] = pd.to_numeric(df[COL_Y_GROWTH], errors="coerce")
    else:
        df["y_growth"] = np.nan

    # survival (already 0/1 in transition table when present)
    if COL_SURV in df.columns:
        df["surv"] = pd.to_numeric(df[COL_SURV], errors="coerce")
    else:
        df["surv"] = np.nan

    # fecundity count (do NOT fill missing with 0; missing means unavailable)
    if COL_TILLERS in df.columns:
        df["fec"] = pd.to_numeric(df[COL_TILLERS], errors="coerce")
    else:
        df["fec"] = np.nan

    # binary fecundity (only defined where fec is present)
    df["fec_bin"] = np.where(df["fec"].isna(), np.nan, (df["fec"] >= 1).astype(int))

    # Optional: warn if fec > 1 exists (fine; we still model binary)
    fec_gt1 = (df["fec"].notna()) & (df["fec"] > 1)
    if fec_gt1.any():
        print("WARNING: Found fecundity counts > 1; treating fecundity as binary (fec_bin).")
        show_cols = ["ecotype", "x", "fec"]
        keep_cols = [c for c in show_cols if c in df.columns]
        print(df.loc[fec_gt1, keep_cols].head(20).to_string(index=False))

    return df


def ecotypes_sorted(df):
    return sorted(df["ecotype"].unique().tolist())


def basevar_for_xform(xform):
    if xform == "identity":
        return "x"
    if xform == "log1p":
        return "log1p_x"
    raise ValueError("Unknown xform: {}".format(xform))


def make_predict_df(xgrid, ecotype=None):
    d = pd.DataFrame({"x": xgrid})
    d["log1p_x"] = np.log1p(d["x"].astype(float))
    if ecotype is not None:
        d["ecotype"] = ecotype
    return d


# ----------------------------
# Readable labels for summary figures
# ----------------------------
def nice_vital(v):
    return {
        "growth_mean": "Growth",
        "survival": "Survival",
        "fecundity": "Fecundity",
    }.get(v, str(v))


def degree_name(degree):
    d = int(degree)
    return {1: "linear", 2: "quadratic", 3: "cubic"}.get(d, "degree {}".format(d))


def form_label(vital, xform, degree):
    base = degree_name(degree)
    xform = str(xform)
    if xform == "log1p":
        base = "log-size {}".format(base)
    if vital in ["survival", "fecundity"]:
        return "logit({})".format(base)
    return base


def scalar_label(s):
    s = str(s)
    if s == "none":
        return "shared across ecotypes"
    term_map = {
        "offset": "ecotype intercept",
        "slope": "ecotype slope",
        "curvature": "ecotype curvature",
    }
    parts = s.split("+")
    readable = [term_map.get(p, p) for p in parts]
    return " + ".join(readable)


# ----------------------------
# Scalar-set definitions
# ----------------------------
def scalarset_name(use_offset, use_slope, use_curv):
    parts = []
    if use_offset:
        parts.append("offset")
    if use_slope:
        parts.append("slope")
    if use_curv:
        parts.append("curvature")
    return "+".join(parts) if parts else "none"


def scalar_sets_for_degree(deg):
    """
    deg=1: only offset/slope combos (curvature forced False)
    deg>=2: all offset/slope/curvature combos
    Returned sorted by increasing complexity.
    """
    sets = []
    for use_offset in [False, True]:
        for use_slope in [False, True]:
            if deg < 2:
                sets.append((use_offset, use_slope, False))
            else:
                for use_curv in [False, True]:
                    sets.append((use_offset, use_slope, use_curv))
    sets = list(set(sets))
    sets.sort(key=lambda t: (t[0] + t[1] + t[2], t))
    return sets


def full_tuple_for_degree(deg):
    """
    Most flexible scalar set within a given degree:
      deg=1: offset+slope
      deg>=2: offset+slope+curvature
    """
    if deg < 2:
        return (True, True, False)
    return (True, True, True)


# ----------------------------
# Formula building (SUM coding + functional forms)
# ----------------------------
def poly_terms(basevar, deg):
    terms = [basevar]
    for d in range(2, deg + 1):
        terms.append("I({}**{})".format(basevar, d))
    return " + ".join(terms)


def build_formula(vital, xform, deg, use_offset, use_slope, use_curv):
    """
    SUM coding; ecotype scalars only on intercept, z, and z^2 where z = transform(x).
    For deg=3, z^3 is shared across ecotypes (no ecotype scalar on z^3).
    """
    E = "C(ecotype, Sum)"
    basevar = basevar_for_xform(xform)
    poly = poly_terms(basevar, deg)

    if vital == "growth_mean":
        lhs = "y_growth"
    elif vital == "survival":
        lhs = "surv"
    elif vital == "fecundity":
        lhs = "fec_bin"
    else:
        raise ValueError("Unknown vital")

    rhs = [poly]

    if use_offset:
        rhs.append(E)
    if use_slope:
        rhs.append("{}:{}".format(E, basevar))
    if use_curv:
        if deg < 2:
            raise ValueError("curvature scalar requested but deg < 2")
        rhs.append("{}:I({}**2)".format(E, basevar))

    return "{} ~ {}".format(lhs, " + ".join(rhs))


# ----------------------------
# Fit + metrics
# ----------------------------
def bic_manual(llf, n, k):
    return np.log(n) * k - 2.0 * llf


def fit_one(df, vital, formula):
    """
    Vital-specific row filtering (so missing survival/tillering doesn't nuke growth rows, etc.)
    """
    if vital == "growth_mean":
        d = df.copy()
        d["y_growth"] = pd.to_numeric(d["y_growth"], errors="coerce")
        d = d.dropna(subset=["y_growth"])
        d = d[d["y_growth"] > 0].copy()
        res = smf.ols(formula, d).fit()

    elif vital == "survival":
        d = df.copy()
        d["surv"] = pd.to_numeric(d["surv"], errors="coerce")
        d = d.dropna(subset=["surv"]).copy()
        res = smf.glm(formula, d, family=sm.families.Binomial()).fit()

    elif vital == "fecundity":
        d = df.copy()
        d["fec_bin"] = pd.to_numeric(d["fec_bin"], errors="coerce")
        d = d.dropna(subset=["fec_bin"]).copy()
        res = smf.glm(formula, d, family=sm.families.Binomial()).fit()

    else:
        raise ValueError("Unknown vital")

    n_used = int(res.nobs)
    k_params = int(len(res.params))
    llf = float(res.llf)

    aic = float(res.aic) if hasattr(res, "aic") and res.aic is not None else (2 * k_params - 2 * llf)
    bic = float(res.bic) if hasattr(res, "bic") and res.bic is not None else bic_manual(llf, n_used, k_params)

    return res, d, n_used, k_params, llf, aic, bic


def lr_test(llf_small, k_small, llf_big, k_big):
    lr_stat = 2.0 * (llf_big - llf_small)
    df_diff = int(k_big - k_small)
    p = chi2.sf(lr_stat, df_diff) if df_diff > 0 else np.nan
    return float(lr_stat), float(p), float(df_diff)


# ----------------------------
# Plotting helpers
# ----------------------------
def save_figure_multi(fig, stem, dpi=200, bbox_inches=None):
    png = os.path.join(OUTDIR, stem + ".png")
    svg = os.path.join(OUTDIR, stem + ".svg")
    pdf = os.path.join(OUTDIR, stem + ".pdf")
    fig.savefig(png, dpi=dpi, bbox_inches=bbox_inches)
    fig.savefig(svg, bbox_inches=bbox_inches)
    fig.savefig(pdf, bbox_inches=bbox_inches)
    plt.close(fig)

def make_ecotype_colors(ecotypes):
    cmap = plt.get_cmap("tab10")
    return dict((e, cmap(i % 10)) for i, e in enumerate(ecotypes))


def _vital_plot_meta(vital):
    if vital == "growth_mean":
        return "Growth mean", "Area at n+1 ({})".format(COL_Y_GROWTH), "y_growth"
    if vital == "survival":
        return "Survival", "Survival (0/1)", "surv"
    if vital == "fecundity":
        return "Fecundity", "Made a daughter tiller (0/1)", "fec_bin"
    raise ValueError("Unknown vital")


def plot_top_models_curves(vital, ecotypes, fitted_models, top_models, df_points):
    colors = make_ecotype_colors(ecotypes)
    xgrid = np.linspace(df_points["x"].min(), df_points["x"].max(), N_GRID)

    title_prefix, ylabel, ycol = _vital_plot_meta(vital)

    fig, ax = plt.subplots(figsize=(9.2, 6.6))

    # scatter
    for e in ecotypes:
        sub = df_points[df_points["ecotype"] == e]
        ax.scatter(
            sub["x"].values,
            sub[ycol].values,
            s=SCATTER_S,
            alpha=SCATTER_ALPHA,
            color=colors[e],
            linewidths=0
        )

    # curves
    model_box = []
    for j, (_, r) in enumerate(top_models.iterrows(), start=1):
        xform = str(r["xform"])
        deg = int(r["degree"])
        sset = str(r["scalar_set"])
        res = fitted_models[(vital, xform, deg, sset)]

        for e in ecotypes:
            pred_df = make_predict_df(xgrid, e)
            yhat = res.predict(pred_df)
            alpha = 0.95 if j == 1 else 0.60
            lw = 2.3 if j == 1 else 1.5
            ax.plot(xgrid, yhat, color=colors[e], alpha=alpha, linewidth=lw)

        model_box.append(
            "{} ) xform={}, deg={}, {}  (AIC={:.2f}, w={:.2f}, dAIC_vs_full={:.2f})".format(
                j, xform, deg, sset, float(r["AIC"]), float(r["AIC_weight"]), float(r["dAIC_vs_full_for_degree_xform"])
            )
        )

    handles = [Line2D([0], [0], color=colors[e], lw=3) for e in ecotypes]
    ax.legend(handles, ecotypes, title="Ecotype", fontsize=9, loc="upper left")

    ax.set_xlabel("Area at n ({})".format(COL_X))
    ax.set_ylabel(ylabel)
    ax.set_title("{}: top {} models (xform × degree × scalar set)".format(title_prefix, TOPK_PLOT))

    ax.text(
        0.02, 0.02, "\n".join(model_box),
        transform=ax.transAxes,
        va="bottom", ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.7")
    )

    fig.tight_layout()
    save_figure_multi(fig, "best_model_curves_{}_deg_sweep".format(vital), dpi=200)


def plot_AIC_by_degree(vital, sel_vital):
    fig, ax = plt.subplots(figsize=(9.6, 6.6))
    for (xform, sset), sub in sel_vital.groupby(["xform", "scalar_set"]):
        sub2 = sub.sort_values("degree")
        ax.plot(
            sub2["degree"].values,
            sub2["AIC"].values,
            marker="o",
            linewidth=1.6,
            alpha=0.80,
            label="{}:{}".format(xform, sset)
        )

    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("AIC")
    ax.set_title("{}: AIC by degree, scalar set, and functional form (Sum coding)".format(vital))
    ax.legend(title="xform:scalar_set", fontsize=8, ncol=2)
    fig.tight_layout()

    save_figure_multi(fig, "AIC_by_degree_{}".format(vital), dpi=200)


def growth_noise_sigma_from_res(res):
    """
    Return residual SD for OLS growth_mean model in response units (area_n_plus_1 units).
    Uses sqrt(MSE_resid).
    """
    try:
        sigma = float(np.sqrt(res.mse_resid))
    except Exception:
        sigma = float("nan")
    return sigma


def plot_pooled_vs_best(vital, ecotypes, res_pooled, res_best, best_xform, best_degree, best_scalar_set, df_points):
    colors = make_ecotype_colors(ecotypes)
    xgrid = np.linspace(df_points["x"].min(), df_points["x"].max(), N_GRID)

    title_prefix, ylabel, ycol = _vital_plot_meta(vital)

    fig, ax = plt.subplots(figsize=(9.2, 6.6))

    # scatter
    for e in ecotypes:
        sub = df_points[df_points["ecotype"] == e]
        ax.scatter(
            sub["x"].values,
            sub[ycol].values,
            s=SCATTER_S,
            alpha=SCATTER_ALPHA,
            color=colors[e],
            linewidths=0
        )

    # pooled curve (drawn but not included in legend)
    pooled_df = make_predict_df(xgrid, None)
    yhat_pooled = res_pooled.predict(pooled_df)
    ax.plot(xgrid, yhat_pooled, color="black", linewidth=3.0, alpha=0.95, linestyle="--")

    # best ecotype curves
    for e in ecotypes:
        pred_df = make_predict_df(xgrid, e)
        yhat = res_best.predict(pred_df)
        ax.plot(xgrid, yhat, color=colors[e], linewidth=2.2, alpha=0.95)

    # Only ecotype legend
    handles = [Line2D([0], [0], color=colors[e], lw=3) for e in ecotypes]
    ax.legend(handles, ecotypes, title="Ecotype", fontsize=9, loc="upper left")

    ax.set_xlabel("Area at n ({})".format(COL_X))
    ax.set_ylabel(ylabel)
    ax.set_title("{}: pooled curve vs best ecotype curves".format(title_prefix))

    fig.tight_layout()
    save_figure_multi(fig, "pooled_vs_best_{}".format(vital), dpi=200)


def plot_model_space_heatmap(vital, sel_vital):
    df = sel_vital.copy()
    df["deltaAIC_global"] = df["AIC"] - df["AIC"].min()

    fig, axes = plt.subplots(1, len(XFORMS), figsize=(13.2, 4.3), sharey=True)
    if len(XFORMS) == 1:
        axes = [axes]

    for ax, xform in zip(axes, XFORMS):
        sub = df[df["xform"] == xform].copy()

        mat = np.full((len(DEGREES), len(SCALARSET_ORDER)), np.nan, dtype=float)

        for i, deg in enumerate(DEGREES):
            for j, sset in enumerate(SCALARSET_ORDER):
                hit = sub[(sub["degree"] == deg) & (sub["scalar_set"] == sset)]
                if hit.shape[0] == 0:
                    continue
                mat[i, j] = float(hit.iloc[0]["deltaAIC_global"])

        im = ax.imshow(mat, aspect="auto")
        ax.set_title("{} | xform={}".format(vital, xform))
        ax.set_xticks(np.arange(len(SCALARSET_ORDER)))
        ax.set_xticklabels(SCALARSET_ORDER, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(DEGREES)))
        ax.set_yticklabels([str(d) for d in DEGREES])
        ax.set_xlabel("scalar_set")
        if ax is axes[0]:
            ax.set_ylabel("degree")

    fig.colorbar(im, ax=axes, shrink=0.9, label="deltaAIC (vs best overall)")
    fig.suptitle("Model-space support: deltaAIC grid (xform × degree × scalar set)", y=1.02)
    fig.tight_layout()

    save_figure_multi(fig, "model_space_heatmap_{}".format(vital), dpi=200, bbox_inches="tight")


def plot_top_model_weights(vital, sel_vital, topn=12):
    sub = sel_vital.sort_values("AIC").head(topn).copy()
    labels = ["{}|d{}|{}".format(r.xform, int(r.degree), r.scalar_set) for _, r in sub.iterrows()]
    weights = sub["AIC_weight"].values

    fig, ax = plt.subplots(figsize=(10.2, 4.6))
    ax.bar(np.arange(len(weights)), weights)
    ax.set_xticks(np.arange(len(weights)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("AIC weight")
    ax.set_title("{}: top {} models by AIC weight".format(vital, topn))
    fig.tight_layout()

    save_figure_multi(fig, "top_model_weights_{}".format(vital), dpi=200)


# ----------------------------
# Paper-style summary figure panels
# ----------------------------
def plot_functional_form_only_profiles(sel):
    func_rows = []
    for (vital, ff_label), g in sel.groupby(["vital_rate", "functional_form_label"], dropna=False):
        best = g.sort_values("AIC").iloc[0]
        func_rows.append({
            "vital_rate": vital,
            "functional_form_label": ff_label,
            "best_AIC": float(best["AIC"]),
            "best_deltaAIC": float(best["deltaAIC"]),
            "best_AIC_weight": float(best["AIC_weight"]),
            "best_ecotype_terms_label": str(best["ecotype_terms_label"]),
            "best_scalar_set": str(best["scalar_set"]),
            "best_xform": str(best["xform"]),
            "best_degree": int(best["degree"]),
        })
    func_df = pd.DataFrame(func_rows)
    func_df.to_csv(OUT_FUNC_ONLY_CSV, index=False)

    vitals_present = [v for v in ["growth_mean", "survival", "fecundity"] if v in set(func_df["vital_rate"])]
    fig, axes = plt.subplots(1, len(vitals_present), figsize=(6.2 * len(vitals_present), 5.8), sharex=True)
    if len(vitals_present) == 1:
        axes = [axes]

    for ax, vital in zip(axes, vitals_present):
        sub = func_df[func_df["vital_rate"] == vital].sort_values("best_AIC").copy()
        sub = sub.sort_values("best_AIC", ascending=False).reset_index(drop=True)
        y = np.arange(len(sub))

        ax.barh(y, sub["best_AIC_weight"], alpha=0.75)
        ax.scatter(sub["best_AIC_weight"], y, s=55, marker="D", zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["functional_form_label"], fontsize=9)
        ax.set_xlabel("AIC weight of best model using this functional form")
        ax.set_title(nice_vital(vital))
        ax.grid(True, axis="x", alpha=0.25)

        for yi, w, d, et in zip(y, sub["best_AIC_weight"], sub["best_deltaAIC"], sub["best_ecotype_terms_label"]):
            ax.text(float(w) + 0.01, yi, "ΔAIC={:.1f}\n{}".format(float(d), et), va="center", fontsize=7)

    axes[0].set_ylabel("Functional form")
    fig.suptitle("Functional form only: best-supported model for each form", y=1.02)
    fig.tight_layout()
    save_figure_multi(fig, "functional_form_only_score_profiles", dpi=300, bbox_inches="tight")


def plot_ecotype_effect_only_profiles(sel):
    eco_rows = []
    for (vital, eco_label), g in sel.groupby(["vital_rate", "ecotype_terms_label"], dropna=False):
        best = g.sort_values("AIC").iloc[0]
        eco_rows.append({
            "vital_rate": vital,
            "ecotype_terms_label": eco_label,
            "best_AIC": float(best["AIC"]),
            "best_deltaAIC": float(best["deltaAIC"]),
            "best_AIC_weight": float(best["AIC_weight"]),
            "best_functional_form_label": str(best["functional_form_label"]),
            "best_xform": str(best["xform"]),
            "best_degree": int(best["degree"]),
            "best_scalar_set": str(best["scalar_set"]),
        })
    eco_df = pd.DataFrame(eco_rows)
    eco_df.to_csv(OUT_ECOTYPE_ONLY_CSV, index=False)

    vitals_present = [v for v in ["growth_mean", "survival", "fecundity"] if v in set(eco_df["vital_rate"])]
    fig, axes = plt.subplots(1, len(vitals_present), figsize=(6.4 * len(vitals_present), 7.0), sharex=True)
    if len(vitals_present) == 1:
        axes = [axes]

    for ax, vital in zip(axes, vitals_present):
        sub = eco_df[eco_df["vital_rate"] == vital].sort_values("best_AIC").copy()
        sub = sub.sort_values("best_AIC", ascending=False).reset_index(drop=True)
        y = np.arange(len(sub))

        ax.barh(y, sub["best_AIC_weight"], alpha=0.75)
        ax.scatter(sub["best_AIC_weight"], y, s=55, marker="D", zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["ecotype_terms_label"], fontsize=8)
        ax.set_xlabel("AIC weight of best model using this ecotype-effect set")
        ax.set_title(nice_vital(vital))
        ax.grid(True, axis="x", alpha=0.25)

        for yi, w, d, ff in zip(y, sub["best_AIC_weight"], sub["best_deltaAIC"], sub["best_functional_form_label"]):
            ax.text(float(w) + 0.01, yi, "ΔAIC={:.1f}\n{}".format(float(d), ff), va="center", fontsize=7)

    axes[0].set_ylabel("Ecotype-effect set")
    fig.suptitle("Ecotype effects only: best-supported model for each ecotype-effect set", y=1.02)
    fig.tight_layout()
    save_figure_multi(fig, "ecotype_effect_only_score_profiles", dpi=300, bbox_inches="tight")


def plot_combined_model_top10_profiles(sel):
    combined_rows = []
    vitals_present = [v for v in ["growth_mean", "survival", "fecundity"] if v in set(sel["vital_rate"])]
    fig, axes = plt.subplots(1, len(vitals_present), figsize=(6.2 * len(vitals_present), 7.4), sharex=True)
    if len(vitals_present) == 1:
        axes = [axes]

    for ax, vital in zip(axes, vitals_present):
        sub_all = sel[sel["vital_rate"] == vital].copy()
        sub = sub_all.sort_values("AIC").head(10).copy()
        sub = sub.sort_values("AIC", ascending=False).reset_index(drop=True)

        y = np.arange(len(sub))
        ax.barh(y, sub["AIC_weight"], alpha=0.75)
        ax.scatter(sub["AIC_weight"], y, s=55, marker="D", zorder=3)

        labels = []
        for _, r in sub.iterrows():
            rank = int((sub_all["AIC"] < r["AIC"]).sum() + 1)
            labels.append("#{} {}\n{}".format(rank, r["functional_form_label"], r["ecotype_terms_label"]))
            combined_rows.append({
                "vital_rate": vital,
                "rank_within_vital": rank,
                "functional_form_label": r["functional_form_label"],
                "ecotype_terms_label": r["ecotype_terms_label"],
                "AIC": float(r["AIC"]),
                "deltaAIC": float(r["deltaAIC"]),
                "AIC_weight": float(r["AIC_weight"]),
                "xform": r["xform"],
                "degree": int(r["degree"]),
                "scalar_set": r["scalar_set"],
            })

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("AIC weight (higher is better)")
        ax.set_title(nice_vital(vital))
        ax.grid(True, axis="x", alpha=0.25)

        for yi, w, d in zip(y, sub["AIC_weight"], sub["deltaAIC"]):
            ax.text(float(w) + 0.01, yi, "ΔAIC={:.1f}".format(float(d)), va="center", fontsize=7)

    fig.suptitle("Combined models: top 10 supported models", y=1.02)
    fig.tight_layout()
    save_figure_multi(fig, "combined_model_top10_score_profiles", dpi=300, bbox_inches="tight")

    pd.DataFrame(combined_rows).to_csv(OUT_COMBINED_TOP10_CSV, index=False)


# ----------------------------
# CPP JSON export helpers
# ----------------------------
def _norm(s):
    return s.replace(" ", "")


def get_base_poly_coefs(res, degree, basevar):
    params = res.params
    out = {"b0": float(params["Intercept"]), "b1": float(params[basevar]), "b2": 0.0, "b3": 0.0}

    if degree >= 2:
        for k, v in params.items():
            if _norm(k) == _norm("I({}**2)".format(basevar)):
                out["b2"] = float(v)
                break

    if degree >= 3:
        for k, v in params.items():
            if _norm(k) == _norm("I({}**3)".format(basevar)):
                out["b3"] = float(v)
                break

    return out


def complete_sumcoded_effects(params, ecotypes, suffix):
    suffix_n = _norm(suffix)
    found = {}
    for term, val in params.items():
        t = _norm(term)
        if t.startswith("C(ecotype,Sum)[S.") and t.endswith("]{}".format(suffix_n)):
            ec = t.split("[S.")[-1].split("]")[0]
            found[ec] = float(val)

    missing = [e for e in ecotypes if e not in found]
    if len(missing) == 1:
        found[missing[0]] = -sum(found.values())

    for e in ecotypes:
        found.setdefault(e, 0.0)

    return found


def link_info(vital):
    if vital == "growth_mean":
        return {"link": "identity", "response_from_eta": "y = eta"}
    if vital == "survival":
        return {"link": "logit", "response_from_eta": "p = 1/(1+exp(-eta))"}
    if vital == "fecundity":
        return {"link": "logit", "response_from_eta": "p = 1/(1+exp(-eta))"}
    raise ValueError("Unknown vital")


def export_json(df, sel):
    ecotypes = ecotypes_sorted(df)
    vitals = ["growth_mean", "survival", "fecundity"]

    def growth_noise_sigma_from_res(res):
        """
        Residual SD for OLS growth_mean model in response units (area_n_plus_1 units).
        Uses sqrt(MSE_resid).
        """
        try:
            return float(np.sqrt(res.mse_resid))
        except Exception:
            return float("nan")

    out = {
        "meta": {
            "coding": "Sum coding. Ecotype deviations sum to 0 across ecotypes.",
            "x_variable": COL_X,
            "ecotype_variable": COL_ECOTYPE,
            "ecotypes_fit": ecotypes,
            "new_ecotype_placeholder_key": "NEW_ECOTYPE",
            "z_transform_note": "Models are fit in z, where z = x (identity) or z = log(1+x) (log1p).",
            "polynomial_eta_form_in_z": "eta(z) = b0 + b1*z + b2*z^2 + b3*z^3  (missing higher terms are 0)",
        },
        "vital_rates": {}
    }

    for vital in vitals:
        best = sel[sel["vital_rate"] == vital].sort_values("AIC").head(1).iloc[0]
        xform = str(best["xform"])
        degree = int(best["degree"])
        scalar_set = str(best["scalar_set"])
        formula = str(best["formula"])
        basevar = basevar_for_xform(xform)

        # Fit the selected best model (and keep res so we can extract sigma for growth)
        res, d_used, n_used, k_params, llf, aic, bic = fit_one(df, vital, formula)
        params = res.params

        # Base polynomial coefficients (in z = x or log1p(x))
        base = get_base_poly_coefs(res, degree, basevar)

        # Which scalar deviations are active
        active = {
            "offset": ("offset" in scalar_set),
            "slope": ("slope" in scalar_set),
            "curvature": ("curvature" in scalar_set and degree >= 2),
        }

        # Extract sum-coded deviations into explicit per-ecotype maps
        scalars = {}

        # Offset deviations
        if active["offset"]:
            m = complete_sumcoded_effects(params, ecotypes, suffix="")
        else:
            m = dict((e, 0.0) for e in ecotypes)
        m["NEW_ECOTYPE"] = 0.0
        scalars["offset"] = m

        # Slope deviations
        slope_suffix = ":{}".format(basevar)
        if active["slope"]:
            m = complete_sumcoded_effects(params, ecotypes, suffix=slope_suffix)
        else:
            m = dict((e, 0.0) for e in ecotypes)
        m["NEW_ECOTYPE"] = 0.0
        scalars["slope"] = m

        # Curvature deviations (only z^2)
        curv_suffix = ":I({}**2)".format(basevar)
        if active["curvature"]:
            m = complete_sumcoded_effects(params, ecotypes, suffix=curv_suffix)
        else:
            m = dict((e, 0.0) for e in ecotypes)
        m["NEW_ECOTYPE"] = 0.0
        scalars["curvature"] = m

        vital_obj = {
            "x_transform": xform,
            "degree": degree,
            "scalar_set": scalar_set,
            "formula": formula,
            **link_info(vital),
            "base_coefficients_in_z": base,
            "scalar_active": active,
            "scalar_deviations": scalars,
            "AIC": float(best["AIC"]),
            "k_params": int(best["k_params"]),
            "n": int(best["n"]),
        }

        # Add growth noise SD for C++ simulation (only growth_mean is OLS)
        if vital == "growth_mean":
            sigma_y = growth_noise_sigma_from_res(res)
            vital_obj["noise_model"] = "Normal(0, sigma^2) additive on y_growth"
            vital_obj["sigma_y"] = float(sigma_y)

        out["vital_rates"][vital] = vital_obj

    with open(OUT_CPP_JSON, "w") as f:
        json.dump(out, f, indent=2)

    print("\nWrote C++ JSON:")
    print(" - {}".format(OUT_CPP_JSON))



# ----------------------------
# Sweep
# ----------------------------
def run_sweep(df):
    vitals = ["growth_mean", "survival", "fecundity"]
    ecotypes = ecotypes_sorted(df)

    rows = []
    fitted_models = {}  # (vital, xform, degree, scalar_set) -> fitted result

    for vital in vitals:
        for xform in XFORMS:
            for deg in DEGREES:
                full_tuple = full_tuple_for_degree(deg)
                full_name = scalarset_name(*full_tuple)
                full_formula = build_formula(vital, xform, deg, *full_tuple)
                full_res, _, full_n, full_k, full_llf, full_aic, full_bic = fit_one(df, vital, full_formula)
                fitted_models[(vital, xform, deg, full_name)] = full_res

                for use_offset, use_slope, use_curv in scalar_sets_for_degree(deg):
                    name = scalarset_name(use_offset, use_slope, use_curv)
                    formula = build_formula(vital, xform, deg, use_offset, use_slope, use_curv)

                    if name == full_name:
                        res, n_used, k_params, llf, aic, bic = full_res, full_n, full_k, full_llf, full_aic, full_bic
                    else:
                        res, _, n_used, k_params, llf, aic, bic = fit_one(df, vital, formula)

                    fitted_models[(vital, xform, deg, name)] = res

                    lr_stat, lr_p, lr_df = lr_test(llf, k_params, full_llf, full_k)
                    dAIC_vs_full = aic - full_aic
                    evidence_ratio = float(np.exp(-0.5 * dAIC_vs_full))

                    rows.append({
                        "vital_rate": vital,
                        "xform": xform,
                        "degree": deg,
                        "scalar_set": name,
                        "formula": formula,
                        "n": n_used,
                        "k_params": k_params,
                        "llf": llf,
                        "AIC": aic,
                        "BIC": bic,

                        "full_for_degree_xform": full_name,
                        "AIC_full_for_degree_xform": full_aic,
                        "dAIC_vs_full_for_degree_xform": dAIC_vs_full,
                        "evidence_ratio_vs_full_for_degree_xform": evidence_ratio,

                        "LR_vs_full_for_degree_xform": lr_stat,
                        "p_vs_full_for_degree_xform": lr_p,
                        "df_vs_full_for_degree_xform": lr_df,
                    })

        vital_rows_idx = [i for i, r in enumerate(rows) if r["vital_rate"] == vital]
        vital_df = pd.DataFrame([rows[i] for i in vital_rows_idx]).copy()

        vital_df["deltaAIC"] = vital_df["AIC"] - vital_df["AIC"].min()
        vital_df["AIC_weight"] = np.exp(-0.5 * vital_df["deltaAIC"])
        vital_df["AIC_weight"] = vital_df["AIC_weight"] / vital_df["AIC_weight"].sum()

        for i, rr in vital_df.iterrows():
            rows[vital_rows_idx[i]]["deltaAIC"] = float(rr["deltaAIC"])
            rows[vital_rows_idx[i]]["AIC_weight"] = float(rr["AIC_weight"])

        # plots per vital
        if vital == "growth_mean":
            df_points = df.copy()
            df_points = df_points.dropna(subset=["y_growth"])
            df_points = df_points[df_points["y_growth"] > 0].copy()
        elif vital == "survival":
            df_points = df.dropna(subset=["surv"]).copy()
        else:
            df_points = df.dropna(subset=["fec_bin"]).copy()

        top_models = vital_df.sort_values("AIC").head(TOPK_PLOT)
        plot_top_models_curves(vital, ecotypes, fitted_models, top_models, df_points)

        plot_AIC_by_degree(vital, vital_df)
        plot_model_space_heatmap(vital, vital_df)
        plot_top_model_weights(vital, vital_df, topn=12)

        best_row = vital_df.sort_values("AIC").head(1).iloc[0]
        best_xform = str(best_row["xform"])
        best_degree = int(best_row["degree"])
        best_scalar_set = str(best_row["scalar_set"])
        res_best = fitted_models[(vital, best_xform, best_degree, best_scalar_set)]

        pooled_formula = build_formula(vital, best_xform, best_degree, False, False, False)
        res_pooled = fit_one(df, vital, pooled_formula)[0]

        plot_pooled_vs_best(vital, ecotypes, res_pooled, res_best, best_xform, best_degree, best_scalar_set, df_points)

    sel = pd.DataFrame(rows).sort_values(["vital_rate", "AIC", "xform", "degree", "scalar_set"]).copy()
    sel["functional_form_label"] = [
        form_label(v, x, d) for v, x, d in zip(sel["vital_rate"], sel["xform"], sel["degree"])
    ]
    sel["ecotype_terms_label"] = sel["scalar_set"].map(scalar_label)
    sel.to_csv(OUT_SELECTION, index=False)
    return sel


# ----------------------------
# Printing summaries
# ----------------------------
def print_top_models(sel, vital, topk=TOPK_PRINT):
    print("\n{}".format(vital))
    sub = sel[sel["vital_rate"] == vital].sort_values("AIC").head(topk)
    cols = [
        "xform", "degree", "scalar_set",
        "AIC", "deltaAIC", "AIC_weight",
        "k_params", "n",
        "dAIC_vs_full_for_degree_xform", "evidence_ratio_vs_full_for_degree_xform",
        "p_vs_full_for_degree_xform"
    ]
    print(sub[cols].to_string(index=False))


def print_best_per_degree(sel, vital):
    print("\n{} (best scalar_set within each xform × degree)".format(vital))
    sub = sel[sel["vital_rate"] == vital].copy()
    best = sub.sort_values("AIC").groupby(["xform", "degree"]).head(1).sort_values(["xform", "degree"])
    cols = [
        "xform", "degree", "scalar_set", "AIC",
        "dAIC_vs_full_for_degree_xform", "evidence_ratio_vs_full_for_degree_xform",
        "p_vs_full_for_degree_xform", "k_params"
    ]
    print(best[cols].to_string(index=False))


# ----------------------------
# Main
# ----------------------------
def main():
    ensure_outdir()
    df = load_data(DATA_PATH)

    print("Loaded rows:", df.shape[0])
    print("Ecotypes:", ecotypes_sorted(df))
    print("Functional forms (x transforms) tested:", XFORMS)
    print("Degrees tested:", DEGREES)
    print("Scalar sets tested:", SCALARSET_ORDER)
    print("Fecundity model: logistic on fec_bin (any daughter tiller vs none)")
    print("Rows with growth (y_growth not null):", int(df["y_growth"].notna().sum()))
    print("Rows with survival (surv not null):", int(df["surv"].notna().sum()))
    print("Rows with fecundity (fec_bin not null):", int(df["fec_bin"].notna().sum()))

    sel = run_sweep(df)

    print("\n=== Top models by AIC (xform × degree × scalar set, Sum coding) ===")
    for vital in ["growth_mean", "survival", "fecundity"]:
        print_top_models(sel, vital, topk=TOPK_PRINT)

    print("\n=== Best per degree (within each xform × degree) ===")
    for vital in ["growth_mean", "survival", "fecundity"]:
        print_best_per_degree(sel, vital)

    print("\nSaved to:", OUTDIR)
    print(" - degree_scalarset_model_selection_sumcoding.csv")
    print(" - functional_form_only_best_scores.csv")
    print(" - ecotype_effect_only_best_scores.csv")
    print(" - combined_model_top10_scores.csv")
    print(" - functional_form_only_score_profiles.(png/svg/pdf)")
    print(" - ecotype_effect_only_score_profiles.(png/svg/pdf)")
    print(" - combined_model_top10_score_profiles.(png/svg/pdf)")
    for vital in ["growth_mean", "survival", "fecundity"]:
        print(" - best_model_curves_{}_deg_sweep.(png/svg/pdf)".format(vital))
        print(" - AIC_by_degree_{}.(png/svg/pdf)".format(vital))
        print(" - pooled_vs_best_{}.(png/svg/pdf)".format(vital))
        print(" - model_space_heatmap_{}.(png/svg/pdf)".format(vital))
        print(" - top_model_weights_{}.(png/svg/pdf)".format(vital))

    plot_functional_form_only_profiles(sel)
    plot_ecotype_effect_only_profiles(sel)
    plot_combined_model_top10_profiles(sel)

    export_json(df, sel)


if __name__ == "__main__":
    main()
