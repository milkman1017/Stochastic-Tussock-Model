#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NON_PARAMETER_COLUMNS = {
    "loss",
    "iteration",
    "fit_loss_raw",
    "diameter_sd_loss_raw",
    "live_tiller_radius_prior_raw",
    "extinct_frac_raw",
    "overflow_frac_raw",
    "fit_loss_weighted",
    "diameter_sd_loss_weighted",
    "live_tiller_radius_prior_weighted",
    "extinct_loss_weighted",
    "overflow_loss_weighted",
    "obs_std",
    "n_fit_sims",
    "pass_frac",
    "low_alive_frac",
    "missing_frac",
    "diameter_sd_weight",
    "live_tiller_radius_prior_weight",
    "extinct_frac_weight",
    "overflow_frac_weight",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Live question-level parameter stability analysis. "
            "Finds optimization_results.csv files under <h-dir>, extracts best parameters "
            "per set, summarizes stability, writes CSVs, and makes plots."
        )
    )
    p.add_argument("--h-dir", required=True)
    p.add_argument("--ecotype", default="")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--expected-sets", type=int, default=50)
    p.add_argument("--plot-dpi", type=int, default=250)
    p.add_argument("--top-n", type=int, default=25)
    return p.parse_args()


def is_set_dir_name(name: str) -> bool:
    return re.fullmatch(r"set_\d+", str(name)) is not None


def infer_set_id(path: Path) -> str:
    for part in path.parts:
        if is_set_dir_name(part):
            return part
    return ""


def find_optimization_files(h_dir: Path, ecotype: str) -> list[Path]:
    files = []

    for f in sorted(h_dir.rglob("optimization_results.csv")):
        parts = f.parts

        if ecotype and ecotype not in parts:
            continue

        if not any(is_set_dir_name(part) for part in parts):
            continue

        files.append(f)

    return files


def parse_metadata(h_dir: Path, result_file: Path) -> dict[str, str]:
    rel = result_file.relative_to(h_dir)
    parts = rel.parts

    question = ""
    ecotype = ""
    set_id = infer_set_id(result_file)

    if "resampled_fits" in parts:
        idx = parts.index("resampled_fits")

        if idx - 1 >= 0:
            question = parts[idx - 1]

        if idx + 1 < len(parts):
            ecotype = parts[idx + 1]
    else:
        question = parts[0] if len(parts) > 0 else ""

    return {
        "question": question,
        "ecotype": ecotype,
        "set_id": set_id,
        "optimization_result_file": str(result_file),
    }


def load_best_row(h_dir: Path, f: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(f)
    except Exception as exc:
        print(f"WARNING: skipping unreadable file, maybe still being written: {f} ({exc})")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    if "loss" not in df.columns:
        print(f"WARNING: skipping file with no loss column: {f}")
        return pd.DataFrame()

    df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
    df = df[np.isfinite(df["loss"])].copy()

    if df.empty:
        print(f"WARNING: skipping file with no finite loss rows: {f}")
        return pd.DataFrame()

    best = df.sort_values("loss", ascending=True).iloc[[0]].copy()

    meta = parse_metadata(h_dir, f)
    for k, v in meta.items():
        best[k] = v

    return best


def load_all_best_params(h_dir: Path, ecotype: str) -> pd.DataFrame:
    files = find_optimization_files(h_dir, ecotype)
    print(f"Found optimization_results.csv files: {len(files)}")

    rows = []

    for f in files:
        row = load_best_row(h_dir, f)
        if not row.empty:
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def get_parameter_columns(df: pd.DataFrame) -> list[str]:
    metadata_cols = {
        "question",
        "ecotype",
        "set_id",
        "optimization_result_file",
    }

    blocked = NON_PARAMETER_COLUMNS | metadata_cols

    param_cols = []

    for col in df.columns:
        if col in blocked:
            continue

        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() > 0:
            param_cols.append(col)

    return param_cols


def summarize_parameter_stability(best_df: pd.DataFrame, param_cols: list[str]) -> pd.DataFrame:
    rows = []

    for question, g in best_df.groupby("question", dropna=False):
        loss_vals = pd.to_numeric(g["loss"], errors="coerce")

        row = {
            "question": question,
            "n_sets_done": int(g["set_id"].nunique()),
            "n_ecotypes": int(g["ecotype"].nunique()),
            "loss_mean": float(loss_vals.mean()),
            "loss_sd": float(loss_vals.std(ddof=1)) if loss_vals.notna().sum() > 1 else 0.0,
            "loss_median": float(loss_vals.median()),
            "loss_min": float(loss_vals.min()),
            "loss_max": float(loss_vals.max()),
        }

        for p in param_cols:
            vals = pd.to_numeric(g[p], errors="coerce").dropna()

            if vals.empty:
                row[f"{p}_mean"] = np.nan
                row[f"{p}_sd"] = np.nan
                row[f"{p}_cv"] = np.nan
                row[f"{p}_median"] = np.nan
                row[f"{p}_iqr"] = np.nan
                row[f"{p}_min"] = np.nan
                row[f"{p}_max"] = np.nan
                row[f"{p}_sign_consistency"] = np.nan
                continue

            mean = float(vals.mean())
            sd = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            median = float(vals.median())
            q25 = float(vals.quantile(0.25))
            q75 = float(vals.quantile(0.75))
            iqr = q75 - q25

            cv = sd / abs(mean) if abs(mean) > 1e-12 else np.nan

            median_sign = np.sign(median)
            if median_sign == 0:
                sign_consistency = float(np.mean(np.sign(vals) == 0))
            else:
                sign_consistency = float(np.mean(np.sign(vals) == median_sign))

            row[f"{p}_mean"] = mean
            row[f"{p}_sd"] = sd
            row[f"{p}_cv"] = cv
            row[f"{p}_median"] = median
            row[f"{p}_iqr"] = iqr
            row[f"{p}_min"] = float(vals.min())
            row[f"{p}_max"] = float(vals.max())
            row[f"{p}_sign_consistency"] = sign_consistency

        rows.append(row)

    return pd.DataFrame(rows).sort_values("question").reset_index(drop=True)


def make_long_stability_table(summary_df: pd.DataFrame, param_cols: list[str]) -> pd.DataFrame:
    rows = []

    for _, r in summary_df.iterrows():
        question = r["question"]

        for p in param_cols:
            rows.append(
                {
                    "question": question,
                    "parameter": p,
                    "mean": r.get(f"{p}_mean", np.nan),
                    "sd": r.get(f"{p}_sd", np.nan),
                    "cv": r.get(f"{p}_cv", np.nan),
                    "median": r.get(f"{p}_median", np.nan),
                    "iqr": r.get(f"{p}_iqr", np.nan),
                    "min": r.get(f"{p}_min", np.nan),
                    "max": r.get(f"{p}_max", np.nan),
                    "sign_consistency": r.get(f"{p}_sign_consistency", np.nan),
                }
            )

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values(["parameter", "cv", "question"], na_position="last")

    return out


def build_progress_table(best_df: pd.DataFrame, expected_sets: int) -> pd.DataFrame:
    progress = (
        best_df.groupby("question", dropna=False)
        .agg(
            n_sets_done=("set_id", "nunique"),
            n_files=("optimization_result_file", "nunique"),
        )
        .reset_index()
        .sort_values("question")
    )

    progress["expected_sets"] = int(expected_sets)
    progress["fraction_done"] = progress["n_sets_done"] / progress["expected_sets"]

    return progress


def save_empty_plot(path: Path, title: str, message: str, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_progress(progress_df: pd.DataFrame, plot_dir: Path, dpi: int) -> None:
    out = plot_dir / "question_progress_live.png"

    if progress_df.empty:
        save_empty_plot(out, "Question progress", "No progress data.", dpi)
        return

    df = progress_df.sort_values("question").copy()
    labels = df["question"].astype(str).to_list()
    y = pd.to_numeric(df["fraction_done"], errors="coerce").fillna(0).to_numpy()

    fig_width = max(9, 0.6 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, 5))

    x = np.arange(len(labels))
    ax.bar(x, y)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction of expected sets completed")
    ax.set_xlabel("Question")
    ax.set_title("Live run progress by question")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.grid(True, axis="y", alpha=0.25)

    for i, row in df.reset_index(drop=True).iterrows():
        done = int(row.get("n_sets_done", 0))
        expected = int(row.get("expected_sets", 0))
        ax.text(i, min(1.02, y[i] + 0.02), f"{done}/{expected}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def plot_cv_heatmap(long_df: pd.DataFrame, plot_dir: Path, dpi: int) -> None:
    out = plot_dir / "parameter_cv_heatmap_live.png"

    if long_df.empty:
        save_empty_plot(out, "Parameter CV heatmap", "No usable CV data.", dpi)
        return

    df = long_df.copy()
    df["cv"] = pd.to_numeric(df["cv"], errors="coerce")

    pivot = df.pivot_table(index="parameter", columns="question", values="cv", aggfunc="median")

    if pivot.empty:
        save_empty_plot(out, "Parameter CV heatmap", "No finite CV values.", dpi)
        return

    fig_width = max(9, 0.75 * len(pivot.columns))
    fig_height = max(5, 0.45 * len(pivot.index))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("CV = SD / abs(mean)")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=60, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    ax.set_title("Parameter instability by question")
    ax.set_xlabel("Question")
    ax.set_ylabel("Parameter")

    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def plot_sign_consistency_heatmap(long_df: pd.DataFrame, plot_dir: Path, dpi: int) -> None:
    out = plot_dir / "parameter_sign_consistency_heatmap_live.png"

    if long_df.empty:
        save_empty_plot(out, "Parameter sign consistency", "No usable sign-consistency data.", dpi)
        return

    df = long_df.copy()
    df["sign_consistency"] = pd.to_numeric(df["sign_consistency"], errors="coerce")

    pivot = df.pivot_table(index="parameter", columns="question", values="sign_consistency", aggfunc="median")

    if pivot.empty:
        save_empty_plot(out, "Parameter sign consistency", "No finite sign-consistency values.", dpi)
        return

    fig_width = max(9, 0.75 * len(pivot.columns))
    fig_height = max(5, 0.45 * len(pivot.index))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", vmin=0, vmax=1)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Sign consistency")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=60, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    ax.set_title("Parameter sign consistency by question")
    ax.set_xlabel("Question")
    ax.set_ylabel("Parameter")

    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def plot_top_unstable_pairs(long_df: pd.DataFrame, plot_dir: Path, dpi: int, top_n: int) -> None:
    out = plot_dir / "top_unstable_parameter_question_pairs_live.png"

    if long_df.empty:
        save_empty_plot(out, "Top unstable parameter/question pairs", "No usable CV data.", dpi)
        return

    df = long_df.copy()
    df["cv"] = pd.to_numeric(df["cv"], errors="coerce")
    df = df[np.isfinite(df["cv"])].copy()

    if df.empty:
        save_empty_plot(out, "Top unstable parameter/question pairs", "No finite CV values.", dpi)
        return

    df["label"] = df["question"].astype(str) + " / " + df["parameter"].astype(str)
    df = df.sort_values("cv", ascending=False).head(top_n).sort_values("cv", ascending=True)

    fig_height = max(6, 0.35 * len(df))
    fig, ax = plt.subplots(figsize=(10, fig_height))

    ax.barh(np.arange(len(df)), df["cv"].to_numpy())
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["label"].to_list())
    ax.set_xlabel("Coefficient of variation")
    ax.set_title(f"Top {len(df)} most unstable parameter/question pairs")
    ax.grid(True, axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def plot_loss_by_question(best_df: pd.DataFrame, plot_dir: Path, dpi: int) -> None:
    out = plot_dir / "best_loss_by_question_live.png"

    if best_df.empty:
        save_empty_plot(out, "Best loss by question", "No usable loss data.", dpi)
        return

    df = best_df.copy()
    df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
    df = df[np.isfinite(df["loss"])].copy()

    if df.empty:
        save_empty_plot(out, "Best loss by question", "No finite loss values.", dpi)
        return

    questions = sorted(df["question"].astype(str).unique())
    data = [df.loc[df["question"].astype(str) == q, "loss"].to_numpy() for q in questions]

    fig_width = max(9, 0.65 * len(questions))
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    ax.boxplot(data, labels=questions, showfliers=True)
    ax.set_ylabel("Best loss per completed set")
    ax.set_xlabel("Question")
    ax.set_title("Best-loss distribution by question")
    ax.tick_params(axis="x", rotation=60)
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def plot_parameter_boxplots(best_df: pd.DataFrame, param_cols: list[str], plot_dir: Path, dpi: int) -> None:
    box_dir = plot_dir / "parameter_boxplots_live"
    box_dir.mkdir(parents=True, exist_ok=True)

    questions = sorted(best_df["question"].astype(str).unique())

    for p in param_cols:
        tmp = best_df[["question", p]].copy()
        tmp[p] = pd.to_numeric(tmp[p], errors="coerce")
        tmp = tmp.dropna()

        if tmp.empty:
            continue

        data = [tmp.loc[tmp["question"].astype(str) == q, p].to_numpy() for q in questions]

        if sum(len(x) for x in data) == 0:
            continue

        fig_width = max(9, 0.65 * len(questions))
        fig, ax = plt.subplots(figsize=(fig_width, 6))

        ax.boxplot(data, labels=questions, showfliers=True)
        ax.set_title(f"Parameter distribution by question: {p}")
        ax.set_xlabel("Question")
        ax.set_ylabel(p)
        ax.tick_params(axis="x", rotation=60)
        ax.grid(True, axis="y", alpha=0.25)

        fig.tight_layout()
        fig.savefig(box_dir / f"parameter_boxplot_{p}.png", dpi=dpi)
        plt.close(fig)


def make_plots(
    best_df: pd.DataFrame,
    long_df: pd.DataFrame,
    progress_df: pd.DataFrame,
    param_cols: list[str],
    out_dir: Path,
    dpi: int,
    top_n: int,
) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_progress(progress_df, plot_dir, dpi)
    plot_cv_heatmap(long_df, plot_dir, dpi)
    plot_sign_consistency_heatmap(long_df, plot_dir, dpi)
    plot_top_unstable_pairs(long_df, plot_dir, dpi, top_n)
    plot_loss_by_question(best_df, plot_dir, dpi)
    plot_parameter_boxplots(best_df, param_cols, plot_dir, dpi)


def main() -> None:
    args = parse_args()

    h_dir = Path(args.h_dir).resolve()
    ecotype = args.ecotype.strip()

    if not h_dir.exists():
        print(f"ERROR: h-dir does not exist: {h_dir}")
        return

    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        if ecotype:
            out_dir = h_dir / f"parameter_stability_live_{ecotype}"
        else:
            out_dir = h_dir / "parameter_stability_live_all_ecotypes"

    out_dir.mkdir(parents=True, exist_ok=True)

    best_df = load_all_best_params(h_dir, ecotype)

    if best_df.empty:
        print("No usable optimization_results.csv files found yet.")
        print(f"h-dir: {h_dir}")
        print(f"ecotype filter: {ecotype if ecotype else '[all ecotypes]'}")
        print("Exiting without error.")
        return

    param_cols = get_parameter_columns(best_df)

    if not param_cols:
        print("No parameter columns detected yet.")
        print("Exiting without error.")
        return

    summary_df = summarize_parameter_stability(best_df, param_cols)
    long_df = make_long_stability_table(summary_df, param_cols)
    progress_df = build_progress_table(best_df, args.expected_sets)

    best_df.to_csv(out_dir / "best_parameter_rows_by_set_live.csv", index=False)
    summary_df.to_csv(out_dir / "parameter_stability_by_question_wide_live.csv", index=False)
    long_df.to_csv(out_dir / "parameter_stability_by_question_long_live.csv", index=False)
    progress_df.to_csv(out_dir / "question_progress_live.csv", index=False)

    make_plots(
        best_df=best_df,
        long_df=long_df,
        progress_df=progress_df,
        param_cols=param_cols,
        out_dir=out_dir,
        dpi=args.plot_dpi,
        top_n=args.top_n,
    )

    print("========================================")
    print(f"h-dir: {h_dir}")
    print(f"ecotype filter: {ecotype if ecotype else '[all ecotypes combined]'}")
    print(f"output directory: {out_dir}")
    print("========================================")
    print(f"usable best rows compiled: {len(best_df)}")
    print(f"questions found: {best_df['question'].nunique()}")
    print(f"sets found: {best_df['set_id'].nunique()}")
    print(f"parameters detected: {', '.join(param_cols)}")
    print("")
    print("CSV outputs:")
    print(f"  {out_dir / 'best_parameter_rows_by_set_live.csv'}")
    print(f"  {out_dir / 'parameter_stability_by_question_wide_live.csv'}")
    print(f"  {out_dir / 'parameter_stability_by_question_long_live.csv'}")
    print(f"  {out_dir / 'question_progress_live.csv'}")
    print("")
    print("Plot outputs:")
    print(f"  {out_dir / 'plots' / 'question_progress_live.png'}")
    print(f"  {out_dir / 'plots' / 'parameter_cv_heatmap_live.png'}")
    print(f"  {out_dir / 'plots' / 'parameter_sign_consistency_heatmap_live.png'}")
    print(f"  {out_dir / 'plots' / 'top_unstable_parameter_question_pairs_live.png'}")
    print(f"  {out_dir / 'plots' / 'best_loss_by_question_live.png'}")
    print(f"  {out_dir / 'plots' / 'parameter_boxplots_live'}")
    print("")
    print("Progress:")
    print(progress_df.to_string(index=False))
    print("")
    print("Most unstable question-parameter pairs so far:")
    show = long_df.sort_values("cv", ascending=False, na_position="last").head(args.top_n)
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()