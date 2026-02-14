#!/usr/bin/env python3
"""
Combine IPM datasets into ONE transition-format CSV.

Inputs (examples):
- 16-17_IPM_data.csv  (wide format with 2016/2017 columns)
- 2022-2024_IPM_data.csv (long-ish summary per year with area already computed)

Output:
- combined_IPM_transitions.csv

Each output row = one transition: year n -> year n+1
Generic columns: area_n, area_n_plus_1, I_n, I_n_plus_1, etc.
Actual years retained in: year_n, year_n_plus_1

Survival/tillering:
- included if present in source; otherwise left empty.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def require_cols(df, cols, name="dataframe"):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing required columns: {missing}\nFound: {list(df.columns)}")


def build_transitions_2016_2017(df):
    """
    Convert the 16-17 wide-format dataset into transition rows.
    Expected columns (from your file):
      'MATLAB ID', 'Garden',
      '2016 Index', '2016 Leaf Area', '2016 Total Leaf Count', '2016 LL1+LL2',
      '2017 Index', '2017 Leaf Area', '2017 Total Leaf Count', '2017 LL1+LL2',
      optional: '2017 Type (1-3) 1=dead', '2017 # S0 (current yr new)', etc.
    """
    require_cols(df, ["MATLAB ID", "Garden"], name="16-17")

    out = pd.DataFrame()
    out["dataset"] = "2016-2017"
    out["id"] = df["MATLAB ID"]
    out["Src"] = df["Garden"]

    out["year_n"] = 2016
    out["year_n_plus_1"] = 2017

    # map common IPM state vars if present
    if "2016 Leaf Area" in df.columns:
        out["area_n"] = df["2016 Leaf Area"]
    if "2017 Leaf Area" in df.columns:
        out["area_n_plus_1"] = df["2017 Leaf Area"]

    if "2016 Index" in df.columns:
        out["I_n"] = df["2016 Index"]
    if "2017 Index" in df.columns:
        out["I_n_plus_1"] = df["2017 Index"]

    if "2016 Total Leaf Count" in df.columns:
        out["n_leaves_n"] = df["2016 Total Leaf Count"]
    if "2017 Total Leaf Count" in df.columns:
        out["n_leaves_n_plus_1"] = df["2017 Total Leaf Count"]

    if "2016 LL1+LL2" in df.columns:
        out["sum_two_longest_n"] = df["2016 LL1+LL2"]
    if "2017 LL1+LL2" in df.columns:
        out["sum_two_longest_n_plus_1"] = df["2017 LL1+LL2"]

    # survival if present (Type==1 means dead)
    # If that column exists, we populate survival; if not, leave empty.
    if "2017 Type (1-3) 1=dead" in df.columns:
        dead = df["2017 Type (1-3) 1=dead"]
        # treat 1 as dead; anything else as alive (including NaN -> NaN survival)
        surv = np.where(dead.isna(), np.nan, np.where(dead.astype(float) == 1.0, 0.0, 1.0))
        out["survival"] = surv

    # tillering if present (current yr new)
    if "2017 # S0 (current yr new)" in df.columns:
        out["tillering"] = df["2017 # S0 (current yr new)"]

    # keep any extra fields that might be useful (optional)
    # Example: stage class, flower, comments
    for extra in [
        "2017 Stage Class",
        "2017 Flower (y/n)",
        "2017 Comment",
        "2017 # S1 (last yr new)?",
        "2016 NumOld", "2016 NumNew",
        "2017 NumOld", "2017 NumNew",
    ]:
        if extra in df.columns:
            out[extra] = df[extra]

    return out


def build_transitions_2022_2024(df):
    """
    Convert the 2022-2024 per-year summary dataset into transition rows.

    Expected columns (from your file):
      Src, Rep, Plot, Ind, Tiller, Yrm,
      doy_max_total_gl, total_gl_at_doy_max, n_leaves_at_doy_max,
      sum_two_longest_gl_at_doy_max, I, tiller_leaf_surface_area
    """
    require_cols(df, ["Src", "Rep", "Plot", "Ind", "Tiller", "Yrm", "tiller_leaf_surface_area"], name="2022-2024")

    key = ["Src", "Rep", "Plot", "Ind", "Tiller"]

    df = df.copy()
    df = df.sort_values(key + ["Yrm"])

    # Build transitions for consecutive years present in the table
    rows = []
    for k, g in df.groupby(key, sort=False):
        years = g["Yrm"].values
        for i in range(len(g) - 1):
            y0 = int(years[i])
            y1 = int(years[i + 1])
            # only keep consecutive year transitions
            if y1 != y0 + 1:
                continue

            r0 = g.iloc[i]
            r1 = g.iloc[i + 1]

            rec = {
                "dataset": "2022-2024",
                "Src": k[0],
                "Rep": k[1],
                "Plot": k[2],
                "Ind": k[3],
                "Tiller": k[4],
                "year_n": y0,
                "year_n_plus_1": y1,
                "area_n": r0.get("tiller_leaf_surface_area", np.nan),
                "area_n_plus_1": r1.get("tiller_leaf_surface_area", np.nan),
                "I_n": r0.get("I", np.nan),
                "I_n_plus_1": r1.get("I", np.nan),
                "n_leaves_n": r0.get("n_leaves_at_doy_max", np.nan),
                "n_leaves_n_plus_1": r1.get("n_leaves_at_doy_max", np.nan),
                "sum_two_longest_n": r0.get("sum_two_longest_gl_at_doy_max", np.nan),
                "sum_two_longest_n_plus_1": r1.get("sum_two_longest_gl_at_doy_max", np.nan),
                "doy_peak_n": r0.get("doy_max_total_gl", np.nan),
                "doy_peak_n_plus_1": r1.get("doy_max_total_gl", np.nan),
                "total_gl_peak_n": r0.get("total_gl_at_doy_max", np.nan),
                "total_gl_peak_n_plus_1": r1.get("total_gl_at_doy_max", np.nan),
            }

            # Survival/tillering: only include if explicitly present in source dataset columns.
            # (Your 2022-2024 file does NOT have them, so these will remain NaN.)
            if "survival" in df.columns:
                rec["survival"] = r1.get("survival", np.nan)
            if "tillering" in df.columns:
                rec["tillering"] = r1.get("tillering", np.nan)

            rows.append(rec)

    out = pd.DataFrame(rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ipm_1", type=str, required=True, help="Path to 16-17_IPM_data.csv")
    ap.add_argument("--ipm_2", type=str, required=True, help="Path to 2022-2024_IPM_data.csv")
    ap.add_argument("--out", type=str, default="combined_IPM_transitions.csv", help="Output CSV path")
    args = ap.parse_args()

    df1 = pd.read_csv(args.ipm_1)
    df2 = pd.read_csv(args.ipm_2)

    t1 = build_transitions_2016_2017(df1)
    t2 = build_transitions_2022_2024(df2)

    # Make columns align (union of columns, missing filled with NaN)
    all_cols = sorted(set(t1.columns).union(set(t2.columns)))
    t1 = t1.reindex(columns=all_cols)
    t2 = t2.reindex(columns=all_cols)

    combined = pd.concat([t1, t2], ignore_index=True)

    # A convenient generic "transition label" for IPM code
    combined["transition"] = "n_to_n_plus_1"

    # Write
    outpath = Path(args.out)
    combined.to_csv(outpath, index=False)

    # Quick console summary
    print("\nWrote:", outpath.resolve())
    print("Rows:", len(combined))
    if "survival" in combined.columns:
        print("Survival non-missing:", combined["survival"].notna().sum())
    if "tillering" in combined.columns:
        print("Tillering non-missing:", combined["tillering"].notna().sum())
    print("\nPreview:")
    print(combined.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
