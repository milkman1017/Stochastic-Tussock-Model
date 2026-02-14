#!/usr/bin/env python3
"""
ONE-SCRIPT PIPELINE

1) Read HistLeafPhenol85-25.csv
2) Filter to:
   - native tussocks = Site matches "home Site" for each Src (home inferred from tr==0 rows)
   - control only (gr == 'control', case-insensitive)
   - years Yrm in {2022, 2023, 2024}
   - keep only tillers present in consecutive years (2022->2023 and/or 2023->2024)
3) Save filtered CSV
4) From filtered CSV, for each (Src,Rep,Plot,Ind,Tiller,Yrm):
   - find doy where total green length (sum gl across leaves) is maximal
   - at that doy compute:
       I = (sum of two longest leaves' gl) * (number of leaves)
       Area = 8.4477833 + 5.3464264*I - 0.0056047*I^2
5) Save area summary CSV
6) Save verification plots for BOTH steps
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List


def require_cols(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: {}\nFound columns: {}".format(missing, list(df.columns))
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default="input_data/HistLeafPhenol85-25.csv",
        help="Input CSV path (default: input_data/HistLeafPhenol85-25.csv)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="native_control_consecutive_and_leaf_area",
        help="Output directory for CSVs + plots",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling verification plots",
    )
    args = parser.parse_args()

    inpath = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inpath)

    # ---- Column checks ----
    require_cols(df, ["Src", "Rep", "Plot", "Ind", "Tiller", "Leaf", "doy", "gl", "Site", "gr", "tr", "Yrm"])

    key_cols = ["Src", "Rep", "Plot", "Ind", "Tiller"]
    key_year_cols = key_cols + ["Yrm"]

    # ============================================================
    # PART A — FILTERING
    # ============================================================

    # A1) Infer "home Site" for each Src from tr==0 rows
    df_tr0 = df[df["tr"] == 0].copy()
    if df_tr0.empty:
        raise ValueError("No rows found with tr==0; cannot infer home Site per Src.")

    # If a Src has >1 unique Site in tr==0, that's ambiguous; fail loudly.
    src_site_counts = df_tr0.groupby("Src")["Site"].nunique()
    ambiguous_src = src_site_counts[src_site_counts > 1].index.tolist()
    if ambiguous_src:
        raise ValueError(
            "Ambiguous home Site for these Src values (multiple Sites in tr==0 rows): {}".format(ambiguous_src)
        )

    home_site = df_tr0.groupby("Src")["Site"].first().to_dict()

    # A2) Native filter: Site matches home_site(Src)
    df_native = df[df["Site"] == df["Src"].map(home_site)].copy()

    # A3) Control only (case-insensitive)
    df_native["gr_lower"] = df_native["gr"].astype(str).str.strip().str.lower()
    df_ctrl = df_native[df_native["gr_lower"] == "control"].copy()

    # A4) Restrict years
    years_keep = [2022, 2023, 2024]
    df_y = df_ctrl[df_ctrl["Yrm"].isin(years_keep)].copy()

    # A5) Keep tillers present in consecutive years
    df_y = df_y.sort_values(key_year_cols + ["doy", "Leaf"])
    df_y["_year_diff"] = df_y.groupby(key_cols)["Yrm"].diff()

    valid_tillers = df_y.loc[df_y["_year_diff"] == 1, key_cols].drop_duplicates()
    df_filtered = df_y.merge(valid_tillers, on=key_cols, how="inner").copy()

    # Save filtered CSV
    filtered_csv = outdir / "native_same_source_site_control_2022_2024_consecutive_years.csv"
    df_filtered.to_csv(str(filtered_csv), index=False)

    # ============================================================
    # PART A — FILTER VERIFICATION PLOTS
    # ============================================================

    # Plot A1: rows by year before vs after consecutive-year filter
    plt.figure()
    years = np.array(years_keep)
    before = df_y["Yrm"].value_counts().reindex(years, fill_value=0).values
    after = df_filtered["Yrm"].value_counts().reindex(years, fill_value=0).values
    x = np.arange(len(years))
    plt.bar(x - 0.2, before, width=0.4, label="Before consecutive-year filter")
    plt.bar(x + 0.2, after, width=0.4, label="After consecutive-year filter")
    plt.xticks(x, years)
    plt.xlabel("Year (Yrm)")
    plt.ylabel("Row count")
    plt.title("Filter check: rows by year (native+control), before vs after consecutive-year filter")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(outdir / "verify_filter_rows_by_year_before_after.png"), dpi=200)
    plt.close()

    # Plot A2: # unique years per tiller (after filter) should be 2 or 3
    years_per_tiller = df_filtered.groupby(key_cols)["Yrm"].apply(lambda s: len(set(s)))
    plt.figure()
    plt.hist(years_per_tiller.values, bins=np.arange(0.5, 4.5, 1.0))
    plt.xlabel("# unique years per tiller (after filter)")
    plt.ylabel("Count of tillers")
    plt.title("Filter check: expected only 2 or 3")
    plt.tight_layout()
    plt.savefig(str(outdir / "verify_filter_years_per_tiller_hist.png"), dpi=200)
    plt.close()

    # Plot A3: timeline scatter for random sample of tillers (after filter)
    unique_tillers = years_per_tiller.index.to_frame(index=False)
    n_sample = min(200, len(unique_tillers))
    sample = unique_tillers.sample(n=n_sample, random_state=args.seed).reset_index(drop=True)
    sample["sample_id"] = np.arange(len(sample))
    df_sample = df_filtered.merge(sample, on=key_cols, how="inner")

    plt.figure()
    plt.scatter(df_sample["Yrm"].values, df_sample["sample_id"].values, s=10)
    plt.yticks([])
    plt.xlabel("Year (Yrm)")
    plt.ylabel("Random tillers (index)")
    plt.title("Filter check: each tiller should appear in consecutive-year pairs")
    plt.tight_layout()
    plt.savefig(str(outdir / "verify_filter_tiller_timelines_sample.png"), dpi=200)
    plt.close()

    # ============================================================
    # PART B — MAX-DOY and LEAF AREA CALCULATION
    # ============================================================

    # B1) Total GL per doy per tiller-year
    gl_by_doy = (
        df_filtered.groupby(key_year_cols + ["doy"], as_index=False)["gl"]
        .sum()
        .rename(columns={"gl": "total_gl"})
    )

    # B2) DOY of max total GL per tiller-year
    idx = gl_by_doy.groupby(key_year_cols)["total_gl"].idxmax()
    doy_max = gl_by_doy.loc[idx].copy()

    # B3) Subset to peak DOY rows
    df_peak = df_filtered.merge(
        doy_max[key_year_cols + ["doy", "total_gl"]],
        on=key_year_cols + ["doy"],
        how="inner"
    )

    # B4) Compute I and area per tiller-year
    records = []
    for keys, g in df_peak.groupby(key_year_cols):
        n_leaves = g["Leaf"].nunique()
        g_sorted = g.sort_values("gl", ascending=False)
        sum_two_longest = g_sorted["gl"].iloc[:2].sum()

        I = sum_two_longest * n_leaves
        area = 8.4477833 + 5.3464264 * I - 0.0056047 * (I ** 2)

        rec = dict(zip(key_year_cols, keys))
        rec.update({
            "doy_max_total_gl": int(g_sorted["doy"].iloc[0]),
            "total_gl_at_doy_max": float(g_sorted["total_gl"].iloc[0]),
            "n_leaves_at_doy_max": int(n_leaves),
            "sum_two_longest_gl_at_doy_max": float(sum_two_longest),
            "I": float(I),
            "tiller_leaf_surface_area": float(area),
        })
        records.append(rec)

    df_area = pd.DataFrame.from_records(records)

    area_csv = outdir / "tiller_leaf_surface_area_at_max_total_gl.csv"
    df_area.to_csv(str(area_csv), index=False)

    # ============================================================
    # PART B — VERIFICATION PLOTS
    # ============================================================

    # Plot B1: total GL vs DOY for a sample of tiller-years, with selected doy marked
    unique_tiller_years = df_area[key_year_cols].drop_duplicates()
    n_sample_ty = min(25, len(unique_tiller_years))
    sample_ty = unique_tiller_years.sample(n=n_sample_ty, random_state=args.seed)

    plt.figure()
    for _, row in sample_ty.iterrows():
        mask = np.ones(len(df_filtered), dtype=bool)
        for c in key_year_cols:
            mask &= (df_filtered[c].values == row[c])

        tmp = df_filtered.loc[mask].groupby("doy")["gl"].sum().reset_index()

        peak_doy = df_area.loc[
            (df_area[key_year_cols] == row[key_year_cols]).all(axis=1),
            "doy_max_total_gl"
        ].iloc[0]

        plt.plot(tmp["doy"], tmp["gl"], alpha=0.4)
        plt.axvline(peak_doy, alpha=0.3)

    plt.xlabel("DOY")
    plt.ylabel("Total green length (sum gl across leaves)")
    plt.title("Area check: DOY chosen at maximum total green length")
    plt.tight_layout()
    plt.savefig(str(outdir / "verify_area_total_gl_vs_doy.png"), dpi=200)
    plt.close()

    # Plot B2: leaf count distribution at selected doy
    plt.figure()
    plt.hist(df_area["n_leaves_at_doy_max"], bins=20)
    plt.xlabel("Number of leaves at DOY of max total GL")
    plt.ylabel("Count of tiller-years")
    plt.title("Area check: leaf counts used in I")
    plt.tight_layout()
    plt.savefig(str(outdir / "verify_area_leaf_counts.png"), dpi=200)
    plt.close()

    # Plot B3: I vs computed leaf surface area
    plt.figure()
    plt.scatter(df_area["I"], df_area["tiller_leaf_surface_area"], s=10)
    plt.xlabel("Index I = (sum of 2 longest leaves) * (number of leaves)")
    plt.ylabel("Tiller Leaf Surface Area")
    plt.title("Area check: quadratic relationship")
    plt.tight_layout()
    plt.savefig(str(outdir / "verify_area_I_vs_surface_area.png"), dpi=200)
    plt.close()

    # ============================================================
    # PRINT SUMMARY
    # ============================================================
    native_ok = (df_filtered["Site"] == df_filtered["Src"].map(home_site)).mean()

    print("\n=== Home site map inferred from tr==0 ===")
    for k, v in sorted(home_site.items()):
        print("  {} -> {}".format(k, v))

    print("\n=== Filter summaries ===")
    print("Rows after native+control+year restriction:", len(df_y))
    print("Rows after consecutive-year tiller filter:", len(df_filtered))
    print("Sanity (Site == home_site(Src)) in filtered rows: {:.6f} (expect 1.0)".format(native_ok))

    print("\n=== Outputs ===")
    print("Filtered CSV:", filtered_csv)
    print("Area summary CSV:", area_csv)
    print("Plots written to:", outdir.resolve())
    print("Done.")


if __name__ == "__main__":
    main()
