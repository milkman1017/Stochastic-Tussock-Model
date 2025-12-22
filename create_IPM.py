import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
import os
from sklearn.linear_model import LogisticRegression, LinearRegression
from scipy.optimize import curve_fit
import statsmodels.api as sm
import statsmodels.formula.api as smf

def load_data(file_path):
    data = pd.read_csv(file_path)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(subset=['2016 Leaf Area'], inplace=True)
    return data




def plot_growth(data):

    os.makedirs("IPM", exist_ok=True)

    growth = data[data['2017 Leaf Area'] > 0].copy()
    ecotypes = growth['Garden'].unique()
    colors = sns.color_palette("tab10", len(ecotypes))

    results = []

    # ======================================================
    # ========== FIT MODELS & PRODUCE DEGREE PLOTS =========
    # ======================================================
    for degree in [1, 2, 3, 4]:

        plt.figure(figsize=(7,6))

        for color, ecotype in zip(colors, ecotypes):

            subset = growth[growth['Garden'] == ecotype]
            x = subset['2016 Leaf Area'].values.reshape(-1,1)
            y = subset['2017 Leaf Area'].values

            poly = PolynomialFeatures(degree)
            X_poly = poly.fit_transform(x)
            model = sm.OLS(y, X_poly).fit()

            # Prediction curve
            xline = np.linspace(x.min(), x.max(), 200).reshape(-1,1)
            Xline_poly = poly.transform(xline)
            yline = model.predict(Xline_poly)

            # Residual SD band
            sd = np.sqrt(np.mean(model.resid**2))
            plt.fill_between(xline.flatten(), yline-sd, yline+sd, color=color, alpha=0.15)

            # Scatter + curve
            plt.scatter(x, y, s=5, color=color, alpha=0.5, label=ecotype if degree==1 else None)
            plt.plot(xline, yline, color=color)

            # Store stats
            results.append({
                "ecotype": ecotype,
                "degree": degree,
                "AIC": model.aic,
                "BIC": model.bic,
                "R2": model.rsquared,
                "resid_SD": sd
            })

        plt.title(f"Growth Fit (Degree {degree})")
        plt.xlabel("2016 Leaf Area")
        plt.ylabel("2017 Leaf Area")
        plt.legend(title="Ecotype")
        plt.tight_layout()
        plt.savefig(f"IPM/growth_degree_{degree}.png")
        plt.close()

        # =====================================
        # ========== RESIDUAL PLOT ============
        # =====================================
        plt.figure(figsize=(7,6))
        for color, ecotype in zip(colors, ecotypes):
            subset = growth[growth['Garden'] == ecotype]
            x = subset['2016 Leaf Area'].values.reshape(-1,1)
            y = subset['2017 Leaf Area'].values

            poly = PolynomialFeatures(degree)
            X_poly = poly.fit_transform(x)
            model = sm.OLS(y, X_poly).fit()

            plt.scatter(model.fittedvalues, model.resid, s=5, color=color, alpha=0.6, label=ecotype)

        plt.axhline(0, color='black')
        plt.title(f"Residual Diagnostics (Degree {degree})")
        plt.xlabel("Fitted Values")
        plt.ylabel("Residuals")
        plt.legend(title="Ecotype")
        plt.tight_layout()
        plt.savefig(f"IPM/growth_residuals_degree_{degree}.png")
        plt.close()

    # ======================================================
    # ========== SAVE MODEL COMPARISON TABLE ============
    # ======================================================
    results_df = pd.DataFrame(results)
    results_df.to_csv("IPM/growth_model_comparison.csv", index=False)

    # Compute ΔAIC per ecotype
    results_df["deltaAIC"] = results_df.groupby("ecotype")["AIC"].transform(lambda x: x - x.min())
    results_df["deltaBIC"] = results_df.groupby("ecotype")["BIC"].transform(lambda x: x - x.min())

    results_df.to_csv("IPM/growth_model_summary.csv", index=False)

    # ===============================================
    # ========= AIC/BIC PROFILES FOR PAPER ==========
    # ===============================================
    plt.figure(figsize=(8,6))
    for ecotype, color in zip(ecotypes, colors):
        sub = results_df[results_df.ecotype == ecotype]
        plt.plot(sub.degree, sub.AIC, marker='o', color=color, label=ecotype)
    plt.xlabel("Polynomial Degree")
    plt.ylabel("AIC")
    plt.title("AIC Profile Across Degrees")
    plt.legend()
    plt.tight_layout()
    plt.savefig("IPM/growth_AIC_profiles.png")
    plt.close()

    plt.figure(figsize=(8,6))
    for ecotype, color in zip(ecotypes, colors):
        sub = results_df[results_df.ecotype == ecotype]
        plt.plot(sub.degree, sub.BIC, marker='o', color=color, label=ecotype)
    plt.xlabel("Polynomial Degree")
    plt.ylabel("BIC")
    plt.title("BIC Profile Across Degrees")
    plt.legend()
    plt.tight_layout()
    plt.savefig("IPM/growth_BIC_profiles.png")
    plt.close()

    # ===============================================
    # ========= ΔAIC HEATMAP (paper-ready) ==========
    # ===============================================
    pivot = results_df.pivot(index="ecotype", columns="degree", values="deltaAIC")
    plt.figure(figsize=(8,5))
    sns.heatmap(pivot, annot=True, cmap="viridis_r", cbar_kws={'label':'ΔAIC'})
    plt.title("ΔAIC per Ecotype per Polynomial Degree")
    plt.tight_layout()
    plt.savefig("IPM/growth_deltaAIC_heatmap.png")
    plt.close()

    # ===============================================
    # ======== PRINT BEST MODEL PER ECOTYPE =========
    # ===============================================
    print("\n=== Best Models Per Ecotype (AIC) ===")
    for ecotype in ecotypes:
        sub = results_df[results_df["ecotype"] == ecotype]
        best = sub.loc[sub["AIC"].idxmin()]
        print(f"{ecotype}: degree {int(best.degree)} "
              f"(AIC={best.AIC:.2f}, BIC={best.BIC:.2f}, SD={best.resid_SD:.3f})")

    print("\nSaved:")
    print(" - growth_model_comparison.csv")
    print(" - growth_model_summary.csv")
    print(" - growth_degree_X.png")
    print(" - growth_residuals_degree_X.png")
    print(" - growth_AIC_profiles.png")
    print(" - growth_BIC_profiles.png")
    print(" - growth_deltaAIC_heatmap.png")

    return results_df

import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

def plot_survival(data):
    surv = data.copy()

    # Recode survival
    surv['surv'] = surv['2017 Type (1-3) 1=dead'].replace({1:0, 2:1, 3:1})

    ecotypes = surv['Garden'].unique()
    colors = sns.color_palette("tab10", len(ecotypes))  # one unique color per ecotype

    plt.figure(figsize=(7,6))

    for color, ecotype in zip(colors, ecotypes):
        subset = surv[surv['Garden'] == ecotype]

        x = subset['2016 Leaf Area'].values.reshape(-1,1)
        y = subset['surv'].values

        # Logistic regression — skip if all 0s or all 1s
        if len(np.unique(y)) < 2:
            continue

        logreg = LogisticRegression().fit(x, y)

        # Prediction line
        xline = np.linspace(x.min(), x.max(), 300).reshape(-1,1)
        yline = logreg.predict_proba(xline)[:,1]

        # Scatter + logistic curve in same color
        plt.scatter(x, y, s=5, color=color, alpha=0.7, label=ecotype)
        plt.plot(xline, yline, color=color, linewidth=1.8)

    plt.xlabel("2016 Leaf Area")
    plt.ylabel("Survival Probability")
    plt.title("Ecotype-specific Logistic Survival Curves")
    plt.legend(title="Ecotype")
    plt.tight_layout()
    plt.savefig("IPM/survival_logistic.png")
    plt.close()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.count_model import ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.count_model import ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP


from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_tillering(data, output="IPM/tillering_sequential.png"):

    df = data.copy()
    df["tiller"] = df["2017 # S0 (current yr new)"].astype(int)

    ecotypes = df["Garden"].unique()
    colors = sns.color_palette("tab10", len(ecotypes))

    max_tillers = df["tiller"].max()
    print(f"Max tillers observed = {max_tillers}")

    # 2×2 plot grid if nmax ≤ 4; else dynamic
    ncols = 2
    nrows = int(np.ceil(max_tillers / 2))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4*nrows))
    axes = axes.flatten()

    # ------------------------------------------------------------
    # For each threshold k (1..max_tillers), fit logistic:
    # P(Y ≥ k | x)
    # ------------------------------------------------------------
    for k in range(1, max_tillers+1):
        ax = axes[k-1]
        ax.set_title(f"P(tillers ≥ {k})")

        for color, ecotype in zip(colors, ecotypes):
            sub = df[df["Garden"] == ecotype]

            x = sub["2016 Leaf Area"].values.reshape(-1,1)
            y = (sub["tiller"] >= k).astype(int)

            # scatter
            ax.scatter(sub["2016 Leaf Area"], y, s=8, alpha=0.5, color=color)

            # If no variation skip
            if len(np.unique(y)) < 2:
                continue

            # logistic regression
            clf = LogisticRegression().fit(x, y)

            xline = np.linspace(x.min(), x.max(), 300).reshape(-1,1)
            yline = clf.predict_proba(xline)[:,1]

            ax.plot(xline, yline, color=color, linewidth=2, label=ecotype if k==1 else "_nolabel_")

        ax.set_xlabel("2016 Leaf Area")
        ax.set_ylabel(f"P(Y ≥ {k})")

    # master legend only once
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Ecotype", loc="upper right")

    plt.tight_layout()
    plt.savefig(output)
    plt.close()

    print(f"Saved sequential plot: {output}")




def plot_flower(data):
    flower = data.copy()

    # Recode survival
    flower['flower'] = flower['2017 Flower (y/n)'].replace({'y':1, 'n':0})

    ecotypes = flower['Garden'].unique()
    colors = sns.color_palette("tab10", len(ecotypes))  # one unique color per ecotype

    plt.figure(figsize=(7,6))

    for color, ecotype in zip(colors, ecotypes):
        subset = flower[flower['Garden'] == ecotype]

        x = subset['2016 Leaf Area'].values.reshape(-1,1)
        y = subset['flower'].values

        # Logistic regression — skip if all 0s or all 1s
        if len(np.unique(y)) < 2:
            continue

        logreg = LogisticRegression().fit(x, y)

        # Prediction line
        xline = np.linspace(x.min(), x.max(), 300).reshape(-1,1)
        yline = logreg.predict_proba(xline)[:,1]

        # Scatter + logistic curve in same color
        plt.scatter(x, y, s=5, color=color, alpha=0.7, label=ecotype)
        plt.plot(xline, yline, color=color, linewidth=1.8)

    plt.xlabel("2016 Leaf Area")
    plt.ylabel("Flowering Probability")
    plt.title("Ecotype-specific Logistic Flowering Curves")
    plt.legend(title="Ecotype")
    plt.tight_layout()
    plt.savefig("IPM/flower_logistic.png")
    plt.close()
    

def main():
    data = load_data('16-17_IPM_data.csv')
    print(data.columns)
    # leaf_area_2016, leaf_area_2017, s0_recruits, s1_recruits, gardens = extract_columns(data)

    plot_growth(data)
    plot_survival(data)
    plot_flower(data)
    plot_tillering(data)

if __name__ == "__main__":
    main()
