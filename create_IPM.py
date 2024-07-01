import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from scipy.optimize import curve_fit

def load_data(file_path):
    return pd.read_csv(file_path)

def extract_columns(data):
    leaf_area_2016 = data['2016 Leaf Area']
    leaf_area_2017 = data['2017 Leaf Area'].copy()
    s0_recruits = data['2017 # S0 (current yr new)']
    s1_recruits = data['2017 # S1 (last yr new)?']
    
    # Set negative values in 2017 leaf area to 0
    leaf_area_2017[leaf_area_2017 < 0] = 0
    
    return leaf_area_2016, leaf_area_2017, s0_recruits, s1_recruits, data['Garden']

def growth_model(x, a, b, c):
    return a * x**b + c

def growth_function(leaf_area_2016, leaf_area_2017, x):
    params, _ = curve_fit(growth_model, leaf_area_2016, leaf_area_2017, maxfev=10000)
    return growth_model(x, *params), params

def survival_function_continuous(leaf_area_2016, leaf_area_2017, x):
    survival_outcome = (leaf_area_2017 > 0).astype(int)
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(leaf_area_2016.values.reshape(-1, 1), survival_outcome)
    return log_reg.predict_proba(x.reshape(-1, 1))[:, 1], log_reg

def reproduction_function_continuous(leaf_area_2016, s0_recruits, s1_recruits, x):
    total_recruits = s0_recruits + s1_recruits
    reproduction_outcome = (total_recruits > 0).astype(int)
    
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(leaf_area_2016.values.reshape(-1, 1), reproduction_outcome)
    return log_reg.predict_proba(x.reshape(-1, 1))[:, 1] * total_recruits.mean(), log_reg

def plot_reproduction_probability(min_leaf_area, max_leaf_area, reproduction_function_continuous, leaf_area_2016, s0_recruits, s1_recruits, gardens, nbins=100):
    leaf_area_range = np.linspace(min_leaf_area, max_leaf_area, nbins)

    plt.figure(figsize=(10, 6))
    for garden in gardens.unique():
        garden_mask = gardens == garden
        reproduction_probs_continuous, _ = reproduction_function_continuous(
            leaf_area_2016[garden_mask], s0_recruits[garden_mask], s1_recruits[garden_mask], leaf_area_range)
        plt.plot(leaf_area_range, reproduction_probs_continuous, linestyle='-', label=garden)

    plt.xlabel('Leaf Area')
    plt.ylabel('Reproduction Probability')
    plt.title('Reproduction Probability vs Leaf Area')
    plt.legend()
    plt.grid(True)
    plt.show()

def project(leaf_area_bins, leaf_area_2016, garden_mask, leaf_area_2017, s0_recruits, s1_recruits, kernels, garden):
    X, Y = np.meshgrid(leaf_area_bins, leaf_area_bins)
    growth_probs, growth_params = growth_function(leaf_area_2016[garden_mask], leaf_area_2017[garden_mask], X)
    survival_probs, _ = survival_function_continuous(leaf_area_2016[garden_mask], leaf_area_2017[garden_mask], X.flatten())
    survival_probs = survival_probs.reshape(X.shape)
    reproduction_probs, _ = reproduction_function_continuous(leaf_area_2016[garden_mask], s0_recruits[garden_mask], s1_recruits[garden_mask], X.flatten())
    reproduction_probs = reproduction_probs.reshape(X.shape)
    kernel = norm.pdf(Y, loc=growth_probs, scale=5) * survival_probs + reproduction_probs
    kernel /= kernel.sum(axis=0)
    kernels[garden] = kernel
    return kernels, growth_params

def plot_ipm_kernel(min_leaf_area, max_leaf_area, growth_function, leaf_area_2016, leaf_area_2017, survival_function_continuous, reproduction_function_continuous, s0_recruits, s1_recruits, gardens, nbins=100):
    n_bins = nbins
    leaf_area_bins = np.linspace(min_leaf_area, max_leaf_area, n_bins)
    
    unique_gardens = gardens.unique()
    fig, axs = plt.subplots(1, len(unique_gardens), figsize=(15, 6), sharey=True)

    for i, garden in enumerate(unique_gardens):
        garden_mask = gardens == garden
        kernels, _ = project(leaf_area_bins, leaf_area_2016, garden_mask, leaf_area_2017, s0_recruits, s1_recruits, {}, garden)

        ax = axs[i]
        im = ax.imshow(kernels[garden], extent=[min_leaf_area, max_leaf_area, min_leaf_area, max_leaf_area], origin='lower', aspect='auto')
        ax.set_title(garden)
        ax.set_xlabel('Leaf Area (t)')
        if i == 0:
            ax.set_ylabel('Leaf Area (t+1)')
        fig.colorbar(im, ax=ax, label='Transition Probability')

    plt.suptitle('IPM Kernel by Garden')
    plt.show()

def plot_survival_probability(min_leaf_area, max_leaf_area, survival_function_continuous, leaf_area_2016, leaf_area_2017, gardens):
    leaf_area_range = np.linspace(min_leaf_area, max_leaf_area, 500)

    plt.figure(figsize=(10, 6))
    for garden in gardens.unique():
        garden_mask = gardens == garden
        survival_probs_continuous, _ = survival_function_continuous(
            leaf_area_2016[garden_mask], leaf_area_2017[garden_mask], leaf_area_range)
        plt.plot(leaf_area_range, survival_probs_continuous, linestyle='-', label=garden)

    plt.xlabel('Leaf Area (2016)')
    plt.ylabel('Survival Probability (2017)')
    plt.title('Survival Probability vs Leaf Area (2016)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_stable_size_distribution(kernels, leaf_area_bins, gardens):
    plt.figure(figsize=(10, 6))
    for garden, kernel in kernels.items():
        eigenvalues, eigenvectors = np.linalg.eig(kernel)
        stable_distribution = np.real(eigenvectors[:, np.argmax(eigenvalues)])
        stable_distribution /= stable_distribution.sum()
        plt.plot(leaf_area_bins, stable_distribution, linestyle='-', label=garden)

    plt.xlabel('Leaf Area')
    plt.ylabel('Stable Size Distribution')
    plt.title('Stable Size Distribution vs Leaf Area')
    plt.legend()
    plt.grid(True)
    plt.show()

def sensitivity_analysis(leaf_area_bins, leaf_area_2016, garden_mask, leaf_area_2017, s0_recruits, s1_recruits, kernels, garden, perturbations):
    original_kernels, growth_params = project(leaf_area_bins, leaf_area_2016, garden_mask, leaf_area_2017, s0_recruits, s1_recruits, kernels, garden)

    perturbed_kernels = {}

    # Perturb growth function parameters
    for i, param in enumerate(growth_params):
        for perturb in perturbations:
            perturbed_params = growth_params.copy()
            perturbed_params[i] *= perturb
            perturbed_growth_probs = growth_model(leaf_area_bins, *perturbed_params)

            perturbed_kernel = norm.pdf(leaf_area_bins[:, None], loc=perturbed_growth_probs, scale=5) * \
                               survival_function_continuous(leaf_area_2016[garden_mask], leaf_area_2017[garden_mask], leaf_area_bins)[0][:, None] + \
                               reproduction_function_continuous(leaf_area_2016[garden_mask], s0_recruits[garden_mask], s1_recruits[garden_mask], leaf_area_bins)[0][:, None]
            perturbed_kernel /= perturbed_kernel.sum(axis=0)
            perturbed_kernels[f'growth_param_{i}_perturb_{perturb}'] = perturbed_kernel

    # Perturb logistic regression coefficients
    for perturb in perturbations:
        survival_probs, survival_log_reg = survival_function_continuous(leaf_area_2016[garden_mask], leaf_area_2017[garden_mask], leaf_area_bins)
        reproduction_probs, reproduction_log_reg = reproduction_function_continuous(leaf_area_2016[garden_mask], s0_recruits[garden_mask], s1_recruits[garden_mask], leaf_area_bins)

        survival_log_reg.coef_ *= perturb
        reproduction_log_reg.coef_ *= perturb

        perturbed_survival_probs = survival_log_reg.predict_proba(leaf_area_bins[:, None])[:, 1]
        perturbed_reproduction_probs = reproduction_log_reg.predict_proba(leaf_area_bins[:, None])[:, 1] * (s0_recruits[garden_mask] + s1_recruits[garden_mask]).mean()

        perturbed_kernel = norm.pdf(leaf_area_bins[:, None], loc=growth_model(leaf_area_bins, *growth_params), scale=5) * \
                           perturbed_survival_probs[:, None] + \
                           perturbed_reproduction_probs[:, None]
        perturbed_kernel /= perturbed_kernel.sum(axis=0)
        perturbed_kernels[f'survival_reproduction_perturb_{perturb}'] = perturbed_kernel

    return perturbed_kernels

def plot_sensitivity_analysis(perturbed_kernels, leaf_area_bins):
    plt.figure(figsize=(15, 10))
    for label, kernel in perturbed_kernels.items():
        eigenvalues, eigenvectors = np.linalg.eig(kernel)
        stable_distribution = np.real(eigenvectors[:, np.argmax(eigenvalues)])
        stable_distribution /= stable_distribution.sum()
        plt.plot(leaf_area_bins, stable_distribution, linestyle='-', label=label)

    plt.xlabel('Leaf Area')
    plt.ylabel('Stable Size Distribution')
    plt.title('Stable Size Distribution Sensitivity Analysis')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    data = load_data('16-17_IPM_data.csv')
    leaf_area_2016, leaf_area_2017, s0_recruits, s1_recruits, gardens = extract_columns(data)

    min_leaf_area = min(leaf_area_2016.min(), leaf_area_2017.min())
    max_leaf_area = max(leaf_area_2016.max(), leaf_area_2017.max())

    n_bins = 100

    plot_reproduction_probability(min_leaf_area, max_leaf_area, reproduction_function_continuous, leaf_area_2016, s0_recruits, s1_recruits, gardens, nbins=n_bins)
    plot_ipm_kernel(min_leaf_area, max_leaf_area, growth_function, leaf_area_2016, leaf_area_2017, survival_function_continuous, reproduction_function_continuous, s0_recruits, s1_recruits, gardens, nbins=n_bins)
    plot_survival_probability(min_leaf_area, max_leaf_area, survival_function_continuous, leaf_area_2016, leaf_area_2017, gardens)
    
    leaf_area_bins = np.linspace(min_leaf_area, max_leaf_area, n_bins)
    kernels = {}
    for garden in gardens.unique():
        garden_mask = gardens == garden
        kernels, _ = project(leaf_area_bins, leaf_area_2016, garden_mask, leaf_area_2017, s0_recruits, s1_recruits, kernels, garden)

    plot_stable_size_distribution(kernels, leaf_area_bins, gardens)

    perturbations = [0.8, 1.2]  # Example perturbations, you can adjust these
    for garden in gardens.unique():
        garden_mask = gardens == garden
        perturbed_kernels = sensitivity_analysis(leaf_area_bins, leaf_area_2016, garden_mask, leaf_area_2017, s0_recruits, s1_recruits, kernels, garden, perturbations)
        plot_sensitivity_analysis(perturbed_kernels, leaf_area_bins)

if __name__ == "__main__":
    main()
