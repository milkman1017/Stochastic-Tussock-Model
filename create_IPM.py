import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Growth function
def growth_function(x, a, b, c):
    return a * x**b + c

# Fit the growth model
def fit_growth_model(data):
    x_data = data['2016 Leaf Area']
    y_data = data['2017 Leaf Area']
    popt, _ = curve_fit(growth_function, x_data, y_data, maxfev=10000)
    return popt

# Survival function
def survival_function(data):
    total_tillers = len(data)
    dead_tillers = len(data[data['2017 Type (1-3) 1=dead'] == 1])
    survival_rate = 1 - (dead_tillers / total_tillers)
    return survival_rate

# Reproduction function
def reproduction_function(data):
    new_tillers = data['2017 # S0 (current yr new)'].sum()
    total_tillers = len(data)
    reproduction_rate = new_tillers / total_tillers
    return reproduction_rate

# Construct the IPM kernel
def construct_kernel(popt, size_range, survival_rate, reproduction_rate):
    a, b, c = popt
    kernel = np.zeros((len(size_range), len(size_range)))
    
    for i, x in enumerate(size_range):
        for j, y in enumerate(size_range):
            growth = growth_function(x, a, b, c)
            
            # Kernel entry: transition from size x to y
            kernel[i, j] = survival_rate * (y - growth)**2 * reproduction_rate
    
    return kernel

# Analyze the IPM
def analyze_ipm(kernel, size_range, category):
    eigenvalues, eigenvectors = np.linalg.eig(kernel.T)
    dominant_index = np.argmax(eigenvalues)
    stable_distribution = eigenvectors[:, dominant_index].real
    stable_distribution /= stable_distribution.sum()
    
    plt.plot(size_range, stable_distribution, label=category)
    plt.xlabel('Leaf Area')
    plt.ylabel('Stable Distribution')
    plt.title('Stable Size Distribution')
    plt.legend()
    plt.savefig(f'stable_distribution_{category}.png')
    plt.clf()

# Plot the kernel
def plot_kernel(kernel, size_range, category):
    plt.imshow(kernel, extent=(size_range[0], size_range[-1], size_range[0], size_range[-1]), origin='lower', aspect='auto', cmap='hot')
    plt.colorbar(label='Kernel Value')
    plt.xlabel('Size at time t')
    plt.ylabel('Size at time t+1')
    plt.title(f'Kernel Distribution for {category}')
    plt.savefig(f'kernel_distribution_{category}.png')
    plt.clf()

# Simulate population dynamics
def simulate_population(kernel, initial_distribution, num_time_steps):
    population_size = np.zeros(num_time_steps)
    average_leaf_area = np.zeros(num_time_steps)
    size_distribution = np.zeros((num_time_steps, len(initial_distribution)))
    
    size_distribution[0, :] = initial_distribution
    population_size[0] = initial_distribution.sum()
    average_leaf_area[0] = (initial_distribution * size_range).sum() / initial_distribution.sum()
    
    for t in range(1, num_time_steps):
        size_distribution[t, :] = kernel.dot(size_distribution[t-1, :])
        population_size[t] = size_distribution[t, :].sum()
        average_leaf_area[t] = (size_distribution[t, :] * size_range).sum() / size_distribution[t, :].sum()
    
    return population_size, average_leaf_area, size_distribution

def main():
    data = pd.read_csv('16_17_IPM_data.csv')
    print(data)
    
    size_range = np.linspace(data['2016 Leaf Area'].min(), data['2016 Leaf Area'].max(), 100)
    num_time_steps = 50
    
    population_sizes = {}
    average_leaf_areas = {}
    size_distributions = {}
    
    for category in data['Source'].unique():
        category_data = data[data['Source'] == category]
        
        if len(category_data) < 2:
            print(f"Not enough data for category {category} to fit a model.")
            continue
        
        popt = fit_growth_model(category_data)
        print(f"Fitted parameters for {category}: {popt}")
        
        survival_rate = survival_function(category_data)
        reproduction_rate = reproduction_function(category_data)
        
        kernel = construct_kernel(popt, size_range, survival_rate, reproduction_rate)
        
        analyze_ipm(kernel, size_range, category)
        plot_kernel(kernel, size_range, category)
        
        initial_distribution = np.ones(len(size_range))
        population_size, average_leaf_area, size_distribution = simulate_population(kernel, initial_distribution, num_time_steps)
        
        population_sizes[category] = population_size
        average_leaf_areas[category] = average_leaf_area
        size_distributions[category] = size_distribution
    
    # Plot population size vs time
    plt.figure()
    for category, population_size in population_sizes.items():
        plt.plot(range(num_time_steps), population_size, label=category)
    plt.xlabel('Time')
    plt.ylabel('Population Size')
    plt.title('Population Size vs Time')
    plt.legend()
    plt.savefig('population_size_vs_time.png')
    plt.clf()
    
    # Plot average leaf area vs time
    plt.figure()
    for category, average_leaf_area in average_leaf_areas.items():
        plt.plot(range(num_time_steps), average_leaf_area, label=category)
    plt.xlabel('Time')
    plt.ylabel('Average Leaf Area')
    plt.title('Average Leaf Area vs Time')
    plt.legend()
    plt.savefig('average_leaf_area_vs_time.png')
    plt.clf()
    
    # Plot size distributions at final time step
    plt.figure()
    for category, size_distribution in size_distributions.items():
        plt.plot(size_range, size_distribution[-1, :], label=category)
    plt.xlabel('Leaf Area')
    plt.ylabel('Frequency')
    plt.title('Size Distribution at Final Time Step')
    plt.legend()
    plt.savefig('size_distribution_final.png')
    plt.clf()

if __name__ == "__main__":
    main()
