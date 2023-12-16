import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import subprocess
import os

#Run $ wget  "https://docs.google.com/spreadsheets/d/1GfVrWWKMBeOzuNMu31YC-pTHkpYPN9guSKM_pvvei5U/export?format=csv&edit#gid=0" -O "parameterization_data.csv" to get most recent data

def graph_input_data(df):

    scatter_plot = sns.scatterplot(x='Estimated_Age', y='Diameter', hue='Location', data=df, palette='viridis')

    plt.xlabel('Estimated Age')
    plt.ylabel('Diameter')
    plt.title('Scatter Plot with Nonlinear Trend Line (All Locations)')

    plt.legend(title='Location')

    plt.show()

def tussock_model():
    makefile_path = 'Makefile'
    make_result = subprocess.run(['make', '-f', makefile_path])

    if make_result.returncode == 0:
        
        sim_time = 250
        num_sims = 10
        outdir = 'lol'
        num_threads = 10

        cpp_input = f"{sim_time}\n{num_sims}\n{outdir}\n{num_threads}\n"

        process = subprocess.Popen('./tussock_model', stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        output, _ = process.communicate(input=cpp_input)
        print(output)

    else:
        print("Makefile execution failed")

def mean_tussock_size_analysis(training_data):
    filepath =  'lol'
    num_sims=10

    mean_tussock_size_objective = 0

    mean_training_diameters = training_data.groupby('Estimated_Age')['Diameter'].mean().values

    sim_diameters = []

    for i in range(num_sims):
        sim_data = pd.read_csv(f'{filepath}/tiller_data_sim_num_{i}.csv')

        sim_diameters.append(sim_data.groupby('TimeStep')['X'].apply(lambda x: x.max() - x.min()))

    sim_diameters = np.array(sim_diameters)
    mean_sim_diameters = np.mean(sim_diameters, axis=0)
    
    for mean_diameter in zip(mean_sim_diameters, mean_training_diameters):
        mean_tussock_size_objective += (mean_diameter[0] - mean_diameter[1])**2

    return mean_tussock_size_objective

def tussock_variation_analysis(training_data):
    filepath =  'lol'
    num_sims=10

    var_tussock_size_objective = 0

    var_training_diameters = training_data.groupby('Estimated_Age')['Diameter'].var().values

    sim_diameters = []

    for i in range(num_sims):
        sim_data = pd.read_csv(f'{filepath}/tiller_data_sim_num_{i}.csv')

        sim_diameters.append(sim_data.groupby('TimeStep')['X'].apply(lambda x: x.max() - x.min()))

    sim_diameters = np.array(sim_diameters)
    var_sim_diameters = np.var(sim_diameters, axis=0)
    
    for var_diameter in zip(var_sim_diameters, var_training_diameters):
        var_tussock_size_objective += (var_diameter[0] - var_diameter[1])**2

    return var_tussock_size_objective

def compute_parameters():
    pass

def main():
    training_data = pd.read_csv('parameterization_data.csv')

    parameters = {
        'ks': 0.25,
        'kr': 0.05,
    }

    output_file_path = 'parameters.txt'

    # Write parameters to the text file
    with open(output_file_path, 'w') as file:
        for param_name, param_value in parameters.items():
            file.write(f"{param_name}={param_value}\n")

    # tussock_model()
    mean_tussock_size_analysis(training_data)
    tussock_variation_analysis(training_data)

if __name__ == '__main__':
    main()