import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import subprocess
import os
import configparser
import random
import csv
from PIL import Image
import seaborn as sns

#Run $ wget  "https://docs.google.com/spreadsheets/d/1GfVrWWKMBeOzuNMu31YC-pTHkpYPN9guSKM_pvvei5U/export?format=csv&edit#gid=0" -O "parameterization_data.csv" to get most recent data

def get_config():
    config = configparser.ConfigParser()
    config.read('parameterization.ini')
    return config

def graph_real_data():

    df = pd.read_csv('./parameterization_data.csv')

    scatter_plot = sns.scatterplot(x='Estimated_Age', y='Diameter', hue='Location', data=df, palette='viridis')

    plt.xlabel('Estimated Age')
    plt.ylabel('Diameter')
    plt.title('Scatter Plot with Nonlinear Trend Line (All Locations)')

    plt.legend(title='Location')

    plt.show()

def tussock_model(config):
    makefile_path = 'Makefile'
    make_result = subprocess.run(['make', '-f', makefile_path])

    num_sims = int(config.get('Tussock Model', 'nsims'))
    outdir = config.get('Tussock Model', 'filepath')
    num_threads = int(config.get('Tussock Model','nthreads'))

    sim_time = int(random.uniform(40, 200))

    if make_result.returncode == 0:

        cpp_input = f"{sim_time}\n{num_sims}\n{outdir}\n{num_threads}\n"

        process = subprocess.Popen('./tussock_model', stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        output, _ = process.communicate(input=cpp_input)
        print(output)

    else:
        print("Makefile execution failed")

def clean_data(sim_data, messy_data, config):
    sim_length = int(config.get('Tussock Model', 'nyears'))

    last_time_step = sim_data['TimeStep'].max()
    last_time_step_rows = sim_data[sim_data['TimeStep'] == last_time_step]

    if messy_data != []:
        messy_last = messy_data[-1]
    else: 
        messy_last = 0

    if any(last_time_step_rows['Status'] == 1):
        while len(messy_data) < sim_length + 1:
            messy_data.append(messy_last * 10)
    else:
        while len(messy_data) < sim_length + 1:
            messy_data.append(0)

    return messy_data

def diameter_objective(config, iteration):
    training_data = pd.read_csv('./tussock_allometry.csv')

    num_sims = int(config.get('Tussock Model', 'nsims'))
    sim_filepath = config.get('Tussock Model', 'filepath')

    training_data['field_davg'] = pd.to_numeric(training_data['field_davg'], errors='coerce')
    training_data['field_davg'] = training_data['field_davg'].astype(float)

    training_diameters = training_data['field_davg'].values

    training_diameters = training_diameters[~np.isnan(training_diameters)]

    obv_counts, obv_bins = np.histogram(training_diameters, bins='auto')

    sim_diameters = []

    for i in range(num_sims):
        sim_data = pd.read_csv(f'{sim_filepath}/tiller_data_sim_num_{i}.csv')

        diameter = sim_data[sim_data['TimeStep'] == sim_data['TimeStep'].max()]['X'].max() - sim_data[sim_data['TimeStep'] == sim_data['TimeStep'].max()]['X'].min()

        sim_diameters.append(diameter)

    model_counts, model_bins = np.histogram(sim_diameters, bins='auto')


    common_bins = np.linspace(min(min(obv_bins), min(model_bins)),
                            max(max(obv_bins), max(model_bins)),
                            num=100)

    common_bins = np.union1d(common_bins, obv_bins[:-1])
    common_bins = np.union1d(common_bins, model_bins[:-1])

    obs_dist = np.interp(common_bins, obv_bins[:-1], obv_counts, left=0, right=0)
    model_dist = np.interp(common_bins, model_bins[:-1], model_counts, left=0, right=0)

    squared_diff = (obs_dist - model_dist) ** 2

    penalty = 1.0  # penality for non-overlapping bins, forces the distributions to align
    non_overlapping_obs_bins = set(obv_bins[:-1]) - set(common_bins)
    non_overlapping_model_bins = set(model_bins[:-1]) - set(common_bins)

    for bin_value in non_overlapping_obs_bins:
        squared_diff += penalty * obs_dist[common_bins == bin_value] ** 2

    for bin_value in non_overlapping_model_bins:
        squared_diff += penalty * model_dist[common_bins == bin_value] ** 2

    rmse = np.sqrt(np.mean(squared_diff))

    output_dir = 'mean_diameter_frames'
    os.makedirs(output_dir, exist_ok=True)

    sns.kdeplot(training_diameters, linewidth=1, label='Observed Tussock Diametesr')
    sns.kdeplot(sim_diameters, linewidth=1, label='Predicted Tussock Diameters')
    plt.ylabel('Kernel Density')
    plt.xlabel('Diameter (cm)')
    plt.legend()

    frame_filename = os.path.join(output_dir, f'Mean_Tuss_diameter_iteration_{iteration}.png')
    plt.savefig(frame_filename)
    plt.close()

    del training_data
    return rmse

def calculate_parameters(parameters, config, iteration, previous_loss):
    optimization_data = pd.read_csv('./optimization_results.csv')
    gradients = dict()

    learning_rate = previous_loss/1000

    print('unclipped learning rate: ', learning_rate)
    # learning_rate = np.clip(learning_rate, 0.00001, max_learning_rate)
    print('clipped learning rate: ', learning_rate)
    for key in parameters:
        gradient = np.gradient(optimization_data['loss'], optimization_data[key])
        print('gradient: ', gradient[-1])
        norm_gradient = gradient / np.linalg.norm(gradient, ord=2)
        print('norm_gradient: ', norm_gradient[-1])
        parameter_change = learning_rate * norm_gradient[-1] 
        print('unclipped parameter change: ', parameter_change)
        parameter_change = np.clip(parameter_change, -0.1, 0.1)
        print('clipped parameter change: ', parameter_change)
        parameters[key] = parameters[key] - parameter_change

    return gradients

def write_parameters(parameters, config):
    outdir = config.get('Parameterization','outdir')

    #save parameters to use in sim
    with open(f'{outdir}/parameters.txt', 'w') as file:
        for param_name, param_value in parameters.items():
            file.write(f"{param_name}={param_value}\n")

def write_optimization_results(parameters, loss, iteration, config):
    outdir = config.get('Parameterization','outdir')

    # save parameters and loss for analysis
    csv_filepath = f'{outdir}/optimization_results.csv'
    mode = 'w' if iteration == 0 else 'a'

    with open(csv_filepath, mode, newline='') as csvfile:
        fieldnames = list(parameters.keys()) + ['loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header only in the first iteration
        if iteration == 0:
            writer.writeheader()

        writer.writerow({**parameters, 'loss': loss})

def animate_fitting(outdir, filename, iteration, outfilename):
    frames = []

    for timestep in range(iteration):
        frame_filename = os.path.join(outdir, f'{filename}_{timestep}.png')
        frames.append(Image.open(frame_filename))

    frames[0].save(outfilename, save_all=True, append_images=frames[1:], duration=50, loop=0)

    for frame_filename in os.listdir(outdir):
        os.remove(os.path.join(outdir, frame_filename))
    os.rmdir(outdir)


def main():
    config = get_config()

    parameters = dict()

    # graph_real_data()

    opt_iteration = 0
    dloss=100

    while opt_iteration <= 1:
        print('Iteration: ', opt_iteration)

        #initialize random variables first
        parameters['ks'] = np.random.uniform(0, 0.5)
        parameters['kr'] = random.uniform(0, 0.5)
        parameters['bs'] = random.uniform(0, 0.1)
        parameters['br'] = random.uniform(0, 0.1)

        write_parameters(parameters, config)

        tussock_model(config)
        
        #is this loss
        loss = diameter_objective(config, opt_iteration)

        write_optimization_results(parameters, loss, opt_iteration, config)

        opt_iteration += 1

    print('dloss: ', dloss, ' loss: ', loss)
    while opt_iteration  < 50:
        print('dloss: ', dloss, ' loss: ', loss)
        print('Iteration: ', opt_iteration)

        gradients = calculate_parameters(parameters, config, opt_iteration, loss)
        write_parameters(parameters, config)

        tussock_model(config)

        previous_loss = loss
        loss = diameter_objective(config, opt_iteration)

        dloss = abs(previous_loss - loss)

        write_optimization_results(parameters, loss, opt_iteration, config)

        opt_iteration+=1

    animate_fitting('mean_diameter_frames', f'Mean_Tuss_diameter_iteration', opt_iteration, 'diameter_dist_fitting.gif')
    
    # animate_fitting('num_tillers_frames', f'num_tillers_iteration', opt_iteration, 'num_tiller_fitting.gif')

if __name__ == '__main__':
    main()