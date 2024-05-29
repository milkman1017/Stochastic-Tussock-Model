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
from scipy.interpolate import interp1d
import logging

def get_config():
    config = configparser.ConfigParser()
    config.read('parameterization.ini')
    return config

def graph_real_data():
    df = pd.read_csv('./tussock_density_tussock_diam.csv')
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
    sim_time = int(config.get('Tussock Model', 'nyears'))
    if make_result.returncode == 0:
        cpp_input = f"{sim_time}\n{num_sims}\n{outdir}\n{num_threads}\n"
        process = subprocess.Popen('./tussock_model', stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        output, _ = process.communicate(input=cpp_input)
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
    training_data = pd.read_csv('./tussock_density_tussock_diam.csv')
    num_sims = int(config.get('Tussock Model', 'nsims'))
    sim_filepath = config.get('Tussock Model', 'filepath')
    training_data['field_davg'] = pd.to_numeric(training_data['diam'], errors='coerce')
    training_data['field_davg'] = training_data['field_davg'].astype(float)
    training_diameters = training_data['field_davg'].values
    training_diameters = training_diameters[~np.isnan(training_diameters)]
    obv_hist, obv_bins = np.histogram(training_diameters, bins='auto', density=True)
    sim_diameters = []

    for i in range(num_sims):
        sim_data = pd.read_csv(f'{sim_filepath}/tiller_data_sim_num_{i}.csv')
        last_time_step_data = sim_data[sim_data['TimeStep'] == sim_data['TimeStep'].max()]
        num_alive_tillers = last_time_step_data['Status'].sum()
        num_dead_tillers = (last_time_step_data['Status'] == 0).sum()

        diameter = abs(sim_data[sim_data['TimeStep'] == sim_data['TimeStep'].max()]['X'].max() - sim_data[sim_data['TimeStep'] == sim_data['TimeStep'].max()]['X'].min())
        
        if diameter == 0.0:
            diameter=0.1

        sim_diameters.append(diameter)

    model_hist, model_bins = np.histogram(sim_diameters, bins=obv_bins, density=True)
    rmse = np.sqrt(np.mean((obv_hist - model_hist)**2))
    output_dir = 'mean_diameter_frames'
    os.makedirs(output_dir, exist_ok=True)
    sns.kdeplot(training_diameters, label='Observed Tussock Diameter Distribution', linewidth=1)
    sns.kdeplot(sim_diameters, label='Modeled Tussock Diameter Distribution', linewidth=1)
    plt.legend()
    plt.title(f'Training Iteration: {iteration}')
    plt.xlabel('Tussock Diameter')
    frame_filename = os.path.join(output_dir, f'Mean_Tuss_diameter_iteration_{iteration}.png')
    plt.savefig(frame_filename)
    plt.close()
    del training_data
    return rmse

def calculate_parameters(parameters, config, iteration, previous_loss, dloss, log_file, opt_iteration):
    optimization_data = pd.read_csv('./optimization_results.csv')
    gradients = dict()

    learning_rate = 0.1
    
    with open(log_file, 'a') as log:
        log.write(f'Iteration: {iteration}\n')
        log.write(f'Learning rate: {learning_rate} \n')
        for key in parameters:
            gradient = np.gradient(optimization_data['loss'], optimization_data[key])
            if gradient[-1] == 0.0:
                parameters[key] = parameters[key]
            else:
                norm_gradient = gradient / np.linalg.norm(gradient, ord=2)
                parameter_change = learning_rate * gradient[-1]
                parameter_change = np.clip(parameter_change, -0.5, 0.5)
                parameters[key] = parameters[key] - parameter_change
            log.write(f'Parameter: {key}, Change: {parameter_change}, Gradient: {gradient[-1]}, Normalized Gradient: {norm_gradient[-1]} \n')
    return gradients

def write_parameters(parameters, config):
    outdir = config.get('Parameterization','outdir')
    with open(f'{outdir}/parameters.txt', 'w') as file:
        for param_name, param_value in parameters.items():
            file.write(f"{param_name}={param_value}\n")

def write_optimization_results(parameters, loss, iteration, config):
    outdir = config.get('Parameterization','outdir')
    csv_filepath = f'{outdir}/optimization_results.csv'
    param_history_filepath = f'{outdir}/parameter_history.csv'

    mode = 'w' if iteration == 0 else 'a'

    with open(csv_filepath, mode, newline='') as csvfile:
        fieldnames = list(parameters.keys()) + ['loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if iteration == 0:
            writer.writeheader()
        writer.writerow({**parameters, 'loss': loss})

def animate_fitting(outdir, filename, iteration, outfilename):
    frames = []
    for timestep in range(iteration):
        frame_filename = os.path.join(outdir, f'{filename}_{timestep}.png')
        frames.append(Image.open(frame_filename))
    frames[0].save(outfilename, save_all=True, append_images=frames[1:], duration=75, loop=0)
    for frame_filename in os.listdir(outdir):
        os.remove(os.path.join(outdir, frame_filename))
    os.rmdir(outdir)

def main():
    config = get_config()
    parameters = dict()
    log_file = 'parameter_updates.log'
    with open(log_file, 'w') as log:
        log.write('Parameter Updates Log\n')

    opt_iteration = 0
    dloss = 100
    while opt_iteration <= 1:
        parameters['ks'] = np.random.uniform(0.3, 0.6)
        parameters['kr'] = random.uniform(0.3, 0.6)
        
        for i in range(8):
            for j in range(9):
                parameters[f'class_transition_matrix_{i}_{j}'] = np.random.uniform(0.0001,1.0)

        for i in range(9):
            parameters[f'survival_matrix_{i}'] = np.random.uniform(0.0001,1.0)
            parameters[f'tillering_matrix_{i}'] = np.random.uniform(0.0001,1.0)

        write_parameters(parameters, config)
        tussock_model(config)
        loss = diameter_objective(config, opt_iteration)
        with open(log_file, 'a') as log:
            log.write(f'Iteration: {opt_iteration}, Loss: {loss}\n')
        write_optimization_results(parameters, loss, opt_iteration, config)
        opt_iteration += 1

    while opt_iteration < 100 and loss > 0.015 and dloss > 0.0001:
        gradients = calculate_parameters(parameters, config, opt_iteration, loss, dloss, log_file, opt_iteration)
        write_parameters(parameters, config)
        tussock_model(config)
        previous_loss = loss
        loss = diameter_objective(config, opt_iteration)
        dloss = abs(previous_loss - loss)
        with open(log_file, 'a') as log:
            print(loss, dloss)
            log.write(f'Iteration: {opt_iteration}, Loss: {loss}, dLoss: {dloss}\n')
        write_optimization_results(parameters, loss, opt_iteration, config)
        opt_iteration += 1

    animate_fitting('mean_diameter_frames', f'Mean_Tuss_diameter_iteration', opt_iteration, 'diameter_dist_fitting.gif')

if __name__ == '__main__':
    main()
