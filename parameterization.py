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
    sim_time = int(config.get('Tussock Model', 'nyears'))
    num_threads = int(config.get('Tussock Model','nthreads'))

    if make_result.returncode == 0:
        
        num_threads = 10

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

    if any(last_time_step_rows['Status'] == 1):
        observed_diameters = sim_data[sim_data['TimeStep'] <= last_time_step]['X'].tolist()
        messy_data.extend(observed_diameters)
    else:
        while len(messy_data) < sim_length + 1:
            messy_data.append(0)

    return messy_data

def mean_diameter_objective(config, iteration):
    training_data = pd.read_csv('./parameterization_data.csv')

    num_sims = int(config.get('Tussock Model', 'nsims'))
    sim_filepath = config.get('Tussock Model', 'filepath')
    sim_length = int(config.get('Tussock Model', 'nyears'))

    mean_tussock_size_objective = 0

    mean_training_diameters = training_data.groupby('Estimated_Age')['Diameter'].mean().values

    sim_diameters = []

    for i in range(num_sims):
        sim_data = pd.read_csv(f'{sim_filepath}/tiller_data_sim_num_{i}.csv')

        sim_diameter = (sim_data.groupby('TimeStep')['X'].apply(lambda x: x.max() - x.min())).values.tolist()

        if len(sim_diameter) != (sim_length + 1): #check if data needs to be cleaned
            sim_diameter = clean_data(sim_data, sim_diameter, config)

        sim_diameters.append(sim_diameter)

    sim_diameters = np.array(sim_diameters)
    mean_sim_diameters = np.mean(sim_diameters, axis=0)

    plt.plot(mean_training_diameters, linewidth=1, label='Real Mean Tussock Diameter')
    plt.plot(mean_sim_diameters, linewidth=1, label='Predicted Mean Tussock Diameters')
    plt.ylabel('Tussock Diameter (cm)')
    plt.xlabel('Time (years)')
    plt.legend()
    plt.savefig(f'Mean_Tuss_diameter_iteration_{iteration}.png')
    plt.close()
    
    for mean_diameter in zip(mean_sim_diameters, mean_training_diameters):
        mean_tussock_size_objective += np.sqrt((mean_diameter[0] - mean_diameter[1])**2)

    del training_data

    return mean_tussock_size_objective

def variation_objective(config, iteration):
    training_data = pd.read_csv('./parameterization_data.csv')

    num_sims = int(config.get('Tussock Model', 'nsims'))
    sim_filepath = config.get('Tussock Model', 'filepath')
    sim_length = int(config.get('Tussock Model', 'nyears'))

    var_tussock_size_objective = 0

    var_training_diameters = training_data.groupby('Estimated_Age')['Diameter'].var().values

    sim_diameters = []

    for i in range(num_sims):
        sim_data = pd.read_csv(f'{sim_filepath}/tiller_data_sim_num_{i}.csv')

        sim_diameter = (sim_data.groupby('TimeStep')['X'].apply(lambda x: x.max() - x.min())).values.tolist()

        if len(sim_diameter) != (sim_length + 1): #check if data needs to be cleaned
            sim_diameter = clean_data(sim_data, sim_diameter, config)

        sim_diameters.append(sim_diameter)

    sim_diameters = np.array(sim_diameters)
    var_sim_diameters = np.var(sim_diameters, axis=0)

    plt.plot(var_training_diameters, linewidth=1, label='Real variation Tussock Diameter')
    plt.plot(var_sim_diameters, linewidth=1, label='Predicted variation Tussock Diameters')
    plt.ylabel('Tussock Diameter (cm)')
    plt.xlabel('Time (years)')
    plt.legend()
    plt.savefig(f'variation_Tuss_diameter_iteration_{iteration}.png')
    plt.close()
    
    for var_diameter in zip(var_sim_diameters, var_training_diameters):
        var_tussock_size_objective += np.sqrt((var_diameter[0] - var_diameter[1])**2)

    del training_data

    return var_tussock_size_objective 

def calculate_parameters(parameters, config):
    optimization_data = pd.read_csv('./optimization_results.csv')

    previous_loss = optimization_data['loss'].iloc[-1]

    min_learning_rate = 1e-6 
    learning_rate = max(previous_loss / 10000, min_learning_rate)

    grad_ks = np.gradient(optimization_data['loss'], optimization_data['ks'])
    grad_kr = np.gradient(optimization_data['loss'], optimization_data['kr'])

    parameters['ks'] = parameters['ks'] - learning_rate * np.sign(grad_ks[-1])
    parameters['kr'] = parameters['kr'] - learning_rate * np.sign(grad_kr[-1])

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

def main():
    config = get_config()

    parameters = dict()

    # graph_real_data()

    opt_iteration = 0
    dloss=100

    while opt_iteration <= 1:
        print('Iteration: ', opt_iteration)

        #initialize random variables first
        parameters['ks'] = random.random()
        parameters['kr'] = random.random()

        write_parameters(parameters, config)

        tussock_model(config)

        #is this loss
        loss = mean_diameter_objective(config, opt_iteration) 

        write_optimization_results(parameters, loss, opt_iteration, config)

        opt_iteration += 1

    while dloss >= 5:
        print('Iteration: ', opt_iteration)

        calculate_parameters(parameters, config)
        write_parameters(parameters, config)

        tussock_model(config)

        previous_loss = loss
        loss = mean_diameter_objective(config, opt_iteration) 

        dloss = abs(previous_loss - loss)

        write_optimization_results(parameters, loss, opt_iteration, config)

        opt_iteration+=1
    
    frames = []
    for timestep in range(opt_iteration):
        frame_filename = f'Mean_Tuss_diameter_iteration_{timestep}.png'
        frames.append(Image.open(frame_filename))

    output_gif_filename = 'mean_tus_fitting.gif'
    frames[0].save(output_gif_filename, save_all=True, append_images=frames[1:], duration=50, loop=0)

    frames = []
    for timestep in range(opt_iteration):
        frame_filename = f'variation_Tuss_diameter_iteration_{timestep}.png'
        frames.append(Image.open(frame_filename))

    output_gif_filename = 'variation_tus_fitting.gif'
    frames[0].save(output_gif_filename, save_all=True, append_images=frames[1:], duration=50, loop=0)


if __name__ == '__main__':
    main()