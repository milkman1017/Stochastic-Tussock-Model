import argparse
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
import logging


# ------------------------------------------------------
# Argument Parsing
# ------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Tussock model parameterization"
    )
    parser.add_argument(
        "--sites",
        nargs="*",
        default=None,
        help=(
            "List of sites to parameterize separately. "
            "Example: --sites SiteA SiteB. "
            "Use '--sites all' to run each site independently."
        )
    )
    return parser.parse_args()



# ------------------------------------------------------
# Config Loading
# ------------------------------------------------------
def get_config():
    config = configparser.ConfigParser()
    config.read('parameterization.ini')
    return config


# ------------------------------------------------------
# Plotting Real Data (saved instead of shown)
# ------------------------------------------------------
def graph_real_data(training_df, site_outdir):
    os.makedirs(os.path.join(site_outdir, "real_data_plots"), exist_ok=True)

    scatter_plot = sns.scatterplot(
        x="Estimated_Age",
        y="Diameter",
        hue="Location",
        data=training_df,
        palette="viridis"
    )

    plt.xlabel('Estimated Age')
    plt.ylabel('Diameter')
    plt.title("Diameter vs Age (filtered site data)")
    plt.legend(title='Location')

    plt.savefig(
        os.path.join(site_outdir, "real_data_plots", "scatter_diameter_vs_age.png"),
        dpi=300
    )
    plt.close()



# ------------------------------------------------------
# Run the C++ Tussock Model
# ------------------------------------------------------
def tussock_model(config):
    makefile_path = "Makefile"
    make_result = subprocess.run(["make", "-f", makefile_path])

    num_sims = int(config.get('Tussock Model', 'nsims'))
    outdir = config.get('Tussock Model', 'filepath')
    num_threads = int(config.get('Tussock Model','nthreads'))
    sim_time = int(config.get('Tussock Model', 'nyears'))

    if make_result.returncode == 0:
        cpp_input = f"{sim_time}\n{num_sims}\n{outdir}\n{num_threads}\n"
        process = subprocess.Popen(
            './tussock_model',
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True
        )
        output, _ = process.communicate(input=cpp_input)
    else:
        print("Makefile execution failed")



# ------------------------------------------------------
# Objective Function: compare model diameters to field data
# ------------------------------------------------------
def diameter_objective(config, iteration, training_data):
    num_sims = int(config.get('Tussock Model', 'nsims'))
    sim_filepath = config.get('Tussock Model', 'filepath')

    training_data['field_davg'] = pd.to_numeric(training_data['diam'], errors='coerce')
    training_diameters = training_data['field_davg'].dropna().values

    obv_hist, obv_bins = np.histogram(training_diameters, bins='auto', density=True)

    sim_diameters = []

    for i in range(num_sims):
        sim_data = pd.read_csv(f"{sim_filepath}/tiller_data_sim_num_{i}.csv")
        last_time_step = sim_data['TimeStep'].max()

        diameter = abs(
            sim_data[sim_data['TimeStep'] == last_time_step]['X'].max() -
            sim_data[sim_data['TimeStep'] == last_time_step]['X'].min()
        )
        if diameter == 0.0:
            diameter = 0.1

        sim_diameters.append(diameter)

    model_hist, model_bins = np.histogram(sim_diameters, bins=obv_bins, density=True)
    rmse = np.sqrt(np.mean((obv_hist - model_hist)**2))

    # Plot KDE frames
    output_dir = "mean_diameter_frames"
    os.makedirs(output_dir, exist_ok=True)

    sns.kdeplot(training_diameters, label='Observed', linewidth=1)
    sns.kdeplot(sim_diameters, label='Modeled', linewidth=1)
    plt.legend()
    plt.title(f'Training Iteration: {iteration}')
    plt.xlabel('Tussock Diameter')

    frame_filename = os.path.join(output_dir, f'Mean_Tuss_diameter_iteration_{iteration}.png')
    plt.savefig(frame_filename, dpi=300)
    plt.close()

    return rmse



# ------------------------------------------------------
# Optimization Helper Functions
# ------------------------------------------------------
def calculate_parameters(parameters, config, iteration, previous_loss, dloss, log_file, opt_iteration):
    optimization_data = pd.read_csv('./optimization_results.csv')
    gradients = dict()

    learning_rate = 0.1

    with open(log_file, 'a') as log:
        log.write(f"Iteration {iteration}\n")
        log.write(f"Learning rate: {learning_rate}\n")

        for key in parameters:
            gradient = np.gradient(optimization_data['loss'], optimization_data[key])
            if gradient[-1] != 0.0:
                norm_gradient = gradient / np.linalg.norm(gradient, ord=2)
                parameter_change = learning_rate * gradient[-1]
                parameter_change = np.clip(parameter_change, -0.5, 0.5)
                parameters[key] -= parameter_change
            else:
                parameter_change = 0
                norm_gradient = gradient

            log.write(
                f"Parameter: {key}, Change: {parameter_change}, "
                f"Gradient: {gradient[-1]}, Normalized: {norm_gradient[-1]}\n"
            )

    return gradients



def write_parameters(parameters, config):
    outdir = config.get('Parameterization', 'outdir')
    with open(f"{outdir}/parameters.txt", "w") as file:
        for param_name, param_value in parameters.items():
            file.write(f"{param_name}={param_value}\n")



def write_optimization_results(parameters, loss, iteration, config):
    outdir = config.get('Parameterization','outdir')
    csv_filepath = f"{outdir}/optimization_results.csv"

    mode = "w" if iteration == 0 else "a"

    with open(csv_filepath, mode, newline='') as csvfile:
        fieldnames = list(parameters.keys()) + ['loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if iteration == 0:
            writer.writeheader()

        writer.writerow({**parameters, 'loss': loss})



def animate_fitting(outdir, filename, iteration, outfilename):
    frames = []
    for timestep in range(iteration):
        frame_filename = os.path.join(outdir, f"{filename}_{timestep}.png")
        frames.append(Image.open(frame_filename))

    frames[0].save(
        outfilename,
        save_all=True,
        append_images=frames[1:],
        duration=75,
        loop=0
    )

    for frame_filename in os.listdir(outdir):
        os.remove(os.path.join(outdir, frame_filename))

    os.rmdir(outdir)



# ------------------------------------------------------
# MAIN: Loop over sites and parameterize each
# ------------------------------------------------------
def main():
    args = parse_args()
    config = get_config()

    # Load dataset once
    full_training_df = pd.read_csv('./tussock_density_tussock_diam.csv')

    # Determine which sites to run
    if args.sites is None:
        site_list = ["ALL"]
    elif len(args.sites) == 1 and args.sites[0].lower() == "all":
        site_list = sorted(full_training_df["site"].unique())
    else:
        site_list = args.sites

    # Loop over requested sites
    for site in site_list:
        print(f"\n====================================")
        print(f"   PARAMETERIZING SITE: {site}")
        print(f"====================================\n")

        # Filter field data
        if site == "ALL":
            training_data = full_training_df.copy()
            site_tag = "ALL"
        else:
            training_data = full_training_df[full_training_df["site"] == site].copy()
            site_tag = site

        # Create site-specific output directory
        base_outdir = config.get('Parameterization', 'outdir')
        site_outdir = os.path.join(base_outdir, site_tag)
        os.makedirs(site_outdir, exist_ok=True)

        # Update model output path
        cpp_outdir = os.path.join(site_outdir, "simulation_outputs")
        os.makedirs(cpp_outdir, exist_ok=True)
        config.set('Tussock Model', 'filepath', cpp_outdir)

        # Log file for this site
        log_file = os.path.join(site_outdir, "parameter_updates.log")
        open(log_file, 'w').write("Parameter Updates Log\n")

        # Initialize
        parameters = {}
        opt_iteration = 0
        dloss = 100

        # ----------------------------------------
        # Initial Random Parameter Search (2 runs)
        # ----------------------------------------
        while opt_iteration <= 1:
            parameters['ks'] = np.random.uniform(0.3, 0.6)
            parameters['kr'] = random.uniform(0.3, 0.6)

            for i in range(8):
                for j in range(9):
                    parameters[f'class_transition_matrix_{i}_{j}'] = np.random.uniform(0.0001,1.0)

            for i in range(9):
                parameters[f'survival_matrix_{i}'] = np.random.uniform(0.0001,1.0)
                parameters[f'tillering_matrix_{i}'] = np.random.uniform(0.0001,1.0)

            # Write parameters.txt inside the site directory
            config.set('Parameterization', 'outdir', site_outdir)
            write_parameters(parameters, config)

            # Run C++ model
            tussock_model(config)

            # Compute loss for this site
            loss = diameter_objective(config, opt_iteration, training_data)

            with open(log_file, 'a') as log:
                log.write(f"Iteration {opt_iteration}, Loss {loss}\n")

            write_optimization_results(parameters, loss, opt_iteration, config)
            opt_iteration += 1

        # -------------------------
        # Gradient descent loop
        # -------------------------
        while opt_iteration < 100 and loss > 0.015 and dloss > 0.0001:

            gradients = calculate_parameters(
                parameters, config, opt_iteration,
                loss, dloss, log_file, opt_iteration
            )

            write_parameters(parameters, config)
            tussock_model(config)

            previous_loss = loss
            loss = diameter_objective(config, opt_iteration, training_data)
            dloss = abs(previous_loss - loss)

            with open(log_file, 'a') as log:
                log.write(f"Iteration: {opt_iteration}, Loss: {loss}, dLoss: {dloss}\n")

            write_optimization_results(parameters, loss, opt_iteration, config)
            opt_iteration += 1

        # ----------------------------------------
        # Save final sims for this site
        # ----------------------------------------
        final_sims_dir = os.path.join(site_outdir, "final_sims")
        os.makedirs(final_sims_dir, exist_ok=True)

        for fname in os.listdir(cpp_outdir):
            if fname.endswith(".csv"):
                src = os.path.join(cpp_outdir, fname)
                dst = os.path.join(final_sims_dir, fname)
                os.replace(src, dst)

        # ----------------------------------------
        # Make the GIF for this site
        # ----------------------------------------
        # animate_fitting(
        #     "mean_diameter_frames",
        #     "Mean_Tuss_diameter_iteration",
        #     opt_iteration,
        #     os.path.join(site_outdir, "diameter_dist_fitting.gif")
        # )

        print(f"Completed site: {site_tag}")


if __name__ == "__main__":
    main()
