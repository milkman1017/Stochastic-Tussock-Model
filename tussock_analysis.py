import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.spatial import ConvexHull, convex_hull_plot_2d

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--nsims', type=int, help='The number of simulations that you wish to simulate')
    parser.add_argument('--filepath', type=str, help='The directory containing the simulations to analyze')
    parser.add_argument('--outdir', type=str, help='Output directory to save the analysis')

    args=parser.parse_args()
    
    return args

def numberOfTillers(df):
    df = df.groupby('TimeStep')

    total_tillers = []
    alive_tillers = []
    dead_tillers = []

    for timestep in df:
        stepdata = timestep[1]

        total_tiller = len(stepdata)
        alive_tiller = len(stepdata[stepdata['Status']==1])
        dead_tiller = len(stepdata[stepdata['Status']==0])

        total_tillers.append(total_tiller)
        alive_tillers.append(alive_tiller)
        dead_tillers.append(dead_tiller)
        
    return total_tillers, alive_tillers, dead_tillers

def graph_tiller_number(tiller_number):
    total_tillers = np.array(tiller_number['total'])
    alive_tillers = np.array(tiller_number['alive'])
    dead_tillers = np.array(tiller_number['dead'])

    for data in alive_tillers:
        plt.plot(data, linewidth=1)
    
    plt.show()

def tussockDiameter(df):
    df = df.groupby("TimeStep")

    diameters = []

    x_diameters = (df['X'].max() - df['X'].min())
    y_diameters = (df['Y'].max() - df['Y'].min())

    for x_diameter, y_diameter in zip(x_diameters, y_diameters):
        diameter = np.mean([x_diameter, y_diameter])
        diameters.append(diameter)

    return diameters

def graph_diameters(diameter_data):

    diameter_growth = np.diff(diameter_data)

    fig, ax = plt.subplots(2)

    ax[0].plot(diameter_data.mean(axis=0), linewidth=1, label='Mean Tussock Diameter')
    ax[0].fill_between(range(len(diameter_data.min(axis=0))), diameter_data.min(axis=0), diameter_data.max(axis=0), alpha=0.2, label='Range of Tussock Diameter')
    ax[0].set_title('Tussock Diameter')
    ax[0].set_ylabel('Diameter (cm)')
    ax[0].set_xlabel('Time (yrs)')

    ax[1].plot(diameter_growth.mean(axis=0), linewidth=1, label='Mean of Growth')
    ax[1].fill_between(range(len(diameter_growth.min(axis=0))), diameter_growth.min(axis=0), diameter_growth.max(axis=0), alpha=0.2, label='Range of Growth')
    ax[1].set_title('Tussock Growth')
    ax[1].set_ylabel('Growth (cm/yr)')
    ax[1].set_xlabel('Time (yrs)')

    plt.tight_layout()
    plt.legend()
    plt.show()

def tussock_height(df):
    df = df.groupby("TimeStep")

    mean_height = df['Z'].mean()
    return mean_height

def graph_heights(height_data):
    fig, ax = plt.subplots()

    ax.plot(height_data.mean(axis=0), linewidth=1, label='Mean Height of Tussock from all Simulations')
    ax.fill_between(range(len(height_data.min(axis=0))), height_data.min(axis=0), height_data.max(axis=0), alpha=0.2, label='Range of Mean Tussock Heights')

    plt.show()

def compute_volumes(df):
    df = df.groupby("TimeStep")

    tussock_volumes = []
    alive_volumes = []
    dead_volumes = []

    for timestep in df:
        step_data = timestep[1]

        all_points = step_data[['X','Y','Z']]
        alive_points = step_data[step_data['Status']==1][['X','Y','Z']]
        dead_points = step_data[step_data['Status']==0][['X','Y','Z']]

        try:
            tussock_volume = ConvexHull(all_points).volume
            tussock_volumes.append(tussock_volume)
        except Exception as e:
            tussock_volumes.append(0)

        try: 
            alive_volume = ConvexHull(alive_points).volume
            alive_volumes.append(alive_volume)
        except Exception as e:
            alive_volumes.append(0)

        try:
            dead_volume = ConvexHull(dead_points).volume
            dead_volumes.append(dead_volume)
        except Exception as e:
            dead_volumes.append(0)
    
    return tussock_volumes, alive_volumes, dead_volumes

def graph_volume(volumes):
    total_volumes = np.array(volumes['total'])
    alive_volumes = np.array(volumes['alive'])
    dead_volumes = np.array(volumes['dead'])

    fig, ax = plt.subplots(3)

    ax[0].plot(total_volumes.mean(axis=0), linewidth=1, color='b', label='Mean Above Ground Volume')
    ax[0].fill_between(range(len(total_volumes.min(axis=0))), total_volumes.min(axis=0), total_volumes.max(axis=0), alpha=0.2, color='b', label='Range of Above Ground Volume')
    ax[0].set_title('Total Tussock Volume')
    ax[0].set_ylabel('Volume (cm^3)')
    ax[0].set_xlabel('Time (yrs)')

    ax[1].plot(alive_volumes.mean(axis=0), linewidth=1, color='g', label='Mean Above Ground Volume')
    ax[1].fill_between(range(len(alive_volumes.min(axis=0))), alive_volumes.min(axis=0), alive_volumes.max(axis=0), alpha=0.2, color='g', label='Range of Above Ground Volume')
    ax[1].set_title('Living Tussock Volume')
    ax[1].set_ylabel('Volume (cm^3)')
    ax[1].set_xlabel('Time (yrs)')

    ax[2].plot(dead_volumes.mean(axis=0), linewidth=1, color='brown', label='Mean Above Ground Volume')
    ax[2].fill_between(range(len(dead_volumes.min(axis=0))), dead_volumes.min(axis=0), dead_volumes.max(axis=0), alpha=0.2, color='brown', label='Range of Above Ground Volume')
    ax[2].set_title('Dead Tussock Volume')
    ax[2].set_ylabel('Volume (cm^3)')
    ax[2].set_xlabel('Time (yrs)')

    plt.tight_layout()
    plt.show()

def calculate_root_volume(df):
    df = df.groupby("TimeStep")

    live_root_volumes = []
    dead_root_volumes = []


    for timestep in df:
        stepdata = timestep[1]

        try:
            dead_root_volume = np.round(dead_root_volumes[-1] + live_root_volumes[-1]/2,2) #assume dead roots are half the diameter as the live ones, but same length, also assume roots dont decay
            dead_root_volumes.append(dead_root_volume)
        except Exception as e:
            dead_root_volumes.append(0)

        live_root_volume = np.round(stepdata['NumRoots'].sum() * np.pi * 0.3**2 * 100, 2)  #volume of roots is assuming they are a cylinder extending down to permafrost (approx 100 cm) and of thickness 6 mm (3 mm diameter)
        live_root_volumes.append(live_root_volume)

    return live_root_volumes, dead_root_volumes

def graph_root_volumes(root_volumes):
    alive_root_volumes = np.array(root_volumes['alive'])
    dead_root_volumes = np.array(root_volumes['dead'])

    fig, ax = plt.subplots(2)

    for data in alive_root_volumes:
        ax[0].plot(data)

    for data in dead_root_volumes:
        ax[1].plot(data)

    plt.show()

def main():
    args = parse_args()

    tussock_diameters = []
    tussock_heights = []

    volumes = dict()
    volumes['total'] = []
    volumes['alive'] = []
    volumes['dead'] = []

    tiller_number = dict()
    tiller_number['total'] = []
    tiller_number['alive'] = []
    tiller_number['dead'] = []

    root_volumes = dict()
    root_volumes['alive'] = []
    root_volumes['dead'] = []
    
    for sim in range(int(args.nsims)):
        sim_data = pd.read_csv(f'{args.filepath}/tiller_data_sim_num_{sim}.csv')

        # tussock_diameters.append(tussockDiameter(sim_data))

        # tussock_heights.append(tussock_height(sim_data))

        # tussock_volume, alive_volume, dead_volume = compute_volumes(sim_data)
        # volumes['total'].append(tussock_volume)
        # volumes['alive'].append(alive_volume)
        # volumes['dead'].append(dead_volume)

        # total_tillers, alive_tillers, dead_tillers = numberOfTillers(sim_data)
        # tiller_number['total'].append(total_tillers)
        # tiller_number['alive'].append(alive_tillers)
        # tiller_number['dead'].append(dead_tillers)\

        alive_root_volumes, dead_root_volumes = calculate_root_volume(sim_data)
        root_volumes['alive'].append(alive_root_volumes)
        root_volumes['dead'].append(dead_root_volumes)

    # tussock_diameters = np.array(tussock_diameters)
    # tussock_heights = np.array(tussock_heights)

    # graph_diameters(tussock_diameters)
    # graph_heights(tussock_heights)
    # graph_volume(volumes)
    # graph_tiller_number(tiller_number)
    graph_root_volumes(root_volumes)

if __name__ == "__main__":
    main()