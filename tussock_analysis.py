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

    for timestep in df:
        stepdata = timestep[1]

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

def rootVolume(df):
    pass

def main():
    args = parse_args()

    tussock_diameters = []
    tussock_heights = []

    volumes = dict()
    volumes['total'] = []
    volumes['alive'] = []
    volumes['dead'] = []
    
    for sim in range(int(args.nsims)):
        sim_data = pd.read_csv(f'{args.filepath}/tiller_data_sim_num_{sim}.csv')

        # tussock_diameters.append(tussockDiameter(sim_data, args))

        # tussock_heights.append(tussock_height(sim_data))

        tussock_volume, alive_volume, dead_volume = compute_volumes(sim_data)
        volumes['total'].append(tussock_volume)
        volumes['alive'].append(alive_volume)
        volumes['dead'].append(dead_volume)

    # tussock_diameters = np.array(tussock_diameters)
    # tussock_heights = np.array(tussock_heights)

    # graph_diameters(tussock_diameters)
    # graph_heights(tussock_heights)
    graph_volume(volumes)

        


if __name__ == "__main__":
    main()