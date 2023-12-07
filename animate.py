import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numpy as np
import os
from PIL import Image
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--sim_id', type=int, help='Sim id that you wish to animate')
    parser.add_argument('--filepath', type=str, help='The directory containing the simulations to analyze')
    parser.add_argument('--outdir', type=str, help='Output directory to save the analysis')

    args=parser.parse_args()
    
    return args

def main():

    args = parse_args()

    data = pd.read_csv(f'{args.filepath}/tiller_data_sim_num_{args.sim_id}.csv')
    
    point_scatter_3d(data)
    compute_volume(data)

def point_scatter_3d(data):
    output_dir = 'scatter_plot_frames'
    os.makedirs(output_dir, exist_ok=True)

    for timestep in data['TimeStep'].unique():
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        timestep_data = data[data['TimeStep'] == timestep]

        ax.scatter(
            timestep_data[timestep_data['Status'] == 1]['X'],
            timestep_data[timestep_data['Status'] == 1]['Y'],
            timestep_data[timestep_data['Status'] == 1]['Z'],
            color='green',
            label='Alive'
        )

        ax.scatter(
            timestep_data[timestep_data['Status'] == 0]['X'],
            timestep_data[timestep_data['Status'] == 0]['Y'],
            timestep_data[timestep_data['Status'] == 0]['Z'],
            color='brown',
            label='Dead'
        )

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Time Step: {timestep}')
        ax.legend()

        # ax.set_xlim(-10, 10)
        # ax.set_ylim(-10, 10)
        # ax.set_zlim(0, 2)

        frame_filename = os.path.join(output_dir, f'frame_{timestep:03d}.png')
        plt.savefig(frame_filename)
        plt.close()

    frames = []
    for timestep in data['TimeStep'].unique():
        frame_filename = os.path.join(output_dir, f'frame_{timestep:03d}.png')
        frames.append(Image.open(frame_filename))

    output_gif_filename = 'growth.gif'
    frames[0].save(output_gif_filename, save_all=True, append_images=frames[1:], duration=50, loop=0)

    for frame_filename in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, frame_filename))
    os.rmdir(output_dir)

def compute_volume(df, output_folder='frames'):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    alive_volumes = [] 
    dead_volumes = []
    root_volumes = []

    for timestep in range(df['TimeStep'].max() + 1):

        timestep_data = df[df['TimeStep'] == timestep]

        alive_tillers = timestep_data[timestep_data['Status'] == 1][['X', 'Y', 'Z']].values
        dead_tillers = timestep_data[timestep_data['Status'] == 0][['X','Y','Z']].values

        if len(alive_tillers) < 3:
            continue

        try:
            alive_volume = ConvexHull(alive_tillers)
            dead_volume = ConvexHull(dead_tillers)

            alive_bottom = np.argsort(alive_volume.points[:, 2])

            root_top = alive_volume.points[alive_bottom]
            root_bottom = np.copy(root_top)
            root_bottom[:, 2] = -100  # Extend the root shape downward to z=-100

            root = np.vstack((root_top, root_bottom))

            root_necromass_volume = ConvexHull(root)

            fig = plt.figure(figsize=(12, 6))

            ax1 = fig.add_subplot(121, projection="3d")
            
            ax1.plot_trisurf(*zip(*alive_tillers[alive_volume.vertices]), color='green', alpha=0.5, label='Alive')
            ax1.plot_trisurf(*zip(*dead_tillers[dead_volume.vertices]), color='brown', alpha=0.5, label='Dead')
            # ax1.plot_trisurf(*zip(*root[root_necromass_volume.vertices]), color='black', alpha=0.5, label='Root Necromass')

            ax1.set_title(f'Time Step: {timestep}')

            ax2 = fig.add_subplot(122)

            alive_volumes.append(alive_volume.volume)
            dead_volumes.append(dead_volume.volume - alive_volume.volume)
            root_volumes.append(root_necromass_volume.volume)

            ax2.plot(range(len(alive_volumes)), alive_volumes, color='green', label='Alive')
            ax2.plot(range(len(dead_volumes)), dead_volumes, color='brown', label='Dead')
            ax2.plot(range(len(root_volumes)), root_volumes, color='black', label='Root Necromass')

            ax2.set_title('Tussock Volume')
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Volume')

            frame_path = os.path.join(output_folder, f'frame_{timestep:03d}.png')
            plt.legend()
            plt.savefig(frame_path)
            plt.close()

        except Exception as e:
            print(f"Error processing frame {timestep}: {e}")

    images = []
    for i in range(df['TimeStep'].max() + 1):
        try:
            images.append(Image.open(os.path.join(output_folder, f'frame_{i:03d}.png')))
        except:
            continue

    images[0].save('tussock_volume.gif', save_all=True, append_images=images[1:], duration=50, loop=0)

    for i in range(df['TimeStep'].max() + 1):
        frame_path = os.path.join(output_folder, f'frame_{i:03d}.png')
        if os.path.exists(frame_path):
            os.remove(frame_path)
    os.rmdir(output_folder)


if __name__ == "__main__":
    main()