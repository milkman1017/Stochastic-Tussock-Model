import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numpy as np
import os
from PIL import Image
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_sims', help='number of sims to analyze')
    args = parser.parse_args()

    size_data = []
    number_tiller_data = []

    for i in range(int(args.num_sims)):
        data = pd.read_csv(f'sim_data/tiller_data_sim_num_{i}.csv')

        # size_data.append(final_size(data))
        # number_tiller_data.append(num_tillers(data))

    # plot_min_max_avg(size_data)
    # plot_tiller_number(number_tiller_data)

    data = pd.read_csv(f'sim_data/tiller_data_sim_num_0.csv')
    point_scatter_3d(data)

def final_size(data):

    sim_width = (data.groupby('TimeStep')['X'].max() - data.groupby('TimeStep')['X'].min())
    
    return sim_width

def plot_min_max_avg(size_data):
    min_values = size_data[0].copy()
    max_values = size_data[0].copy()
    avg_values = size_data[0].copy()

    for i in range(1, len(size_data)):
        min_values = pd.concat([min_values, size_data[i]], axis=1).min(axis=1)
        max_values = pd.concat([max_values, size_data[i]], axis=1).max(axis=1)
        avg_values += size_data[i]

    avg_values /= len(size_data)

    plt.plot(avg_values, label='Average', color='blue')

    plt.fill_between(range(len(avg_values)), min_values, max_values, color='blue', alpha=0.3, label='Min-Max Range')

    plt.legend()
    plt.show()

def num_tillers(data):
    grouped = data.groupby(["TimeStep", "Status"]).size().unstack(fill_value=0)

    total = grouped.sum(axis=1)
    alive = grouped[1]  # represents alive status
    dead = grouped[0]   # represents dead status

    return total, alive, dead

def plot_tiller_number(number_tiller_data):
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))

    for i, label, color in zip([0, 1, 2], ["Total Tillers", "Alive Tillers", "Dead Tillers"], ['blue', 'green', 'brown']):
        axs[i].set_title(label)

        min_values = number_tiller_data[0][i].copy()
        max_values = number_tiller_data[0][i].copy()
        mean_values = number_tiller_data[0][i].copy()

        for j, data in enumerate(number_tiller_data):
            min_values = pd.concat([min_values, data[i]], axis=1).min(axis=1)
            max_values = pd.concat([max_values, data[i]], axis=1).max(axis=1)
            mean_values += data[i]

        mean_values /= len(number_tiller_data)

        axs[i].plot(mean_values, label='Mean', color=color)

        axs[i].fill_between(range(len(mean_values)), min_values, max_values, color=color, alpha=0.3, label='Min-Max Range')

        axs[i].legend()

    plt.tight_layout()
    plt.xlabel('Time (yrs)')
    plt.ylabel('Number')
    plt.show()
    
    
def tillering_rate(data):
    pass

def point_scatter_3d(data):
    output_dir = 'scatter_plot_frames'
    os.makedirs(output_dir, exist_ok=True)

    # Plot and save frames
    for timestep in data['TimeStep'].unique():
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Filter data for the current timestep
        timestep_data = data[data['TimeStep'] == timestep]

        # Plot the 3D scatter plot with different colors for Status 1 and Status 0
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

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Time Step: {timestep}')
        ax.legend()
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_zlim(0, 2)
        # Save the frame as an image
        frame_filename = os.path.join(output_dir, f'frame_{timestep:03d}.png')
        plt.savefig(frame_filename)
        plt.close()

    # Create a GIF from the frames
    frames = []
    for timestep in data['TimeStep'].unique():
        frame_filename = os.path.join(output_dir, f'frame_{timestep:03d}.png')
        frames.append(Image.open(frame_filename))

    output_gif_filename = 'growth.gif'
    frames[0].save(output_gif_filename, save_all=True, append_images=frames[1:], duration=50, loop=0)

    # Remove the temporary frame images
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

            # Define the root shape using the bottom 10 points of the alive volume
            root_top = alive_volume.points[alive_bottom]
            root_bottom = np.copy(root_top)
            root_bottom[:, 2] = -100  # Extend the root shape downward to z=-100
            print(root_bottom)
            root = np.vstack((root_top, root_bottom))

            root_necromass_volume = ConvexHull(root)

            fig = plt.figure(figsize=(12, 6))

            # 3D plot
            ax1 = fig.add_subplot(121, projection="3d")\
            
            ax1.plot_trisurf(*zip(*alive_tillers[alive_volume.vertices]), color='green', alpha=0.5, label='Alive')
            ax1.plot_trisurf(*zip(*dead_tillers[dead_volume.vertices]), color='brown', alpha=0.5, label='Dead')
            # ax1.plot_trisurf(*zip(*root[root_necromass_volume.vertices]), color='black', alpha=0.5, label='Root Necromass')

            ax1.set_title(f'Time Step: {timestep}')

            # Volume graph
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

            # Save the frame as an image
            frame_path = os.path.join(output_folder, f'frame_{timestep:03d}.png')
            plt.legend()
            plt.savefig(frame_path)
            plt.close()

        except Exception as e:
            print(f"Error processing frame {timestep}: {e}")

    # Create animated GIF
    images = []
    for i in range(df['TimeStep'].max() + 1):
        try:
            images.append(Image.open(os.path.join(output_folder, f'frame_{i:03d}.png')))
        except:
            continue

    images[0].save('tussock_volume.gif', save_all=True, append_images=images[1:], duration=50, loop=0)

    # Remove individual frames after creating the GIF
    for i in range(df['TimeStep'].max() + 1):
        frame_path = os.path.join(output_folder, f'frame_{i:03d}.png')
        if os.path.exists(frame_path):
            os.remove(frame_path)
    os.rmdir(output_folder)


if __name__ == "__main__":
    main()