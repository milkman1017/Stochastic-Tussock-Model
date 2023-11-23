import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numpy as np
import os
from PIL import Image

def main():

    data = pd.read_csv("tiller_data.csv")
    print(data)

    # num_tillers(data)
    # avg_distance(data)

    # point_scatter_3d(data)

    compute_volume(data)

def avg_distance(data):
    data['distance'] = (data['X']**2 + data['Y']**2)**0.5

    # Calculate average radius for total, status 1, and status 0
    average_radius_total = data.groupby('TimeStep')['distance'].mean()
    average_radius_alive = data[data['Status'] == 1].groupby('TimeStep')['distance'].mean()
    average_radius_dead = data[data['Status'] == 0].groupby('TimeStep')['distance'].mean()

    average_height_total = data.groupby('TimeStep')['Z'].mean()
    average_height_alive = data[data['Status']==1].groupby('TimeStep')['Z'].mean()
    average_height_dead = data[data['Status']==0].groupby('TimeStep')['Z'].mean()

    # Plot the lines
    fig, ax = plt.subplots(2)

    ax[0].plot(average_radius_total, label='Total', linewidth=1)
    ax[0].plot(average_radius_alive, label='Alive', linewidth=1, color='g', linestyle='--')
    ax[0].plot(average_radius_dead, label='Dead', linewidth=1, color='brown', linestyle='--')
    ax[0].set_title('Average Distance from Center of Tussock')

    ax[1].plot(average_height_total, label='Total', linewidth=1)
    ax[1].plot(average_height_alive, label='Alive', linewidth=1, color='g', linestyle='--')
    ax[1].plot(average_height_dead, label='Dead', linewidth=1, color='brown', linestyle='--')
    ax[1].set_title('Average Height Above Soil')

    plt.show()


def num_tillers(data):
    # Group the data by "TimeStep" and "Status"
    grouped = data.groupby(["TimeStep", "Status"]).size().unstack(fill_value=0)

    total = grouped.sum(axis=1)
    alive = grouped[1]  # Assuming 1 represents alive status
    dead = grouped[0]   # Assuming 0 represents dead status

    plt.plot(total, label='Total Tillers', linewidth=1)
    plt.plot(alive, label='Alive Tillers', linewidth=1, linestyle='--', color='g')
    plt.plot(dead, label='Dead Tillers', linewidth=1, linestyle='--', color='brown')

    plt.legend()
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

    alive_volumes = []  # List to store volumes for each timestep
    dead_volumes = []

    for timestep in range(df['TimeStep'].max() + 1):

        timestep_data = df[df['TimeStep'] == timestep]

        alive_tillers = timestep_data[timestep_data['Status'] == 1][['X', 'Y', 'Z']].values
        dead_tillers = timestep_data[timestep_data['Status'] == 0][['X','Y','Z']].values

        if len(alive_tillers) < 3:
            continue

        try:
            alive_volume = ConvexHull(alive_tillers)
            dead_volume = ConvexHull(dead_tillers)

            fig = plt.figure(figsize=(12, 6))

            # 3D plot
            ax1 = fig.add_subplot(121, projection="3d")\
            
            ax1.set_xlim(-10, 10)
            ax1.set_ylim(-10, 10)
            ax1.set_zlim(0, 2)

            ax1.plot_trisurf(*zip(*alive_tillers[alive_volume.vertices]), color='green', alpha=0.5, label='Alive')
            ax1.plot_trisurf(*zip(*dead_tillers[dead_volume.vertices]), color='brown', alpha=0.5, label='Dead')

            ax1.set_title(f'Time Step: {timestep}')

            #volume graph
            ax2 = fig.add_subplot(122)

            alive_volumes.append(alive_volume.volume)
            dead_volumes.append(dead_volume.volume - alive_volume.volume)

            ax2.plot(range(len(alive_volumes)), alive_volumes, color='green', label='Alive')
            ax2.plot(range(len(dead_volumes)), dead_volumes, color='brown', label='Dead')

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