import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numpy as np

def main():

    data = pd.read_csv("tiller_data.csv")
    print(data)

    num_tillers(data)
    avg_distance(data)

def avg_distance(data):

    data['distance'] = (data['X']**2 + data['Y']**2 + data['Z']**2)**0.5
    average_radius = data.groupby('TimeStep')['distance'].mean()
    
    fig, ax = plt.subplots()

    ax.plot(average_radius, linewidth=1)
    plt.title("Average Distance from the Center")
    plt.xlabel("Time Step (yrs)")
    plt.ylabel("Radius (cm)")

    plt.show()


def num_tillers(data):
    # Group the data by "TimeStep" and "Status"
    grouped = data.groupby(["TimeStep", "Status"]).size().unstack(fill_value=0)

    total = grouped.sum(axis=1)
    alive = grouped[1]  # Assuming 1 represents alive status
    dead = grouped[0]   # Assuming 0 represents dead status

    plt.plot(total, label='Total Tillers', linewidth=1)
    plt.plot(alive, label='Alive Tillers', linewidth=1, linestyle='--')
    plt.plot(dead, label='Dead Tillers', linewidth=1, linestyle='--')

    plt.legend()
    plt.show()
    
def tillering_rate(data):
    pass

def point_scatter_3d(data):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Tiller Data Over Time')

    grouped_data = data.groupby('TimeStep')

    def update(frame):
        ax.cla()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Time Step {frame}')
        
        time_step_data = grouped_data.get_group(frame)
        
        ax.scatter(time_step_data['X'], time_step_data['Y'], time_step_data['Z'], s=time_step_data['Radius']*10, c='g', marker='o')
        
        # Save the frame as a PNG image
        # plt.savefig(f'model_gif_images/frame_{frame:04d}.png')

    ani = FuncAnimation(fig, update, frames=data['TimeStep'], repeat=False)

    # Display the animation
    plt.show()

if __name__ == "__main__":
    main()
