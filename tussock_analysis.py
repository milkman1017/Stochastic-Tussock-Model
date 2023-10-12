import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('tiller_data.csv', header=None, names=['time_step', 'x', 'y', 'z', 'status'])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(df['x'].min(), df['x'].max())
ax.set_ylim(df['y'].min(), df['y'].max())
ax.set_zlim(df['z'].min(), df['z'].max())

scatter = ax.scatter([], [], [], marker='o')

def update(frame):
    subset = df[df['time_step'] == frame]
    scatter._offsets3d = (subset['x'], subset['y'], subset['z'])
    scatter.set_array(subset['status'])  # Use 'status' as a colormap

num_time_steps = df['time_step'].nunique()
ani = FuncAnimation(fig, update, frames=range(num_time_steps), repeat=False, blit=False)

plt.show()