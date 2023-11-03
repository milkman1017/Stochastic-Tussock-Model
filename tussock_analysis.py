import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

data = pd.read_csv("tiller_data.csv")

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