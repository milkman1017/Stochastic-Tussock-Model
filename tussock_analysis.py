import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--nsims', type=int, help='The number of simulations that you wish to simulate')
    parser.add_argument('--filepath', type=str, help='The directory containing the simulations to analyze')
    parser.add_argument('--outdir', type=str, help='Output directory to save the analysis')

    args=parser.parse_args()
    
    return args

def get_num_tillers(data):

    last_time_step_data = data[data['TimeStep'] == data['TimeStep'].max()]

    last_step_num_tillers = len(last_time_step_data)
    last_step_alive_tillers = len(last_time_step_data[last_time_step_data['Status']==1])
    last_step_dead_tillers = len(last_time_step_data[last_time_step_data['Status']==0])

    return last_step_num_tillers, last_step_alive_tillers, last_step_dead_tillers

def graph_num_tillers(num_tillers,args):
    fig, ax = plt.subplots()

    for key in num_tillers:
        sns.kdeplot(num_tillers[key], label=key, linewidth=1)

    plt.legend()
    plt.xlabel('Number of Tillers')
    plt.ylabel('Density')
    plt.title('Distribution of the Number of Tillers')
    plt.savefig(f'{args.outdir}/num_tiller_dist.png')
    plt.show()

def get_diameter(data):
    last_time_step_data = data[data['TimeStep'] == data['TimeStep'].max()]

    diameter = (last_time_step_data['X'].max() - last_time_step_data['X'].min())

    return diameter

def graph_diameter_distributions(diameters, args):
    sns.kdeplot(diameters)

    plt.xlabel('Tussock Diameter (cm)')
    plt.ylabel('Density')
    plt.title('Distribution of Tussock Diameters')
    plt.savefig(f'{args.outdir}/diameters_dist.png')
    plt.show()

def get_height(data):
    last_time_step_data = data[data['TimeStep'] == data['TimeStep'].max()]

    height  = last_time_step_data[last_time_step_data['Status']==1]['Z'].mean()
    
    return height

def graph_height_distribution(heights, args):
    sns.kdeplot(heights)

    plt.xlabel('Height Above Mineral Soil (cm)')
    plt.ylabel('Distribution')
    plt.title('Distribution of Tussock Heights')
    plt.savefig(f'{args.outdir}/height_dist.png')
    plt.show()

def graph_height_vs_radius(diameters, heights, args):
    radii = np.array(diameters) / 2

    plt.scatter(radii, heights, s=2)

    # z = np.polyfit(radii, heights, 1)
    # p = np.poly1d(z)
    # plt.plot(radii, p(radii))

    plt.xlabel('Tussock Radius (cm)')
    plt.ylabel('Height Above Mineral Soil (cm)')
    plt.title('Tussock Height vs Radius')
    plt.savefig(f'{args.outdir}/height_vs_radius.png')
    plt.show()

def graph_radius_vs_tillers(diameters, num_tillers, args):
    radii = np.array(diameters) / 2

    plt.scatter(radii, num_tillers['all'], s=2)

    z = np.polyfit(radii, num_tillers['all'], 2)
    p = np.poly1d(z)
    radii_fit = np.linspace(min(radii), max(radii), 100)

    plt.plot(radii_fit, p(radii_fit), linewidth=1, label=f'Trend Line')

    plt.xlabel('Tussock Radius (cm)')
    plt.ylabel('Number of Tillers')
    plt.title('Tussock Radius vs Number of Tillers')
    plt.legend()
    plt.savefig(f'{args.outdir}/radius_vs_tillers.png')
    plt.show()

def graph_packing_index(diameters, num_tillers):
    radii = np.array(diameters)
    num_tillers = np.array(num_tillers['all'])

    packing_index = radii/num_tillers * 10
    sns.kdeplot(packing_index)
    plt.show()

def get_root_necrovolume(data):
    alive = data[data['Status']==1]
    year_data = alive.groupby('TimeStep').agg({'NumRoots': 'sum'})
   
    yearly_alive_volume = year_data['NumRoots'] * 0.03 * 50 #estimate roots as a cylinder with radius of 3 mm and a length of 50 cm
    yearly_necro_volume = yearly_alive_volume / 2 #estimate that root volume decreases in half when dead

    necro_volume = yearly_necro_volume.cumsum().iloc[-1]

    return necro_volume

def graph_radius_vs_necrovolume(diameters, necrovolumes, args):
    radii = np.array(diameters) / 2
    necrovolumes = np.array(necrovolumes)
    # Assuming necromass has a density of 0.9 g/cm^3, needs to be validated
    necromass = (necrovolumes * 0.11) / 1000

    plt.scatter(necromass, radii, s=1, label='Data Points')

    z = np.polyfit(necromass, radii, 4)
    p = np.poly1d(z)

    trendline_x = np.linspace(min(necromass), max(necromass), 100)
    trendline_y = p(trendline_x)

    plt.plot(trendline_x, trendline_y, 'r-', label='Trend Line', linewidth=1)
    plt.xlabel('Root Necromass (kg)')
    plt.ylabel('Tussock Radius (cm)')
    plt.legend()
    plt.show()

def get_ages(data):
    alive_tillers = data[data['Status']==1]
    dead_tillers = data[data['Status']==0]

    return alive_tillers['Age'].tolist(), dead_tillers['Age'].tolist()

def age_hist(ages):
    fig, ax = plt.subplots(2, figsize=(10, 8))

    # Alive tillers
    mean_alive = np.array(ages['alive']).mean()
    ax[0].hist(ages['alive'], bins='auto', color='skyblue', edgecolor='black', density=True)
    ax[0].axvline(mean_alive, color='red', linestyle='dashed', linewidth=1)
    ax[0].text(mean_alive, ax[0].get_ylim()[1]*0.9, f'Mean: {mean_alive:.2f}', color='red')
    ax[0].set_title('Alive Tillers')

    # Dead tillers
    mean_dead = np.array(ages['dead']).mean()
    ax[1].hist(ages['dead'], bins='auto', color='lightgreen', edgecolor='black', density=True)
    ax[1].axvline(mean_dead, color='red', linestyle='dashed', linewidth=1)
    ax[1].text(mean_dead, ax[1].get_ylim()[1]*0.9, f'Mean: {mean_dead:.2f}', color='red')
    ax[1].set_title('Dead Tillers')

    plt.tight_layout()
    plt.show()

def main():
    args = parse_args()

    filepath = args.filepath

    num_tillers = dict()
    num_tillers['all'] = []
    num_tillers['alive'] = []
    num_tillers['dead'] = []

    diameters = []

    heights = []

    necrovolumes = []

    ages = dict()
    ages['alive'] = []
    ages['dead'] = []

    for sim_id in range(int(args.nsims)):
        data = pd.read_csv(f'{filepath}/tiller_data_sim_num_{sim_id}.csv')

        num_tiller, num_alive_tiller, num_dead_tiller = get_num_tillers(data)
        num_tillers['all'].append(num_tiller)
        num_tillers['alive'].append(num_alive_tiller)
        num_tillers['dead'].append(num_dead_tiller)

        diameters.append(get_diameter(data))

        heights.append(get_height(data))

        necrovolumes.append(get_root_necrovolume(data))

        alive_age, dead_age = get_ages(data)
        ages['alive'].extend(alive_age)
        ages['dead'].extend(dead_age)

    graph_num_tillers(num_tillers, args)
    graph_diameter_distributions(diameters, args)
    graph_height_distribution(heights, args)
    graph_height_vs_radius(diameters, heights, args)
    graph_radius_vs_tillers(diameters, num_tillers, args)
    graph_packing_index(diameters, num_tillers)
    graph_radius_vs_necrovolume(diameters, necrovolumes, args)
    age_hist(ages)
        
if __name__ == "__main__":
    main()