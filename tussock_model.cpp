#include <iostream>
#include <fstream>
#include <vector>
#include "tussock_model.h"
#include <string>
#include <thread>
#include <filesystem>
#include <random>


double calculater0(const Tiller& tiller) {
    return std::sqrt(tiller.getX() * tiller.getX() + tiller.getY() * tiller.getY());
}

double calculateTillersDistance(const Tiller& tiller1, const Tiller& tiller2){
    return std::sqrt((tiller1.getX() - tiller2.getX()) * (tiller1.getX() - tiller2.getX()) + ((tiller1.getY() - tiller2.getY()) * (tiller1.getY() - tiller2.getY())));

}

void resolveOverlaps(std::vector<Tiller>& tillers, std::random_device& rd) {
    bool overlapsExist = true;

    while (overlapsExist) {
        overlapsExist = false;  // Assume no overlaps initially
        for (size_t i = 0; i < tillers.size(); ++i) {
            for (size_t j = i + 1; j < tillers.size(); ++j) {
               
                if (calculateTillersDistance(tillers[i], tillers[j]) < 3.5 && tillers[i].isOverlapping(tillers[j])) {
                    overlapsExist = true;

                    double angle = atan2(tillers[j].getY() - tillers[i].getY(), tillers[j].getX() - tillers[i].getX());

                    double distance = calculateTillersDistance(tillers[i], tillers[j]);
                    double overlappingDistance = distance - (tillers[i].getRadius() + tillers[j].getRadius());

                    tillers[i].move(angle, overlappingDistance);
                    tillers[j].move(angle + M_PI, overlappingDistance);

                }
            }
        }
    }
}

void readFromFile(const std::string& filename, double& ks, double& kr, double& bs, double& br, double& g_min, double& g_max) {
    std::ifstream inputFile(filename);

    if (inputFile.is_open()) {
        // Assuming the file format is "ks=value" and "kr=value etc"
        inputFile.ignore(256, '=');
        inputFile >> ks;

        inputFile.ignore(256, '=');
        inputFile >> kr;

        inputFile.ignore(256, '=');
        inputFile >> bs;

        inputFile.ignore(256, '=');
        inputFile >> br;

        inputFile.ignore(256, '=');
        inputFile >> g_min;

        inputFile.ignore(256, '=');
        inputFile >> g_max;
        
        inputFile.close();

    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
}

void input(int &sim_time, int &num_sims, std::string &outdir, long unsigned int &num_threads){
    std::cout << "Enter Simulation time in Years: ";
    std::cin >> sim_time;

    std::cout << "Enter Number of Simulations: ";
    std::cin >> num_sims;

    std::cout << "Enter output (relative) directory: ";
    std::cin >> outdir;
    std::filesystem::create_directory(outdir);

    std::cout << "Enter the number of threads: ";
    std::cin >> num_threads;
}

void simulate(const int max_sim_time, const int sim_id, const std::string outdir) {
    double ks, kr, bs, br, g_min, g_max;
    readFromFile("parameters.txt", ks, kr, bs, br, g_min, g_max);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    std::uniform_real_distribution<double> g(g_min, g_max);
    std::uniform_real_distribution<double> time(40, max_sim_time);

    int sim_time = time(gen);

    std::cout << sim_time << "\n";

    double global = g(gen);

    //the "tiller" object contains the following:
    // tiller size class, 
    // radius (assumed to be 0.5)
    // x,y,z coordinates
    // boolean status, dead or alive

    //both below matrices dervied from Jim McGraw's data
    double class_transition_matrix[8][9] = {                                //probability of a tiller of size class n transitioning to a class m
        {0.3716,0.4259,0.2133,0.1066,0.0579,0.0367,0.0270,0.0305,0.0459},   //n and m are indices on the matrix
        {0.1789,0.1230,0.2678,0.1588,0.0802,0.0446,0.0154,0.0076,0.0204},
        {0.1468,0.0599,0.1991,0.2417,0.1470,0.0971,0.0541,0.0267,0.0102},
        {0.1239,0.0189,0.1303,0.2109,0.2272,0.1286,0.0888,0.0496,0.0102},
        {0.1009,0.0095,0.0616,0.1043,0.1670,0.2441,0.1506,0.1031,0.0612},
        {0.0367,0.0095,0.0118,0.0474,0.1225,0.1837,0.2124,0.1412,0.0969},
        {0.0367,0.0000,0.0000,0.0284,0.0601,0.1312,0.2124,0.2710,0.2092},
        {0.0046,0.0000,0.0000,0.0000,0.0134,0.0341,0.1467,0.2672,0.5000},
    };

    double survival_matrix[9] = {
        1.0000,0.6467,0.8839,0.8981,0.8753,0.9003,0.9073,0.8969,0.9541    //probability of a tiller size class n surviving the time step
    };

    double tillering_matrix[9] = {
        0.0000,0.0063,0.0782,0.1730,0.2739,0.3202,0.4054,0.3779,0.3776   //probability of tiller classes producing a daughter tiller
    };

    std::string out_file_name = outdir + "/tiller_data_sim_num_" + std::to_string(sim_id) + ".csv";
    std::ofstream outputFile(out_file_name, std::ios::ate);  // Open CSV file in append mode


    Tiller first_tiller(1,1, 0.5, 0.001, 0, 0, 3, 1); // initalize the first tiller at coords 0,0,0,...attributes are age, size class, radius, x,y,z, num fo roots, and status
    std::vector<Tiller> previous_step;
    previous_step.push_back(first_tiller);

    std::vector<std::string> buffer;

    for (int time_step = 0; time_step <= sim_time; time_step++) {
        std::vector<Tiller> step_data;
        std::vector<Tiller> newTillers; // Store new tillers separately

        // std::cout << "Time Step: " << time_step <<" Number of Tillers: " << previous_step.size() << '\n';

        int alive_tillers = 0;

        for (Tiller& tiller : previous_step) {
            if (tiller.getStatus() == 1) { // Check if the tiller is alive

                alive_tillers +=1;

                double distance = calculater0(tiller);
                int size_class = tiller.getSizeClass();
                double surviveEvent = dis(gen);

                //first determine if tiller lives or dies
                if (surviveEvent < (survival_matrix[size_class-1] / (ks*distance)+bs) * global) {  //if tiller lives, determine new size class from transition probabilities
                    //for now just divide the survival matrix probabilities by the distance from the center
                    //will need to get actual data to validate that this is good enough
                    //also to see what kind of relationship between distance and survival there is (linear, exponentional, etc)
    
                    double tillerEvent = dis(gen);

                    if (tillerEvent < (tillering_matrix[size_class-1] / (kr*distance)+br) * global) {
                        Tiller newTiller = tiller.makeDaughter();
                        newTillers.push_back(newTiller); //store new tiller separately, add into total data at the end of iterating through every current tiller
                        // resolveOverlaps(newTillers)
                    }

                    double transition_prob = dis(gen);

                    double cumulative_prob = 0.0;
                    int new_size_class = 0;

                    for (int m = 0; m < 9; m++) {
                        cumulative_prob += class_transition_matrix[size_class - 1][m];
                        if (transition_prob < cumulative_prob) {
                            new_size_class = m + 1; // Adding 1 to convert from 0-based index to 1-based size class
                            tiller.setClass(new_size_class);
                            break;
                        }
                    }

                    tiller.growRadius(0.01);
                    tiller.mature(1);  //increase age by one year, this will also increase the stem base by 2 mm

                    std::uniform_int_distribution<int> int_dis(2,4);
                    int new_roots = int_dis(gen);
                    tiller.growRoots(new_roots);

            //if it didnt survive, it died:
            } else {
                tiller.setStatus(0);

            };

                step_data.emplace_back(tiller.getAge(), tiller.getSizeClass(), tiller.getRadius(), tiller.getX(), tiller.getY(), tiller.getZ(), tiller.getNumRoots(), tiller.getStatus());
            } 
            else { //now iterate through dead tillers

                //simulate decay, only add tillers into the data which are not decayed for now decay is estimates at -10% biomass per year, lots of conflcting info, will need to do more lit review
                tiller.decay();
                tiller.growRoots(0);
                if (tiller.getRadius() >= 0.01) {
                    step_data.emplace_back(tiller.getAge(), tiller.getSizeClass(), tiller.getRadius(), tiller.getX(), tiller.getY(), tiller.getZ(), tiller.getNumRoots(), tiller.getStatus());
                }

            }
        }

        // Append new tillers to the main vector after the loop
        step_data.insert(step_data.end(), newTillers.begin(), newTillers.end());

        resolveOverlaps(step_data, rd);

        for (Tiller& data : step_data) {
            buffer.emplace_back(std::to_string(time_step) + ',' + std::to_string(data.getAge()) + ',' + std::to_string(data.getSizeClass()) + ','+ std::to_string(data.getRadius()) + ',' + std::to_string(data.getX()) + ',' + std::to_string(data.getY()) + ',' + std::to_string(data.getZ()) + "," + std::to_string(data.getNumRoots()) + "," + std::to_string(data.getStatus()) + '\n');
        }

        previous_step = step_data;

        if (alive_tillers > 400) {
            std::cout << "Too many alive tillers in simulation number: " << sim_id << "\n";
            break;
        }
    }

    std::cout << "Finished sim number: " << sim_id <<"\n";

    std::string big_buffer = "TimeStep,Age,SizeClass,Radius,X,Y,Z,NumRoots,Status\n";
    for(std::string& step: buffer){
        big_buffer += step;
    }
    outputFile << big_buffer;
}

int main() {
    int max_sim_time;
    int num_sims;
    std::string outdir;
    unsigned long int num_threads;

    input(max_sim_time, num_sims, outdir, num_threads);

    std::vector<std::thread> threads;

    for (int sim_id = 0; sim_id < num_sims; sim_id++) {
        std::cout << "Starting Simulation Number: " << sim_id << "\n";


        threads.emplace_back(simulate, max_sim_time, sim_id, outdir);

        if ((threads.size() == num_threads) || (sim_id == num_sims - 1)) {
            for (auto& thread : threads) {
                thread.join();
            }
            threads.clear();
        }
    }

    return 0;
}