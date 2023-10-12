#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include "tussock_model.h"

double calculateDistance(const Tiller& tiller) {
    return std::sqrt(tiller.getX() * tiller.getX() + tiller.getY() * tiller.getY() + tiller.getZ() * tiller.getZ());
}

int main() {
    //the "tiller" object contains the following:
    // tiller radius, leaf length, root length (assumed to be 0.5, 20, and 100)
    // x,y,z coordinates
    // quaternion x,y,z,w
    // boolean status, dead or alive

    std::ofstream outputFile("tiller_data.csv", std::ios::app);  // Open CSV file in append mode
    outputFile << "TimeStep,Radius,LeafLength,RootLength,Volume,X,Y,Z\n";

    Tiller initial_tiller(0.5,15,100,0.001,0,0,0,0,0,1,1); // initalize the first tiller at coords 0,0,0,0, and parallel with the z axis 
    std::vector<Tiller> tillers;
    tillers.push_back(initial_tiller);

    //define sim parameters
    double kr = 2; //constant for creating a daughter tiller
    double kd = 2; //constant for a tiller dying
    double kg = 100; //constant for a tiller growing 

    int sim_time = 100; //total length of the sim in years

    for (int time_step = 0; time_step <= sim_time; time_step++) {
        std::vector<Tiller> step_data;
        std::vector<Tiller> newTillers; // Store new tillers separately

        std::cout << "Time Step: " << time_step <<" Number of Tillers: " << tillers.size() << '\n';

        for (Tiller& tiller : tillers) {
            if (tiller.getStatus() == 1) { // Check if the tiller is alive
                // Rest of your code for event probability and actions remains the same
                double distance = calculateDistance(tiller);
                double totalProb = kr * distance + kd * distance * distance + kg;
                double reproProb = (kr / distance) / totalProb;
                double dieProb = (kd * distance * distance) / totalProb;
                double growProb = kg / totalProb;

                double eventProb = static_cast<double>(std::rand()) / RAND_MAX;

                if (eventProb < reproProb) {  // Reproducing
                    Tiller newTiller = tiller.makeDaughter();
                    newTillers.push_back(newTiller); // Store new tiller separately
                } else if (eventProb < (reproProb + dieProb)) {  // Dying
                    tiller.setStatus(false);
                } else {  // Growing
                    tiller.setRadius(0.001);
                }

                step_data.emplace_back(tiller.getRadius(), tiller.getLeafLength(), tiller.getRootLength(), tiller.getX(), tiller.getY(), tiller.getZ(), tiller.getQuatX(), tiller.getQuatY(), tiller.getQuatZ(), tiller.getQuatW(), tiller.getStatus());
            }
        }

        // Append new tillers to the main vector after the loop
        tillers.insert(tillers.end(), newTillers.begin(), newTillers.end());

        for (Tiller& data : step_data) {
            outputFile << time_step << "," << data.getRadius() <<','<<data.getLeafLength()<<','<<data.getRootLength()<< ',' << data.volume() << ',' <<data.getX()<<','<<data.getY()<<','<<data.getZ() << '\n';
        }
    }

    return 0;
}