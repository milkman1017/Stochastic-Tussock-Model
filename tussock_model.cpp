#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include "tussock_model.h"

struct TillerData {
    double x, y, z;
    bool status;

    TillerData(double x, double y, double z, bool status) : x(x), y(y), z(z), status(status) {}
};

double calculateDistance(const Tiller& tiller) {
    return std::sqrt(tiller.getX() * tiller.getX() + tiller.getY() * tiller.getY() + tiller.getZ() * tiller.getZ());
}

void resolveOverlaps(std::vector<Tiller>& tillers) {
    bool overlaps = true;
    while (overlaps) {
        overlaps = false;
        for (size_t i = 0; i < tillers.size(); ++i) {
            for (size_t j = i + 1; j < tillers.size(); ++j) {
                if (tillers[i].isOverlapping(tillers[j])) {
                    overlaps = true;
                    // Move the old Tiller
                    tillers[i].move();
                }
            }
        }
    }
}

int main() {
    int kr = 2;  // Reproduction constant
    int kd = 2;  // Death constant
    int kg = 10;  // Growth constant

    int sim_time = 200;

    std::ofstream outputFile("tiller_data.csv", std::ios::app);  // Open CSV file in append mode

    Tiller initialTiller(1.0, 0.001, 0.0, 0.0, true);
    std::vector<Tiller> tillers;
    tillers.push_back(initialTiller);

    for (int time_step = 0; time_step <= sim_time; time_step++) {
        std::vector<TillerData> stepData;
        std::cout << "Time step: " << time_step << " Num of tillers: " << tillers.size() << "\n";

        for (Tiller& tiller : tillers) {
            double distance = calculateDistance(tiller);

            double totalProb = kr * distance + kd * distance + kg;
            double reproProb = (kr / distance) / totalProb;
            double dieProb = (kd * distance) / totalProb;

            double eventProb = static_cast<double>(std::rand()) / RAND_MAX; // Random number between 0 and 1

            if (eventProb < reproProb) { // Reproduction
                Tiller newTiller = tiller.makeDaughter();
                resolveOverlaps(tillers);
                tillers.push_back(newTiller);
            }
            else if (eventProb < (reproProb + dieProb)) { // Death
                tiller.setStatus(false);
            }
            stepData.emplace_back(tiller.getX(), tiller.getY(), tiller.getZ(), tiller.getStatus());
        }

        // Write the data for this time step to the CSV file
        for (const TillerData& data : stepData) {
            outputFile << time_step << "," << data.x << "," << data.y << "," << data.z << "," << data.status << "\n";
        }
    }

    outputFile.close();  // Close the CSV file

    return 0;
}