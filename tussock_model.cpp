#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <cmath>
#include "tussock_model.h"

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
    int kr = 10;  // Reproduction constant
    int kd = 2.5;  // Death constant
    int kg = 10;  // Growth constant

    int sim_time = 200;

    std::vector<Tiller> tillers;
    std::vector<Tiller> newTillers;

    Tiller initialTiller(1.0, 0.001, 0.0, 0.0, true);
    tillers.push_back(initialTiller);

    for (int time_step = 0; time_step <= sim_time; time_step++) {
        int aliveTillers = 0; // Counter for alive tillers
        std::cout << "Time Step " << time_step << " - Number of Tillers: " << tillers.size() << "\n";

        for (Tiller& tiller : tillers) {
            if (tiller.getStatus()) {
                aliveTillers++;

                double distance = calculateDistance(tiller);

                double totalProb = kr * distance + kd * distance + kg;
                double reproProb = (kr / distance) / totalProb;
                double dieProb = (kd * distance) / totalProb;

                double eventProb = static_cast<double>(std::rand()) / RAND_MAX; // Random number between 0 and 1

                if (eventProb < reproProb) { // Reproduction
                    Tiller newTiller = tiller.makeDaughter();
                    newTillers.push_back(newTiller);
                    resolveOverlaps(newTillers);
                }
                else if (eventProb < (reproProb + dieProb)) { // Death
                    tiller.setStatus(false);
                }
            }
        }

        std::cout << "Number of Alive Tillers: " << aliveTillers << "\n"; 
        tillers.insert(tillers.end(), newTillers.begin(), newTillers.end());
        newTillers.clear();
    }

    return 0;
}