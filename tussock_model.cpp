#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <cmath>
#include "tussock_model.h"

// Function to check and resolve overlaps
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
    int kr = 5;  // Tillering constant
    int kd = 1;  // Death constant
    int kg = 2;  // Growth constant

    int sim_time = 200;

    std::vector<Tiller> tillers;
    std::vector<Tiller> newTillers;

    Tiller initialTiller(1.0, 0.0, 0.0, 0.0, true);
    tillers.push_back(initialTiller);

    for (int time_step = 0; time_step <= sim_time; time_step++) {
        std::cout << "Time Step " << time_step << " - Number of Tillers: " << tillers.size() << "\n";

        for (Tiller& tiller : tillers) {
            if (tiller.getStatus()) {
                std::srand(static_cast<unsigned>(std::time(0)));
                int event = std::rand() % (kr + kd + kg);

                if (event < kr) { //daughter
                    Tiller newTiller = tiller.makeDaughter();

                    // Check and resolve overlaps with existing Tillers
                    newTillers.push_back(newTiller);
                    resolveOverlaps(newTillers);
                }
                else if (event > kr && event < kd + kg) { //die
                    tiller.setStatus(false);
                }
            }
        }

        // Append newTillers to the existing tillers
        tillers.insert(tillers.end(), newTillers.begin(), newTillers.end());
        newTillers.clear();
    }

    return 0;
}