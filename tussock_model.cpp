#include <iostream>
#include <cstdlib>
#include <ctime>  


int main() {
    int num_tillers, sim_time, kr, kd, event;
    sim_time = 200;
    num_tillers = 1;
    kr = 50; //reproduction constant
    kd = 10; //dying constant

    std::srand(std::time(0)); // Seed the random number generator

    for (int time_step = 0; time_step < sim_time; time_step++) {
        event = std::rand() % (kr + kd);

        if (event < kr) { // reproduce
            num_tillers++;
        } else if (event > kr && event < kr + kd) { // die
            num_tillers -= 1;
        }

        std::cout << num_tillers << "\n";
    }

    return 0;
}