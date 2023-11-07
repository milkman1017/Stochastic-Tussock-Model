#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <cmath>

class Tiller {
    public:
        Tiller (int size_class, double radius, double x, double y, double z, bool status) : size_class(size_class), radius(radius), x(x), y(y), z(z), status(status) {}

        int getSizeClass() const {return size_class;}

        double getRadius() const {return radius;}

        double getX() const {return x;}
        double getY() const {return y;}
        double getZ() const {return z;}

        bool getStatus() const {return status;}

        void growRadius(double dRadius){
            radius += dRadius;
        }

        void setStatus(bool new_status) {
            status = new_status;
        }

        void setClass(int new_class) {
            size_class = new_class;
        }

        bool isOverlapping(const Tiller& other) const {
            double distance = std::sqrt(std::pow(x - other.getX(), 2) + std::pow(y - other.getY(), 2) + std::pow(z - other.getZ(), 2));
            double sumOfRadii = getRadius() + other.getRadius(); // Assuming Tiller has a getRadius() method

            return (distance <= sumOfRadii);
        }
        void move() {

            std::srand(static_cast<double>(std::rand())); 

            double move_angle = (std::rand() % 360) * (3.141 / 180);
            double move_radius = 0.1;
            x += move_radius * std::cos(move_angle);
            y += move_radius * std::sin(move_angle);
        }


        Tiller makeDaughter() {
            std::srand(static_cast<unsigned>(std::rand()));

            double randomRadius = 1.0 * static_cast<double>(std::rand()) / RAND_MAX;
            double randomAngle = 2.0 * 3.141 * static_cast<double>(std::rand()) / RAND_MAX;
            double xOffset = randomRadius * std::cos(randomAngle);
            double yOffset = randomRadius * std::sin(randomAngle);

            double zOffset = 0.5 * static_cast<double>(std::rand()) / RAND_MAX;

            double newX = x + xOffset;
            double newY = y + yOffset;
            double newZ = z + zOffset;

            return Tiller(1, 0.5, newX, newY, newZ, 1);

        }

    private:
        int size_class; // size class as defined by Jim McGraw
        double radius, x, y, z; // radius of tiller, x,y,z coords
        bool status; // 1 for alive, 0 for dead
};