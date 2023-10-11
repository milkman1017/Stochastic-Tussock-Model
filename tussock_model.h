#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <cmath>

class Tiller {
    public:
        Tiller(double radius, double x, double y, double z, bool status): radius(radius), x(x), y(y), z(z), status(status) {}

        double getRadius() const { return radius; }

        double getX() const {return x; }
        double getY() const {return y; }
        double getZ() const {return z;}

        double getStatus() const {return status;}

        void move() {

            std::srand(static_cast<unsigned>(std::time(0))); 

            double move_angle = (std::rand() % 360) * (3.141 / 180);
            double move_radius = 1.0;
            x += move_radius * std::cos(move_angle);
            y += move_radius * std::sin(move_angle);
        }

        bool isOverlapping(const Tiller& other) const {
            double distance = std::sqrt(std::pow(x - other.getX(), 2) + std::pow(y - other.getY(), 2));
            return (distance < 1.0);
        }

        void setStatus(bool newStatus) {
            status = newStatus;
        }

        Tiller makeDaughter() {
        std::srand(static_cast<unsigned>(std::time(0)));

        double xOffset = static_cast<double>(std::rand() % 200 - 100) / 100.0; // Random value between -1 and 1
        double yOffset = static_cast<double>(std::rand() % 200 - 100) / 100.0; // Random value between -1 and 1
        double zOffset = static_cast<double>(std::rand() % 200 - 100) / 100.0; // Random value between -1 and 1

        // Calculate new coordinates
        double newX = x + xOffset;
        double newY = y + yOffset;
        double newZ = z + zOffset;

        // Create and return the new Tiller
        return Tiller(1.0, newX, newY, newZ, true);
    }

    private:
        double radius;

        double x;
        double y;
        double z;

        bool status;

};

class Leaf: public Tiller {
    public:
        Leaf(double length, double radius, double x, double y, double z, bool status)
        : Tiller(radius, x, y, z, status) {}
};


class Root : public Tiller {
public:
    Root(double length, double radius, double x, double y, double z, bool status)
        : Tiller(radius, x, y, z, status) {

    }
};