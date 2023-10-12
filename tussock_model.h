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

            std::srand(static_cast<double>(std::rand())); 

            double move_angle = (std::rand() % 360) * (3.141 / 180);
            double move_radius = 0.5;
            x += move_radius * std::cos(move_angle);
            y += move_radius * std::sin(move_angle);
        }

        bool isOverlapping(const Tiller& other) const {
            double distance = std::sqrt(std::pow(x - other.getX(), 2) + std::pow(y - other.getY(), 2) + std::pow(z - other.getZ(),2));
            return (distance < 0.1);
        }

        void setStatus(bool newStatus) {
            status = newStatus;
        }

        void setRadius(double growth) {
            radius += growth;
        }

        Tiller makeDaughter() {
            std::srand(static_cast<double>(std::rand()));

            double radius = 0.5;  
            double angle = (std::rand() % 360) * (3.141 / 180);
            double distance = (std::rand() % 100) / 100.0 * radius;

            double xOffset = distance * std::cos(angle);
            double yOffset = distance * std::sin(angle);
            double zOffset = static_cast<double>(std::rand()) / (2.0 * RAND_MAX);  // Random value between 0 and 0.5

            double newX = x + xOffset;
            double newY = y + yOffset;
            double newZ = z + zOffset;

            return Tiller(radius, newX, newY, newZ, true);
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