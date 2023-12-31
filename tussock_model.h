#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <cmath>

class Tiller {
    public:
    Tiller(int age, int size_class, double radius, double x, double y, double z, int num_roots, bool status)
        : age(age), size_class(size_class), radius(radius), x(x), y(y), z(z), num_roots(num_roots), status(status) {}


        int getSizeClass() const {return size_class;}

        double getRadius() const {return radius;}

        double getX() const {return x;}
        double getY() const {return y;}
        double getZ() const {return z;}

        bool getStatus() const {return status;}

        int getAge() const {return age;}

        int getNumRoots() const {return num_roots;}

        void growRadius(double dRadius){
            radius += dRadius;
        }

        void setStatus(bool new_status) {
            status = new_status;
        }

        void setClass(int new_class) {
            size_class = new_class;
        }

        void mature (int age_growth) {
            age += age_growth;
            z += 0.2;   //simulates the growth of the stem base every year
        }

        void growRoots(int new_roots) {
            num_roots = new_roots;
        }

        bool isOverlapping(const Tiller& other) const {
            double distance = std::sqrt(std::pow(x - other.getX(), 2) + std::pow(y - other.getY(), 2));
            double sumOfRadii = getRadius() + other.getRadius();

            double tiller1_base_height = 0.2 * age * z;
            double tiller2_base_height = 0.2 * other.getAge() * other.getZ();

            double tillers_height = tiller1_base_height + tiller2_base_height;
            double not_overlapping_height = 0.2 * age + 0.2 * other.getAge();

            return(distance <= sumOfRadii && tillers_height >= not_overlapping_height);

        }

        void move(double move_angle, double move_distance){
        
            x += move_distance * std::cos(move_angle);
            y += move_distance * std::sin(move_angle);
        }

        void decay() {
            double decay_amount = 1 - 0.02;

            radius *= decay_amount;
        }

        Tiller makeDaughter() {

            double randomRadius = 1.0 * static_cast<double>(std::rand()) / RAND_MAX;
            double randomAngle = 2.0 * 3.141 * static_cast<double>(std::rand()) / RAND_MAX;
            double xOffset = randomRadius * std::cos(randomAngle);
            double yOffset = randomRadius * std::sin(randomAngle);

            //0.1 is to ensure that the daughter tiller is made slightly above the parent
            double zOffset = 0.1 * static_cast<double>(std::rand()) / RAND_MAX;

            double newX = x + xOffset;
            double newY = y + yOffset;
            double newZ = z + zOffset;
            // double newZ = z;

            return Tiller(1,1, 0.5, newX, newY, newZ, 3, 1);

        }

    private:
        int age;
        int size_class; // size class as defined by Jim McGraw
        double radius, x, y, z; // radius of tiller, x,y,z coords
        int num_roots;
        bool status; // 1 for alive, 0 for dead
};