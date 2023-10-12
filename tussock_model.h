#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <cmath>

class Tiller {
    public:
        Tiller (double radius, double leaf_length, double root_length, double x, double y, double z, double quat_x, double quat_y, double quat_z, double quat_w, bool status) : radius(radius), leaf_length(leaf_length), root_length(root_length), x(x), y(y), z(z), status(status) {}

        double getRadius() const {return radius;}
        double getLeafLength() const {return leaf_length;}
        double getRootLength() const {return root_length;}

        double getX() const {return x;}
        double getY() const {return y;}
        double getZ() const {return z;}

        double getQuatX() const {return quat_x;}
        double getQuatY() const {return quat_y;}
        double getQuatZ() const {return quat_z;}
        double getQuatW() const {return quat_w;}

        bool getStatus() const {return status;}

        double volume() {  // volume is approximated by assuming the root and leaf are cones
                         // the leaf extends straight out from the angle of the tiller
                         // the root extends stauight down along the z axis no matter the angle of the tiller
            double root_vol = (1.0 / 3.0) * 3.14 * root_length * radius * radius;
            double leaf_vol = (1.0 / 3.0) * 3.14 * leaf_length * radius * radius;
            return root_vol + leaf_vol;

        }

        void setRadius(double dRadius){
            radius += dRadius;
        }

        void setLeaf(double dLeaf){
            leaf_length += dLeaf;
        }

        void setRoot(double dRoot){
            root_length += dRoot;
        }

        void setStatus(bool new_status) {
            status = new_status;
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

            return Tiller(0.5, 15, 100, newX, newY, newZ, 0, 0, 0, 1, true);

        }

    private:
        double radius, leaf_length, root_length, x, y, z, quat_x, quat_y, quat_z, quat_w; // radius of tiller, length of leaf, length of root, x,y,z coords, quaternion representation of the angle of the tillers
        bool status; // 1 for alive, 0 for dead
};