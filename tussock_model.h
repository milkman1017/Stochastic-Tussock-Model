
class Stem {
    public:
        Stem(double length, double radius, double x, double y, double z): length(length), radius(radius), x(x), y(y), z(z) {}

        double getRadius() const { return radius; }
        double getLength() const { return length; }

        double getX() const {return x; }
        double getY() const {return y; }
        double getZ() const {return z;}

        double getVolume() const {3.141 * radius * length;}
            
    private:
        double radius;
        double length;

        double x;
        double y;
        double z;

};
