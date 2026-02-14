// tussock_model.h
#pragma once

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <cmath>

class Tiller {
public:
    // Added dead_leaf_area with default = 0
    // Root necromass volumes (pool + cumulative)
    Tiller(int age,
           double radius,
           double x,
           double y,
           double z,
           int num_roots,
           bool status,
           float leaf_area,
           float dead_leaf_area = 0.0f,
           float root_necro_vol = 0.0f,
           float root_necro_vol_cum = 0.0f)
        : age(age),
          radius(radius),
          x(x),
          y(y),
          z(z),
          num_roots(num_roots),
          status(status),
          leaf_area(leaf_area),
          dead_leaf_area(dead_leaf_area),
          root_necro_vol(root_necro_vol),
          root_necro_vol_cum(root_necro_vol_cum) {}

    double getRadius() const { return radius; }

    double getX() const { return x; }
    double getY() const { return y; }
    double getZ() const { return z; }

    bool getStatus() const { return status; }

    int getAge() const { return age; }

    int getNumRoots() const { return num_roots; }

    float getLeafArea() const { return leaf_area; }

    // Dead leaf pool (area units)
    float getDeadLeafArea() const { return dead_leaf_area; }

    // Root necromass pools (volume units)
    float getRootNecroVol() const { return root_necro_vol; }
    float getRootNecroVolCum() const { return root_necro_vol_cum; }

    // --- conversions (from Methods) ---
    // SLA = 98 cm^2 g^-1  (Schedlbauer et al. 2018)
    static constexpr float SLA_CM2_PER_G = 98.0f;

    // Root tissue density ~ 0.21 g/cm^3 (Carex mean from GRooT proxy)
    static constexpr float RHO_ROOT_G_PER_CM3 = 0.21f;

    // Computed necromass (dry mass)
    float getDeadLeafMass() const {
        // Assumes dead_leaf_area is in cm^2. If dead_leaf_area is mm^2, convert first: *0.01f.
        return dead_leaf_area / SLA_CM2_PER_G;
    }

    float getRootNecroMass() const {
        return root_necro_vol * RHO_ROOT_G_PER_CM3;
    }

    float getRootNecroMassCum() const {
        return root_necro_vol_cum * RHO_ROOT_G_PER_CM3;
    }

    void growRadius(double dRadius) { radius += dRadius; }

    void setStatus(bool new_status) { status = new_status; }

    void addZ(double dz) { z += dz; }

    void mature(int age_growth) {
        age += age_growth;
        z += 0.2; // simulates the growth of the stem base every year
    }

    void growRoots(int new_roots) { num_roots = new_roots; }

    bool isOverlapping(const Tiller& other) const {
        double distance = std::sqrt(std::pow(x - other.getX(), 2) + std::pow(y - other.getY(), 2));
        double sumOfRadii = getRadius() + other.getRadius();

        double tiller1_base_height = 0.2 * age;
        double tiller2_base_height = 0.2 * other.getAge();

        double tillers_height = tiller1_base_height + tiller2_base_height;
        double not_overlapping_height = 0.2 * age + 0.2 * other.getAge();

        return (distance <= sumOfRadii && tillers_height >= not_overlapping_height);
    }

    void move(double move_angle, double move_distance) {
        x += move_distance * std::cos(move_angle);
        y += move_distance * std::sin(move_angle);
    }

    // Dead leaf pool: D_{t+1} = 0.75 * D_t + A_t  (25% mass/volume loss per year)
    void accumulateDeadLeafArea(float prev_leaf_area) {
        dead_leaf_area = 0.75f * dead_leaf_area + prev_leaf_area;
        if (dead_leaf_area < 0) dead_leaf_area = 0;
    }

    void decayDeadLeafArea() {
        dead_leaf_area *= 0.75f;
        if (dead_leaf_area < 0) dead_leaf_area = 0;
    }

    // Root necromass accounting
    // Assumption: previous year's roots die/turn over and enter necromass once per year.
    // pool decays by 15%/yr (i.e., multiply by 0.85), cumulative never decays.
    static constexpr float ROOT_CONE_DIAMETER_MM = 1.0f;
    static constexpr float ROOT_LENGTH_CM = 50.0f;

    static inline float perRootConeVolumeCm3() {
        // diameter 1 mm => radius 0.5 mm = 0.05 cm
        const float r_cm = (ROOT_CONE_DIAMETER_MM * 0.1f) * 0.5f; // 1 mm = 0.1 cm
        const float h_cm = ROOT_LENGTH_CM;
        const float pi = 3.14159265358979323846f;
        return (1.0f / 3.0f) * pi * r_cm * r_cm * h_cm;
    }

    void accumulateRootNecroFromPrevRoots(int prev_roots) {
        if (prev_roots <= 0) return;
        const float v_per_root = perRootConeVolumeCm3();
        const float add = v_per_root * static_cast<float>(prev_roots);

        // decaying necro pool (15%/yr decay + additions)
        root_necro_vol = 0.85f * root_necro_vol + add;
        if (root_necro_vol < 0) root_necro_vol = 0;

        // cumulative necro (no decay)
        root_necro_vol_cum += add;
        if (root_necro_vol_cum < 0) root_necro_vol_cum = 0;
    }

    void decayRootNecroPool() {
        root_necro_vol *= 0.85f;
        if (root_necro_vol < 0) root_necro_vol = 0;
        // cumulative does not decay
    }

    void decay() {
        // For dead tillers: living leaf tissue is gone; only dead pools persist.
        leaf_area = 0.0f;

        // --- necromass proportional decay ---
        // Leaf litter: 25% y-o-y mass loss => multiply mass/volume by 0.75
        // If we assume constant bulk density and fixed height, volume ∝ R^2,
        // so radius scales as sqrt(volume fraction).
        static constexpr double LEAF_NECRO_FRAC = 0.75;
        radius *= std::sqrt(LEAF_NECRO_FRAC);

        // Dead leaf pool decays by 25% every year
        dead_leaf_area *= 0.75f;

        // Root necro pool decays by 15% every year
        root_necro_vol *= 0.85f;

        if (radius < 0) radius = 0;
        if (dead_leaf_area < 0) dead_leaf_area = 0;
        if (root_necro_vol < 0) root_necro_vol = 0;
        if (root_necro_vol_cum < 0) root_necro_vol_cum = 0;
    }

    void grow_leaves(float area_change) {
        leaf_area += area_change;
        if (leaf_area < 0) leaf_area = 0;
    }

    void setLeafArea(float new_leaf_area) {
        leaf_area = new_leaf_area;
        if (leaf_area < 0) leaf_area = 0;
    }

    Tiller makeDaughter() {
        double randomRadius = 1.0 * static_cast<double>(std::rand()) / RAND_MAX;
        double randomAngle  = 2.0 * 3.141 * static_cast<double>(std::rand()) / RAND_MAX;
        double xOffset      = randomRadius * std::cos(randomAngle);
        double yOffset      = randomRadius * std::sin(randomAngle);

        // 0.1 is to ensure that the daughter tiller is made slightly above the parent
        double zOffset = 0.1 * static_cast<double>(std::rand()) / RAND_MAX;

        double newX = x + xOffset;
        double newY = y + yOffset;
        double newZ = z + zOffset;

        // daughter starts with dead pools = 0
        return Tiller(1, 0.25, newX, newY, newZ, 3, 1, 50, 0.0f, 0.0f, 0.0f);
    }

private:
    int age;
    double radius, x, y, z; // radius of tiller, x,y,z coords
    int num_roots;
    bool status; // 1 for alive, 0 for dead
    float leaf_area;

    // dead leaf pool
    float dead_leaf_area;

    // root necromass (volume)
    float root_necro_vol;      // decaying pool
    float root_necro_vol_cum;  // cumulative total
};
