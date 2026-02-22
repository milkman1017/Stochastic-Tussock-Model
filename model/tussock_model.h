// tussock_model.h
#pragma once

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <cmath>
#include <algorithm>

class Tiller {
public:
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
           float root_necro_vol_cum = 0.0f,
           float root_diam_mm = 1.0f,
           double c_store = 0.0) // NEW
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
          root_necro_vol_cum(root_necro_vol_cum),
          root_diam_mm(root_diam_mm),
          c_store(c_store) {}

    double getRadius() const { return radius; }

    double getX() const { return x; }
    double getY() const { return y; }
    double getZ() const { return z; }

    bool getStatus() const { return status; }

    int getAge() const { return age; }

    int getNumRoots() const { return num_roots; }

    float getLeafArea() const { return leaf_area; }

    float getDeadLeafArea() const { return dead_leaf_area; }

    float getRootNecroVol() const { return root_necro_vol; }
    float getRootNecroVolCum() const { return root_necro_vol_cum; }

    float getRootDiamMM() const { return root_diam_mm; }

    // NEW: stored carbon pool (gC), carried between years
    double getCStore() const { return c_store; }
    void setCStore(double v) { c_store = (std::isfinite(v) && v > 0.0) ? v : 0.0; }

    // SLA = 98 cm^2 g^-1  (Schedlbauer et al. 2018)
    static constexpr float SLA_CM2_PER_G = 98.0f;

    // Root tissue density ~ 0.21 g/cm^3 (Carex mean from GRooT proxy)
    static constexpr float RHO_ROOT_G_PER_CM3 = 0.21f;

    float getDeadLeafMass() const {
        return dead_leaf_area / SLA_CM2_PER_G;
    }

    float getRootNecroMass() const {
        return root_necro_vol * RHO_ROOT_G_PER_CM3;
    }

    float getRootNecroMassCum() const {
        return root_necro_vol_cum * RHO_ROOT_G_PER_CM3;
    }

    // ---- growth / state updates ----
    void growRadius(double dRadius) { radius += dRadius; }

    void setStatus(bool new_status) { status = new_status; }

    void addZ(double dz) { z += dz; }

    void mature(int age_growth) {
        age += age_growth;
        z += 0.2; // legacy "stem base growth" marker; geometry now uses Fetcher-based h_cm in main.cpp
    }

    void setRoots(int new_roots, float diam_mm) {
        num_roots = new_roots;
        root_diam_mm = diam_mm;

        if (num_roots < 0) num_roots = 0;

        // Clamp consistent with your carbon-mechanized model assumptions
        root_diam_mm = std::max(0.5f, std::min(5.0f, root_diam_mm));
    }

    void setLeafArea(float new_leaf_area) {
        leaf_area = new_leaf_area;
        if (leaf_area < 0) leaf_area = 0;
    }

    void grow_leaves(float area_change) {
        leaf_area += area_change;
        if (leaf_area < 0) leaf_area = 0;
    }

    // ---- geometry / interactions ----
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

    // ---- necromass bookkeeping ----
    void accumulateDeadLeafArea(float prev_leaf_area) {
        dead_leaf_area = 0.75f * dead_leaf_area + prev_leaf_area;
        if (dead_leaf_area < 0) dead_leaf_area = 0;
    }

    void decayDeadLeafArea() {
        dead_leaf_area *= 0.75f;
        if (dead_leaf_area < 0) dead_leaf_area = 0;
    }

    static constexpr float ROOT_LENGTH_CM = 50.0f;

    static inline float perRootConeVolumeCm3(float diam_mm) {
        // diameter mm -> radius cm
        const float r_cm = (diam_mm * 0.1f) * 0.5f; // 1 mm = 0.1 cm
        const float h_cm = ROOT_LENGTH_CM;
        const float pi = 3.14159265358979323846f;
        return (1.0f / 3.0f) * pi * r_cm * r_cm * h_cm;
    }

    void accumulateRootNecroFromPrevRoots(int prev_roots, float prev_diam_mm) {
        if (prev_roots <= 0) return;
        const float v_per_root = perRootConeVolumeCm3(prev_diam_mm);
        const float add = v_per_root * static_cast<float>(prev_roots);

        root_necro_vol = 0.85f * root_necro_vol + add;
        if (root_necro_vol < 0) root_necro_vol = 0;

        root_necro_vol_cum += add;
        if (root_necro_vol_cum < 0) root_necro_vol_cum = 0;
    }

    void decayRootNecroPool() {
        root_necro_vol *= 0.85f;
        if (root_necro_vol < 0) root_necro_vol = 0;
    }

    // ---- death decay ----
    void decay() {
        leaf_area = 0.0f;

        static constexpr double LEAF_NECRO_FRAC = 0.75;
        radius *= std::sqrt(LEAF_NECRO_FRAC);

        dead_leaf_area *= 0.75f;
        root_necro_vol *= 0.85f;

        if (radius < 0) radius = 0;
        if (dead_leaf_area < 0) dead_leaf_area = 0;
        if (root_necro_vol < 0) root_necro_vol = 0;
        if (root_necro_vol_cum < 0) root_necro_vol_cum = 0;

        c_store = 0.0; // stored labile carbon gone
    }

    // ---- reproduction ----
    Tiller makeDaughter() {
        double randomRadius = 1.0 * static_cast<double>(std::rand()) / RAND_MAX;
        double randomAngle  = 2.0 * 3.141 * static_cast<double>(std::rand()) / RAND_MAX;
        double xOffset      = randomRadius * std::cos(randomAngle);
        double yOffset      = randomRadius * std::sin(randomAngle);

        double zOffset = 0.1 * static_cast<double>(std::rand()) / RAND_MAX;

        double newX = x + xOffset;
        double newY = y + yOffset;
        double newZ = z + zOffset;

        // Daughter tillers start with leaf area = 50, no stored carbon initially
        return Tiller(1, 0.1, newX, newY, newZ, 3, 1, 50.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0);
    }

private:
    int age;
    double radius, x, y, z;
    int num_roots;
    bool status;
    float leaf_area;

    float dead_leaf_area;

    float root_necro_vol;
    float root_necro_vol_cum;

    float root_diam_mm;

    double c_store; 
};
