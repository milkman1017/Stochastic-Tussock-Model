#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>

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
           int tiller_id = -1,
           int parent_tiller_id = -1)
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
          tiller_id(tiller_id),
          parent_tiller_id(parent_tiller_id) {}

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
    int getTillerId() const { return tiller_id; }
    int getParentTillerId() const { return parent_tiller_id; }
    void setTillerId(int id) { tiller_id = id; }
    void setParentTillerId(int id) { parent_tiller_id = id; }

    static constexpr float SLA_CM2_PER_G = 98.0f;
    static constexpr float RHO_ROOT_G_PER_CM3 = 0.21f;
    static constexpr float ROOT_LENGTH_CM = 50.0f;

    float getDeadLeafMass() const { return dead_leaf_area / SLA_CM2_PER_G; }
    float getRootNecroMass() const { return root_necro_vol * RHO_ROOT_G_PER_CM3; }
    float getRootNecroMassCum() const { return root_necro_vol_cum * RHO_ROOT_G_PER_CM3; }

    void growRadius(double dRadius) { radius += dRadius; }
    void setStatus(bool new_status) { status = new_status; }
    void addZ(double dz) { z += dz; }
    void mature(int age_growth) { age += age_growth; z += 0.2; }
    void setRoots(int new_roots, float diam_mm) {
        num_roots = std::max(0, new_roots);
        root_diam_mm = std::max(0.5f, std::min(5.0f, diam_mm));
    }
    void setLeafArea(float new_leaf_area) { leaf_area = std::max(0.0f, new_leaf_area); }
    void grow_leaves(float area_change) { leaf_area = std::max(0.0f, leaf_area + area_change); }

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

    void accumulateDeadLeafArea(float prev_leaf_area) {
        dead_leaf_area = std::max(0.0f, 0.75f * dead_leaf_area + prev_leaf_area);
    }

    static inline float perRootCylinderVolumeCm3(float diam_mm) {
        const float r_cm = (diam_mm * 0.1f) * 0.5f;
        const float h_cm = ROOT_LENGTH_CM;
        const float pi = 3.14159265358979323846f;
        return pi * r_cm * r_cm * h_cm;
    }

    void accumulateRootNecroFromPrevRoots(int prev_roots, float prev_diam_mm) {
        if (prev_roots <= 0) return;
        const float v_per_root = perRootCylinderVolumeCm3(prev_diam_mm);
        const float add = v_per_root * static_cast<float>(prev_roots);
        root_necro_vol = std::max(0.0f, 0.85f * root_necro_vol + add);
        root_necro_vol_cum = std::max(0.0f, root_necro_vol_cum + add);
    }

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
    }

    Tiller makeDaughter(int new_id) const {
        double randomRadius = 1.0 * static_cast<double>(std::rand()) / RAND_MAX;
        double randomAngle = 2.0 * 3.14159265358979323846 * static_cast<double>(std::rand()) / RAND_MAX;
        double xOffset = randomRadius * std::cos(randomAngle);
        double yOffset = randomRadius * std::sin(randomAngle);
        double zOffset = 0.1 * static_cast<double>(std::rand()) / RAND_MAX;
        return Tiller(1, 0.1, x + xOffset, y + yOffset, z + zOffset, 3, 1, 50.0f, 0.0f, 0.0f, 0.0f, 1.0f, new_id, tiller_id);
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
    int tiller_id;
    int parent_tiller_id;
};
