#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>

class Tiller {
public:
    static constexpr float SLA_CM2_PER_G = 98.0f;
    static constexpr float RHO_ROOT_G_PER_CM3 = 0.21f;
    static constexpr float ROOT_LENGTH_CM = 50.0f;

    // Pooled positive median of the 2016 and 2017 Leaf Area columns.
    static constexpr double FOOTPRINT_REFERENCE_LEAF_AREA_CM2 = 310.3036469;

    // Curasi et al. model-data-fusion prior bounds for individual tiller radius.
    // The archived model uses r_tiller in [0.0992, 0.25] cm. In this IBM,
    // each tiller receives an intrinsic reference radius drawn uniformly from
    // that interval at birth.
    static constexpr double FOOTPRINT_REFERENCE_RADIUS_MIN_CM = 0.0992;
    static constexpr double FOOTPRINT_REFERENCE_RADIUS_MAX_CM = 0.25;

    Tiller(int age,
           double reference_footprint_radius_cm,
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
          reference_footprint_radius(std::clamp(
              reference_footprint_radius_cm,
              FOOTPRINT_REFERENCE_RADIUS_MIN_CM,
              FOOTPRINT_REFERENCE_RADIUS_MAX_CM
          )),
          effective_footprint_radius(0.0),
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
          parent_tiller_id(parent_tiller_id) {
        if (status) {
            updateEffectiveFootprintRadiusFromLeafArea();
        }
    }

    double getReferenceFootprintRadius() const { return reference_footprint_radius; }
    double getEffectiveFootprintRadius() const { return effective_footprint_radius; }

    // Compatibility alias for code that has not yet been renamed.
    double getRadius() const { return getEffectiveFootprintRadius(); }

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

    float getDeadLeafMass() const { return dead_leaf_area / SLA_CM2_PER_G; }
    float getRootNecroMass() const { return root_necro_vol * RHO_ROOT_G_PER_CM3; }
    float getRootNecroMassCum() const { return root_necro_vol_cum * RHO_ROOT_G_PER_CM3; }

    static double sampleReferenceFootprintRadius(std::mt19937& gen) {
        std::uniform_real_distribution<double> radius_prior(
            FOOTPRINT_REFERENCE_RADIUS_MIN_CM,
            FOOTPRINT_REFERENCE_RADIUS_MAX_CM
        );
        return radius_prior(gen);
    }

    static double effectiveFootprintRadiusFromLeafArea(
        float area_cm2,
        double reference_radius_cm
    ) {
        const double area = std::max(0.0, static_cast<double>(area_cm2));
        const double reference_radius = std::clamp(
            reference_radius_cm,
            FOOTPRINT_REFERENCE_RADIUS_MIN_CM,
            FOOTPRINT_REFERENCE_RADIUS_MAX_CM
        );
        const double raw_radius =
            reference_radius
            * std::sqrt(area / FOOTPRINT_REFERENCE_LEAF_AREA_CM2);

        return std::clamp(
            raw_radius,
            FOOTPRINT_REFERENCE_RADIUS_MIN_CM,
            FOOTPRINT_REFERENCE_RADIUS_MAX_CM
        );
    }

    void updateEffectiveFootprintRadiusFromLeafArea() {
        effective_footprint_radius = effectiveFootprintRadiusFromLeafArea(
            leaf_area,
            reference_footprint_radius
        );
    }

    void setStatus(bool new_status) { status = new_status; }
    void addZ(double dz) { z += dz; }
    void mature(int age_growth) { age += age_growth; z += 0.2; }

    void setRoots(int new_roots, float diam_mm) {
        num_roots = std::max(0, new_roots);
        root_diam_mm = std::max(0.5f, std::min(5.0f, diam_mm));
    }

    void setLeafArea(float new_leaf_area) {
        leaf_area = std::max(0.0f, new_leaf_area);
        if (status) {
            updateEffectiveFootprintRadiusFromLeafArea();
        }
    }

    void grow_leaves(float area_change) {
        leaf_area = std::max(0.0f, leaf_area + area_change);
        if (status) {
            updateEffectiveFootprintRadiusFromLeafArea();
        }
    }

    bool isOverlapping(const Tiller& other) const {
        const double dx = x - other.getX();
        const double dy = y - other.getY();
        const double distance_squared = dx * dx + dy * dy;

        const double combined_footprint_radius =
            getEffectiveFootprintRadius()
            + other.getEffectiveFootprintRadius();

        return distance_squared
            <= combined_footprint_radius * combined_footprint_radius;
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
        effective_footprint_radius *= std::sqrt(LEAF_NECRO_FRAC);

        dead_leaf_area *= 0.75f;
        root_necro_vol *= 0.85f;

        if (effective_footprint_radius < 0.0) effective_footprint_radius = 0.0;
        if (dead_leaf_area < 0.0f) dead_leaf_area = 0.0f;
        if (root_necro_vol < 0.0f) root_necro_vol = 0.0f;
        if (root_necro_vol_cum < 0.0f) root_necro_vol_cum = 0.0f;
    }

    Tiller makeDaughter(int new_id, std::mt19937& gen) const {
        std::uniform_real_distribution<double> unit(0.0, 1.0);
        std::uniform_real_distribution<double> insertion_fraction(0.35, 0.65);

        static constexpr double PI = 3.14159265358979323846;
        static constexpr float DAUGHTER_LEAF_AREA_CM2 = 50.0f;

        const double daughter_reference_radius =
            sampleReferenceFootprintRadius(gen);
        const double daughter_effective_radius =
            effectiveFootprintRadiusFromLeafArea(
                DAUGHTER_LEAF_AREA_CM2,
                daughter_reference_radius
            );

        const double combined_radius =
            getEffectiveFootprintRadius() + daughter_effective_radius;
        const double daughter_distance =
            combined_radius * insertion_fraction(gen);
        const double daughter_angle = 2.0 * PI * unit(gen);

        const double x_offset = daughter_distance * std::cos(daughter_angle);
        const double y_offset = daughter_distance * std::sin(daughter_angle);
        const double z_offset = 0.1 * unit(gen);

        return Tiller(
            1,
            daughter_reference_radius,
            x + x_offset,
            y + y_offset,
            z + z_offset,
            3,
            true,
            DAUGHTER_LEAF_AREA_CM2,
            0.0f,
            0.0f,
            0.0f,
            1.0f,
            new_id,
            tiller_id
        );
    }

private:
    int age;
    double reference_footprint_radius;
    double effective_footprint_radius;
    double x;
    double y;
    double z;
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