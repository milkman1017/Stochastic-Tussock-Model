#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits.h>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <deque>
#if defined(__linux__)
#include <unistd.h>
#endif

#include "tussock_model.h"

static std::mutex g_print_mutex;
static constexpr int DISABLE_REPRO = 0;


static std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static std::uint64_t read_base_seed() {
    const char* raw = std::getenv("TUSSOCK_BASE_SEED");
    if (raw != nullptr && *raw != '\0') {
        try {
            std::size_t used = 0;
            const std::string value_text(raw);
            const std::uint64_t value = std::stoull(value_text, &used, 10);
            if (used == value_text.size()) return value;
        } catch (...) {
        }
    }

    return static_cast<std::uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count()
    );
}

static inline std::string trim(const std::string& s) {
    size_t a = 0;
    while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) a++;
    size_t b = s.size();
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) b--;
    return s.substr(a, b - a);
}

static inline double clamp01(double p) { return std::max(0.0, std::min(1.0, p)); }
static inline double logistic(double z) { return 1.0 / (1.0 + std::exp(-z)); }
static inline double clamp(double x, double lo, double hi) { return std::max(lo, std::min(hi, x)); }


static double vector_quantile(std::vector<double> values, double q) {
    if (values.empty()) return 0.0;

    q = std::max(0.0, std::min(1.0, q));
    std::sort(values.begin(), values.end());

    if (values.size() == 1) return values[0];

    const double index = q * static_cast<double>(values.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(index));
    const std::size_t hi = static_cast<std::size_t>(std::ceil(index));
    const double fraction = index - static_cast<double>(lo);

    return values[lo] * (1.0 - fraction) + values[hi] * fraction;
}

static double calculate_robust_tussock_diameter(
    const std::vector<Tiller>& tillers,
    double lower_q = 0.025,
    double upper_q = 0.975
) {
    std::vector<double> x_lower_edges;
    std::vector<double> x_upper_edges;
    std::vector<double> y_lower_edges;
    std::vector<double> y_upper_edges;

    x_lower_edges.reserve(tillers.size());
    x_upper_edges.reserve(tillers.size());
    y_lower_edges.reserve(tillers.size());
    y_upper_edges.reserve(tillers.size());

    for (const auto& t : tillers) {
        const double r = t.getEffectiveFootprintRadius();

        x_lower_edges.push_back(t.getX() - r);
        x_upper_edges.push_back(t.getX() + r);
        y_lower_edges.push_back(t.getY() - r);
        y_upper_edges.push_back(t.getY() + r);
    }

    if (tillers.empty()) return 0.0;

    const double width_x =
        vector_quantile(x_upper_edges, upper_q)
        - vector_quantile(x_lower_edges, lower_q);

    const double width_y =
        vector_quantile(y_upper_edges, upper_q)
        - vector_quantile(y_lower_edges, lower_q);

    return 0.5 * (
        std::max(0.0, width_x)
        + std::max(0.0, width_y)
    );
}

static std::filesystem::path get_project_root() {
    try {
#if defined(__linux__)
        char buf[PATH_MAX];
        ssize_t len = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
        if (len > 0) {
            buf[len] = '\0';
            std::filesystem::path exe_path = std::filesystem::canonical(std::filesystem::path(buf));
            return exe_path.parent_path().parent_path();
        }
#endif
        return std::filesystem::current_path();
    } catch (...) {
        return std::filesystem::current_path();
    }
}

std::string ini_get(const std::string& ini_path,
                    const std::string& wanted_section,
                    const std::string& wanted_key,
                    const std::string& fallback) {
    std::ifstream in(ini_path);
    if (!in.is_open()) return fallback;

    std::string line, current_section;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == ';' || line[0] == '#') continue;
        if (line.front() == '[' && line.back() == ']') {
            current_section = trim(line.substr(1, line.size() - 2));
            continue;
        }
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = trim(line.substr(0, eq));
        std::string val = trim(line.substr(eq + 1));
        if (current_section == wanted_section && key == wanted_key) {
            return val.empty() ? fallback : val;
        }
    }
    return fallback;
}

static bool ini_get_bool(const std::string& ini_path,
                         const std::string& section,
                         const std::string& key,
                         bool fallback) {
    std::string v = ini_get(ini_path, section, key, fallback ? "true" : "false");
    std::string lv = trim(v);
    std::transform(lv.begin(), lv.end(), lv.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return (lv == "1" || lv == "true" || lv == "yes" || lv == "on");
}

double calculater0(const Tiller& tiller) {
    return std::sqrt(tiller.getX() * tiller.getX() + tiller.getY() * tiller.getY());
}

static inline std::int64_t cell_key(int cx, int cy) {
    return (static_cast<std::int64_t>(cx) << 32) ^ (static_cast<std::uint32_t>(cy));
}

struct OverlapStats {
    int passes = 0;
    long long candidates = 0;
    long long overlapped = 0;
    long long z_adjusts = 0;
    double max_penetration = 0.0;
    double ms = 0.0;
};

struct MechanismConfig {
    bool use_spatial_survival = false;
    bool use_spatial_reproduction = false;
    bool use_spatial_establishment = false;
    bool use_crowding_survival = false;
    bool use_crowding_reproduction = false;
    bool use_crowding_establishment = false;
    double crowding_radius_cm = 2.0;
    std::string spatial_survival_form = "linear";
    std::string spatial_reproduction_form = "linear";
    std::string spatial_establishment_form = "linear";
};

struct ModelParams {
    double ks = 0.0;
    double kr = 0.0;
    double bs = 0.0;
    double br = 0.0;
    double ke = 0.0;
    double be = 0.0;
    double c_space_survival = 0.5;
    double c_space_reproduction = 0.5;
    double c_space_establishment = 0.5;
    double k_crowd_survival = 0.0;
    double k_crowd_reproduction = 0.0;
    double k_crowd_establishment = 0.0;
    double leaf_offset = 0.0;
    double base_establishment = 0.5;
};

struct SimSummary {
    int sim_id = -1;
    int final_t = -1;
    double final_diameter = 0.0;
    int alive_y = -1;
    double rmax_y = 0.0;
    int overflow_t = -1;
    int extinct_t = -1;
    int missing_year = 1;
    int alive_final = 0;
    double leafarea_mean_y = std::nan("");
    long long cumulative_attempted_daughters = 0;
    long long cumulative_established_births = 0;
    long long cumulative_deaths = 0;
    long long cumulative_tillers_created = 1;
    double mean_reproduction_probability = std::nan("");
    double mean_establishment_probability = std::nan("");
};

struct RuntimeConfig {
    std::filesystem::path config_path;
    std::filesystem::path project_root;
    std::filesystem::path config_dir;
    std::filesystem::path output_root;
    std::filesystem::path param_file_path;
    int constraint_year = 25;
    int alive_overflow_threshold = 600;
};

static RuntimeConfig load_runtime_config(const std::string& config_path_raw) {
    RuntimeConfig rc;
    rc.project_root = get_project_root();

    std::filesystem::path config_path = std::filesystem::path(config_path_raw);
    if (config_path.is_relative()) config_path = rc.project_root / config_path;
    rc.config_path = std::filesystem::weakly_canonical(config_path);
    rc.config_dir = rc.config_path.parent_path();

    std::string output_dir_raw = ini_get(rc.config_path.string(), "Paths", "output_dir", "parameterization_outputs");
    std::filesystem::path output_root = std::filesystem::path(output_dir_raw);
    if (output_root.is_relative()) output_root = rc.config_dir / output_root;
    rc.output_root = output_root;

    rc.param_file_path = rc.output_root / "parameters.txt";
    rc.constraint_year = std::stoi(ini_get(rc.config_path.string(), "Constraints", "constraint_year", "25"));
    rc.alive_overflow_threshold = std::stoi(ini_get(rc.config_path.string(), "Constraints", "alive_overflow_threshold", "600"));

    return rc;
}

static inline bool should_prune_dead(const Tiller& t) {
    constexpr double EPS_R = 1e-6;
    constexpr double EPS_A = 1e-6;
    constexpr double EPS_M = 1e-8;

    if (t.getStatus() == 1) return false;

    const double r = t.getEffectiveFootprintRadius();
    const double la = t.getLeafArea();
    const double dla = t.getDeadLeafArea();
    const double dlm = t.getDeadLeafMass();
    const double rnv = t.getRootNecroVol();
    const double rnvc = t.getRootNecroVolCum();
    const double rnm = t.getRootNecroMass();
    const double rnmc = t.getRootNecroMassCum();

    return (std::abs(r) <= EPS_R)
        && (std::abs(la) <= EPS_A)
        && (std::abs(dla) <= EPS_M)
        && (std::abs(dlm) <= EPS_M)
        && (std::abs(rnv) <= EPS_M)
        && (std::abs(rnm) <= EPS_M)
        && (std::abs(rnvc) <= EPS_M)
        && (std::abs(rnmc) <= EPS_M);
}

void resolveOverlaps(std::vector<Tiller>& tillers, OverlapStats& stats) {
    auto t0 = std::chrono::high_resolution_clock::now();

    stats = OverlapStats{};
    stats.max_penetration = 0.0;

    if (tillers.size() < 2) {
        auto t1 = std::chrono::high_resolution_clock::now();
        stats.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return;
    }

    const int MAX_PASSES = 40;
    const double PEN_TOL = 1e-7;
    const double SLOP = 1e-4;
    const double STIFFNESS = 0.85;
    const double STOP_TOTAL_CORRECTION = 1e-6;
    const double EPS = 1e-12;

    double max_rsum = 0.0;
    for (const auto& t : tillers) {
        max_rsum = std::max(max_rsum, 2.0 * t.getEffectiveFootprintRadius());
    }

    const double CELL = std::max(3.5, max_rsum);

    auto cell_of = [&](double x, double y) -> std::pair<int, int> {
        return {
            static_cast<int>(std::floor(x / CELL)),
            static_cast<int>(std::floor(y / CELL))
        };
    };

    for (int pass = 0; pass < MAX_PASSES; ++pass) {
        std::unordered_map<std::int64_t, std::vector<int>> grid;
        grid.reserve(tillers.size() * 2);

        for (int i = 0; i < static_cast<int>(tillers.size()); ++i) {
            auto [cx, cy] = cell_of(tillers[i].getX(), tillers[i].getY());
            grid[cell_key(cx, cy)].push_back(i);
        }

        long long overlaps_this_pass = 0;
        double total_correction = 0.0;
        double max_pen_this_pass = 0.0;

        for (const auto& kv : grid) {
            std::int64_t key = kv.first;
            const std::vector<int>& ids = kv.second;

            int cx = static_cast<int>(key >> 32);
            int cy = static_cast<int>(static_cast<std::int32_t>(key & 0xffffffff));

            for (int i : ids) {
                for (int gx = -1; gx <= 1; ++gx) {
                    for (int gy = -1; gy <= 1; ++gy) {
                        auto it = grid.find(cell_key(cx + gx, cy + gy));
                        if (it == grid.end()) continue;

                        for (int j : it->second) {
                            if (j <= i) continue;

                            double dx = tillers[j].getX() - tillers[i].getX();
                            double dy = tillers[j].getY() - tillers[i].getY();

                            double d2 = dx * dx + dy * dy;
                            double rsum = tillers[i].getEffectiveFootprintRadius() + tillers[j].getEffectiveFootprintRadius();

                            if (rsum <= 0.0) continue;
                            if (d2 >= rsum * rsum) continue;

                            double dist = std::sqrt(std::max(d2, EPS));
                            double penetration = rsum - dist;

                            if (penetration <= PEN_TOL) continue;
                            if (penetration <= SLOP) continue;

                            if (!tillers[i].isOverlapping(tillers[j])) continue;

                            double angle;

                            if (d2 <= EPS) {
                                // Perfectly coincident centers: choose a deterministic direction
                                // based on indices so the pair can separate.
                                double theta = 2.39996322972865332 * static_cast<double>(i + j + 1);
                                angle = theta;
                            } else {
                                angle = std::atan2(dy, dx);
                            }

                            double correction = STIFFNESS * (penetration - SLOP);
                            double step = 0.5 * correction;

                            tillers[i].move(angle + M_PI, step);
                            tillers[j].move(angle, step);

                            overlaps_this_pass++;
                            total_correction += correction;
                            max_pen_this_pass = std::max(max_pen_this_pass, penetration);
                        }
                    }
                }
            }
        }

        stats.passes = pass + 1;
        stats.overlapped += overlaps_this_pass;
        stats.max_penetration = std::max(stats.max_penetration, max_pen_this_pass);

        if (overlaps_this_pass == 0 || total_correction < STOP_TOTAL_CORRECTION) {
            break;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    stats.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static int count_alive_neighbors_at_position(const std::vector<Tiller>& tillers,
                                             double x,
                                             double y,
                                             double R,
                                             int excluded_tiller_id = -1) {
    const double R2 = R * R;
    int count = 0;
    for (const auto& other : tillers) {
        if (other.getStatus() != 1) continue;
        if (other.getTillerId() == excluded_tiller_id) continue;
        const double dx = x - other.getX();
        const double dy = y - other.getY();
        if (dx * dx + dy * dy <= R2) count++;
    }
    return count;
}

static std::vector<int> compute_local_crowding_alive(const std::vector<Tiller>& tillers, double R) {
    const double R2 = R * R;
    const double CELL = R;
    std::vector<int> crowd(tillers.size(), 0);
    if (tillers.size() < 2) return crowd;

    std::unordered_map<std::int64_t, std::vector<int>> grid;
    grid.reserve(tillers.size() * 2);

    auto cell_of = [&](double x, double y) -> std::pair<int, int> {
        return {static_cast<int>(std::floor(x / CELL)), static_cast<int>(std::floor(y / CELL))};
    };

    for (int i = 0; i < static_cast<int>(tillers.size()); ++i) {
        if (tillers[i].getStatus() != 1) continue;
        auto [cx, cy] = cell_of(tillers[i].getX(), tillers[i].getY());
        grid[cell_key(cx, cy)].push_back(i);
    }

    for (int i = 0; i < static_cast<int>(tillers.size()); ++i) {
        if (tillers[i].getStatus() != 1) continue;
        auto [cx, cy] = cell_of(tillers[i].getX(), tillers[i].getY());
        int count = 0;
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                auto it = grid.find(cell_key(cx + dx, cy + dy));
                if (it == grid.end()) continue;
                for (int j : it->second) {
                    if (j == i) continue;
                    double ddx = tillers[i].getX() - tillers[j].getX();
                    double ddy = tillers[i].getY() - tillers[j].getY();
                    if (ddx * ddx + ddy * ddy <= R2) count++;
                }
            }
        }
        crowd[i] = count;
    }

    return crowd;
}

void readFromFile(const std::string& filename, ModelParams& p) {
    std::ifstream inputFile(filename);
    if (!inputFile.is_open()) {
        std::lock_guard<std::mutex> lk(g_print_mutex);
        std::cerr << "Unable to open file: " << filename << "\n";
        return;
    }

    std::unordered_map<std::string, double> kv;
    std::string line;
    while (std::getline(inputFile, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#' || line[0] == ';') continue;
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        try {
            kv[trim(line.substr(0, eq))] = std::stod(trim(line.substr(eq + 1)));
        } catch (...) {}
    }

    auto set_if = [&](const char* key, double& ref) {
        if (kv.count(key)) ref = kv[key];
    };

    set_if("ks", p.ks);
    set_if("kr", p.kr);
    set_if("bs", p.bs);
    set_if("br", p.br);
    set_if("ke", p.ke);
    set_if("be", p.be);
    set_if("c_space_survival", p.c_space_survival);
    set_if("c_space_reproduction", p.c_space_reproduction);
    set_if("c_space_establishment", p.c_space_establishment);
    set_if("k_crowd_survival", p.k_crowd_survival);
    set_if("k_crowd_reproduction", p.k_crowd_reproduction);
    set_if("k_crowd_establishment", p.k_crowd_establishment);
    set_if("leaf_offset", p.leaf_offset);
    set_if("base_establishment", p.base_establishment);
}

enum class OutputMode : int { FULL = 0, SUMMARY = 1 };

void input(int& sim_time, int& num_sims, std::string& outdir, unsigned long int& num_threads, OutputMode& mode) {
    std::cout << "Enter Simulation time in Years: ";
    std::cin >> sim_time;
    std::cout << "Enter Number of Simulations: ";
    std::cin >> num_sims;
    std::cout << "Enter output directory: ";
    std::cin >> outdir;
    std::filesystem::create_directories(outdir);
    std::cout << "Enter the number of threads: ";
    std::cin >> num_threads;
    int m = 1;
    std::cout << "Enter output mode (0=full CSV, 1=summary only): ";
    std::cin >> m;
    mode = (m == 0) ? OutputMode::FULL : OutputMode::SUMMARY;
}

static inline double leaf_ipm_next_mean(double A, double leaf_offset) {
    const double b0 = 34.56271744473715;
    const double b1 = 1.043331450132405;
    const double b2 = -0.00030329319726520824;
    return (b0 + b1 * A + b2 * A * A) + leaf_offset;
}

static inline double baseline_survival_ipm(const Tiller& t) {
    const double s0 = -0.4759392738089253;
    const double s1 = 0.010061994950699203;
    return clamp01(logistic(s0 + s1 * std::max(0.0, (double)t.getLeafArea())));
}

static inline double apply_spatial_modifier(const std::string& form, double a, double b, double r0) {
    std::string lf = form;
    std::transform(lf.begin(), lf.end(), lf.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    
    if (lf == "logit") {
        return clamp01(logistic(a + b * r0));
    } else if (lf == "inverse") {
        return clamp01(a + b / (r0 + 1.0));
    } else if (lf == "exponential" || lf == "exp") {
        return clamp01(std::exp(a - b * r0));
    }
    // default linear
    return clamp01(a - b * r0);
}

static inline double spatial_survival_modifier(const Tiller& t, const ModelParams& p, const std::string& form) {
    return apply_spatial_modifier(form, p.bs, p.ks, calculater0(t));
}

static inline double baseline_reproduction_ipm(const Tiller& t) {
    double A = clamp((double)t.getLeafArea(), 0.0, 2000.0);
    const double b0 = -4.23966202048902;
    const double b1 = 0.01478880984282271;
    const double b2 = -1.334715680346952e-05;
    return clamp01(logistic(b0 + b1 * A + b2 * A * A));
}

static inline double spatial_reproduction_modifier(const Tiller& t, const ModelParams& p, const std::string& form) {
    return apply_spatial_modifier(form, p.br, p.kr, calculater0(t));
}

static inline double spatial_establishment_modifier(const Tiller& daughter, const ModelParams& p, const std::string& form) {
    return apply_spatial_modifier(form, p.be, p.ke, calculater0(daughter));
}

static inline double apply_blend(double base_p, double mod_p, double weight) {
    return clamp01((1.0 - clamp(weight, 0.0, 1.0)) * base_p + clamp(weight, 0.0, 1.0) * mod_p);
}

static inline double apply_crowding_penalty(double p, int nnbr, double k_crowd) {
    if (k_crowd <= 0.0 || nnbr <= 0) return clamp01(p);
    double denom = 1.0 + k_crowd * (double)nnbr;
    if (!std::isfinite(denom) || denom <= 0.0) denom = 1.0;
    return clamp01(p / denom);
}

static MechanismConfig read_mechanisms(const std::string& combined_ini_path) {
    MechanismConfig cfg;
    cfg.use_spatial_survival = ini_get_bool(combined_ini_path, "Mechanisms", "use_spatial_survival", false);
    cfg.use_spatial_reproduction = ini_get_bool(combined_ini_path, "Mechanisms", "use_spatial_reproduction", false);
    cfg.use_spatial_establishment = ini_get_bool(combined_ini_path, "Mechanisms", "use_spatial_establishment", false);
    cfg.use_crowding_survival = ini_get_bool(combined_ini_path, "Mechanisms", "use_crowding_survival", false);
    cfg.use_crowding_reproduction = ini_get_bool(combined_ini_path, "Mechanisms", "use_crowding_reproduction", false);
    cfg.use_crowding_establishment = ini_get_bool(combined_ini_path, "Mechanisms", "use_crowding_establishment", false);
    cfg.crowding_radius_cm = std::stod(ini_get(combined_ini_path, "Mechanisms", "crowding_radius_cm", "2.0"));
    cfg.spatial_survival_form = trim(ini_get(combined_ini_path, "Mechanisms", "spatial_survival_form", "linear"));
    cfg.spatial_reproduction_form = trim(ini_get(combined_ini_path, "Mechanisms", "spatial_reproduction_form", "linear"));
    cfg.spatial_establishment_form = trim(ini_get(combined_ini_path, "Mechanisms", "spatial_establishment_form", "linear"));
    return cfg;
}

void simulate(const int max_sim_time,
              const int sim_id,
              const std::uint64_t base_seed,
              const std::string& outdir,
              const std::string& combined_ini_path,
              const std::string& param_file_path,
              const OutputMode mode,
              const int constraint_year,
              const int alive_overflow_threshold) {
    ModelParams params;
    readFromFile(param_file_path, params);
    MechanismConfig cfg = read_mechanisms(combined_ini_path);

    const std::uint64_t seed64 = splitmix64(
        base_seed + static_cast<std::uint64_t>(sim_id)
    );
    std::seed_seq seed_words{
        static_cast<std::uint32_t>(seed64),
        static_cast<std::uint32_t>(seed64 >> 32),
        static_cast<std::uint32_t>(sim_id),
        static_cast<std::uint32_t>(base_seed),
        static_cast<std::uint32_t>(base_seed >> 32)
    };
    std::mt19937 gen(seed_words);
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    std::uniform_int_distribution<int> root_num_dis(1, 4);
    std::uniform_real_distribution<double> root_diam_dis(0.2, 1.5);
    std::normal_distribution<double> leafNoise(0.0, 147.74558691423508);

    std::filesystem::create_directories(outdir);
    std::string summary_dir = outdir + "/summaries";
    std::filesystem::create_directories(summary_dir);
    std::ofstream summary(summary_dir + "/summary_" + std::to_string(sim_id) + ".csv", std::ios::trunc);
    summary << "sim_id,final_t,final_diameter,alive_y,rmax_y,overflow_t,extinct_t,missing_year,alive_final,LeafArea,attempted_daughters,established_births,cumulative_deaths,cumulative_tillers_created,mean_reproduction_probability,mean_establishment_probability\n";

    // Yearly summaries are written in both SUMMARY and FULL mode so that the
    // Python parameterization script can apply time-series biological priors
    // without writing the full per-tiller CSV during optimization.
    std::string yearly_dir = outdir + "/yearly_summaries";
    std::filesystem::create_directories(yearly_dir);
    std::ofstream yearly(yearly_dir + "/yearly_summary_" + std::to_string(sim_id) + ".csv", std::ios::trunc);
    yearly << "sim_id,time_step,n_total,n_alive,n_dead,attempted_daughters,established_daughters,mean_reproduction_probability,mean_establishment_probability,cumulative_attempted_daughters,cumulative_established_births,cumulative_deaths,cumulative_tillers_created,diameter,radius,leaf_area_mean,overflow\n";

    std::ofstream outputFile, simlog;
    std::vector<char> filebuf;
    if (mode == OutputMode::FULL) {
        outputFile.open(outdir + "/tiller_data_sim_num_" + std::to_string(sim_id) + ".csv", std::ios::trunc);
        filebuf.resize(8 * 1024 * 1024);
        outputFile.rdbuf()->pubsetbuf(filebuf.data(), (std::streamsize)filebuf.size());
        outputFile << "TimeStep,TillerID,ParentTillerID,Age,ReferenceFootprintRadius,EffectiveFootprintRadius,LeafArea,DeadLeafArea,DeadLeafMass,RootNecroVol,RootNecroVolCum,RootNecroMass,RootNecroMassCum,X,Y,Z,NumRoots,RootDiamMM,Status\n";
        std::filesystem::create_directories(outdir + "/sim_logs");
        simlog.open(outdir + "/sim_logs/sim_" + std::to_string(sim_id) + ".log", std::ios::trunc);
        simlog << "TimeStep\tN_total\tN_alive\tN_dead\tAttemptedDaughters\tEstablishedDaughters\tMeanPRepro\tMeanPEst\tCumulativeAttemptedDaughters\tCumulativeEstablishedBirths\tCumulativeDeaths\tCumulativeTillersCreated\tDiameter\tOverlap_passes\tCandidates\tOverlapped\tZ_adjusts\tMaxPen\tOverlap_ms\n";
    }

    int next_tiller_id = 1;
    const double founder_reference_radius =
        Tiller::sampleReferenceFootprintRadius(gen);
    Tiller first_tiller(
        1,
        founder_reference_radius,
        0.0,
        0.0,
        0.0,
        3,
        true,
        50.0f,
        0.0f,
        0.0f,
        0.0f,
        1.0f,
        next_tiller_id++,
        -1
    );
    std::vector<Tiller> previous_step;
    previous_step.reserve(1024);
    previous_step.push_back(first_tiller);

    SimSummary ss;
    ss.sim_id = sim_id;
    int final_t = -1;
    const double LEAFAREA_MIN = 0.0;
    const double LEAFAREA_MAX = 2500.0;

    long long cumulative_attempted_daughters = 0;
    long long cumulative_established_births = 0;
    long long cumulative_deaths = 0;
    long long cumulative_tillers_created = 1; // includes the founding tiller
    double cumulative_repro_probability_sum = 0.0;
    long long cumulative_repro_probability_n = 0;
    double cumulative_est_probability_sum = 0.0;
    long long cumulative_est_probability_n = 0;

    for (int time_step = 0; time_step <= max_sim_time; ++time_step) {
        final_t = time_step;
        std::vector<int> crowd_prev = compute_local_crowding_alive(previous_step, cfg.crowding_radius_cm);
        std::vector<Tiller> step_data;
        std::vector<Tiller> newTillers;
        step_data.reserve(previous_step.size() + 256);
        newTillers.reserve(256);

        int attempted_daughters = 0;
        int established_daughters = 0;
        int deaths_this_step = 0;
        double repro_probability_sum = 0.0;
        int repro_probability_n = 0;
        double establishment_probability_sum = 0.0;
        int establishment_probability_n = 0;

        for (int idx = 0; idx < static_cast<int>(previous_step.size()); ++idx) {
            Tiller& tiller = previous_step[idx];
            int local_crowding = (idx >= 0 && idx < static_cast<int>(crowd_prev.size())) ? crowd_prev[idx] : 0;

            if (tiller.getStatus() == 1) {
                double prev_area = std::max(0.0, (double)tiller.getLeafArea());
                int prev_roots = tiller.getNumRoots();
                float prev_root_diam_mm = tiller.getRootDiamMM();

                double p_survive = baseline_survival_ipm(tiller);
                if (cfg.use_spatial_survival) {
                    p_survive = apply_blend(p_survive, spatial_survival_modifier(tiller, params, cfg.spatial_survival_form), params.c_space_survival);
                }
                if (cfg.use_crowding_survival) {
                    p_survive = apply_crowding_penalty(p_survive, local_crowding, params.k_crowd_survival);
                }

                if (dis(gen) < p_survive) {
                    tiller.accumulateDeadLeafArea((float)prev_area);
                    tiller.accumulateRootNecroFromPrevRoots(prev_roots, prev_root_diam_mm);

                    // Update growth first, then evaluate reproduction from the current year's size.
                    tiller.mature(1);
                    double Aclamp = clamp(prev_area, 0.0, 2000.0);
                    double A_next = leaf_ipm_next_mean(Aclamp, params.leaf_offset) + leafNoise(gen);
                    if (!std::isfinite(A_next)) A_next = 0.0;
                    tiller.setLeafArea((float)clamp(A_next, LEAFAREA_MIN, LEAFAREA_MAX));
                    tiller.setRoots(root_num_dis(gen), (float)root_diam_dis(gen));

                    double p_repro = 0.0;
                    if (!DISABLE_REPRO) {
                        p_repro = baseline_reproduction_ipm(tiller);
                        if (cfg.use_spatial_reproduction) {
                            p_repro = apply_blend(p_repro, spatial_reproduction_modifier(tiller, params, cfg.spatial_reproduction_form), params.c_space_reproduction);
                        }
                        if (cfg.use_crowding_reproduction) {
                            p_repro = apply_crowding_penalty(p_repro, local_crowding, params.k_crowd_reproduction);
                        }
                        repro_probability_sum += p_repro;
                        repro_probability_n++;
                    }

                    if (!DISABLE_REPRO && dis(gen) < p_repro) {
                        attempted_daughters++;
                        cumulative_attempted_daughters++;

                        Tiller daughter = tiller.makeDaughter(next_tiller_id++, gen);
                        cumulative_tillers_created++;

                        double p_est = params.base_establishment;
                        if (cfg.use_spatial_establishment) {
                            p_est = apply_blend(p_est,
                                                spatial_establishment_modifier(daughter, params, cfg.spatial_establishment_form),
                                                params.c_space_establishment);
                        }
                        if (cfg.use_crowding_establishment) {
                            // Establishment crowding is evaluated at the daughter's coordinates.
                            // The parent is excluded while the candidate daughter is age 1 or 2.
                            const int excluded_parent_id =
                                (daughter.getAge() <= 2) ? daughter.getParentTillerId() : -1;
                            int daughter_crowding = count_alive_neighbors_at_position(
                                previous_step,
                                daughter.getX(),
                                daughter.getY(),
                                cfg.crowding_radius_cm,
                                excluded_parent_id);
                            daughter_crowding += count_alive_neighbors_at_position(
                                newTillers,
                                daughter.getX(),
                                daughter.getY(),
                                cfg.crowding_radius_cm,
                                -1);
                            p_est = apply_crowding_penalty(p_est, daughter_crowding, params.k_crowd_establishment);
                        }

                        establishment_probability_sum += p_est;
                        establishment_probability_n++;

                        if (dis(gen) < p_est) {
                            newTillers.push_back(daughter);
                            established_daughters++;
                            cumulative_established_births++;
                        }
                    }
                } else {
                    tiller.accumulateDeadLeafArea((float)prev_area);
                    tiller.accumulateRootNecroFromPrevRoots(prev_roots, prev_root_diam_mm);
                    tiller.setStatus(0);
                    deaths_this_step++;
                    cumulative_deaths++;
                }

                step_data.push_back(tiller);
            } else {
                tiller.decay();
                tiller.setRoots(0, tiller.getRootDiamMM());
                if (!should_prune_dead(tiller)) step_data.push_back(tiller);
            }
        }

        const double mean_p_repro =
            (repro_probability_n > 0) ? repro_probability_sum / (double)repro_probability_n : std::nan("");
        const double mean_p_est =
            (establishment_probability_n > 0) ? establishment_probability_sum / (double)establishment_probability_n : std::nan("");
        cumulative_repro_probability_sum += repro_probability_sum;
        cumulative_repro_probability_n += repro_probability_n;
        cumulative_est_probability_sum += establishment_probability_sum;
        cumulative_est_probability_n += establishment_probability_n;

        step_data.insert(step_data.end(), newTillers.begin(), newTillers.end());

        int n_total = (int)step_data.size();
        int n_alive = 0;
        for (const auto& tt : step_data) n_alive += (tt.getStatus() == 1);
        int n_dead = n_total - n_alive;

        if (ss.extinct_t < 0 && n_alive == 0) ss.extinct_t = time_step;
        if (ss.overflow_t < 0 && n_alive > alive_overflow_threshold) ss.overflow_t = time_step;

        bool stop_due_to_overflow = false;
        if (n_alive > alive_overflow_threshold) {
            stop_due_to_overflow = true;
        }

        step_data.erase(std::remove_if(step_data.begin(), step_data.end(),
                                       [](const Tiller& t) { return t.getEffectiveFootprintRadius() <= 1e-6; }),
                        step_data.end());

        OverlapStats ostats;
        resolveOverlaps(step_data, ostats);

        // Tussock-level yearly summary used by the Python loss function.
        // Diameter includes living and retained dead tillers, uses each
        // tiller's effective footprint edges on both axes, and trims the
        // outermost 2.5% on each side.
        const double yearly_diam = calculate_robust_tussock_diameter(step_data);
        const double yearly_radius = 0.5 * yearly_diam;
        double yearly_leaf_sum = 0.0;
        int yearly_leaf_n = 0;
        for (const auto& tt : step_data) {
            if (tt.getStatus() == 1) {
                const double la = tt.getLeafArea();
                if (std::isfinite(la)) {
                    yearly_leaf_sum += la;
                    yearly_leaf_n++;
                }
            }
        }
        double yearly_leaf_mean = (yearly_leaf_n > 0) ? (yearly_leaf_sum / (double)yearly_leaf_n) : std::nan("");
        int overflow_now = (ss.overflow_t >= 0) ? 1 : 0;
        yearly << sim_id << ',' << time_step << ',' << n_total << ',' << n_alive << ',' << n_dead << ','
               << attempted_daughters << ',' << established_daughters << ',';
        if (std::isfinite(mean_p_repro)) yearly << mean_p_repro;
        yearly << ',';
        if (std::isfinite(mean_p_est)) yearly << mean_p_est;
        yearly << ',' << cumulative_attempted_daughters << ',' << cumulative_established_births << ','
               << cumulative_deaths << ',' << cumulative_tillers_created << ',' << yearly_diam << ','
               << yearly_radius << ',';
        if (std::isfinite(yearly_leaf_mean)) yearly << yearly_leaf_mean;
        yearly << ',' << overflow_now << "\n";

        if (time_step == constraint_year) {
            ss.missing_year = 0;
            ss.alive_y = n_alive;
            double rmax = 0.0, leaf_sum = 0.0;
            int leaf_n = 0;
            for (const auto& tt : step_data) {
                if (tt.getStatus() == 1) {
                    rmax = std::max(rmax, tt.getEffectiveFootprintRadius());
                    double la = tt.getLeafArea();
                    if (std::isfinite(la)) {
                        leaf_sum += la;
                        leaf_n++;
                    }
                }
            }
            ss.rmax_y = rmax;
            ss.leafarea_mean_y = (leaf_n > 0) ? (leaf_sum / (double)leaf_n) : std::nan("");
        }

        if (mode == OutputMode::FULL) {
            const double diam = calculate_robust_tussock_diameter(step_data);

            simlog << time_step << "\t" << n_total << "\t" << n_alive << "\t" << n_dead << "\t"
                   << attempted_daughters << "\t" << established_daughters << "\t";
            if (std::isfinite(mean_p_repro)) simlog << mean_p_repro;
            simlog << "\t";
            if (std::isfinite(mean_p_est)) simlog << mean_p_est;
            simlog << "\t" << cumulative_attempted_daughters << "\t" << cumulative_established_births
                   << "\t" << cumulative_deaths << "\t" << cumulative_tillers_created << "\t" << diam
                   << "\t" << ostats.passes << "\t" << ostats.candidates << "\t" << ostats.overlapped
                   << "\t" << ostats.z_adjusts << "\t" << ostats.max_penetration << "\t" << ostats.ms << "\n";

            for (const Tiller& data : step_data) {
                outputFile << time_step << ',' << data.getTillerId() << ',' << data.getParentTillerId()
                           << ',' << data.getAge() << ',' << data.getReferenceFootprintRadius()
                           << ',' << data.getEffectiveFootprintRadius() << ',' << data.getLeafArea()
                           << ',' << data.getDeadLeafArea() << ',' << data.getDeadLeafMass() << ','
                           << data.getRootNecroVol() << ',' << data.getRootNecroVolCum() << ','
                           << data.getRootNecroMass() << ',' << data.getRootNecroMassCum() << ','
                           << data.getX() << ',' << data.getY() << ',' << data.getZ() << ','
                           << data.getNumRoots() << ',' << data.getRootDiamMM() << ',' << data.getStatus()
                           << '\n';
            }
        }

        previous_step = std::move(step_data);
        if (stop_due_to_overflow) break;
    }

    ss.final_t = final_t;
    int alive_final = 0;
    for (const auto& tt : previous_step) alive_final += (tt.getStatus() == 1);
    ss.alive_final = alive_final;

    const double final_diam = calculate_robust_tussock_diameter(previous_step);
    ss.final_diameter = final_diam;
    ss.cumulative_attempted_daughters = cumulative_attempted_daughters;
    ss.cumulative_established_births = cumulative_established_births;
    ss.cumulative_deaths = cumulative_deaths;
    ss.cumulative_tillers_created = cumulative_tillers_created;
    ss.mean_reproduction_probability =
        (cumulative_repro_probability_n > 0)
            ? cumulative_repro_probability_sum / (double)cumulative_repro_probability_n
            : std::nan("");
    ss.mean_establishment_probability =
        (cumulative_est_probability_n > 0)
            ? cumulative_est_probability_sum / (double)cumulative_est_probability_n
            : std::nan("");

    summary << ss.sim_id << ',' << ss.final_t << ',' << ss.final_diameter << ',' << ss.alive_y << ','
            << ss.rmax_y << ',' << ss.overflow_t << ',' << ss.extinct_t << ',' << ss.missing_year
            << ',' << ss.alive_final << ',';
    if (std::isfinite(ss.leafarea_mean_y)) summary << ss.leafarea_mean_y;
    summary << ',' << ss.cumulative_attempted_daughters << ',' << ss.cumulative_established_births
            << ',' << ss.cumulative_deaths << ',' << ss.cumulative_tillers_created << ',';
    if (std::isfinite(ss.mean_reproduction_probability)) summary << ss.mean_reproduction_probability;
    summary << ',';
    if (std::isfinite(ss.mean_establishment_probability)) summary << ss.mean_establishment_probability;
    summary << "\n";
}

static std::string get_arg_value(int argc, char** argv, const std::string& name) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == name && i + 1 < argc) return argv[i + 1];
        const std::string prefix = name + "=";
        if (a.rfind(prefix, 0) == 0) return a.substr(prefix.size());
    }
    return "";
}

int main(int argc, char** argv) {
    std::string config_arg = get_arg_value(argc, argv, "--config");
    if (config_arg.empty()) {
        std::cerr << "Usage: tussock_model --config path/to/config.ini\n";
        return 1;
    }

    RuntimeConfig runtime = load_runtime_config(config_arg);
    const std::uint64_t base_seed = read_base_seed();

    if (!std::filesystem::exists(runtime.config_path)) {
        std::cerr << "Config file does not exist: " << runtime.config_path << "\n";
        return 1;
    }

    int max_sim_time, num_sims;
    std::string outdir;
    unsigned long int num_threads;
    OutputMode mode = OutputMode::SUMMARY;

    input(max_sim_time, num_sims, outdir, num_threads, mode);

    std::filesystem::path outdir_path = std::filesystem::path(outdir);
    if (outdir_path.is_relative()) outdir_path = runtime.project_root / outdir_path;
    outdir = outdir_path.string();

    std::vector<std::thread> threads;
    threads.reserve((size_t)num_threads);

    for (int sim_id = 0; sim_id < num_sims; ++sim_id) {
        threads.emplace_back(simulate,
                             max_sim_time,
                             sim_id,
                             base_seed,
                             outdir,
                             runtime.config_path.string(),
                             runtime.param_file_path.string(),
                             mode,
                             runtime.constraint_year,
                             runtime.alive_overflow_threshold);

        if ((threads.size() == num_threads) || (sim_id == num_sims - 1)) {
            for (auto& thread : threads) thread.join();
            threads.clear();
        }
    }

    return 0;
}