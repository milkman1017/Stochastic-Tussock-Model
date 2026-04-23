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
#include <unordered_map>
#include <vector>
#if defined(__linux__)
#include <unistd.h>
#endif

#include "tussock_model.h"

static std::mutex g_print_mutex;
static constexpr int DISABLE_REPRO = 0;

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

std::string ini_get(const std::string& ini_path, const std::string& wanted_section, const std::string& wanted_key, const std::string& fallback) {
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
        if (current_section == wanted_section && key == wanted_key) return val.empty() ? fallback : val;
    }
    return fallback;
}

double calculater0(const Tiller& tiller) { return std::sqrt(tiller.getX() * tiller.getX() + tiller.getY() * tiller.getY()); }

static inline double dist2_xy(const Tiller& a, const Tiller& b) {
    double dx = a.getX() - b.getX();
    double dy = a.getY() - b.getY();
    return dx * dx + dy * dy;
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
};

struct ModelParams {
    double ks = 0.0;
    double kr = 0.0;
    double bs = 0.0;
    double br = 0.0;
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
};

static inline bool should_prune_dead(const Tiller& t) {
    constexpr double EPS_R = 1e-6;
    constexpr double EPS_A = 1e-6;
    constexpr double EPS_M = 1e-8;
    if (t.getStatus() == 1) return false;
    const double r = t.getRadius();
    const double la = t.getLeafArea();
    const double dla = t.getDeadLeafArea();
    const double dlm = t.getDeadLeafMass();
    const double rnv = t.getRootNecroVol();
    const double rnvc = t.getRootNecroVolCum();
    const double rnm = t.getRootNecroMass();
    const double rnmc = t.getRootNecroMassCum();
    return (std::abs(r) <= EPS_R) && (std::abs(la) <= EPS_A) && (std::abs(dla) <= EPS_M) &&
           (std::abs(dlm) <= EPS_M) && (std::abs(rnv) <= EPS_M) && (std::abs(rnm) <= EPS_M) &&
           (std::abs(rnvc) <= EPS_M) && (std::abs(rnmc) <= EPS_M);
}

void resolveOverlaps(std::vector<Tiller>& tillers, OverlapStats& stats) {
    auto t0 = std::chrono::high_resolution_clock::now();
    stats = OverlapStats{};
    stats.max_penetration = 0.0;
    const double CUTOFF = 3.5;
    const double CUTOFF2 = CUTOFF * CUTOFF;
    const double EPS = 1e-6;
    const double CELL = CUTOFF;
    const double DAMP = 0.7;
    const int MAX_PASSES = 80;
    if (tillers.size() < 2) {
        auto t1 = std::chrono::high_resolution_clock::now();
        stats.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return;
    }
    std::unordered_map<std::int64_t, std::vector<int>> grid;
    grid.reserve(tillers.size() * 2);
    auto cell_of = [&](double x, double y) -> std::pair<int, int> {
        return {static_cast<int>(std::floor(x / CELL)), static_cast<int>(std::floor(y / CELL))};
    };
    auto rebuild_grid = [&]() {
        grid.clear();
        for (int i = 0; i < static_cast<int>(tillers.size()); ++i) {
            auto [cx, cy] = cell_of(tillers[i].getX(), tillers[i].getY());
            grid[cell_key(cx, cy)].push_back(i);
        }
    };
    for (int pass = 0; pass < MAX_PASSES; ++pass) {
        rebuild_grid();
        bool any_overlap = false;
        for (int i = 0; i < static_cast<int>(tillers.size()); ++i) {
            auto [cx, cy] = cell_of(tillers[i].getX(), tillers[i].getY());
            for (int gx = -1; gx <= 1; ++gx) {
                for (int gy = -1; gy <= 1; ++gy) {
                    auto it = grid.find(cell_key(cx + gx, cy + gy));
                    if (it == grid.end()) continue;
                    for (int j : it->second) {
                        if (j <= i) continue;
                        stats.candidates++;
                        double dx = tillers[j].getX() - tillers[i].getX();
                        double dy = tillers[j].getY() - tillers[i].getY();
                        double d2 = dx * dx + dy * dy;
                        if (d2 >= CUTOFF2) continue;
                        double rsum = tillers[i].getRadius() + tillers[j].getRadius();
                        if (d2 >= rsum * rsum) continue;
                        if (!tillers[i].isOverlapping(tillers[j])) continue;
                        any_overlap = true;
                        stats.overlapped++;
                        double angle = std::atan2(dy, dx);
                        double dist = std::sqrt(std::max(d2, EPS));
                        double pen = dist - rsum;
                        stats.max_penetration = std::min(stats.max_penetration, pen);
                        tillers[i].move(angle, DAMP * pen);
                        tillers[j].move(angle + M_PI, DAMP * pen);
                    }
                }
            }
        }
        stats.passes = pass + 1;
        if (!any_overlap) break;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    stats.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
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
        try { kv[trim(line.substr(0, eq))] = std::stod(trim(line.substr(eq + 1))); } catch (...) {}
    }
    auto set_if = [&](const char* key, double& ref) { if (kv.count(key)) ref = kv[key]; };
    set_if("ks", p.ks); set_if("kr", p.kr); set_if("bs", p.bs); set_if("br", p.br);
    set_if("c_space_survival", p.c_space_survival); set_if("c_space_reproduction", p.c_space_reproduction); set_if("c_space_establishment", p.c_space_establishment);
    set_if("k_crowd_survival", p.k_crowd_survival); set_if("k_crowd_reproduction", p.k_crowd_reproduction); set_if("k_crowd_establishment", p.k_crowd_establishment);
    set_if("leaf_offset", p.leaf_offset); set_if("base_establishment", p.base_establishment);
}

enum class OutputMode : int { FULL = 0, SUMMARY = 1 };

void input(int& sim_time, int& num_sims, std::string& outdir, unsigned long int& num_threads, OutputMode& mode) {
    std::cout << "Enter Simulation time in Years: "; std::cin >> sim_time;
    std::cout << "Enter Number of Simulations: "; std::cin >> num_sims;
    std::cout << "Enter output (relative) directory: "; std::cin >> outdir; std::filesystem::create_directories(outdir);
    std::cout << "Enter the number of threads: "; std::cin >> num_threads;
    int m = 1; std::cout << "Enter output mode (0=full CSV, 1=summary only): "; std::cin >> m;
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

static inline double spatial_survival_modifier(const Tiller& t, const ModelParams& p) {
    return clamp01(logistic(p.bs - p.ks * calculater0(t)));
}

static inline double baseline_reproduction_ipm(const Tiller& t) {
    double A = clamp((double)t.getLeafArea(), 0.0, 2000.0);
    const double b0 = -4.23966202048902;
    const double b1 = 0.01478880984282271;
    const double b2 = -1.334715680346952e-05;
    return clamp01(logistic(b0 + b1 * A + b2 * A * A));
}

static inline double spatial_reproduction_modifier(const Tiller& t, const ModelParams& p) {
    return clamp01(logistic(p.br - p.kr * calculater0(t)));
}

static inline double baseline_establishment_ipm(const ModelParams& p) {
    return clamp01(p.base_establishment);
}

static inline double spatial_establishment_modifier(const Tiller& daughter, const ModelParams& p) {
    return clamp01(logistic(p.br - p.kr * calculater0(daughter)));
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
    auto getb = [&](const std::string& key, bool fallback) {
        std::string v = ini_get(combined_ini_path, "Mechanisms", key, fallback ? "true" : "false");
        std::string lv = trim(v);
        std::transform(lv.begin(), lv.end(), lv.begin(), [](unsigned char c) { return (char)std::tolower(c); });
        return (lv == "1" || lv == "true" || lv == "yes" || lv == "on");
    };
    cfg.use_spatial_survival = getb("use_spatial_survival", false);
    cfg.use_spatial_reproduction = getb("use_spatial_reproduction", false);
    cfg.use_spatial_establishment = getb("use_spatial_establishment", false);
    cfg.use_crowding_survival = getb("use_crowding_survival", false);
    cfg.use_crowding_reproduction = getb("use_crowding_reproduction", false);
    cfg.use_crowding_establishment = getb("use_crowding_establishment", false);
    cfg.crowding_radius_cm = std::stod(ini_get(combined_ini_path, "Mechanisms", "crowding_radius_cm", "2.0"));
    return cfg;
}

void simulate(const int max_sim_time,
              const int sim_id,
              const std::string& outdir,
              const std::string& combined_ini_path,
              const std::string& param_file_path,
              const OutputMode mode,
              const int constraint_year,
              const int alive_overflow_threshold) {
    ModelParams params;
    readFromFile(param_file_path, params);
    MechanismConfig cfg = read_mechanisms(combined_ini_path);

    std::uint64_t t = (std::uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::uint32_t seed = (std::uint32_t)(t ^ (0x9e3779b97f4a7c15ULL + (std::uint64_t)sim_id * 0xBF58476D1CE4E5B9ULL));
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    std::normal_distribution<double> growRadiusDist(0.01, 0.0025);
    std::uniform_int_distribution<int> root_num_dis(1, 4);
    std::uniform_real_distribution<double> root_diam_dis(0.2, 1.5);
    std::normal_distribution<double> leafNoise(0.0, 147.74558691423508);

    std::filesystem::create_directories(outdir);
    std::string summary_dir = outdir + "/summaries";
    std::filesystem::create_directories(summary_dir);
    std::ofstream summary(summary_dir + "/summary_" + std::to_string(sim_id) + ".csv", std::ios::trunc);
    summary << "sim_id,final_t,final_diameter,alive_y,rmax_y,overflow_t,extinct_t,missing_year,alive_final,LeafArea\n";

    std::ofstream outputFile, simlog;
    std::vector<char> filebuf;
    if (mode == OutputMode::FULL) {
        outputFile.open(outdir + "/tiller_data_sim_num_" + std::to_string(sim_id) + ".csv", std::ios::trunc);
        filebuf.resize(8 * 1024 * 1024);
        outputFile.rdbuf()->pubsetbuf(filebuf.data(), (std::streamsize)filebuf.size());
        outputFile << "TimeStep,TillerID,ParentTillerID,Age,Radius,LeafArea,DeadLeafArea,DeadLeafMass,RootNecroVol,RootNecroVolCum,RootNecroMass,RootNecroMassCum,X,Y,Z,NumRoots,RootDiamMM,Status\n";
        std::filesystem::create_directories(outdir + "/sim_logs");
        simlog.open(outdir + "/sim_logs/sim_" + std::to_string(sim_id) + ".log", std::ios::trunc);
        simlog << "TimeStep\tN_total\tN_alive\tN_dead\tN_newborn\tDiameter\tOverlap_passes\tCandidates\tOverlapped\tZ_adjusts\tMaxPen\tOverlap_ms\n";
    }

    int next_tiller_id = 1;
    Tiller first_tiller(1, 0.1, 0.0, 0.0, 0.0, 3, 1, 50.0f, 0.0f, 0.0f, 0.0f, 1.0f, next_tiller_id++, -1);
    std::vector<Tiller> previous_step;
    previous_step.reserve(1024);
    previous_step.push_back(first_tiller);

    SimSummary ss;
    ss.sim_id = sim_id;
    int final_t = -1;
    const double LEAFAREA_MIN = 0.0;
    const double LEAFAREA_MAX = 2500.0;
    const double SENTINEL_SCALE = 2000.0;

    for (int time_step = 0; time_step <= max_sim_time; ++time_step) {
        final_t = time_step;
        std::vector<int> crowd_prev = compute_local_crowding_alive(previous_step, cfg.crowding_radius_cm);
        std::vector<Tiller> step_data;
        std::vector<Tiller> newTillers;
        step_data.reserve(previous_step.size() + 256);
        newTillers.reserve(256);

        for (int idx = 0; idx < static_cast<int>(previous_step.size()); ++idx) {
            Tiller& tiller = previous_step[idx];
            int local_crowding = (idx >= 0 && idx < static_cast<int>(crowd_prev.size())) ? crowd_prev[idx] : 0;
            if (tiller.getStatus() == 1) {
                double current_area = std::max(0.0, (double)tiller.getLeafArea());
                double prev_area = current_area;
                int prev_roots = tiller.getNumRoots();
                float prev_root_diam_mm = tiller.getRootDiamMM();
                double p_survive = baseline_survival_ipm(tiller);
                if (cfg.use_spatial_survival) p_survive = apply_blend(p_survive, spatial_survival_modifier(tiller, params), params.c_space_survival);
                if (cfg.use_crowding_survival) p_survive = apply_crowding_penalty(p_survive, local_crowding, params.k_crowd_survival);
                if (dis(gen) < p_survive) {
                    tiller.accumulateDeadLeafArea((float)prev_area);
                    tiller.accumulateRootNecroFromPrevRoots(prev_roots, prev_root_diam_mm);
                    double p_repro = 0.0;
                    if (!DISABLE_REPRO) {
                        p_repro = baseline_reproduction_ipm(tiller);
                        if (cfg.use_spatial_reproduction) p_repro = apply_blend(p_repro, spatial_reproduction_modifier(tiller, params), params.c_space_reproduction);
                        if (cfg.use_crowding_reproduction) p_repro = apply_crowding_penalty(p_repro, local_crowding, params.k_crowd_reproduction);
                    }
                    if (!DISABLE_REPRO && (dis(gen) < p_repro)) {
                        Tiller daughter = tiller.makeDaughter(next_tiller_id++);
                        double p_est = baseline_establishment_ipm(params);
                        if (cfg.use_spatial_establishment) p_est = apply_blend(p_est, spatial_establishment_modifier(daughter, params), params.c_space_establishment);
                        if (cfg.use_crowding_establishment) p_est = apply_crowding_penalty(p_est, local_crowding, params.k_crowd_establishment);
                        if (dis(gen) < p_est) newTillers.push_back(daughter);
                    }
                    tiller.mature(1);
                    double Aclamp = clamp(current_area, 0.0, 2000.0);
                    double A_next = leaf_ipm_next_mean(Aclamp, params.leaf_offset) + leafNoise(gen);
                    if (!std::isfinite(A_next)) A_next = 0.0;
                    tiller.setLeafArea((float)clamp(A_next, LEAFAREA_MIN, LEAFAREA_MAX));
                    tiller.setRoots(root_num_dis(gen), (float)root_diam_dis(gen));
                    double dr_base = growRadiusDist(gen);
                    if (!std::isfinite(dr_base) || dr_base < 0.0) dr_base = 0.01;
                    tiller.growRadius(dr_base);
                } else {
                    tiller.accumulateDeadLeafArea((float)prev_area);
                    tiller.accumulateRootNecroFromPrevRoots(prev_roots, prev_root_diam_mm);
                    tiller.setStatus(0);
                }
                step_data.push_back(tiller);
            } else {
                tiller.decay();
                tiller.setRoots(0, tiller.getRootDiamMM());
                if (!should_prune_dead(tiller)) step_data.push_back(tiller);
            }
        }

        step_data.insert(step_data.end(), newTillers.begin(), newTillers.end());
        int n_total = (int)step_data.size();
        int n_alive = 0;
        for (const auto& tt : step_data) n_alive += (tt.getStatus() == 1);
        int n_dead = n_total - n_alive;
        if (ss.extinct_t < 0 && n_alive == 0) ss.extinct_t = time_step;
        if (ss.overflow_t < 0 && n_alive > alive_overflow_threshold) ss.overflow_t = time_step;
        bool stop_due_to_overflow = false;
        if (n_alive > alive_overflow_threshold) {
            double d = SENTINEL_SCALE / (double(time_step) + 1.0);
            d = std::max(50.0, std::min(5000.0, d));
            step_data.emplace_back(Tiller(1, 0.5, +d, 0.0, 0.0, 3, 1, 50.0f, 0.0f, 0.0f, 0.0f, 1.0f, next_tiller_id++, -1));
            step_data.emplace_back(Tiller(1, 0.5, -d, 0.0, 0.0, 3, 1, 50.0f, 0.0f, 0.0f, 0.0f, 1.0f, next_tiller_id++, -1));
            stop_due_to_overflow = true;
        }
        step_data.erase(std::remove_if(step_data.begin(), step_data.end(), [](const Tiller& t) { return t.getRadius() <= 1e-6; }), step_data.end());
        OverlapStats ostats;
        resolveOverlaps(step_data, ostats);

        if (time_step == constraint_year) {
            ss.missing_year = 0;
            ss.alive_y = n_alive;
            double rmax = 0.0, leaf_sum = 0.0; int leaf_n = 0;
            for (const auto& tt : step_data) if (tt.getStatus() == 1) {
                rmax = std::max(rmax, tt.getRadius());
                double la = tt.getLeafArea();
                if (std::isfinite(la)) { leaf_sum += la; leaf_n++; }
            }
            ss.rmax_y = rmax;
            ss.leafarea_mean_y = (leaf_n > 0) ? (leaf_sum / (double)leaf_n) : std::nan("");
        }

        if (mode == OutputMode::FULL) {
            double diam = 0.0;
            if (!step_data.empty()) {
                double xmin = step_data[0].getX(), xmax = step_data[0].getX();
                for (const auto& tt : step_data) { xmin = std::min(xmin, tt.getX()); xmax = std::max(xmax, tt.getX()); }
                diam = xmax - xmin;
            }
            simlog << time_step << "\t" << n_total << "\t" << n_alive << "\t" << n_dead << "\t" << (int)newTillers.size() << "\t" << diam << "\t" << ostats.passes << "\t" << ostats.candidates << "\t" << ostats.overlapped << "\t" << ostats.z_adjusts << "\t" << ostats.max_penetration << "\t" << ostats.ms << "\n";
            for (const Tiller& data : step_data) {
                outputFile << time_step << ',' << data.getTillerId() << ',' << data.getParentTillerId() << ',' << data.getAge() << ',' << data.getRadius() << ',' << data.getLeafArea() << ',' << data.getDeadLeafArea() << ',' << data.getDeadLeafMass() << ',' << data.getRootNecroVol() << ',' << data.getRootNecroVolCum() << ',' << data.getRootNecroMass() << ',' << data.getRootNecroMassCum() << ',' << data.getX() << ',' << data.getY() << ',' << data.getZ() << ',' << data.getNumRoots() << ',' << data.getRootDiamMM() << ',' << data.getStatus() << '\n';
            }
        }

        previous_step = std::move(step_data);
        if (stop_due_to_overflow) break;
    }

    ss.final_t = final_t;
    int alive_final = 0; for (const auto& tt : previous_step) alive_final += (tt.getStatus() == 1);
    ss.alive_final = alive_final;
    double final_diam = 0.0;
    if (!previous_step.empty()) {
        double xmin = previous_step[0].getX(), xmax = previous_step[0].getX();
        for (const auto& tt : previous_step) { xmin = std::min(xmin, tt.getX()); xmax = std::max(xmax, tt.getX()); }
        final_diam = xmax - xmin;
    }
    ss.final_diameter = final_diam;
    summary << ss.sim_id << ',' << ss.final_t << ',' << ss.final_diameter << ',' << ss.alive_y << ',' << ss.rmax_y << ',' << ss.overflow_t << ',' << ss.extinct_t << ',' << ss.missing_year << ',' << ss.alive_final << ',';
    if (std::isfinite(ss.leafarea_mean_y)) summary << ss.leafarea_mean_y;
    summary << "\n";
}

int main() {
    std::srand((unsigned)std::time(nullptr));
    int max_sim_time, num_sims; std::string outdir; unsigned long int num_threads; OutputMode mode = OutputMode::SUMMARY;
    const std::filesystem::path project_root = get_project_root();
    const std::filesystem::path runtime_ini = project_root / "parameterization.ini";
    std::string config_file_raw = ini_get(runtime_ini.string(), "Parameterization", "config_file", "");
    if (config_file_raw.empty()) {
        std::cerr << "Missing config_file in parameterization.ini\n";
        return 1;
    }
    std::filesystem::path combined_ini = std::filesystem::path(config_file_raw);
    if (combined_ini.is_relative()) combined_ini = project_root / combined_ini;
    std::string param_file_raw = ini_get(runtime_ini.string(), "Parameterization", "param_file", "parameters/parameters.txt");
    std::filesystem::path param_file_path = std::filesystem::path(param_file_raw);
    if (param_file_path.is_relative()) param_file_path = project_root / param_file_path;
    const int constraint_year = std::stoi(ini_get(runtime_ini.string(), "Parameterization", "constraint_year", "25"));
    const int alive_overflow_threshold = std::stoi(ini_get(runtime_ini.string(), "Parameterization", "alive_overflow_threshold", "600"));
    input(max_sim_time, num_sims, outdir, num_threads, mode);
    std::filesystem::path outdir_path = std::filesystem::path(outdir);
    if (outdir_path.is_relative()) outdir_path = project_root / outdir_path;
    outdir = outdir_path.string();
    std::vector<std::thread> threads;
    threads.reserve((size_t)num_threads);
    for (int sim_id = 0; sim_id < num_sims; ++sim_id) {
        threads.emplace_back(simulate, max_sim_time, sim_id, outdir, combined_ini.string(), param_file_path.string(), mode, constraint_year, alive_overflow_threshold);
        if ((threads.size() == num_threads) || (sim_id == num_sims - 1)) {
            for (auto& thread : threads) thread.join();
            threads.clear();
        }
    }
    return 0;
}
