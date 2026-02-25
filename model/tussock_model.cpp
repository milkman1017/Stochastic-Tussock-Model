// main.cpp  (build: g++ -O3 -DNDEBUG -march=native -flto -pthread -std=c++17 -o tussock_model main.cpp)

#include <iostream>
#include <fstream>
#include <vector>
#include "tussock_model.h"
#include <string>
#include <thread>
#include <filesystem>
#include <random>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <sstream>
#include <cctype>
#include <ctime>
#include <cstdlib>
#include <chrono>
#include <cstdint>
#include <mutex>

#if defined(__linux__)
#include <unistd.h>
#include <limits.h>
#endif

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

static inline bool should_prune_dead(const Tiller& t) {
    constexpr double EPS_R = 1e-6;
    constexpr double EPS_A = 1e-6;
    constexpr double EPS_M = 1e-8;

    if (t.getStatus() == 1) return false;

    const double r  = (double)t.getRadius();
    const double la = (double)t.getLeafArea();

    const double dla = (double)t.getDeadLeafArea();
    const double dlm = (double)t.getDeadLeafMass();
    const double rnv = (double)t.getRootNecroVol();
    const double rnvc = (double)t.getRootNecroVolCum();
    const double rnm = (double)t.getRootNecroMass();
    const double rnmc = (double)t.getRootNecroMassCum();

    bool empty =
        (std::abs(r)  <= EPS_R) &&
        (std::abs(la) <= EPS_A) &&
        (std::abs(dla) <= EPS_M) &&
        (std::abs(dlm) <= EPS_M) &&
        (std::abs(rnv) <= EPS_M) &&
        (std::abs(rnm) <= EPS_M) &&
        (std::abs(rnvc) <= EPS_M) &&
        (std::abs(rnmc) <= EPS_M);

    return empty;
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

std::string ini_get(
    const std::string& ini_path,
    const std::string& wanted_section,
    const std::string& wanted_key,
    const std::string& fallback
) {
    std::ifstream in(ini_path);
    if (!in.is_open()) return fallback;

    std::string line;
    std::string current_section;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty()) continue;
        if (line[0] == ';' || line[0] == '#') continue;

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

double calculater0(const Tiller& tiller) {
    return std::sqrt(tiller.getX() * tiller.getX() + tiller.getY() * tiller.getY());
}

static inline double dist2_xy(const Tiller& a, const Tiller& b) {
    double dx = a.getX() - b.getX();
    double dy = a.getY() - b.getY();
    return dx*dx + dy*dy;
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

static inline bool allow_rare_z_adjust(int i, int j, int pass) {
    std::uint64_t x = 1469598103934665603ULL;
    x ^= (std::uint64_t)i + 0x9e3779b97f4a7c15ULL; x *= 1099511628211ULL;
    x ^= (std::uint64_t)j + 0xbf58476d1ce4e5b9ULL; x *= 1099511628211ULL;
    x ^= (std::uint64_t)pass + 0x94d049bb133111ebULL; x *= 1099511628211ULL;
    return (x % 20ULL) == 0ULL;
}

void resolveOverlaps(std::vector<Tiller>& tillers, OverlapStats& stats) {
    auto t0 = std::chrono::high_resolution_clock::now();
    stats = OverlapStats{};
    stats.max_penetration = 0.0;

    const double CUTOFF = 3.5;
    const double CUTOFF2 = CUTOFF * CUTOFF;
    const double EPS    = 1e-6;

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

    std::vector<int> nbr(tillers.size(), 0);

    auto cell_of = [&](double x, double y) -> std::pair<int,int> {
        int cx = static_cast<int>(std::floor(x / CELL));
        int cy = static_cast<int>(std::floor(y / CELL));
        return {cx, cy};
    };

    auto rebuild_grid = [&]() {
        grid.clear();
        for (int i = 0; i < static_cast<int>(tillers.size()); ++i) {
            auto [cx, cy] = cell_of(tillers[i].getX(), tillers[i].getY());
            grid[cell_key(cx, cy)].push_back(i);
        }
    };

    auto compute_neighbor_counts = [&]() {
        std::fill(nbr.begin(), nbr.end(), 0);

        for (int i = 0; i < static_cast<int>(tillers.size()); ++i) {
            auto [cx, cy] = cell_of(tillers[i].getX(), tillers[i].getY());

            int count = 0;
            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    auto it = grid.find(cell_key(cx + dx, cy + dy));
                    if (it == grid.end()) continue;
                    const auto& bucket = it->second;

                    for (int j : bucket) {
                        if (j == i) continue;
                        double ddx = tillers[i].getX() - tillers[j].getX();
                        double ddy = tillers[i].getY() - tillers[j].getY();
                        double d2 = ddx*ddx + ddy*ddy;
                        if (d2 < CUTOFF2) count++;
                    }
                }
            }
            nbr[i] = count;
        }
    };

    for (int pass = 0; pass < MAX_PASSES; ++pass) {
        rebuild_grid();
        compute_neighbor_counts();

        double mean_nbrs = 0.0;
        for (int v : nbr) mean_nbrs += static_cast<double>(v);
        mean_nbrs /= std::max(1, static_cast<int>(nbr.size()));

        bool any_overlap = false;

        for (int i = 0; i < static_cast<int>(tillers.size()); ++i) {
            auto [cx, cy] = cell_of(tillers[i].getX(), tillers[i].getY());

            for (int gx = -1; gx <= 1; ++gx) {
                for (int gy = -1; gy <= 1; ++gy) {
                    auto it = grid.find(cell_key(cx + gx, cy + gy));
                    if (it == grid.end()) continue;
                    const auto& bucket = it->second;

                    for (int j : bucket) {
                        if (j <= i) continue;

                        stats.candidates++;

                        double dx = tillers[j].getX() - tillers[i].getX();
                        double dy = tillers[j].getY() - tillers[i].getY();
                        double d2 = dx*dx + dy*dy;
                        if (d2 >= CUTOFF2) continue;

                        double rsum = tillers[i].getRadius() + tillers[j].getRadius();
                        double rsum2 = rsum * rsum;
                        if (d2 >= rsum2) continue;

                        if (!tillers[i].isOverlapping(tillers[j])) continue;

                        any_overlap = true;
                        stats.overlapped++;

                        double angle = std::atan2(dy, dx);
                        double dist = std::sqrt(std::max(d2, EPS));
                        double pen = dist - rsum;
                        stats.max_penetration = std::min(stats.max_penetration, pen);

                        tillers[i].move(angle,        DAMP * pen);
                        tillers[j].move(angle + M_PI, DAMP * pen);

                        if (tillers[i].isOverlapping(tillers[j])) {
                            if (pass >= 5 && allow_rare_z_adjust(i, j, pass)) {
                                double dxy2 = dist2_xy(tillers[i], tillers[j]);
                                if (dxy2 < rsum2) {
                                    double dz_need = std::sqrt(std::max(0.0, rsum2 - dxy2)) + EPS;

                                    int ni = nbr[i];
                                    int nj = nbr[j];

                                    auto signed_dz = [&](int n) {
                                        return (static_cast<double>(n) < mean_nbrs) ? -dz_need : +dz_need;
                                    };

                                    if (ni <= nj) {
                                        tillers[i].addZ(signed_dz(ni));
                                        stats.z_adjusts++;
                                    } else {
                                        tillers[j].addZ(signed_dz(nj));
                                        stats.z_adjusts++;
                                    }
                                }
                            }
                        }
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

    auto cell_of = [&](double x, double y) -> std::pair<int,int> {
        int cx = static_cast<int>(std::floor(x / CELL));
        int cy = static_cast<int>(std::floor(y / CELL));
        return {cx, cy};
    };

    for (int i = 0; i < static_cast<int>(tillers.size()); ++i) {
        if (tillers[i].getStatus() != 1) continue;
        auto [cx, cy] = cell_of(tillers[i].getX(), tillers[i].getY());
        grid[cell_key(cx, cy)].push_back(i);
    }

    for (int i = 0; i < static_cast<int>(tillers.size()); ++i) {
        if (tillers[i].getStatus() != 1) { crowd[i] = 0; continue; }
        auto [cx, cy] = cell_of(tillers[i].getX(), tillers[i].getY());

        int count = 0;
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                auto it = grid.find(cell_key(cx + dx, cy + dy));
                if (it == grid.end()) continue;
                const auto& bucket = it->second;

                for (int j : bucket) {
                    if (j == i) continue;
                    double ddx = tillers[i].getX() - tillers[j].getX();
                    double ddy = tillers[i].getY() - tillers[j].getY();
                    double d2 = ddx*ddx + ddy*ddy;
                    if (d2 <= R2) count++;
                }
            }
        }
        crowd[i] = count;
    }

    return crowd;
}

void readFromFile(const std::string& filename,
                  double& ks, double& kr, double& bs, double& br,
                  double& c_space,
                  double& c_repro,
                  double& fr,
                  double& k_crowd,
                  double& leaf_offset) {
    std::ifstream inputFile(filename);
    if (!inputFile.is_open()) {
        std::lock_guard<std::mutex> lk(g_print_mutex);
        std::cerr << "Unable to open file: " << filename << "\n";
        std::cerr.flush();
        return;
    }

    std::unordered_map<std::string, double> p;
    std::string line;
    while (std::getline(inputFile, line)) {
        line = trim(line);
        if (line.empty()) continue;
        if (line[0] == '#' || line[0] == ';') continue;

        auto eq = line.find('=');
        if (eq == std::string::npos) continue;

        std::string key = trim(line.substr(0, eq));
        std::string val = trim(line.substr(eq + 1));

        try { p[key] = std::stod(val); } catch (...) {}
    }

    if (p.count("ks")) ks = p["ks"];
    if (p.count("kr")) kr = p["kr"];
    if (p.count("bs")) bs = p["bs"];
    if (p.count("br")) br = p["br"];

    if (p.count("c_space")) c_space = p["c_space"];
    if (p.count("c_repro")) c_repro = p["c_repro"];

    if (p.count("fr")) fr = p["fr"];
    if (p.count("k_crowd")) k_crowd = p["k_crowd"];

    if (p.count("leaf_offset")) leaf_offset = p["leaf_offset"];
}

enum class OutputMode : int { FULL = 0, SUMMARY = 1 };

void input(int &sim_time, int &num_sims, std::string &outdir, unsigned long int &num_threads, OutputMode& mode) {
    std::cout << "Enter Simulation time in Years: ";
    std::cin >> sim_time;

    std::cout << "Enter Number of Simulations: ";
    std::cin >> num_sims;

    std::cout << "Enter output (relative) directory: ";
    std::cin >> outdir;
    std::filesystem::create_directories(outdir);

    std::cout << "Enter the number of threads: ";
    std::cin >> num_threads;

    int m = 1;
    std::cout << "Enter output mode (0=full CSV, 1=summary only): ";
    std::cin >> m;
    mode = (m == 0) ? OutputMode::FULL : OutputMode::SUMMARY;
}

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

static inline double leaf_ipm_next_mean(double A, double leaf_offset) {
    // growth_mean: y = b0 + b1*A + b2*A^2 + offset
    const double b0 = 34.56271744473715;
    const double b1 = 1.043331450132405;
    const double b2 = -0.00030329319726520824;
    return (b0 + b1 * A + b2 * A * A) + leaf_offset;
}

void simulate(const int max_sim_time,
              const int sim_id,
              const std::string& outdir,
              const std::string& param_file_path,
              const OutputMode mode,
              const int constraint_year,
              const int alive_overflow_threshold) {

    double ks = 0.0, kr = 0.0, bs = 0.0, br = 0.0;
    double c_space = 1.0;
    double c_repro = 0.5;
    double fr = 0.5;
    double k_crowd = 0.0;
    double leaf_offset = 0.0;

    readFromFile(param_file_path, ks, kr, bs, br, c_space, c_repro, fr, k_crowd, leaf_offset);

    c_space = clamp(c_space, 0.0, 1.0);
    c_repro = clamp(c_repro, 0.0, 1.0);
    fr      = clamp(fr, 0.0, 1.0);

    if (!std::isfinite(k_crowd) || k_crowd < 0.0) k_crowd = 0.0;
    if (!std::isfinite(leaf_offset)) leaf_offset = 0.0;

    std::uint64_t t = (std::uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::uint32_t seed = (std::uint32_t)(t ^ (0x9e3779b97f4a7c15ULL + (std::uint64_t)sim_id * 0xBF58476D1CE4E5B9ULL));
    std::mt19937 gen(seed);

    std::uniform_real_distribution<double> dis(0.0, 1.0);
    std::normal_distribution<double> growRadiusDist(0.01, 0.0025);

    std::uniform_int_distribution<int> root_num_dis(1, 4);
    std::uniform_real_distribution<double> root_diam_dis(0.5, 5.0);

    // IPM growth noise (additive on y)
    const double sigma_leaf = 147.74558691423508;
    std::normal_distribution<double> leafNoise(0.0, sigma_leaf);

    std::filesystem::create_directories(outdir);
    std::string logs_dir = outdir + "/sim_logs";
    std::filesystem::create_directories(logs_dir);

    std::string summary_dir = outdir + "/summaries";
    std::filesystem::create_directories(summary_dir);
    std::string summary_name = summary_dir + "/summary_" + std::to_string(sim_id) + ".csv";
    std::ofstream summary(summary_name, std::ios::trunc);
    summary << "sim_id,final_t,final_diameter,alive_y,rmax_y,overflow_t,extinct_t,missing_year,alive_final,LeafArea\n";

    std::ofstream outputFile;
    std::vector<char> filebuf;
    if (mode == OutputMode::FULL) {
        std::string out_file_name = outdir + "/tiller_data_sim_num_" + std::to_string(sim_id) + ".csv";
        outputFile.open(out_file_name, std::ios::trunc);

        filebuf.resize(8 * 1024 * 1024);
        outputFile.rdbuf()->pubsetbuf(filebuf.data(), (std::streamsize)filebuf.size());

        outputFile << "TimeStep,Age,Radius,LeafArea,DeadLeafArea,DeadLeafMass,RootNecroVol,RootNecroVolCum,RootNecroMass,RootNecroMassCum,X,Y,Z,NumRoots,RootDiamMM,Status\n";
    }

    std::ofstream simlog;
    if (mode == OutputMode::FULL) {
        std::string simlog_name = logs_dir + "/sim_" + std::to_string(sim_id) + ".log";
        simlog.open(simlog_name, std::ios::trunc);
        simlog << "TimeStep\tN_total\tN_alive\tN_dead\tN_newborn\tDiameter"
                  "\tOverlap_passes\tCandidates\tOverlapped\tZ_adjusts\tMaxPen\tOverlap_ms\n";
    }

    Tiller first_tiller(
        1,
        0.1,
        0.0,
        0.0,
        0.0,
        3,
        1,
        50.0f,
        0.0f,
        0.0f,
        0.0f,
        1.0f,
        0.0
    );

    std::vector<Tiller> previous_step;
    previous_step.reserve(1024);
    previous_step.push_back(first_tiller);

    const double SENTINEL_SCALE = 2000.0;

    SimSummary ss;
    ss.sim_id = sim_id;

    int final_t = -1;

    static constexpr double LEAFAREA_MIN = 0.0;
    static constexpr double LEAFAREA_MAX = 2500.0;

    static constexpr double ASSIM_C_PER_G_LEAF = 2.0;
    static constexpr double CARBON_PER_G_BIOMASS = 0.45;

    static constexpr double RHO_STEM_G_PER_CM3 = 0.30;
    static constexpr double STEM_TNC_FRAC = 0.375;

    static constexpr double TDD_JJA = 500.0;
    static constexpr double STEM_INC_MM_PER_YR = (0.0052 * TDD_JJA - 0.5461);
    static constexpr double STEM_INC_CM_PER_YR = (STEM_INC_MM_PER_YR > 0.0 ? STEM_INC_MM_PER_YR / 10.0 : 0.0);

    static constexpr double R_CROWD_CM = 2.0;

    for (int time_step = 0; time_step <= max_sim_time; time_step++) {
        final_t = time_step;

        std::vector<int> crowd_prev = compute_local_crowding_alive(previous_step, R_CROWD_CM);

        std::vector<Tiller> step_data;
        std::vector<Tiller> newTillers;
        step_data.reserve(previous_step.size() + 256);
        newTillers.reserve(256);

        for (int idx = 0; idx < static_cast<int>(previous_step.size()); ++idx) {
            Tiller& tiller = previous_step[idx];

            if (tiller.getStatus() == 1) {
                double distance = calculater0(tiller);

                double current_area = static_cast<double>(tiller.getLeafArea());
                if (!std::isfinite(current_area) || current_area < 0.0) current_area = 0.0;

                const double A_t = clamp(current_area, LEAFAREA_MIN, LEAFAREA_MAX);
                double prev_area = current_area;

                const int   prev_roots = tiller.getNumRoots();
                const float prev_root_diam_mm = tiller.getRootDiamMM();

                double p_spatial = clamp01(logistic(bs - ks * distance));

                const double s0 = -0.4759392738089253;
                const double s1 =  0.010061994950699203;
                double eta_size = s0 + s1 * current_area;
                double p_size = clamp01(logistic(eta_size));

                double w_space = clamp(c_space, 0.0, 1.0);
                double p_survive = (1.0 - w_space) * p_size + w_space * p_spatial;
                p_survive = clamp01(p_survive);

                if (dis(gen) < p_survive) {

                    tiller.accumulateDeadLeafArea(static_cast<float>(prev_area));
                    tiller.accumulateRootNecroFromPrevRoots(prev_roots, prev_root_diam_mm);

                    double p_event = 0.0;
                    if (!DISABLE_REPRO) {
                        double eta_repro_spatial = br - kr * distance;
                        double p_repro_spatial   = clamp01(logistic(eta_repro_spatial));

                        const double b0_f = -4.23966202048902;
                        const double b1_f =  0.01478880984282271;
                        const double b2_f = -1.334715680346952e-05;

                        const double A_MIN_FEC = 0.0;
                        const double A_MAX_FEC = 2000.0;
                        double A = std::max(A_MIN_FEC, std::min(A_MAX_FEC, current_area));
                        double eta_f = b0_f + b1_f * A + b2_f * A * A;
                        double p_event_size = clamp01(logistic(eta_f));

                        double w_repro = clamp(c_repro, 0.0, 1.0);
                        p_event = clamp01(w_repro * p_repro_spatial + (1.0 - w_repro) * p_event_size);

                        int Nnbr = 0;
                        if (idx >= 0 && idx < static_cast<int>(crowd_prev.size())) Nnbr = crowd_prev[idx];
                        if (Nnbr < 0) Nnbr = 0;

                        if (k_crowd > 0.0 && Nnbr > 0) {
                            double denom = 1.0 + k_crowd * static_cast<double>(Nnbr);
                            if (!std::isfinite(denom) || denom <= 0.0) denom = 1.0;
                            p_event = clamp01(p_event / denom);
                        }
                    }

                    if (!DISABLE_REPRO && (dis(gen) < p_event)) {
                        newTillers.push_back(tiller.makeDaughter());
                    }

                    tiller.mature(1);

                    // Leaf area update from IPM growth function (A_{t+1})
                    {
                        double Aclamp = clamp(A_t, 0.0, 2000.0);
                        double mu = leaf_ipm_next_mean(Aclamp, leaf_offset);
                        double A_next = mu + leafNoise(gen);
                        if (!std::isfinite(A_next)) A_next = 0.0;
                        A_next = clamp(A_next, LEAFAREA_MIN, LEAFAREA_MAX);
                        tiller.setLeafArea(static_cast<float>(A_next));
                    }

                    // Carbon supply driven by THIS year's A_t (photosynthesis proxy)
                    double leaf_mass_t_g = (A_t / (double)Tiller::SLA_CM2_PER_G);
                    if (!std::isfinite(leaf_mass_t_g) || leaf_mass_t_g < 0.0) leaf_mass_t_g = 0.0;

                    double supply_C = leaf_mass_t_g * ASSIM_C_PER_G_LEAF;
                    if (!std::isfinite(supply_C) || supply_C < 0.0) supply_C = 0.0;

                    double C_store_prev = tiller.getCStore();
                    if (!std::isfinite(C_store_prev) || C_store_prev < 0.0) C_store_prev = 0.0;

                    double C_avail = supply_C + C_store_prev;
                    if (!std::isfinite(C_avail) || C_avail < 0.0) C_avail = 0.0;

                    double C_root = fr * C_avail;
                    double C_stem = C_avail - C_root;
                    if (C_stem < 0.0) C_stem = 0.0;

                    // Roots: sample desired, then downscale to fit C_root
                    int n_roots = root_num_dis(gen);
                    double diam_mm = root_diam_dis(gen);

                    auto root_cost_C = [&](int n, double dmm) -> double {
                        double vol_cm3 = (double)Tiller::perRootConeVolumeCm3((float)dmm) * (double)n;
                        double mass_g = vol_cm3 * (double)Tiller::RHO_ROOT_G_PER_CM3;
                        return mass_g * CARBON_PER_G_BIOMASS;
                    };

                    double costR = root_cost_C(n_roots, diam_mm);

                    while (n_roots > 1 && costR > C_root) {
                        n_roots--;
                        costR = root_cost_C(n_roots, diam_mm);
                    }

                    if (costR > C_root && C_root > 0.0) {
                        double f = std::sqrt(C_root / std::max(1e-18, costR));
                        diam_mm *= f;
                        diam_mm = clamp(diam_mm, 0.5, 5.0);
                        costR = root_cost_C(n_roots, diam_mm);
                    }

                    if (C_root <= 0.0) {
                        n_roots = 1;
                        diam_mm = 0.5;
                    }

                    tiller.setRoots(n_roots, (float)diam_mm);

                    // Radius growth: limited by C_stem
                    double dr_base = growRadiusDist(gen);
                    if (!std::isfinite(dr_base) || dr_base < 0.0) dr_base = 0.01;

                    double r = tiller.getRadius();

                    double dh_cm = (STEM_INC_CM_PER_YR > 0.0 ? STEM_INC_CM_PER_YR : 0.01);
                    double h_cm = dh_cm * (double)tiller.getAge();
                    h_cm = std::max(dh_cm, h_cm);

                    auto stem_cost_C = [&](double dr) -> double {
                        double dV = M_PI * (2.0 * r * dr + dr * dr) * h_cm;
                        double mass_g = dV * RHO_STEM_G_PER_CM3;
                        return mass_g * CARBON_PER_G_BIOMASS;
                    };

                    double costS = stem_cost_C(dr_base);
                    double dr_use = dr_base;

                    if (costS > C_stem && C_stem > 0.0) {
                        double K = M_PI * h_cm * RHO_STEM_G_PER_CM3 * CARBON_PER_G_BIOMASS;
                        double target = C_stem / std::max(1e-18, K);

                        double disc = r*r + target;
                        if (disc < 0.0) disc = 0.0;
                        dr_use = -r + std::sqrt(disc);
                        if (!std::isfinite(dr_use) || dr_use < 0.0) dr_use = 0.0;
                    } else if (C_stem <= 0.0) {
                        dr_use = 0.0;
                    }

                    tiller.growRadius(dr_use);

                    double r_cm_now = (double)tiller.getRadius();
                    double V_stem_cm3 = M_PI * r_cm_now * r_cm_now * h_cm;
                    if (!std::isfinite(V_stem_cm3) || V_stem_cm3 < 0.0) V_stem_cm3 = 0.0;

                    double M_stem_g = V_stem_cm3 * RHO_STEM_G_PER_CM3;
                    if (!std::isfinite(M_stem_g) || M_stem_g < 0.0) M_stem_g = 0.0;

                    double C_store_next = STEM_TNC_FRAC * M_stem_g * CARBON_PER_G_BIOMASS;
                    if (!std::isfinite(C_store_next) || C_store_next < 0.0) C_store_next = 0.0;

                    tiller.setCStore(C_store_next);

                } else {
                    tiller.accumulateDeadLeafArea(static_cast<float>(prev_area));
                    tiller.accumulateRootNecroFromPrevRoots(prev_roots, prev_root_diam_mm);
                    tiller.setStatus(0);
                }

                step_data.push_back(tiller);

            } else {
                tiller.decay();
                tiller.setRoots(0, tiller.getRootDiamMM());
                tiller.setCStore(0.0);

                if (!should_prune_dead(tiller)) {
                    step_data.push_back(tiller);
                }
            }
        }

        step_data.insert(step_data.end(), newTillers.begin(), newTillers.end());

        int n_total = static_cast<int>(step_data.size());
        int n_alive = 0;
        for (const auto& tt : step_data) n_alive += (tt.getStatus() == 1);
        int n_dead = n_total - n_alive;

        if (ss.extinct_t < 0 && n_alive == 0) ss.extinct_t = time_step;
        if (ss.overflow_t < 0 && n_alive > alive_overflow_threshold) ss.overflow_t = time_step;

        bool stop_due_to_overflow = false;

        if (n_alive > alive_overflow_threshold) {
            double d = SENTINEL_SCALE / (static_cast<double>(time_step) + 1.0);
            d = std::max(50.0, std::min(5000.0, d));

            step_data.emplace_back(Tiller(1, 0.5,  +d, 0.0, 0.0, 3, 1, 50.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0));
            step_data.emplace_back(Tiller(1, 0.5,  -d, 0.0, 0.0, 3, 1, 50.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0));
            stop_due_to_overflow = true;
        }

        step_data.erase(
            std::remove_if(step_data.begin(), step_data.end(),
                           [](const Tiller& t) { return (double)t.getRadius() <= 1e-6; }),
            step_data.end()
        );

        OverlapStats ostats;
        resolveOverlaps(step_data, ostats);

        if (time_step == constraint_year) {
            ss.missing_year = 0;
            ss.alive_y = n_alive;

            double rmax = 0.0;
            double leaf_sum = 0.0;
            int leaf_n = 0;

            for (const auto& tt : step_data) {
                if (tt.getStatus() == 1) {
                    rmax = std::max(rmax, (double)tt.getRadius());

                    double la = (double)tt.getLeafArea();
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
            double diam = 0.0;
            if (!step_data.empty()) {
                double xmin = step_data[0].getX(), xmax = step_data[0].getX();
                for (const auto& tt : step_data) { xmin = std::min(xmin, tt.getX()); xmax = std::max(xmax, tt.getX()); }
                diam = xmax - xmin;
            }

            simlog << time_step << "\t" << n_total << "\t" << n_alive << "\t" << n_dead << "\t"
                   << static_cast<int>(newTillers.size()) << "\t" << diam << "\t"
                   << ostats.passes << "\t" << ostats.candidates << "\t" << ostats.overlapped << "\t"
                   << ostats.z_adjusts << "\t" << ostats.max_penetration << "\t" << ostats.ms << "\n";

            for (const Tiller& data : step_data) {
                outputFile << time_step << ','
                           << data.getAge() << ','
                           << data.getRadius() << ','
                           << data.getLeafArea() << ','
                           << data.getDeadLeafArea() << ','
                           << data.getDeadLeafMass() << ','
                           << data.getRootNecroVol() << ','
                           << data.getRootNecroVolCum() << ','
                           << data.getRootNecroMass() << ','
                           << data.getRootNecroMassCum() << ','
                           << data.getX() << ','
                           << data.getY() << ','
                           << data.getZ() << ','
                           << data.getNumRoots() << ','
                           << data.getRootDiamMM() << ','
                           << data.getStatus() << '\n';
            }
        }

        previous_step = std::move(step_data);
        if (stop_due_to_overflow) break;
    }

    ss.final_t = final_t;

    int alive_final = 0;
    for (const auto& tt : previous_step) alive_final += (tt.getStatus() == 1);
    ss.alive_final = alive_final;

    double final_diam = 0.0;
    if (!previous_step.empty()) {
        double xmin = previous_step[0].getX(), xmax = previous_step[0].getX();
        for (const auto& tt : previous_step) { xmin = std::min(xmin, tt.getX()); xmax = std::max(xmax, tt.getX()); }
        final_diam = xmax - xmin;
    }
    ss.final_diameter = final_diam;

    summary << ss.sim_id << "," << ss.final_t << "," << ss.final_diameter << ","
            << ss.alive_y << "," << ss.rmax_y << "," << ss.overflow_t << ","
            << ss.extinct_t << "," << ss.missing_year << "," << ss.alive_final << ",";

    if (std::isfinite(ss.leafarea_mean_y)) summary << ss.leafarea_mean_y;
    else summary << "";

    summary << "\n";

    if (mode == OutputMode::FULL) { outputFile.close(); simlog.close(); }
    summary.close();
}

int main() {
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);

    int max_sim_time;
    int num_sims;
    std::string outdir;
    unsigned long int num_threads;
    OutputMode mode = OutputMode::SUMMARY;

    const std::filesystem::path project_root = get_project_root();
    const std::filesystem::path ini_path = project_root / "parameterization.ini";

    std::string param_file_raw =
        ini_get(ini_path.string(), "Parameterization", "param_file", "parameters/parameters.txt");
    std::filesystem::path param_file_path = std::filesystem::path(param_file_raw);
    if (param_file_path.is_relative()) {
        param_file_path = project_root / param_file_path;
    }

    const int constraint_year = std::stoi(ini_get(ini_path.string(), "Parameterization", "constraint_year", "25"));
    const int alive_overflow_threshold = std::stoi(ini_get(ini_path.string(), "Parameterization", "alive_overflow_threshold", "600"));

    input(max_sim_time, num_sims, outdir, num_threads, mode);

    std::filesystem::path outdir_path = std::filesystem::path(outdir);
    if (outdir_path.is_relative()) outdir_path = project_root / outdir_path;
    outdir = outdir_path.string();

    std::vector<std::thread> threads;
    threads.reserve((size_t)num_threads);

    for (int sim_id = 0; sim_id < num_sims; sim_id++) {
        threads.emplace_back(
            simulate,
            max_sim_time,
            sim_id,
            outdir,
            param_file_path.string(),
            mode,
            constraint_year,
            alive_overflow_threshold
        );

        if ((threads.size() == num_threads) || (sim_id == num_sims - 1)) {
            for (auto& thread : threads) thread.join();
            threads.clear();
        }
    }

    return 0;
}