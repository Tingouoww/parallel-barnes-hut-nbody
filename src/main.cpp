// Simulation driver: time-step loop, per-stage timing, validation and CSV output.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

#include "bh/backend.hpp"
#include "bh/generator.hpp"
#include "bh/validate.hpp"

namespace {

struct Config {
    std::string dist = "plummer";
    std::size_t n = 10000;
    std::uint64_t seed = 42;
    int steps = 10;
    int warmup = 0;          // leading steps left out of the summary
    bh::Params prm;
    bh::Options opt;
    int validate_every = 0;  // 0 = never
    std::size_t samples = 1000;
    std::size_t energy_max_n = 20000;
    bool check_tree = false;
    double max_err = -1;     // fail when the p99 relative error exceeds this
    double max_drift = -1;   // fail when |energy drift| exceeds this
    std::string csv;
    bool quiet = false;
};

const char* kUsage =
    "usage: bh [options]\n"
    "  workload\n"
    "    --dist NAME           uniform | plummer | collision | degenerate   [plummer]\n"
    "    --n N                 number of particles                          [10000]\n"
    "    --seed S              random seed                                  [42]\n"
    "  simulation\n"
    "    --steps K             time steps to run                            [10]\n"
    "    --theta T             opening angle                                [0.5]\n"
    "    --dt DT               time step                                    [0.01]\n"
    "    --eps E               softening length                             [0.01]\n"
    "  backend\n"
    "    --backend NAME        seq | omp | cuda                             [seq]\n"
    "    --build NAME          tree build strategy (omp backend)            [seq]\n"
    "    --force NAME          force strategy (omp backend)                 [seq]\n"
    "                          seq | static | dynamic | guided | costzones\n"
    "    --force-order NAME    morton | original (parallel force only)      [morton]\n"
    "    --chunk K             0 = contiguous static / chunk 1 dynamic,guided [0]\n"
    "    --threads T           OpenMP threads (omp backend)                 [OpenMP default]\n"
    "  validation (never timed)\n"
    "    --validate-every K    compare with direct summation every K steps  [0 = off]\n"
    "    --samples M           particles checked per validation             [1000]\n"
    "    --energy-max-n N      compute the total energy only when n <= N    [20000]\n"
    "    --check-tree          verify the tree structure on validated steps\n"
    "    --max-err X           exit with status 2 if a p99 relative error exceeds X\n"
    "    --max-drift X         exit with status 2 if |energy drift| exceeds X\n"
    "  output\n"
    "    --warmup W            leading steps excluded from the summary      [0]\n"
    "    --csv FILE            append one row per step to FILE\n"
    "    --quiet               print only the summary\n";

Config parse(int argc, char** argv) {
    Config c;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            return argv[++i];
        };
        if (a == "--help" || a == "-h") {
            std::fputs(kUsage, stdout);
            std::exit(0);
        } else if (a == "--dist") c.dist = value();
        else if (a == "--n") c.n = static_cast<std::size_t>(std::stod(value()));
        else if (a == "--seed") c.seed = std::stoull(value());
        else if (a == "--steps") c.steps = std::stoi(value());
        else if (a == "--warmup") c.warmup = std::stoi(value());
        else if (a == "--theta") c.prm.theta = static_cast<bh::real>(std::stod(value()));
        else if (a == "--dt") c.prm.dt = static_cast<bh::real>(std::stod(value()));
        else if (a == "--eps") c.prm.eps = static_cast<bh::real>(std::stod(value()));
        else if (a == "--backend") c.opt.backend = value();
        else if (a == "--build") c.opt.build = value();
        else if (a == "--force") c.opt.force = value();
        else if (a == "--force-order") c.opt.force_order = value();
        else if (a == "--chunk") c.opt.chunk = std::stoi(value());
        else if (a == "--threads") c.opt.threads = std::stoi(value());
        else if (a == "--validate-every") c.validate_every = std::stoi(value());
        else if (a == "--samples") c.samples = static_cast<std::size_t>(std::stod(value()));
        else if (a == "--energy-max-n") c.energy_max_n = static_cast<std::size_t>(std::stod(value()));
        else if (a == "--check-tree") c.check_tree = true;
        else if (a == "--max-err") c.max_err = std::stod(value());
        else if (a == "--max-drift") c.max_drift = std::stod(value());
        else if (a == "--csv") c.csv = value();
        else if (a == "--quiet") c.quiet = true;
        else throw std::runtime_error("unknown option " + a + " (see --help)");
    }
    if (c.steps < 1) throw std::runtime_error("--steps must be at least 1");
    if (c.warmup < 0 || c.warmup >= c.steps) throw std::runtime_error("--warmup must be in [0, steps)");
    if (c.n > 2000000000u) throw std::runtime_error("--n is limited to 2e9 (32-bit particle indices)");
    if (c.opt.chunk < 0) throw std::runtime_error("--chunk must be nonnegative");
    if (c.opt.threads < 0) throw std::runtime_error("--threads must be nonnegative");
    return c;
}

const char* kStageNames[5] = {"bbox", "build", "com", "force", "integrate"};

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

template <class F>
double timed(F&& stage) {
    const auto t0 = std::chrono::steady_clock::now();
    stage();
    return seconds_since(t0);
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const std::size_t k = v.size();
    return k % 2 ? v[k / 2] : 0.5 * (v[k / 2 - 1] + v[k / 2]);
}

int run(const Config& c) {
    const int threads = c.opt.backend == "omp"
                            ? (c.opt.threads > 0 ? c.opt.threads : omp_get_max_threads())
                            : 1;
    std::printf("bh: dist=%s n=%zu seed=%llu steps=%d theta=%g eps=%g dt=%g precision=%s "
                "backend=%s build=%s force=%s threads=%d force_order=%s chunk=%d\n",
                c.dist.c_str(), c.n, static_cast<unsigned long long>(c.seed), c.steps,
                double(c.prm.theta), double(c.prm.eps), double(c.prm.dt), BH_PRECISION_NAME,
                c.opt.backend.c_str(), c.opt.build.c_str(), c.opt.force.c_str(), threads,
                c.opt.force == "seq" ? "original" : c.opt.force_order.c_str(), c.opt.chunk);

    bh::Particles particles = bh::generate(c.dist, c.n, c.seed);
    auto backend = bh::make_backend(c.opt, c.prm);
    backend->load(particles);

    std::ofstream csv;
    if (!c.csv.empty()) {
        const std::string header =
            "dist,n,seed,theta,eps,dt,precision,backend,build,force,threads,step,"
            "bbox_s,build_s,com_s,force_s,integrate_s,total_s,nodes,interactions,"
            "err_median,err_p99,err_rms,energy,energy_drift,force_order,chunk,"
            "actual_threads,thread_interactions,load_imbalance,force_order_s,force_partition_s,"
            "force_walk_s,warmup,is_warmup";
        bool need_header = true;
        {
            std::ifstream probe(c.csv);
            need_header = !probe.good() || probe.peek() == std::ifstream::traits_type::eof();
            if (!need_header) {
                std::string existing;
                std::getline(probe, existing);
                if (!existing.empty() && existing.back() == '\r') existing.pop_back();
                if (existing != header)
                    throw std::runtime_error("CSV header mismatch: use a new file for " + c.csv);
            }
        }
        csv.open(c.csv, std::ios::app);
        if (!csv) throw std::runtime_error("cannot open " + c.csv);
        csv << std::setprecision(17);
        if (need_header) csv << header << '\n';
    }

    std::vector<double> history[5], totals;
    double energy0 = 0;
    bool have_energy0 = false;
    bool failed = false;
    bh::StepStats stats;

    for (int step = 0; step < c.steps; ++step) {
        double t[5];
        t[0] = timed([&] { backend->compute_bbox(); });
        t[1] = timed([&] { backend->build_tree(); });
        t[2] = timed([&] { backend->compute_com(); });
        t[3] = timed([&] { backend->compute_forces(); });
        stats = backend->step_stats();

        // Validate here: positions and accelerations belong to the same instant.
        const bool validating = c.validate_every > 0 && step % c.validate_every == 0;
        bh::ValidationResult v;
        double drift = 0;
        std::string tree_report;
        if (validating) {
            backend->store(particles);
            v = bh::validate(particles, c.prm, step == 0 ? 0.0 : 0.5 * double(c.prm.dt), c.samples,
                             c.energy_max_n);
            if (v.has_energy) {
                if (!have_energy0) {
                    energy0 = v.energy;
                    have_energy0 = true;
                }
                drift = (v.energy - energy0) / std::fabs(energy0);
            }
            if (c.check_tree) {
                const bh::Octree* tree = backend->host_tree();
                if (tree == nullptr) throw std::runtime_error("--check-tree needs a backend with a host tree");
                tree_report = bh::check_tree(particles, *tree);
            }
        }

        // Half kick on the first step: velocities then lag the positions by dt / 2.
        const bh::real kick = step == 0 ? c.prm.dt / 2 : c.prm.dt;
        t[4] = timed([&] { backend->integrate(kick, c.prm.dt); });

        const double total = t[0] + t[1] + t[2] + t[3] + t[4];
        if (step >= c.warmup) {
            for (int s = 0; s < 5; ++s) history[s].push_back(t[s]);
            totals.push_back(total);
        }

        if (!c.quiet) {
            std::printf("step %4d  bbox %.5f  build %.5f  com %.5f  force %.5f  integrate %.5f"
                        "  total %.5f s  nodes %llu  inter/particle %.1f\n",
                        step, t[0], t[1], t[2], t[3], t[4], total,
                        static_cast<unsigned long long>(stats.nodes),
                        double(stats.interactions) / double(c.n));
            if (!stats.thread_interactions.empty())
                std::printf("           force team %zu  load imbalance %.4f\n",
                            stats.thread_interactions.size(), stats.load_imbalance);
        }
        if (validating) {
            if (!c.quiet) {
                std::printf("           error vs direct sum (%zu samples): median %.3e  p99 %.3e  max %.3e"
                            "  rms %.3e\n", v.samples, v.err_median, v.err_p99, v.err_max, v.err_rms);
                if (v.has_energy)
                    std::printf("           energy %.8f (K %.6f, U %.6f)  drift %+.3e\n", v.energy,
                                v.kinetic, v.potential, drift);
                if (c.check_tree && tree_report.empty()) std::printf("           tree check passed\n");
            }
            if (!tree_report.empty()) {
                std::printf("step %d: tree check FAILED\n%s", step, tree_report.c_str());
                failed = true;
            }
            if (c.max_err >= 0 && !(v.err_p99 <= c.max_err)) {
                std::printf("step %d: p99 relative error %.3e exceeds --max-err %.3e\n", step, v.err_p99,
                            c.max_err);
                failed = true;
            }
            if (c.max_drift >= 0 && v.has_energy && !(std::fabs(drift) <= c.max_drift)) {
                std::printf("step %d: energy drift %+.3e exceeds --max-drift %.3e\n", step, drift,
                            c.max_drift);
                failed = true;
            }
        }

        if (csv.is_open()) {
            csv << c.dist << ',' << c.n << ',' << c.seed << ',' << double(c.prm.theta) << ','
                << double(c.prm.eps) << ',' << double(c.prm.dt) << ',' << BH_PRECISION_NAME << ','
                << c.opt.backend << ',' << c.opt.build << ',' << c.opt.force << ',' << threads << ','
                << step;
            for (int s = 0; s < 5; ++s) csv << ',' << t[s];
            csv << ',' << total << ',' << stats.nodes << ',' << stats.interactions;
            if (validating) {
                csv << ',' << v.err_median << ',' << v.err_p99 << ',' << v.err_rms;
                if (v.has_energy) csv << ',' << v.energy << ',' << drift;
                else csv << ",,";
            } else {
                csv << ",,,,,";
            }
            csv << ',' << (c.opt.force == "seq" ? "original" : c.opt.force_order) << ',' << c.opt.chunk
                << ',' << stats.thread_interactions.size() << ',';
            for (std::size_t rank = 0; rank < stats.thread_interactions.size(); ++rank) {
                if (rank) csv << ';';
                csv << stats.thread_interactions[rank];
            }
            csv << ',' << stats.load_imbalance << ',' << stats.force_order_s << ','
                << stats.force_partition_s << ',' << stats.force_walk_s << ',' << c.warmup << ','
                << (step < c.warmup ? 1 : 0) << '\n';
        }
    }

    const double total_median = median(totals);
    std::printf("summary over %zu steps (median seconds per step, share of total)\n", totals.size());
    for (int s = 0; s < 5; ++s) {
        const double m = median(history[s]);
        std::printf("  %-10s %.6f  %5.1f%%\n", kStageNames[s], m,
                    total_median > 0 ? 100.0 * m / total_median : 0.0);
    }
    std::printf("  %-10s %.6f\n", "total", total_median);
    if (failed) {
        std::printf("VALIDATION FAILED\n");
        return 2;
    }
    if (c.validate_every > 0) std::printf("validation passed\n");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(parse(argc, argv));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
