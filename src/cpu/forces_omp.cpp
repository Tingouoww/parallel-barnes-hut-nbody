#include "bh/forces.hpp"
#include "bh/force_kernel.hpp"
#include "bh/parallel.hpp"
#include "bh/timer.hpp"

#include <algorithm>
#include <stdexcept>
#include <omp.h>

namespace bh {
namespace {
void fill_costzones(const std::vector<std::uint64_t>& prefix, int team, std::size_t* cuts) {
    const std::size_t n = prefix.size() - 1;
    const std::uint64_t total = prefix.back();
    cuts[0] = 0;
    cuts[team] = n;
    for (int t = 1; t < team; ++t) {
        const std::uint64_t target = (total / team) * t + ((total % team) * t) / team;
        cuts[t] = total == 0 ? partition_begin(n, t, team)
                            : static_cast<std::size_t>(std::lower_bound(prefix.begin(), prefix.end(), target)
                                                       - prefix.begin());
    }
}
}  // namespace

std::vector<std::size_t> costzone_boundaries(const std::vector<std::uint64_t>& prefix, int team) {
    if (prefix.empty() || team < 1) throw std::invalid_argument("invalid costzone prefix/team");
    std::vector<std::size_t> cuts(team + 1);
    fill_costzones(prefix, team, cuts.data());
    return cuts;
}

namespace {
enum class Schedule { Static, Dynamic, Guided, Costzones };

ForceMetrics run(Particles& p, const Octree& tree, const Params& prm,
                 const ForceConfig& cfg, Schedule schedule) {
    if (cfg.order != "morton" && cfg.order != "original")
        throw std::invalid_argument("--force-order must be morton or original");
    if (schedule == Schedule::Costzones && cfg.order != "morton")
        throw std::invalid_argument("costzones requires --force-order morton");
    if (cfg.chunk < 0) throw std::invalid_argument("--chunk must be nonnegative");
    ForceMetrics metrics;
    const long long n = static_cast<long long>(p.size());
    const bool sorted = cfg.order == "morton";
    const double order_start = steady_seconds();
    std::vector<std::int32_t> order;
    std::vector<std::int32_t> inverse;
    Particles packed;
    if (sorted) {
        radix_order_omp(tree.key, order);
        inverse.resize(p.size());
        // Velocity is irrelevant to a force-only view; gather just the force
        // inputs and history, then scatter outputs back to the persistent SoA.
        for (auto* v : {&packed.x, &packed.y, &packed.z, &packed.m,
                        &packed.ax, &packed.ay, &packed.az}) v->resize(p.size());
        packed.cost.resize(p.size());
#pragma omp parallel for schedule(static)
        for (long long j = 0; j < n; ++j) {
            const auto i = order[j];
            inverse[i] = static_cast<std::int32_t>(j);
            packed.x[j] = p.x[i]; packed.y[j] = p.y[i]; packed.z[j] = p.z[i];
            packed.m[j] = p.m[i]; packed.cost[j] = p.cost[i];
        }
    }
    metrics.order_s = steady_seconds() - order_start;

    const int chunk = cfg.chunk > 0 ? cfg.chunk : 1;
    std::vector<std::uint64_t> prefix;
    const double partition_start = steady_seconds();
    if (schedule == Schedule::Costzones) {
        // Snapshot every historical cost BEFORE any thread overwrites p.cost.
        std::vector<std::uint64_t> costs(p.size(), 1);
        if (cfg.have_history) {
#pragma omp parallel for schedule(static)
            for (long long j = 0; j < n; ++j) costs[j] = packed.cost[j];
        }
        exclusive_scan_omp(costs, prefix);
    }

    metrics.thread_work.resize(omp_get_max_threads(), 0);
    // Allocate before entering the team (no allocations/exceptions in workers).
    std::vector<std::size_t> cuts(metrics.thread_work.size() + 1);
    int actual_team = 1;
    double walk_start = steady_seconds();
#pragma omp parallel
    {
        const int rank = omp_get_thread_num(), team = omp_get_num_threads();
        std::uint64_t work = 0;
        auto particle = [&](long long j) {
            if (sorted) {
                force_on_sorted_particle(packed, tree, prm, j, order[j], inverse.data());
                work += packed.cost[j];
            } else {
                force_on_particle(p, tree, prm, j);
                work += p.cost[j];
            }
        };
#pragma omp single
        {
            actual_team = team;
            if (schedule == Schedule::Costzones) {
                fill_costzones(prefix, team, cuts.data());
                metrics.partition_s = steady_seconds() - partition_start;
                walk_start = steady_seconds();
            }
        }
        if (schedule == Schedule::Costzones) {
            for (std::size_t j = cuts[rank]; j < cuts[rank + 1]; ++j) particle(j);
        } else if (schedule == Schedule::Static && cfg.chunk == 0) {
#pragma omp for schedule(static)
            for (long long j = 0; j < n; ++j) particle(j);
        } else if (schedule == Schedule::Static) {
#pragma omp for schedule(static, chunk)
            for (long long j = 0; j < n; ++j) particle(j);
        } else if (schedule == Schedule::Dynamic) {
#pragma omp for schedule(dynamic, chunk)
            for (long long j = 0; j < n; ++j) particle(j);
        } else {
#pragma omp for schedule(guided, chunk)
            for (long long j = 0; j < n; ++j) particle(j);
        }
        metrics.thread_work[rank] = work;
    }
    metrics.walk_s = steady_seconds() - walk_start;
    if (sorted) {
        const double scatter_start = steady_seconds();
#pragma omp parallel for schedule(static)
        for (long long j = 0; j < n; ++j) {
            const auto i = order[j];
            p.ax[i] = packed.ax[j]; p.ay[i] = packed.ay[j]; p.az[i] = packed.az[j];
            p.cost[i] = packed.cost[j];
        }
        metrics.order_s += steady_seconds() - scatter_start;
    }
    metrics.thread_work.resize(actual_team);
    return metrics;
}
}  // namespace

ForceMetrics forces_omp_static(Particles& p, const Octree& t, const Params& prm, const ForceConfig& c) {
    return run(p, t, prm, c, Schedule::Static);
}
ForceMetrics forces_omp_dynamic(Particles& p, const Octree& t, const Params& prm, const ForceConfig& c) {
    return run(p, t, prm, c, Schedule::Dynamic);
}
ForceMetrics forces_omp_guided(Particles& p, const Octree& t, const Params& prm, const ForceConfig& c) {
    return run(p, t, prm, c, Schedule::Guided);
}
ForceMetrics forces_omp_costzones(Particles& p, const Octree& t, const Params& prm, const ForceConfig& c) {
    return run(p, t, prm, c, Schedule::Costzones);
}

}  // namespace bh
