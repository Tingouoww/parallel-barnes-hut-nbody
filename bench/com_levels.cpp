// Per-level profile of the level-by-level centre-of-mass calculation.
//
//   build/float/com_levels DIST N THREADS [BUILD = morton] [REPS = 20]
//
// Builds one tree, runs com_level() and com_seq() REPS times on it and prints
// CSV, one row per non-empty tree level (medians over the repetitions; team is
// the thread count the runtime actually gave, over which the statistics run):
//   busy_mean_s, busy_max_s   time a thread spends on its nodes of the level
//   wait_mean_s, wait_min_s   time it then spends at the level's barrier;
//                             the minimum is the cost of the barrier itself
//   total_s, seq_s            whole com_level() and com_seq() calls
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

#include <omp.h>

#include "bh/generator.hpp"
#include "bh/stages.hpp"

namespace {

template <class F>
double timed(F&& f) {
    const auto t0 = std::chrono::steady_clock::now();
    f();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s DIST N THREADS [BUILD = morton] [REPS = 20]\n", argv[0]);
        return 1;
    }
    const std::string dist = argv[1], build = argc > 4 ? argv[4] : "morton";
    const std::size_t n = static_cast<std::size_t>(std::atof(argv[2]));
    const int threads = std::atoi(argv[3]), reps = argc > 5 ? std::atoi(argv[5]) : 20;
    omp_set_num_threads(threads);

    const bh::Particles p = bh::generate(dist, n, 42);
    bh::Octree tree;
    const bh::BBox box = bh::bbox_omp(p);
    if (build == "morton") bh::build_tree_morton(p, box, tree);
    else if (build == "lock") bh::build_tree_lock(p, box, tree);
    else bh::build_tree_seq(p, box, tree);

    constexpr int kLevels = bh::Octree::MAX_DEPTH;
    enum { BusyMean, BusyMax, WaitMean, WaitMin, kStats };
    std::vector<double> level[kLevels][kStats], total, seq;
    bh::ComLevelProfile prof;
    bh::com_level(p, tree);  // starts the thread team
    bh::com_level_profile = &prof;
    for (int r = 0; r < reps; ++r) {
        seq.push_back(timed([&] { bh::com_seq(p, tree); }));
        total.push_back(timed([&] { bh::com_level(p, tree); }));
        for (int l = 0; l < kLevels; ++l) {
            const int team = prof.threads;
            const double* busy = &prof.busy[static_cast<std::size_t>(l) * team];
            const double* wait = &prof.wait[static_cast<std::size_t>(l) * team];
            level[l][BusyMean].push_back(std::accumulate(busy, busy + team, 0.0) / team);
            level[l][BusyMax].push_back(*std::max_element(busy, busy + team));
            level[l][WaitMean].push_back(std::accumulate(wait, wait + team, 0.0) / team);
            level[l][WaitMin].push_back(*std::min_element(wait, wait + team));
        }
    }

    const int levels = static_cast<int>(std::count_if(prof.nodes.begin(), prof.nodes.end(),
                                                      [](std::int32_t count) { return count > 0; }));
    std::printf("dist,n,build,threads,team,levels,level,nodes,busy_mean_s,busy_max_s,wait_mean_s,wait_min_s,"
                "total_s,seq_s\n");
    for (int l = 0; l < kLevels; ++l) {
        if (prof.nodes[l] == 0) continue;
        std::printf("%s,%zu,%s,%d,%d,%d,%d,%d,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n", dist.c_str(), n, build.c_str(),
                    threads, prof.threads, levels, l, prof.nodes[l], median(level[l][BusyMean]), median(level[l][BusyMax]),
                    median(level[l][WaitMean]), median(level[l][WaitMin]), median(total), median(seq));
    }
}
