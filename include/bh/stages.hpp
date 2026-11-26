// CPU implementations of the five stages. Variants of a stage share one
// signature and are registered in src/cpu/cpu_backend.cpp.
#pragma once

#include <cstdint>
#include <vector>

#include "bh/octree.hpp"
#include "bh/types.hpp"

namespace bh {

BBox bbox_seq(const Particles& p);
BBox bbox_omp(const Particles& p);

void build_tree_seq(const Particles& p, const BBox& box, Octree& tree);
void build_tree_lock(const Particles& p, const BBox& box, Octree& tree);
void build_tree_morton(const Particles& p, const BBox& box, Octree& tree);
void build_tree_karras(const Particles& p, const BBox& box, Octree& tree);

// Sets the root cube and tree.key; order[a] is the particle with the a-th
// smallest (key, index) and sorted[a] its key. Skips the sort, and returns
// true, when the particles are already in Morton order.
bool morton_order(const Particles& p, const BBox& box, Octree& tree, std::vector<std::int32_t>& order,
                  std::vector<std::uint64_t>& sorted);

void com_seq(const Particles& p, Octree& tree);
void com_level(const Particles& p, Octree& tree);

// Per-level timings of com_level(), recorded while com_level_profile is set
// (bench/com_levels.cpp). busy is the time a thread spends on its nodes of a
// level, wait the time it then spends at the barrier that ends the level.
struct ComLevelProfile {
    int threads = 0;                  // team size the runtime actually gave
    std::vector<std::int32_t> nodes;  // [level]
    std::vector<double> busy, wait;   // [level * threads + thread], seconds
};
extern ComLevelProfile* com_level_profile;

void forces_seq(Particles& p, const Octree& tree, const Params& prm);

void integrate_seq(Particles& p, real dt_kick, real dt_drift);
void integrate_omp(Particles& p, real dt_kick, real dt_drift);

}  // namespace bh
