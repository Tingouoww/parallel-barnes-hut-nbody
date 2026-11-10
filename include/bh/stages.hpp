// CPU implementations of the five stages. Every variant of a stage has the
// same signature, so a new strategy is one function plus one line in the
// table in src/cpu/cpu_backend.cpp.
#pragma once

#include "bh/octree.hpp"
#include "bh/types.hpp"

namespace bh {

// 1. Bounding box
BBox bbox_seq(const Particles& p);
BBox bbox_omp(const Particles& p);

// 2. Tree construction
void build_tree_seq(const Particles& p, const BBox& box, Octree& tree);

// 3. Mass / centre-of-mass summarization
// Requires every child node to have a larger index than its parent.
void summarize_seq(const Particles& p, Octree& tree);

// 4. Force calculation (fills p.ax/ay/az and p.cost)
void forces_seq(Particles& p, const Octree& tree, const Params& prm);

// 5. Integration: v += a * dt_kick, then x += v * dt_drift
void integrate_seq(Particles& p, real dt_kick, real dt_drift);
void integrate_omp(Particles& p, real dt_kick, real dt_drift);

}  // namespace bh
