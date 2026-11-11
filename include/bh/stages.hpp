// CPU implementations of the five stages. Variants of a stage share one
// signature and are registered in src/cpu/cpu_backend.cpp.
#pragma once

#include "bh/octree.hpp"
#include "bh/types.hpp"

namespace bh {

BBox bbox_seq(const Particles& p);
BBox bbox_omp(const Particles& p);

void build_tree_seq(const Particles& p, const BBox& box, Octree& tree);
void build_tree_lock(const Particles& p, const BBox& box, Octree& tree);

void com_seq(const Particles& p, Octree& tree);

void forces_seq(Particles& p, const Octree& tree, const Params& prm);

void integrate_seq(Particles& p, real dt_kick, real dt_drift);
void integrate_omp(Particles& p, real dt_kick, real dt_drift);

}  // namespace bh
