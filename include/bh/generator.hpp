// Workload generator: reproducible initial conditions with different degrees
// of load imbalance. Units: G = 1, total mass = 1.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "bh/types.hpp"

namespace bh {

// dist is one of
//   uniform     uniform in the cube [-1, 1]^3, at rest (most balanced)
//   plummer     one Plummer sphere, scale radius 1 (dense centre)
//   collision   two Plummer spheres on a parabolic collision course
//   degenerate  test input: coincident and nearly coincident particles
// The same (dist, n, seed) always gives the same particles on every platform.
// Throws std::runtime_error for an unknown name or n == 0.
Particles generate(const std::string& dist, std::size_t n, std::uint64_t seed);

}  // namespace bh
