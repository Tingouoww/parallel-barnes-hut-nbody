// Validation against O(N^2) direct summation. Never part of the timed stages.
#pragma once

#include <cstddef>
#include <string>

#include "bh/octree.hpp"
#include "bh/types.hpp"

namespace bh {

struct ValidationResult {
    std::size_t samples = 0;
    double err_median = 0, err_p99 = 0, err_max = 0;  // |a - a_direct| / |a_direct| per particle
    double err_rms = 0;                               // sqrt(sum |a - a_direct|^2 / sum |a_direct|^2)
    bool has_energy = false;
    double kinetic = 0, potential = 0, energy = 0;
};

// Checks the accelerations of up to max_samples particles; computes the total
// energy when N <= energy_max_n. half_kick (dt / 2, or 0 at step 0) brings the
// leapfrog velocities level with the positions.
ValidationResult validate(const Particles& p, const Params& prm, double half_kick,
                          std::size_t max_samples, std::size_t energy_max_n);

// Structural check of a tree after the centre-of-mass stage. Returns "" when it is valid.
std::string check_tree(const Particles& p, const Octree& tree);

}  // namespace bh
