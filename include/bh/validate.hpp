// Validation module: compares a backend's accelerations against O(N^2) direct
// summation, tracks the total energy, and checks the structure of a tree.
// Everything here runs outside the timed stages.
#pragma once

#include <cstddef>
#include <string>

#include "bh/octree.hpp"
#include "bh/types.hpp"

namespace bh {

struct ValidationResult {
    std::size_t samples = 0;  // particles whose acceleration was checked
    // Relative acceleration error |a - a_direct| / |a_direct| over the samples.
    double err_median = 0, err_p99 = 0, err_max = 0;
    // sqrt(sum |a - a_direct|^2 / sum |a_direct|^2): not inflated by particles
    // whose true acceleration is close to zero.
    double err_rms = 0;

    bool has_energy = false;  // false when N is above the energy limit
    double kinetic = 0, potential = 0, energy = 0;
};

// Checks p.ax/ay/az against direct summation in double precision for at most
// `max_samples` particles spread evenly over the array. When N <= energy_max_n
// the total energy is computed too (the potential costs O(N^2)).
// Velocities in the leapfrog scheme lag half a step behind the positions;
// `half_kick` (dt/2, or 0 at the very first step) brings them level before the
// kinetic energy is taken.
ValidationResult validate(const Particles& p, const Params& prm, double half_kick,
                          std::size_t max_samples, std::size_t energy_max_n);

// Structural check of a summarized tree: every particle is in exactly one
// leaf and inside that leaf's cell, node sizes match their depth, only cells
// at MAX_DEPTH hold several particles, and the mass and centre of mass of
// every node agree with its children. Returns an empty string when the tree is
// valid, otherwise a description of the first problems found.
std::string check_tree(const Particles& p, const Octree& tree);

}  // namespace bh
