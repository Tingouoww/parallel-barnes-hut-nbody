#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "bh/types.hpp"

namespace bh {

// dist: uniform | plummer | collision | degenerate. Units: G = 1, total mass = 1.
// The same (dist, n, seed) gives the same particles on every platform.
Particles generate(const std::string& dist, std::size_t n, std::uint64_t seed);

}  // namespace bh
