#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include "bh/octree.hpp"
#include "bh/types.hpp"

namespace bh {

struct ForceConfig {
    std::string order = "morton";  // packed force SoA, mapped to stable original particle IDs
    int chunk = 0;                 // 0: contiguous static, chunk 1 dynamic/guided
    bool have_history = false;
};

struct ForceMetrics {
    std::vector<std::uint64_t> thread_work;
    double order_s = 0, partition_s = 0, walk_s = 0;
};

ForceMetrics forces_omp_static(Particles&, const Octree&, const Params&, const ForceConfig&);
ForceMetrics forces_omp_dynamic(Particles&, const Octree&, const Params&, const ForceConfig&);
ForceMetrics forces_omp_guided(Particles&, const Octree&, const Params&, const ForceConfig&);
ForceMetrics forces_omp_costzones(Particles&, const Octree&, const Params&, const ForceConfig&);

// Prefix is in Morton order, and includes both endpoints (n+1 entries).
// Returns team+1 monotone boundaries, including 0 and n. All-zero costs
// use balanced particle counts. Exposed for independent partition tests.
std::vector<std::size_t> costzone_boundaries(const std::vector<std::uint64_t>& prefix, int team);

}  // namespace bh
