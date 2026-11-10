// Shared basic types: floating-point precision, simulation parameters and
// the particle container used by every backend.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace bh {

// Single precision by default so CPU and GPU results are comparable.
// Build with `make PRECISION=double` to switch.
#ifdef BH_DOUBLE
using real = double;
#define BH_PRECISION_NAME "double"
#else
using real = float;
#define BH_PRECISION_NAME "float"
#endif

struct Params {
    real theta = real(0.5);   // opening criterion: accept a cell when size / distance < theta
    real eps   = real(0.01);  // Plummer softening length
    real G     = real(1.0);   // gravitational constant
    real dt    = real(0.01);  // time step
};

// Structure-of-arrays particle storage (same layout on the CPU and the GPU).
struct Particles {
    std::vector<real> x, y, z;     // position
    std::vector<real> vx, vy, vz;  // velocity
    std::vector<real> ax, ay, az;  // acceleration from the last force stage
    std::vector<real> m;           // mass
    // Number of interactions the last force stage computed for each particle.
    // This is the per-particle cost that costzones feeds back into the next step.
    std::vector<std::uint32_t> cost;

    std::size_t size() const { return x.size(); }

    void resize(std::size_t n) {
        for (auto* v : {&x, &y, &z, &vx, &vy, &vz, &ax, &ay, &az, &m}) v->assign(n, real(0));
        cost.assign(n, 0);
    }
};

// Axis-aligned bounding box of all particle positions.
struct BBox {
    real min[3];
    real max[3];
};

}  // namespace bh
