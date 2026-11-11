// One time step is five stages; every backend implements the same five calls
// so the driver can time them separately.
#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "bh/octree.hpp"
#include "bh/types.hpp"

namespace bh {

struct Options {
    std::string backend = "seq";  // seq | omp | cuda
    std::string build   = "seq";  // tree build strategy (omp backend)
    std::string force   = "seq";  // force strategy (omp backend)
    int threads = 0;              // 0 = OpenMP default
};

struct StepStats {
    std::uint64_t nodes = 0;
    std::uint64_t interactions = 0;
};

class Backend {
public:
    virtual ~Backend() = default;

    virtual void load(const Particles& p) = 0;

    // Each stage must be complete when it returns (a GPU backend synchronizes first).
    virtual void compute_bbox() = 0;
    virtual void build_tree() = 0;
    virtual void compute_com() = 0;
    virtual void compute_forces() = 0;
    virtual void integrate(real dt_kick, real dt_drift) = 0;  // v += a * dt_kick; x += v * dt_drift

    // Copies positions, velocities, accelerations and costs back to the host (not timed).
    virtual void store(Particles& out) = 0;

    virtual StepStats step_stats() const { return {}; }
    virtual const Octree* host_tree() const { return nullptr; }  // nullptr if the tree is not on the host
};

std::unique_ptr<Backend> make_backend(const Options& opt, const Params& prm);

// Defined in src/cuda/, linked only with `make CUDA=1`.
std::unique_ptr<Backend> make_cuda_backend(const Options& opt, const Params& prm);

}  // namespace bh
