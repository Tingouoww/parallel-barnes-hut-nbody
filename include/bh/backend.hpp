// Backend interface: one time step is five stages, and every backend
// (sequential, OpenMP, CUDA) implements the same five calls so the driver can
// time them one by one.
#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "bh/octree.hpp"
#include "bh/types.hpp"

namespace bh {

struct Options {
    std::string backend = "seq";  // seq | omp | cuda
    std::string build   = "seq";  // tree construction strategy (omp backend)
    std::string force   = "seq";  // force calculation strategy (omp backend)
    int threads = 0;              // 0 = leave the OpenMP default
};

struct StepStats {
    std::uint64_t nodes = 0;         // internal nodes in the current tree
    std::uint64_t interactions = 0;  // interactions computed by the last force stage
};

class Backend {
public:
    virtual ~Backend() = default;

    // Copies the initial state in (to the device, for a GPU backend).
    virtual void load(const Particles& p) = 0;

    // The five stages. Each one must have finished when the call returns
    // (a GPU backend synchronizes before returning), because the driver times
    // each call separately.
    virtual void compute_bbox() = 0;
    virtual void build_tree() = 0;
    virtual void summarize() = 0;        // total mass and centre of mass of every node
    virtual void compute_forces() = 0;   // fills the accelerations and the per-particle cost
    // Leapfrog: v += a * dt_kick, then x += v * dt_drift.
    virtual void integrate(real dt_kick, real dt_drift) = 0;

    // Copies positions, velocities, accelerations and costs back to the host.
    // Never timed; used for validation and output.
    virtual void store(Particles& out) = 0;

    virtual StepStats step_stats() const { return {}; }

    // The current tree if it lives in host memory, otherwise nullptr.
    virtual const Octree* host_tree() const { return nullptr; }
};

// Throws std::runtime_error for an unknown backend or strategy name.
std::unique_ptr<Backend> make_backend(const Options& opt, const Params& prm);

// Implemented in src/cuda/ and only linked when built with `make CUDA=1`.
std::unique_ptr<Backend> make_cuda_backend(const Options& opt, const Params& prm);

}  // namespace bh
