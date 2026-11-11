// CPU backend ("seq" and "omp"): keeps the particles and the tree in host
// memory and runs each stage with the strategy chosen on the command line.
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <omp.h>

#include "bh/backend.hpp"
#include "bh/stages.hpp"

namespace bh {
namespace {

using BuildFn = void (*)(const Particles&, const BBox&, Octree&);
using ForceFn = void (*)(Particles&, const Octree&, const Params&);

// Strategy tables: the name is the value of --build / --force.
const std::vector<std::pair<std::string, BuildFn>> kBuild = {
    {"seq", build_tree_seq},
    {"lock", build_tree_lock},
};
const std::vector<std::pair<std::string, ForceFn>> kForce = {
    {"seq", forces_seq},
};

template <class Fn>
Fn lookup(const std::vector<std::pair<std::string, Fn>>& table, const std::string& name,
          const char* what) {
    std::string known;
    for (const auto& e : table) {
        if (e.first == name) return e.second;
        known += (known.empty() ? "" : ", ") + e.first;
    }
    throw std::runtime_error(std::string("unknown ") + what + " strategy '" + name +
                             "' (available: " + known + ")");
}

class CpuBackend final : public Backend {
public:
    CpuBackend(const Options& opt, const Params& prm, bool parallel)
        : prm_(prm), parallel_(parallel), build_(lookup(kBuild, opt.build, "tree build")),
          force_(lookup(kForce, opt.force, "force")) {
        if (parallel && opt.threads > 0) omp_set_num_threads(opt.threads);
    }

    void load(const Particles& p) override { p_ = p; }
    void store(Particles& out) override { out = p_; }

    void compute_bbox() override { box_ = parallel_ ? bbox_omp(p_) : bbox_seq(p_); }
    void build_tree() override { build_(p_, box_, tree_); }
    void compute_com() override { com_seq(p_, tree_); }
    void compute_forces() override { force_(p_, tree_, prm_); }
    void integrate(real dt_kick, real dt_drift) override {
        if (parallel_) integrate_omp(p_, dt_kick, dt_drift);
        else integrate_seq(p_, dt_kick, dt_drift);
    }

    StepStats step_stats() const override {
        StepStats s;
        s.nodes = static_cast<std::uint64_t>(tree_.num_nodes() - tree_.blank_nodes);
        for (std::uint32_t c : p_.cost) s.interactions += c;
        return s;
    }

    const Octree* host_tree() const override { return &tree_; }

private:
    Params prm_;
    bool parallel_;
    BuildFn build_;
    ForceFn force_;
    Particles p_;
    Octree tree_;
    BBox box_{};
};

}  // namespace

std::unique_ptr<Backend> make_backend(const Options& opt, const Params& prm) {
    if (opt.backend == "seq") {
        if (opt.build != "seq" || opt.force != "seq")
            throw std::runtime_error("--build / --force strategies need --backend omp");
        return std::make_unique<CpuBackend>(opt, prm, false);
    }
    if (opt.backend == "omp") return std::make_unique<CpuBackend>(opt, prm, true);
    if (opt.backend == "cuda") {
#ifdef BH_CUDA
        return make_cuda_backend(opt, prm);
#else
        throw std::runtime_error("this binary was built without CUDA (use `make CUDA=1`)");
#endif
    }
    throw std::runtime_error("unknown backend '" + opt.backend + "' (available: seq, omp, cuda)");
}

}  // namespace bh
