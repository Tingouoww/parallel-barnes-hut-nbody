// CPU backend ("seq" and "omp"): owns the particles and the tree in host
// memory and dispatches each stage to the strategy chosen on the command line.
#include <cstddef>
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
using SummarizeFn = void (*)(const Particles&, Octree&);
using ForceFn = void (*)(Particles&, const Octree&, const Params&);

// ---- Strategy tables: add a new variant here ------------------------------
const std::vector<std::pair<std::string, BuildFn>> kBuild = {
    {"seq", build_tree_seq},
};
const std::vector<std::pair<std::string, ForceFn>> kForce = {
    {"seq", forces_seq},
};
// ---------------------------------------------------------------------------

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
    CpuBackend(const Options& opt, const Params& prm, bool parallel) : prm_(prm) {
        build_ = lookup(kBuild, opt.build, "tree build");
        force_ = lookup(kForce, opt.force, "force");
        summarize_ = summarize_seq;
        bbox_ = parallel ? bbox_omp : bbox_seq;
        integrate_ = parallel ? integrate_omp : integrate_seq;
        if (parallel && opt.threads > 0) omp_set_num_threads(opt.threads);
    }

    void load(const Particles& p) override { p_ = p; }

    void compute_bbox() override { box_ = bbox_(p_); }
    void build_tree() override { build_(p_, box_, tree_); }
    void summarize() override { summarize_(p_, tree_); }
    void compute_forces() override { force_(p_, tree_, prm_); }
    void integrate(real dt_kick, real dt_drift) override { integrate_(p_, dt_kick, dt_drift); }

    void store(Particles& out) override { out = p_; }

    StepStats step_stats() const override {
        StepStats s;
        s.nodes = static_cast<std::uint64_t>(tree_.num_nodes());
        for (std::uint32_t c : p_.cost) s.interactions += c;
        return s;
    }

    const Octree* host_tree() const override { return &tree_; }

private:
    Params prm_;
    Particles p_;
    Octree tree_;
    BBox box_{};

    BBox (*bbox_)(const Particles&) = nullptr;
    BuildFn build_ = nullptr;
    SummarizeFn summarize_ = nullptr;
    ForceFn force_ = nullptr;
    void (*integrate_)(Particles&, real, real) = nullptr;
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
