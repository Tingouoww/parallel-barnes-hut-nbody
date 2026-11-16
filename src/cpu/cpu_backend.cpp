// CPU backend ("seq" and "omp"): keeps the particles and the tree in host
// memory and runs each stage with the strategy chosen on the command line.
#include <memory>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <omp.h>

#include "bh/backend.hpp"
#include "bh/stages.hpp"
#include "bh/forces.hpp"
#include "bh/timer.hpp"

namespace bh {
namespace {

using BuildFn = void (*)(const Particles&, const BBox&, Octree&);
using ForceFn = ForceMetrics (*)(Particles&, const Octree&, const Params&, const ForceConfig&);

ForceMetrics measured_seq(Particles& p, const Octree& t, const Params& prm, const ForceConfig&) {
    const double start = steady_seconds();
    forces_seq(p, t, prm);
    ForceMetrics m;
    m.walk_s = steady_seconds() - start;
    m.thread_work = {std::accumulate(p.cost.begin(), p.cost.end(), std::uint64_t(0))};
    return m;
}

// Strategy tables: the name is the value of --build / --force.
const std::vector<std::pair<std::string, BuildFn>> kBuild = {
    {"seq", build_tree_seq},
    {"lock", build_tree_lock},
};
const std::vector<std::pair<std::string, ForceFn>> kForce = {
    {"seq", measured_seq},
    {"static", forces_omp_static},
    {"dynamic", forces_omp_dynamic},
    {"guided", forces_omp_guided},
    {"costzones", forces_omp_costzones},
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
        force_config_.order = opt.force_order;
        force_config_.chunk = opt.chunk;
        if (opt.force_order != "morton" && opt.force_order != "original")
            throw std::runtime_error("--force-order must be morton or original");
        if (opt.chunk < 0) throw std::runtime_error("--chunk must be nonnegative");
        if (opt.force == "costzones" && opt.force_order != "morton")
            throw std::runtime_error("costzones requires --force-order morton");
    }

    void load(const Particles& p) override {
        p_ = p;
        force_config_.have_history = false;
        metrics_ = {};
    }
    void store(Particles& out) override { out = p_; }

    void compute_bbox() override { box_ = parallel_ ? bbox_omp(p_) : bbox_seq(p_); }
    void build_tree() override { build_(p_, box_, tree_); }
    void compute_com() override { com_seq(p_, tree_); }
    void compute_forces() override {
        metrics_ = force_(p_, tree_, prm_, force_config_);
        force_config_.have_history = true;
    }
    void integrate(real dt_kick, real dt_drift) override {
        if (parallel_) integrate_omp(p_, dt_kick, dt_drift);
        else integrate_seq(p_, dt_kick, dt_drift);
    }

    StepStats step_stats() const override {
        StepStats s;
        s.nodes = static_cast<std::uint64_t>(tree_.num_nodes() - tree_.blank_nodes);
        s.thread_interactions = metrics_.thread_work;
        for (auto c : s.thread_interactions) s.interactions += c;
        if (s.interactions > 0)
            s.load_imbalance = double(*std::max_element(s.thread_interactions.begin(), s.thread_interactions.end()))
                               * s.thread_interactions.size() / double(s.interactions);
        s.force_order_s = metrics_.order_s;
        s.force_partition_s = metrics_.partition_s;
        s.force_walk_s = metrics_.walk_s;
        return s;
    }

    const Octree* host_tree() const override { return &tree_; }

private:
    Params prm_;
    bool parallel_;
    BuildFn build_;
    ForceFn force_;
    ForceConfig force_config_;
    ForceMetrics metrics_;
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
