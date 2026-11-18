// Centre-of-mass calculation: total mass and centre of mass of every node, bottom-up.
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <omp.h>

#include "bh/stages.hpp"

namespace bh {
namespace {

struct Sum {
    double m = 0, x = 0, y = 0, z = 0;  // mass and mass-weighted position

    void add(double mass, double px, double py, double pz) {
        m += mass;
        x += mass * px;
        y += mass * py;
        z += mass * pz;
    }
    void add_leaf(const Particles& p, const Octree& t, std::int32_t c) {
        for (std::int32_t q = Octree::leaf_head(c); q >= 0; q = t.next[q]) add(p.m[q], p.x[q], p.y[q], p.z[q]);
    }
    // Children are added from their stored (rounded) values, so every
    // centre-of-mass variant produces bit-identical nodes.
    void add_node(const Octree& t, std::int32_t c) { add(t.mass[c], t.comx[c], t.comy[c], t.comz[c]); }

    void write(Octree& t, std::int32_t node) const {
        const double inv = m > 0 ? 1.0 / m : 0.0;
        t.mass[node] = static_cast<real>(m);
        t.comx[node] = static_cast<real>(x * inv);
        t.comy[node] = static_cast<real>(y * inv);
        t.comz[node] = static_cast<real>(z * inv);
    }
};

// The children of `node` must be finished.
void sum_children(const Particles& p, Octree& t, std::int32_t node) {
    Sum s;
    for (int k = 0; k < 8; ++k) {
        const std::int32_t c = t.child[8 * static_cast<std::size_t>(node) + k];
        if (c == Octree::EMPTY) continue;
        if (Octree::is_node(c)) s.add_node(t, c);
        else s.add_leaf(p, t, c);
    }
    s.write(t, node);
}

// Children have larger indices than their parent: a backward scan is bottom-up.
void com_backwards(const Particles& p, Octree& t) {
    for (std::int32_t node = t.num_nodes() - 1; node >= 0; --node) sum_children(p, t, node);
}

// Any node numbering: post-order walk with an explicit stack.
void com_postorder(const Particles& p, Octree& t) {
    struct Frame {
        std::int32_t node;
        int k;
        Sum sum;
    };
    Frame stack[Octree::MAX_DEPTH + 1];
    int top = 0;
    stack[0] = Frame{0, 0, {}};

    while (top >= 0) {
        Frame& f = stack[top];
        if (f.k == 8) {
            f.sum.write(t, f.node);
            if (--top >= 0) stack[top].sum.add_node(t, f.node);
            continue;
        }
        const std::int32_t c = t.child[8 * static_cast<std::size_t>(f.node) + f.k++];
        if (c == Octree::EMPTY) continue;
        if (Octree::is_node(c)) stack[++top] = Frame{c, 0, {}};
        else f.sum.add_leaf(p, t, c);
    }
}

}  // namespace

ComLevelProfile* com_level_profile = nullptr;

void com_seq(const Particles& p, Octree& tree) {
    if (tree.parent_first) com_backwards(p, tree);
    else com_postorder(p, tree);
}

// Level by level from the deepest: the nodes of one level are independent and
// run in parallel, the barrier that ends each loop separates the levels.
void com_level(const Particles& p, Octree& t) {
    constexpr int kLevels = Octree::MAX_DEPTH;  // internal nodes have depth 0 .. MAX_DEPTH - 1
    static std::vector<std::int32_t> order;     // nodes sorted by depth
    const std::int32_t n = t.num_nodes();
    order.resize(static_cast<std::size_t>(n));

    // A node's edge is root edge / 2^depth, exactly. Blank nodes have size 0.
    const int root_exp = std::ilogb(t.size[0]);
    const auto depth = [&](std::int32_t node) { return root_exp - std::ilogb(t.size[node]); };

    std::vector<std::int32_t> start(static_cast<std::size_t>(omp_get_max_threads()) * kLevels, 0);
    std::int32_t begin[kLevels + 1];

    ComLevelProfile* const prof = com_level_profile;
    const auto now = [] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    };

#pragma omp parallel
    {
        // Counting sort by depth; both loops give a thread the same nodes.
        std::int32_t* const shared = &start[static_cast<std::size_t>(omp_get_thread_num()) * kLevels];
        std::int32_t mine[kLevels] = {};
#pragma omp for schedule(static) nowait
        for (std::int32_t node = 0; node < n; ++node)
            if (t.size[node] > 0) ++mine[depth(node)];
        for (int l = 0; l < kLevels; ++l) shared[l] = mine[l];
#pragma omp barrier
#pragma omp single
        {
            std::int32_t total = 0;
            for (int l = 0; l < kLevels; ++l) {
                begin[l] = total;
                for (std::size_t i = l; i < start.size(); i += kLevels) {
                    const std::int32_t count = start[i];
                    start[i] = total;
                    total += count;
                }
            }
            begin[kLevels] = total;
            if (prof) {  // the runtime may give fewer threads than requested
                prof->threads = omp_get_num_threads();
                prof->busy.assign(static_cast<std::size_t>(kLevels) * prof->threads, 0);
                prof->wait = prof->busy;
            }
        }
        for (int l = 0; l < kLevels; ++l) mine[l] = shared[l];
#pragma omp for schedule(static)
        for (std::int32_t node = 0; node < n; ++node)
            if (t.size[node] > 0) order[mine[depth(node)]++] = node;

        for (int l = kLevels - 1; l >= 0; --l) {
            if (begin[l] == begin[l + 1]) continue;
            const double t0 = now();
#pragma omp for schedule(static) nowait
            for (std::int32_t i = begin[l]; i < begin[l + 1]; ++i) sum_children(p, t, order[i]);
            const double t1 = now();
#pragma omp barrier
            if (prof) {
                const std::size_t slot = static_cast<std::size_t>(l) * prof->threads + omp_get_thread_num();
                prof->busy[slot] = t1 - t0;
                prof->wait[slot] = now() - t1;
            }
        }
    }
    if (prof) {
        prof->nodes.resize(kLevels);
        for (int l = 0; l < kLevels; ++l) prof->nodes[l] = begin[l + 1] - begin[l];
    }
}

}  // namespace bh
