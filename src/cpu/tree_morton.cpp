// Tree construction from sorted Morton keys, without locks. It builds the
// octree directly; tree_karras.cpp goes through a binary radix tree instead.
//
// Once the keys are sorted, the particles under a node are a contiguous run,
// and the number of leading octants two keys share is the depth of their
// deepest common node. A node of depth d exists for every maximal run of two
// or more keys that share their first d octants. Nodes are numbered by the
// first particle `a` of their run: those starting at `a` have depths
// before(a) + 1 .. shared(a), so a prefix sum over these counts gives every
// node its index, and all nodes can then be filled independently.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "bh/parallel.hpp"
#include "bh/stages.hpp"

namespace bh {
namespace {

constexpr int kDeepest = Octree::MAX_DEPTH - 1;  // depth of the deepest internal nodes

// Largest e in [s, hi] whose key has the same first `depth` octants as key[s].
std::int32_t run_end(const std::uint64_t* key, std::int32_t s, std::int32_t hi, int depth) {
    const int shift = 63 - 3 * depth;
    const auto same = [&](std::int64_t j) { return j <= hi && (key[j] ^ key[s]) >> shift == 0; };
    std::int64_t e = s, step = 1;
    while (same(e + step)) {
        e += step;
        step *= 2;
    }
    while ((step /= 2) > 0)
        if (same(e + step)) e += step;
    return static_cast<std::int32_t>(e);
}

}  // namespace

bool morton_order(const Particles& p, const BBox& box, Octree& t, std::vector<std::int32_t>& order,
                  std::vector<std::uint64_t>& sorted) {
    const std::int32_t n = static_cast<std::int32_t>(p.size());
    t.set_root(box);
    t.key.resize(p.size());
    sorted.resize(p.size());
#pragma omp parallel for schedule(static)
    for (std::int32_t i = 0; i < n; ++i) t.key[i] = t.morton_key(p.x[i], p.y[i], p.z[i]);
    int unsorted = 0;
#pragma omp parallel for schedule(static) reduction(| : unsorted)
    for (std::int32_t i = 1; i < n; ++i) unsorted |= t.key[i] < t.key[i - 1];
    if (unsorted) radix_order_omp(t.key, order);
    else order.resize(p.size());
#pragma omp parallel for schedule(static)
    for (std::int32_t a = 0; a < n; ++a) {
        if (!unsorted) order[a] = a;
        sorted[a] = t.key[order[a]];
    }
    return !unsorted;
}

void build_tree_morton(const Particles& p, const BBox& box, Octree& t) {
    static std::vector<std::uint64_t> key;   // sorted keys
    static std::vector<std::int32_t> order;  // order[a]: particle with the a-th smallest key
    static std::vector<std::uint64_t> count, first;  // nodes that start at a, index of the first of them
    const std::int32_t n = static_cast<std::int32_t>(p.size());
    t.parent_first = true;
    if (n == 0) {
        t.reset(0, box);
        return;
    }
    t.next.resize(p.size());
    count.resize(p.size());
    morton_order(p, box, t, order, key);

    // Octants shared by the sorted keys a and a + 1, and by a - 1 and a.
    const auto shared = [&](std::int32_t a) {
        if (a + 1 >= n) return 0;
        const std::uint64_t diff = key[a] ^ key[a + 1];
        return diff == 0 ? kDeepest : std::min(kDeepest, (__builtin_clzll(diff) - 1) / 3);
    };
    const auto before = [&](std::int32_t a) { return a > 0 ? shared(a - 1) : -1; };

#pragma omp parallel for schedule(static)
    for (std::int32_t a = 0; a < n; ++a) count[a] = std::max(0, shared(a) - before(a));
    exclusive_scan_omp(count, first);
    const auto total = static_cast<std::int32_t>(first[n]);
    t.reserve_nodes(static_cast<std::size_t>(total));
    t.n_nodes = total;
    t.blank_nodes = 0;

#pragma omp parallel for schedule(static)
    for (std::int32_t a = 0; a < n; ++a) {
        const int above = before(a), deepest = shared(a);
        std::int32_t end = n - 1;
        for (int d = above + 1; d <= deepest; ++d) {
            const auto node = static_cast<std::int32_t>(first[a]) + d - above - 1;
            std::int32_t* const child = &t.child[8 * static_cast<std::size_t>(node)];
            for (int k = 0; k < 8; ++k) child[k] = Octree::EMPTY;
            t.size[node] = static_cast<real>(std::ldexp(t.root_size, -d));
            end = run_end(key.data(), a, end, d);

            // One child per run of keys that also share the next octant.
            const int shift = Octree::ROOT_SHIFT - 3 * d;
            for (std::int32_t s = a, e; s <= end; s = e + 1) {
                e = run_end(key.data(), s, end, d + 1);
                std::int32_t& slot = child[(key[s] >> shift) & 7];
                if (e == s || d == kDeepest) {
                    slot = Octree::make_leaf(order[s]);
                    for (std::int32_t j = s; j < e; ++j) t.next[order[j]] = order[j + 1];
                    t.next[order[e]] = -1;
                } else {
                    slot = static_cast<std::int32_t>(first[s]) + d - before(s);  // depth d + 1, starting at s
                }
            }
        }
    }
}

}  // namespace bh
