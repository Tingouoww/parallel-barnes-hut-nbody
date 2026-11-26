// Tree construction through a binary radix tree (Karras 2012).
//
// The n sorted keys are the leaves of a binary radix tree with n - 1 internal
// nodes, node 0 being the root. Each internal node finds its key range and its
// split on its own, so all of them are built in parallel. A binary node whose
// keys share a prefix of L bits, under a parent sharing P bits, holds the octree
// nodes of depths P / 3 + 1 .. L / 3 (from depth 0 for the root). A prefix sum
// over these counts numbers the octree nodes; each is then linked under the
// deepest octree node of its nearest binary ancestor that holds any.
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

// Child slot, in a node of depth d, of the cell holding key k.
inline std::size_t slot(std::int32_t node, std::uint64_t k, int d) {
    return 8 * static_cast<std::size_t>(node) + ((k >> (Octree::ROOT_SHIFT - 3 * d)) & 7);
}

}  // namespace

void build_tree_karras(const Particles& p, const BBox& box, Octree& t) {
    static std::vector<std::uint64_t> key, count, first;
    static std::vector<std::int32_t> order, parent, leaf_parent, prefix;
    const std::int32_t n = static_cast<std::int32_t>(p.size());
    t.parent_first = false;
    if (n == 0) {
        t.reset(0, box);
        return;
    }
    t.next.resize(p.size());
    morton_order(p, box, t, order, key);
    if (n == 1) {
        t.reset(1, box);
        t.child[slot(0, t.key[0], 0)] = Octree::make_leaf(0);
        return;
    }
    const std::int32_t m = n - 1;
    prefix.resize(m);
    parent.resize(m);
    count.resize(m);
    leaf_parent.resize(p.size());

    // Bits shared by sorted keys i and j; equal keys are told apart by position.
    const auto delta = [&](std::int64_t i, std::int64_t j) {
        if (j < 0 || j >= n) return -1;
        const std::uint64_t diff = key[i] ^ key[j];
        return diff ? __builtin_clzll(diff) - 1 : 63 + __builtin_clz(static_cast<std::uint32_t>(i ^ j));
    };
    const auto deepest = [&](std::int32_t c) { return std::min(prefix[c] / 3, kDeepest); };

    parent[0] = -1;
#pragma omp parallel for schedule(static)
    for (std::int32_t i = 0; i < m; ++i) {
        const int d = delta(i, i + 1) > delta(i, i - 1) ? 1 : -1;
        const int lower = delta(i, i - d);
        std::int64_t far = 2, len = 0;
        while (delta(i, i + far * d) > lower) far *= 2;
        for (std::int64_t step = far / 2; step >= 1; step /= 2)
            if (delta(i, i + (len + step) * d) > lower) len += step;
        const std::int64_t j = i + len * d;
        prefix[i] = delta(i, j);
        std::int64_t s = 0;
        for (std::int64_t div = 2;; div *= 2) {
            const std::int64_t step = (len + div - 1) / div;
            if (delta(i, i + (s + step) * d) > prefix[i]) s += step;
            if (step == 1) break;
        }
        const auto split = static_cast<std::int32_t>(i + s * d + std::min(d, 0));
        (std::min<std::int64_t>(i, j) == split ? leaf_parent : parent)[split] = i;
        (std::max<std::int64_t>(i, j) == split + 1 ? leaf_parent : parent)[split + 1] = i;
    }

#pragma omp parallel for schedule(static)
    for (std::int32_t c = 0; c < m; ++c) count[c] = deepest(c) - (c == 0 ? -1 : deepest(parent[c]));
    exclusive_scan_omp(count, first);
    const auto total = static_cast<std::int32_t>(first[m]);
    t.reserve_nodes(static_cast<std::size_t>(total));
    t.n_nodes = total;
    t.blank_nodes = 0;

    const auto top = [&](std::int32_t c) { return deepest(c) - static_cast<int>(count[c]) + 1; };
    const auto bottom = [&](std::int32_t c) { return static_cast<std::int32_t>(first[c] + count[c]) - 1; };
    const auto holder = [&](std::int32_t c) {  // nearest binary node, c or above, that holds octree nodes
        while (count[c] == 0) c = parent[c];
        return c;
    };

#pragma omp parallel for schedule(static)
    for (std::int32_t c = 0; c < m; ++c)
        for (int d = top(c); d <= deepest(c); ++d) {
            const auto node = static_cast<std::int32_t>(first[c]) + d - top(c);
            std::fill_n(&t.child[8 * static_cast<std::size_t>(node)], 8, Octree::EMPTY);
            t.size[node] = static_cast<real>(std::ldexp(t.root_size, -d));
        }

    // Key c lies in the range of binary node c, so it carries the node's prefix.
#pragma omp parallel for schedule(static)
    for (std::int32_t c = 0; c < m; ++c) {
        if (count[c] == 0) continue;
        const auto node = static_cast<std::int32_t>(first[c]);
        for (int d = top(c); d < deepest(c); ++d) t.child[slot(node + d - top(c), key[c], d)] = node + d - top(c) + 1;
        if (c > 0) {
            const std::int32_t above = holder(parent[c]);
            t.child[slot(bottom(above), key[c], deepest(above))] = node;
        }
    }

    // Equal keys share one cell at MAX_DEPTH: the first holds the slot, the rest are chained.
#pragma omp parallel for schedule(static)
    for (std::int32_t s = 0; s < n; ++s) {
        t.next[order[s]] = s + 1 < n && key[s + 1] == key[s] ? order[s + 1] : -1;
        if (s > 0 && key[s - 1] == key[s]) continue;
        const std::int32_t above = holder(leaf_parent[s]);
        t.child[slot(bottom(above), key[s], deepest(above))] = Octree::make_leaf(order[s]);
    }
}

}  // namespace bh
