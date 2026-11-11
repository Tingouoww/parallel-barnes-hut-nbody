// Octree stored as arrays. Only internal nodes are stored; node 0 is the root.
//
// child[8 * node + k] is EMPTY, an internal node index (>= 0), or a leaf
// (<= -2, holding particle leaf_head(c)). The octant k of a particle comes
// from its Morton key: a node at depth d uses (key >> (ROOT_SHIFT - 3 * d)) & 7,
// where bit 0 / 1 / 2 is the upper half in x / y / z.
//
// A leaf holds one particle, except in a cell at MAX_DEPTH, which cannot be
// split: its particles (equal keys) are chained through next[] (-1 ends it).
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "bh/types.hpp"

namespace bh {

struct Octree {
    static constexpr std::int32_t EMPTY = -1;
    static constexpr int MAX_DEPTH = 21;  // bits per axis of a Morton key
    static constexpr int ROOT_SHIFT = 3 * (MAX_DEPTH - 1);

    static bool is_node(std::int32_t c) { return c >= 0; }
    static std::int32_t make_leaf(std::int32_t particle) { return -2 - particle; }
    static std::int32_t leaf_head(std::int32_t c) { return -2 - c; }

    double x0 = 0, y0 = 0, z0 = 0, root_size = 1;  // root cube: minimum corner and edge length

    // Per-node arrays, sized to the capacity; the nodes are [0, n_nodes).
    std::vector<std::int32_t> child;
    std::vector<real> size;                    // edge length of the cell
    std::vector<real> mass, comx, comy, comz;  // filled by the centre-of-mass stage

    // Per-particle arrays.
    std::vector<std::int32_t> next;
    std::vector<std::uint64_t> key;  // Morton key within the root cube, set by the build

    std::int32_t n_nodes = 0;
    std::int32_t blank_nodes = 0;  // unused, unreachable nodes in [0, n_nodes) left by a parallel build
    bool parent_first = false;     // every child has a larger index than its parent

    std::int32_t num_nodes() const { return n_nodes; }
    std::size_t capacity() const { return size.size(); }

    std::uint64_t morton_key(double x, double y, double z) const {
        const double scale = double(1 << MAX_DEPTH) / root_size;
        return spread((x - x0) * scale) | spread((y - y0) * scale) << 1 | spread((z - z0) * scale) << 2;
    }

    void reserve_nodes(std::size_t cap) {
        if (cap <= capacity()) return;
        child.resize(8 * cap, EMPTY);
        for (auto* v : {&size, &mass, &comx, &comy, &comz}) v->resize(cap, real(0));
    }

    void set_root(const BBox& box) {
        double extent = 0;
        for (int d = 0; d < 3; ++d) {
            const double e = double(box.max[d]) - double(box.min[d]);
            if (e > extent) extent = e;
        }
        root_size = extent > 0 ? extent * (1.0 + 1e-6) : 1.0;
        x0 = 0.5 * (double(box.min[0]) + double(box.max[0]) - root_size);
        y0 = 0.5 * (double(box.min[1]) + double(box.max[1]) - root_size);
        z0 = 0.5 * (double(box.min[2]) + double(box.max[2]) - root_size);
    }

    // Empties the tree and creates the root. Not thread-safe, like new_node.
    void reset(std::size_t n_particles, const BBox& box) {
        set_root(box);
        n_nodes = 0;
        blank_nodes = 0;
        next.assign(n_particles, -1);
        key.resize(n_particles);
        new_node(static_cast<real>(root_size));
    }

    std::int32_t new_node(real node_size) {
        if (static_cast<std::size_t>(n_nodes) == capacity())
            reserve_nodes(capacity() < 512 ? 1024 : 2 * capacity());
        const std::int32_t id = n_nodes++;
        for (int k = 0; k < 8; ++k) child[8 * static_cast<std::size_t>(id) + k] = EMPTY;
        size[id] = node_size;
        return id;
    }

private:
    // Clamps to MAX_DEPTH bits and moves bit b to bit 3b.
    static std::uint64_t spread(double coordinate) {
        const double top = double((1 << MAX_DEPTH) - 1);
        std::uint64_t v = static_cast<std::uint64_t>(coordinate < 0 ? 0 : coordinate > top ? top : coordinate);
        v = (v | v << 32) & 0x001f00000000ffffULL;
        v = (v | v << 16) & 0x001f0000ff0000ffULL;
        v = (v | v << 8) & 0x100f00f00f00f00fULL;
        v = (v | v << 4) & 0x10c30c30c30c30c3ULL;
        v = (v | v << 2) & 0x1249249249249249ULL;
        return v;
    }
};

}  // namespace bh
