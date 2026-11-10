// Pointer-free octree shared by every tree builder and every force kernel.
//
// Only internal nodes are stored. Node 0 is the root. Each node has 8 child
// slots, child[8 * node + k], where octant k has bit 0 = upper half in x,
// bit 1 = upper half in y, bit 2 = upper half in z. A slot holds
//
//   EMPTY (-1)        nothing in that octant
//   c >= 0            index of an internal node
//   c <= -2           a leaf; make_leaf(p) / leaf_head(c) convert to and from
//                     the index p of the particle stored there
//
// A leaf normally holds exactly one particle. A cell at depth MAX_DEPTH cannot
// be split any further (21 levels = the resolution of a 63-bit Morton key), so
// particles that still collide there are chained through next[]: start at
// leaf_head(c) and follow next[p] until it is -1.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "bh/types.hpp"

namespace bh {

struct Octree {
    static constexpr std::int32_t EMPTY = -1;
    static constexpr int MAX_DEPTH = 21;  // root is depth 0; leaf cells are at depth <= MAX_DEPTH

    static bool is_node(std::int32_t c) { return c >= 0; }
    static bool is_leaf(std::int32_t c) { return c < EMPTY; }
    static std::int32_t make_leaf(std::int32_t particle) { return -2 - particle; }
    static std::int32_t leaf_head(std::int32_t c) { return -2 - c; }

    // Root cell: a cube centred at (cx, cy, cz) with edge length root_size.
    double cx = 0, cy = 0, cz = 0, root_size = 1;

    // Per-node arrays (structure-of-arrays).
    std::vector<std::int32_t> child;      // 8 slots per node
    std::vector<real> size;               // edge length of the node's cell
    std::vector<real> mass;               // total mass (filled by the summarize stage)
    std::vector<real> comx, comy, comz;   // centre of mass (filled by the summarize stage)

    // Per-particle chain for leaves that hold more than one particle.
    std::vector<std::int32_t> next;

    std::int32_t num_nodes() const { return static_cast<std::int32_t>(size.size()); }

    // Empties the tree, fits the root cube around `box` and creates the root node.
    void reset(std::size_t n_particles, const BBox& box) {
        double extent = 0;
        for (int d = 0; d < 3; ++d) {
            const double e = double(box.max[d]) - double(box.min[d]);
            if (e > extent) extent = e;
        }
        cx = 0.5 * (double(box.min[0]) + double(box.max[0]));
        cy = 0.5 * (double(box.min[1]) + double(box.max[1]));
        cz = 0.5 * (double(box.min[2]) + double(box.max[2]));
        // Slightly enlarged so particles on the boundary are strictly inside.
        root_size = extent > 0 ? extent * (1.0 + 1e-6) : 1.0;

        child.clear();
        size.clear();
        mass.clear();
        comx.clear();
        comy.clear();
        comz.clear();
        next.assign(n_particles, -1);
        new_node(static_cast<real>(root_size));
    }

    // Appends an empty node and returns its index. Not thread-safe.
    std::int32_t new_node(real node_size) {
        const std::int32_t id = num_nodes();
        child.insert(child.end(), 8, EMPTY);
        size.push_back(node_size);
        mass.push_back(real(0));
        comx.push_back(real(0));
        comy.push_back(real(0));
        comz.push_back(real(0));
        return id;
    }
};

// Octant of point (x, y, z) inside a cell centred at (cx, cy, cz).
inline int octant(double x, double y, double z, double cx, double cy, double cz) {
    return int(x >= cx) | (int(y >= cy) << 1) | (int(z >= cz) << 2);
}

// Turns (centre, half edge) of a cell into those of its child in octant k.
inline void enter_child(int k, double& cx, double& cy, double& cz, double& half) {
    half *= 0.5;
    cx += (k & 1) ? half : -half;
    cy += (k & 2) ? half : -half;
    cz += (k & 4) ? half : -half;
}

}  // namespace bh
