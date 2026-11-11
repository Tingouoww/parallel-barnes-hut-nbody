// Sequential tree construction and centre-of-mass calculation.
#include <cstddef>
#include <cstdint>

#include "bh/stages.hpp"

namespace bh {

void build_tree_seq(const Particles& p, const BBox& box, Octree& tree) {
    const std::size_t n = p.size();
    tree.reset(n, box);
    tree.parent_first = true;

    for (std::size_t i = 0; i < n; ++i) {
        const std::int32_t leaf_i = Octree::make_leaf(static_cast<std::int32_t>(i));
        const std::uint64_t key = tree.key[i] = tree.morton_key(p.x[i], p.y[i], p.z[i]);
        std::int32_t node = 0;
        double size = tree.root_size;

        for (int shift = Octree::ROOT_SHIFT;; shift -= 3) {
            const std::size_t slot = 8 * static_cast<std::size_t>(node) + ((key >> shift) & 7);
            const std::int32_t c = tree.child[slot];
            if (c == Octree::EMPTY) {
                tree.child[slot] = leaf_i;
                break;
            }
            size *= 0.5;
            if (Octree::is_node(c)) {
                node = c;
                continue;
            }
            const std::int32_t q = Octree::leaf_head(c);
            if (shift == 0) {  // the cell is at MAX_DEPTH: chain i in front of the leaf
                tree.next[i] = q;
                tree.child[slot] = leaf_i;
                break;
            }
            // Split: a new node takes the old particle, then i keeps descending into it.
            node = tree.new_node(static_cast<real>(size));
            tree.child[slot] = node;
            tree.child[8 * static_cast<std::size_t>(node) + ((tree.key[q] >> (shift - 3)) & 7)] = c;
        }
    }
}

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

// Children have larger indices than their parent: a backward scan is bottom-up.
void com_backwards(const Particles& p, Octree& t) {
    for (std::int32_t node = t.num_nodes() - 1; node >= 0; --node) {
        Sum s;
        for (int k = 0; k < 8; ++k) {
            const std::int32_t c = t.child[8 * static_cast<std::size_t>(node) + k];
            if (c == Octree::EMPTY) continue;
            if (Octree::is_node(c)) s.add_node(t, c);
            else s.add_leaf(p, t, c);
        }
        s.write(t, node);
    }
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

void com_seq(const Particles& p, Octree& tree) {
    if (tree.parent_first) com_backwards(p, tree);
    else com_postorder(p, tree);
}

}  // namespace bh
