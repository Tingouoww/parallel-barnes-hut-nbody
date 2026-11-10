// Sequential octree construction (one particle inserted at a time) and
// sequential bottom-up mass summarization.
#include <cstddef>
#include <cstdint>

#include "bh/stages.hpp"

namespace bh {

void build_tree_seq(const Particles& p, const BBox& box, Octree& tree) {
    const std::size_t n = p.size();
    tree.reset(n, box);
    tree.parent_first = true;  // nodes are appended, so a child always follows its parent

    for (std::size_t i = 0; i < n; ++i) {
        const double x = p.x[i], y = p.y[i], z = p.z[i];
        // (cx, cy, cz, half) describe the cell of `node`, which is at `depth`.
        double cx = tree.cx, cy = tree.cy, cz = tree.cz, half = 0.5 * tree.root_size;
        std::int32_t node = 0;
        int depth = 0;

        for (;;) {
            const int k = octant(x, y, z, cx, cy, cz);
            const std::size_t slot = 8 * static_cast<std::size_t>(node) + k;
            const std::int32_t c = tree.child[slot];

            if (c == Octree::EMPTY) {
                tree.child[slot] = Octree::make_leaf(static_cast<std::int32_t>(i));
                break;
            }

            enter_child(k, cx, cy, cz, half);  // now the cell of child k
            if (Octree::is_node(c)) {
                node = c;
                ++depth;
                continue;
            }

            // The octant already holds a particle.
            if (depth + 1 >= Octree::MAX_DEPTH) {
                // The child cell cannot be split: chain the particles in one leaf.
                tree.next[i] = Octree::leaf_head(c);
                tree.child[slot] = Octree::make_leaf(static_cast<std::int32_t>(i));
                break;
            }

            // Split: replace the leaf by a new node, move the old particle into
            // it, then keep descending with particle i.
            const std::int32_t q = Octree::leaf_head(c);
            const std::int32_t fresh = tree.new_node(static_cast<real>(2 * half));
            tree.child[slot] = fresh;
            tree.child[8 * static_cast<std::size_t>(fresh) + octant(p.x[q], p.y[q], p.z[q], cx, cy, cz)] = c;
            node = fresh;
            ++depth;
        }
    }
}

namespace {

// Sums the mass and mass-weighted position of the particles in a leaf.
inline void add_leaf(const Particles& p, const Octree& tree, std::int32_t c, double& m, double& sx,
                     double& sy, double& sz) {
    for (std::int32_t q = Octree::leaf_head(c); q >= 0; q = tree.next[q]) {
        const double mq = p.m[q];
        m += mq;
        sx += mq * p.x[q];
        sy += mq * p.y[q];
        sz += mq * p.z[q];
    }
}

// For trees whose children all have a larger index than their parent:
// walking the node array backwards visits every child before its parent.
void summarize_backwards(const Particles& p, Octree& tree) {
    for (std::int32_t node = tree.num_nodes() - 1; node >= 0; --node) {
        double m = 0, sx = 0, sy = 0, sz = 0;
        const std::int32_t* ch = &tree.child[8 * static_cast<std::size_t>(node)];
        for (int k = 0; k < 8; ++k) {
            const std::int32_t c = ch[k];
            if (c == Octree::EMPTY) continue;
            if (Octree::is_node(c)) {
                const double mc = tree.mass[c];
                m += mc;
                sx += mc * tree.comx[c];
                sy += mc * tree.comy[c];
                sz += mc * tree.comz[c];
            } else {
                add_leaf(p, tree, c, m, sx, sy, sz);
            }
        }
        const double inv = m > 0 ? 1.0 / m : 0.0;
        tree.mass[node] = static_cast<real>(m);
        tree.comx[node] = static_cast<real>(sx * inv);
        tree.comy[node] = static_cast<real>(sy * inv);
        tree.comz[node] = static_cast<real>(sz * inv);
    }
}

}  // namespace

void summarize_seq(const Particles& p, Octree& tree) {
    if (tree.parent_first) {
        summarize_backwards(p, tree);
        return;
    }

    // Any other node numbering: post-order walk with an explicit stack, where
    // a node is finished once all of its children are.
    struct Frame {
        std::int32_t node;
        int k;                 // next child slot to look at
        double m, sx, sy, sz;  // mass and mass-weighted position gathered so far
    };
    Frame stack[Octree::MAX_DEPTH + 1];
    int top = 0;
    stack[0] = Frame{0, 0, 0, 0, 0, 0};

    while (top >= 0) {
        Frame& f = stack[top];
        if (f.k == 8) {
            tree.mass[f.node] = static_cast<real>(f.m);
            const double inv = f.m > 0 ? 1.0 / f.m : 0.0;
            tree.comx[f.node] = static_cast<real>(f.sx * inv);
            tree.comy[f.node] = static_cast<real>(f.sy * inv);
            tree.comz[f.node] = static_cast<real>(f.sz * inv);
            if (--top >= 0) {
                // Add the stored (rounded) values, exactly as the backward
                // scan does, so both orders give bit-identical trees.
                const std::int32_t c = f.node;
                const double mc = tree.mass[c];
                Frame& parent = stack[top];
                parent.m += mc;
                parent.sx += mc * tree.comx[c];
                parent.sy += mc * tree.comy[c];
                parent.sz += mc * tree.comz[c];
            }
            continue;
        }
        const std::int32_t c = tree.child[8 * static_cast<std::size_t>(f.node) + f.k++];
        if (c == Octree::EMPTY) continue;
        if (Octree::is_node(c)) {
            stack[++top] = Frame{c, 0, 0, 0, 0, 0};
        } else {
            add_leaf(p, tree, c, f.m, f.sx, f.sy, f.sz);
        }
    }
}

}  // namespace bh
