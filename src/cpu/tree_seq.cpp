// Sequential octree construction (one particle inserted at a time) and
// sequential bottom-up mass summarization.
#include <cstddef>
#include <cstdint>

#include "bh/stages.hpp"

namespace bh {

void build_tree_seq(const Particles& p, const BBox& box, Octree& tree) {
    const std::size_t n = p.size();
    tree.reset(n, box);

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

void summarize_seq(const Particles& p, Octree& tree) {
    // Children are always created after their parent, so walking the node
    // array backwards visits every child before its parent.
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
                for (std::int32_t q = Octree::leaf_head(c); q >= 0; q = tree.next[q]) {
                    const double mq = p.m[q];
                    m += mq;
                    sx += mq * p.x[q];
                    sy += mq * p.y[q];
                    sz += mq * p.z[q];
                }
            }
        }
        tree.mass[node] = static_cast<real>(m);
        if (m > 0) {
            tree.comx[node] = static_cast<real>(sx / m);
            tree.comy[node] = static_cast<real>(sy / m);
            tree.comz[node] = static_cast<real>(sz / m);
        } else {
            tree.comx[node] = tree.comy[node] = tree.comz[node] = real(0);
        }
    }
}

}  // namespace bh
