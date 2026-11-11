// Sequential tree construction.
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

}  // namespace bh
