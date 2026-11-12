// Tree walk for one particle, shared by every CPU force strategy.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "bh/octree.hpp"
#include "bh/types.hpp"

namespace bh {

// Writes p.ax/ay/az[i] and p.cost[i]. Calls for different i are independent.
//
// A node is replaced by its total mass at its centre of mass when
// size / distance < theta and it does not contain particle i; nodes that
// contain i are always opened.
template <bool Reordered>
inline void force_on_particle_impl(Particles& p, const Octree& tree, const Params& prm,
                                  std::size_t i, std::int32_t self,
                                  const std::int32_t* original_to_sorted) {
    const real xi = p.x[i], yi = p.y[i], zi = p.z[i];
    const real theta2 = prm.theta * prm.theta, eps2 = prm.eps * prm.eps;
    const std::uint64_t key = tree.key[self];
    real ax = 0, ay = 0, az = 0;
    std::uint32_t count = 0;
    std::int32_t stack[8 * (Octree::MAX_DEPTH + 1)];
    int sp = 0;

    auto add = [&](real mass, real dx, real dy, real dz, real d2) {
        const real inv = real(1) / std::sqrt(d2 + eps2);
        const real f = mass * inv * inv * inv;
        ax += f * dx;
        ay += f * dy;
        az += f * dz;
        ++count;
    };
    // A child that does not contain i: use it, or push it to be opened.
    auto visit = [&](std::int32_t c) {
        if (c == Octree::EMPTY) return;
        if (Octree::is_node(c)) {
            const real dx = tree.comx[c] - xi, dy = tree.comy[c] - yi, dz = tree.comz[c] - zi;
            const real d2 = dx * dx + dy * dy + dz * dz;
            if (tree.size[c] * tree.size[c] < theta2 * d2) add(tree.mass[c], dx, dy, dz, d2);
            else stack[sp++] = c;
        } else {
            for (std::int32_t q = Octree::leaf_head(c); q >= 0; q = tree.next[q]) {
                if (q == self) continue;
                std::int32_t source = q;
                if constexpr (Reordered) source = original_to_sorted[q];
                const real dx = p.x[source] - xi, dy = p.y[source] - yi, dz = p.z[source] - zi;
                add(p.m[source], dx, dy, dz, dx * dx + dy * dy + dz * dz);
            }
        }
    };

    // `node` follows the chain of nodes that contain i, from the root down.
    std::int32_t node = 0;
    for (int shift = Octree::ROOT_SHIFT; node >= 0; shift -= 3) {
        const std::int32_t* ch = &tree.child[8 * static_cast<std::size_t>(node)];
        const int own = (key >> shift) & 7;
        node = Octree::is_node(ch[own]) ? ch[own] : -1;
        for (int k = 0; k < 8; ++k)
            if (k != own || node < 0) visit(ch[k]);
        while (sp > 0) {
            const std::int32_t* sub = &tree.child[8 * static_cast<std::size_t>(stack[--sp])];
            for (int k = 0; k < 8; ++k) visit(sub[k]);
        }
    }

    p.ax[i] = prm.G * ax;
    p.ay[i] = prm.G * ay;
    p.az[i] = prm.G * az;
    p.cost[i] = count;
}

inline void force_on_particle(Particles& p, const Octree& tree, const Params& prm, std::size_t i) {
    force_on_particle_impl<false>(p, tree, prm, i, static_cast<std::int32_t>(i), nullptr);
}

// Force fields are physically packed in Morton order, but tree leaf links
// and keys still use original IDs. Only source leaf loads need remapping.
inline void force_on_sorted_particle(Particles& sorted, const Octree& tree, const Params& prm,
                                     std::size_t target, std::int32_t original,
                                     const std::int32_t* original_to_sorted) {
    force_on_particle_impl<true>(sorted, tree, prm, target, original, original_to_sorted);
}

}  // namespace bh
