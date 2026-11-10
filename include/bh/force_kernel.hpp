// Barnes-Hut tree walk for a single particle.
//
// Every CPU force strategy (sequential, OpenMP static / dynamic / guided,
// costzones) calls this same function, so the strategies differ only in how
// particles are handed to threads.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "bh/octree.hpp"
#include "bh/types.hpp"

namespace bh {

// Computes the acceleration of particle i, writing p.ax/ay/az[i] and the
// number of interactions into p.cost[i]. Reads the tree and the positions and
// masses of other particles only, so calls for different i are independent.
inline void force_on_particle(Particles& p, const Octree& tree, const Params& prm, std::size_t i) {
    const real xi = p.x[i], yi = p.y[i], zi = p.z[i];
    const real theta2 = prm.theta * prm.theta;
    const real eps2 = prm.eps * prm.eps;
    const std::int32_t self = static_cast<std::int32_t>(i);

    real ax = 0, ay = 0, az = 0;
    std::uint32_t count = 0;

    // Depth-first walk. A popped node pushes at most 8 children and internal
    // nodes are at most MAX_DEPTH levels deep, which bounds the stack.
    std::int32_t stack[8 * (Octree::MAX_DEPTH + 1)];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        const std::int32_t* ch = &tree.child[8 * static_cast<std::size_t>(stack[--sp])];
        for (int k = 0; k < 8; ++k) {
            const std::int32_t c = ch[k];
            if (c == Octree::EMPTY) continue;

            if (Octree::is_node(c)) {
                const real dx = tree.comx[c] - xi;
                const real dy = tree.comy[c] - yi;
                const real dz = tree.comz[c] - zi;
                const real d2 = dx * dx + dy * dy + dz * dz;
                const real s = tree.size[c];
                if (s * s < theta2 * d2) {
                    // Far enough: use the cell's total mass at its centre of mass.
                    const real inv = real(1) / std::sqrt(d2 + eps2);
                    const real f = tree.mass[c] * inv * inv * inv;
                    ax += f * dx;
                    ay += f * dy;
                    az += f * dz;
                    ++count;
                } else {
                    stack[sp++] = c;
                }
            } else {
                for (std::int32_t q = Octree::leaf_head(c); q >= 0; q = tree.next[q]) {
                    if (q == self) continue;
                    const real dx = p.x[q] - xi;
                    const real dy = p.y[q] - yi;
                    const real dz = p.z[q] - zi;
                    const real inv = real(1) / std::sqrt(dx * dx + dy * dy + dz * dz + eps2);
                    const real f = p.m[q] * inv * inv * inv;
                    ax += f * dx;
                    ay += f * dy;
                    az += f * dz;
                    ++count;
                }
            }
        }
    }

    p.ax[i] = prm.G * ax;
    p.ay[i] = prm.G * ay;
    p.az[i] = prm.G * az;
    p.cost[i] = count;
}

}  // namespace bh
