// Bounding box and integration.
#include <algorithm>
#include <cstddef>

#include "bh/stages.hpp"

namespace bh {

BBox bbox_seq(const Particles& p) {
    BBox b{{p.x[0], p.y[0], p.z[0]}, {p.x[0], p.y[0], p.z[0]}};
    for (std::size_t i = 1; i < p.size(); ++i) {
        b.min[0] = std::min(b.min[0], p.x[i]);
        b.max[0] = std::max(b.max[0], p.x[i]);
        b.min[1] = std::min(b.min[1], p.y[i]);
        b.max[1] = std::max(b.max[1], p.y[i]);
        b.min[2] = std::min(b.min[2], p.z[i]);
        b.max[2] = std::max(b.max[2], p.z[i]);
    }
    return b;
}

BBox bbox_omp(const Particles& p) {
    const long long n = static_cast<long long>(p.size());
    real x0 = p.x[0], x1 = x0, y0 = p.y[0], y1 = y0, z0 = p.z[0], z1 = z0;
#pragma omp parallel for schedule(static) reduction(min : x0, y0, z0) reduction(max : x1, y1, z1)
    for (long long i = 1; i < n; ++i) {
        x0 = std::min(x0, p.x[i]);
        x1 = std::max(x1, p.x[i]);
        y0 = std::min(y0, p.y[i]);
        y1 = std::max(y1, p.y[i]);
        z0 = std::min(z0, p.z[i]);
        z1 = std::max(z1, p.z[i]);
    }
    return BBox{{x0, y0, z0}, {x1, y1, z1}};
}

static inline void kick_drift(Particles& p, long long i, real dt_kick, real dt_drift) {
    p.vx[i] += p.ax[i] * dt_kick;
    p.vy[i] += p.ay[i] * dt_kick;
    p.vz[i] += p.az[i] * dt_kick;
    p.x[i] += p.vx[i] * dt_drift;
    p.y[i] += p.vy[i] * dt_drift;
    p.z[i] += p.vz[i] * dt_drift;
}

void integrate_seq(Particles& p, real dt_kick, real dt_drift) {
    const long long n = static_cast<long long>(p.size());
    for (long long i = 0; i < n; ++i) kick_drift(p, i, dt_kick, dt_drift);
}

void integrate_omp(Particles& p, real dt_kick, real dt_drift) {
    const long long n = static_cast<long long>(p.size());
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < n; ++i) kick_drift(p, i, dt_kick, dt_drift);
}

}  // namespace bh
