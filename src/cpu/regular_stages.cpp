// The two regular, data-parallel stages: bounding box and integration.
#include <cstddef>

#include "bh/stages.hpp"

namespace bh {

BBox bbox_seq(const Particles& p) {
    const std::size_t n = p.size();
    BBox b;
    b.min[0] = b.max[0] = p.x[0];
    b.min[1] = b.max[1] = p.y[0];
    b.min[2] = b.max[2] = p.z[0];
    for (std::size_t i = 1; i < n; ++i) {
        if (p.x[i] < b.min[0]) b.min[0] = p.x[i];
        if (p.x[i] > b.max[0]) b.max[0] = p.x[i];
        if (p.y[i] < b.min[1]) b.min[1] = p.y[i];
        if (p.y[i] > b.max[1]) b.max[1] = p.y[i];
        if (p.z[i] < b.min[2]) b.min[2] = p.z[i];
        if (p.z[i] > b.max[2]) b.max[2] = p.z[i];
    }
    return b;
}

BBox bbox_omp(const Particles& p) {
    const long long n = static_cast<long long>(p.size());
    real x0 = p.x[0], x1 = p.x[0];
    real y0 = p.y[0], y1 = p.y[0];
    real z0 = p.z[0], z1 = p.z[0];
#pragma omp parallel for schedule(static) reduction(min : x0, y0, z0) reduction(max : x1, y1, z1)
    for (long long i = 1; i < n; ++i) {
        if (p.x[i] < x0) x0 = p.x[i];
        if (p.x[i] > x1) x1 = p.x[i];
        if (p.y[i] < y0) y0 = p.y[i];
        if (p.y[i] > y1) y1 = p.y[i];
        if (p.z[i] < z0) z0 = p.z[i];
        if (p.z[i] > z1) z1 = p.z[i];
    }
    return BBox{{x0, y0, z0}, {x1, y1, z1}};
}

void integrate_seq(Particles& p, real dt_kick, real dt_drift) {
    const std::size_t n = p.size();
    for (std::size_t i = 0; i < n; ++i) {
        p.vx[i] += p.ax[i] * dt_kick;
        p.vy[i] += p.ay[i] * dt_kick;
        p.vz[i] += p.az[i] * dt_kick;
        p.x[i] += p.vx[i] * dt_drift;
        p.y[i] += p.vy[i] * dt_drift;
        p.z[i] += p.vz[i] * dt_drift;
    }
}

void integrate_omp(Particles& p, real dt_kick, real dt_drift) {
    const long long n = static_cast<long long>(p.size());
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < n; ++i) {
        p.vx[i] += p.ax[i] * dt_kick;
        p.vy[i] += p.ay[i] * dt_kick;
        p.vz[i] += p.az[i] * dt_kick;
        p.x[i] += p.vx[i] * dt_drift;
        p.y[i] += p.vy[i] * dt_drift;
        p.z[i] += p.vz[i] * dt_drift;
    }
}

}  // namespace bh
