#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace bh {

#ifdef BH_DOUBLE
using real = double;
#define BH_PRECISION_NAME "double"
#else
using real = float;
#define BH_PRECISION_NAME "float"
#endif

struct Params {
    real theta = real(0.5);   // a cell is accepted when size / distance < theta
    real eps   = real(0.01);  // softening length
    real G     = real(1.0);
    real dt    = real(0.01);
};

struct Particles {
    std::vector<real> x, y, z, vx, vy, vz, ax, ay, az, m;
    std::vector<std::uint32_t> cost;  // interactions per particle in the last force stage

    std::size_t size() const { return x.size(); }

    void resize(std::size_t n) {
        for (auto* v : {&x, &y, &z, &vx, &vy, &vz, &ax, &ay, &az, &m}) v->assign(n, real(0));
        cost.assign(n, 0);
    }
};

struct BBox {
    real min[3], max[3];
};

}  // namespace bh
