#include "bh/generator.hpp"

#include <cmath>
#include <random>
#include <stdexcept>

namespace bh {
namespace {

const double kPi = 3.14159265358979323846;

// The standard distributions differ between platforms, so [0, 1) is derived by hand.
struct Rng {
    std::mt19937_64 engine;
    explicit Rng(std::uint64_t seed) : engine(seed) {}
    double uniform() { return static_cast<double>(engine() >> 11) * (1.0 / 9007199254740992.0); }
};

void random_direction(Rng& rng, double length, double out[3]) {
    const double u = 1.0 - 2.0 * rng.uniform();
    const double phi = 2.0 * kPi * rng.uniform();
    const double s = std::sqrt(1.0 - u * u);
    out[0] = length * s * std::cos(phi);
    out[1] = length * s * std::sin(phi);
    out[2] = length * u;
}

// Plummer sphere (scale radius 1) in particles [first, first + count), sampled
// as in Aarseth, Henon & Wielen (1974) and truncated at kMaxRadius.
void plummer(Particles& p, std::size_t first, std::size_t count, Rng& rng, double mass,
             const double centre[3], const double bulk[3]) {
    const double kMaxRadius = 20.0;
    const double mi = mass / static_cast<double>(count);
    double com[3] = {0, 0, 0}, cov[3] = {0, 0, 0};
    std::vector<double> pos(3 * count), vel(3 * count);

    for (std::size_t j = 0; j < count; ++j) {
        double r;
        do {
            const double u = rng.uniform();
            r = u > 0 ? 1.0 / std::sqrt(std::pow(u, -2.0 / 3.0) - 1.0) : 0.0;
        } while (!(r <= kMaxRadius));
        random_direction(rng, r, &pos[3 * j]);

        // Speed = q * escape speed, with q distributed as q^2 (1 - q^2)^(7/2).
        double q, g;
        do {
            q = rng.uniform();
            g = 0.1 * rng.uniform();
        } while (g > q * q * std::pow(1.0 - q * q, 3.5));
        const double v_escape = std::sqrt(2.0 * mass) * std::pow(1.0 + r * r, -0.25);
        random_direction(rng, q * v_escape, &vel[3 * j]);

        for (int d = 0; d < 3; ++d) {
            com[d] += pos[3 * j + d];
            cov[d] += vel[3 * j + d];
        }
    }

    // Put the centre of mass and its velocity exactly at `centre` and `bulk`.
    for (int d = 0; d < 3; ++d) {
        com[d] /= static_cast<double>(count);
        cov[d] /= static_cast<double>(count);
    }
    for (std::size_t j = 0; j < count; ++j) {
        const std::size_t i = first + j;
        p.x[i] = static_cast<real>(pos[3 * j + 0] - com[0] + centre[0]);
        p.y[i] = static_cast<real>(pos[3 * j + 1] - com[1] + centre[1]);
        p.z[i] = static_cast<real>(pos[3 * j + 2] - com[2] + centre[2]);
        p.vx[i] = static_cast<real>(vel[3 * j + 0] - cov[0] + bulk[0]);
        p.vy[i] = static_cast<real>(vel[3 * j + 1] - cov[1] + bulk[1]);
        p.vz[i] = static_cast<real>(vel[3 * j + 2] - cov[2] + bulk[2]);
        p.m[i] = static_cast<real>(mi);
    }
}

void uniform_cube(Particles& p, Rng& rng) {
    const std::size_t n = p.size();
    for (std::size_t i = 0; i < n; ++i) {
        p.x[i] = static_cast<real>(2.0 * rng.uniform() - 1.0);
        p.y[i] = static_cast<real>(2.0 * rng.uniform() - 1.0);
        p.z[i] = static_cast<real>(2.0 * rng.uniform() - 1.0);
        p.m[i] = static_cast<real>(1.0 / static_cast<double>(n));
    }
}

// Two equal Plummer spheres, 8 scale radii apart with impact parameter 1, on
// a parabolic orbit; the total centre of mass is at rest at the origin.
void collision(Particles& p, Rng& rng) {
    const std::size_t n = p.size();
    const std::size_t n1 = n / 2, n2 = n - n1;
    const double m1 = static_cast<double>(n1) / static_cast<double>(n), m2 = 1.0 - m1;
    const double sep_x = 8.0, sep_y = 1.0;
    const double v_rel = std::sqrt(2.0 / std::sqrt(sep_x * sep_x + sep_y * sep_y));
    const double c1[3] = {-sep_x * m2, -sep_y * m2, 0}, c2[3] = {sep_x * m1, sep_y * m1, 0};
    const double b1[3] = {v_rel * m2, 0, 0}, b2[3] = {-v_rel * m1, 0, 0};
    if (n1 > 0) plummer(p, 0, n1, rng, m1, c1, b1);
    plummer(p, n1, n2, rng, m2, c2, b2);
}

// Test input: a quarter of the particles share three points, a quarter are
// a hair away from another particle, the rest are uniform.
void degenerate(Particles& p, Rng& rng) {
    uniform_cube(p, rng);
    const std::size_t n = p.size();
    for (std::size_t i = 0; i < n; ++i) {
        if (i % 4 == 1) {
            const std::size_t spot = (i / 4) % 3;
            p.x[i] = static_cast<real>(0.25 * static_cast<double>(spot));
            p.y[i] = static_cast<real>(-0.5);
            p.z[i] = static_cast<real>(0.125);
        } else if (i % 4 == 3) {
            p.x[i] = p.x[i - 1] * (real(1) + real(4e-7));
            p.y[i] = p.y[i - 1];
            p.z[i] = p.z[i - 1];
        }
    }
}

}  // namespace

Particles generate(const std::string& dist, std::size_t n, std::uint64_t seed) {
    if (n == 0) throw std::runtime_error("the number of particles must be positive");
    Particles p;
    p.resize(n);
    Rng rng(seed);
    if (dist == "uniform") {
        uniform_cube(p, rng);
    } else if (dist == "plummer") {
        const double zero[3] = {0, 0, 0};
        plummer(p, 0, n, rng, 1.0, zero, zero);
    } else if (dist == "collision") {
        collision(p, rng);
    } else if (dist == "degenerate") {
        degenerate(p, rng);
    } else {
        throw std::runtime_error("unknown distribution '" + dist +
                                 "' (available: uniform, plummer, collision, degenerate)");
    }
    return p;
}

}  // namespace bh
