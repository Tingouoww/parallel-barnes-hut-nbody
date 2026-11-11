#include "bh/validate.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <vector>

namespace bh {

ValidationResult validate(const Particles& p, const Params& prm, double half_kick,
                          std::size_t max_samples, std::size_t energy_max_n) {
    ValidationResult r;
    const std::size_t n = p.size();
    const long long ln = static_cast<long long>(n);
    const double eps2 = double(prm.eps) * double(prm.eps);
    const double G = prm.G;

    // Acceleration error on evenly spaced sample particles.
    const std::size_t k = std::min(n, max_samples);
    r.samples = k;
    std::vector<double> err(k);
    double diff2 = 0, ref2 = 0;
#pragma omp parallel for schedule(dynamic, 8) reduction(+ : diff2, ref2)
    for (long long s = 0; s < static_cast<long long>(k); ++s) {
        const std::size_t i = static_cast<std::size_t>(s) * n / k;
        const double xi = p.x[i], yi = p.y[i], zi = p.z[i];
        double ax = 0, ay = 0, az = 0;
        for (std::size_t j = 0; j < n; ++j) {
            if (j == i) continue;
            const double dx = p.x[j] - xi, dy = p.y[j] - yi, dz = p.z[j] - zi;
            const double inv = 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz + eps2);
            const double f = p.m[j] * inv * inv * inv;
            ax += f * dx;
            ay += f * dy;
            az += f * dz;
        }
        ax *= G;
        ay *= G;
        az *= G;
        const double ex = p.ax[i] - ax, ey = p.ay[i] - ay, ez = p.az[i] - az;
        const double d2 = ex * ex + ey * ey + ez * ez;
        const double a2 = ax * ax + ay * ay + az * az;
        err[s] = a2 > 0 ? std::sqrt(d2 / a2) : (d2 > 0 ? INFINITY : 0.0);
        diff2 += d2;
        ref2 += a2;
    }
    if (k > 0) {
        std::sort(err.begin(), err.end());
        r.err_median = err[k / 2];
        r.err_p99 = err[std::min(k - 1, static_cast<std::size_t>(0.99 * static_cast<double>(k)))];
        r.err_max = err[k - 1];
        r.err_rms = ref2 > 0 ? std::sqrt(diff2 / ref2) : 0.0;
    }

    if (n <= energy_max_n) {
        double kinetic = 0, potential = 0;
#pragma omp parallel for schedule(dynamic, 64) reduction(+ : kinetic, potential)
        for (long long i = 0; i < ln; ++i) {
            const double vx = p.vx[i] + half_kick * p.ax[i];
            const double vy = p.vy[i] + half_kick * p.ay[i];
            const double vz = p.vz[i] + half_kick * p.az[i];
            kinetic += 0.5 * p.m[i] * (vx * vx + vy * vy + vz * vz);
            double u = 0;
            for (long long j = i + 1; j < ln; ++j) {
                const double dx = double(p.x[j]) - p.x[i];
                const double dy = double(p.y[j]) - p.y[i];
                const double dz = double(p.z[j]) - p.z[i];
                u += p.m[j] / std::sqrt(dx * dx + dy * dy + dz * dz + eps2);
            }
            potential -= G * p.m[i] * u;
        }
        r.has_energy = true;
        r.kinetic = kinetic;
        r.potential = potential;
        r.energy = kinetic + potential;
    }
    return r;
}

namespace {

struct Aggregate {
    double m = 0, sx = 0, sy = 0, sz = 0;
};

struct TreeChecker {
    const Particles& p;
    const Octree& t;
    std::vector<char> particle_seen, node_seen;
    std::ostringstream out;
    int problems = 0;

    TreeChecker(const Particles& particles, const Octree& tree)
        : p(particles), t(tree), particle_seen(particles.size(), 0),
          node_seen(static_cast<std::size_t>(tree.num_nodes()), 0) {}

    void fail(const std::string& what) {
        if (problems++ < 8) out << "  " << what << '\n';
    }

    // `prefix` is the leading 3 * depth key bits shared by every particle under `node`.
    Aggregate visit(std::int32_t node, std::uint64_t prefix, int depth) {
        Aggregate sum;
        std::ostringstream id;
        id << "node " << node << " (depth " << depth << ")";

        if (node_seen[node]) {
            fail(id.str() + " is reachable through more than one parent");
            return sum;
        }
        node_seen[node] = 1;
        if (depth >= Octree::MAX_DEPTH) {
            fail(id.str() + " is deeper than MAX_DEPTH allows");
            return sum;
        }
        const double cell = std::ldexp(t.root_size, -depth);
        if (std::fabs(double(t.size[node]) - cell) > 1e-5 * cell) fail(id.str() + " has the wrong cell size");

        const int shift = Octree::ROOT_SHIFT - 3 * depth;
        bool any_child = false;
        for (int k = 0; k < 8; ++k) {
            const std::int32_t c = t.child[8 * static_cast<std::size_t>(node) + k];
            if (c == Octree::EMPTY) continue;
            any_child = true;
            const std::uint64_t child_prefix = prefix << 3 | static_cast<std::uint64_t>(k);

            if (Octree::is_node(c)) {
                if (c >= t.num_nodes()) {
                    fail(id.str() + " points at a node index out of range");
                    continue;
                }
                const Aggregate a = visit(c, child_prefix, depth + 1);
                sum.m += a.m;
                sum.sx += a.sx;
                sum.sy += a.sy;
                sum.sz += a.sz;
                continue;
            }

            std::size_t in_leaf = 0;
            for (std::int32_t q = Octree::leaf_head(c); q >= 0; q = t.next[q]) {
                if (static_cast<std::size_t>(q) >= p.size() || ++in_leaf > p.size()) {
                    fail(id.str() + " has a broken leaf chain");
                    break;
                }
                if (particle_seen[q]) {
                    fail("particle " + std::to_string(q) + " appears in more than one leaf");
                    break;
                }
                particle_seen[q] = 1;
                if (t.key[q] != t.morton_key(p.x[q], p.y[q], p.z[q]))
                    fail("particle " + std::to_string(q) + " has the wrong Morton key");
                else if (t.key[q] >> shift != child_prefix)
                    fail("particle " + std::to_string(q) + " lies outside its leaf cell under " + id.str());
                sum.m += p.m[q];
                sum.sx += double(p.m[q]) * p.x[q];
                sum.sy += double(p.m[q]) * p.y[q];
                sum.sz += double(p.m[q]) * p.z[q];
            }
            if (in_leaf > 1 && shift != 0)
                fail(id.str() + " has a multi-particle leaf above MAX_DEPTH");
        }

        if (!any_child && !(node == 0 && p.size() == 0)) fail(id.str() + " has no children");

        if (std::fabs(t.mass[node] - sum.m) > 1e-4 * sum.m) fail(id.str() + " has the wrong mass");
        if (sum.m > 0) {
            const double ctol = 1e-4 * t.root_size;
            if (std::fabs(t.comx[node] - sum.sx / sum.m) > ctol ||
                std::fabs(t.comy[node] - sum.sy / sum.m) > ctol ||
                std::fabs(t.comz[node] - sum.sz / sum.m) > ctol)
                fail(id.str() + " has the wrong centre of mass");
        }
        return sum;
    }
};

}  // namespace

std::string check_tree(const Particles& p, const Octree& tree) {
    if (tree.num_nodes() < 1) return "  the tree has no root node\n";
    if (static_cast<std::size_t>(tree.num_nodes()) > tree.capacity())
        return "  n_nodes is larger than the node arrays\n";
    if (tree.next.size() != p.size() || tree.key.size() != p.size())
        return "  next[] or key[] does not have one entry per particle\n";

    TreeChecker checker(p, tree);
    checker.visit(0, 0, 0);

    std::size_t missing = 0;
    for (char seen : checker.particle_seen) missing += seen ? 0 : 1;
    if (missing > 0) checker.fail(std::to_string(missing) + " particles are not in the tree");
    // Unreachable nodes must be exactly the blank ones.
    std::size_t orphans = 0;
    for (std::int32_t node = 0; node < tree.num_nodes(); ++node) {
        if (checker.node_seen[node]) continue;
        ++orphans;
        for (int k = 0; k < 8; ++k)
            if (tree.child[8 * static_cast<std::size_t>(node) + k] != Octree::EMPTY) {
                checker.fail("unreachable node " + std::to_string(node) + " is not blank");
                break;
            }
    }
    if (orphans != static_cast<std::size_t>(tree.blank_nodes))
        checker.fail(std::to_string(orphans) + " nodes are not reachable from the root, but blank_nodes is " +
                     std::to_string(tree.blank_nodes));

    if (checker.problems > 8)
        checker.out << "  ... and " << (checker.problems - 8) << " more problems\n";
    return checker.out.str();
}

}  // namespace bh
