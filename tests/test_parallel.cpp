#include "bh/parallel.hpp"
#include "bh/forces.hpp"
#include "bh/backend.hpp"
#include "bh/generator.hpp"
#include "bh/stages.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <omp.h>

namespace {
std::string context;
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(context + ": " + message);
}

void primitives() {
    std::mt19937_64 rng(47);
    for (int team : {1, 2, 3, 8}) {
        omp_set_num_threads(team);
        for (std::size_t n : {0, 1, 2, 7, 257, 4099}) {
            std::vector<std::uint64_t> input(n), keys(n);
            for (std::size_t i = 0; i < n; ++i) {
                input[i] = i % 5 == 0 ? (1ULL << 40) : rng() % 100;
                keys[i] = i % 3 == 0 ? 42 : rng();
            }
            if (n > 1) { keys[0] = 0; keys[1] = ~0ULL; }
            std::vector<std::uint64_t> prefix;
            bh::exclusive_scan_omp(input, prefix);
            require(prefix.size() == n + 1 && prefix[0] == 0, "scan endpoints");
            std::uint64_t sum = 0;
            for (std::size_t i = 0; i < n; ++i) {
                sum += input[i];
                require(prefix[i + 1] == sum, "scan mismatch");
            }
            std::vector<std::int32_t> order, want(n);
            std::iota(want.begin(), want.end(), 0);
            std::stable_sort(want.begin(), want.end(), [&](auto a, auto b) { return keys[a] < keys[b]; });
            bh::radix_order_omp(keys, order);
            require(order == want, "radix mismatch/stability");
            std::fill(keys.begin(), keys.end(), (1ULL << 63) - 1);
            bh::radix_order_omp(keys, order);
            for (std::size_t i = 0; i < n; ++i) require(order[i] == static_cast<int>(i), "equal keys unstable");
            std::fill(input.begin(), input.end(), 0);
            bh::exclusive_scan_omp(input, prefix);
            for (auto v : prefix) require(v == 0, "zero scan mismatch");
        }
    }
    std::vector<std::uint64_t> same{1};
    bool rejected = false;
    try { bh::exclusive_scan_omp(same, same); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "aliased scan was accepted");
}

void partitions() {
    require(bh::costzone_boundaries({0, 8, 9, 10, 16}, 2) ==
            std::vector<std::size_t>({0, 1, 4}), "weighted partition ignored costs");
    require(bh::costzone_boundaries({0, 1, 2, 3, 4, 5}, 3) ==
            std::vector<std::size_t>({0, 1, 3, 5}), "first-step uniform cost split");
    const std::vector<std::vector<std::uint64_t>> inputs = {
        {0}, {0, 0, 0, 0}, {0, 0, 10, 10, 11}, {0, 1, 2, 3},
        {0, (1ULL << 63), (1ULL << 63) + 100}
    };
    for (const auto& prefix : inputs) {
        for (int team : {1, 2, 3, 8}) {
            const auto cuts = bh::costzone_boundaries(prefix, team);
            require(cuts.size() == static_cast<std::size_t>(team + 1), "wrong cut count");
            require(cuts.front() == 0 && cuts.back() == prefix.size() - 1, "cut endpoints");
            std::vector<int> visits(prefix.size() - 1);
            for (int t = 0; t < team; ++t) {
                require(cuts[t] <= cuts[t + 1], "non-monotone cuts");
                for (auto j = cuts[t]; j < cuts[t + 1]; ++j) ++visits[j];
            }
            for (auto v : visits) require(v == 1, "missing or repeated particle");
        }
    }
}

void equal(const bh::Particles& a, const bh::Particles& b) {
    require(a.x == b.x && a.y == b.y && a.z == b.z, "position mismatch");
    require(a.vx == b.vx && a.vy == b.vy && a.vz == b.vz, "velocity mismatch");
    require(a.ax == b.ax && a.ay == b.ay && a.az == b.az, "acceleration mismatch");
    require(a.cost == b.cost && a.m == b.m, "cost/mass mismatch");
}

void force_strategies() {
    for (int team : {1, 3, 8}) {
        for (const std::string build : {"seq", "lock"}) {
            for (const std::string strategy : {"static", "dynamic", "guided", "costzones"}) {
                for (const std::string order : {"morton", "original"}) {
                    if (strategy == "costzones" && order == "original") continue;
                    for (const std::string dist : {"uniform", "plummer", "collision", "degenerate"}) {
                        for (int n : {1, 2, 9, 257}) {
                            bh::Params prm;
                            // Both exact opening and approximate walks are exercised.
                            prm.theta = n == 9 ? 0 : bh::real(0.5);
                            bh::Options opt;
                            opt.backend = "omp"; opt.build = build; opt.force = strategy;
                            opt.threads = team; opt.force_order = order;
                            opt.chunk = n == 257 ? 7 : 0;
                            auto parallel = bh::make_backend(opt, prm);
                            const auto initial = bh::generate(dist, n, 42);
                            auto want = initial;
                            parallel->load(initial);
                            for (int step = 0; step < 4; ++step) {
                                context = build + "/" + strategy + "/" + order + "/" + dist +
                                    "/n=" + std::to_string(n) + "/threads=" + std::to_string(team) +
                                    "/step=" + std::to_string(step);
                                parallel->compute_bbox(); parallel->build_tree(); parallel->compute_com();
                                parallel->compute_forces();
                                // Use the identical tree: lock insertion can change the
                                // summation order inside duplicate-key leaf chains.
                                bh::forces_seq(want, *parallel->host_tree(), prm);
                                bh::Particles got;
                                parallel->store(got);
                                equal(got, want);
                                const auto s = parallel->step_stats();
                                const auto total = std::accumulate(got.cost.begin(), got.cost.end(), std::uint64_t(0));
                                require(total == s.interactions, "work total mismatch");
                                require(std::accumulate(s.thread_interactions.begin(), s.thread_interactions.end(),
                                                        std::uint64_t(0)) == total, "thread total mismatch");
                                require(!s.thread_interactions.empty() && s.thread_interactions.size() <=
                                        static_cast<std::size_t>(team), "actual team mismatch");
                                const double want_lambda = total == 0 ? 1.0 :
                                    double(*std::max_element(s.thread_interactions.begin(), s.thread_interactions.end())) *
                                    s.thread_interactions.size() / double(total);
                                require(s.load_imbalance == want_lambda, "lambda mismatch");
                                require(s.force_order_s >= 0 && s.force_partition_s >= 0 && s.force_walk_s >= 0,
                                        "negative timing");
                                if (n == 1) require(s.load_imbalance == 1, "zero work lambda");
                                const auto kick = step == 0 ? prm.dt / 2 : prm.dt;
                                parallel->integrate(kick, prm.dt);
                                bh::integrate_seq(want, kick, prm.dt);
                                parallel->store(got);
                                equal(got, want);
                            }
                            // Reloading invalidates historical costs and remains reproducible.
                            parallel->load(initial);
                            parallel->compute_bbox(); parallel->build_tree(); parallel->compute_com();
                            parallel->compute_forces();
                        }
                    }
                }
            }
        }
    }
}
}  // namespace

int main() {
    try {
        omp_set_dynamic(0);
        primitives();
        partitions();
        force_strategies();
        // An implementation may shrink teams in dynamic mode.
        omp_set_dynamic(1);
        primitives();
        std::cout << "parallel primitives, partitions, multi-step force equality and metrics passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
