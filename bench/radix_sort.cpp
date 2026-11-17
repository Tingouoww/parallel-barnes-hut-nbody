// Reproducible sort-only benchmark. Optional baseline object is compiled
// from the parent revision with radix_order_omp renamed to *_baseline.
#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

#include "bh/generator.hpp"
#include "bh/parallel.hpp"
#include "bh/stages.hpp"
#include "bh/timer.hpp"

#ifdef BH_RADIX_BASELINE
namespace bh {
void radix_order_omp_baseline(const std::vector<std::uint64_t>&,
                              std::vector<std::int32_t>&);
}
#endif

int main(int argc, char** argv) {
    try {
        const std::size_t n = argc > 1 ? std::stoull(argv[1]) : 1000000;
        const int repeats = argc > 2 ? std::stoi(argv[2]) : 15;
        if (repeats < 1) throw std::invalid_argument("repeats must be positive");
        using Sort = void (*)(const std::vector<std::uint64_t>&, std::vector<std::int32_t>&);
        struct Variant { const char* name; Sort sort; };
        const std::vector<Variant> variants = {
#ifdef BH_RADIX_BASELINE
            {"baseline", bh::radix_order_omp_baseline},
#endif
            {"current", bh::radix_order_omp},
        };
        omp_set_dynamic(0);
        std::cout << "implementation,dist,n,requested_threads,actual_threads,repeat,seconds\n"
                  << std::setprecision(17);
        for (const std::string dist : {"uniform", "plummer"}) {
            const auto particles = bh::generate(dist, n, 42);
            bh::Octree tree;
            tree.set_root(bh::bbox_seq(particles));
            std::vector<std::uint64_t> keys(n);
            for (std::size_t i = 0; i < n; ++i)
                keys[i] = tree.morton_key(particles.x[i], particles.y[i], particles.z[i]);
            std::vector<std::int32_t> expected(n), order;
            std::iota(expected.begin(), expected.end(), 0);
            std::stable_sort(expected.begin(), expected.end(),
                             [&](auto a, auto b) { return keys[a] < keys[b]; });
            for (int threads : {1, 2, 4, 8, 16}) {
                omp_set_num_threads(threads);
                int actual_threads = 1;
#pragma omp parallel
                {
#pragma omp single
                    actual_threads = omp_get_num_threads();
                }
                for (const auto& variant : variants) {
                    // Warm up workers and allocation, then check every output.
                    for (int warmup = 0; warmup < 2; ++warmup) variant.sort(keys, order);
                    if (order != expected) throw std::runtime_error("sort reference mismatch");
                }
                for (int repeat = 0; repeat < repeats; ++repeat) {
                    // Alternate invocation order to reduce systematic ordering bias.
                    for (std::size_t k = 0; k < variants.size(); ++k) {
                        const auto& variant = variants[(k + repeat) % variants.size()];
                        const double begin = bh::steady_seconds();
                        variant.sort(keys, order);
                        const double seconds = bh::steady_seconds() - begin;
                        if (order != expected) throw std::runtime_error("sort reference mismatch");
                        std::cout << variant.name << ',' << dist << ',' << n << ',' << threads << ','
                                  << actual_threads << ',' << repeat << ',' << seconds << '\n';
                    }
                }
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
