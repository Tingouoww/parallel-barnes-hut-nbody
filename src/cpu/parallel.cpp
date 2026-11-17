#include "bh/parallel.hpp"

#include <array>
#include <limits>
#include <stdexcept>
#include <omp.h>

namespace bh {

void exclusive_scan_omp(const std::vector<std::uint64_t>& input,
                        std::vector<std::uint64_t>& prefix) {
    if (&input == &prefix) throw std::invalid_argument("exclusive_scan_omp needs distinct arrays");
    const std::size_t n = input.size();
    prefix.resize(n + 1);
    prefix[0] = 0;
    std::vector<std::uint64_t> offsets(omp_get_max_threads());
#pragma omp parallel
    {
        const int rank = omp_get_thread_num(), team = omp_get_num_threads();
        const std::size_t first = partition_begin(n, rank, team);
        const std::size_t last = partition_begin(n, rank + 1, team);
        std::uint64_t sum = 0;
        for (std::size_t i = first; i < last; ++i) {
            sum += input[i];
            prefix[i + 1] = sum;
        }
        offsets[rank] = sum;
#pragma omp barrier
#pragma omp single
        {
            std::uint64_t total = 0;
            for (int t = 0; t < team; ++t) {
                const auto local = offsets[t];
                offsets[t] = total;
                total += local;
            }
        }
        for (std::size_t i = first; i < last; ++i) prefix[i + 1] += offsets[rank];
    }
}

void radix_order_omp(const std::vector<std::uint64_t>& keys,
                     std::vector<std::int32_t>& order) {
    const std::size_t n = keys.size();
    if (n > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
        throw std::length_error("radix_order_omp: too many particle indices");
    order.resize(n);
    if (n == 0) return;
    std::vector<std::int32_t> scratch(n);
    using Histogram = std::array<std::size_t, 256>;
    std::vector<Histogram> histograms(omp_get_max_threads());
#pragma omp parallel
    {
        const int rank = omp_get_thread_num(), team = omp_get_num_threads();
        const std::size_t first = partition_begin(n, rank, team);
        const std::size_t last = partition_begin(n, rank + 1, team);
        for (std::size_t i = first; i < last; ++i) order[i] = static_cast<std::int32_t>(i);
#pragma omp barrier
        // Eight stable LSD passes. Each thread owns a contiguous input slice;
        // bucket offsets follow thread order, preserving ties across slices.
        for (int shift = 0; shift < 64; shift += 8) {
            auto& local = histograms[rank];
            local.fill(0);
            for (std::size_t i = first; i < last; ++i) ++local[(keys[order[i]] >> shift) & 255];
#pragma omp barrier
#pragma omp single
            {
                std::size_t offset = 0;
                for (int digit = 0; digit < 256; ++digit) {
                    for (int t = 0; t < team; ++t) {
                        const auto count = histograms[t][digit];
                        histograms[t][digit] = offset;
                        offset += count;
                    }
                }
            }
            for (std::size_t i = first; i < last; ++i) {
                const auto id = order[i];
                scratch[local[(keys[id] >> shift) & 255]++] = id;
            }
#pragma omp barrier
#pragma omp single
            order.swap(scratch);
        }
    }
}

}  // namespace bh
