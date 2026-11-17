#include "bh/parallel.hpp"

#include <array>
#include <memory>
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

namespace {
constexpr std::size_t kRadixBuckets = 256;
constexpr std::size_t kLocalSortLimit = 16384;

// Sort a disjoint range with thread-local histograms. Small ranges stay in
// cache for their remaining LSD passes. Larger ranges split by their next
// most significant byte so independent subranges can run as OpenMP tasks.
void sort_remaining(std::uint64_t* keys, std::int32_t* indices,
                    std::uint64_t* scratch_keys, std::int32_t* scratch_indices,
                    std::size_t count, int remaining_bits, std::int32_t* output) {
    if (count <= 1 || remaining_bits == 0) {
        if (indices != output) std::copy_n(indices, count, output);
        return;
    }
    std::array<std::size_t, kRadixBuckets> offsets{};
    if (count <= kLocalSortLimit || remaining_bits <= 8) {
        for (int shift = 0; shift < remaining_bits; shift += 8) {
            offsets.fill(0);
            for (std::size_t i = 0; i < count; ++i) ++offsets[(keys[i] >> shift) & 255];
            std::size_t position = 0;
            for (auto& offset : offsets) {
                const auto bucket_count = offset;
                offset = position;
                position += bucket_count;
            }
            for (std::size_t i = 0; i < count; ++i) {
                const auto key = keys[i];
                const auto destination = offsets[(key >> shift) & 255]++;
                scratch_keys[destination] = key;
                scratch_indices[destination] = indices[i];
            }
            std::swap(keys, scratch_keys);
            std::swap(indices, scratch_indices);
        }
        if (indices != output) std::copy_n(indices, count, output);
        return;
    }
    const int shift = remaining_bits - 8;
    for (std::size_t i = 0; i < count; ++i) ++offsets[(keys[i] >> shift) & 255];
    // Shared high bytes need no scatter, including tightly clustered keys.
    if (*std::max_element(offsets.begin(), offsets.end()) == count) {
        sort_remaining(keys, indices, scratch_keys, scratch_indices, count, shift, output);
        return;
    }
    std::size_t position = 0;
    for (auto& offset : offsets) {
        const auto bucket_count = offset;
        offset = position;
        position += bucket_count;
    }
    for (std::size_t i = 0; i < count; ++i) {
        const auto key = keys[i];
        const auto destination = offsets[(key >> shift) & 255]++;
        scratch_keys[destination] = key;
        scratch_indices[destination] = indices[i];
    }
    // Scatter cursors now hold each bucket's end. Siblings own disjoint
    // ranges in both buffers and in the final output, even when they swap.
    std::size_t first = 0;
    for (const auto last : offsets) {
        const std::size_t child_count = last - first;
        if (child_count > 0) {
#pragma omp task firstprivate(first, child_count, shift, keys, indices, scratch_keys, scratch_indices, output)
            sort_remaining(scratch_keys + first, scratch_indices + first,
                           keys + first, indices + first, child_count, shift, output + first);
        }
        first = last;
    }
#pragma omp taskwait
}
}  // namespace

void radix_order_omp(const std::vector<std::uint64_t>& keys,
                     std::vector<std::int32_t>& order) {
    const std::size_t n = keys.size();
    if (n > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
        throw std::length_error("radix_order_omp: too many particle indices");
    order.resize(n);
    if (n == 0) return;
    // Every temporary element is written before being read. Default-initialized
    // arrays avoid serial zero-filling of large buffers before parallel work.
    std::unique_ptr<std::uint64_t[]> sorted_keys(new std::uint64_t[n]);
    std::unique_ptr<std::uint64_t[]> scratch_keys(new std::uint64_t[n]);
    std::unique_ptr<std::int32_t[]> scratch_order(new std::int32_t[n]);
    using Histogram = std::array<std::size_t, kRadixBuckets>;
    std::vector<Histogram> histograms(omp_get_max_threads());
    std::array<std::size_t, kRadixBuckets + 1> boundaries;
    std::uint64_t different = 0;
#pragma omp parallel
    {
        const int rank = omp_get_thread_num(), team = omp_get_num_threads();
        const std::size_t first = partition_begin(n, rank, team);
        const std::size_t last = partition_begin(n, rank + 1, team);
#pragma omp for schedule(static) reduction(| : different)
        for (std::size_t i = 0; i < n; ++i) {
            order[i] = static_cast<std::int32_t>(i);
            different |= keys[i] ^ keys[0];
        }
        // Equal keys already have the required stable index order. All other
        // inputs first get one stable parallel partition by their high byte.
        if (different != 0) {
            auto& local = histograms[rank];
            local.fill(0);
            for (std::size_t i = first; i < last; ++i) ++local[keys[i] >> 56];
#pragma omp barrier
#pragma omp single
            {
                std::size_t position = 0;
                for (std::size_t digit = 0; digit < kRadixBuckets; ++digit) {
                    boundaries[digit] = position;
                    for (int thread = 0; thread < team; ++thread) {
                        const auto count = histograms[thread][digit];
                        histograms[thread][digit] = position;
                        position += count;
                    }
                }
                boundaries[kRadixBuckets] = n;
            }
            for (std::size_t i = first; i < last; ++i) {
                const auto key = keys[i];
                const auto destination = local[key >> 56]++;
                scratch_keys[destination] = key;
                scratch_order[destination] = static_cast<std::int32_t>(i);
            }
#pragma omp barrier
#pragma omp single
            {
                for (std::size_t digit = 0; digit < kRadixBuckets; ++digit) {
                    const auto begin = boundaries[digit];
                    const auto count = boundaries[digit + 1] - begin;
                    if (count > 0) {
#pragma omp task firstprivate(begin, count)
                        sort_remaining(scratch_keys.get() + begin, scratch_order.get() + begin,
                                       sorted_keys.get() + begin, order.data() + begin,
                                       count, 56, order.data() + begin);
                    }
                }
#pragma omp taskwait
            }
        }
    }
}

}  // namespace bh
