// Reusable CPU primitives. No standard parallel algorithms or third-party runtimes.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace bh {

// Balanced contiguous ranges, including empty ranges when team > n.
inline std::size_t partition_begin(std::size_t n, int rank, int team) {
    return (n / team) * rank + std::min(n % team, static_cast<std::size_t>(rank));
}

// prefix has n+1 elements: prefix[0]=0, prefix[i+1]=prefix[i]+input[i].
// The input and output must be distinct. Sums use uint64_t arithmetic;
// callers must ensure the mathematical total fits (all particle costs do).
void exclusive_scan_omp(const std::vector<std::uint64_t>& input,
                        std::vector<std::uint64_t>& prefix);

// Stable ascending (key, original index) order. Keys remain in particle order.
// Accepts all 64 key bits, including the 63 used by Octree::morton_key().
void radix_order_omp(const std::vector<std::uint64_t>& keys,
                     std::vector<std::int32_t>& order);

}  // namespace bh
