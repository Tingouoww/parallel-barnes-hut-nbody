// Sequential stand-ins for the parallel prefix sum and radix sort.
#include "bh/primitives.hpp"

namespace bh {

std::int32_t exclusive_scan(std::int32_t* v, std::size_t n) {
    std::int32_t sum = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const std::int32_t x = v[i];
        v[i] = sum;
        sum += x;
    }
    return sum;
}

// LSD radix sort, 11 bits per pass.
void sort_by_key(std::vector<std::uint64_t>& key, std::vector<std::int32_t>& value) {
    constexpr int kBits = 11;
    constexpr std::size_t kBuckets = std::size_t(1) << kBits;
    static std::vector<std::uint64_t> key_out;
    static std::vector<std::int32_t> value_out;
    const std::size_t n = key.size();
    key_out.resize(n);
    value_out.resize(n);

    for (int shift = 0; shift < 64; shift += kBits) {
        std::size_t start[kBuckets] = {};
        for (std::size_t i = 0; i < n; ++i) ++start[(key[i] >> shift) & (kBuckets - 1)];
        std::size_t sum = 0;
        for (std::size_t& s : start) {
            const std::size_t count = s;
            s = sum;
            sum += count;
        }
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t at = start[(key[i] >> shift) & (kBuckets - 1)]++;
            key_out[at] = key[i];
            value_out[at] = value[i];
        }
        key.swap(key_out);
        value.swap(value_out);
    }
}

}  // namespace bh
