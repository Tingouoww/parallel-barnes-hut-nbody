// Building blocks shared by the Morton tree build and costzones.
// src/cpu/primitives.cpp holds sequential stand-ins until the parallel
// versions are written; the signatures are what the callers rely on.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace bh {

// In place: v[i] becomes v[0] + ... + v[i - 1]. Returns the sum of all n values.
std::int32_t exclusive_scan(std::int32_t* v, std::size_t n);

// Sorts the keys in ascending order and moves each value with its key.
// Stable: equal keys keep their order.
void sort_by_key(std::vector<std::uint64_t>& key, std::vector<std::int32_t>& value);

}  // namespace bh
