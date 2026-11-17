#pragma once
#include <chrono>

namespace bh {
// Match the driver's monotonic clock. Some OpenMP runtimes on Windows have
// coarse omp_get_wtime() resolution, unsuitable for sub-millisecond stages.
inline double steady_seconds() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace bh
