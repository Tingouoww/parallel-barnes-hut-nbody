# Parallel radix sort performance

Issue [#7](https://github.com/Tingouoww/parallel-barnes-hut-nbody/issues/7) reports that the original eight-pass sort repeatedly reads Morton keys through the shuffled index array. The revised implementation preserves the existing API, immutable input keys, all 64 key bits, and ascending original indices for equal keys.

## Implementation

The first parallel pass stably partitions keys and indices by the highest byte. Buckets larger than 16,384 particles split further by their next high byte using OpenMP tasks. Smaller ranges finish their remaining 8-bit LSD passes with local histograms. Each range moves keys and indices together, improving sequential key reads and cache reuse. All-equal keys retain the initialized index order, and shared high bytes skip scattering.

All temporary storage belongs to the current call. Each task owns disjoint ranges of both buffers and the final output. Buffers are default-initialized and completely written before being read. Two temporary key arrays add 16 bytes per particle, or 16 MB at N=1,000,000, compared with the original index-only implementation.

An 11-bit, six-pass global LSD variant was evaluated. It improved single-thread time but increased multi-thread scatter costs in this environment. The selected implementation uses 8-bit partitions and local passes.

## Reproduction

The standalone benchmark generates uniform and Plummer Morton keys with seed 42, performs two warmups, and checks every output against an independent stable-sort reference. It reports all measured calls as CSV. Timing includes scratch allocation, initialization, sorting, output placement, and cleanup.

~~~bash
make -j
make bench-radix
~~~

To compare against the original implementation in a Bash environment:

~~~bash
git show ac05f15:src/cpu/parallel.cpp > build/radix-baseline.cpp
g++ -Iinclude -O3 -std=c++17 -Wall -Wextra -fopenmp \
  -Dradix_order_omp=radix_order_omp_baseline \
  -Dexclusive_scan_omp=exclusive_scan_omp_baseline \
  -c build/radix-baseline.cpp -o build/radix-baseline.o
g++ -Iinclude -O3 -std=c++17 -Wall -Wextra -fopenmp -DBH_RADIX_BASELINE \
  bench/radix_sort.cpp build/radix-baseline.o build/float/src/cpu/parallel.o \
  build/float/src/generator.o build/float/src/cpu/regular_stages.o \
  -o build/bench-radix-compare
OMP_DYNAMIC=FALSE OMP_PROC_BIND=FALSE OMP_WAIT_POLICY=PASSIVE \
  ./build/bench-radix-compare 1000000 15 > radix-comparison.csv
~~~

## Measurements

Measured on 2026-11-17 with an Intel Core i5-14400, 16 logical processors, Windows 11, and GCC 15.2.0. Flags were -O3 -std=c++17 -Wall -Wextra -fopenmp. OpenMP dynamic teams and binding were disabled, with passive waiting. The benchmark alternates implementation order over 15 repeats. Values are medians in milliseconds. The raw records and exact environment are in [bench/results](../bench/results/).

| Distribution | Threads | Original sort | Revised sort | Speedup |
|---|---:|---:|---:|---:|
| uniform | 1 | 42.029 | 24.575 | 1.71x |
| uniform | 2 | 22.551 | 14.148 | 1.59x |
| uniform | 4 | 13.932 | 11.374 | 1.22x |
| uniform | 8 | 11.816 | 9.661 | 1.22x |
| uniform | 16 | 11.497 | 7.646 | 1.50x |
| plummer | 1 | 40.812 | 23.738 | 1.72x |
| plummer | 2 | 22.009 | 14.017 | 1.57x |
| plummer | 4 | 12.940 | 9.231 | 1.40x |
| plummer | 8 | 12.807 | 9.170 | 1.40x |
| plummer | 16 | 10.892 | 8.926 | 1.22x |

### Morton tree integration

The remote Morton tree source at commit 249ceea was exercised through an adapter for its existing scan and key/value-sort interfaces. Both compared versions use the same adapter and scan implementation. At N=1,000,000 and 15 repeats, full tree-build medians were:

| Distribution | Threads | Original sort in tree build | Revised sort in tree build |
|---|---:|---:|---:|
| uniform | 1 | 82.679 | 68.702 |
| uniform | 8 | 31.709 | 28.676 |
| uniform | 16 | 29.144 | 27.948 |
| plummer | 1 | 81.194 | 66.327 |
| plummer | 8 | 33.035 | 31.475 |
| plummer | 16 | 28.746 | 27.511 |

This adapter benchmark includes key/value remapping costs. The issue's Docker measurements can be repeated using the standalone comparison above on that machine.

### Force-stage integration

Normal CLI runs used N=20,000, seed 42, theta=0.5, Morton force order, default chunks, and all four force schedulers. Each configuration used three processes with ten steps and two excluded warmup steps. Values below take the median of each run's measured steps, then the median across processes.

At eight threads, the measured order/gather/scatter interval fell from approximately 2.2–2.4 ms to 1.1–1.2 ms. At sixteen threads it fell from approximately 3.5–3.9 ms to 1.2–1.4 ms. Full force-stage time improved across the sampled configurations. All matching inputs and steps retained identical node and interaction counts. Per-run medians are preserved in [force-sort-issue7.csv](../bench/results/force-sort-issue7.csv).

## Validation

The primitive tests compare against an independent stable-sort reference, including empty and singleton inputs, duplicate and full-width keys, uneven worker slices, all-equal keys, and 65,537-element inputs with shared high prefixes. Both precisions pass multi-step force equality and work-metric checks, plus CLI/CSV integration tests.

Morton integration additionally checked 60 tree configurations across all four workload distributions and 1/3/8 threads. Tree structures and per-particle forces matched the original sort exactly, with structural and direct-summation checks passing.

The full single-precision physical suite passed 141 cases with zero failures. The double-precision costzones physical suite passed 18 cases with zero failures. CLI/CSV integration passed four test groups in each precision.
