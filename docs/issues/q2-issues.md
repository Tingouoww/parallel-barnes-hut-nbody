## 1. [Q2] Implement reusable OpenMP prefix sum and stable Morton radix sort

Owner: @mizu5555.

Source: the division of work and CPU parallelization plan in the local `docs/proposal/main.tex`.

The existing repository has no shared parallel scan or radix sort. These primitives are needed by CPU Morton tree construction and costzones.

Acceptance criteria:
- [ ] Expose a reusable 64-bit OpenMP exclusive scan API, including the final total.
- [ ] Implement stable parallel radix sorting of Morton keys into particle indices; break equal-key ties by original index.
- [ ] Use no third-party parallel algorithms (TBB, Thrust, CUB, or parallel STL).
- [ ] Test empty/singleton inputs, duplicate and boundary keys, uneven blocks, large totals, and more threads than elements.
- [ ] Document the API for the CPU tree-construction owner.

Validation: compare against independent serial scan and stable-sort references across multiple actual OpenMP team sizes.

## 2. [Q2] Add static, dynamic, and guided OpenMP force scheduling with Morton order

Owner: @mizu5555.

Source: proposal CPU force scheduling and README CPU strategy extension instructions.

Only sequential force evaluation is currently registered. Reuse `force_on_particle()` so scheduling does not alter a particle's tree walk.

Acceptance criteria:
- [ ] Register static, dynamic, and guided force strategies and expose chunk size through the CLI.
- [ ] Sort Morton indices, physically pack the force SoA, and scatter outputs back while preserving particle identity and the existing tree's indices.
- [ ] Offer original-index order as a locality control, especially for static/dynamic scheduling.
- [ ] Record the effective configuration in CSV and include ordering overhead in total force time.
- [ ] Test multiple steps, tiny and degenerate inputs, all workload distributions, seq/lock trees, and multiple thread counts.

Validation: require bit-identical accelerations and interaction counts when using the same tree. Independently validate physical error and tree structure. Note that existing lock insertion can reorder duplicate-key leaf chains across separate builds.

Dependency: reusable CPU scan/radix-sort issue.

## 3. [Q2] Implement Morton costzones using previous-step interaction costs

Owner: @mizu5555.

Source: proposal CPU costzones method.

Acceptance criteria:
- [ ] Register `--force costzones` and require Morton work order.
- [ ] Use equal particle counts on the first step and snapshot each particle's previous-step interaction cost before any worker overwrites it.
- [ ] Run the shared parallel prefix sum in the current Morton order and split it into contiguous ranges of approximately equal estimated cost.
- [ ] Cover every target exactly once, using the actual OpenMP team size.
- [ ] Handle zero total costs, repeated Morton keys, moving particles, uneven counts, idle threads, and reloading a backend.
- [ ] Include sorting and partition overhead in measured force time.

Validation: independently test weighted partitions, coverage, first-step fallback, and multi-step collision trajectories against sequential walks over the same tree.

Dependencies: shared CPU primitives and OpenMP scheduling issues.

## 4. [Q2] Measure per-thread work, load imbalance, and force scheduling overhead

Owner: @mizu5555.

Source: proposal Q2 metrics, lambda = max(Wp) / mean(Wp).

Existing CSV records stage timings and total interactions but lacks per-thread work and scheduling diagnostics.

Acceptance criteria:
- [ ] Record actual force team size and per-thread interaction totals, including idle threads.
- [ ] Compute lambda and document the all-zero-work convention.
- [ ] Extend CSV with ordering, chunk, actual threads, per-thread work, lambda, and ordering/partition/walk timings.
- [ ] Preserve existing columns, record warmup rows explicitly, and reject incompatible CSV headers when appending.
- [ ] Include all necessary force-strategy overhead in force_s.

Validation: sum(Wp) equals sum(particle.cost), independently check lambda, verify CSV parsing and safe append behavior, and confirm physics remains unchanged.

## 5. [Q2] Add reproducible scaling and collision experiments, plots, and analysis

Owner: @mizu5555.

Source: proposal Q2 experiments, deliverables, and reporting responsibilities.

Acceptance criteria:
- [ ] Automate uniform/Plummer/collision workloads, all four schedulers, Morton/original controls, and chunk sensitivity.
- [ ] Measure strong scaling at fixed N and weak scaling at fixed N/threads; define matching-N speedup and efficiency separately from weak time efficiency.
- [ ] Fix seeds/parameters, exclude warmup steps, repeat runs, and save commands, environment, compiler, and hardware metadata.
- [ ] Produce raw CSV, summary tables, stage breakdowns, load imbalance, locality/scaling plots, and collision time series.
- [ ] Run a practical local experiment and write conclusions supported by its measured data, stating the scope and hardware limitations.
- [ ] Supply full experiment settings and reusable methods/report guidance; update README progress and usage.

Validation: reproduce all figures from saved raw data. Clearly retain unrun final-hardware or large-scale experiments as pending work.

Dependencies: CPU primitives, scheduling, costzones, and metrics issues.

## Published tracking and local completion

All five issue drafts were published in English as [#1](https://github.com/Tingouoww/parallel-barnes-hut-nbody/issues/1), [#2](https://github.com/Tingouoww/parallel-barnes-hut-nbody/issues/2), [#3](https://github.com/Tingouoww/parallel-barnes-hut-nbody/issues/3), [#4](https://github.com/Tingouoww/parallel-barnes-hut-nbody/issues/4), and [#5](https://github.com/Tingouoww/parallel-barnes-hut-nbody/issues/5).

The implementation, tests, 609-run local study, and analysis are complete in this checkout. The checklists above preserve the original task definitions; current local validation status is recorded in the GitHub issues. They remain open until source publication and review. See [methods](../q2.md) and [measured analysis](../results/q2-local/analysis.md). Final-hardware experiments and group report/slides remain team integration work.
