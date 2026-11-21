#!/usr/bin/env python3
"""Q4: summarise the accuracy, work and timings recorded by run_q4.py.

    python3 scripts/analyze_q4.py results/q4-local

Writes accuracy.csv, timing.csv, summary.md and, when matplotlib is installed,
tradeoff.png and, when one thread was measured, one efficiency-n<N>.png per
problem size.

accuracy.csv  per phase, distribution, size and theta: interactions per particle,
              error against direct summation (median over the validated steps)
              and the largest |energy drift|
timing.csv    per distribution, size, theta, backend and thread count: force and
              whole-step time (median over measured steps, then over repeats),
              speedup over the sequential backend and efficiency, which is the
              speedup over the same strategy on one thread divided by the threads
"""
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ERRORS = ("err_median", "err_p99", "err_rms")


def load(folder):
    warmup = json.loads((folder / "meta.json").read_text(encoding="utf-8"))["config"]["warmup"]
    validated = defaultdict(list)
    timed = defaultdict(lambda: defaultdict(list))
    with (folder / "steps.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            shape = (row["dist"], int(row["n"]), round(float(row["theta"]), 6))
            if row["phase"] != "timing":
                if row["err_median"]:
                    validated[(row["phase"],) + shape].append(row)
            elif int(row["step"]) >= warmup:
                timed[shape + (row["backend"], int(row["threads"]))][row["repeat"]].append(row)
    return validated, timed


def accuracy(validated):
    table = {}
    for key, rows in validated.items():
        drift = [abs(float(r["energy_drift"])) for r in rows if r["energy_drift"]]
        table[key] = dict(steps=len(rows),
                          interactions=statistics.median(int(r["interactions"]) for r in rows) / key[2],
                          **{f: statistics.median(float(r[f]) for r in rows) for f in ERRORS},
                          energy_drift=max(drift) if drift else "")
    return table


def timing(timed):
    table = {}
    for key, repeats in timed.items():
        row = {stage: statistics.median(statistics.median(float(r[stage]) for r in rows)
                                        for rows in repeats.values()) for stage in ("force_s", "total_s")}
        table[key] = dict(repeats=len(repeats), **row, force_share=row["force_s"] / row["total_s"])
    for (dist, n, theta, backend, threads), row in table.items():
        seq, one = table[dist, n, theta, "seq", 1], table.get((dist, n, theta, "omp", 1))
        for stage in ("force", "total"):
            row[stage + "_speedup"] = seq[stage + "_s"] / row[stage + "_s"]
            row[stage + "_efficiency"] = (one[stage + "_s"] / (threads * row[stage + "_s"])
                                          if backend == "omp" and one else "")
    return table


def write_csv(table, names, path):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=names + list(next(iter(table.values()))))
        writer.writeheader()
        for key, row in sorted(table.items()):
            writer.writerow(dict(row, **dict(zip(names, key))))


def markdown(acc, times):
    lines = []
    for dist in sorted({key[1] for key in acc}):
        lines += [f"**{dist}：精確度與計算量**", "",
                  "| N | θ | 每粒子交互作用次數 | 誤差中位數 | 誤差 p99 | RMS 誤差 | 循序受力計算 | 占整步 |",
                  "|---|---|---|---|---|---|---|---|"]
        for (phase, d, n, theta), r in sorted(acc.items()):
            if (phase, d) != ("error", dist):
                continue
            seq = times.get((dist, n, theta, "seq", 1))
            cost = f"{seq['force_s']:.3f} s | {100 * seq['force_share']:.1f}%" if seq else " | "
            lines.append(f"| {n:,} | {theta:g} | {r['interactions']:.0f} | {r['err_median']:.1e} | "
                         f"{r['err_p99']:.1e} | {r['err_rms']:.1e} | {cost} |")
        lines.append("")

    energy = {key[1:]: r for key, r in acc.items() if key[0] == "energy"}
    if energy:
        thetas = sorted({theta for _, _, theta in energy})
        (n, steps), = {(n, r["steps"]) for (_, n, _), r in energy.items()}
        lines += [f"**能量漂移**（N = {n:,}，{steps} 次驗證中的最大絕對值）與同一段模擬的誤差 p99 中位數", "",
                  "| 分布 | " + " | ".join(f"θ = {t:g}" for t in thetas) + " |", "|---|" + "---|" * len(thetas)]
        for dist in sorted({dist for dist, _, _ in energy}):
            cells = [f"{energy[dist, n, t]['energy_drift']:.1e}（{energy[dist, n, t]['err_p99']:.1e}）"
                     for t in thetas]
            lines.append(f"| {dist} | " + " | ".join(cells) + " |")
        lines.append("")

    threads = sorted({key[4] for key in times if key[3] == "omp"})
    if threads and threads[0] != 1:
        lines += ["沒有 1 執行緒的量測，不計算平行效率。", ""]
    for dist, n in sorted({key[:2] for key in times} if len(threads) > 1 and threads[0] == 1 else []):
        lines += [f"**{dist}，N = {n:,}：平行效率**（受力計算 / 整步，相對於同一策略的 1 執行緒）", "",
                  "| θ | 1 執行緒受力計算 | " + " | ".join(f"{t} 執行緒" for t in threads[1:]) + " |",
                  "|---|---|" + "---|" * (len(threads) - 1)]
        for theta in sorted({key[2] for key in times if key[:2] == (dist, n)}):
            rows = [times[dist, n, theta, "omp", t] for t in threads]
            cells = [f"{100 * r['force_efficiency']:.0f}% / {100 * r['total_efficiency']:.0f}%" for r in rows[1:]]
            lines.append(f"| {theta:g} | {rows[0]['force_s']:.3f} s | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def plot(acc, times, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dists = sorted({key[1] for key in acc})
    fig, axes = plt.subplots(1, len(dists), figsize=(4.5 * len(dists), 4), squeeze=False, sharey=True)
    for ax, dist in zip(axes[0], dists):
        for n in sorted({key[2] for key in acc if key[:2] == ("error", dist)}):
            rows = sorted((theta, r) for (phase, d, m, theta), r in acc.items() if (phase, d, m) == ("error", dist, n))
            work = [r["interactions"] for _, r in rows]
            line, = ax.plot(work, [r["err_p99"] for _, r in rows], marker="o", label=f"N = {n:,}, p99")
            ax.plot(work, [r["err_median"] for _, r in rows], marker="o", linestyle="--", color=line.get_color(),
                    label="median")
            for theta, r in rows:
                ax.annotate(f"{theta:g}", (r["interactions"], r["err_p99"]), textcoords="offset points",
                            xytext=(4, 4), fontsize=8)
        ax.set(title=dist, xlabel="interactions per particle", ylabel="relative force error", xscale="log",
               yscale="log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Accuracy against work (points are labelled with theta)")
    fig.tight_layout()
    fig.savefig(folder / "tradeoff.png", dpi=150)
    plt.close(fig)

    threads = sorted({key[4] for key in times if key[3] == "omp"})
    if len(threads) < 2 or threads[0] != 1:
        return
    dists = sorted({key[0] for key in times})
    for n in sorted({key[1] for key in times}):
        fig, axes = plt.subplots(2, len(dists), figsize=(4.2 * len(dists), 7), squeeze=False, sharey=True)
        for col, dist in enumerate(dists):
            for ax, stage in zip(axes[:, col], ("force", "total")):
                for theta in sorted({key[2] for key in times if key[:2] == (dist, n)}):
                    ax.plot(threads, [100 * times[dist, n, theta, "omp", t][stage + "_efficiency"] for t in threads],
                            marker="o", label=f"theta = {theta:g}")
                ax.set(title=f"{dist}: {'force stage' if stage == 'force' else 'whole step'}", xlabel="threads",
                       ylabel="parallel efficiency (%)", xscale="log")
                ax.set_xticks(threads, [str(t) for t in threads])
                ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
                ax.grid(alpha=0.3)
                ax.legend(fontsize=8)
        fig.suptitle(f"Parallel efficiency, N = {n:,} (relative to one thread of the same strategy)")
        fig.tight_layout()
        fig.savefig(folder / f"efficiency-n{n}.png", dpi=150)
        plt.close(fig)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    folder = Path(sys.argv[1])
    validated, timed = load(folder)
    acc, times = accuracy(validated), timing(timed)
    write_csv(acc, ["phase", "dist", "n", "theta"], folder / "accuracy.csv")
    write_csv(times, ["dist", "n", "theta", "backend", "threads"], folder / "timing.csv")
    text = markdown(acc, times)
    (folder / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    try:
        plot(acc, times, folder)
    except ImportError:
        print("matplotlib is not installed: no figures written")


if __name__ == "__main__":
    main()
