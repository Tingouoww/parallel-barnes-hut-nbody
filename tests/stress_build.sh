#!/usr/bin/env bash
# Stress test for a parallel tree build strategy (takes a few minutes).
# theta is huge so the force stage costs almost nothing; correctness comes
# from the tree structure check on every step.
#
#   bash tests/stress_build.sh            # --build lock
#   BUILD=morton bash tests/stress_build.sh
BH=${BH:-./bh}
BUILD=${BUILD:-lock}
fails=0
runs=0

run() {
    runs=$((runs + 1))
    if ! out=$(timeout 120 "$BH" "$@" --quiet 2>&1); then
        fails=$((fails + 1))
        echo "FAIL: $BH $*"
        echo "$out" | head -12
    fi
}

# High contention: few particles, many threads (including more threads than cores), many steps.
for t in 2 4 16 64; do
    for n in 8 64 500 5000 50000; do
        for d in plummer degenerate uniform; do
            steps=400
            [ "$n" -ge 50000 ] && steps=40
            run --backend omp --build "$BUILD" --threads $t --dist $d --n $n --steps $steps --theta 1000 \
                --validate-every 1 --check-tree --samples 1 --energy-max-n 0
        done
    done
done

# Fresh starts with degenerate input, which needs about 4N nodes and so
# exercises the path that grows the node arrays.
for s in 1 2 3 4 5 6 7 8; do
    run --backend omp --build "$BUILD" --threads 16 --seed $s --dist degenerate --n 200000 --steps 3 \
        --theta 1000 --validate-every 1 --check-tree --samples 1 --energy-max-n 0
done

echo "stress ($BUILD): $runs runs, $fails failures"
[ "$fails" -eq 0 ]
