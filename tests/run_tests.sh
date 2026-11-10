#!/usr/bin/env bash
# Validation tests: every case runs the simulation with the validation module
# switched on and fails if the error against direct summation, the energy
# drift or the tree structure check is out of bounds.
#
#   bash tests/run_tests.sh                 # tests ./bh with the seq and omp backends
#   BH=./bh_double bash tests/run_tests.sh
#   BACKENDS="omp" EXTRA="--build lock" bash tests/run_tests.sh   # test one strategy
set -u

BH=${BH:-./bh}
BACKENDS=${BACKENDS:-"seq omp"}
EXTRA=${EXTRA:-}
pass=0
fail=0

check() {
    local name=$1
    shift
    local out
    if out=$("$BH" "$@" --quiet 2>&1); then
        pass=$((pass + 1))
        echo "ok    $name"
    else
        fail=$((fail + 1))
        echo "FAIL  $name"
        echo "      $BH $*"
        echo "$out" | sed 's/^/      /'
    fi
}

for backend in $BACKENDS; do
    common="--backend $backend $EXTRA --validate-every 1 --check-tree"
    [ "$backend" = seq ] && common="--backend seq --validate-every 1 --check-tree"

    # theta = 0 opens every cell, so the tree walk must reproduce direct summation.
    for dist in uniform plummer collision degenerate; do
        check "$backend: $dist, theta=0 equals direct sum" \
            $common --dist $dist --n 3000 --steps 2 --theta 0 --max-err 1e-3
    done

    # Usual opening angles: the error must stay at the level expected for Barnes-Hut.
    for dist in plummer collision; do
        check "$backend: $dist, theta=0.5" \
            $common --dist $dist --n 20000 --steps 3 --theta 0.5 --max-err 0.03
        check "$backend: $dist, theta=1.0" \
            $common --dist $dist --n 20000 --steps 3 --theta 1.0 --max-err 0.2
    done
    # Accelerations nearly cancel at the centre of a uniform cube, which
    # inflates the relative error there; the bound is looser.
    check "$backend: uniform, theta=0.5" \
        $common --dist uniform --n 20000 --steps 3 --theta 0.5 --max-err 0.1

    # Coincident and nearly coincident particles (multi-particle leaves at MAX_DEPTH).
    check "$backend: degenerate, theta=0.5" \
        $common --dist degenerate --n 20000 --steps 3 --theta 0.5 --max-err 0.1

    # Tiny inputs (theta = 0 again, so the result must be exact).
    for n in 1 2 3 9; do
        check "$backend: n=$n" $common --dist plummer --n $n --steps 3 --theta 0 --max-err 1e-3
    done

    # Leapfrog energy conservation over 200 steps.
    check "$backend: energy conservation" \
        --backend $backend $( [ "$backend" = seq ] || echo "$EXTRA" ) --dist plummer --n 3000 \
        --steps 200 --theta 0.3 --validate-every 20 --max-drift 2e-3
done

echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
