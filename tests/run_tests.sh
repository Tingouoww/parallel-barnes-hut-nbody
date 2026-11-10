#!/usr/bin/env bash
# Validation tests: every case runs the simulation with the validation module
# switched on and fails if the error against direct summation, the energy
# drift or the tree structure check is out of bounds.
#
#   bash tests/run_tests.sh                    # tests ./bh in every configuration below
#   BH=./bh_double bash tests/run_tests.sh
#   CONFIGS="--backend omp --force dynamic" bash tests/run_tests.sh    # one configuration
#   CONFIGS="--backend omp --build lock --threads 2;--backend omp --build lock --threads 64" ...
set -u

BH=${BH:-./bh}
# Backend / strategy flags to test, separated by ';'.
CONFIGS=${CONFIGS:-"--backend seq;--backend omp;--backend omp --build lock;--backend omp --build lock --threads 3"}
pass=0
fail=0

ok() {
    pass=$((pass + 1))
    echo "ok    $1"
}

bad() {
    fail=$((fail + 1))
    echo "FAIL  $1"
    shift
    printf '%s\n' "$@" | sed 's/^/      /'
}

# check NAME ARGS...: the run must exit with status 0.
check() {
    local name=$1
    shift
    local out
    if out=$("$BH" "$@" --quiet 2>&1); then ok "$name"; else bad "$name" "$BH $*" "$out"; fi
}

# Physical results of a run (everything except timings), for comparing two configurations.
results() {
    "$BH" "$@" --validate-every 1 2>&1 |
        sed -n -e 's/^step *\([0-9]*\) .*\(nodes .*\)$/step \1 \2/p' -e '/error vs/p' -e '/energy/p'
}

IFS=';' read -r -a configs <<<"$CONFIGS"
for config in "${configs[@]}"; do
    # shellcheck disable=SC2086
    set -- $config
    common=("$@" --validate-every 1 --check-tree)

    # theta = 0 opens every cell, so the tree walk must reproduce direct summation.
    for dist in uniform plummer collision degenerate; do
        check "[$config] $dist, theta=0 equals direct sum" \
            "${common[@]}" --dist $dist --n 3000 --steps 2 --theta 0 --max-err 1e-3
    done

    # Usual opening angles: the error must stay at the level expected for Barnes-Hut.
    for dist in plummer collision; do
        check "[$config] $dist, theta=0.5" \
            "${common[@]}" --dist $dist --n 20000 --steps 3 --theta 0.5 --max-err 0.03
        check "[$config] $dist, theta=1.0" \
            "${common[@]}" --dist $dist --n 20000 --steps 3 --theta 1.0 --max-err 0.2
    done
    # Accelerations nearly cancel at the centre of a uniform cube, which
    # inflates the relative error there; the bound is looser.
    check "[$config] uniform, theta=0.5" \
        "${common[@]}" --dist uniform --n 20000 --steps 3 --theta 0.5 --max-err 0.1

    # Coincident and nearly coincident particles (multi-particle leaves at MAX_DEPTH).
    check "[$config] degenerate, theta=0.5" \
        "${common[@]}" --dist degenerate --n 20000 --steps 3 --theta 0.5 --max-err 0.1

    # Tiny inputs (theta = 0 again, so the result must be exact).
    for n in 1 2 3 9; do
        check "[$config] n=$n" "${common[@]}" --dist plummer --n $n --steps 3 --theta 0 --max-err 1e-3
    done

    # Leapfrog energy conservation over 200 steps.
    check "[$config] energy conservation" \
        "$@" --dist plummer --n 3000 --steps 200 --theta 0.3 --validate-every 20 --max-drift 2e-3

    # Every configuration builds the same tree and walks it in the same order,
    # so node counts, interaction counts, errors and energies must be
    # identical to the sequential backend, digit for digit.
    if [ "$config" != "--backend seq" ]; then
        for dist in uniform plummer collision; do
            args=(--dist $dist --n 20000 --steps 6 --theta 0.5)
            want=$(results --backend seq "${args[@]}")
            got=$(results "$@" "${args[@]}")
            if [ -n "$want" ] && [ "$want" = "$got" ]; then
                ok "[$config] $dist, identical to seq"
            else
                bad "[$config] $dist, identical to seq" "$(diff <(echo "$want") <(echo "$got") | head -8)"
            fi
        done
    fi
done

echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
