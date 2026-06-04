#!/usr/bin/env bash
# One-shot verification that the scaling work is healthy:
#   1. iter-199 vector-halo regression tests (3 tests, ~15 s)
#   2. Conservation harness across all 12 (case × grid × dycore) cells
#   3. ``--scan-steps`` equivalence verifier (3 dycores)
#   4. iter-220 ``--scan-steps auto`` dispatch-table pin (2 tests, ~0.2 s)
#   5. iter-228 PE-dycore inline-import drift audit (4 tests, ~0.05 s)
#
# Use:
#   ./scripts/validate_scaling.sh
#
# Exit 0 if all green, exit 1 otherwise.  Designed to run inside an
# iteration budget (~2 min total wall) so it can be the first thing a
# new iteration calls before adding code.

set -e
cd "$(dirname "$0")/../.."

GREEN=$'\033[0;32m'
RED=$'\033[0;31m'
RESET=$'\033[0m'
PASS=0
FAIL=0

check() {
    local label="$1"; shift
    if "$@" > /tmp/validate_scaling_last.log 2>&1; then
        echo "${GREEN}✓ $label${RESET}"
        PASS=$((PASS + 1))
    else
        echo "${RED}✗ $label${RESET}"
        tail -5 /tmp/validate_scaling_last.log | sed 's/^/    /'
        FAIL=$((FAIL + 1))
    fi
}

echo "==== Scaling-work validation ===="
echo

echo "[1/5] iter-199 vector-halo regression tests"
check "TestHydrostaticToFV3VectorHalo (3 tests)" \
    bash -c "JAX_ENABLE_X64=1 PYTHONPATH=. .venv/bin/python -m pytest \
        tests/atmosphere/hydrostatic/unit/test_primitive_eq.py::TestHydrostaticToFV3VectorHalo \
        -q --no-header"

echo
echo "[2/5] Conservation harness (12 cells, no fixers)"
check "All cells finite + bounded drift" \
    bash -c "PYTHONPATH=. JAX_ENABLE_X64=1 .venv/bin/python \
        scripts/matrix/check_conservation_all.py 2>&1 | \
        grep -q ' OK ' && \
        ! ( PYTHONPATH=. JAX_ENABLE_X64=1 .venv/bin/python \
            scripts/matrix/check_conservation_all.py 2>&1 | grep -q ERROR )"

echo
echo "[3/5] --scan-steps equivalence (per-dycore bit-or-fp32)"
check "scan=12 vs Python loop agreement" \
    bash -c "JAX_ENABLE_X64=1 PYTHONPATH=. .venv/bin/python \
        scripts/validate/verify_scan_steps_equivalence.py 2>&1 | grep -q DIVERGED && \
        exit 1 || exit 0"

echo
echo "[4/5] iter-220 --scan-steps auto dispatch-table pin"
check "AUTO_SCAN_STEPS = iter-212 honest values" \
    bash -c "JAX_ENABLE_X64=1 PYTHONPATH=. .venv/bin/python -m pytest \
        tests/test_bcw_benchmark_scan_steps.py \
        -q --no-header"

echo
echo "[5/5] iter-228 PE-dycore inline-import drift audit"
check "All four PE files within inline-import budget" \
    bash -c "JAX_ENABLE_X64=1 PYTHONPATH=. .venv/bin/python -m pytest \
        tests/test_pe_dycore_inline_imports.py \
        -q --no-header"

echo
echo "==== $PASS passed, $FAIL failed ===="
[ "$FAIL" -eq 0 ]
