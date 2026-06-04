#!/usr/bin/env bash
# Re-run the ocean cases that errored in the Phase-1 matrix run.
# Writes each case-family's logs/summaries under results/ocean_rerun/<case>.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUTBASE="results/ocean_rerun"
mkdir -p "$OUTBASE"

for case_pat in rest_state_uniform barotropic_double_gyre global_barotropic_wind inertia_gravity_wave lock_exchange phillips_two_layer stommel_gyre_tracer; do
    echo "=== Re-running $case_pat ==="
    JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_ocean_test_matrix.py --quick \
        --only "$case_pat" --output "$OUTBASE/$case_pat" \
        > "$OUTBASE/${case_pat}.log" 2>&1 || true
done

echo "=== Ocean rerun complete ==="
