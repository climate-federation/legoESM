#!/usr/bin/env bash
# Runs ocean → sea_ice → validator once atmosphere has finished.
# The atmosphere log is written to results/atmosphere_run.log in advance.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Wait until atmosphere summary.json exists (written at end of run).
until [ -f results/atmosphere/summary.json ]; do sleep 10; done

echo "=== Atmosphere matrix done.  Launching ocean matrix ==="
JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_ocean_test_matrix.py \
    --quick --output results/ocean \
    > results/ocean_run.log 2>&1 || true

echo "=== Ocean matrix done.  Launching sea_ice matrix ==="
JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_sea_ice_test_matrix.py \
    --quick --output results/sea_ice \
    > results/sea_ice_run.log 2>&1 || true

echo "=== All matrices done.  Running validator ==="
.venv/bin/python scripts/matrix/validate_matrix_report.py \
    > results/validator_stdout.log 2>&1 || true

echo "=== Post-matrix chain complete ==="
