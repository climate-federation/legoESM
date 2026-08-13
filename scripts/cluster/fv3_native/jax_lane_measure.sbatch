#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=fv3jaxmeas
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/fv3jax_measure_%j.log
# Measurement + smoke job for the FV3 duo-grid JAX lane (hop B: NumPy -> JAX).
#
# Purpose, in order:
#   1. ruff/import health of the new JAX modules,
#   2. run every new certification gate,
#   3. print the MEASURED JAX-vs-NumPy and jit-vs-eager errors so the
#      TOL-PENDING provisional bounds can be replaced by
#      `measured X, bound = measured x N` per the strategy doc's tolerance
#      policy (docs/atmosphere/fv3_duo_jax_lane_strategy.md section 4).
#
# TRAP (earned): SLURM sets TMPDIR=/local, which gets reaped mid-job on this
# cluster.  pytest --basetemp is forced onto a burg path below.
set -uo pipefail

REPO=/burg-archive/glab/users/pg2328/legoESM_fv3jax
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
BASETMP=/burg-archive/glab/users/pg2328/fv3_duo_gaps/pytest_basetemp_$SLURM_JOB_ID
mkdir -p "$BASETMP"

export PYTHONPATH=$REPO/packages/core:$REPO/packages/atmosphere:$REPO/src
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1

echo "host: $(hostname)"
cd "$REPO"
echo "PINNED sha: $(git rev-parse HEAD)"
echo "dirty files: $(git status --porcelain | wc -l)"

NEW_MODULES="
packages/core/legoesm/grids/fv3_duo_halos.py
packages/core/legoesm/core/fv3_tp_core.py
packages/core/legoesm/core/fv3_pgrad.py
packages/core/legoesm/core/fv3_mapz.py
packages/core/legoesm/core/fv3_nh_core.py
"
NEW_TESTS="
tests/grids/test_fv3_duo_halos.py
tests/grids/test_fv3_tp_core.py
tests/grids/test_fv3_pgrad.py
tests/grids/test_fv3_mapz.py
tests/grids/test_fv3_nh_core.py
"

echo "=== 0. which of the expected files exist ==="
for f in $NEW_MODULES $NEW_TESTS; do
  if [ -f "$REPO/$f" ]; then echo "  PRESENT $f ($(wc -l < "$REPO/$f") lines)";
  else echo "  MISSING $f"; fi
done

echo "=== 1. TOL-PENDING census (every one must be gone before this ships) ==="
grep -rn "TOL-PENDING" $REPO/packages $REPO/tests 2>/dev/null | sed "s|$REPO/||" || echo "  none"

echo "=== 2. ruff ==="
for f in $NEW_MODULES $NEW_TESTS; do
  [ -f "$REPO/$f" ] && $PY -m ruff check "$REPO/$f"
done

echo "=== 3. import health (each module alone, isolated interpreter) ==="
for f in $NEW_MODULES; do
  [ -f "$REPO/$f" ] || continue
  mod=$(echo "$f" | sed 's|^packages/[^/]*/||; s|/|.|g; s|\.py$||')
  $PY -c "import importlib,sys; importlib.import_module('$mod'); print('  OK   $mod')" \
    || echo "  FAIL $mod"
done

echo "=== 4. new certification gates (verbose: -s keeps measured prints) ==="
for t in $NEW_TESTS; do
  [ -f "$REPO/$t" ] || continue
  echo "--- $t"
  $PY -m pytest "$REPO/$t" -q -s --basetemp="$BASETMP" --tb=short
  echo "    pytest rc=$?"
done

echo "=== 5. NumPy-lane regression (the hop-A authority must stay green) ==="
$PY -m pytest "$REPO/tests/grids" -q --basetemp="$BASETMP" \
     -k "fv3_native" --tb=short
echo "    pytest rc=$?"

echo "=== 6. gradient gates only (the reason this lane exists) ==="
$PY -m pytest "$REPO/tests/grids" -q -s --basetemp="$BASETMP" \
     -k "grad and fv3" --tb=short
echo "    pytest rc=$?"

rm -rf "$BASETMP"
echo FV3JAX_MEASURE_DONE
