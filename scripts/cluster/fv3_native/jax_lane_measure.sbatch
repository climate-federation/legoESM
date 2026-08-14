#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=fv3jaxmeas
#SBATCH --time=08:00:00
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

# REPO is derived from the script's SUBMITTED path, never edited per run.
#
# Two earlier attempts failed and both are worth recording:
#   * editing REPO= inside a pinned worktree makes it dirty, so the preflight
#     below (correctly) refuses -- that killed job 9408879;
#   * ${BASH_SOURCE[0]} does NOT work under sbatch: SLURM copies the batch
#     script into its spool dir, so it resolved to /cm/local/apps/slurm and the
#     job ran with REPO pointing at the scheduler -- that killed job 9409086.
# scontrol reports the path actually submitted, which is the only one that
# refers to the worktree.  Falls back to BASH_SOURCE for a bare `bash` run.
if [ -n "${SLURM_JOB_ID:-}" ] && command -v scontrol >/dev/null 2>&1; then
  _SELF=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
          | tr ' ' '\n' | sed -n 's/^Command=//p' | head -1)
fi
[ -n "${_SELF:-}" ] || _SELF="${BASH_SOURCE[0]}"
REPO=$(cd "$(dirname "$_SELF")/../../.." && pwd)
if [ ! -e "$REPO/.git" ]; then
  echo "PREFLIGHT FAILED: REPO=$REPO is not a git worktree (resolved from"
  echo "$_SELF). Refusing rather than running against the wrong tree."
  exit 5
fi
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
BASETMP=/burg-archive/glab/users/pg2328/fv3_duo_gaps/pytest_basetemp_$SLURM_JOB_ID
mkdir -p "$BASETMP"

export PYTHONPATH=$REPO/packages/core:$REPO/packages/atmosphere:$REPO/src
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1

echo "host: $(hostname)"
cd "$REPO"

# ---------------------------------------------------------------- PREFLIGHT
# Earned 2026-08-14, twice in five minutes.  `cp` is aliased to prompt on this
# machine, so BOTH a plain `cp` and a `cp -f` into a pinned worktree silently
# did nothing -- and a job was submitted against a stale file, which would have
# been reported as a result.  `/bin/cp` bypasses the alias, but the real fix is
# to never hand-copy into a pinned tree: COMMIT FIRST, create the worktree at
# that SHA, and let git place the files.
#
# This block makes a stale run impossible to mistake for a good one: it aborts
# if the tree is dirty, and it prints the SHA next to every result so a log can
# never be attributed to the wrong code.
DIRTY=$(git status --porcelain | wc -l)
echo "PINNED sha: $(git rev-parse HEAD)"
echo "dirty files: $DIRTY"
if [ "$DIRTY" -ne 0 ]; then
  echo "PREFLIGHT FAILED: worktree is DIRTY. A pinned run must be reproducible"
  echo "from its SHA alone; an uncommitted file means the log cannot be"
  echo "attributed to any commit. Commit, re-create the worktree, resubmit."
  git status --porcelain | head -20
  exit 3
fi

NEW_MODULES="
packages/core/legoesm/grids/fv3_duo_halos.py
packages/core/legoesm/core/fv3_tp_core.py
packages/core/legoesm/core/fv3_pgrad.py
packages/core/legoesm/core/fv3_mapz.py
packages/core/legoesm/core/fv3_nh_core.py
packages/core/legoesm/core/fv3_duo_sw_core.py
packages/core/legoesm/core/fv3_duo_stepper.py
packages/core/legoesm/core/fv3_cgrid_phase_3d.py
"
NEW_TESTS="
tests/grids/test_fv3_duo_halos.py
tests/grids/test_fv3_tp_core.py
tests/grids/test_fv3_pgrad.py
tests/grids/test_fv3_mapz.py
tests/grids/test_fv3_nh_core.py
tests/grids/test_fv3_duo_sw_core.py
tests/grids/test_fv3_duo_stepper.py
tests/grids/test_fv3_cgrid_phase_3d.py
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
