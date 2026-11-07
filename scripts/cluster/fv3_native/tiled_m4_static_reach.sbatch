#!/bin/bash
#SBATCH --account=glab
#SBATCH --job-name=m4_reach
#SBATCH --partition=short
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/m4_static_reach_%j.log

# Static (program-derived) halo pad-depth ceiling for the acoustic substep.
# The taint interpreter is pure numpy on boolean arrays, so this is CPU-only;
# the CUDA plugin banner in the log is expected and harmless.

set -u
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3jax
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
cd "$REPO" || exit 1

echo "host: $(hostname)"
echo "PINNED sha: $(git rev-parse HEAD)"
echo "dirty: $(git status --porcelain | wc -l)"

# Without this the interpreter picks up the legoesm installed from ANOTHER
# worktree, where dynamics.gcm does not exist -- job 9611171 died on
# "No module named 'legoesm.atmosphere.dynamics.gcm'" for exactly that reason.
export PYTHONPATH="$REPO/packages/core:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ocean:$REPO/packages/ml:$REPO/packages/tools:$REPO/src"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

# The perturbation ladder's largest measured reach (job 9610385).  The probe
# may read LESS than the static bound (a limiter can swallow a perturbation);
# it may never exceed it, and --compare exits non-zero if it does.
LADDER=7
# km=10, NOT 5: at km=5 the level-major intermediates have km+1 = 6 levels,
# which the other-face leak detector cannot tell from the 6 faces (job
# 9631126 false-alarmed on exactly that).

# NON-VACUITY: the pre-fix analyser (whole-output gather/scatter/dyn-slice
# fallbacks) must FAIL the new selftest case; a case both versions pass proves
# nothing about the axis-aware rules.
OLD=/burg-archive/glab/users/pg2328/fv3_duo_gaps/tiled_m4_static_reach_prefix.py
echo "===== SELFTEST on PRE-FIX analyser (expect MISMATCH on the fori case) ====="
"$PY" "$OLD" --selftest; echo "OLD_SELFTEST_RC=$?"

echo "===== SELFTEST: toy programs of KNOWN reach ====="
"$PY" scripts/validate/fv3_native/tiled_m4_static_reach.py --selftest
echo "SELFTEST_RC=$?"

for ARM in "24 hydro" "48 hydro" "96 hydro" "48 nh" "96 nh"; do
  set -- $ARM
  N=$1; MODE=$2
  NHFLAG=""
  [ "$MODE" = "nh" ] && NHFLAG="--nh"
  echo "===== static reach, ${MODE} km=${KM:-10}, C${N} ====="
  "$PY" scripts/validate/fv3_native/tiled_m4_static_reach.py \
      --n "$N" --km ${KM:-10} $NHFLAG --compare "$LADDER"
  echo "ARM_RC=$? (4=unknown primitive, 1=contract violated)"
done

echo "DONE m4_static_reach"
