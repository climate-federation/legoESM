#!/bin/bash
#SBATCH --account=glab
#SBATCH --job-name=m4_reach
#SBATCH --partition=short
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
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

export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=8

# The perturbation ladder's largest measured reach (job 9610385).  The probe
# may read LESS than the static bound (a limiter can swallow a perturbation);
# it may never exceed it, and --compare exits non-zero if it does.
LADDER=7

echo "===== SELFTEST: toy programs of KNOWN reach ====="
"$PY" scripts/validate/fv3_native/tiled_m4_static_reach.py --selftest
echo "SELFTEST_RC=$?"

for ARM in "24 hydro" "48 hydro" "24 nh"; do
  set -- $ARM
  N=$1; MODE=$2
  NHFLAG=""
  [ "$MODE" = "nh" ] && NHFLAG="--nh"
  echo "===== static reach, ${MODE} km=5, C${N} ====="
  "$PY" scripts/validate/fv3_native/tiled_m4_static_reach.py \
      --n "$N" --km 5 $NHFLAG --compare "$LADDER"
  echo "ARM_RC=$? (4=unknown primitive, 1=contract violated)"
done

echo "DONE m4_static_reach"
