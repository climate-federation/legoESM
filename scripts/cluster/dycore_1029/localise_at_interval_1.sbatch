#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=hs1029loc3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/results/dycore_1029/loc3_%j.out
#
# #1029 -- localise the burst by looking BEFORE the state has gone.
#
# The repaired diagnostic settled that a per-step statistic cannot localise
# this failure at the default cadence: at the trip step 414,700 of 414,720
# wind cells are already non-finite, so every extremum and every "first bad
# cell" degenerates to index 0.  The reason is the CHECK interval, not the
# statistic -- the blow-up test runs every 100 steps, i.e. the state has 5.6
# model-hours to go globally NaN before anyone looks.  This is the same defect
# class as an OMIP blow-up that was reported "at step 36" when 36 was simply
# the first sample.
#
# So: same two seeds, same everything, LEGOESM_BLOWUP_CHECK_INTERVAL=1.
#
# PRE-REGISTERED READING RULE:
#   * trip step lands well before the interval-100 step, and n_nonfinite_wind
#     is a SMALL fraction of 414,720 -> the burst is localisable; read
#     first_nonfinite_lat/lon/k against the three suspects (the ridge near
#     30N/270E, |lat| > 80 polar/metric, the model top).  This is the outcome
#     the issue has been trying to reach since July.
#   * trip step within a few steps of the interval-100 step AND the field is
#     still wholly non-finite -> the transition from healthy to globally NaN
#     takes fewer than ~1 step, and localisation needs a different instrument
#     (a per-substage check inside the RK3, or a bisect from the last clean
#     snapshot), not a finer outer stride.
#
# Note the trip steps to beat, measured at interval 100: seed 65537 -> 4800,
# seed 42 -> 23600.  At interval 1 those become upper bounds; anything lower
# is the resolution gained.
#
# COST: two idealized arms.  Checking every step costs a device-to-host sync
# per step, so expect these to be several times slower than the 1-4 min
# interval-100 arms -- that is the price of the resolution and it is still
# minutes, not hours.
set -uo pipefail

ROOT=/burg-archive/glab/users/pg2328/_sweep0923_wt
cd "$ROOT" || exit 2
OUT="/burg-archive/glab/users/pg2328/legoESM/results/dycore_1029/loc3_$SLURM_JOB_ID"
mkdir -p "$OUT"
export PYTHONPATH="$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1
export LEGOESM_BLOWUP_CHECK_INTERVAL=1
PY=/burg-archive/glab/users/ac5006/legoESM/.venv/bin/python
[ -x "$PY" ] || { echo "FATAL: interpreter $PY missing"; exit 2; }
echo "sha=$(git -C "$ROOT" rev-parse --short HEAD)$([ -n "$(git -C "$ROOT" status --porcelain)" ] && echo -dirty)  $(date -u)"

for s in 65537 42; do
  echo "=== seed $s, check every step (interval-100 trip was 4800 / 23600) $(date -u) ==="
  LEGOESM_TOPO_SEED=$s LEGOESM_TOPO_H0=2000 "$PY" -u \
      scripts/matrix/run_atmosphere_test_matrix.py \
      --only hydro --test "=held_suarez_topo" --grid latlon \
      --no-cross-grid-plots -o "$OUT/s$s" 2>&1 \
    | grep -E "BLOWUP|XFAIL|XPASS|CHECK_INTERVAL"
  r=$(find "$OUT/s$s" -name results.txt | head -1)
  [ -n "$r" ] && grep '^notes:' "$r"
done
echo "TOTAL WIND CELLS for scale: 72 x 144 x 40 = $((72*144*40))"
echo DONE $(date -u)
