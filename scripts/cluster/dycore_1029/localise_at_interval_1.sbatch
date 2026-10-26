#!/bin/bash
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --job-name=hs1029loc4
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/work/bd1083/b309178/diffESM/legoesm_pg/amip_runs/dycore_1029_anatomy/loc4_%j.out
#
# #1029 -- localise the burst by looking BEFORE the state has gone.
#
# History: run 2026-09-23 (job 9947903, other cluster) on the OLD mountain
# (398 km ridges, 30N) and no del-4: bursts at 36N/252E k=18 and 16N/255E
# k=16, lower troposphere.  Since then #1907 (DCMIP widths 3pi/4, pi/16,
# centre 0N/270E; del-4 nu=2.88e15 on u,v,T) and #1908 (SB81 omega/p on)
# landed; the case still dies (interval-100: day 101.85, step 44000, seed 42,
# field 99.99% NaN at the trip).  This re-runs the SAME localisation on the
# CURRENT setup.  Usage:  sbatch localise_at_interval_1.sbatch <code-tree>
#
# STAGE 1: matrix runner, LEGOESM_BLOWUP_CHECK_INTERVAL=1, seeds 65537, 42.
# STAGE 2: burst_anatomy_probe.py (--nu4-matrix), same seed, per-step samples
#          over the last 500 steps before the stage-1 trip.
#
# (a) NUMBERS: per seed, interval-1 trip step, n_nonfinite_wind at the trip,
#     argmax (lat, lon, k) of the cell-centre wind at the trip, its
#     great-circle distance r from the mountain centre (0N, 270E) and p/p_s.
#     Stage 2: per-step umax, n_unst_low (theta inversions, lowest 20 levels),
#     dp_min, over the last 500 steps.
# (b) READING RULE (written before the run):
#   CONTROL (stage 2 is read only if it holds): probe trip step == stage-1
#     trip step for that seed.  Else the probe is not the matrix model.
#   LOCALISABLE iff n_nonfinite_wind < 1% of 414,720 at the trip; otherwise
#     UNLOCALISABLE at outer-step resolution (needs a per-RK-stage check).
#   MOUNTAIN / LOWER TROPOSPHERE CONFIRMED (both seeds): argmax over terrain
#     (r < R_m = 135 deg) AND p/p_s >= 0.5 AND |lat| <= 80.
#   REFUTED (any seed): |lat| > 80 (polar/metric), or p/p_s < 0.5 (upper
#     levels / lid), or r >= R_m (flat ground).
#   Stage 2 anatomy: n_unst_low growing >= 10x before umax grows 10x ->
#     static instability first; umax jumping > 10x in one step with
#     n_unst_low flat -> one-step numerical singularity.  Neither -> report
#     the series, no mechanism claimed.
# (c) No cheaper test: the location needs the state within one step of the
#     trip; checkpoints are ~20 days apart.  COST: interval-100 arm was 131 s
#     for 44000 steps; expect <= 30 min on 1 GPU (cap 1 GPU-h).
set -uo pipefail
ROOT=${1:?usage: sbatch localise_at_interval_1.sbatch <code-tree>}
cd "$ROOT" || exit 2
OUT=/work/bd1083/b309178/diffESM/legoesm_pg/amip_runs/dycore_1029_anatomy/loc4_$SLURM_JOB_ID
mkdir -p "$OUT"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | tr '\n' ':')"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python
[ -x "$PY" ] || { echo "FATAL: interpreter $PY missing"; exit 2; }
echo "tree=$ROOT sha=$(cat "$ROOT/SHA" 2>/dev/null || git -C "$ROOT" rev-parse --short HEAD)  $(date -u)"

for s in 65537 42; do
  echo "=== seed $s, matrix, check every step $(date -u) ==="
  LEGOESM_BLOWUP_CHECK_INTERVAL=1 LEGOESM_TOPO_SEED=$s LEGOESM_TOPO_H0=2000 "$PY" -u \
      scripts/matrix/run_atmosphere_test_matrix.py \
      --only hydro --test "=held_suarez_topo" --grid latlon \
      --no-cross-grid-plots -o "$OUT/s$s" > "$OUT/s$s.log" 2>&1
  grep -E "BLOWUP|XFAIL|XPASS|CHECK_INTERVAL|dissipation|exponent" "$OUT/s$s.log"
  r=$(find "$OUT/s$s" -name results.txt | head -1)
  [ -n "$r" ] && grep '^notes:' "$r"
  trip=$(grep -oE "BLOWUP at step [0-9]+" "$OUT/s$s.log" | head -1 | grep -oE "[0-9]+$")
  if [ -n "$trip" ]; then
    echo "=== seed $s, probe, per-step from $((trip-500)) $(date -u) ==="
    "$PY" -u scripts/cluster/dycore_1029/burst_anatomy_probe.py --seed $s --h0 2000 \
        --nu4-matrix --every 1000 --fine-from $((trip-500)) --max-steps $((trip+50)) \
        --out "$OUT/probe_s$s" > "$OUT/probe_s$s.log" 2>&1
    grep -E "^TRIP|^DONE" "$OUT/probe_s$s.log" | cut -c1-400
    echo "CONTROL seed $s: matrix trip $trip, probe trip $("$PY" -c "import json;print(json.load(open('$OUT/probe_s$s/meta.json'))['trip_step'])" 2>/dev/null)"
  fi
done
echo "TOTAL WIND CELLS for scale: 72 x 144 x 40 = $((72*144*40))"
echo DONE $(date -u)
