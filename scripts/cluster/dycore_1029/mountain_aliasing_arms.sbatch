#!/bin/bash
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --job-name=hs1029arms
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=/work/bd1083/b309178/diffESM/legoesm_pg/amip_runs/dycore_1029_anatomy/arms_%j.out
#
# #1029 -- is the ALIASED DCMIP ridge spacing the generator?  Diagnosis only;
# every override is a scratch env var read by scripts/cluster/dycore_1029/mountain_override_arm.py,
# nothing committed moves.
# Read off code: dcmip_2_0_0_mountain uses zeta = R/16 (398 km) and
# R_m = 0.75 R; its docstring, the repo's DCMIP 1-3 mountain and DCMIP v1.7
# use zeta = pi/16 rad (1251 km), R_m = 3pi/4 rad, centre 0N (code: 30N).
# On 72x144 the current mountain is a cell-to-cell checkerboard (max 2dx
# content 1654 m of 1920 m; with zeta = pi/16: 605 m).  CPU spin-up, seed
# 65537, day 5.6: max wind 82 m/s current, 7.1 zeta-fixed, 2.8 flat.
#
# GROUP Z (one variable: zeta_frac = pi/16), 6 seeds; controls: seeds 99,
# 104729 with NO override must blow up (exp-3 job 27926385: 7.6 / 8.8 d).
#   (a) survival day (censored 200 d) + final max|v|.
#   (b) CONFIRMED: >= 5/6 survive 200 d, final max|v| < 90 m/s.
#       REFUTED: <= 1/6 survive AND median < 2x baseline median 13.9 d.
#       Else PARTIAL.  Read only if both controls blow up.
# GROUP D (candidate case definition: DCMIP as published -- zeta, R_m and
# centre latitude all change; a NEW BASELINE, not an attribution arm), 6
# seeds.  VIABLE >= 5/6 survive with max|v| < 90; NOT VIABLE <= 1/6.
# GROUP X (cross-lane, 10 d, default seed): spectral / icosahedral /
# cubed_sphere current mountain, latlon flat -- max wind over days 3-6
# against the lat-lon 36-82 m/s.  Reported, not a verdict input.
# (c) no cheaper test: survival scatters 5.5x across seeds; no offline proxy
#     has been validated against survival.
# COST: 18 arms, 4 GPUs, <= 3 h wall (<= ~4 GPU-hours, most arms short).
set -uo pipefail
ROOT=/work/bd1083/b309178/diffESM/legoesm_pg/wt_1029
OUT=/work/bd1083/b309178/diffESM/legoesm_pg/amip_runs/dycore_1029_anatomy/arms_$SLURM_JOB_ID
mkdir -p $OUT; cd $ROOT
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | tr '\n' ':')"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python
echo "sha=$(git -c safe.directory=$ROOT rev-parse --short HEAD) dirty=$(git -c safe.directory=$ROOT status --porcelain | wc -l) $(date -u)"
PI16=0.19634954084936207; R3PI4=2.356194490192345
# tag grid days seed zeta rm latc  ('-' = unset)
cat > $OUT/arms.txt <<ARMS
base_s99 latlon 200 99 - - -
base_s104729 latlon 200 104729 - - -
Z_s42 latlon 200 42 $PI16 - -
Z_s65537 latlon 200 65537 $PI16 - -
Z_s7 latlon 200 7 $PI16 - -
Z_s99 latlon 200 99 $PI16 - -
Z_s1234 latlon 200 1234 $PI16 - -
Z_s2024 latlon 200 2024 $PI16 - -
D_s42 latlon 200 42 $PI16 $R3PI4 0.0
D_s65537 latlon 200 65537 $PI16 $R3PI4 0.0
D_s7 latlon 200 7 $PI16 $R3PI4 0.0
D_s99 latlon 200 99 $PI16 $R3PI4 0.0
D_s1234 latlon 200 1234 $PI16 $R3PI4 0.0
D_s2024 latlon 200 2024 $PI16 $R3PI4 0.0
X_spectral spectral 10 - - - -
X_ico icosahedral 10 - - - -
X_cube cubed_sphere 10 - - - -
X_latlon_flat latlon 10 - - - -
ARMS
run_arm () {  # gpu line...
  local gpu=$1 tag=$2 grid=$3 days=$4 seed=$5 zf=$6 rm=$7 lc=$8
  local env=(CUDA_VISIBLE_DEVICES=$gpu)
  [ "$seed" != "-" ] && env+=(LEGOESM_TOPO_SEED=$seed)
  [ "$zf" != "-" ] && env+=(LEGOESM_PROBE_ZETA_FRAC=$zf)
  [ "$rm" != "-" ] && env+=(LEGOESM_PROBE_RM_FRAC=$rm)
  [ "$lc" != "-" ] && env+=(LEGOESM_PROBE_LAT_C=$lc)
  [ "$tag" = "X_latlon_flat" ] && env+=(LEGOESM_TOPO_H0=0)
  env "${env[@]}" $PY -u scripts/cluster/dycore_1029/mountain_override_arm.py --only hydro --test "=held_suarez_topo" \
      --grid $grid --days $days --no-cross-grid-plots -o $OUT/$tag > $OUT/$tag.log 2>&1
}
# 4 GPU workers pulling from the arm list
for g in 0 1 2 3; do
  ( awk -v g=$g 'NR%4==g' $OUT/arms.txt | while read -r line; do run_arm $g $line; done ) &
done
wait
for f in $OUT/*.log; do echo "== $(basename $f .log): $(grep -E 'PROBE' $f | head -1 | cut -c1-160)"; grep -E "BLOWUP at|^\s+\*?\*?(PASS|FAIL|XFAIL|XPASS)" $f | tail -2 | cut -c1-220; done
echo DONE $(date -u)
