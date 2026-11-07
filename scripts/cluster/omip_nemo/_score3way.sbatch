#!/bin/bash
#SBATCH --job-name=score3way
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/score3way_%j.out
#
# CPU-only scoring pass (no GPU): put the matched cross-grid pair and the
# real-freshwater arms on the NEMO oracle, with maps + zonal means + biases.
#   1. three-way tripole/MPAS/NEMO scorecard (xgrid matched pair, day 30)
#   2. per-arm compare_omip_nemo verdicts (same protocol as the run history)
#   3. real-freshwater arms (day 90 vs NEMO month 3)
# Every arm's caveats travel in its own report.json.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM
cd $REPO || exit 1
echo "=== HEAD $(git -C $REPO rev-parse HEAD)"
export PYTHONPATH="$REPO/src:$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
R=$REPO/results/omip_nemo
GT=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_REF/ORCA1_1m_20000101_20041231_grid_T.nc
[ -r "$GT" ] || { echo "FATAL: NEMO grid_T unreadable"; exit 1; }

echo; echo "########## 1. THREE-WAY xgrid matched pair, day 30 vs NEMO month 1"
$PY -u scripts/validate/ocean_fidelity/compare_three_way_nemo.py \
  --tripole $R/xgrid_trp_d30/snapshot_final.npz \
  --mpas    $R/xgrid_mpas_d30/snapshot_final.npz \
  --nemo-gridt "$GT" --nemo-month 1 \
  --label-tripole "tripole eORCA1" --label-mpas "MPAS ico7" \
  --out-dir $R/xgrid3_d30_m01
rc1=$?; echo "=== three-way rc=$rc1"

echo; echo "########## 2. PER-ARM verdicts (history-comparable protocol)"
for arm in trp mpas; do
  $PY -u scripts/validate/compare_omip_nemo.py \
    --legoesm-snapshot $R/xgrid_${arm}_d30/snapshot_final.npz \
    --nemo-gridt "$GT" --nemo-month 1 --freeze-clamp-C -1.8 --sss-faithful \
    --grid-label "xgrid ${arm} d30" \
    --output-dir $R/compare_xgrid_${arm}_d30_m01
  echo "=== arm $arm rc=$?"
done

echo; echo "########## 3. REAL-FRESHWATER arms, day 90 vs NEMO month 3"
for arm in fwreal fwrealnn; do
  s=$R/nemolev_trp_${arm}_d90/snapshot_day0090.npz
  if [ -r "$s" ]; then
    $PY -u scripts/validate/compare_omip_nemo.py \
      --legoesm-snapshot "$s" \
      --nemo-gridt "$GT" --nemo-month 3 --freeze-clamp-C -1.8 --sss-faithful \
      --grid-label "tripole ${arm} d90" \
      --output-dir $R/compare_${arm}_d90_m03
    echo "=== $arm rc=$?"
  else
    echo "SKIP $arm: $s not readable"
  fi
done
echo "=== ALL DONE ==="
