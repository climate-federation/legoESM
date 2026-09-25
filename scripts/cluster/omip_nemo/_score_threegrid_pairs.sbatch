#!/bin/bash
#SBATCH --job-name=threegrid
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/threegrid_%j.out
# THREE-GRID comparison at matched days 30/60/90 vs GATEWAY recs 5/11/17:
# pairwise runs of the existing two-arm three-way tool (no new code), so every
# grid is scored vs NEMO AND vs each other grid on the common 1-deg mask.
# Arms: tripole trp_unified180, MPAS mpas_unified_180d, FESOM fesom_b5_d90.
# 2026-09-04: the first pass scored trp_visc180/mpas_fullharm_180d, which
# PREDATE the SI3 lead-heat fix (116071f74) and carry SST down to -10 C under
# Arctic ice -> a spurious "northern cold bias" on both grids. The unified
# arms select --ice-lead-freeze-source nemo_qlead (SST min -1.8 C).
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu
GW_T=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
R=$ROOT/results/omip_nemo
for pair in "30 5" "60 11" "90 17"; do
  set -- $pair; printf -v day %04d $1; idx=$2
  t=$R/trp_unified180/snapshot_day$day.npz
  m=$R/mpas_unified_180d/snapshot_day$day.npz
  f=$R/fesom_b5_d90/snapshot_day$day.npz
  for combo in "tripole:$t:MPAS:$m" "tripole:$t:FESOM:$f" "MPAS:$m:FESOM:$f"; do
    IFS=: read la pa lb pb <<< "$combo"
    echo; echo "########## d$day  $la vs $lb vs NEMO rec $idx"
    $PY -u scripts/validate/ocean_fidelity/compare_three_way_nemo.py \
      --tripole "$pa" --mpas "$pb" --label-tripole "$la" --label-mpas "$lb" \
      --nemo-gridt "$GW_T" --nemo-time-idx $idx \
      --out-dir $R/threegrid_uni_${la}_${lb}_d$day 2>&1 | grep -E "^\[|rmse|bias|corr|SST|SSS|MLD" | head -n 40
  done
done
echo DONE
