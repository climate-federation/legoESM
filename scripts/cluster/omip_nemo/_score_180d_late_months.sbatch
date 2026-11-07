#!/bin/bash
#SBATCH --job-name=score180_late
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/score180_late_%j.out
#
# Score the 180d matched pair PAST day 90. RUN_GATEWAY (5d, 18 recs) ends at
# day 90; RUN_REF is the SAME namelist/IC/forcing run for 5 years with MONTHLY
# output, and its first three months are BIT-IDENTICAL to RUN_GATEWAY (checked
# 2026-09-02: max|tos,sos,mldr10_1 diff| = 0.0 on Jan and Mar). So RUN_REF
# year-1 monthly means idx 3/4/5 (Apr/May/Jun, centred ~day 106/136/166) are
# the matched cold-start oracle for our day 105/135/165 snapshots.
# RESIDUAL: our snapshots are INSTANTANEOUS, NEMO records are MONTHLY means.
# Also: three-way (tripole/MPAS/NEMO) maps + zonal means at d30/d90/d105.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
echo "HEAD $(git rev-parse --short HEAD)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu
N=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00
GW_T=$N/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
REF_T=$N/RUN_REF/ORCA1_1m_20000101_20041231_grid_T.nc
R=$ROOT/results/omip_nemo
declare -A ARM=( [trpvisc180]=trp_visc180 [mpasfullharm180]=mpas_fullharm_180d )
for g in trpvisc180 mpasfullharm180; do
  for pair in "105 3" "135 4" "165 5"; do
    set -- $pair; day=$1; idx=$2
    s=$R/${ARM[$g]}/snapshot_day0$day.npz
    [ -r "$s" ] || { echo "SKIP $g d$day: no snapshot yet"; continue; }
    echo; echo "########## $g day $day vs RUN_REF 1m idx $idx"
    $PY scripts/validate/compare_omip_nemo.py \
      --legoesm-snapshot "$s" --nemo-gridt "$REF_T" --nemo-time-idx $idx \
      --freeze-clamp-C -1.8 --sss-faithful --grid-label ${g}_d${day} \
      --output-dir $R/compare_${g}_d0${day} 2>&1 | tail -38
  done
done
echo; echo "########## THREE-WAY maps + zonal means"
for pair in "30 $GW_T 5" "90 $GW_T 17" "105 $REF_T 3"; do
  set -- $pair; printf -v day %04d $1; f=$2; idx=$3
  t=$R/trp_visc180/snapshot_day$day.npz; m=$R/mpas_fullharm_180d/snapshot_day$day.npz
  [ -r "$t" ] && [ -r "$m" ] || { echo "SKIP three-way d$day"; continue; }
  $PY -u scripts/validate/ocean_fidelity/compare_three_way_nemo.py \
    --tripole "$t" --mpas "$m" --nemo-gridt "$f" --nemo-time-idx $idx \
    --label-tripole "tripole visc180" --label-mpas "MPAS fullharm180" \
    --out-dir $R/threeway180_d$day 2>&1 | tail -30
  echo "=== three-way d$day rc=$?"
done
echo DONE
