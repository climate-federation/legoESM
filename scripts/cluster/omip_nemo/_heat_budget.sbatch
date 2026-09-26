#!/bin/bash
#SBATCH --job-name=heatbud
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/heatbud_%j.out
# The codex+GLM discriminator (2026-09-08) between "the cold 20 m is advective"
# and "it is the mixing closure": all three heat-budget terms on both sides,
# same box, matched 5-day window, ours checked against its own observed dT/dt.
set -u
WT=/burg-archive/glab/users/pg2328/_omip_wt
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
NE=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00
GW=$NE/RUN_GATEWAY
RES=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/trp_base2_d30
echo "HEAD: $(git -C $WT log -1 --oneline)"
cd $WT || exit 1
export PYTHONPATH="$WT/src:$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT:$WT/scripts/validate"
$PY -m pytest tests/ocean/unit/test_equatorial_heat_budget.py -q 2>&1 | tail -3
AVG=$(for d in 11 12 13 14 15; do printf "%s/snapshot_day00%s.npz," "/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/trp_daily_d15" "$d"; done)
for LAYER in 15,25 0,25 50,150; do
  echo; echo "################ layer $LAYER ################"
  $PY scripts/validate/ocean_fidelity/equatorial_thermocline.py \
    --legoesm-snapshot /burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/trp_daily_d15/snapshot_day0015.npz \
    --heat-budget-average "${AVG%,}" \
    --heat-budget-prev /burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/trp_daily_d15/snapshot_day0010.npz \
    --nemo-gridt $GW/ORCA1_5d_20000101_20000331_grid_T.nc \
    --nemo-t-recs 2:3 --nemo-w-recs 2:3 \
    --nemo-wfile $GW/ORCA1_5d_20000101_20000331_grid_W.nc --nemo-w-var wo \
    --nemo-ufile $GW/ORCA1_5d_20000101_20000331_grid_U.nc \
    --nemo-vfile $GW/ORCA1_5d_20000101_20000331_grid_V.nc \
    --mesh-mask $WT/data/grids/eORCA1.2_mesh_mask.nc \
    --box-budget 220,240 --budget-layer $LAYER --budget-lat-halfwidth 2 \
    --heat-budget --label trp_base2_d30 2>&1 | grep -vE "^\[|Warning|warn" | tail -22
done
echo DONE
