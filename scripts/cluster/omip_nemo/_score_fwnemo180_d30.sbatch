#!/bin/bash
#SBATCH --job-name=score_fw180
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/score_fw180_%j.out
# Real-freshwater 180d arms vs NEMO at matched day 30 (same instrument as the
# unified virtual-salt baselines -> freshwater stack is the ONE variable), plus
# the MPAS round-off twin vs its sibling at day 30 (cold-start divergence bound,
# GLM blocker) via the three-way tool with the twin as the second arm.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_wt; cd $ROOT || exit 1
echo "HEAD $(git rev-parse --short HEAD)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu
N=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00
GW_T=$N/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
R=$ROOT/results/omip_nemo
declare -A ARM=( [mpasfw180]=mpas_fwnemo_180d [trpfw180]=trp_fwnemo_180d [mpasfw180twin]=mpas_fwnemo_180d_twin )
for g in mpasfw180 trpfw180 mpasfw180twin; do
  for pair in "0030 5" "0060 11" "0090 17"; do
    set -- $pair; day=$1; idx=$2
    s=$R/${ARM[$g]}/snapshot_day$day.npz
    [ -r "$s" ] || { echo "SKIP $g d$day: no snapshot yet"; continue; }
    [ -f $R/compare_${g}_d$day/report.json ] && { echo "have $g d$day"; continue; }
    echo; echo "########## $g day $day vs GATEWAY 5d idx $idx"
    $PY scripts/validate/compare_omip_nemo.py \
      --legoesm-snapshot "$s" --nemo-gridt "$GW_T" --nemo-time-idx $idx \
      --freeze-clamp-C -1.8 --sss-faithful --grid-label ${g}_d${day} \
      --output-dir $R/compare_${g}_d$day 2>&1 | grep -E "^\[|rmse|bias|WARNING" | head -30
  done
done
for pair in "0030 5" "0060 11" "0090 17"; do
  set -- $pair; day=$1; idx=$2
  a=$R/mpas_fwnemo_180d/snapshot_day$day.npz; b=$R/mpas_fwnemo_180d_twin/snapshot_day$day.npz
  [ -r "$a" ] && [ -r "$b" ] || continue
  echo; echo "########## TWIN d$day: main vs twin (grid-vs-grid rmse = round-off divergence)"
  $PY -u scripts/validate/ocean_fidelity/compare_three_way_nemo.py \
    --tripole "$a" --mpas "$b" --label-tripole main --label-mpas twin \
    --nemo-gridt "$GW_T" --nemo-time-idx $idx \
    --out-dir $R/twin_mpasfw_d$day 2>&1 | grep -E "main_vs_twin|global" | head -12
done
echo DONE
