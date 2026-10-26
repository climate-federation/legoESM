#!/bin/bash
#SBATCH --job-name=score180
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/score180_%j.out
#
# long180 drift test vs NEMO's matched GATEWAY cold start (5-day means):
# rec 5 = days 26-30, rec 11 = days 56-60, rec 17 = days 86-90.  The absolute
# nino3 trajectory is seasonally confounded (Mar-Apr is the cold tongue's warm
# season); the drift verdict must be the BIAS at matched records.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu

GW_T=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc

score () {   # day  record  label
  local day=$1 rec=$2 label=$3
  local snap=$ROOT/results/omip_nemo/nemolev_trp_long180/snapshot_day${day}.npz
  echo
  echo "############ long180 day $day vs GATEWAY rec $rec ############"
  [ -f "$snap" ] || { echo "SKIP: no $snap"; return; }
  $PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$snap" \
    --nemo-gridt "$GW_T" --nemo-time-idx "$rec" --freeze-clamp-C -1.8 \
    --sss-faithful --grid-label "long180_${label}" \
    --output-dir $ROOT/results/omip_nemo/compare_gw_long180_${label} 2>&1 | tail -38
  echo "### rc=$?"
}

score 0030 5  d30
score 0060 11 d60
score 0090 17 d90
echo DONE
