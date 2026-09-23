#!/bin/bash
#SBATCH --job-name=score_uni60
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/score_uni60_%j.out
#
# Score the NEW harmonized (unified physics/BC/IC) 180d arms at matched days
# against RUN_GATEWAY, with the SAME protocol as the previous baseline's
# reports (compare_mpasfullharm180_d00{30,45,60}, compare_trpvisc180_d00*):
# same comparator, same --freeze-clamp-C -1.8 --sss-faithful, same 5-day
# GATEWAY record index (day/5 - 1). One variable: the code the run used.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_wt; cd $ROOT || exit 1
echo "HEAD $(git rev-parse --short HEAD)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu
N=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00
GW_T=$N/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
R=$ROOT/results/omip_nemo
declare -A ARM=( [mpasuni180]=mpas_unified_180d [trpuni180]=trp_unified180 )
for g in trpuni180; do
  for pair in "0060 11"; do
    set -- $pair; day=$1; idx=$2
    s=$R/${ARM[$g]}/snapshot_day$day.npz
    [ -r "$s" ] || { echo "SKIP $g d$day: no snapshot yet"; continue; }
    echo; echo "########## $g day $day vs GATEWAY 5d idx $idx"
    $PY scripts/validate/compare_omip_nemo.py \
      --legoesm-snapshot "$s" --nemo-gridt "$GW_T" --nemo-time-idx $idx \
      --freeze-clamp-C -1.8 --sss-faithful --grid-label ${g}_d${day} \
      --output-dir $R/compare_${g}_d$day 2>&1 | tail -40
  done
done
echo DONE
