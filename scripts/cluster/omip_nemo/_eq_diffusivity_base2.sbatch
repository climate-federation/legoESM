#!/bin/bash
#SBATCH --job-name=eqdiffb2
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/eqdiffb2_%j.out
#
# Our vertical diffusivity/viscosity profile against NEMO's, on the base2
# (re-baseline) tripole arm, in the cold-tongue box where the shallow
# tropical cell lives.  DAY=15 -> NEMO record 2 (days 11-15) and our
# snapshot_day0015; DAY=30 -> record 5 and snapshot_day0030.  The fwfix arm
# is included as a within-model control: it differs from base2 only in the
# freshwater closure, so any K difference between the two is not the operator
# we are looking for.
set -uo pipefail
R=/burg-archive/glab/users/pg2328/legoESM; cd $R || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$R/src:$(ls -d $R/packages/*/ | sed 's:/$::' | tr '\n' ':')$R:$R/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
GW=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_W.nc
DAY=${DAY:-15}
case $DAY in 15) REC=2;; 30) REC=5;; *) echo "DAY must be 15 or 30"; exit 2;; esac
RES=/burg-archive/glab/users/pg2328/_base2_wt/results/omip_nemo
SNAPS="$RES/trp_base2_d30/snapshot_day00${DAY}.npz $RES/trp_fwfix_d30/snapshot_day00${DAY}.npz"
for s in $SNAPS; do [ -f "$s" ] || { echo "missing $s"; exit 3; }; done

for box in "200 260 cold-tongue" "140 180 warm-pool-control"; do
  set -- $box
  echo "############ day $DAY, NEMO rec $REC, $3 box $1-$2E ############"
  $PY -u scripts/validate/ocean_fidelity/equatorial_diffusivity_oracle.py \
    --nemo-gridw "$GW" --rec $REC --lon-west $1 --lon-east $2 \
    --legoesm-snapshot $SNAPS
  echo "### rc=$?"
done
echo DONE
