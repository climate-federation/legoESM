#!/bin/bash
#SBATCH --job-name=mxlflux3
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/mxlflux3_%j.out
#
# Third attempt, and this one was read out of the source rather than guessed.
# --heat-budget is a SUB-flag: the budget block lives inside the section gated
# by --box-budget 'lo,hi', so passing --heat-budget alone printed nothing
# twice. The tell was that the section's own mass-budget header ("=== ... Sv")
# never appeared either.
#
# NO WINDOW IS CHOSEN HERE. The layer is the quantity in dispute -- the
# mixing bases sit at 14.70 m (mixing length 3), 15.12 m (length 4) and 17.95 m
# (NEMO), while the probe's own default layer is 50-150 m, which would answer
# a different question -- so BOTH are run and both reported. The longitude box
# is 200-260E, the same box the K-profile discriminator used, so the two
# instruments are not looking at different water.
set -uo pipefail
R=/burg-archive/glab/users/pg2328/legoESM; cd $R || exit 1
M=/burg-archive/glab/users/pg2328/_omip_merge
export PYTHONPATH="$R/src:$(ls -d $R/packages/*/ | sed 's:/$::' | tr '\n' ':')$R:$R/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
GT=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
CTL=$R/results/omip_nemo/trp_faithz0_d30/snapshot_day0005.npz
ARM=$R/results/omip_nemo/trp_mxl4_d5/snapshot_final.npz
for f in "$GT" "$CTL" "$ARM"; do
  [ -r "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done

rc=0
for LAYER in 10,25 50,150; do
  for pair in "mxl3_ctl:$CTL" "mxl4_arm:$ARM"; do
    LAB=${pair%%:*}; SNAP=${pair#*:}
    echo "############ $LAB  layer=$LAYER ############"
    $PY -u $M/scripts/validate/ocean_fidelity/equatorial_thermocline.py \
      --legoesm-snapshot "$SNAP" --nemo-gridt "$GT" --nemo-t-recs 0:1 \
      --label "${LAB}_${LAYER}" --box-budget 200,260 --budget-layer "$LAYER" \
      --heat-budget 2>&1 | tail -80
    r=${PIPESTATUS[0]}; echo "### $LAB $LAYER rc=$r"; [ "$r" -eq 0 ] || rc=$r
  done
done
echo DONE
exit $rc
