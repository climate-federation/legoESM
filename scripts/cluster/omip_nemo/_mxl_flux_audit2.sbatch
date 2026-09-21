#!/bin/bash
#SBATCH --job-name=mxlflux2
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/mxlflux2_%j.out
#
# The first flux audit (9901417) printed thermocline structure and no flux,
# because the diffusive term is gated behind --heat-budget and I did not pass
# it. A probe that runs clean is not a probe that answered.
#
# Worth keeping from that run: Z20 gap and tilt are IDENTICAL between mixing
# length 3 and 4 (-1.0 m over 8/8 longitudes, 104% of NEMO's tilt, both arms),
# so the thermocline itself did not move.
#
# The question here is whether the diffusivity change that IS present -- down
# 10-12% above 13 m, up 6-18% at 15-21 m -- reaches the heat budget, computed
# per column before averaging.
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
for pair in "mxl3_ctl:$CTL" "mxl4_arm:$ARM"; do
  LAB=${pair%%:*}; SNAP=${pair#*:}
  echo "############ $LAB ############"
  $PY -u $M/scripts/validate/ocean_fidelity/equatorial_thermocline.py \
    --legoesm-snapshot "$SNAP" --nemo-gridt "$GT" --nemo-t-recs 0:1 \
    --label "$LAB" --heat-budget 2>&1 | tail -260
  r=${PIPESTATUS[0]}; echo "### $LAB rc=$r"; [ "$r" -eq 0 ] || rc=$r
done
echo DONE
exit $rc
