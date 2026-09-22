#!/bin/bash
#SBATCH --job-name=equpw
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/equpw_%j.out
#
# IS THE EQUATORIAL WARM BIAS AN ASCENT DEFICIT OR AN UNDERCURRENT DEFICIT?
#
# Tonight's surface-flux measurement makes this the right question. Our net
# heat into nino3 is 30-40 W/m2 BELOW NEMO's while we sit 1.34 K WARMER, so
# the surface is a heat SINK relative to the oracle and the warmth cannot be
# arriving through the air-sea interface. It is dynamical. That also matches
# the standing budget result: at the mixing base the column balance is an
# advective near-cancellation, -75.7 against +72.4 K/month, where a 3% error
# in either term is the whole bias.
#
# An earlier attempt (9907300) failed rc=1 precisely because the NEMO half was
# not supplied. Those files all exist, so this supplies them:
#   --nemo-wfile  the 5-day grid_W, whose `wo` is a VELOCITY and is comparable
#                 to our mass_flux_w directly (the probe's other option,
#                 `wocetr_eff`, is a TRANSPORT and would need an area divide)
#   --nemo-ufile  the 5-day grid_U, enabling the undercurrent profile
#   --nemo-vfile  the 5-day grid_V, the Ekman-divergence carrier
# An absent ascent and an absent undercurrent need DIFFERENT fixes, which is
# why both blocks are wanted in one pass.
#
# THE CONTROL ARM ONLY. This asks how OUR production-faithful configuration
# (nemo_z0) compares with the oracle; it is not the flag A/B, and mixing the
# two questions into one card is how a comparison stops being controlled.
#
# BOTH WINDOWS, AND THE ORDINARY ONE IS PRIMARY. Day 25 is a normal window.
# Day 30 sits inside a wind relaxation event -- measured, not assumed: our
# nino3 stress falls 0.0780 to 0.0356 and NEMO's 0.0834 to 0.0533 across it --
# so the equatorial circulation there is in transient adjustment and is the
# wrong place to read a mean ascent. Records follow the window: 4:5 is day 25,
# 5:6 is day 30.
set -uo pipefail
M=/burg-archive/glab/users/pg2328/_fluxacc_wt; cd $M || exit 1
export PYTHONPATH="$M/src:$(ls -d $M/packages/*/ | sed 's:/$::' | tr '\n' ':')$M:$M/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo
G=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY
STEM=$G/ORCA1_5d_20000101_20000331
MESH=/burg-archive/glab/users/pg2328/_omip_merge/data/grids/eORCA1.2_mesh_mask.nc
for f in ${STEM}_grid_T.nc ${STEM}_grid_U.nc ${STEM}_grid_V.nc ${STEM}_grid_W.nc "$MESH"; do
  [ -r "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done

rc=0
for pair in "d25:4:5:snapshot_day0025.npz" "d30:5:6:snapshot_final.npz"; do
  TAG=${pair%%:*}; rest=${pair#*:}
  R0=${rest%%:*}; rest=${rest#*:}
  R1=${rest%%:*}; SNAPNAME=${rest#*:}
  SNAP=$R/trp_faithz0_d30/$SNAPNAME
  [ -r "$SNAP" ] || { echo "FATAL: missing $SNAP"; rc=1; continue; }
  echo "############ control nemo_z0  $TAG  (NEMO recs $R0:$R1) ############"
  $PY -u scripts/validate/ocean_fidelity/equatorial_thermocline.py \
    --legoesm-snapshot "$SNAP" --nemo-gridt ${STEM}_grid_T.nc \
    --nemo-t-recs "$R0:$R1" --label "ctl_$TAG" \
    --nemo-wfile ${STEM}_grid_W.nc --nemo-w-var wo --nemo-w-recs "$R0:$R1" \
    --nemo-ufile ${STEM}_grid_U.nc --nemo-vfile ${STEM}_grid_V.nc \
    --mesh-mask "$MESH" 2>&1 | tail -120
  r=${PIPESTATUS[0]}; echo "### ctl $TAG rc=$r"; [ "$r" -eq 0 ] || rc=$r
done
echo "=== card exit: $rc ==="
exit $rc
