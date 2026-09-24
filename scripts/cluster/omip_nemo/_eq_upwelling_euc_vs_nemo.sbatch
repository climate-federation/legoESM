#!/bin/bash
#SBATCH --job-name=equpw
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/equpw_%j.out
#
# IS THE EQUATORIAL WARM BIAS AN ASCENT DEFICIT OR AN UNDERCURRENT DEFICIT?
#
# RETRACTED, AND THE RETRACTION IS THE REASON THIS HEADER WAS REWRITTEN (R24,
# cd2c15d75). The card originally justified itself with: "our net heat into
# nino3 is 30-40 W/m2 BELOW NEMO's while we sit 1.34 K WARMER, so the surface
# is a heat SINK and the warmth must be dynamical." That inference is
# CIRCULAR. With a prescribed atmosphere the bulk formulae restore SST at
# about 38 W/m2 per K, so a warm ocean MUST lose extra heat -- 38 x 1.34 K
# predicts ~51 W/m2, MORE than the 30-40 observed. The heat deficit is the
# EXPECTED RESPONSE to being warm, not independent evidence of a dynamical
# cause, and the restoring coefficient itself is a tautology (R23): with a
# prescribed atmosphere it is simply our own bulk's slope.
#
# The card is still worth running, on the narrower ground that it MEASURES
# the circulation directly instead of inferring it. The standing budget
# result stands on its own: at the mixing base the column balance is an
# advective near-cancellation, -75.7 against +72.4 K/month, so a 3% error in
# either term is the whole bias.
#
# ONE FURTHER CAVEAT ON WHAT THIS CARD CAN CONCLUDE: the upwelling comparison
# may be UNINFORMATIVE (also R24). Equatorial ascent is Ekman divergence, and
# our wind stress is MATCHED to NEMO's by construction, so a ratio near 1.00
# is what a locked input predicts and validates nothing. The undercurrent
# blocks do not share that defect.
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
# ALL SIX MATCHED DAYS, because a single day cannot tell an initial-condition
# offset from a momentum sink. GLM's discriminator: an offset present from the
# first day means the ICs differ and nothing is being removed; ours
# decelerating while NEMO's accelerates means a local sink; both steady at
# different values means an equilibrium difference. Day 30 stays in the series
# but its MEAN circulation is not readable -- it sits inside a measured wind
# relaxation (our nino3 stress falls 0.0780 to 0.0356, NEMO's 0.0834 to
# 0.0533) and its upwelling ratio reads 1.54 against 1.00 on every other day.
#
# THE RUN AND THE SAMPLE DAYS ARE NOW PARAMETERS, defaulting to exactly what
# this card has always done. RUN_DIR picks the run; PAIRS is a list of
# tag:rec0:rec1:snapshot entries. The 90-day extension of this same recipe
# lives in trp_faithz0_d90 -- a manifest diff of the two resolved configs
# shows ONE differing row, total_days, so its series continues this one
# rather than being a separate experiment -- and reading it answers the
# question a 30-day window cannot:
#
#   * the core-speed ratio RECOVERS toward 1 by day 90 -> the day-30 gap is
#     SPIN-UP RATE, the undercurrent simply equilibrates more slowly here
#     than in the oracle, and every day-30 equatorial conclusion in this
#     campaign is scoring rate rather than fidelity.
#   * it STAYS near 0.35 or falls further -> an equilibrium error, and the
#     vertical-momentum thread is worth the instrument work it needs.
#   * it OVERSHOOTS above about 1.3 -> adjustment overshoot, a third reading
#     that is neither of the above and would need its own explanation.
# Records follow the campaign convention REC = DAY/5 - 1, so day 90 is
# record 17 and the file's 18 records make "17:18" the last valid window.
RUN_DIR="${RUN_DIR:-trp_faithz0_d30}"
PAIRS="${PAIRS:-d05:0:1:snapshot_day0005.npz d10:1:2:snapshot_day0010.npz d15:2:3:snapshot_day0015.npz d20:3:4:snapshot_day0020.npz d25:4:5:snapshot_day0025.npz d30:5:6:snapshot_final.npz}"
echo "RUN_DIR=$RUN_DIR"
for pair in $PAIRS; do
  TAG=${pair%%:*}; rest=${pair#*:}
  R0=${rest%%:*}; rest=${rest#*:}
  R1=${rest%%:*}; SNAPNAME=${rest#*:}
  SNAP=$R/$RUN_DIR/$SNAPNAME
  [ -r "$SNAP" ] || { echo "FATAL: missing $SNAP"; rc=1; continue; }
  echo "############ control nemo_z0  $TAG  (NEMO recs $R0:$R1) ############"
  $PY -u scripts/validate/ocean_fidelity/equatorial_thermocline.py \
    --legoesm-snapshot "$SNAP" --nemo-gridt ${STEM}_grid_T.nc \
    --nemo-t-recs "$R0:$R1" --label "ctl_$TAG" \
    --nemo-wfile ${STEM}_grid_W.nc --nemo-w-var wo --nemo-w-recs "$R0:$R1" \
    --nemo-ufile ${STEM}_grid_U.nc --nemo-vfile ${STEM}_grid_V.nc \
    --mesh-mask "$MESH" 2>&1 | tail -400
  r=${PIPESTATUS[0]}; echo "### ctl $TAG rc=$r"; [ "$r" -eq 0 ] || rc=$r
done
echo "=== card exit: $rc ==="
exit $rc
