#!/bin/bash
#SBATCH --job-name=equpwell
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/equpwell_%j.out
#
# IS THE EQUATORIAL ASCENT ACTUALLY TOO WEAK?  The direct measurement.
#
# The isopycnal probe (7f0523105) showed that between day 15 and day 30 NEMO
# LIFTS every density surface in the cold tongue by 6-12 m while ours SINK
# below about 45 m, with the temperature ON those surfaces agreeing to a few
# tenths.  From that I inferred "the upwelling that removes heat is too weak".
# That is an INFERENCE from a displacement.  This card measures the vertical
# velocity itself, so the inference either survives a direct test or does not.
#
# It also answers the two questions a band mean cannot, both of which this
# probe already implements and neither of which I had to build:
#   * DISPLACED vs ABSENT ascent -- --meridional-lon prints w(lat) at four
#     longitudes, because upwelling that moved off the equator and upwelling
#     that stopped look identical in a 2S-2N mean and need different fixes;
#   * the CARRIER -- --nemo-vfile prints the meridional surface current, the
#     Ekman divergence that drives the ascent, and --nemo-ufile prints the
#     undercurrent.  An absent undercurrent and an absent ascent need
#     different fixes too.
#
# THE NEMO VARIABLE IS `wo`, NOT THE PROBE'S DEFAULT.  --nemo-w-var defaults to
# `wocetr_eff`, which is an effective vertical TRANSPORT in m3/s.  The active
# OCE_W output group in this configuration (field_def_nemo-oce.xml:1429) writes
# `woce` under the name `wo`, a VELOCITY in m/s, and does not write
# wocetr_eff at all.  Taking the default would have compared m3/s against our
# m/s, or failed on a missing variable.  A default is not a record.
#
# PRE-REGISTERED READING, written before the job was submitted.
#
# The metric is the cold-tongue band mean of vertical velocity at ~50 m,
# 180-280E, |lat| <= 2, ours (instantaneous, day 30) against NEMO (5-day mean,
# record 5 = days 25-30).  The probe's own docstring sets the resolution: a
# 10-20% difference is inside the snapshot-versus-mean mismatch and MUST NOT be
# read; a factor of ~2 is outside it.
#
#   SIGNS DISAGREE (ours descends where NEMO ascends)
#       -> NOT read as physics on this evidence.  The probe says outright it
#          cannot separate a convention difference from a reversed cell, so
#          this outcome is reported as "the sign test is inconclusive" and the
#          next step is a convention check, not a mechanism.
#   OURS ASCENDS, WEAKER BY MORE THAN A FACTOR 1.5
#       -> the inference is CONFIRMED by direct measurement: the ascent is
#          too weak, and the size of the deficit is now a number.
#   OURS ASCENDS, WITHIN 20%
#       -> the inference is NOT SUPPORTED.  The mean ascent at 50 m is right,
#          so the sinking isopycnals come from somewhere else -- depth
#          structure below 50 m, or horizontal advection.  The next step is
#          then this same probe at a second depth, not a new hypothesis.
#   OURS ASCENDS, STRONGER BY MORE THAN A FACTOR 1.5
#       -> the inference is REFUTED and the isopycnal reading needs
#          re-examination before anything is built on it.
#   BETWEEN 20% AND A FACTOR 1.5
#       -> the labelled grey zone.  Inside the instrument's own stated
#          uncertainty, reported as UNREADABLE, not as a small effect.  This
#          band is named explicitly so the answer cannot fall into a gap.
#
# THE INSTRUMENT CHECK COMES FIRST AND IS DECIDED BEFORE THE COMPARISON IS
# READ.  Our side is one instant, NEMO's is a five-day mean -- the same
# asymmetry that made the wind-stress channel unusable, where our
# instantaneous value fell by a factor 2.2 between two samples five days
# apart.  So this card runs the identical measurement at day 25 against
# record 4 as well.  If OUR band mean moves by more than 30% between day 25
# and day 30, our instantaneous vertical velocity is too variable to compare
# against a five-day mean and the channel is reported UNUSABLE, exactly as
# the stress channel was, whatever the gap to NEMO turns out to be.
#
# ONE GRID ON PURPOSE: the tripole lane is the only one running NEMO's own
# lateral viscosity (d70eb5cf4), so putting MPAS or FESOM beside it here would
# be a confounded comparison.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/trp_orca1faithvisc_d30
G=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY
GT=$G/ORCA1_5d_20000101_20000331_grid_T.nc
GW=$G/ORCA1_5d_20000101_20000331_grid_W.nc
GU=$G/ORCA1_5d_20000101_20000331_grid_U.nc
GV=$G/ORCA1_5d_20000101_20000331_grid_V.nc
for f in "$GT" "$GW" "$GU" "$GV"; do
  [ -e "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done

RC=0
one () {   # one <day> <snapshot>
  local day="$1" snap="$2"
  local rec=$(( day / 5 - 1 ))            # the record ENDING on day D is D/5-1
  echo; echo "########## day $day, NEMO record $rec (days $((rec*5))-$(((rec+1)*5)))"
  echo "snapshot: $snap"
  [ -e "$snap" ] || { echo "FATAL: missing $snap"; RC=1; return; }
  $PY -u scripts/validate/ocean_fidelity/equatorial_thermocline.py \
    --legoesm-snapshot "$snap" --label "tripole_d${day}" \
    --nemo-gridt "$GT" --nemo-t-recs "${rec}:$((rec+1))" \
    --nemo-wfile "$GW" --nemo-w-var wo --nemo-w-recs "${rec}:$((rec+1))" \
    --w-depth-m 50 \
    --nemo-ufile "$GU" --nemo-vfile "$GV" --v-depth-m 10 \
    --meridional-lon 200,220,240,260 \
    --lat-halfwidth 2.0
  local rc=$?
  echo "=== day $day rc=$rc"
  [ $rc -eq 0 ] || RC=1
}

one 25 $R/snapshot_day0025.npz
one 30 $R/snapshot_final.npz

echo; echo "=== ALL DONE (rc=$RC) ==="
exit $RC
