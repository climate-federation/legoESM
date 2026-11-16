#!/bin/bash
#SBATCH --job-name=eqprandtl
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/eqprandtl_%j.out
#
# DOES OUR CLOSURE MIX MOMENTUM TOO DEEP?  The carrier of the undercurrent gap.
#
# Job 9884840 measured the shape of the error rather than just its size: at
# 220E our equatorial zonal velocity is NEMO's profile displaced roughly 0.35
# to 0.42 m/s westward, nearly uniformly from the surface down to about 150 m,
# and the displacement vanishes by 200 m.  The undercurrent core is a third to
# a half of NEMO's and sits 25-30 m too deep, while the thermocline depth and
# its east-west tilt now agree with the oracle to within a few percent -- so
# the density structure and the pressure gradient driving the jet are right
# and the flow's response to them is not.
#
# A vertical viscosity that is too large does exactly that: it carries the
# surface westward momentum downward, thickens the westward layer and erodes
# the eastward core underneath it.  This card measures that coefficient
# instead of arguing about it.
#
# NOTHING NEW IS WRITTEN.  equatorial_diffusivity_oracle.py already exists and
# was built for this question -- its own docstring opens on an undercurrent
# that is too weak under a correct pressure gradient, and names the turbulent
# Prandtl number as the single quantity that would explain the jet and the
# thermocline together.  It reads NEMO's avm, avt, avs and bn2 straight out of
# the five-day grid_W file, so the oracle's profile is a file read rather than
# a remembered number, and it prints the oracle on its own before any
# comparison so the target cannot move when our side does.  It reads our side
# from K_M_diag and K_H_diag, which the snapshots already carry -- verified
# before submitting: shape (332, 362, 74), one value per interface between the
# 75 centres, magnitudes 1e-6 to 1e-1 m2/s, which is the vertical range and
# not the lateral one.  No new run is needed.
#
# PRE-REGISTERED READING, written before the job was submitted.
#
# The metric is the equatorial band profile of vertical viscosity, |lat| <= 2,
# 200-260E, over 0-300 m, ours against NEMO's avm on the matched record.  The
# layer that matters is 50-150 m, where the undercurrent core lives and where
# the westward displacement dies away.
#
#   OURS EXCEEDS NEMO'S BY MORE THAN A FACTOR 2 THROUGH 50-150 m
#       -> CONFIRMS the carrier: momentum is being mixed too deep, which is
#          the mechanism that produces a uniform westward offset over a thick
#          layer rather than a differently shaped jet.
#   OURS WITHIN 30% OF NEMO'S
#       -> the vertical viscosity is NOT the carrier.  The momentum then has
#          to be arriving wrong at the surface, and the next instrument is the
#          wind stress -- which is an OPEN ASK, because our stress is one
#          instant against NEMO's five-day mean and is unusable until our side
#          is accumulated the same way.
#   OURS BELOW NEMO'S
#       -> REFUTES the too-much-mixing mechanism outright.  A weak undercurrent
#          under a LOW viscosity points at the momentum source, not at its
#          redistribution, and that is a different investigation.
#   BETWEEN 30% AND A FACTOR 2
#       -> the labelled grey zone.  Named so the answer cannot fall into a gap
#          between bands, and reported as unreadable on this instrument alone.
#
# THE INSTRUMENT CHECK IS DECIDED FIRST AND IT HAS TWO HALVES, because the
# last two channels died on exactly this.  Our side is one instant and NEMO's
# is a five-day mean.
#   (a) OUR OWN 50-150 m band mean at day 25 against day 30.  More than 30%
#       and our side is too variable to compare, same threshold the vertical
#       velocity was held to.
#   (b) NEMO'S OWN record 4 against record 5.  This is the half that actually
#       killed the vertical-velocity channel: ours moved -7% there while the
#       ORACLE moved -42%, which is larger than the factor 1.5 that channel
#       set out to resolve.  Whatever NEMO's own motion turns out to be here
#       is the resolution floor, and a gap smaller than it CANNOT be read no
#       matter how stable our side is.
#
# ONE GRID ON PURPOSE: the tripole is the only lane on NEMO's own lateral
# viscosity, so the other two would be a confounded comparison here.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/trp_orca1faithvisc_d30
GW=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_W.nc
[ -e "$GW" ] || { echo "FATAL: missing $GW"; exit 1; }

RC=0
one () {   # one <day> <snapshot>
  local day="$1" snap="$2"
  local rec=$(( day / 5 - 1 ))            # the record ENDING on day D is D/5-1
  echo; echo "########## day $day, NEMO record $rec (days $((rec*5))-$(((rec+1)*5)))"
  echo "snapshot: $snap"
  [ -e "$snap" ] || { echo "FATAL: missing $snap"; RC=1; return; }
  $PY -u scripts/validate/ocean_fidelity/equatorial_diffusivity_oracle.py \
    --nemo-gridw "$GW" --rec $rec \
    --legoesm-snapshot "$snap" \
    --lat-halfwidth 2.0 --lon-west 200.0 --lon-east 260.0 --depth-max 300.0
  local rc=$?
  echo "=== day $day rc=$rc"
  [ $rc -eq 0 ] || RC=1
}

one 25 $R/snapshot_day0025.npz
one 30 $R/snapshot_final.npz

echo; echo "=== ALL DONE (rc=$RC) ==="
exit $RC
