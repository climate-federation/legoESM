#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=viscscore
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/viscscore_%j.out
#
# WHAT THE VISCOSITY FIX DID TO SST, SSS AND MIXED-LAYER DEPTH.
#
# Every transport number so far says the fix removed about a quarter of the
# Florida error and nothing else moved much.  Transports are one slice.  The
# campaign's actual targets -- Pacific SST in the ENSO region, Southern Ocean
# SST and mixed layer, polar salinity -- have NEVER been scored on a run that
# applies NEMO's lateral viscosity correctly, because until 2026-09-19 no run
# did.  The fixed run is therefore a NEW BASELINE whose scorecard is unknown.
#
# BOTH ARMS GO THROUGH ONE SCORER INVOCATION.  The comparator takes two
# legoESM slots and intersects their wet footprints into ONE common mask, so
# putting the halved run in slot A and the fixed run in slot B scores both
# against the same oracle record on exactly the same cells.  Running them as
# two separate invocations would mask them separately and the two scorecards
# would not be comparable -- the confound this card exists to avoid.
#
# ONE VARIABLE: the two snapshots differ only in the lateral viscosity the
# Laplacian applied.  Same card, same forcing, same initial condition, same
# thirty days.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

RES=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo
NEMOT=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc

# The record ENDING on the scored day, derived from the day rather than typed,
# and then checked by the scorer itself: --expect-day makes a mismatch a hard
# error as of 2026-09-19, so a card that got this wrong would now stop instead
# of quietly scoring against day 90.
DAY=${DAY:-30}
REC=$(( DAY / 5 - 1 ))

# DAY=25 re-runs the identical comparison one record earlier.  A structural
# disagreement reproduces; a single-snapshot fluke does not.  Note what this
# does NOT settle: both days compare an INSTANTANEOUS model snapshot against a
# FIVE-DAY MEAN oracle field, and mixed-layer depth is the diagnostic most
# sensitive to that asymmetry because it responds to individual storms.  Two
# days sharing the same asymmetry cannot measure it.
DD=$(printf '%04d' $DAY)
HALF=$RES/trp_orca1faithtint_d30/snapshot_day$DD.npz      # viscosity halved
FULL=$RES/trp_orca1faithvisc_d30/snapshot_day$DD.npz      # NEMO's own value
[ -e "$FULL" ] || FULL=$RES/trp_orca1faithvisc_d30/snapshot_final.npz
OUT=$RES/score_viscpair_d$DD

for f in "$NEMOT" "$HALF" "$FULL"; do
  [ -e "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done
echo "half: $HALF"; echo "full: $FULL"; echo "nemo record: $REC (days $((REC*5))-$(((REC+1)*5)))"

JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 $PY -u \
  scripts/validate/ocean_fidelity/compare_three_way_nemo.py \
  --tripole "$HALF" --mpas "$FULL" --nemo-gridt "$NEMOT" \
  --expect-day "$DAY" --nemo-time-idx "$REC" \
  --label-tripole halfvisc --label-mpas fullvisc \
  --out-dir "$OUT"
RC=$?
echo "=== scorer rc=$RC ==="
echo "OUT=$OUT"
echo DONE
exit $RC
