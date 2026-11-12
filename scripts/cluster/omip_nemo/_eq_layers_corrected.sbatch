#!/bin/bash
#SBATCH --job-name=eqlayers
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/eqlayers_%j.out
#
# WHERE DOES THE HEAT GO, asked of the CORRECTED run.
#
# The surface budget just eliminated forcing: we put about 30 W/m2 LESS heat
# into the cold tongue than NEMO at both day 25 and day 30 and are still
# 1.15 degC too warm, so the defect is in how heat LEAVES that box, not how it
# arrives.  This is the next term of the same budget, and the probe for it
# already exists: does the cooling come from VERTICAL REDISTRIBUTION -- the
# top slab cooling while 50 to 150 m warms -- or from NET LOSS, every slab
# cooling together?  Those have different causes.  Upwelling and the
# undercurrent redistribute; mixing and advection out of the box remove.
#
# ONE GRID ON PURPOSE.  The tripole lane is the only one running NEMO's own
# lateral viscosity; MPAS uses a uniform 100000 m2/s and FESOM's value is not
# even recorded (d70eb5cf4).  Putting them beside each other here would be a
# confounded comparison, so this card asks the question of the corrected
# tripole run against the oracle alone and says so rather than filling the
# panel.
#
# The early and late records are DERIVED from the days, never typed: the
# record ending on day D is D/5-1, so day 15 is record 2 and day 30 record 5.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/trp_orca1faithvisc_d30
GT=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc

DAY_LATE=30 ; DAY_EARLY=15
REC_LATE=$(( DAY_LATE / 5 - 1 )) ; REC_EARLY=$(( DAY_EARLY / 5 - 1 ))
LATE=$R/snapshot_final.npz              # day 30, rewritten by leg 2 at 18:53
EARLY=$R/snapshot_leg1end_day15.npz     # leg 1's day 15, preserved on purpose
for f in "$GT" "$LATE" "$EARLY"; do
  [ -e "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done
echo "early: $EARLY (day $DAY_EARLY, NEMO record $REC_EARLY)"
echo "late : $LATE (day $DAY_LATE, NEMO record $REC_LATE)"

echo; echo "########## the probe's own unit test first"
$PY -m pytest -q tests/ocean/unit/test_equatorial_thermocline_structure.py 2>&1 | tail -2

echo; echo "########## equatorial layer structure, corrected tripole vs NEMO"
$PY -u scripts/validate/ocean_fidelity/equatorial_thermocline_structure.py \
  --snapshot "$LATE" --label tripole_fullvisc \
  --snapshot-early "$EARLY" \
  --nemo-gridt "$GT" --nemo-rec $REC_LATE --nemo-rec-early $REC_EARLY \
  --isopycnal --surface-profile 2>&1 | grep -vE "Warning|warn"
RC=${PIPESTATUS[0]}
echo "=== rc=$RC ==="
exit $RC
