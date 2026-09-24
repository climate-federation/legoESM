#!/bin/bash
#SBATCH --job-name=sharporig
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/sharporig_%j.out
#
# IS THE OVER-SHARP EQUATORIAL THERMOCLINE INHERITED OR GROWN?
#
# The cold-tongue attribution now runs: thermocline 3.3x too sharp -> zonal
# density gradient too strong -> baroclinic pressure gradient cancels the
# barotropic push -> net eastward force at 29% of the oracle's -> undercurrent
# at half strength -> equatorial ascent at 0.58 -> cold tongue 2.6 K too warm.
# Every link there is measured. The sharpness itself is not explained, and it
# is the only link left.
#
# THERE ARE EXACTLY TWO ORIGINS AND THEY NEED OPPOSITE FIXES. Both models
# start from the same WOA climatology, so if our front is already too sharp in
# the first days the defect is in how that climatology is put onto our 75
# levels, and no amount of mixing work will touch it. If the front starts
# right and steepens, the defect is in the physics that maintains it and the
# initial condition is innocent. Nothing measured so far distinguishes these,
# and the campaign has spent time on mixing arms that would be worthless under
# the first reading.
#
# PRE-REGISTERED, on the ratio of our thermocline sharpness to the oracle's in
# the cold tongue, comparing day 5 with day 90:
#   * already >= 2.5 at day 5      -> INHERITED. The initial condition carries
#     it; look at the vertical interpolation of the climatology, not at mixing.
#   * <= 1.3 at day 5 and >= 2.5 at day 90 -> GROWN. The initial condition is
#     innocent and the maintaining physics owns it.
#   * anything else -> report both numbers and the ratio between them, and
#     claim neither origin. In particular a day-5 value between 1.3 and 2.5 is
#     a MIXED origin and must be reported as such rather than rounded into
#     whichever branch is more convenient.
# All branches are reachable: the interpolation could preserve WOA's own
# gradient exactly, or sharpen it, and the 90-day physics could do either.
#
# WHY DAY 5 RATHER THAN DAY 0. We do not archive a day-0 snapshot, and the
# oracle's first record is a five-day mean centred inside the same window, so
# day 5 against record 0 is the earliest MATCHED pair available. Five days of
# a 90-day adjustment is close enough to the initial condition to separate the
# two origins, and it is honest about being five days rather than zero.
#
# ONE TRAJECTORY, TWO DIRECTORIES. Day 5 comes from trp_faithz0_d30 and day 90
# from trp_faithz0_d90. Those are the same recipe: a flattened diff of their
# resolved configurations differs in exactly one row, total_days, and their
# global mean sea surface temperature at day 30 agrees to six figures
# (13.9601 both). Splicing them is therefore reading one trajectory, not
# comparing two experiments.
#
# REUSED, NOT REBUILT: equatorial_thermocline_structure.py already computes
# the sharpness metric -- the mean over columns of each column's maximum
# |dT/dz| in the top 400 m -- beside the isotherm depth, on each model's own
# levels with its own area weights. Its unit test runs first as the gate.
set -uo pipefail
M=/burg-archive/glab/users/pg2328/_fluxacc_wt; cd $M || exit 1
echo "HEAD: $(git -C $M log --oneline -1)"
echo "dirty: $(git -C $M status --porcelain | wc -l)"
export PYTHONPATH="$M/src:$(ls -d $M/packages/*/ | sed 's:/$::' | tr '\n' ':')$M:$M/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo
GT=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc

echo "=== probe unit test ==="
$PY -m pytest -q tests/ocean/unit/test_equatorial_thermocline_structure.py 2>&1 | tail -4
rc=${PIPESTATUS[0]}
[ "$rc" -eq 0 ] || { echo "FATAL: probe test is red (rc=$rc) -- refusing to measure"; exit 1; }
[ -r "$GT" ] || { echo "FATAL: no oracle grid_T"; exit 1; }

LATE=$R/trp_faithz0_d90/snapshot_final.npz
EARLY=$R/trp_faithz0_d30/snapshot_day0005.npz
for f in "$LATE" "$EARLY"; do
  [ -r "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done

# Day stamps, same discipline as every other card here: a lane that records
# its day must agree with the mark, and one that records nothing says so
# rather than being quietly trusted.
$PY - "$LATE" "$EARLY" <<'STAMP' || { echo "FATAL: a snapshot is not at its expected day"; exit 1; }
import sys
import numpy as np
for path, want in ((sys.argv[1], 90.0), (sys.argv[2], 5.0)):
    d = np.load(path, allow_pickle=True)
    if "time_days" not in d.files:
        print(f"  {want:g}: NO STAMP (unverified)")
        continue
    got = float(np.asarray(d["time_days"]).ravel()[0])
    print(f"  stamp {got:g} (want {want:g})")
    if abs(got - want) > 1e-6:
        sys.exit(1)
STAMP

# REC = DAY/5 - 1, bounds-verified for this archive: day 90 is record 17 and
# day 5 is record 0.
$PY -u scripts/validate/ocean_fidelity/equatorial_thermocline_structure.py \
  --snapshot "$LATE" --label d90 --snapshot-early "$EARLY" \
  --nemo-gridt "$GT" --nemo-rec 17 --nemo-rec-early 0
rc=$?
echo "=== card exit: $rc ==="
exit $rc
