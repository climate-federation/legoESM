#!/bin/bash
#SBATCH --job-name=eqthermo
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/eqthermo_%j.out
#
# THE DOMINANT ERROR, MEASURED WHERE IT LIVES.
#
# Ranking the day-30 scorecard against everything this campaign was asked to
# get right, the equatorial Pacific is the only large term: Nino3 surface
# temperature is +1.135 K biased with rmse 1.500, and the equatorial Pacific
# +1.212 / 1.643, while the Southern Ocean sits at -0.107 / 0.338, off North
# America at -0.012 / 0.457 and the Arctic at +0.010 / 0.330. Five to ten
# times everything else.
#
# WHY THE THERMOCLINE AND NOT SOMETHING ELSE. The undercurrent force balance
# measured earlier in this session found our net core force at 220E to be 0.29
# of NEMO's, paired t -2.58 on 4.5 effective samples, and the mean-gap
# decomposition -- an identity, no statistics -- put 63% of that gap in the
# DENSITY STRUCTURE rather than in sea level. A zonal density-gradient deficit
# along the equator IS a thermocline-slope deficit, so this measures the same
# claim in the variable it is usually stated in, where it is visible and where
# the oracle can be compared directly.
#
# REUSED, NOT REBUILT: scripts/validate/ocean_fidelity/
# equatorial_thermocline_structure.py already exists and already answers this.
# Its own docstring names the discriminator -- mixed-layer depth, the depth of
# the 20 C isotherm, and the SHARPNESS of the thermocline, all at once -- built
# to separate a model that is right from one that is right by cancellation
# between a deep smeared thermocline and too much vertical mixing. There is a
# second probe, equatorial_thermocline.py, which does one lane at a time; this
# one takes several, so it also answers the three-grid question in one pass.
# Its unit test runs first as the gate.
#
# PRE-REGISTERED, on the EAST-WEST TILT of the 20 C isotherm along 2S-2N,
# tripole against NEMO at day 30:
#   * our tilt is significantly FLATTER than NEMO's  ->  the density-structure
#     deficit found in the force balance is confirmed in the variable that
#     causes it, and the thermocline becomes the lever.
#   * our tilt MATCHES NEMO's  ->  the zonal density structure is right and the
#     63% density share of the force gap is NOT a thermocline-slope problem;
#     it would then have to sit in the vertical distribution rather than the
#     zonal one, and the sharpness and mixed-layer rows say which.
#   * our tilt is STEEPER  ->  the force-balance decomposition and this
#     measurement disagree, and neither is usable until that is resolved.
# All three are reachable: tilt is a difference of two depths each tens of
# metres, our warm bias is consistent with any of them a priori, and nothing
# measured so far constrains the sign.
#
# READ THE SHARPNESS AND MIXED-LAYER ROWS BEFORE THE TILT. A thermocline that
# is deep and smeared upwells warmer water regardless of its slope, and that
# is the other family of cause for the same surface bias. The probe prints
# both so the two are not confused.
#
# WHAT THIS CANNOT SETTLE: the warm bias is partly transient -- it plateaus
# from +1.42 at day 30 toward +0.70 by day 180 -- so a day-30 thermocline
# difference is an upper bound on the equilibrium one. Day 15 is included as
# the early mark so the direction of travel is visible.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_fluxacc_wt; cd $ROOT || exit 1
echo "HEAD: $(git -C $ROOT log --oneline -1)"
echo "dirty: $(git -C $ROOT status --porcelain | wc -l)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo
GT=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc

echo "=== probe unit test ==="
$PY -m pytest -q tests/ocean/unit/test_equatorial_thermocline_structure.py 2>&1 | tail -4
rc=${PIPESTATUS[0]}
[ "$rc" -eq 0 ] || { echo "FATAL: probe test is red (rc=$rc) -- refusing to measure"; exit 1; }
[ -r "$GT" ] || { echo "FATAL: no oracle grid_T at $GT"; exit 1; }

# The oracle record convention for this campaign is REC = DAY/5 - 1, so day 30
# is record 5 and day 15 is record 2.
ARGS=""
# evdarm is the enhanced-diffusion arm (job 9959646). It is listed IMMEDIATELY
# after the tripole because those two are the controlled pair -- identical
# command lines but for the convection flags, verified by diffing the line the
# run itself echoed -- so the verdict is read from ADJACENT rows of one table,
# scored by the same probe on the same box against the same oracle record.
for a in tripole=trp_pinned_d30 evdarm=trp_evdeq_d30 mpas=mpas_orca1faithtint_d30 fesom=fesom_orca1faithtint_d30; do
  lab=${a%%=*}; dir=${a#*=}
  late=$R/$dir/snapshot_day0030.npz
  [ -r "$late" ] || late=$R/$dir/snapshot_final.npz
  early=$R/$dir/snapshot_day0015.npz
  if [ ! -r "$late" ] || [ ! -r "$early" ]; then
    echo "FATAL: missing a snapshot for $lab ($dir)"; exit 1
  fi
  echo "  $lab: late=$(basename $late) early=$(basename $early)"
  # A run scored before it reaches day 30 would be compared against the
  # oracle's day-30 record and silently read as a result. Lanes that stamp
  # their day must therefore AGREE with the mark; lanes that stamp nothing
  # (the node-based one does not) are reported as unverified rather than
  # blocked, so this cannot turn into a lane-specific gate nobody notices.
  $PY - "$late" "$early" "$lab" <<'STAMP' || { echo "FATAL: $lab snapshot is not at the expected day"; exit 1; }
import sys
import numpy as np
for path, want in ((sys.argv[1], 30.0), (sys.argv[2], 15.0)):
    d = np.load(path, allow_pickle=True)
    if "time_days" not in d.files:
        print(f"    {want:g}: NO STAMP (unverified)")
        continue
    got = float(np.asarray(d["time_days"]).ravel()[0])
    print(f"    stamp {got:g} (want {want:g})")
    if abs(got - want) > 1e-6:
        sys.exit(1)
STAMP
  ARGS="$ARGS --snapshot $late --label $lab --snapshot-early $early"
done

$PY -u scripts/validate/ocean_fidelity/equatorial_thermocline_structure.py $ARGS \
  --nemo-gridt $GT --nemo-rec 5 --nemo-rec-early 2
rc=$?

# THE DISCRIMINATOR ADDED AFTER THE FIRST RUN, and it is a test of the
# INSTRUMENT rather than of the model. Our snapshots are INSTANTANEOUS, every
# lane runs --dm2dc so the shortwave has a diurnal cycle, the top level centre
# is 0.512 m, and every mark sits at exactly day N.0 -- the same clock hour
# each time. The oracle's records are 5-DAY MEANS, which average the diurnal
# cycle away. A fixed-phase instantaneous sample of a half-metre level against
# a 5-day mean is exactly the aliasing this campaign has been burned by.
#
# The near-surface profile separates the two readings outright:
#   * the warm anomaly confined to the top one to three levels, decaying with
#     an e-folding of a few metres  ->  DIURNAL SKIN, the finding is a
#     sampling artefact and the model may be fine.
#   * the anomaly still large at 10-20 m  ->  a real mixing defect, and the
#     decoupled-surface reading stands.
for box in "220 240" "240 260"; do
  set -- $box
  echo
  echo "===== near-surface profile ${1}-${2}E ====="
  $PY -u scripts/validate/ocean_fidelity/equatorial_thermocline_structure.py $ARGS \
    --nemo-gridt $GT --nemo-rec 5 --nemo-rec-early 2 --surface-profile \
    --surface-lon-lo $1 --surface-lon-hi $2 2>&1 | sed -n '/=== Near-surface/,$p' || rc=1
done
echo "=== card exit: $rc ==="
exit $rc
