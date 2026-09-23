#!/bin/bash
#SBATCH --job-name=sssrunoff
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/sssrunoff_%j.out
#
# DOES THE COASTAL SALINITY GAP LIVE WHERE THE RIVERS ARE?
#
# The FESOM lane's coastal surface-salinity error is about twice the tripole's,
# it sits almost entirely in the ring of cells touching the edge of the scored
# domain, and the day-5 control showed it is already at full relative strength
# five days in. So nothing slow builds it. Two candidates survive, both
# switching on at t=0: the initial condition interpolated onto FESOM's mesh,
# and where the two lanes PUT river runoff.
#
# THE LANE DIFFERENCE IS REAL AND IT IS DOCUMENTED, read off the manifests
# rather than the cards: trp_pinned_d30 and mpas_orca1faithtint_d30 both ran
# --runoff-depth-nemo-ini; fesom_orca1faithtint_d30's argv carries only
# --runoff and --river-mouth-restoring-gate. That flag is NEMO's
# ln_rnf_depth_ini, which spreads each river's freshwater down to a depth set
# by the local runoff climatology, up to 150 m. Without it the driver's own
# help calls the behaviour "legacy top-cell": every drop lands in the surface
# cell. The FESOM card lists --runoff-depth-nemo-ini among the flags its
# stage-B4 allowlist REJECTS, and ocean_model_fesom.py carries no
# runoff_depth_spread_map, so this is a lane capability gap, not a card
# oversight. The driver's own comment at the wiring site claims the lever
# reaches "all three grids"; the code below it has two branches, MPAS and the
# structured C-grids, and FESOM is neither.
#
# So the mechanism is CONFIRMED as a difference between the lanes. Whether it
# is what drives the measured excess is NOT, and that is what this measures.
#
# THE TEST IS GEOGRAPHIC, WHICH IS WHY IT CAN SUCCEED WHERE THE TIME SERIES
# CANNOT. Both candidates act at t=0, so no arrangement of marks separates
# them. They differ in WHERE: a runoff-placement difference can only act where
# there is runoff, while an interpolated initial condition is wrong on every
# coast, including the runoff-free ones.
#
# THE FIRST VERSION OF THIS PRE-REGISTRATION WAS VACUOUS, and it is left
# described here rather than quietly replaced. It banded the zero-runoff
# stratum's SHARE of the gap, on the stated premise that "runoff-free cells
# are most of the ring by area". Measured, they were 289 of 4127 cells and
# 1.5% OF THE AREA, because the first version averaged runoff over a
# 2.5-degree radius and left almost no coastal cell at exactly zero. A stratum
# that small cannot carry 55% of the gap under any hypothesis, so the refuting
# branch was unreachable and the passing branch meant nothing. Withdrawn
# before it was claimed. See 90111a318.
#
# TWO FIXES, and both are in the probe rather than in the wording:
#   * runoff is carried to the target by its NEAREST SOURCE CELL, not an
#     inverse-distance average, so "no river here" is a fact about the
#     coastline instead of about the search radius.
#   * the statistic is LIFT = the stratum's share of the gap divided by its
#     share of the area. That is 1.0 for a gap that simply follows the
#     coastline, whatever the stratum's size, and near 0 for a gap that needs
#     rivers. It cannot be defeated by a stratum turning out small.
#
# PRE-REGISTERED, on the LIFT of the exactly-zero-runoff stratum in ring 1:
#   * lift <= 0.25  -> runoff placement LEADS. River-free coast carries far
#     less of the gap than its area, so the excess needs rivers, and closing
#     the depth-spread gap on the FESOM lane is the first thing to try.
#   * lift >= 0.75  -> runoff placement is REFUTED as the leading term. The
#     excess is on river-free coast in proportion to its area, and the initial
#     condition carries it.
#   * 0.25 to 0.75  -> NO VERDICT, both contribute and neither leads.
# A stratifier blind to runoff returns lift 1.0 everywhere by construction,
# which is what the probe's own uniform-excess test asserts.
#
# READ THE WHOLE LIFT COLUMN, NOT ONLY THAT ROW. Lift rising monotonically
# with runoff is the dose-response the hypothesis predicts. It is also, on its
# own, CONFOUNDED: river-mouth cells carry the sharpest surface-salinity
# gradients in the ocean and are therefore the hardest cells for any
# interpolation, so "the IC interpolation is worst where gradients are
# sharpest" predicts the same shape. The zero-runoff row is the one place the
# two candidates genuinely disagree, which is why it alone is banded.
#
# The strata come from the ring's OWN nonzero runoff percentiles, so no
# threshold is chosen by hand. Shares are computed on area-weighted SQUARED
# error, the only form that adds across strata, and a recombination mismatch
# is fatal.
#
# RUN AT BOTH MARKS. Day 5 is the one the pre-registration is about -- it is
# closest to t=0, where both candidates are cleanest. Day 30 is printed beside
# it because a mechanism that is real should not change stratum between them.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_fluxacc_wt; cd $ROOT || exit 1
echo "HEAD: $(git -C $ROOT log --oneline -1)"
echo "dirty: $(git -C $ROOT status --porcelain | wc -l)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

WORK=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/_nodedist_work
RUNOFF=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/input_fields/runoff-icb_DaiTrenberth_Depoorter.nc

# The probe's own gate first. A diagnostic that decides a verdict is code, and
# unreviewed code does not get to print a number.
echo "=== probe unit tests ==="
JAX_ENABLE_X64=1 $PY -m pytest -q tests/validate/test_coastal_sss_by_runoff.py 2>&1 | tail -15
rc=${PIPESTATUS[0]}
[ "$rc" -eq 0 ] || { echo "FATAL: probe tests are red (rc=$rc) -- refusing to measure"; exit 1; }

[ -r "$RUNOFF" ] || { echo "FATAL: no runoff file at $RUNOFF"; exit 1; }

rc=0
for TAG in d0005 d30; do
  DUMP=$WORK/cells_${TAG}.npz
  if [ ! -r "$DUMP" ]; then
    echo "FATAL: missing dump $DUMP"; rc=1; continue
  fi
  echo
  echo "===== ${TAG} ====="
  JAX_ENABLE_X64=1 $PY -u scripts/validate/ocean_fidelity/coastal_sss_by_runoff.py \
    --dump "$DUMP" --runoff-file "$RUNOFF" --ring 1 || rc=1
done
echo "=== card exit: $rc ==="
exit $rc
