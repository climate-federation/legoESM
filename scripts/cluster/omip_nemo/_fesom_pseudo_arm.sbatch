#!/bin/bash
#SBATCH --job-name=pseudoarm
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/pseudoarm_%j.out
#
# CAN GEOMETRY ALONE MANUFACTURE THE FESOM SALINITY GAP?
#
# Where this stands. FESOM's day-30 surface salinity scores rmse 0.465 against
# the ORCA1 oracle where the tripole scores 0.258, the two arms' mean offsets
# agree to 7e-4 so it is scatter, and 86% of the gap sits in the SINGLE ring of
# target cells touching something not scored (job 9934775). The smoothing
# ladder (job 9934795) refuted the obvious explanation: smoothing every source
# to a common 330 km left the FESOM/tripole salinity ratio at 1.80-1.85 across
# the whole ladder while SST's ratio fell 1.18 -> 1.07, so the instrument can
# see and remove a small-scale artifact and there is none in the salinity gap.
#
# Two explanations survive and no measurement yet separates them. The remap may
# be drawing values ACROSS a coastline or barrier -- the scorer's own docstring
# already flags that it picks neighbours by distance and would need a
# topology-aware remapper -- and smoothing does not remove that, because a
# cross-barrier draw is an OFFSET rather than noise. Or the two models may
# genuinely differ at coasts, for which the FESOM lane offers no shortage of
# candidates given its 12x timestep and its refusal of restratification, eddy
# parameterisation and the NEMO bottom drag.
#
# THE CONTROL. Carry the ORACLE'S OWN FIELD through FESOM's mesh and mask and
# then through the scorer's own path, and score it against the oracle. Model
# error is zero by construction, so any rmse it produces was manufactured by
# the scoring path. This was GLM's first-ranked test.
#
# WHAT IT TURNED OUT TO MEASURE, written after the fact and REPLACING an
# earlier claim in this header that it was an UPPER BOUND on the geometry term
# and that it discriminated the two hypotheses. It is neither.
#   * NOT A BOUND. The oracle-to-node leg inflates the control (it can itself
#     draw across a barrier) and deflates it (the pseudo field is band-limited
#     to the oracle's resolution and shares the reference's kernel). The
#     deflation half was then MEASURED and found absent -- pseudo error
#     correlates with the tripole's at +0.04 in ring 1 -- but the label stays
#     off. It is an estimate.
#   * NOT A DISCRIMINATOR. This FESOM snapshot carries NO LAND, every node is
#     wet, so the control wears the ORACLE'S coastline while the real arm
#     carries its own. A disagreement about where the ocean ends is invisible
#     to it by construction.
#   * WHAT IT DID ANSWER: removing the scoring path's own error takes FESOM's
#     ring-1 salinity rmse from 1.3917 to 1.2882 and its global figure from
#     0.4650 to 0.4372. Six to seven per cent. Not the explanation. Note that
#     the magnitude ratio pseudo/excess reads 0.20 and OVERSTATES this, because
#     it ignores that the two error fields only partly point the same way.
#
# THE STENCIL-REACH TEST, added after all of the above and the reason this card
# now also takes the tripole. FESOM's node cloud has no land, so nothing stops
# the four-neighbour stencil at a coastal target cell from averaging in nodes
# across a headland or a bay -- the failure the scorer's own docstring names
# when it says a verdict there needs a topology-aware remapper. Narrowing each
# arm to its SINGLE nearest source removes three of the four chances to do it.
# ONE-SIDED: k=1 also drops the averaging, which alone would RAISE the error,
# so a FALL in FESOM's ring 1 is evidence of cross-barrier contamination while
# no change or a rise is ambiguous and must be reported as ambiguous. The
# tripole is the control that makes it readable -- it shares the oracle's grid
# and land mask, so both arms moving alike means the change is about averaging
# rather than barriers.
#
# GATES, run before the control produces anything. The probe reproduces the
# oracle and FESOM fields the dump already holds (if it cannot rebuild the
# scorecard it stands on, nothing downstream is readable), and checks its
# point-target interpolator against the grid regridder to 1e-12 on that
# function's own targets. The unit tests run FIRST and include a non-vacuity
# partner showing the stencil is live, so the equivalence is arithmetic rather
# than both routines collapsing onto the nearest neighbour.
set -uo pipefail
M=/burg-archive/glab/users/pg2328/_fluxacc_wt; cd $M || exit 1
export PYTHONPATH="$M/src:$(ls -d $M/packages/*/ | sed 's:/$::' | tr '\n' ':')$M:$M/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo
G=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY
STEM=$G/ORCA1_5d_20000101_20000331
DUMP=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/_nodedist_work/cells_d30.npz
FES=$R/fesom_orca1faithtint_d30/snapshot_day0030.npz
TRP=$R/trp_pinned_d30/snapshot_final.npz

for f in "$DUMP" "$FES" "$TRP" ${STEM}_grid_T.nc; do
  [ -r "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done

echo "=== unit tests for the new interpolator ==="
$PY -m pytest -q tests/validate/test_fesom_pseudo_arm.py 2>&1 | tail -12
rc=${PIPESTATUS[0]}
[ "$rc" -eq 0 ] || { echo "FATAL: unit tests failed rc=$rc; not running the probe"; exit $rc; }

echo
echo "=== the control ==="
# --nemo-time-idx 5 is the record the day-30 scorecard used (records are 5-day
# means, so the day-N record is N/5 - 1); the dump was written from it.
$PY -u scripts/validate/ocean_fidelity/fesom_pseudo_arm.py \
  --dump "$DUMP" --fesom "$FES" --tripole "$TRP" \
  --nemo-gridt ${STEM}_grid_T.nc --nemo-time-idx 5 --res-deg 1.0
rc2=$?
echo "=== card exit: $rc2 ==="
exit $rc2
