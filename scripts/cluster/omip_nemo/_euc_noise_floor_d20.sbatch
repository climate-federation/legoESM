#!/bin/bash
#SBATCH --job-name=eucnull
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/eucnull_%j.out
#
# THE CONTROL THE UNDERCURRENT THREAD HAS NEVER HAD: A SAME-CONFIGURATION NULL.
#
# Every number in that thread is a difference between our run and NEMO's, and
# not one of them has ever been compared against the difference between two of
# OUR OWN runs that should agree. Without that, "we carry 4% of NEMO's
# transport" has no error bar and a reviewer is entitled to ignore it.
#
# The null arrived by accident. The 90-day run resumes from the 30-day run's
# day-15 restart, so both integrate days 15 to 20 from the same state, and
# their resolved configurations agree on 626 of 626 leaves apart from
# output_path and total_days. total_days reaches only the step count and the
# manifest -- grep says it never touches physics. The two runs should have
# produced the same day-20 ocean.
#
# THEY DID NOT. Field-by-field at day 20: temperature differs by up to 2.61 K,
# zonal velocity by 0.333 m/s (22% of its own maximum), and roughly 42% of wet
# cells differ in T, S and both mass fluxes. The mixing diagnostics differ too,
# which is the likely amplifier: convection and the enhanced-diffusivity
# trigger are switches, so a rounding-level difference flips a cell into or out
# of a large diffusivity and the difference stops being rounding-level
# immediately.
#
# WHY THEY DIVERGED AT ALL, PLAUSIBLE AND NOT ESTABLISHED: the two runs used
# DIFFERENT GPUs -- the 30-day leg ran on g048, this one runs on g184.
# Different hardware means different kernel selection and different reduction
# order, hence different rounding from the first step. Proving it needs a
# same-node repeat, which is not what this card is for.
#
# WHAT THIS CARD MEASURES, AND IT IS THE POINT: run the SAME undercurrent
# probe, at the SAME day, on the two snapshots that should have agreed. The
# 30-day run's day-20 numbers are already on record -- 220E core +0.405 m/s,
# 220E net transport +28.27 m2/s, 240E net transport -29.22 m2/s, against
# NEMO's +0.667, +44.01 and -5.93. Whatever this run reports instead IS the
# noise floor of those metrics under an identical configuration.
#
# PRE-REGISTERED, because the number decides how much of the thread survives:
#   * core speed and transport reproduce to a few percent of the ours-minus-
#     NEMO gap: the gap is real and the thread stands.
#   * they move by a sizeable fraction of that gap: every single-pair
#     undercurrent number in this session needs an ensemble before it can be
#     quoted, and "the disagreement grows" cannot be read off one run.
#   * they move by as much as the gap: the undercurrent comparison has been
#     measuring noise and the thread collapses.
# NEMO is included in the same call so the oracle half is printed from the
# identical code path at the identical day; its record for day 20 is 3.
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
for pair in "d30run:$R/trp_faithz0_d30/snapshot_day0020.npz" \
            "d90run:$R/trp_faithz0_d90/snapshot_day0020.npz"; do
  TAG=${pair%%:*}; SNAP=${pair#*:}
  [ -r "$SNAP" ] || { echo "FATAL: missing $SNAP"; rc=1; continue; }
  echo "############ $TAG  day 20  (NEMO rec 3:4) ############"
  $PY -u scripts/validate/ocean_fidelity/equatorial_thermocline.py \
    --legoesm-snapshot "$SNAP" --nemo-gridt ${STEM}_grid_T.nc \
    --nemo-t-recs "3:4" --label "null_$TAG" \
    --nemo-wfile ${STEM}_grid_W.nc --nemo-w-var wo --nemo-w-recs "3:4" \
    --nemo-ufile ${STEM}_grid_U.nc --nemo-vfile ${STEM}_grid_V.nc \
    --mesh-mask "$MESH" 2>&1 | tail -400
  r=${PIPESTATUS[0]}; echo "### $TAG rc=$r"; [ "$r" -eq 0 ] || rc=$r
done
echo "=== card exit: $rc ==="
exit $rc
