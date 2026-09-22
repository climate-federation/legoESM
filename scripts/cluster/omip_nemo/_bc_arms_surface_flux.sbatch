#!/bin/bash
#SBATCH --job-name=bcflux
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/bcflux_%j.out
#
# Does the surface TKE boundary-condition flag change the SURFACE HEAT AND
# MOMENTUM FLUX, or only the interior mixing?
#
# This is GLM's corroboration, asked for twice and now owed. Its objection is
# that identical mixed-layer depth with diverging SST smells like an air-sea
# flux feedback or an implementation bug rather than faithful physics. If the
# two arms put DIFFERENT heat through the surface, the flag has a physical
# pathway to SST. If the surface fluxes are identical to within the probe's
# resolution, the difference is interior and the feedback branch is dead.
#
# NO NEW PROBE WAS WRITTEN. Searched scripts/validate/ocean_fidelity/ and
# found nino3_flux_vs_nemo.py, which already rebuilds our net heat and wind
# stress over the nino3 box through the SAME production bulk path the run used
# (air_sea_fluxes), at the forcing record whose centre matches the snapshot
# day, and scores both against NEMO's own qt_oce and taum on the matched
# 5-day file. That is exactly the measurement, against the oracle as a bonus.
#
# BOTH WINDOWS, NOT ONE. Day 30 is the window whose arm-minus-control gap
# tripled (0.204 against a 0.044-0.083 band over the first twenty-five days),
# and day 25 is the last ordinary window. GLM's reading is that the day-30
# jump, being synchronised across both arms, is a shared forcing event that
# the control absorbs about 40% worse than the arm -- an interaction, which
# would make day 30 the most informative day rather than the one to discard.
# Running only day 30 could not tell those apart; running both can.
#
# Four runs: {control nemo_z0, arm interior_pinned} x {day 25, day 30}.
set -uo pipefail
M=/burg-archive/glab/users/pg2328/_fluxacc_wt; cd $M || exit 1
export PYTHONPATH="$M/src:$(ls -d $M/packages/*/ | sed 's:/$::' | tr '\n' ':')$M:$M/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo
SBC=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_SBC.nc
# The probe's --mesh default is RELATIVE (data/grids/...), and data/ is
# runtime-only so it exists under _omip_merge and not under this worktree.
# Running from the editable tree therefore needs the mesh named absolutely.
MESH=/burg-archive/glab/users/pg2328/_omip_merge/data/grids/eORCA1.2_mesh_mask.nc
for f in "$SBC" "$MESH"; do
  [ -r "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done

rc=0
for pair in "d25:25:4:5:snapshot_day0025.npz" "d30:30:5:6:snapshot_final.npz"; do
  TAG=${pair%%:*}; rest=${pair#*:}
  DAY=${rest%%:*}; rest=${rest#*:}
  R0=${rest%%:*}; rest=${rest#*:}
  R1=${rest%%:*}; SNAPNAME=${rest#*:}
  for arm in "ctl_nemo_z0:trp_faithz0_d30" "arm_interior_pinned:trp_pinned_d30"; do
    LAB=${arm%%:*}; DIR=${arm#*:}
    SNAP=$R/$DIR/$SNAPNAME
    [ -r "$SNAP" ] || { echo "FATAL: missing $SNAP"; rc=1; continue; }
    echo "########## $LAB  day $DAY  (NEMO recs $R0:$R1) ##########"
    $PY -u scripts/validate/ocean_fidelity/nino3_flux_vs_nemo.py \
      --snapshot "$SNAP" --nemo-sbc "$SBC" --nemo-recs "$R0:$R1" --day "$DAY" \
      --mesh "$MESH" 2>&1 | tail -40
    r=${PIPESTATUS[0]}; echo "### $LAB $TAG rc=$r"; [ "$r" -eq 0 ] || rc=$r
  done
done
echo "=== card exit: $rc ==="
exit $rc
