#!/bin/bash
#SBATCH --job-name=arcsalt
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/arcsalt_%j.out
#
# DECIDES whether the 13 GPU-h water-flux arm is worth launching.
#
# S_mean = M_salt / V, so the common-mode Arctic +1.4 psu bias is either a
# SALT-inventory excess or a VOLUME deficit.  Real-water restoring conserves
# h*S -- it dilutes but cannot remove salt MASS -- and is suppressed under ice.
# So a salt-dominated bias REFUTES the water-routing hypothesis outright.
#
# PRE-REGISTERED (repeated from the probe docstring so the verdict cannot
# drift): salt-dominated -> do NOT launch the arm; volume-dominated -> launch;
# neither dominant -> report both, claim no direction.
#
# The probe validates itself first: its Arctic volume integral must match the
# mesh-implied volume to round-off, an empty region is FATAL, and any NaN in a
# wet cell is FATAL.  Numbers are not quoted unless that check prints.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM
cd $REPO || exit 1
echo "=== HEAD $(git -C $REPO rev-parse HEAD)"
export PYTHONPATH="$REPO/src:$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
R=$REPO/results/omip_nemo
MESH=$REPO/data/grids/eORCA1.2_mesh_mask.nc
[ -r "$MESH" ] || { echo "FATAL: mesh unreadable"; exit 1; }

NFAIL=0
run_stage () {
  local lab="$1"; shift
  echo; echo "########## $lab"
  "$@"
  local rc=$?
  echo "=== $lab rc=$rc"
  [ $rc -eq 0 ] || NFAIL=$((NFAIL + 1))
}

# Ours: the tripole day-30 snapshot (the arm's control lineage).
run_stage "tripole d30 Arctic salt vs volume" \
  $PY -u scripts/validate/ocean_fidelity/arctic_salt_vs_volume.py \
    --snapshot $R/xgrid_trp_d30/snapshot_final.npz --mesh-mask "$MESH" \
    --label tripole_d30 \
    --out-json $R/arctic_salt_volume/tripole_d30.json

# The control arm too, so the number is tied to the run the arm would compare
# against rather than to a different lineage.
if [ -r "$R/nemolev_trp_gwcorr_d30/snapshot_final.npz" ]; then
  run_stage "control (gwcorr) d30 Arctic salt vs volume" \
    $PY -u scripts/validate/ocean_fidelity/arctic_salt_vs_volume.py \
      --snapshot $R/nemolev_trp_gwcorr_d30/snapshot_final.npz --mesh-mask "$MESH" \
      --label control_gwcorr_d30 --profile \
      --out-json $R/arctic_salt_volume/control_gwcorr_d30.json
fi

# THE REFERENCE ARM.  Without NEMO there is no dM_salt or dV and therefore no
# verdict -- the first version of this gate compared ours against ours, which
# answers a different question entirely.
GT=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_REF/ORCA1_1m_20000101_20041231_grid_T.nc
run_stage "NEMO month 1 Arctic salt vs volume" \
  $PY -u scripts/validate/ocean_fidelity/arctic_salt_vs_volume.py \
    --nemo-gridt "$GT" --nemo-month 1 --mesh-mask "$MESH" \
    --label nemo_m01 --profile \
    --out-json $R/arctic_salt_volume/nemo_m01.json

echo; echo "=== ALL DONE (failed stages: $NFAIL) ==="
exit $((NFAIL > 0))
