#!/bin/bash
#SBATCH --job-name=mlepsi
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/mlepsi_%j.out
#
# Discriminate the MLE over-restratification mechanism (day 30, matched pair):
#   H^2 amplification at the deep-MLD-excess columns  vs
#   freshwater-forcing interaction at river mouths     vs
#   centered-flux instability (bolus_cfl_cap=0).
# dt=150 s from the run manifest.  Offline, CPU, existing snapshots.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM
cd $REPO || exit 1
echo "=== HEAD $(git -C $REPO rev-parse HEAD)"
export PYTHONPATH="$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
R=$REPO/results/omip_nemo

$PY -u scripts/validate/ocean_fidelity/mle_psi_diagnostics.py \
  --mle-snapshot $R/nemolev_trp_mle_d90/snapshot_day0030.npz \
  --control-snapshot $R/nemolev_trp_fwreal_d90/snapshot_day0030.npz \
  --mle-ce 0.06 --dt-s 150 \
  --mesh-mask $REPO/data/grids/eORCA1.2_mesh_mask.nc \
  --out-dir $R/mle_psi_diag_d30
rc=$?
echo "=== rc=$rc ==="
exit $rc
