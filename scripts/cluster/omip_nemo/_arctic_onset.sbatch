#!/bin/bash
#SBATCH --job-name=arconset
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/arconset_%j.out
#
# ONSET vs FEEDBACK.  The day-30 state shows the Arctic halocline gone and the
# near-surface N^2 100-360x weaker than NEMO's -- but the run STARTED from a
# stratified NEMO-monthly IC, and once the halocline is gone, weak N^2 keeps it
# gone (l_b = sqrt(2e)/N grows as N falls).  So the day-30 state cannot
# distinguish what DESTROYED the halocline from what SUSTAINS its absence.
#
# THIS ADDS TIME.  The day-5 snapshot (hoist identity run, control physics --
# NO experiment flags) is profiled with the SAME probe as day 30.
#
# PRE-REGISTERED READING:
#   * halocline ALREADY largely gone at day 5 (top-20 m spread << NEMO's 0.83)
#     => onset is FAST, within days: look at the initial adjustment / the first
#     steps, not at a slow seasonal process.
#   * halocline still largely INTACT at day 5 => onset is gradual over weeks,
#     and the day-5 vs day-30 difference bounds the erosion rate.
# Either way the number is the top-20 m salinity spread, directly comparable to
# NEMO's 0.83 psu and to our day-30 value of 0.018.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM
cd $REPO || exit 1
echo "=== HEAD $(git -C $REPO rev-parse HEAD)"
export PYTHONPATH="$REPO/src:$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
MESH=$REPO/data/grids/eORCA1.2_mesh_mask.nc
R=$REPO/results/omip_nemo

# Day 5, control physics (the identity run carries NO experiment flags).
$PY -u scripts/validate/ocean_fidelity/arctic_salt_vs_volume.py \
  --snapshot $R/nemolev_trp_hoist_d5/snapshot_final.npz --mesh-mask "$MESH" \
  --label hoist_d5 --profile \
  --out-json $R/arctic_salt_volume/hoist_d5.json
echo "=== rc=$? ==="
