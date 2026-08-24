#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=prpeek
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:25:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/prpeek_%j.out
#
# EARLY read of our equatorial Prandtl vs NEMO, off the mxl4 day-10 snapshot
# the running arm already wrote (with --kprofile-snapshots).  Reads existing
# output, runs no model.  Day 10 is NOT the pre-registered day 30, and this is
# ONE arm, so it is a PLAUSIBLE early signal on whether the Prandtl hypothesis
# is alive at all -- avm~NEMO with avt<<NEMO (ratio defect) vs both<<NEMO
# (starvation) -- not a verdict.  NEMO grid_W rec 1 = days 6-10 (matched).
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu
GW_W=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_W.nc
SNAP=$ROOT/results/omip_nemo/nemolev_trp_mxl4_d30/snapshot_day0010.npz
[ -f "$SNAP" ] || { echo "no day-10 snapshot yet"; exit 1; }
$PY -u scripts/validate/ocean_fidelity/equatorial_diffusivity_oracle.py \
  --nemo-gridw "$GW_W" --rec 1 --legoesm-snapshot "$SNAP" 2>&1 | tail -120
echo DONE
