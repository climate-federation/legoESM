#!/bin/bash
#SBATCH --job-name=arcday0
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/arcday0_%j.out
#
# THE DAY-0 CONTROL, without which the onset claim is unsupported.
# Top-19 m Arctic salinity spread so far:  NEMO 0.827 | ours d5 0.263 | d30 0.018
# If the INITIAL CONDITION already has a weak spread, mixing is not destroying
# the halocline -- the IC never had one, and this whole mixing line is wrong.
# The IC is `presalt` (12, 75, 331, 360): already on the model's 75 levels and
# already native, so no vertical interpolation can be blamed either way.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM
cd $REPO || exit 1
export PYTHONPATH="$REPO/src:$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
IC=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/initial_conditions/woce_salt_monthly_init_4p2.nc
$PY -u scripts/validate/ocean_fidelity/arctic_salt_vs_volume.py \
  --native-salt "presalt:$IC" --nemo-month 1 \
  --mesh-mask $REPO/data/grids/eORCA1.2_mesh_mask.nc --label IC_day0_jan
echo "=== rc=$? ==="
