#!/bin/bash
#SBATCH --job-name=l8_e3w
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/l8_e3w_%j.out
# Level-8 arm died at step 1: "raw-mesh e3w_int must contain only finite values > 0".
# Localise the cells (level 7 as control) before touching code.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/_omip_wt; cd $REPO || exit 1
export PYTHONPATH="$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
for L in 7; do
  echo "########## level $L"
  $PY -u scripts/validate/ocean_fidelity/probe_mpas_e3w_geometry.py --level $L --nemo-monthly-init /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/initial_conditions/woce_temp_monthly_init_4p2.nc /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/initial_conditions/woce_salt_monthly_init_4p2.nc 2>&1 | grep -vE "^\[|^\s*$" | tail -n 80
  echo "rc=$?"
done
