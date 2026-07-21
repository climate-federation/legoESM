#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=test_leadlat
#SBATCH --cpus-per-task=2
#SBATCH --mem=14G
#SBATCH --time=00:25:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/test_leadlat_%j.out
set +e
WT=/burg-archive/glab/users/pg2328/legoESM_worktrees/vfs-tke-fix; cd $WT
export PYTHONPATH="$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT:$WT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 $PY -u -m pytest -q \
  tests/ice/unit/test_lead_freeze_latent.py \
  tests/ice/unit/test_ice_brine_salinity_faithful.py \
  tests/unit/test_sea_ice_new_physics.py 2>&1 | tail -12
echo "EXIT $?"
