#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=omip_mesh
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/mesh_%j.out

# OMIP-faithful step D: download canonical eORCA1.2 mesh_mask (Zenodo 4436658,
# ~484MB) for legoESM `tripole` grid (= run_omip.py's expected grid file).
set +e
cd /burg-archive/glab/users/pg2328/legoESM
export PYTHONPATH=/burg-archive/glab/users/pg2328/legoESM/src:$PYTHONPATH
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
echo "=== HOST ==="; hostname; date
JAX_PLATFORMS=cpu $PY -c "from legoesm.grids.tripole import download_orca1_grid; p=download_orca1_grid('data/grids'); print('mesh at', p)" 2>&1 | tail -8
echo "rc=$?"; ls -lah data/grids/eORCA1.2_mesh_mask.nc 2>&1
echo "=== DONE ==="
