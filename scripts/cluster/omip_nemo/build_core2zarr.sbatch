#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=omip_c2z
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/c2z_%j.out

# OMIP-faithful: build legoESM CORE-II NYF zarr from NEMO COREv2 INPUTS so both
# models share identical forcing.
set +e
cd /burg-archive/glab/users/pg2328/legoESM
export PYTHONPATH=/burg-archive/glab/users/pg2328/legoESM/src:$PYTHONPATH
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
INPUTS=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS
echo "=== HOST ==="; hostname; date
JAX_PLATFORMS=cpu $PY scripts/data/build_core2_nyf_zarr.py --inputs-dir "$INPUTS" 2>&1 | tail -30
echo "c2z_rc=$?"
echo "=== DONE ==="
