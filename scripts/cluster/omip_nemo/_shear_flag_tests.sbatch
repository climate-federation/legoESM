#!/bin/bash
#SBATCH --job-name=sheartest
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/sheartest_%j.out
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/legoESM; cd $WT || exit 1
export PYTHONPATH="$WT/src:$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python -m pytest -q \
  tests/unit/test_run_omip_core2_kappa_convention_cli.py \
  tests/ocean/unit/test_tripole_vmix_cli.py \
  tests/unit/test_run_omip_core2_kpp_override.py \
  tests/ocean/unit/test_mpas_tke.py
echo "=== rc=$? ==="
