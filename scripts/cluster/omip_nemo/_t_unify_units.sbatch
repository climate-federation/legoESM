#!/bin/bash
#SBATCH --job-name=t_unify
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/t_unify_%j.out
set -u
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 $PY -m pytest -q \
  tests/ocean/unit/test_isf_prescribed_melt.py \
  tests/ocean/unit/test_iwm_forcing_loader.py \
  tests/ocean/unit/test_mpas_iwm.py \
  tests/ocean/unit/test_mpas_tke.py \
  tests/ocean/unit/test_bbl_adv_mpas.py tests/ocean/unit/test_bbl_adv.py \
  "$@" 2>&1 | tail -12
echo DONE
