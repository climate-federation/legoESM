#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=modn8b48
#SBATCH --time=70:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=16G
#SBATCH --exclude=g161,g203
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_recon/modon8_c48_bounded_%j.log
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3grid
export PYTHONPATH=$REPO/packages/core:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
cd $REPO
# Zenodo case-8 outer cadence exactly: dt_atmos=1200, n_split=7
# (inner dt 1200/7; entry A-scalar exchange only on the first inner
# step of each block — dyn_core.F90:432-439)
/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python \
  scripts/validate/fv3_native/run_duo_stepper_modon.py --n 48 \
  --dt-atmos 1200 --n-split 7 --days 60 --frame-days 5 --preset case8 \
  --out /burg-archive/glab/users/pg2328/fv3_recon/modon8_c48_bounded.npz
