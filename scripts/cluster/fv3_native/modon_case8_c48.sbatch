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
# dt = 1200/7 = the Zenodo case-8 small step exactly
/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python \
  scripts/validate/fv3_native/run_duo_stepper_modon.py --n 48 \
  --dt 171.42857142857142 --days 60 --frame-days 5 --preset case8 \
  --out /burg-archive/glab/users/pg2328/fv3_recon/modon8_c48_bounded.npz
