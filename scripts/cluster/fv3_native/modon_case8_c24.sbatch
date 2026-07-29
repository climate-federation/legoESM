#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=modn8b24
#SBATCH --time=60:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=12G
#SBATCH --exclude=g161,g203
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_recon/modon8_c24_bounded_%j.log
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3grid
export PYTHONPATH=$REPO/packages/core:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
cd $REPO
/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python \
  scripts/validate/fv3_native/run_duo_stepper_modon.py --n 24 --dt 360 \
  --days 60 --preset case8 --frame-days 5 \
  --out /burg-archive/glab/users/pg2328/fv3_recon/modon8_c24_bounded.npz
