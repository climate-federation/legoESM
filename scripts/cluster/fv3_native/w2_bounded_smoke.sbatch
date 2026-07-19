#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=w2bnd12
#SBATCH --time=02:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G
#SBATCH --exclude=g161,g203
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_recon/w2_c12_bounded_c2l_%j.log
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3grid
export PYTHONPATH=$REPO/packages/core:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
P=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
cd $REPO
$P scripts/validate/fv3_native/run_duo_stepper_w2.py --n 12 --dt 900 \
  --days 1 --ext-bundle --oracle-conventions \
  --out /burg-archive/glab/users/pg2328/fv3_recon/w2_c12_bounded_c2l.npz
