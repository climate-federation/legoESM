#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=duow2
#SBATCH --time=11:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --exclude=g161
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_recon/duow2_%j.log
set -euo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3grid
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH=$REPO/packages/core:$REPO/src JAX_PLATFORMS=cpu
N=${1:-24}
DT=${2:-450}
OUT=/burg-archive/glab/users/pg2328/fv3_recon/w2_duo_stepper_c${N}.npz
$PY $REPO/scripts/validate/fv3_native/run_duo_stepper_w2.py \
    --n $N --dt $DT --days 5 --out $OUT
$PY $REPO/scripts/validate/fv3_native/w2_duo_oracle_gate.py $OUT --res $N || true
echo DUOW2_DONE
