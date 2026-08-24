#!/bin/bash
#SBATCH --job-name=arcn2
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/arcn2_%j.out
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM
cd $REPO || exit 1
echo "=== HEAD $(git -C $REPO rev-parse HEAD)"
export PYTHONPATH="$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
GT=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_REF/ORCA1_1m_20000101_20041231_grid_T.nc
$PY -u scripts/validate/ocean_fidelity/arctic_n2_compare.py \
  --snapshot $REPO/results/omip_nemo/nemolev_trp_gwcorr_d30/snapshot_final.npz \
  --nemo-gridt "$GT" --nemo-month 1 \
  --mesh-mask $REPO/data/grids/eORCA1.2_mesh_mask.nc
echo "=== rc=$? ==="
