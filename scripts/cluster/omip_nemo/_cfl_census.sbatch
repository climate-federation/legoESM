#!/bin/bash
#SBATCH --job-name=cflcensus
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=0:20:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cfl_%j.out
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_wt; cd $ROOT || exit 1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
for s in trp_unified180/snapshot_day0030.npz trp_unified180/snapshot_day0090.npz trp_unified180/snapshot_day0150.npz; do
  $PY scripts/validate/ocean_fidelity/top_cell_vertical_cfl_census.py --snapshot results/omip_nemo/$s --dt 150 900 1800 3600
done
echo DONE
