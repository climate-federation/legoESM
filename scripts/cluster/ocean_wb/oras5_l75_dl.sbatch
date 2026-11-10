#!/bin/bash
#SBATCH -A bb1596
#SBATCH -J owb_oras5
#SBATCH -p shared
#SBATCH -t 08:00:00
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH -o /work/bb1596/b309178/diffESM/ocean_wb/logs/oras5_%A_%a.out
# ORAS5 daily, 75 levels, 1993-01..2022-12 = month indices 0-359. Shared partition has internet.
#   sbatch --array=0-359%6 scripts/cluster/ocean_wb/oras5_l75_dl.sbatch
set -euo pipefail
S2S=/work/bd1083/b309178/diffESM/S2S_model
REPO=${REPO:-/work/bd1083/b309178/diffESM/legoesm_pg/wt_oceanwb}
"$S2S/.venv-cmems/bin/python" -I "$REPO/scripts/data/download_oras5_cmems_l75.py" \
    "$SLURM_ARRAY_TASK_ID" --out /work/bb1596/b309178/diffESM/ocean_wb/oras5_l75 \
    --raw /scratch/b/b381103/oceanwb/oras5_raw --s2s "$S2S" \
    --client "$S2S/.venv-cmems/bin/copernicusmarine"
