#!/bin/bash
#SBATCH -A bb1596
#SBATCH -J owb_era5
#SBATCH -p shared
#SBATCH -t 02:00:00
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH -o /work/bb1596/b309178/diffESM/ocean_wb/logs/era5_%A_%a.out
# ERA5 daily ocean forcing 1993-01..2022-12 = month indices 0-359 (on-disk DKRZ ERA5, no network).
#   sbatch --array=0-359%12 scripts/cluster/ocean_wb/era5_ocean_forcing.sbatch
set -euo pipefail
source /usr/share/Modules/init/bash; module load cdo/2.6.0-gcc-11.2.0
REPO=${REPO:-/work/bd1083/b309178/diffESM/legoesm_pg/wt_oceanwb}
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python "$REPO/scripts/data/build_era5_ocean_forcing.py" \
    "$SLURM_ARRAY_TASK_ID" --out /work/bb1596/b309178/diffESM/ocean_wb/era5_daily \
    --work /scratch/b/b381103/oceanwb/era5_work
