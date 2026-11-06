#!/bin/bash
#SBATCH --account=bb1596
#SBATCH --partition=gpu
#SBATCH --job-name=rrtmgp_audit
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100_80
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/b/b381103/rrtmgp_audit/audit_%j.log
set -euo pipefail
WT=/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/wt_rrtmgp_audit
ROOT=/work/bd1083/b309178/diffESM/legoesm_pg/legoESM
A=/scratch/b/b381103/rrtmgp_audit
export PYTHONPATH="$WT/src:$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
module load gcc/13.4.0-gcc-13.4.0 netcdf-fortran/4.6.2-openmpi-5.0.10-gcc-13.4.0
$ROOT/.venv/bin/python $WT/scripts/validate/oracle/rrtmgp_faithfulness/audit.py \
  --oracle $A/build/oracle_driver --data $A/ref/rrtmgp-data --out $A/run_${TABLES:-ours} \
  --tables ${TABLES:-ours} ${EXTRA:-}
