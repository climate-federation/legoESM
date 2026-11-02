#!/bin/bash
#SBATCH --job-name=bgc_offline
#SBATCH --partition=gpu
#SBATCH --account=bd1179_gpu
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1

cd /work/bd1083/b309178/diffESM/legoesm_ggn
JAX_ENABLE_X64=1 .venv/bin/python scripts/run_bgc_offline_oras5.py 2>&1
