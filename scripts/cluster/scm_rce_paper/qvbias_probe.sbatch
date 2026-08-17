#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=qvbias
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/qvbias_%j.out
set +e
WT=/burg-archive/glab/users/pg2328/_wt_tqtune
export JAX_PLATFORMS=cpu
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$WT
export LEGOESM_JIT_CACHE_DIR=""
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
JAX_ENABLE_X64=1 $PY /tmp/qvbias.py 2>&1 | grep -v "^E[0-9]\|pjrt_executable\|cpu_aot_loader\|cuInit\|CUDA"
echo ALLDONE
