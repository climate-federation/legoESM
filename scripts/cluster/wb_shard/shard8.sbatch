#!/bin/bash
#SBATCH --job-name=shard8
#SBATCH --account=bb1596
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/b/b381103/shard8_%j.out

set -u
WT=/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/wt_wbmem
cd "$WT"

# srun refuses to start when it is handed two different CPU counts -- the
# batch step exports SLURM_CPUS_PER_TASK from the node's full core count
# while --cpus-per-task sets SLURM_TRES_PER_TASK, and the mismatch is fatal
# before any Python runs.  Clear both and state the count on the srun line.
unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK

PP=$(ls -d packages/*/ | sed 's|/$||' | tr '\n' ':')src
export PYTHONPATH="$PP"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export SHARD_RAD=1
export LEGOESM_DIST=1
export DECK=config/wb/campaign/spectral_t63_amip.yaml

echo "nodes=$SLURM_JOB_NUM_NODES account=bb1596 deck=$DECK"
# No --gpus-per-task here: it hands each rank ONE device, so the job saw
# two GPUs instead of eight.  The node-level --gres=gpu:4 already gives
# each rank all four of its node's cards.
srun --ntasks=2 --ntasks-per-node=1 --cpus-per-task=48 --cpu-bind=none \
     ../.venv/bin/python scripts/validate/wb_shard_arena.py 63 32 8
