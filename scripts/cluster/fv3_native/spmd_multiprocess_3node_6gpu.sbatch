#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=fv3_mp6gpu
#SBATCH --nodes=3
#SBATCH --ntasks=6
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM_fv3jax/scripts/tmp/fv3_mp6gpu_%j.out

# 6-DEVICE MULTI-NODE SPMD (tail item 1, 2026-08-27): the prior 3-node run
# saw only 1 GPU/task = 3 global devices. --gpus-per-task=2 is UNSUPPORTED
# by this cluster's SLURM SelectType plugin, so the fix is the proven
# eORCA025-multinode resource shape instead: --gres=gpu:2 --ntasks-per-node=2
# over 3 nodes = 6 tasks, ONE GPU per task, 6 GLOBAL devices = 6 cube faces
# (1 face/device). Bare jax.distributed.initialize() under SLURM auto-derives
# each task's local_device_ids from SLURM_LOCALID (jax's own SLURM cluster
# detection), so the two co-located tasks bind DIFFERENT GPUs with no manual
# --gpu-bind (which the eORCA025 template warns deadlocks the cross-node
# topology exchange).
#
# PREFLIGHT (HARD): refuse a dirty TRACKED tree -- must run the pinned commit.
set -uo pipefail
if [ -n "${SLURM_JOB_ID:-}" ] && command -v scontrol >/dev/null 2>&1; then
  _SELF=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
          | tr ' ' '\n' | sed -n 's/^Command=//p' | head -1)
fi
[ -n "${_SELF:-}" ] || _SELF="${BASH_SOURCE[0]}"
REPO=$(cd "$(dirname "$_SELF")/../../.." && pwd)
[ -e "$REPO/.git" ] || { echo "PREFLIGHT FAILED: REPO=$REPO"; exit 5; }
DIRTY=$(git -C "$REPO" status --porcelain --untracked-files=no | wc -l)  # tracked edits only; untracked scratch from other sessions is expected
[ "$DIRTY" -eq 0 ] || { echo "PREFLIGHT FAILED: $REPO dirty ($DIRTY)"; exit 3; }
echo "host: $(hostname)"
echo "PINNED sha: $(git -C "$REPO" rev-parse HEAD)"
echo "nodes: $SLURM_JOB_NODELIST  ntasks: $SLURM_NTASKS"
scontrol show hostnames "$SLURM_JOB_NODELIST"

PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH="$REPO/packages/core:$REPO/packages/atmosphere:$REPO/src"
cd "$REPO"

srun --mpi=pmix \
  env JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \
  $PY scripts/validate/fv3_native/spmd_multiprocess_parity.py \
    --resolution 48 --km 5 --n-steps 3 --dt 120.0

echo "spmd_mp_3node_6gpu_rc=$?"
echo DONE
