#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=fv3_mp6gpu
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM_fv3jax/scripts/tmp/fv3_mp6gpu_%j.out

# PER-TASK MULTI-GPU BINDING FIX (tail item 1, 2026-08-27): the prior 3-node
# spmd_multiprocess_parity run bound only 1 GPU/task (3 global devices) --
# SLURM's default GPU allocation is 1 GPU/task unless told otherwise. This job
# requests ONE task per NODE (--ntasks-per-node=1) and TWO GPUs per that task
# (--gpus-per-task=2), so SLURM sets CUDA_VISIBLE_DEVICES to both node GPUs
# for the single task -- jax.distributed.initialize()'s SLURM auto-detect then
# sees 2 local devices per process with no cross-task GPU contention (unlike
# the eORCA025 2-tasks/node template, which needs local_device_ids=SLURM_LOCALID
# to disambiguate SHARED node GPUs -- not needed here since each task owns its
# node's GPUs exclusively). 3 nodes x 2 GPUs = 6 global devices = 6 cube faces,
# the natural (1 face/device, no ring-neighbor grouping) SPMD case.
#
# PREFLIGHT (HARD rule): refuse a dirty tree -- a job must run the PINNED
# commit, never uncommitted edits a later commit could silently change under it.
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
