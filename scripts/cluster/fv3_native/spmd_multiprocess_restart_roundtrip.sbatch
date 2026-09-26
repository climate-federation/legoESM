#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=fv3_mprt
#SBATCH --nodes=3
#SBATCH --ntasks=6
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM_fv3jax/scripts/tmp/fv3_mprt_%j.out

# Multi-process fv3_duo checkpoint SAVE->LOAD->STEP bitwise round-trip
# (tail item 3 correctness gate). Same 6-device shape as
# spmd_multiprocess_3node_6gpu.sbatch: --gres=gpu:2 --ntasks-per-node=2
# over 3 nodes = 6 tasks x 1 GPU = 6 global devices = 6 cube faces
# (--gpus-per-task is unsupported by this cluster's SLURM). Bare
# jax.distributed.initialize() auto-derives each task's GPU from
# SLURM_LOCALID.
# Proves a checkpoint+restart run is BITWISE the uninterrupted run across
# the multi-process gather/broadcast/reshard I/O path.
#
# PREFLIGHT (HARD): refuse a dirty tree -- must run the PINNED commit.
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
export PYTHONPATH="$REPO/packages/core:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ocean:$REPO/packages/ml:$REPO/packages/tools:$REPO/src"
cd "$REPO"

# The checkpoint dir MUST be on a shared FS visible to rank 0's writer AND
# every reader (NOT node-local /local or /tmp).
OUT=/burg-archive/glab/users/pg2328/fv3_duo_gaps/mprt_ckpt_${SLURM_JOB_ID}

srun --mpi=pmix \
  env JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \
  $PY scripts/validate/fv3_native/spmd_multiprocess_restart_roundtrip.py \
    --resolution 48 --km 5 --n-steps 4 --dt 120.0 --out-dir "$OUT"

echo "spmd_mp_restart_roundtrip_rc=$?"
rm -rf "$OUT" 2>/dev/null
echo DONE
