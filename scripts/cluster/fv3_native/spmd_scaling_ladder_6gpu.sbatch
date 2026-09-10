#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=fv3_scale
#SBATCH --nodes=3
#SBATCH --ntasks=6
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM_fv3jax/scripts/tmp/fv3_scale_%j.out

# FV3 duo multi-NODE SPMD scaling ladder: sharded (6 devices, 3 nodes,
# 1 face/device) vs single-device wall time, at C192 and C384 km=10.
# The one regime the 2xPCIe verdict left open (multi-node aggregate
# bandwidth). --constraint=a100 PINS the card (the 2-GPU timing trap was
# an accidental A100 landing; timing must not straddle card types).
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
DIRTY=$(git -C "$REPO" status --porcelain --untracked-files=no | wc -l)  # tracked edits only
[ "$DIRTY" -eq 0 ] || { echo "PREFLIGHT FAILED: $REPO dirty ($DIRTY)"; exit 3; }
echo "host: $(hostname)"
echo "PINNED sha: $(git -C "$REPO" rev-parse HEAD)"
echo "nodes: $SLURM_JOB_NODELIST  ntasks: $SLURM_NTASKS"
scontrol show hostnames "$SLURM_JOB_NODELIST"

PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH="$REPO/packages/core:$REPO/packages/atmosphere:$REPO/src"
cd "$REPO"
BENCH=scripts/bench/bench_fv3_duo_spmd_scaling.py

for RES in ${RESLIST:-96}; do
  echo "=== C${RES} km=10 : sharded (6 dev, 3 nodes) ==="
  srun --mpi=pmix --nodes=3 --ntasks=6 --ntasks-per-node=2 \
    env JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \
    $PY $BENCH --mode sharded --resolution $RES --km 10 --dt 120.0 \
      --n-warmup 3 --n-timed 20
  echo "C${RES}_sharded_rc=$?"

  echo "=== C${RES} km=10 : single-device (1 GPU) ==="
  srun --nodes=1 --ntasks=1 --gres=gpu:1 \
    env JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \
    $PY $BENCH --mode single --resolution $RES --km 10 --dt 120.0 \
      --n-warmup 3 --n-timed 20
  echo "C${RES}_single_rc=$?"
done
echo DONE
