#!/usr/bin/env bash
#SBATCH --job-name=aimip_gpu_smoke
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=results/aimip_multistep_001/smoke-%j.out
#SBATCH --error=results/aimip_multistep_001/smoke-%j.err
#
# 30-min smoke test: confirms JAX-on-GPU works and the multi-step
# loss path traces end-to-end on the real ERA5 ingestion + spectral
# PE dycore.  Uses ``--smoke`` to force T21 L8 1 epoch 2 train days.
# The new ``aimip_suite_multistep.yaml`` keeps the ``multi_step_hours``
# loss field active, so the smoke path exercises the K=4 chained
# rollouts that the production submission needs.

set -euo pipefail

REPO=/burg-archive/glab/users/pg2328/legoESM
cd "$REPO"
mkdir -p results/aimip_multistep_001

PYTHON=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export HOME=${HOME:-/burg-archive/home/pg2328}

echo "[smoke] $(date)  host=$(hostname)"
echo "[smoke] GPU info:"
nvidia-smi 2>&1 | head -20 || echo "  nvidia-smi unavailable"
echo "[smoke] JAX devices:"
$PYTHON - <<'PY'
import os
os.environ.setdefault("JAX_PLATFORMS", "cuda")
import jax
print("  jax:", jax.__version__)
print("  devices:", jax.devices())
print("  default backend:", jax.default_backend())
PY

# Run smoke variant of the multi-step suite (only the classical
# variant -- if that traces, the column_nn and sfno_physics variants
# will too since they share the training loop).
$PYTHON -u scripts/run/run_aimip.py \
    --suite config/aimip/aimip_suite_multistep.yaml \
    --variants classical \
    --smoke

ls -la results/aimip_multistep_001_smoke 2>&1 | head -10 || true
echo "[smoke] Scorecard:"
find results -name "aimip_scorecard.json" -mmin -60 -exec cat {} \;
echo "[smoke] Done $(date)"
