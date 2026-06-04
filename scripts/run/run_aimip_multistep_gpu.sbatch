#!/usr/bin/env bash
#SBATCH --job-name=aimip_multistep
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=results/aimip_multistep_001/slurm-%j.out
#SBATCH --error=results/aimip_multistep_001/slurm-%j.err
#
# GenCast-style multi-step autoregressive + CRPS training for the
# AIMIP intercomparison suite (classical / column_nn / sfno_physics).
# All three variants train sequentially on a single GPU; per-variant
# expected wall clock at T21 L8 gray-radiation + K=4 chained 6h
# rollouts is ~30-60 min, total ~2-3 h.
#
# Submit with:
#   sbatch scripts/run_aimip_multistep_gpu.sbatch
#
# Re-running is safe: ``run_aimip.py`` merges new variant results into
# any existing ``aimip_scorecard.json``.

set -euo pipefail

REPO=/burg-archive/glab/users/pg2328/legoESM
cd "$REPO"
mkdir -p results/aimip_multistep_001

# Conda env with jax 0.9.1 + jax-cuda12-plugin (pip-bundled CUDA
# 12.8 wheels) -- self-contained, no system CUDA module needed.
PYTHON=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

# Use pg2328's repo (where the multi-step training code lives), not
# the jn2808 editable-install path the conda env imports by default.
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"

export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1
# Lower XLA preallocate so RRTMGP gradient buffers don't OOM; the
# per-variant compile already allocates >1 GiB so leave half for the
# scan buffers.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
# Forward HOME so pip/conda credentials/configs are found.
export HOME=${HOME:-/burg-archive/home/pg2328}

echo "[aimip-multistep] $(date)  host=$(hostname)  GPU info:"
nvidia-smi -L 2>&1 | head -5 || echo "  (nvidia-smi unavailable)"
echo "[aimip-multistep] JAX devices:"
$PYTHON - <<'PY'
import os
os.environ.setdefault("JAX_PLATFORMS", "cuda")
import jax
print("  jax:", jax.__version__, " devices:", jax.devices())
PY

echo "[aimip-multistep] Suite manifest: config/aimip/aimip_suite_multistep.yaml"
$PYTHON -u scripts/run/run_aimip.py \
    --suite config/aimip/aimip_suite_multistep.yaml

echo "[aimip-multistep] Scorecard:"
ls -la results/aimip_multistep_001/aimip_scorecard.json || true
cat results/aimip_multistep_001/aimip_scorecard.json 2>/dev/null | head -100 || true

echo "[aimip-multistep] Done $(date)"
