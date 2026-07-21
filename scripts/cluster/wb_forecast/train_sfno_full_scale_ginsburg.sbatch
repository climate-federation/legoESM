#!/bin/bash
#SBATCH --job-name=sfno_scale
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/results/aimip_wbcompare_scale/logs/%x_%j.out
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=72:00:00
# sfno_full GraphCast-scale training on Ginsburg (Derecho sibling of
# scripts/cluster/derecho/train_sfno_full_scale.pbs). Rollout curriculum
# 6h->120h + ~3200 seasonal pairs. Single A100/RTX8000; sfno_full has NO
# data-parallel. JAX_PLATFORMS=cuda,cpu for the host-resident large-data loader.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM
cd "$REPO"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
_pp="$REPO"; for d in "$REPO"/packages/*/; do [ -d "${d}legoesm" ] && _pp="$_pp:$d"; done
export PYTHONPATH="${_pp}:${PYTHONPATH:-}"
export MPI4JAX_NO_WARN_JAX_VERSION=1
export JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export TMPDIR="/local/sfno_scale_${SLURM_JOB_ID}"; mkdir -p "$TMPDIR" results/aimip_wbcompare_scale/logs
SUITE="${WB_SUITE:-config/aimip/wbcompare/suite_sfno_full_scale.yaml}"
echo "[$(date)] sfno_full GraphCast-scale train on $(hostname) | suite=$SUITE"
nvidia-smi -L
"$PY" -u scripts/run/run_aimip.py --suite "$SUITE" ${WB_RESUME:+--resume}
echo "[$(date)] done"
rm -rf "$TMPDIR"
