#!/bin/bash
#SBATCH --job-name=wb_aimip_neural
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/results/aimip_wbcompare/logs/%x_%j.out
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
# Train the two NEURAL WB-compare variants (column_nn radiation=none + sfno_full)
# via run_aimip's suite runner. No rrtmgp -> fast. classical (winner+rrtmgp) comes
# from the dense classical run. Targets the CURRENT checkout (has run_aimip +
# sfno_full support + the wbcompare configs). PYTHONPATH = this repo's packages/*
# namespace (memory python-env); reuses jn2808's conda GPU python.
set -uo pipefail

REPO=/burg-archive/glab/users/pg2328/legoESM
cd "$REPO"
PY_GPU="/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python"
_pp="$REPO"
for _d in "$REPO"/packages/*/; do
  [ -d "${_d}legoesm" ] && _pp="${_pp}:${_d}"
done
export PYTHONPATH="${_pp}:${PYTHONPATH:-}"
export MPI4JAX_NO_WARN_JAX_VERSION=1
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export TMPDIR="/local/wb_aimip_neural_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" results/aimip_wbcompare/logs

SUITE="${WB_NEURAL_SUITE:-config/aimip/wbcompare/suite_neural.yaml}"
echo "[$(date)] WB-compare neural train on $(hostname) | suite=$SUITE"
nvidia-smi -L
"$PY_GPU" -u scripts/run/run_aimip.py --suite "$SUITE" ${WB_RESUME:+--resume}
echo "[$(date)] done -> results/aimip_wbcompare/{column_nn,sfno_full}"
rm -rf "$TMPDIR"
