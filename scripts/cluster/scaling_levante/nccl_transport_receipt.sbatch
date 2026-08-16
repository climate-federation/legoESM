#!/bin/bash -l
#SBATCH --job-name=nccl_net_receipt
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu-devel
#SBATCH --nodes=2
#SBATCH --gpus-per-node=2
#SBATCH --time=00:20:00
#SBATCH --output=nccl_net_receipt.%j.log
# TRANSPORT RECEIPT (theoretical_limit_assessment_2026-08-16.md section 0).
# The early_init warning claims cross-node collectives "will likely run on
# TCP SOCKETS" because no NCCL net plugin is visible. On Mellanox IB the
# builtin NET/IB verbs transport needs NO plugin, so the warning is
# plausibly a false alarm — this job decides it.
#   CONFIRMS (warning false alarm): NCCL INFO line "Using network IB".
#   REFUTES  (warning real):        "Using network Socket" — then every
#   multi-node point in the scaling figure is transport-degraded and the
#   plugin/env fix outranks every other lever.
# Cheap by design: 2 vader nodes x 2 A100_40, tiny LL256 atm case, 5 steps.
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/nccl_receipt_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"
JAX_ENABLE_X64=0 srun --ntasks=4 --ntasks-per-node=2 \
    --gpus-per-node=2 --gpu-bind=none --kill-on-bad-exit=1 \
  "$PY" scripts/bench/bench_atm_latlon_spmd_scaling.py \
    --multicontroller --n-devices 4 --mode strong \
    --n-lat 256 --n-lon 512 --nlev 26 --steps 5 --warmup 2 \
    --out "$OUTDIR/receipt.jsonl" || { echo "bench FAILED"; }
echo "=== TRANSPORT VERDICT ==="
# The decisive NCCL line; grep the job's own log (NCCL prints to stderr).
grep -h -m4 "Using network" "nccl_net_receipt.${SLURM_JOB_ID}.log" \
  || echo "NO 'Using network' LINE FOUND — verdict UNDECIDED, rerun"
