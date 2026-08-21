#!/bin/bash -l
#SBATCH --job-name=wide_parity_x64
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:45:00
#SBATCH --exclude=l50003,l50139,l50027,l50009,l50100
#SBATCH --output=wide_parity_x64.%j.log
# ===========================================================================
# Is the s6@8 wide-halo u-parity mismatch (2.662e-03, IDENTICAL under
# ragged and coloured ppermute — job 27039649 arms a/c/e) a real wide-path
# defect or an f32 tolerance artifact?
# Rule: an fp32 mismatch is NOT a failure until re-run with x64.
#   arm 1: s6@8 wide=1 ppermute x64 — mismatch persists ~1e-3 => REAL
#          defect in the wide path at 8 parts; collapses to ~1e-12 =>
#          f32 accumulation artifact, parity tolerance needs precision/
#          partition-count scaling.
#   arm 2: s6@4 wide=1 ppermute x64 — the receipted-good config, control.
#   arm 3: s6@8 wide=0 ppermute x64 — narrow at 8 parts, isolates WIDE
#          (if this also mismatches, the 8-part split itself is off).
# ===========================================================================
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_XLA_OVERLAP=0
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
export LEGOESM_JIT_CACHE_DIR="$SCRATCH/legoesm_jit_cache_j${SLURM_JOB_ID}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/wparity_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
export XLA_FLAGS="--xla_gpu_shard_autotuning=false"

arm () { # tag ndev nodes wide
  local TAG=$1 NDEV=$2 NNODES=$3 WIDE=$4
  echo "=== arm $TAG ndev=$NDEV wide=$WIDE x64=1 $(date +%T) ==="
  env LEGOESM_MPAS_WIDE_HALO="$WIDE" LEGOESM_MPAS_RAGGED_HALO=0 \
      JAX_ENABLE_X64=1 \
    timeout 600 srun --export=ALL --nodes="$NNODES" --ntasks="$NDEV" \
      --ntasks-per-node=4 --gpus-per-node=4 \
      --gpu-bind=none --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
      --multicontroller --n-devices "$NDEV" --subdivision 6 --nlev 8 \
      --steps 4 --warmup 1 --lloyd 0 --partition-method sfc \
      --halo-strategy ppermute --parity-gate \
      --out "$OUTDIR/${TAG}.jsonl" \
    && echo "arm $TAG PARITY OK $(date +%T)" \
    || echo "arm $TAG PARITY FAIL $(date +%T)"
  sleep 10
}

arm wide8 8 2 1
arm wide4 4 1 1
arm narrow8 8 2 0
echo "DONE"; exit 0
