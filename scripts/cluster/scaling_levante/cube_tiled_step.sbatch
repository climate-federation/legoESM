#!/bin/bash
#SBATCH --job-name=legoesm_cube_tiled
#SBATCH --partition=gpu
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=single:1
#SBATCH --time=01:30:00
#SBATCH --output=cube_tiled_%j.log
# ===========================================================================
# Sub-face TILED cube hydrostatic step on Levante: 24 GPUs (6 x 4 A100),
# kt=2 -> (6,2,2) mesh — the >6-GPU strong-scaling lane (audit item 5).
# Stage A: CPU-virtual parity gate; Stage B: multicontroller timed run
# (SLURM auto-detected jax.distributed; --gpu-bind=single:1 pins one GPU
# per task so local_device_ids=[0] is the pinned device).
# See scripts/cluster/scaling_derecho/cube_tiled_step.pbs for the lane's
# contract (dynamics-only envelope; HLO anti-replication refusals).
# ===========================================================================
set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if   [ -f "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh" ]; then
    SCRIPT_DIR="${SUBMIT_DIR}/scripts/cluster/scaling_levante"
elif [ -f "${SUBMIT_DIR}/_env.sh" ]; then
    SCRIPT_DIR="${SUBMIT_DIR}"
else
    echo "ERROR: cannot find _env.sh from SLURM_SUBMIT_DIR=${SUBMIT_DIR}" >&2
    exit 1
fi

export LEGOESM_CONDA_ENV="${LEGOESM_CONDA_ENV:-legoesm-gpu}"
# shellcheck source=_env.sh
source "${SCRIPT_DIR}/_env.sh"
cd "$REPO"

OUT="${OUT:-results/a1/cube_tiled_scaling_levante.jsonl}"
RES="${RES:-192}"
NLEV="${NLEV:-60}"
KT="${KT:-2}"
# This job reserves EXACTLY 24 GPUs; a different kt needs its own
# --nodes/--ntasks allocation (see the PBS twin's refusal note).
if [ "${KT}" != "2" ]; then
    echo "ERROR: this 24-GPU template is kt=2 only; clone it with a" \
         "matching allocation for KT=${KT}." >&2
    exit 1
fi

echo "== Stage A: CPU-virtual parity gates =="
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=24 \
srun -N1 -n1 "$PY" scripts/bench/bench_cube_tiled_step_scaling.py \
    --kt "${KT}" --resolution 48 --nlev 10 --steps 4 --warmup 1 \
    --parity-gate --out "${OUT%.jsonl}_parity.jsonl"
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=24 \
srun -N1 -n1 "$PY" scripts/bench/bench_cube_tiled_step_scaling.py \
    --kt "${KT}" --resolution 48 --nlev 10 --steps 4 --warmup 1 \
    --closed-loop --parity-gate --out "${OUT%.jsonl}_loop_parity.jsonl"

echo "== Stage B: 24-GPU multicontroller timed runs =="
export JAX_PLATFORMS=cuda
srun "$PY" scripts/bench/bench_cube_tiled_step_scaling.py \
    --kt "${KT}" --resolution "${RES}" --nlev "${NLEV}" \
    --steps 12 --warmup 2 --multicontroller --out "${OUT}"
# Closed-loop production lane (state resident tile-sharded across steps).
srun "$PY" scripts/bench/bench_cube_tiled_step_scaling.py \
    --kt "${KT}" --resolution "${RES}" --nlev "${NLEV}" \
    --steps 12 --warmup 2 --multicontroller --closed-loop --out "${OUT}"

echo "done: ${OUT}"
