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
source scripts/cluster/scaling_levante/_env.sh 2>/dev/null || true

OUT=results/a1/cube_tiled_scaling_levante.jsonl
RES=${RES:-192}
NLEV=${NLEV:-60}
KT=${KT:-2}

echo "== Stage A: CPU-virtual parity gate =="
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=24 \
srun -N1 -n1 python scripts/bench/bench_cube_tiled_step_scaling.py \
    --kt "${KT}" --resolution 48 --nlev 10 --steps 4 --warmup 1 \
    --parity-gate --out "${OUT%.jsonl}_parity.jsonl"

echo "== Stage B: 24-GPU multicontroller timed run =="
srun python scripts/bench/bench_cube_tiled_step_scaling.py \
    --kt "${KT}" --resolution "${RES}" --nlev "${NLEV}" \
    --steps 12 --warmup 2 --multicontroller --out "${OUT}"

echo "done: ${OUT}"
