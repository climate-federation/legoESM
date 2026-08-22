#!/bin/bash -l
# Can a layout constraint stop XLA converting the scan carry back and forth?
# See scripts/validate/scan_carry_layout_probe.py for what is measured and how
# to read it. One GPU, seconds; this decides whether the last two layout
# conversions on the lat-lon lane are reachable without changing the model's
# state axis order.
#SBATCH --job-name=ll_carry
#SBATCH --account=bd1179_gpu
#SBATCH --partition=gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=ll_carry.%j.log
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
export XLA_FLAGS="--xla_gpu_shard_autotuning=false"
"$PY" scripts/validate/scan_carry_layout_probe.py --selftest || exit 1
echo "=== the shard's shape at 128 devices ==="
srun --export=ALL --nodes=1 --ntasks=1 --gpus-per-node=1 --gpu-bind=none \
  "$PY" scripts/validate/scan_carry_layout_probe.py --rows 16 --lon 4096 --lev 26
echo "=== the shard's shape at 32 devices ==="
srun --export=ALL --nodes=1 --ntasks=1 --gpus-per-node=1 --gpu-bind=none \
  "$PY" scripts/validate/scan_carry_layout_probe.py --rows 64 --lon 4096 --lev 26
echo DONE
