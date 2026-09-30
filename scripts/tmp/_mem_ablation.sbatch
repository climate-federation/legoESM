#!/usr/bin/env bash
# Which component holds the WB training gradient's memory? Compile-only.
#
# One arm per component, each differing from `base` in EXACTLY ONE deck key
# (audited when the decks were written). Reports XLA's own temp/arg/out bytes,
# NOT the HLO-buffer ranking -- that ranking is what pointed the first
# diagnosis at the biggest TENSOR (radiation) rather than the biggest
# CONSUMER, and blocking radiation moved the total by 0.3%.
#
# CPU on purpose: the compile is the cost, no GPU is needed to read the
# estimate, and CPU scratch (490.7 GiB measured) tracked the GPU arena
# (376 GiB) closely enough to RANK. The winner gets confirmed on a GPU after.
#SBATCH --job-name=wbmemabl
#SBATCH --partition=compute
#SBATCH --account=bb1596
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=wbmemabl_%j.out
set -uo pipefail
export LEGOESM_REPO="${LEGOESM_REPO:-${SLURM_SUBMIT_DIR:?set LEGOESM_REPO}}"
source "$LEGOESM_REPO/scripts/cluster/scaling_levante/_env.sh" || exit 1
cd "$REPO" || exit 1
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cpu
export LEGOESM_WB_MEM_PROBE=1
export LEGOESM_JIT_CACHE_DIR=""

echo "[memabl] git $(git rev-parse --short HEAD) $(date)"
for arm in ${ARMS:?set ARMS}; do
  export LEGOESM_WB_MEM_ARM="$arm"
  echo "---- arm $arm $(date +%H:%M) ----"
  timeout 5400 "$PY" -u scripts/run/train_weatherbench_scale.py \
    --config "config/wb/campaign/_mem_${arm}.yaml" \
    --mode physics --training-core spectral \
    --out "$REPO/results/_mem_${arm}" 2>&1 | grep -E "MEMRESULT|Error|error:|Traceback|MemoryError" | head -5
done
echo "[memabl] done $(date)"
grep -h MEMRESULT "$SLURM_SUBMIT_DIR/wbmemabl_${SLURM_JOB_ID}.out" || true
