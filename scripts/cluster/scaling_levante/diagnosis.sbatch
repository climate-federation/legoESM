#!/bin/bash -l
#SBATCH --job-name=legoesm_diag
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=legoesm_diag.%j.log
# ===========================================================================
# BOTTLENECK DIAGNOSIS on a Levante GPU node (4x A100-80GB, NVLink).
# Levante twin of scaling_derecho/diagnosis.pbs — identical tool + ladder.
#
# Runs scripts/bench/run_scaling_diagnosis.py — the per-phase bottleneck tool
# (halo bandwidth, reduction latency, roofline, compute/comm overlap, and the
# STATIC collective census: collective-permute + all-reduce + all-gather per
# step) — which the throughput jobs (gpu_scaling.sbatch) do NOT capture.
#
# Why this matters: the measured f64==f32 GPU strong-scaling curves mean the
# multi-GPU leg is LATENCY-bound, so the per-step MESSAGE COUNT — captured by
# the census below — is the lever, not the byte volume.  Cube face-sharding
# requires the device count to DIVIDE 6, so a 4-GPU node climbs the 1/2/3
# ladder (>6 uses cube_tiled_step.sbatch); the count-vs-shard curve across
# 1->2->3 IS the anti-scaling signal.
#
# Account: export SBATCH_ACCOUNT=<project> (or sbatch -A <project> ...).
# SUBMIT:
#   sbatch scripts/cluster/scaling_levante/diagnosis.sbatch
# OVERRIDE (env prefix or --export):
#   N=96 NLEV=40 MODE=full sbatch --export=ALL scripts/.../diagnosis.sbatch
#
# Knobs:
#   N (default 96)  NLEV (40)  MODE (full|census, default full)
#   PRECISION (float64)  NDEV_LIST (default "1 2 3"; must divide 6)
# ===========================================================================
set -uo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if   [ -f "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh" ]; then
    SCRIPT_DIR="${SUBMIT_DIR}/scripts/cluster/scaling_levante"
elif [ -f "${SUBMIT_DIR}/_env.sh" ]; then
    SCRIPT_DIR="${SUBMIT_DIR}"
else
    echo "ERROR: cannot find _env.sh from SLURM_SUBMIT_DIR=${SUBMIT_DIR}" >&2
    echo "       sbatch from the repo root or from scripts/cluster/scaling_levante/." >&2
    exit 1
fi

export JAX_PLATFORMS=cuda
export LEGOESM_CONDA_ENV="${LEGOESM_CONDA_ENV:-legoesm-gpu}"
# shellcheck source=_env.sh
source "${SCRIPT_DIR}/_env.sh"
cd "$REPO"

N="${N:-96}"
NLEV="${NLEV:-40}"
MODE="${MODE:-full}"
PRECISION="${PRECISION:-float64}"
NDEV_LIST="${NDEV_LIST:-1 2 3}"

STAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/diagnosis_${STAMP}_j${SLURM_JOB_ID:-local}}"
mkdir -p "$OUTDIR"

echo "=== HOST ==="; hostname; date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -8 || true
echo "=== CONFIG ==="
echo "grid=cubed-sphere N=$N NLEV=$NLEV mode=$MODE precision=$PRECISION ndev=[$NDEV_LIST]"
echo "outdir=$OUTDIR"

case "$MODE" in
    full|census) : ;;
    *) echo "ERROR: MODE=$MODE (full|census)"; exit 2 ;;
esac

# --- 1) Collective-census ladder: message-count vs shard count --------------
rc_all=0
for nd in $NDEV_LIST; do
    if [ $((6 % nd)) -ne 0 ]; then
        echo "SKIP nd=$nd (does not divide 6 — not a valid cube face-shard; " \
             ">6 uses cube_tiled_step.sbatch)"; continue
    fi
    vis=$(seq -s, 0 $((nd - 1)))
    echo "--- census: ${nd} device(s) (CUDA_VISIBLE_DEVICES=$vis) ---"
    CUDA_VISIBLE_DEVICES="$vis" "$PY" scripts/bench/run_scaling_diagnosis.py \
        --mode census --grid cubed-sphere --n "$N" --nlev "$NLEV" \
        --precision "$PRECISION" --expect-devices "$nd" \
        --output-dir "$OUTDIR/census_nd${nd}" --no-timestamp
    rc=$?
    [ "$rc" -ne 0 ] && { echo "census nd=$nd FAILED rc=$rc"; rc_all=$rc; }
done

# --- 2) Full per-phase diagnosis at the top of the ladder -------------------
if [ "$MODE" = "full" ]; then
    top_nd=1
    for nd in $NDEV_LIST; do
        [ $((6 % nd)) -eq 0 ] && [ "$nd" -gt "$top_nd" ] && top_nd=$nd
    done
    vis=$(seq -s, 0 $((top_nd - 1)))
    echo "=== FULL per-phase diagnosis: ${top_nd} device(s) ==="
    CUDA_VISIBLE_DEVICES="$vis" "$PY" scripts/bench/run_scaling_diagnosis.py \
        --mode full --grid cubed-sphere --n "$N" --nlev "$NLEV" \
        --precision "$PRECISION" --expect-devices "$top_nd" \
        --output-dir "$OUTDIR/full_nd${top_nd}" --no-timestamp
    rc=$?
    [ "$rc" -ne 0 ] && { echo "full diagnosis FAILED rc=$rc"; rc_all=$rc; }
fi

echo "=== RESULTS ==="
find "$OUTDIR" -name "summary.json" -print 2>/dev/null || true
echo "Pull back + analyze:"
echo "  scp -r levante:$OUTDIR ."
echo "  python scripts/bench/analyze_scaling_results.py $OUTDIR/full_nd*/ --format markdown"
echo "=== DONE rc=$rc_all ==="
exit "$rc_all"
