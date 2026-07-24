#!/bin/bash -l
#SBATCH --job-name=legoesm_gpu_scaling
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=legoesm_gpu_scaling.%j.log
# ===========================================================================
# ATM + OCEAN single-node GPU weak+strong scaling on a Levante GPU node
# (2x AMD 7763 + 4x NVIDIA A100-80GB, NVLink). SINGLE PROCESS sharding over
# the node's GPUs — no MPI, no mpi4jax; plain `legoesm-gpu` env.
#
#   ATM   — scripts/bench/run_levante_gpu_scaling.py, cubed-sphere AMIP
#           physics (device subsets 1/2/3 from the 4 allocated; the cube is
#           face-sharded so 4 is not a valid count — 6 needs 2 nodes, see
#           gpu_multinode_scaling.sbatch); icosahedral runs any count.
#   OCEAN — scripts/bench/bench_ocean_mpi_scaling.py --transport spmd
#           (full lat-lon C-grid step sharded over 1/2/4 A100s), with a
#           fail-fast parity + conservation smoke first.
#
# Account: export SBATCH_ACCOUNT=<project> (or sbatch -A <project> ...).
# SUBMIT:
#   sbatch scripts/cluster/scaling_levante/gpu_scaling.sbatch
# OVERRIDE (env, prefix or --export):
#   MODE=strong PRECISION=float64 sbatch --export=ALL ...
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

RUN_ATM="${RUN_ATM:-1}"
RUN_OCEAN="${RUN_OCEAN:-1}"
MODE="${MODE:-both}"
PRECISION="${PRECISION:-both}"
ATM_GRID="${ATM_GRID:-cubed-sphere}"
ATM_PHYSICS="${ATM_PHYSICS:-gray_sbm}"
NLEV_ATM="${NLEV_ATM:-26}"
OC_STRONG_RES="${OC_STRONG_RES:-192}"
OC_WEAK_ROWS="${OC_WEAK_ROWS:-64}"
OC_NLEV="${OC_NLEV:-20}"
# Ocean SPMD bench runs the production-default config (from_flat).
OC_NDEV_LIST="${OC_NDEV_LIST:-1 2 4}"
N_WARMUP="${N_WARMUP:-3}"
N_TIMING="${N_TIMING:-30}"

STAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/gpu_${STAMP}}"
mkdir -p "$OUTDIR"

echo "=== HOST ==="; hostname; date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -8 || true
echo "=== CONFIG ==="
echo "atm=$RUN_ATM ocean=$RUN_OCEAN mode=$MODE precision=$PRECISION"
echo "outdir=$OUTDIR"

rc_all=0

if [ "$RUN_ATM" = "1" ]; then
    echo "=== ATM: $ATM_GRID $ATM_PHYSICS (single-process multi-GPU sweep) ==="
    "$PY" scripts/bench/run_levante_gpu_scaling.py \
        --grid "$ATM_GRID" \
        --physics "$ATM_PHYSICS" \
        --mode "$MODE" \
        --precision "$PRECISION" \
        --n-levels "$NLEV_ATM" \
        --n-gpus 0 \
        --n-warmup "$N_WARMUP" --n-timing "$N_TIMING" \
        --output-dir "$OUTDIR/atm"
    rc=$?
    [ "$rc" -ne 0 ] && { echo "ATM lane FAILED rc=$rc"; rc_all=$rc; }
fi

if [ "$RUN_OCEAN" = "1" ]; then
    echo "=== OCEAN SMOKE: parity + conservation gates (LL32/L6, nd=2) ==="
    "$PY" scripts/bench/bench_ocean_latlon_spmd_scaling.py \
        --n-devices 2 \
        --n-lat 32 --n-lon 64 --nlev 6 \
        --steps 4 --warmup 1 \
        --parity-gate --check-conservation --cons-rtol 1e-6 \
        --out "$OUTDIR/ocean_smoke/ocean_spmd_scaling.jsonl"
    rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "OCEAN SMOKE GATE FAILED rc=$rc — skipping the ocean ladder." >&2
        rc_all=$rc
    else
        case "$MODE" in
            weak)   OC_MODES="weak" ;;
            strong) OC_MODES="strong" ;;
            *)      OC_MODES="strong weak" ;;
        esac
        case "$PRECISION" in
            float32) OC_PRECS="float32" ;;
            float64) OC_PRECS="float64" ;;
            *)       OC_PRECS="float64 float32" ;;
        esac
        for prec in $OC_PRECS; do
            for mode in $OC_MODES; do
                for nd in $OC_NDEV_LIST; do
                    if [ "$mode" = "strong" ]; then res="$OC_STRONG_RES"; else res="$OC_WEAK_ROWS"; fi
                    echo "--- ocean spmd: mode=$mode prec=$prec nd=$nd res=$res ---"
                    if [ "$prec" = "float64" ]; then _X64=1; else _X64=0; fi
                    if [ "$mode" = "strong" ]; then
                        _GEOM="--mode strong --n-lat $res --n-lon $((2 * res))"
                    else
                        _GEOM="--mode weak --nlat-per-dev $res --n-lon $((2 * res))"
                    fi
                    # --force-pcg: solver-matched ladder (nd=1 otherwise
                    # times adaptive stock CG vs fixed-M PCG at nd>1).
                    JAX_ENABLE_X64=$_X64 "$PY" \
                        scripts/bench/bench_ocean_latlon_spmd_scaling.py \
                        --n-devices "$nd" $_GEOM --nlev "$OC_NLEV" \
                        --force-pcg \
                        --steps $((N_WARMUP + N_TIMING)) --warmup "$N_WARMUP" \
                        --out "$OUTDIR/ocean/ocean_spmd_scaling.jsonl"
                    rc=$?
                    [ "$rc" -ne 0 ] && { echo "ocean case FAILED rc=$rc"; rc_all=$rc; }
                done
            done
        done
    fi
fi

echo "=== RESULTS ==="
find "$OUTDIR" -name "*.csv" | head -40
cat "$OUTDIR/ocean/ocean_scaling.csv" 2>/dev/null || true
echo "=== DONE rc=$rc_all ==="
exit "$rc_all"
