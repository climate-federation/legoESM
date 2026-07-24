#!/bin/bash -l
#SBATCH --job-name=legoesm_cpu_scaling
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=legoesm_cpu_scaling.%j.log
# ===========================================================================
# ATM + OCEAN CPU-MPI weak+strong scaling on Levante `compute` nodes
# (2x AMD 7763 = 128 cores). Mirrors scaling_derecho/{cpu,ocean_cpu}_scaling
# so the CPU curves are directly comparable across machines.
#
#   ATM   — run_cpu_mpi_scaling.py sweep (latlon latitude-band + icosahedral
#           cell-partition; moist baroclinic wave), rank ladder <= MAX_RANKS.
#   OCEAN — bench_ocean_mpi_scaling.py (lat-lon C-grid band MPI), weak+strong
#           ladder with a fail-fast 2-rank parity+conservation smoke.
#
# Env: `legoesm-mpi` — jax/jaxlib 0.9.2 CPU + mpi4py/mpi4jax source-built
# against the Levante OpenMPI (module openmpi/4.1.x; see README).
#
# Rank policy: the Ginsburg campaign measured DRAM-bandwidth contention above
# ~8-16 ranks/node for the bandwidth-bound dynamics; the within-node ladder
# here measures that envelope on Milan. Multi-node: sbatch --nodes=N and
# raise MAX_RANKS/RANKS; srun spreads tasks round-robin.
#
# Account: export SBATCH_ACCOUNT=<project> (or sbatch -A <project> ...).
# SUBMIT:
#   sbatch scripts/cluster/scaling_levante/cpu_scaling.sbatch
# ===========================================================================
set -uo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if   [ -f "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh" ]; then
    SCRIPT_DIR="${SUBMIT_DIR}/scripts/cluster/scaling_levante"
elif [ -f "${SUBMIT_DIR}/_env.sh" ]; then
    SCRIPT_DIR="${SUBMIT_DIR}"
else
    echo "ERROR: cannot find _env.sh from SLURM_SUBMIT_DIR=${SUBMIT_DIR}" >&2
    exit 1
fi

export JAX_PLATFORMS=cpu
export LEGOESM_CONDA_ENV="${LEGOESM_CONDA_ENV:-legoesm-mpi}"
# shellcheck source=_env.sh
source "${SCRIPT_DIR}/_env.sh"
cd "$REPO"

module load openmpi 2>/dev/null || true

RUN_ATM="${RUN_ATM:-1}"
RUN_OCEAN="${RUN_OCEAN:-1}"
GRID="${GRID:-latlon}"              # atm: latlon | icosahedral
PHYSICS="${PHYSICS:-moist}"
MODE="${MODE:-both}"
PRECISION="${PRECISION:-float64}"
NLEV_ATM="${NLEV_ATM:-26}"
MAX_RANKS="${MAX_RANKS:-64}"
OC_STRONG_RES="${OC_STRONG_RES:-192}"
OC_WEAK_ROWS="${OC_WEAK_ROWS:-64}"
OC_NLEV="${OC_NLEV:-20}"
OC_BARO="${OC_BARO:-implicit_cn}"
OC_RANKS="${OC_RANKS:-1 2 4 8 16 32}"
N_WARMUP="${N_WARMUP:-3}"
N_TIMING="${N_TIMING:-30}"

STAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/cpu_${STAMP}}"
mkdir -p "$OUTDIR"

echo "=== HOST ==="; hostname; date
echo "=== CONFIG ==="
echo "atm=$RUN_ATM($GRID/$PHYSICS) ocean=$RUN_OCEAN(baro=$OC_BARO) mode=$MODE prec=$PRECISION max_ranks=$MAX_RANKS"
echo "outdir=$OUTDIR"

rc_all=0

if [ "$RUN_ATM" = "1" ]; then
    echo "=== ATM sweep: $GRID $PHYSICS ==="
    CASE_FILE="$OUTDIR/atm_cases.jsonl"
    "$PY" scripts/bench/run_cpu_mpi_scaling.py --sweep \
        --grid "$GRID" --mode "$MODE" --physics "$PHYSICS" \
        --precision "$PRECISION" --n-levels "$NLEV_ATM" \
        --max-ranks "$MAX_RANKS" > "$CASE_FILE"
    cat "$CASE_FILE"
    i=0
    while IFS= read -r -u 9 CASE; do
        [ -z "$CASE" ] && continue
        N_RANKS=$("$PY" -c 'import sys,json; print(json.loads(sys.argv[1])["n_ranks"])' "$CASE")
        echo "--- atm case $i: n_ranks=$N_RANKS ---"
        srun --ntasks="$N_RANKS" --kill-on-bad-exit=1 \
            "$PY" scripts/bench/run_cpu_mpi_scaling.py \
            --case "$CASE" \
            --n-warmup "$N_WARMUP" --n-timing "$N_TIMING" \
            --output-dir "$OUTDIR/atm" < /dev/null
        rc=$?
        [ "$rc" -ne 0 ] && { echo "atm case $i FAILED rc=$rc"; rc_all=$rc; }
        i=$((i + 1))
    done 9< "$CASE_FILE"
fi

if [ "$RUN_OCEAN" = "1" ]; then
    echo "=== OCEAN SMOKE: parity + conservation gates (LL64, np=2) ==="
    srun --ntasks=2 --kill-on-bad-exit=1 \
        "$PY" scripts/bench/bench_ocean_mpi_scaling.py \
        --mode strong --resolution 64 --n-levels 6 \
        --baro-solver explicit_substep \
        --n-warmup 2 --n-timing 6 \
        --parity-gate --check-conservation --cons-rtol 1e-6 \
        --output-dir "$OUTDIR/ocean_smoke" < /dev/null
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
        # Solver-matched ladder (codex 2026-07-24): implicit_cn dispatches
        # adaptive stock CG at np=1 but fixed-M PCG at np>1 — pass
        # --force-pcg so every leg times the SAME solver.
        OC_FORCE=""
        [ "$OC_BARO" = "implicit_cn" ] && OC_FORCE="--force-pcg"
        for mode in $OC_MODES; do
            if [ "$mode" = "strong" ]; then res="$OC_STRONG_RES"; else res="$OC_WEAK_ROWS"; fi
            for np in $OC_RANKS; do
                echo "--- ocean mpi: mode=$mode np=$np res=$res baro=$OC_BARO $OC_FORCE ---"
                # shellcheck disable=SC2086
                srun --ntasks="$np" --kill-on-bad-exit=1 \
                    "$PY" scripts/bench/bench_ocean_mpi_scaling.py \
                    --mode "$mode" --resolution "$res" --n-levels "$OC_NLEV" \
                    --baro-solver "$OC_BARO" $OC_FORCE --precision "$PRECISION" \
                    --n-warmup "$N_WARMUP" --n-timing "$N_TIMING" \
                    --output-dir "$OUTDIR/ocean" < /dev/null
                rc=$?
                [ "$rc" -ne 0 ] && { echo "ocean case FAILED rc=$rc"; rc_all=$rc; }
            done
        done
    fi
fi

echo "=== RESULTS ==="
cat "$OUTDIR/ocean/ocean_scaling.csv" 2>/dev/null || true
echo "=== DONE rc=$rc_all ==="
exit "$rc_all"
