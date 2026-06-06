#!/bin/bash
# Local runner for CPU MPI scaling benchmarks (no SLURM needed).
#
# Usage:
#   bash scripts/run_cpu_mpi_scaling_local.sh --grid latlon --physics held_suarez --max-ranks 4
#   bash scripts/run_cpu_mpi_scaling_local.sh --grid icosahedral --physics held_suarez --max-ranks 8
#   bash scripts/run_cpu_mpi_scaling_local.sh --grid cubed-sphere --physics gray_sbm --max-ranks 6

set -euo pipefail

GRID="${1:---grid}"
GRID_VAL="${2:-latlon}"
PHYSICS="${3:---physics}"
PHYSICS_VAL="${4:-held_suarez}"
MAX_RANKS_FLAG="${5:---max-ranks}"
MAX_RANKS="${6:-4}"

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --grid) GRID_VAL="$2"; shift 2;;
        --physics) PHYSICS_VAL="$2"; shift 2;;
        --max-ranks) MAX_RANKS="$2"; shift 2;;
        --mode) MODE="$2"; shift 2;;
        --precision) PRECISION="$2"; shift 2;;
        --n-timing) N_TIMING="$2"; shift 2;;
        *) shift;;
    esac
done

MODE="${MODE:-both}"
PRECISION="${PRECISION:-float64}"
N_TIMING="${N_TIMING:-50}"

echo "=================================================="
echo "  CPU MPI Scaling Benchmark (local)"
echo "=================================================="
echo "  Grid:      $GRID_VAL"
echo "  Physics:   $PHYSICS_VAL"
echo "  Max ranks: $MAX_RANKS"
echo "  Mode:      $MODE"
echo "  Precision: $PRECISION"
echo "=================================================="

# Generate cases
CASES=$(python scripts/bench/run_cpu_mpi_scaling.py --sweep \
    --grid "$GRID_VAL" --mode "$MODE" --physics "$PHYSICS_VAL" \
    --max-ranks "$MAX_RANKS" --precision "$PRECISION")

echo "$CASES" | while IFS= read -r CASE; do
    N_RANKS=$(echo "$CASE" | python3 -c "import sys,json; print(json.load(sys.stdin)['n_ranks'])")
    echo ""
    echo "--- Running: n_ranks=$N_RANKS ---"
    echo "    Case: $CASE"
    mpirun -np "$N_RANKS" python scripts/bench/run_cpu_mpi_scaling.py \
        --case "$CASE" --n-timing "$N_TIMING"
done

echo ""
echo "=== Aggregating results ==="
python scripts/bench/aggregate_scaling_results.py results/cpu_scaling/
echo "Done."
