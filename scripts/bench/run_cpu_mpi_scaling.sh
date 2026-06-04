#!/bin/bash
#SBATCH --job-name=legoesm-cpu-scaling
#SBATCH --partition=compute
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=results/cpu_scaling/slurm_%A_%a.out
#
# CPU MPI scaling benchmark launcher.
#
# Usage:
#   # Generate cases and submit as SLURM array:
#   CASES=$(python scripts/run_cpu_mpi_scaling.py --sweep \
#       --grid latlon --mode both --physics held_suarez --max-ranks 64)
#   N_CASES=$(echo "$CASES" | wc -l)
#   echo "$CASES" > /tmp/scaling_cases.txt
#   sbatch --array=0-$((N_CASES-1)) scripts/run_cpu_mpi_scaling.sh /tmp/scaling_cases.txt
#
#   # Or run directly (no SLURM):
#   bash scripts/run_cpu_mpi_scaling.sh /tmp/scaling_cases.txt 0

set -euo pipefail

CASE_FILE="${1:?Usage: $0 <case_file> [task_id]}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-${2:-0}}"

# Read the case JSON for this task
CASE=$(sed -n "$((TASK_ID + 1))p" "$CASE_FILE")
if [ -z "$CASE" ]; then
    echo "ERROR: No case at index $TASK_ID in $CASE_FILE"
    exit 1
fi

# Extract n_ranks from the case
N_RANKS=$(echo "$CASE" | python3 -c "import sys,json; print(json.load(sys.stdin)['n_ranks'])")

echo "Task $TASK_ID: n_ranks=$N_RANKS, case=$CASE"

# Run the benchmark
if command -v srun &> /dev/null; then
    srun --ntasks="$N_RANKS" --cpus-per-task=1 \
        python scripts/run_cpu_mpi_scaling.py --case "$CASE"
else
    mpirun -np "$N_RANKS" \
        python scripts/run_cpu_mpi_scaling.py --case "$CASE"
fi
