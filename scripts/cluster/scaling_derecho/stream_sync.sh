#!/bin/bash
# Node memory bandwidth with the model's rank layout: 16 STREAM triad copies,
# 8 threads each on cores 8i..8i+7, one common start, timed over the window in
# which all copies run (scripts/bench/microbench/stream_sync.py).
#   stream_sync.sh OUTDIR [N_MILLIONS=200] [REPS=30]
# Env: CC (default gcc), LDFLAGS (e.g. -Wl,-rpath for a non-system libgomp).
# The published numbers used spack GCC 14.3 with -march=native.
set -euo pipefail
OUT=$1; MEL=${2:-200}; REPS=${3:-30}; mkdir -p "$OUT"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
B="$OUT/stream_triad"; LOG="$OUT/stream_sync_$(hostname).txt"
${CC:-gcc} -O3 -march=native -fopenmp ${LDFLAGS:-} "$REPO/scripts/bench/microbench/stream_triad.c" -o "$B"
T=$(( $(date +%s) + 20 ))
for i in $(seq 0 15); do
  OMP_NUM_THREADS=8 OMP_PROC_BIND=close OMP_PLACES=cores taskset -c $((8*i))-$((8*i+7)) \
    "$B" "$MEL" "$T" "$REPS" | sed "s/^/rank=$i Mel=$MEL /" &
done > "$LOG"; wait
python3 "$REPO/scripts/bench/microbench/stream_sync.py" "$LOG" "$MEL" 16 "$REPS"
