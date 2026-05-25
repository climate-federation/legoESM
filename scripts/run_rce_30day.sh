#!/usr/bin/env bash
# 30-day RCE @ 132x132, dx=2 km, 12 MPI ranks, full physics.
# Hourly 3D MSE/qv/T snapshots for GIF, daily surface snapshots, 5-day profiles.
#
# Usage:
#   scripts/run_rce_30day.sh                  # default output dir results/rce_30day
#   scripts/run_rce_30day.sh results/myrun    # custom output dir
#   DAYS=1 scripts/run_rce_30day.sh results/rce_smoke   # short smoke run
#   RANKS=4 scripts/run_rce_30day.sh ...      # custom rank count
#
# Env vars:
#   DAYS    simulation days (default 30)
#   RANKS   MPI ranks (default 12)
#   DT      outer timestep [s] (default 6.0)
#   NX,NY   grid dims (default 132)
#   PYBIN   python interpreter (default .venv/bin/python)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

OUTPUT="${1:-results/rce_30day}"
DAYS="${DAYS:-30}"
RANKS="${RANKS:-12}"
DT="${DT:-6.0}"
NX="${NX:-132}"
NY="${NY:-132}"
PYBIN="${PYBIN:-.venv/bin/python}"

mkdir -p "$OUTPUT"
LOGFILE="$OUTPUT/mpi.stdout.log"

echo "RCE run: $DAYS days, ${NY}x${NX} grid, $RANKS ranks, dt=$DT s"
echo "Output:  $OUTPUT"
echo "Logs:    $LOGFILE"

# JAX_PLATFORMS=cpu forces CPU (default in driver). Set to metal for GPU/Metal.
exec mpirun -np "$RANKS" "$PYBIN" \
    "$REPO_ROOT/scripts/run_rce_mpi_long.py" \
    --nx "$NX" --ny "$NY" \
    --days "$DAYS" --dt "$DT" \
    --snapshot-hours 24.0 \
    --snapshot-3d-hours 1.0 \
    --profile-days 5.0 \
    --log-every-steps 100 \
    --output "$OUTPUT" \
    2>&1 | tee "$LOGFILE"
