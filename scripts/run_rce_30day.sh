#!/usr/bin/env bash
# 30-day RCE @ 132x132, dx=2 km, 12 MPI ranks, full physics.
# Hourly 3D MSE/qv/T snapshots for GIF, daily surface snapshots, 5-day profiles.
#
# Stability constraints (measured by scripts/diag_bare_dycore_stability.py
# at nx=ny=48, nlev=30, dx=2 km, H=33 km, --semi-implicit-acoustic):
#   dt=0.5 s -> stable (max|w| ~ 7e-3 m/s through 200 steps)
#   dt=1.0 s -> stable
#   dt=1.5 s -> growing instability (max|w| ~ 4 m/s by step 100)
#   dt=2.0 s -> blows up by step 70 (max|w| > 100 m/s)
# Conclusion: outer dt must satisfy dt <= ~1.0 s with the Wing 2018 IC
# (warm bubble at z<1 km) until the buoyancy/w mode amplification at
# the SSP-RK3 outer step is fixed (e.g. by porting Klemp-Wilhelmson
# 1978 implicit-buoyancy to the OUTER step, not just the acoustic
# substep — see CRM_implementation.md).
#
# Usage:
#   scripts/run_rce_30day.sh                  # default output dir results/rce_30day
#   scripts/run_rce_30day.sh results/myrun    # custom output dir
#   DAYS=1 scripts/run_rce_30day.sh results/rce_smoke   # short smoke run
#   RANKS=4 scripts/run_rce_30day.sh ...      # custom rank count
#
# Env vars:
#   DAYS         simulation days (default 30)
#   RANKS        MPI ranks (default 12)
#   DT           outer timestep [s] (default 1.0 — stable at production scale)
#   NX,NY        grid dims (default 132)
#   N_ACOUSTIC   acoustic substeps per outer step (default 24, matches dt=1.0s)
#   ADVECTION    upwind1 | weno5  (default upwind1)
#   PYBIN        python interpreter (default .venv/bin/python)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

OUTPUT="${1:-results/rce_30day}"
DAYS="${DAYS:-30}"
RANKS="${RANKS:-12}"
DT="${DT:-1.0}"
NX="${NX:-132}"
NY="${NY:-132}"
N_ACOUSTIC="${N_ACOUSTIC:-24}"
ADVECTION="${ADVECTION:-upwind1}"
HYPERDIFF="${HYPERDIFF:-5.0e6}"
BUBBLE_K="${BUBBLE_K:-0.0}"
QV_NOISE="${QV_NOISE:-0.0}"
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
    --semi-implicit-acoustic \
    --acoustic-off-centering 0.1 \
    --n-acoustic-substeps "$N_ACOUSTIC" \
    --advection "$ADVECTION" \
    --hyperdiff "$HYPERDIFF" \
    --bubble-theta-pert "$BUBBLE_K" \
    --qv-noise-amp "$QV_NOISE" \
    --snapshot-hours 24.0 \
    --snapshot-3d-hours 1.0 \
    --profile-days 5.0 \
    --log-every-steps 100 \
    --output "$OUTPUT" \
    2>&1 | tee "$LOGFILE"
