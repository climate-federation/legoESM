#!/usr/bin/env bash
# 30-day RCE @ 132x132, dx=2 km, 12 MPI ranks, full physics.
#
# Default outer dt: 5.0 s. F10 finding (2026-05 iter-9): bare-dycore
# stable at dt up to 10 s with the clean F8 IC (no bubble, no qv
# noise). Full-physics smoke at dt=5 s ran 864 steps stably. This
# replaces the iter-2 conservative 1 s default that was set based
# on the bubble-driven F1 instability — fixed by F7 (bubble made
# opt-in).
# Hourly 3D MSE/qv/T snapshots for GIF, daily surface snapshots, 5-day profiles.
#
# Stability history (measured by scripts/diag_bare_dycore_stability.py
# at nx=ny=48, nlev=30, dx=2 km, H=33 km, --semi-implicit-acoustic):
# F1 (iter-1, with bubble IC):
#   dt=1.0 s -> stable; dt=1.5 s -> growing; dt=2.0 s -> blows up step 70
# F7 (iter-2) found the bubble IC was the destabiliser, not dt itself.
# F8/F10 (iter-2/9) verified: clean Wing IC (no bubble, no qv noise) is
# stable at dt up to 10 s; full-physics smoke at dt=5 s ran 864 steps
# stably. iter-14 measured 132x132 1-sim-hour PASS at dt=5 s + N_ACOUSTIC=12
# (max|w|=6.1e-3 m/s); iter-38 locks this as a structural regression.
# R9 (Klemp-Wilhelmson 1978 outer-step implicit buoyancy) no longer on
# critical path: F10 lifted the dt constraint without it.
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
#   DT           outer timestep [s] (default 5.0 — F10 production stable)
#   NX,NY        grid dims (default 132)
#   N_ACOUSTIC   acoustic substeps per outer step (default 12, matches dt=5.0s
#                + iter-14 production measurement)
#   ADVECTION    upwind1 | weno5  (default upwind1)
#   NO_MASS_FIXER  1 = pass --no-mass-fixer to driver (iter-95b RCE-spinup
#                fix; default 1 for this 30-day RCE wrapper). 0 keeps
#                the legacy fix_moist_mass_plane ON (use only for
#                gravity-wave / hydrostatic smokes — not 30-day RCE).
#   PYBIN        python interpreter (default .venv/bin/python)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

OUTPUT="${1:-results/rce_30day}"
DAYS="${DAYS:-30}"
RANKS="${RANKS:-12}"
DT="${DT:-5.0}"
NX="${NX:-132}"
NY="${NY:-132}"
N_ACOUSTIC="${N_ACOUSTIC:-12}"
ADVECTION="${ADVECTION:-upwind1}"
HYPERDIFF="${HYPERDIFF:-5.0e6}"
BUBBLE_K="${BUBBLE_K:-0.0}"
QV_NOISE="${QV_NOISE:-0.0}"
USE_DD="${USE_DD:-0}"
NO_MASS_FIXER="${NO_MASS_FIXER:-1}"
PYBIN="${PYBIN:-.venv/bin/python}"

# USE_DD=1 switches the driver from the legacy rank-0-dycore-broadcast
# pattern to true per-rank domain decomposition via step_halo + the R7
# MPI mass fixer. Required for any real MPI scaling claim. Verify on
# DAYS=0.05 + RANKS=2 before committing to a multi-day production run.
DD_FLAG=""
if [ "$USE_DD" = "1" ]; then
    DD_FLAG="--use-dd"
fi

# NO_MASS_FIXER=1 (default for this 30-day RCE wrapper) passes
# --no-mass-fixer to the driver. iter-95b found fix_moist_mass_plane
# rescales total water back to IC every outer step, killing RCE
# spinup because surface flux must NET ADD moisture until precip
# balances at equilibrium. Set NO_MASS_FIXER=0 to keep the legacy
# fixer ON (appropriate for gravity-wave / hydrostatic smokes where
# total water IS conserved; not appropriate for 30-day RCE).
NMF_FLAG=""
if [ "$NO_MASS_FIXER" = "1" ]; then
    NMF_FLAG="--no-mass-fixer"
fi

mkdir -p "$OUTPUT"
LOGFILE="$OUTPUT/mpi.stdout.log"

echo "RCE run: $DAYS days, ${NY}x${NX} grid, $RANKS ranks, dt=$DT s"
echo "Output:  $OUTPUT"
echo "Logs:    $LOGFILE"

# JAX_PLATFORMS=cpu forces CPU (default in driver). Set to metal for GPU/Metal.
# iter-99: dropped the leading ``exec`` so the post-run trajectory
# summarizer below actually executes (``exec`` replaces the shell
# process with mpirun and skips every later line). ``set -o
# pipefail`` (already enabled above) preserves mpirun's exit status
# through the ``| tee`` pipe.
mpirun -np "$RANKS" "$PYBIN" \
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
    $DD_FLAG \
    $NMF_FLAG \
    --snapshot-hours 24.0 \
    --snapshot-3d-hours 1.0 \
    --profile-days 5.0 \
    --log-every-steps 100 \
    --output "$OUTPUT" \
    2>&1 | tee "$LOGFILE"

# iter-99: per-day RCE trajectory summary as a post-run artefact.
# Reads <OUTPUT>/snapshots/snap_day_*.npz + <OUTPUT>/profiles/prof_day_*.npz
# and writes <OUTPUT>/trajectory.csv + a one-line console table per
# day. Pure numpy; no JAX / MPI requirement (runs on the rank-0
# wrapper host after mpirun exits).
#
# Exit status (iter-99 Codex MEDIUM#1 fix): a successful mpirun + a
# failed summarizer used to mask each other out, so callers could
# not rely on wrapper exit 0 to mean ``trajectory.csv exists``. Now:
#   * mpirun nonzero  → wrapper exits nonzero (set -e + pipefail).
#   * mpirun ok + summarizer nonzero → wrapper exits 1 by default.
#     Set ``ALLOW_SUMMARY_FAILURE=1`` to downgrade to a warning (e.g.
#     for runs aborted before any snapshot landed, where the
#     summarizer's FileNotFoundError is expected).
ALLOW_SUMMARY_FAILURE="${ALLOW_SUMMARY_FAILURE:-0}"
echo "Computing per-day RCE trajectory summary..."
set +e
"$PYBIN" "$REPO_ROOT/scripts/summarize_rce_trajectory.py" "$OUTPUT" \
    > "$OUTPUT/trajectory.txt" 2>&1
summary_status=$?
set -e
if [ "$summary_status" -eq 0 ]; then
    echo "Wrote $OUTPUT/trajectory.csv"
elif [ "$ALLOW_SUMMARY_FAILURE" = "1" ]; then
    echo "WARN: summarize_rce_trajectory.py failed (status $summary_status); see $OUTPUT/trajectory.txt" >&2
    echo "WARN: ALLOW_SUMMARY_FAILURE=1 — wrapper exiting 0 anyway." >&2
else
    echo "ERROR: summarize_rce_trajectory.py failed (status $summary_status); see $OUTPUT/trajectory.txt" >&2
    echo "Set ALLOW_SUMMARY_FAILURE=1 to downgrade to a warning." >&2
    exit "$summary_status"
fi
