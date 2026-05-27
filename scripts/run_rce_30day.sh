#!/usr/bin/env bash
# 30-day RCE @ 132x132, dx=2 km, 12 MPI ranks, full physics.
#
# Default outer dt: 20.0 s (iter-183 production refresh — 4x faster
# than the iter-14 dt=5 s contract). iter-179 added Van Leer TVD
# horizontal advection; iter-180/183 measured Van Leer at dt=20 +
# beta=0.2 acoustic off-centering gives 3x wall-time speedup vs the
# iter-14 dt=10 + upwind1 baseline at the same sim time. F10
# verified the bare dycore is stable at dt up to 10 s with the
# clean F8 IC; iter-180/183 pushed dt up to 20 s by pairing with
# WENO5 / Van Leer (less numerical diffusion -> looser advective
# CFL) and stronger acoustic off-centering (beta=0.2 vs iter-14's
# 0.0/0.1).
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
#   DT           outer timestep [s] (default 20.0 — iter-183 production
#                contract pairing with ADVECTION=van_leer + beta=0.2)
#   NX,NY        grid dims (default 132)
#   N_ACOUSTIC   acoustic substeps per outer step (default 12). With
#                the iter-183 dt=20 s outer step the inner acoustic
#                dt = 20/12 = 1.67 s and the SI substep runs at
#                Ca_substep = 0.87 (well under the SI-relaxed acoustic
#                CFL bound). iter-14 measured the same N_ACOUSTIC=12
#                paired with dt=5 s at Ca_substep ~ 0.22; iter-183
#                kept N_ACOUSTIC fixed and lifted dt because Van Leer
#                + stronger beta absorb the extra inner stiffness.
#   ADVECTION    upwind1 | van_leer | weno5  (default van_leer — iter-183
#                production: 2nd-order TVD, stencil 4, 3x wall-time
#                speedup vs upwind1 at dt=10. upwind1 = legacy; weno5
#                = sharp-front opt-in but ~3x per-step cost so SLOWER
#                end-to-end than upwind1 at dt=10).
#   HYPERDIFF    horizontal hyperdiffusion coefficient (default 5.0e6;
#                iter-9 F6 found weaker values let bubble IC blow up).
#   BUBBLE_K     warm-bubble IC perturbation [K] (default 0.0 = no
#                bubble per F7/F10 — clean Wing IC). Non-zero only
#                for bare-dycore stability probing.
#   QV_NOISE     qv noise IC amplitude [kg/kg] (default 0.0 per
#                iter-181 F11: any qv noise > 0 + gray radiation
#                triggers an exponential instability that blows up
#                in 25-50 outer steps. The amplitude does not
#                matter (1e-8, 1e-6, 1e-4 all blow up) so this is
#                a structural radiation-feedback bug not a stiffness
#                issue. iter-181 added --theta-noise-amp as a
#                proposed alternative symmetry-breaker, but the
#                same blowup pattern repeats at any nonzero theta'
#                noise. F11 needs separate physics-side debugging
#                (likely radiation+dycore coupling, not the dycore
#                proper). Default 0.0 keeps the wrapper functional
#                at the iter-183 dt=20+van_leer+beta=0.2 production
#                contract; convection remains column-symmetric
#                (iter-149) so realistic precipitation is gated on
#                F11 resolution.
#   USE_DD       1 = pass --use-dd to driver (true per-rank domain
#                decomposition via step_halo + R7 MPI mass fixer).
#                Default 0 keeps the legacy rank-0-broadcast pattern
#                (F8-stable). Required for any real MPI scaling claim.
#   NO_MASS_FIXER  1 = pass --no-mass-fixer to driver (iter-95b RCE-spinup
#                fix; default 1 for this 30-day RCE wrapper). 0 keeps
#                the legacy fix_moist_mass_plane ON (use only for
#                gravity-wave / hydrostatic smokes — not 30-day RCE).
#   EVALUATE_DOD 0 (default) = skip post-run grading.
#                1            = pass --evaluate (10-day SPINUP gate,
#                               5 % MSE drift). Use on DAYS>=10
#                               smokes / mid-run health checks.
#                stability    = pass --evaluate --no-plateau-check.
#                               Skips the plateau / MSE-drift gates
#                               and folds INSUFFICIENT into PASS;
#                               only stability (finite + max|U|_sfc
#                               + stuck) checks fire. Use for
#                               in-flight progress monitoring (e.g.
#                               iter-105 day 5 of 30). iter-128.
#                final        = pass --final-dod (30-day PRODUCTION
#                               gate, 1 % MSE drift, requires
#                               >=30 days of snapshots). Use on
#                               the canonical 30-day production
#                               run — iter-112 contract.
#                FAIL / INSUFFICIENT propagate as non-zero wrapper
#                exit unless ALLOW_SUMMARY_FAILURE=1.
#   ALLOW_SUMMARY_FAILURE  1 = downgrade a non-zero summarizer exit to
#                a WARN on stderr (wrapper still exits 0). Default 0
#                propagates the failure as the wrapper's exit code.
#                Set to 1 only for runs aborted before any snapshot
#                landed (where the FileNotFoundError is expected).
#   EMIT_TRAJECTORY_PNG  Trajectory PNG rendering mode:
#                0      (default) — skip the PNG step.
#                1      — best-effort. Render via scripts/plot_rce_log.py;
#                         WARN on failure but DO NOT change wrapper exit.
#                strict — render + propagate failure as the wrapper's
#                         exit code. Use in CI gates that REQUIRE the
#                         PNG artefact.
#                matplotlib + Agg is needed for the PNG step.
#   CHECK_LOG_MAX_W  1 = thread --check-log-max-w to the summarizer
#                so the run-wide 3D max|w| from log.txt is gated
#                against DOD criterion 1 (50 m/s). Default 0 keeps
#                short smokes ungated. iter-137 contract.
#   PYBIN        python interpreter (default .venv/bin/python)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# iter-136: --help / -h prints the docstring header (everything
# above ``set -e``) and exits 0. Without this, ``run_rce_30day.sh
# --help`` would be interpreted as ``OUTPUT=--help`` and write
# garbage to the literal path. The header captures every env var
# + the typical invocation patterns.
if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    # macOS BSD sed does not support ``\?``; use ``-E`` (extended
    # regex) so ``# ?`` strips the leading ``# `` (with optional
    # trailing space).
    # iter-143/iter-144/iter-179/iter-181/iter-186: extracted range
    # widened to cover every documented env var. iter-136 stopped at
    # line 72 (pre-iter-138 CHECK_LOG_MAX_W docs); iter-143 widened
    # to 77 (post-iter-138); iter-144 added HYPERDIFF/BUBBLE_K/
    # QV_NOISE/USE_DD docs + PYBIN moved to 89. iter-179 expanded
    # the QV_NOISE docblock (column-symmetry-trap rationale) which
    # pushed CHECK_LOG_MAX_W + PYBIN to 93-97. iter-181 further
    # expanded the QV_NOISE docblock (F11 radiation-feedback
    # finding) so CHECK_LOG_MAX_W is now at 98-101 and PYBIN at 102.
    # iter-186 (Codex LOW) refreshed the iter-183 production
    # contract docs at the top (DT=20, ADVECTION=van_leer, beta=0.2);
    # CHECK_LOG_MAX_W is now at 107-110 and PYBIN at 111. iter-190
    # expanded the N_ACOUSTIC env-var docblock with the
    # Ca_substep=0.87 measurement, pushing CHECK_LOG_MAX_W to
    # 113-116 and PYBIN to 117 — range widened to 117 in lockstep.
    sed -n '2,117p' "$0" | sed -E 's/^# ?//'
    exit 0
fi

OUTPUT="${1:-results/rce_30day}"
DAYS="${DAYS:-30}"
RANKS="${RANKS:-12}"
DT="${DT:-20.0}"
NX="${NX:-132}"
NY="${NY:-132}"
N_ACOUSTIC="${N_ACOUSTIC:-12}"
ADVECTION="${ADVECTION:-van_leer}"
HYPERDIFF="${HYPERDIFF:-5.0e6}"
BUBBLE_K="${BUBBLE_K:-0.0}"
QV_NOISE="${QV_NOISE:-0.0}"
USE_DD="${USE_DD:-0}"
NO_MASS_FIXER="${NO_MASS_FIXER:-1}"
EVALUATE_DOD="${EVALUATE_DOD:-0}"
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
    --acoustic-off-centering 0.2 \
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
# iter-103 / iter-113: EVALUATE_DOD threads either ``--evaluate``
# (spinup 5 % gate) or ``--final-dod`` (production 1 % gate +
# >=30 days) into the post-run summarizer. Three valid values:
#   0     (default) — skip the grade step.
#   1     — spinup gate. 10-day window, 5 % MSE drift tolerance.
#   final — 30-day production gate (iter-112). 10-day window,
#           1 % MSE drift tolerance, requires >=30 days of snapshots.
# Anything else falls through to no flag (also the default). The
# summarizer's ``--evaluate`` and ``--final-dod`` are mutually
# exclusive (driver argparse enforces); the case below picks at
# most one.
EVAL_FLAG=""
case "$EVALUATE_DOD" in
    0)         EVAL_FLAG="" ;;
    1)         EVAL_FLAG="--evaluate" ;;
    stability) EVAL_FLAG="--evaluate --no-plateau-check" ;;
    final)     EVAL_FLAG="--final-dod" ;;
    # iter-114 Codex MEDIUM#4: a typo like ``Final`` or ``spinup``
    # silently disabled grading pre-iter-114. Refuse instead.
    *)
        echo "ERROR: EVALUATE_DOD='$EVALUATE_DOD' not recognised." >&2
        echo "  Valid values: 0 (default, no grading), 1 (spinup gate)," >&2
        echo "                stability (spinup gate without plateau check," >&2
        echo "                            iter-128), final (30-day production gate)." >&2
        exit 1
        ;;
esac
# iter-138: thread --check-log-max-w to the summarizer when
# CHECK_LOG_MAX_W=1. Independent of the EVALUATE_DOD case (both
# gates are additive — criterion 1 vs criterion 2).
CHECK_LOG_MAX_W="${CHECK_LOG_MAX_W:-0}"
if [ "$CHECK_LOG_MAX_W" = "1" ]; then
    EVAL_FLAG="$EVAL_FLAG --check-log-max-w"
fi
echo "Computing per-day RCE trajectory summary..."
set +e
"$PYBIN" "$REPO_ROOT/scripts/summarize_rce_trajectory.py" "$OUTPUT" \
    $EVAL_FLAG \
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

# iter-124 / iter-126: if the user ran a >=30-day production run
# without EVALUATE_DOD=final, the wrapper landed trajectory.csv but
# did NOT grade it against the production DOD criterion 2 (1 % MSE
# drift, Wing 2018 plateau range). Print a hint so the manual
# follow-up is obvious.
#
# iter-126 Codex MEDIUM#1: condition is now ``!= "final"`` (was
# ``= "0"``). EVALUATE_DOD=1 (spinup 5 % gate) ALSO benefits from
# the hint because spinup-gate PASS != production-DOD PASS — the
# 1 % gate is strictly tighter.
#
# DAYS edge cases: bash arithmetic ``[ "${DAYS%.*}" -ge 30 ]`` with
# 2>/dev/null silently skips the hint on non-numeric / empty /
# negative DAYS (test cases covered in iter-126). ``${DAYS%.*}``
# strips trailing decimals so 30.5 floors to 30 (hint fires);
# 29.99 floors to 29 (skipped).
if [ "$EVALUATE_DOD" != "final" ] && [ "${DAYS%.*}" -ge 30 ] 2>/dev/null; then
    echo ""
    echo "Hint: this is a >=30-day production run with EVALUATE_DOD=$EVALUATE_DOD."
    echo "  To grade the trajectory against the final DOD criterion 2"
    echo "  (Wing 2018 plateau, 1 % MSE drift over last 10 days), run:"
    echo "    $PYBIN scripts/summarize_rce_trajectory.py \\"
    echo "      $OUTPUT --final-dod --check-log-max-w"
    echo "  --check-log-max-w covers criterion 1 (max|w| < 50 m/s)"
    echo "  and requires $OUTPUT/log.txt to be present; drop it if the"
    echo "  log was rotated / lost (criterion 1 then UNCHECKED)."
    echo "  Or re-launch this wrapper with EVALUATE_DOD=final and"
    echo "  CHECK_LOG_MAX_W=1 to gate on BOTH DOD criteria (1 + 2)"
    echo "  (non-zero exit on FAIL)."
fi

# iter-125 / iter-126: optional trajectory PNG via plot_rce_log.py.
# Three modes: 0 = skip, 1 = best-effort (WARN-on-fail), strict =
# propagate-on-fail (Codex iter-125 MEDIUM#2 — CI gates that REQUIRE
# the PNG artefact can opt into propagation).
# iter-177: reject typo / unrecognised values explicitly (mirroring
# the iter-114 EVALUATE_DOD typo-rejection). Pre-iter-177 a value
# like ``Strict`` or ``yes`` silently fell through to the implicit
# "skip" branch — a user expecting strict propagation got a silent
# no-PNG instead, with no error to surface the mismatch.
EMIT_TRAJECTORY_PNG="${EMIT_TRAJECTORY_PNG:-0}"
case "$EMIT_TRAJECTORY_PNG" in
    0|1|strict) ;;
    *)
        echo "ERROR: EMIT_TRAJECTORY_PNG='$EMIT_TRAJECTORY_PNG' not recognised." >&2
        echo "  Valid values: 0 (default, skip), 1 (best-effort," >&2
        echo "                WARN-on-fail), strict (propagate-on-fail)." >&2
        exit 1
        ;;
esac
if [ "$EMIT_TRAJECTORY_PNG" = "1" ] || [ "$EMIT_TRAJECTORY_PNG" = "strict" ]; then
    echo "Rendering trajectory PNG (mode=$EMIT_TRAJECTORY_PNG)..."
    set +e
    "$PYBIN" "$REPO_ROOT/scripts/plot_rce_log.py" "$OUTPUT" \
        > "$OUTPUT/trajectory_plot.log" 2>&1
    plot_status=$?
    set -e
    if [ "$plot_status" -eq 0 ]; then
        echo "Wrote $OUTPUT/trajectory.png"
    elif [ "$EMIT_TRAJECTORY_PNG" = "strict" ]; then
        echo "ERROR: plot_rce_log.py failed (status $plot_status); see $OUTPUT/trajectory_plot.log" >&2
        echo "ERROR: EMIT_TRAJECTORY_PNG=strict — propagating exit code." >&2
        exit "$plot_status"
    else
        echo "WARN: plot_rce_log.py failed (status $plot_status); see $OUTPUT/trajectory_plot.log" >&2
        echo "WARN: trajectory.csv still emitted; PNG is best-effort." >&2
        echo "WARN: set EMIT_TRAJECTORY_PNG=strict to propagate this failure." >&2
    fi
fi
