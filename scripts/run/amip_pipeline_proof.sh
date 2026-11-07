#!/usr/bin/env bash
# 30-year 2.5-degree AMIP CMIP runs on a single local GPU, all four grids.
#
# Sequential queue (one GPU): cubed-sphere -> lat-lon -> gaussian -> voronoi.
# Every run is CHECKPOINTED yearly (--checkpoint-days 365) and this script
# resumes from the latest checkpoint on restart, so it is safe to interrupt
# and re-invoke (cron/systemd/manual).  Move to SLURM by wrapping each
# invocation in an sbatch chain (see scripts/cluster/ for templates).
#
# Physics: production deck defaults — RRTMG radiation, Morrison
# double-moment microphysics (M2005/MG), SBM convection, Louis PBL,
# Sundqvist cloud fraction, aerosol-CCN (Andreae 2009) + dynamic zenith
# ocean albedo on the pipeline grids.  Full CMIP6 forcing deck
# (SST/SIC, transient GHG, ozone, solar, aerosol, volcanic), 1979-2009.
#
# Throughput (RTX 5090 laptop, fp32, 1-h radiation): ~0.6 h per
# simulated day at C36 => a single 30-year chain is a MULTI-MONTH
# wall-clock job locally.  Use --rad-update-steps 18 (3-h radiation)
# for ~2-3x, or run on a cluster.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$REPO/.venv/bin/python"
DECK="$REPO/scripts/run/run_amip_smoke_deck.py"
FORCING="${FORCING_DIR:-$REPO/forcing_amip}"
OUT_BASE="${OUT_BASE:-$REPO/results/amip_pipeline_proof}"
START_YEAR=1979
END_YEAR=1981
DAYS=30              # pipeline proof: full production stack + 1 checkpoint per grid
RAD_STEPS="${RAD_STEPS:-18}"  # 18 = 3-hourly at dt=600 (production choice
                              # 2026-06-10); 6 = 1-hourly for fidelity runs

export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1

# grid_type discretization resolution extra...
CASES=(
  "cubed_sphere finite_volume 36"
  "latlon       finite_volume 72"
  "gaussian     spectral      47"
  "voronoi      mpas          5  --turbulence none"
)

FAILED=()
for case in "${CASES[@]}"; do
  read -r GRID DISC RES REST <<<"$case"
  OUT="$OUT_BASE/${GRID}_${DISC}_n${RES}"
  mkdir -p "$OUT"

  # Pass-through args forwarded to run_amip via the deck's SINGLE --extra.  The
  # deck's --extra is argparse.REMAINDER, so it must appear exactly ONCE and
  # LAST: two --extra tokens make the first swallow the literal second and
  # forward a bogus "--extra" to run_amip (which rejects it).  Collect the case
  # extras (e.g. voronoi "--turbulence none") plus, on resume, the restart flag.
  PASS=()
  [[ -n "${REST:-}" ]] && PASS+=($REST)

  # Resume support: find the latest checkpoint day.
  LATEST=$(ls "$OUT"/checkpoint_day_*.npz 2>/dev/null \
             | sed 's/.*checkpoint_day_0*\([0-9]*\).npz/\1/' \
             | sort -n | tail -1 || true)
  if [[ -n "${LATEST:-}" ]]; then
    if (( LATEST >= DAYS )); then
      echo "[30y] $GRID/$DISC already complete (day $LATEST) — skip"
      continue
    fi
    echo "[30y] $GRID/$DISC resuming from day $LATEST"
    # NOTE: spectral/FV restart honors start_step (runs start_day -> DAYS, total
    # semantics).  MPAS/voronoi restart loops range(n_steps_total) ignoring
    # start_step (model_driver._run_mpas), so a resumed voronoi run does a FRESH
    # DAYS from the checkpoint and OVERSHOOTS to LATEST+DAYS instead of stopping
    # at DAYS.  For a proof that means it runs long, not short — acceptable and
    # flagged here; a true total-duration MPAS resume needs a driver-side fix.
    PASS+=(--restart-from "$OUT/checkpoint_day_$(printf '%04d' "$LATEST").npz")
  fi

  EXTRA_ARGS=()
  ((${#PASS[@]})) && EXTRA_ARGS=(--extra "${PASS[@]}")

  echo "[30y] launching $GRID/$DISC n=$RES -> $OUT"
  # --dt-auto picks each grid's ladder-validated stable timestep
  # (C36->150 s, latlon72->75 s, T47->150 s, voronoi->300 s) so a long
  # run cannot blow up at the over-large default dt mid-chain.
  "$PY" "$DECK" \
    --forcing-dir "$FORCING" --auto-generate \
    --start-year $START_YEAR --end-year $END_YEAR \
    --grid-type "$GRID" --discretization "$DISC" \
    --resolution "$RES" --nlev 30 \
    --days $DAYS --dt-auto \
    --rad-update-steps "$RAD_STEPS" \
    --diag-days 10 --checkpoint-days 15 \
    --radiation rrtmg \
    --output "$OUT" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a "$OUT/run.log"

  # Validation must FAIL LOUDLY: a numerically-complete but invalid run is a
  # failed proof, not a pass.  pipefail makes the pipe status the validator's;
  # `if !` keeps set -e from aborting so every grid is still attempted.
  if ! "$PY" "$REPO/scripts/validate/validate_amip_run.py" "$OUT" \
        2>&1 | tee -a "$OUT/validate.log"; then
    echo "[30y] VALIDATION FAILED for $GRID/$DISC (see $OUT/validate.log)" >&2
    FAILED+=("$GRID/$DISC")
  fi
done

if ((${#FAILED[@]})); then
  echo "[30y] queue finished with VALIDATION FAILURES: ${FAILED[*]}" >&2
  exit 1
fi
echo "[30y] queue complete"
