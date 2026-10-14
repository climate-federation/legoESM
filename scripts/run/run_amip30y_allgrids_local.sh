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
OUT_BASE="${OUT_BASE:-$REPO/results/amip30y}"
START_YEAR=1979
END_YEAR=2009
DAYS=10950           # 30 years
RAD_STEPS="${RAD_STEPS:-18}"  # 18 = 3-hourly at dt=600 (production choice
                              # 2026-06-10); 6 = 1-hourly for fidelity runs
# Initial condition: era5 (realistic winds + moisture) is the production
# default and needs an ERA5 Zarr store / GCS URI in ERA5_IC_PATH.  Set
# IC=default to run the (less realistic) uniform-IC cold start instead.
IC="${IC:-era5}"
# ERA5 IC source: defaults to the PUBLIC ARCO ERA5 store on GCS (no
# credentials).  Override with a local zarr via ERA5_IC_PATH.
ERA5_IC_PATH="${ERA5_IC_PATH:-gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3}"
# Real prescribed SST: point SST_FILE at an input4MIPs AMIP II bcs file
# (tosbcs K, siconcbcs percent) for a FAITHFUL run.  Empty -> the
# synthetic deck SST (pipeline-valid, not observed) is used instead.
SST_FILE="${SST_FILE:-}"
if [[ -z "$SST_FILE" ]]; then
  echo "[30y] NOTE: SST_FILE unset -> using SYNTHETIC deck SST (not the" >&2
  echo "[30y]   observed input4MIPs AMIP II bcs). For a faithful run, run" >&2
  echo "[30y]   scripts/data/stage_amip_realdata.py --print-esgf and set" >&2
  echo "[30y]   export SST_FILE=<tosbcs input4MIPs file>." >&2
fi

export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1

# grid_type discretization resolution extra...
CASES=(
  "cubed_sphere finite_volume 36"
  "latlon       finite_volume 72"
  "gaussian     spectral      47"
  "voronoi      mpas          5  --turbulence none"
)

for case in "${CASES[@]}"; do
  read -r GRID DISC RES EXTRA <<<"$case"
  OUT="$OUT_BASE/${GRID}_${DISC}_n${RES}"
  mkdir -p "$OUT"

  # Resume support: find the latest checkpoint day, run the remainder.
  RESTART_ARGS=()
  LATEST=$(ls "$OUT"/checkpoint_day_*.npz 2>/dev/null \
             | sed 's/.*checkpoint_day_0*\([0-9]*\).npz/\1/' \
             | sort -n | tail -1 || true)
  if [[ -n "${LATEST:-}" ]]; then
    if (( LATEST >= DAYS )); then
      echo "[30y] $GRID/$DISC already complete (day $LATEST) — skip"
      continue
    fi
    echo "[30y] $GRID/$DISC resuming from day $LATEST"
    RESTART_ARGS=(--extra --restart-from "$OUT/checkpoint_day_$(printf '%04d' "$LATEST").npz")
  fi

  echo "[30y] launching $GRID/$DISC n=$RES -> $OUT"
  # --dt-auto picks each grid's ladder-validated stable timestep
  # (C36->150 s, latlon72->75 s, T47->150 s, voronoi->300 s) so a long
  # run cannot blow up at the over-large default dt mid-chain.
  # IC: era5 on cube/latlon/gaussian; the deck auto-falls voronoi/mpas
  # back to --ic default (era5_to_mpas_carry not yet wired).
  IC_ARGS=(--ic "$IC")
  if [[ "$IC" == "era5" ]]; then
    IC_ARGS+=(--ic-path "$ERA5_IC_PATH")
  fi
  # Real observed SST overrides the synthetic deck SST when provided.
  if [[ -n "$SST_FILE" ]]; then
    IC_ARGS+=(--sst-file "$SST_FILE")
  fi
  "$PY" "$DECK" \
    --forcing-dir "$FORCING" --auto-generate \
    --start-year $START_YEAR --end-year $END_YEAR \
    --grid-type "$GRID" --discretization "$DISC" \
    --resolution "$RES" --nlev 30 \
    --days $DAYS --dt-auto \
    --rad-update-steps "$RAD_STEPS" \
    --diag-days 30 --checkpoint-days 365 \
    --radiation rrtmg \
    "${IC_ARGS[@]}" \
    --output "$OUT" \
    ${EXTRA:+--extra $EXTRA} \
    "${RESTART_ARGS[@]}" \
    2>&1 | tee -a "$OUT/run.log"

  "$PY" "$REPO/scripts/validate/validate_amip_run.py" "$OUT" \
    2>&1 | tee -a "$OUT/validate.log" || true
done
echo "[30y] queue complete"
