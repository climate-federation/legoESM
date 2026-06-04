#!/usr/bin/env bash
# End-to-end smoke test of the JRA55-do RYF pipeline on real data.
#
# Pre-req:
#   - WOA18 in data/woa18/  (run scripts/data/download_omip_data.sh)
#   - JRA55-do IAF 1990+1991 in data/jra55_iaf/
#       (run scripts/data/download_omip_data.sh --jra55-only)
#
# Stages (each idempotent — skipped if its output already exists):
#   1. make_ryf.py  → data/jra55_ryf/RYF9091.zarr
#   2. prepare_omip_forcing.py → data/jra55_ryf_cache/<...>.zarr
#   3. run_omip.py 1-day smoke at 1° → results/jra55_ryf_smoke_1day/
#   4. plot_jra55_tropical_progress.py → 4 PNGs in the run dir
#
# Override the smoke length / resolution via env vars:
#   SMOKE_DAYS=7 SMOKE_RES=180x360 SMOKE_NLEV=20 ./scripts/run_ryf_smoke.sh
#
# Laptop-friendly defaults (1 day, 1° lat-lon, 20 levels) keep the
# integration to ~30 min on a modern CPU; bump to 50 levels and 30 days
# once the pipeline is moved to a GPU box.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="$REPO_ROOT/data"
RESULTS_DIR="$REPO_ROOT/results"

IAF_DIR="$DATA_DIR/jra55_iaf"
RYF_PATH="$DATA_DIR/jra55_ryf/RYF9091.zarr"
CACHE_DIR="$DATA_DIR/jra55_ryf_cache"
CACHE_NAME="jra55_do_v14_omip2_1deg_noleap.zarr"
CACHE_PATH="$CACHE_DIR/$CACHE_NAME"
WOA_T="$DATA_DIR/woa18/woa18_decav_t00_01.nc"
WOA_S="$DATA_DIR/woa18/woa18_decav_s00_01.nc"

SMOKE_RES="${SMOKE_RES:-180x360}"
SMOKE_NLEV="${SMOKE_NLEV:-20}"
SMOKE_DAYS="${SMOKE_DAYS:-1}"
SMOKE_DT="${SMOKE_DT:-300}"
SMOKE_OUT="${SMOKE_OUT:-$RESULTS_DIR/jra55_ryf_smoke_${SMOKE_DAYS}day}"
RUN_DIR="$SMOKE_OUT/latlon/$SMOKE_RES"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

echo "=========================================="
echo "JRA55-do RYF smoke test pipeline"
echo "  IAF:     $IAF_DIR"
echo "  WOA18:   $WOA_T"
echo "           $WOA_S"
echo "  resolution: $SMOKE_RES   nlev: $SMOKE_NLEV   dt: ${SMOKE_DT}s"
echo "  duration:   $SMOKE_DAYS day(s)"
echo "  output:     $SMOKE_OUT"
echo "=========================================="

if [[ ! -d "$IAF_DIR" ]]; then
    echo "ERROR: IAF dir not found: $IAF_DIR" >&2
    echo "  Run scripts/data/download_omip_data.sh --jra55-only first." >&2
    exit 1
fi

n_iaf=$(find "$IAF_DIR" -name "*.nc" 2>/dev/null | wc -l | tr -d ' ')
if [[ "$n_iaf" -lt 20 ]]; then
    echo "ERROR: IAF dir has only $n_iaf NetCDFs; need 20." >&2
    echo "  Wait for the download to finish or re-run the downloader." >&2
    exit 1
fi

if [[ ! -f "$WOA_T" || ! -f "$WOA_S" ]]; then
    echo "ERROR: WOA18 NetCDFs missing." >&2
    echo "  Run scripts/data/download_omip_data.sh first." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Stage 1: make_ryf
# ---------------------------------------------------------------------------

if [[ -e "$RYF_PATH" ]]; then
    echo ""
    echo "[1/4] RYF Zarr exists, skipping build: $RYF_PATH"
else
    echo ""
    echo "[1/4] Building RYF9091 (Stewart 2020 splice) ..."
    JAX_ENABLE_X64=1 python "$REPO_ROOT/scripts/data/make_ryf.py" \
        --iaf-dir "$IAF_DIR" \
        --year1 1990 --year2 1991 \
        --out "$RYF_PATH"
fi

# ---------------------------------------------------------------------------
# Stage 2: prepare_omip_forcing
# ---------------------------------------------------------------------------

if [[ -d "$CACHE_PATH" ]]; then
    echo ""
    echo "[2/4] 1° cache exists, skipping build: $CACHE_PATH"
else
    echo ""
    echo "[2/4] Building 1° regular cache from RYF Zarr ..."
    # Read the RYF's reference year from its time-axis units.  Per
    # Stewart 2020 / make_ryf.py, the RYF time axis is re-based to
    # 1900-01-01.  We pass --years 1900 1900 + --ref-year 1900 to
    # match.
    JAX_ENABLE_X64=1 python "$REPO_ROOT/scripts/data/prepare_omip_forcing.py" \
        --source "$RYF_PATH" \
        --years 1900 1900 \
        --target-resolution-deg 1.0 \
        --cache-dir "$CACHE_DIR" \
        --cache-filename "$CACHE_NAME" \
        --ref-year 1900
fi

# ---------------------------------------------------------------------------
# Stage 3: run_omip.py smoke
# ---------------------------------------------------------------------------

echo ""
echo "[3/4] Running ${SMOKE_DAYS}-day RYF smoke @ $SMOKE_RES, $SMOKE_NLEV levels, dt=${SMOKE_DT}s ..."
JAX_ENABLE_X64=1 python "$REPO_ROOT/scripts/run/run_omip.py" \
    --grid latlon \
    --resolution "$SMOKE_RES" \
    --nlev "$SMOKE_NLEV" \
    --dt "$SMOKE_DT" \
    --days "$SMOKE_DAYS" \
    --forcing-mode jra55_do_tropical \
    --jra55-cache "$CACHE_PATH" \
    --jra55-cycle \
    --woa-t "$WOA_T" \
    --woa-s "$WOA_S" \
    --output "$SMOKE_OUT" \
    --checkpoint-days 0.5

# ---------------------------------------------------------------------------
# Stage 4: plot
# ---------------------------------------------------------------------------

echo ""
echo "[4/4] Generating progress plots ..."
if [[ -d "$RUN_DIR" ]]; then
    JAX_ENABLE_X64=1 python "$REPO_ROOT/scripts/plot/plot_jra55_tropical_progress.py" \
        --run-dir "$RUN_DIR" || echo "  (plotter exited non-zero — check for missing matplotlib?)"
else
    echo "  WARNING: run dir not found: $RUN_DIR"
fi

echo ""
echo "=========================================="
echo "Smoke test complete."
echo ""
echo "Inspect:"
echo "  $RUN_DIR/timeseries.csv                  — scalar diagnostics"
echo "  $RUN_DIR/results.json                    — run metadata"
echo "  $RUN_DIR/timeseries_progress.png         — scalar evolution"
echo "  $RUN_DIR/snapshots_progress.png          — η, SST, surface speed"
echo "  $RUN_DIR/moc_progress.png                — meridional overturning"
echo "  $RUN_DIR/barotropic_streamfunction_progress.png — wind-driven gyres"
echo "=========================================="
