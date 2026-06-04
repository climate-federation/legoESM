#!/usr/bin/env bash
# Run a real AMIP simulation (with actual GHG / aerosol / ozone /
# solar forcing from CMIP6-compatible files) on each supported grid
# type and produce a cross-grid comparison via the atmosphere-matrix
# plotter.
#
# Iter-41: addresses the user's prompt item "hydrostatic AMIP with
# realistic GHG, aerosol and ozone forcing, on all grid types".
# This wrapper invokes ``scripts/run_amip.py`` (NOT the matrix
# runner's stub ``run_amip``, which is HS + optional RRTMGP without
# external forcing files) once per grid; output is laid out at
# ``$OUTPUT/hydrostatic/amip/<grid>/<resolution>/`` so the matrix
# runner's ``--cross-grid-plots-only`` path picks it up.
#
# Usage:
#     scripts/run_amip_cross_grid.sh OUTPUT [DAYS] [GHG_FILE] \
#         [OZONE_FILE] [AEROSOL_FILE]
#
# Defaults:
#     DAYS=30
#     GHG/ozone/aerosol files: empty (fall back to constant present-
#     day GHG, standard ozone climatology, off aerosol).  Pass a
#     CMIP6 input4MIPs file path to use real time-varying forcing.
#
# Examples:
#     # Quick smoke (constant forcing, 5 days):
#     scripts/run_amip_cross_grid.sh /tmp/amip_smoke 5
#
#     # Full realistic forcing (CMIP6 historical):
#     scripts/run_amip_cross_grid.sh /tmp/amip_full 30 \
#         data/forcings/ghg_historical.nc \
#         data/forcings/vmro3_input4MIPs_ozone-mmrConcentrations.nc \
#         data/forcings/aerosol_kinne_v2.nc

set -e

OUTPUT=${1:?usage: $0 OUTPUT [DAYS] [GHG_FILE] [OZONE_FILE] [AEROSOL_FILE]}
DAYS=${2:-30}
GHG_FILE=${3:-}
OZONE_FILE=${4:-}
AEROSOL_FILE=${5:-}
ANY_FAILED=0

# Grid configuration table (parallel-array form; macOS default
# Bash 3.2 lacks ``declare -A`` — same iter-7 / iter-32 fix as the
# RCE wrapper). Each row is
#   "<grid_type>:<resolution>:<discretization>:<output_folder>:<extra_flags>:<dt>"
# Empty trailing fields stay empty. iter-32 wires the dt column to
# the iter-13/iter-26 stability ladder (legoesm.driver.rce_dt.
# auto_dt_rce) so AMIP cross-grid at C48 picks dt=150 instead of
# the run_amip.py default of 600 (which is the iter-13-banned
# value that produced the C48 BLOWUP in RCE).
#
# iter-74: ``run_amip.py --resolution`` is ``type=int`` and means
# different things per grid:
#   cubed_sphere → n      (C48 → 48 cells per face per face dim)
#   latlon       → n_lat  (run_amip.py sets n_lon = 2 * n_lat)
#   voronoi      → MPAS level
#   gaussian     → ignored (use --truncation instead)
GRID_TABLE=(
    "cubed_sphere:48:cdgrid:cubed_sphere::150"
    "latlon:90:latlon_cgrid:latlon::"
    "voronoi:6:mpas:icosahedral::60"
    "gaussian:42:spectral:spectral:--truncation 42:150"
)
# Voronoi dt notes (iter-32 Codex HIGH):
#   scripts/validate/smoke_test_amip_all_grids.py uses --dt 60 at V4 explicitly,
#   noting "the MPAS hydrostatic dycore is unstable at the default
#   600 s step despite the CFL diagnostic reporting 0.09". V6 is
#   ~4x as many cells as V4 (10242 vs 2562) so MUST be at least as
#   tight as V4. Pinned dt=60 here. iter-32 measurement of V6
#   30-day at dt=60 is the pending follow-up.

# Build the optional-forcing flag set once.
EXTRA_FLAGS=""
if [ -n "$GHG_FILE" ]; then
    EXTRA_FLAGS+=" --ghg-forcing external --ghg-file $GHG_FILE"
fi
if [ -n "$OZONE_FILE" ]; then
    EXTRA_FLAGS+=" --ozone-forcing external --ozone-file $OZONE_FILE"
fi
if [ -n "$AEROSOL_FILE" ]; then
    EXTRA_FLAGS+=" --aerosol-forcing external --aerosol-file $AEROSOL_FILE"
fi

for ENTRY in "${GRID_TABLE[@]}"; do
    IFS=':' read -r GRID RES DISC FOLDER TRUNC DT_OVERRIDE <<< "$ENTRY"
    OUTDIR="$OUTPUT/hydrostatic/amip/$FOLDER/$RES"
    echo "=================================================="
    echo "  AMIP on $GRID/$DISC (resolution=$RES, days=$DAYS)"
    if [ -n "$EXTRA_FLAGS" ]; then
        echo "  Forcing: $EXTRA_FLAGS"
    else
        echo "  Forcing: constant GHG / standard ozone / off aerosol"
    fi
    echo "  → $OUTDIR"
    echo "=================================================="
    mkdir -p "$OUTDIR"
    # iter-43 codex HIGH: remove stale ``timeseries.npz`` from a
    # prior successful run BEFORE invoking ``run_amip.py``.  If the
    # new run fails, no ``timeseries.npz`` will exist, which makes
    # the converter's purge path fire (and it also returns non-zero
    # so ANY_FAILED is set correctly).  Without this delete, a
    # failed re-run into an existing OUTDIR would re-converge old
    # data on the next cross-grid pass.
    rm -f "$OUTDIR/timeseries.npz"
    rm -f "$OUTDIR/mean_timeseries.csv"
    # iter-74: ``run_amip.py`` only emits diagnostics every
    # ``--diag-days`` (default 5).  Short-day smokes (DAYS<=5)
    # produce empty ``timeseries.npz``, which the iter-42
    # converter then purges as failed output.  Force daily
    # diagnostics so even 1-2 day smokes accumulate data.
    # iter-73 made the same fix for the RCE wrapper.
    #
    # iter-74 ALSO fixes a latent bug: the wrapper computed
    # ``$RES`` from the GRID_RES dict for the OUTDIR path but
    # never passed it to ``run_amip.py``.  Result: every grid
    # ran at ``--resolution`` default (16), regardless of the
    # GRID_RES value.  Now ``--resolution "$RES"`` is passed
    # explicitly so cube C48 runs at C48, latlon at the right
    # n_lat, etc.  ``$TRUNC`` (--truncation) is the gaussian-
    # specific override that the iter-41 wrapper already had.
    # iter-32: pass iter-13-validated dt to AMIP at C48/T42 to
    # avoid the iter-13-banned dt=600 default (which produced
    # C48 30-day BLOWUP in RCE iter-12). Other grids fall through
    # to run_amip.py's default of 600 until measured otherwise.
    # iter-7 RCE wrapper Metal fix replicated for the AMIP wrapper.
    DT_FLAG=""
    if [ -n "$DT_OVERRIDE" ]; then
        DT_FLAG="--dt $DT_OVERRIDE"
    fi
    JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}" JAX_ENABLE_X64=1 .venv/bin/python scripts/run_amip.py \
        --grid-type "$GRID" --discretization "$DISC" \
        --resolution "$RES" \
        $TRUNC $DT_FLAG --days "$DAYS" --diag-days 1 --output "$OUTDIR" \
        $EXTRA_FLAGS || {
        echo "  WARNING: run_amip.py failed for $GRID; continuing"
        echo "  with cross-grid loop so other grids still produce"
        echo "  output.  The format converter below will purge the"
        echo "  $GRID directory of stale matrix-format files."
        # iter-104 codex HIGH-1: previously this block did NOT
        # set ANY_FAILED=1, relying on the converter (line ~148)
        # to detect failure via missing timeseries.npz.  But the
        # iter-100 ``run_amip.py`` exit-1 on NaN happens AFTER
        # ``driver.run()`` has already written a partial
        # timeseries.npz, so the converter could succeed on
        # garbage and ANY_FAILED would never get set.  Mark
        # failure HERE, where ``run_amip.py``'s exit code is
        # the unambiguous truth.
        ANY_FAILED=1
    }
    # iter-42: convert run_amip.py outputs (timeseries.npz +
    # free-form results.txt) to matrix-runner-compatible files
    # (mean_timeseries.csv + key:value results.txt) so the
    # cross-grid plot collector below can pick them up.
    # iter-42 codex HIGH: converter returns non-zero on missing
    # timeseries.npz (= failed AMIP run); we honor that exit
    # status by setting ANY_FAILED so the cross-grid plot step
    # can warn the user.
    if ! .venv/bin/python scripts/_amip_to_matrix_format.py "$OUTDIR"; then
        echo "  WARNING: $GRID has no usable AMIP output; cross-"
        echo "  grid collection will skip it."
        ANY_FAILED=1
    fi
done

echo ""
echo "=================================================="
echo "  Generating cross-grid comparison plots"
echo "=================================================="
if [ "$ANY_FAILED" = "1" ]; then
    echo "  NOTE: at least one grid had no usable AMIP output;"
    echo "  the cross-grid plot will only show the grids that"
    echo "  succeeded.  Re-run the failing grid(s) to get a"
    echo "  complete cross-grid comparison."
fi
# iter-104 codex MEDIUM-4: wrap the comparison-plot step.
# See ``run_omip_cross_grid.sh`` for the rationale.
PLOT_FAILED=0
JAX_ENABLE_X64=1 .venv/bin/python scripts/run_atmosphere_test_matrix.py \
    --cross-grid-plots-only --test amip --output "$OUTPUT" \
    || PLOT_FAILED=1
if [ "$PLOT_FAILED" = "1" ]; then
    echo "  WARNING: comparison-plot step failed; partial plots"
    echo "  may still exist for grids that succeeded."
fi

# iter-103/104: propagate ANY_FAILED || PLOT_FAILED to the
# wrapper's own exit code so CI can detect per-grid failures
# AND plot-step failures via ``$?``.
exit $((ANY_FAILED || PLOT_FAILED))
