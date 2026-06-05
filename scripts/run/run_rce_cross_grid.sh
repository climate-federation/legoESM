#!/usr/bin/env bash
# Run moist RCE on each supported grid type and produce a cross-grid
# time-series comparison via the atmosphere-matrix plotter.
#
# Iter-24: addresses the user's prompt item "RCE on all grid types".
# ``scripts/run/run_rce.py`` (now with iter-24 mean_timeseries.csv +
# results.txt outputs) is invoked once per grid; output is laid out at
# ``$OUTPUT/hydrostatic/rce/<grid>/<resolution>/`` so
# ``run_atmosphere_test_matrix.py --cross-grid-plots-only`` picks it up.
#
# Usage:
#     scripts/run_rce_cross_grid.sh OUTPUT [DAYS] [DIAG_DAYS]
# Defaults: DAYS=30, DIAG_DAYS=1 (one diag entry per day so
# short-day smokes accumulate data).

set -e

OUTPUT=${1:?usage: $0 OUTPUT [DAYS] [DIAG_DAYS]}
DAYS=${2:-30}
ANY_FAILED=0
# iter-73: ``run_rce.py`` only writes ``mean_timeseries.csv`` +
# matrix-format ``results.txt`` if its diag_log is non-empty,
# which requires ``DAYS >= --diag-days`` (default 5).  For
# short smoke tests (DAYS=2 or less), force daily diagnostics
# so the cross-grid plotter has something to read.
DIAG_DAYS=${3:-1}

# Grid configuration table (parallel arrays — macOS default bash 3.2
# does not support `declare -A` associative arrays). Each entry is
# "<grid_type>:<resolution>:<discretization>:<output_folder>". iter-26
# codex HIGH: --discretization is REQUIRED for non-cubed grids;
# run_rce.py's default cdgrid is rejected for latlon/voronoi/gaussian.
GRID_TABLE=(
    "cubed_sphere:24:cdgrid:cubed_sphere"
    "latlon:32:latlon_cgrid:latlon"
    "voronoi:4:mpas:icosahedral"
    "gaussian:21:spectral:spectral"
)

for ENTRY in "${GRID_TABLE[@]}"; do
    IFS=':' read -r GRID RES DISC FOLDER <<< "$ENTRY"
    OUTDIR="$OUTPUT/hydrostatic/rce/$FOLDER/$RES"
    echo "=================================================="
    echo "  RCE on $GRID/$DISC (resolution=$RES, days=$DAYS)"
    echo "  → $OUTDIR"
    echo "=================================================="
    mkdir -p "$OUTDIR"
    # iter-75 codex HIGH: purge stale matrix-format files from a
    # prior run BEFORE invoking ``run_rce.py``.  Without this, a
    # failed re-run into an existing OUTDIR could leave old
    # ``mean_timeseries.csv`` / ``results.txt`` in place which
    # the cross-grid plotter would treat as current success.
    # Mirrors the iter-43 AMIP wrapper pattern.
    rm -f "$OUTDIR/mean_timeseries.csv"
    rm -f "$OUTDIR/results.txt"
    rm -f "$OUTDIR/timeseries.npz"
    # iter-73: mirror the iter-43 / iter-72 single-grid-failure
    # tolerance pattern so a per-grid blowup (e.g., the iter-73
    # voronoi RCE BLOWUP) does not abort the entire cross-grid
    # run.  iter-101 made ``run_rce.py`` exit 1 on BLOWUP
    # (matching iter-97 ``run_omip.py`` and iter-100
    # ``run_amip.py`` conventions), so this guard now reliably
    # catches the expected per-grid failures via ``$?`` and
    # records ``ANY_FAILED=1``.
    # JAX_PLATFORMS=${JAX_PLATFORMS:-cpu}: pin CPU by default; the
    # spectral + voronoi paths trigger an MLIR legalisation error on
    # Apple Metal ("func.func op ... data types not supported"). User
    # can override with JAX_PLATFORMS=metal at their own risk.
    JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}" JAX_ENABLE_X64=1 .venv/bin/python scripts/run/run_rce.py \
        --grid-type "$GRID" --discretization "$DISC" \
        --resolution "$RES" --days "$DAYS" --diag-days "$DIAG_DAYS" \
        --output "$OUTDIR" || {
        echo "  WARNING: run_rce.py FAILED for $GRID; continuing"
        echo "  with cross-grid loop so other grids still produce"
        echo "  output."
        ANY_FAILED=1
    }
done

echo ""
echo "=================================================="
echo "  Generating cross-grid comparison plots"
echo "=================================================="
if [ "$ANY_FAILED" = "1" ]; then
    echo "  NOTE: at least one grid had no usable RCE output;"
    echo "  the cross-grid plot will only show the grids that"
    echo "  succeeded.  Re-run the failing grid(s) to get a"
    echo "  complete cross-grid comparison."
fi
# iter-104 codex MEDIUM-4: wrap the comparison-plot step.
# See ``run_omip_cross_grid.sh`` for the rationale.
#
# iter-28: same JAX_PLATFORMS=cpu default as the per-grid runs.
# iter-7 documented the Metal MLIR legalisation crash on the
# spectral-grid plot path; without this pin a user with
# JAX_PLATFORMS=metal exported in their shell would always see the
# comparison-plot step crash even though every per-grid run wrote
# valid output.
PLOT_FAILED=0
JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}" JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_atmosphere_test_matrix.py \
    --cross-grid-plots-only --test rce --output "$OUTPUT" \
    || PLOT_FAILED=1
if [ "$PLOT_FAILED" = "1" ]; then
    echo "  WARNING: comparison-plot step failed; partial plots"
    echo "  may still exist for grids that succeeded."
fi

# iter-103/104: propagate ANY_FAILED || PLOT_FAILED to the
# wrapper's own exit code.  This pairs with iter-101 which
# fixed ``run_rce.py`` to exit 1 on BLOWUP.
exit $((ANY_FAILED || PLOT_FAILED))
