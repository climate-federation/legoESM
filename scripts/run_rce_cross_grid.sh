#!/bin/bash
# Run moist RCE on each supported grid type and produce a cross-grid
# time-series comparison via the atmosphere-matrix plotter.
#
# Iter-24: addresses the user's prompt item "RCE on all grid types".
# ``scripts/run_rce.py`` (now with iter-24 mean_timeseries.csv +
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

# Map grid_type → (resolution, discretization, folder) appropriate
# for ~1.25-3 deg coverage.  iter-26 codex HIGH: ``--discretization``
# is REQUIRED for non-cubed grids — ``run_rce.py``'s default
# ``cdgrid`` is rejected for latlon/voronoi/gaussian.
declare -A GRID_RES=(
    [cubed_sphere]="24"
    [latlon]="32"
    [voronoi]="4"
    [gaussian]="21"
)
declare -A GRID_DISC=(
    [cubed_sphere]="cdgrid"
    [latlon]="latlon_cgrid"
    [voronoi]="mpas"
    [gaussian]="spectral"
)
declare -A GRID_FOLDER=(
    [cubed_sphere]="cubed_sphere"
    [latlon]="latlon"
    [voronoi]="icosahedral"
    [gaussian]="spectral"
)

for GRID in cubed_sphere latlon voronoi gaussian; do
    RES=${GRID_RES[$GRID]}
    DISC=${GRID_DISC[$GRID]}
    FOLDER=${GRID_FOLDER[$GRID]}
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
    JAX_ENABLE_X64=1 .venv/bin/python scripts/run_rce.py \
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
PLOT_FAILED=0
JAX_ENABLE_X64=1 .venv/bin/python scripts/run_atmosphere_test_matrix.py \
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
