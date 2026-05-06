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
#     scripts/run_rce_cross_grid.sh OUTPUT [DAYS]
# Default DAYS = 30.

set -e

OUTPUT=${1:?usage: $0 OUTPUT [DAYS]}
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
    # iter-73: mirror the iter-43 / iter-72 single-grid-failure
    # tolerance pattern so a per-grid blowup (e.g., the iter-71
    # cube-side dycore weakness) does not abort the entire
    # cross-grid run.  ``run_rce.py`` does not currently exit
    # non-zero on FAIL the way ``run_omip.py`` does, so this is
    # a defensive guard for future regressions.
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
JAX_ENABLE_X64=1 .venv/bin/python scripts/run_atmosphere_test_matrix.py \
    --cross-grid-plots-only --test rce --output "$OUTPUT"
