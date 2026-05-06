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

# Map grid_type → (resolution, dt) appropriate for ~1.25-3 deg coverage.
declare -A GRID_RES=(
    [cubed_sphere]="24"
    [latlon]="32"
    [voronoi]="4"
    [gaussian]="21"
)
declare -A GRID_FOLDER=(
    [cubed_sphere]="cubed_sphere"
    [latlon]="latlon"
    [voronoi]="icosahedral"
    [gaussian]="spectral"
)

for GRID in cubed_sphere latlon voronoi gaussian; do
    RES=${GRID_RES[$GRID]}
    FOLDER=${GRID_FOLDER[$GRID]}
    OUTDIR="$OUTPUT/hydrostatic/rce/$FOLDER/$RES"
    echo "=================================================="
    echo "  RCE on $GRID (resolution=$RES, days=$DAYS)"
    echo "  → $OUTDIR"
    echo "=================================================="
    mkdir -p "$OUTDIR"
    JAX_ENABLE_X64=1 .venv/bin/python scripts/run_rce.py \
        --grid-type "$GRID" --resolution "$RES" \
        --days "$DAYS" --output "$OUTDIR"
done

echo ""
echo "=================================================="
echo "  Generating cross-grid comparison plots"
echo "=================================================="
JAX_ENABLE_X64=1 .venv/bin/python scripts/run_atmosphere_test_matrix.py \
    --cross-grid-plots-only --test rce --output "$OUTPUT"
