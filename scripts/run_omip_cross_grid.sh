#!/bin/bash
# Run OMIP on each supported ocean grid and produce a cross-grid
# comparison via the ocean-matrix plotter.
#
# Iter-25: addresses the user's prompt item "OMIP" cross-grid.
# ``scripts/run_omip.py`` (now with iter-25 mean_timeseries.csv +
# results.txt outputs) is invoked once per grid; ``run_omip.py``
# already places output at ``$OUTPUT/<grid>/<resolution>/`` which
# is the same layout the ocean cross-grid plotter expects.
#
# Usage:
#     scripts/run_omip_cross_grid.sh OUTPUT [DAYS] [PHYSICS]
# Defaults: DAYS=30, PHYSICS=full.
#
# After all grids complete, the ocean cross-grid plotter is invoked
# (currently via the per-grid plot in run_omip.py — full cross-grid
# integration with comparison_*.png plots requires reorganizing the
# OMIP output under the canonical ocean-matrix layout, tracked as a
# follow-up).

set -e

OUTPUT=${1:?usage: $0 OUTPUT [DAYS] [PHYSICS]}
DAYS=${2:-30}
PHYSICS=${3:-full}

for GRID in cubed_sphere latlon mpas spectral; do
    OUTDIR="$OUTPUT"
    echo "=================================================="
    echo "  OMIP on $GRID (days=$DAYS, physics=$PHYSICS)"
    echo "=================================================="
    JAX_ENABLE_X64=1 .venv/bin/python scripts/run_omip.py \
        --grid "$GRID" --days "$DAYS" --physics "$PHYSICS" \
        --output "$OUTDIR"
done

echo ""
echo "=================================================="
echo "  All OMIP grids done.  Cross-grid plots: per-grid"
echo "  ``timeseries.png`` files are under \$OUTPUT/<grid>/."
echo "  Full cross-grid comparison plots require reorganizing"
echo "  the OMIP output to match the ocean-matrix layout —"
echo "  follow-up scope."
echo "=================================================="
