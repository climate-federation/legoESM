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
    # iter-53: write under ``$OUTPUT/omip/`` so the test-case
    # parent directory has the canonical name for the
    # ocean-matrix collector + replot path.  ``run_omip.py``
    # appends ``/<grid>/<resolution>/`` itself, so the final
    # layout is ``$OUTPUT/omip/<grid>/<resolution>/`` matching
    # the matrix-runner convention.
    OUTDIR="$OUTPUT/omip"
    echo "=================================================="
    echo "  OMIP on $GRID (days=$DAYS, physics=$PHYSICS)"
    echo "=================================================="
    JAX_ENABLE_X64=1 .venv/bin/python scripts/run_omip.py \
        --grid "$GRID" --days "$DAYS" --physics "$PHYSICS" \
        --output "$OUTDIR"
done

echo ""
echo "=================================================="
echo "  Generating cross-grid comparison plots"
echo "=================================================="
# iter-53: invoke the ocean matrix's ``--replot`` discovery
# path so the iter-49-relaxed collector produces the OMIP
# cross-grid comparison plots.  Without this step the
# wrapper produces only per-grid plots.
JAX_ENABLE_X64=1 .venv/bin/python scripts/run_ocean_test_matrix.py \
    --replot --only omip --output "$OUTPUT"
