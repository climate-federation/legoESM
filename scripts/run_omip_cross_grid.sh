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
ANY_FAILED=0

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
    # iter-75 codex HIGH: purge stale matrix-format files
    # from a prior successful run BEFORE invoking
    # ``run_omip.py``.  Without this, a failed re-run could
    # leave the cross-grid plotter reading old data.
    # ``run_omip.py`` writes to ``$OUTDIR/<grid>/<resolution>/``
    # so we purge per-grid (not per-resolution since the
    # resolution is determined by ``run_omip.py``'s own
    # GRID_DEFAULTS).  The ``-rf`` is scoped tightly to the
    # known ``$OUTDIR/$GRID`` subdirectory to avoid wider
    # damage if ``$OUTDIR`` is unset (the ``${1:?}`` guard
    # at line 23 catches the unset OUTPUT case anyway).
    rm -rf "$OUTDIR/$GRID"
    # iter-72: ``run_omip.py`` exits with code 1 on FAIL (e.g.,
    # the iter-71 cube C24 BLOWUP).  Without the ``|| { ... }``
    # guard, ``set -e`` aborts the whole wrapper after the
    # first grid failure — preventing the other 3 grids from
    # contributing to the cross-grid plot.  Mirrors the iter-43
    # AMIP wrapper pattern: log a warning, set ANY_FAILED=1,
    # and continue.
    JAX_ENABLE_X64=1 .venv/bin/python scripts/run_omip.py \
        --grid "$GRID" --days "$DAYS" --physics "$PHYSICS" \
        --output "$OUTDIR" || {
        echo "  WARNING: run_omip.py FAILED for $GRID"
        echo "  (e.g., the iter-71 cube C24 BLOWUP).  Continuing"
        echo "  with cross-grid loop so the other grids still"
        echo "  contribute to the cross-grid plot."
        ANY_FAILED=1
    }
done

echo ""
echo "=================================================="
echo "  Generating cross-grid comparison plots"
echo "=================================================="
if [ "$ANY_FAILED" = "1" ]; then
    echo "  NOTE: at least one grid had no usable OMIP output;"
    echo "  the cross-grid plot will only show the grids that"
    echo "  succeeded.  Re-run the failing grid(s) to get a"
    echo "  complete cross-grid comparison."
fi
# iter-53: invoke the ocean matrix's ``--replot`` discovery
# path so the iter-49-relaxed collector produces the OMIP
# cross-grid comparison plots.  Without this step the
# wrapper produces only per-grid plots.
JAX_ENABLE_X64=1 .venv/bin/python scripts/run_ocean_test_matrix.py \
    --replot --only omip --output "$OUTPUT"
