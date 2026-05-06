#!/bin/bash
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

# Map grid_type → (resolution, discretization, folder) appropriate
# for ~3 deg coverage (matrix-runner default for AMIP).  AMIP needs
# realistic SSTs, so the 1-degree grid is overkill for cross-grid
# consistency testing.
declare -A GRID_RES=(
    [cubed_sphere]="48"
    [latlon]="90x180"
    [voronoi]="6"
    [gaussian]="42"
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
declare -A GRID_TRUNC=(
    [gaussian]="--truncation 42"
)

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

for GRID in cubed_sphere latlon voronoi gaussian; do
    RES=${GRID_RES[$GRID]}
    DISC=${GRID_DISC[$GRID]}
    FOLDER=${GRID_FOLDER[$GRID]}
    TRUNC=${GRID_TRUNC[$GRID]:-}
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
    JAX_ENABLE_X64=1 .venv/bin/python scripts/run_amip.py \
        --grid-type "$GRID" --discretization "$DISC" \
        $TRUNC --days "$DAYS" --output "$OUTDIR" \
        $EXTRA_FLAGS
done

echo ""
echo "=================================================="
echo "  Generating cross-grid comparison plots"
echo "=================================================="
JAX_ENABLE_X64=1 .venv/bin/python scripts/run_atmosphere_test_matrix.py \
    --cross-grid-plots-only --test amip --output "$OUTPUT"
