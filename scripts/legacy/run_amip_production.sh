#!/bin/bash
# ==========================================================================
# Production 10-year AMIP run at C48/L40 (Task 12 capstone)
#
# This script launches the full-physics AMIP experiment with:
#   - C48 cubed-sphere grid (~200 km resolution)
#   - L40 hybrid sigma-pressure levels (p_top = 200 Pa)
#   - RRTMGP correlated-k radiation with diurnal cycle
#   - Analytical ozone (latitude-dependent)
#   - Xu-Randall cloud fraction + RRTMGP cloud optics
#   - Kessler microphysics (or sundqvist for more realism)
#   - SBM convection
#   - Temperature/zenith-dependent surface albedo
#   - Real topography (ETOPO1, if available)
#   - Monthly checkpointing for restart
#   - Monthly-mean diagnostic output
#
# Usage:
#   # With real SST forcing:
#   bash scripts/run_amip_production.sh /path/to/MODEL.SST.COBE-SST2.nc
#
#   # With analytical forcing (no data needed):
#   bash scripts/run_amip_production.sh analytical
#
#   # Restart from checkpoint:
#   bash scripts/run_amip_production.sh /path/to/sst.nc --restart results/amip/.../checkpoint_day_0365.npz
# ==========================================================================

set -e

FORCING_PATH="${1:?Usage: $0 <forcing-path-or-analytical> [--restart checkpoint.npz] [--topo topography.nc]}"
shift

# Parse optional args
RESTART_ARG=""
TOPO_ARG="--topography flat"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --restart)
            RESTART_ARG="--restart-from $2"
            shift 2
            ;;
        --topo)
            TOPO_ARG="--topography $2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Determine dataset flag
if [[ "$FORCING_PATH" == "analytical" ]]; then
    DATASET_ARGS="--dataset analytical"
else
    DATASET_ARGS="--dataset cobe --forcing-path $FORCING_PATH"
fi

echo "=========================================="
echo "  legoESM Production AMIP (Task 12)"
echo "=========================================="
echo "  Forcing: $FORCING_PATH"
echo "  Topography: $TOPO_ARG"
echo ""

JAX_ENABLE_X64=1 python scripts/run_amip.py \
    $DATASET_ARGS \
    --resolution 48 \
    --nlev 40 \
    --vertical-coord hybrid \
    --dt 450 \
    --days 3650 \
    --start-day 0 \
    --diag-days 5 \
    --checkpoint-days 30 \
    --radiation rrtmg \
    --rad-update-steps 3 \
    --diurnal-cycle \
    --ozone-source analytical \
    --clouds xu_randall \
    --microphysics kessler \
    --dynamic-albedo \
    --monthly-means \
    $TOPO_ARG \
    $RESTART_ARG \
    $EXTRA_ARGS

echo ""
echo "  Run complete. Validating output..."
echo ""

# Auto-validate if output directory exists
OUTPUT_DIR=$(ls -td results/amip/C48_L40_3650d_* 2>/dev/null | head -1)
if [[ -n "$OUTPUT_DIR" ]]; then
    python scripts/validate_amip.py "$OUTPUT_DIR" --spinup 365
fi
