#!/usr/bin/env bash
# =============================================================================
# legoESM AMIP — GINSBURG (Columbia) machine paths for the authoritative config
# =============================================================================
# Burg/Ginsburg twin of config/amip/amip_production.sh (which pins Levante paths).
# The authoritative PHYSICS/grid/dycore/forcing-selector/diagnostics parameter set
# stays in config/amip/amip_production.yaml — do NOT fork physics here.
#
#   source config/amip/amip_production.ginsburg.sh
#   JAX_ENABLE_X64=1 "${PY}" -u scripts/run/run_amip.py \
#       --config config/amip/amip_production.yaml \
#       "${AMIP_PATH_FLAGS[@]}" --days "${DAYS:-10}" --output "${OUTDIR:-results/amip/ginsburg}"
#
# Inputs are LOCAL (downloaded once, not streamed), staged by the prep helpers:
#   ERA5 IC  : scripts/data/prep_era5_ic_from_zarr.py  (public ARCO/WB2 -> local zarr)
#   ETOPO    : scripts/data/prep_etopo_topography.py   (an elevation NetCDF -> 1/4deg topo)
#   forcing  : scripts/data/generate_amip_forcing.py   (synthetic CMIP6-schema deck)
# Observed SST/SIC (faithful, Stage 3) needs an ESGF account — see the override below.
# =============================================================================
set -uo pipefail

REPO="${LEGOESM_REPO:-/burg-archive/glab/users/ac5006/legoESM}"
: "${PY:=${REPO}/.venv/bin/python}"                 # the GPU .venv (jax 0.10.2)
DATA="${AMIP_DATA:-${REPO}/data/amip}"

# --- LOCAL inputs (override any via the environment) --------------------------
: "${ERA5_IC:=${DATA}/era5_ic_1979-01-01.zarr}"     # prep_era5_ic_from_zarr.py output
: "${ETOPO:=${DATA}/etopo_0p25deg.nc}"              # prep_etopo_topography.py output (PRODUCE THIS)
: "${FORCING_DIR:=${DATA}/forcing_amip}"            # generate_amip_forcing.py deck

# --- SST/SIC source -----------------------------------------------------------
# DEFAULT = the REAL observed input4MIPs PCMDI AMIP II bcs (tosbcs [degC] / siconcbcs [%],
# 1870-2022 monthly, 1deg), Globus-staged from the ALCF ESGF data node into $DATA.
# The loader's units guard converts degC->K and the YAML sic_scale=0.01 does %->fraction.
# For a quick no-real-SST smoke instead, override with the synthetic deck:
#   SST_FILE=$DATA/forcing_amip/sst_sic_amip_1979-2014.nc SIC_FILE=$DATA/forcing_amip/sst_sic_amip_1979-2014.nc \
#   SST_VAR=sst SIC_VAR=sic source config/amip/amip_production.ginsburg.sh
: "${SST_FILE:=${DATA}/tosbcs_input4MIPs_PCMDI-AMIP-1-1-9_gn_187001-202212.nc}"
: "${SIC_FILE:=${DATA}/siconcbcs_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-9_gn_187001-202212.nc}"
: "${SST_VAR:=tosbcs}"
: "${SIC_VAR:=siconcbcs}"

# --- external CMIP6 forcing (synthetic deck by default; drop real input4MIPs in) -
: "${SOLAR:=${FORCING_DIR}/solar_amip_1979-2014.nc}"
: "${OZONE:=${FORCING_DIR}/ozone_amip_clim.nc}"
: "${GHG:=${FORCING_DIR}/ghg_amip_1979-2014.nc}"
: "${AEROSOL:=${FORCING_DIR}/aerosol_amip_clim.nc}"
: "${VOLCANIC:=${FORCING_DIR}/volcanic_amip_1979-2014.nc}"

# --- machine-specific PATH flags (everything else is in the YAML) -------------
AMIP_PATH_FLAGS=(
  --ic-path "${ERA5_IC}"
  --topography "${ETOPO}"
  --forcing-path "${SST_FILE}" --sst-var "${SST_VAR}"
  --sic-path "${SIC_FILE}"     --sic-var "${SIC_VAR}"
  --solar-file "${SOLAR}"
  --ozone-file "${OZONE}"
  --ghg-file "${GHG}"
  --aerosol-file "${AEROSOL}"
  --volcanic-aerosol-file "${VOLCANIC}"
)

# Warn (don't fail) on a missing local input so a Stage-0 flat-topo smoke still works.
for _f in "${ETOPO}"; do
  [ -e "${_f}" ] || echo "[ginsburg] NOTE: ${_f} not found — produce it with " \
    "scripts/data/prep_etopo_topography.py, or run a flat-topo smoke (--topography flat)." >&2
done
export PY AMIP_PATH_FLAGS
