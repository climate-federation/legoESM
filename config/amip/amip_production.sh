#!/usr/bin/env bash
# =============================================================================
# legoESM AMIP — machine paths + launcher for the AUTHORITATIVE production config
# =============================================================================
# The authoritative PHYSICS / grid / dycore / forcing-selector / diagnostics
# parameter set lives in ONE machine-independent file:
#
#     config/amip/amip_production.yaml   (--config single source of truth)
#
# This shell file supplies ONLY the machine-specific PATHS (which are not
# committed into the physics config) and a ready launcher. Do NOT fork per-run
# physics here — edit the YAML (under review) so all runs stay comparable.
#
#   source config/amip/amip_production.sh
#   "${PY}" -u scripts/run/run_amip.py \
#       --config config/amip/amip_production.yaml \
#       "${AMIP_PATH_FLAGS[@]}" --days "${DAYS}" --output "${OUTDIR}"
#
# Provenance + validation: see the header of config/amip/amip_production.yaml
# (VALIDATED at C48/L40, job 25918469: albedo 0.292, rsut 99, pr 3.2, net TOA
# +6). Tracking: GitHub issue #636.
#
# OPEN ITEMS — need Pierre/Veronika to finalize (do NOT silently diverge):
#   * RUNNER: this is the run_amip.py (prescribed-SST) --config path. Pierre's
#     tuned config is a run_coupled.py --config YAML (cmip_ocean_slab.yaml).
#   * LAND: this uses our tiled COARE3/MOST + bucket + stomatal tile. Pierre's
#     config uses slab_richards. The simplified AMIP land tile has NO
#     snow-albedo feedback (snow tracked thermodynamically but radiatively
#     invisible); snow_albedo_feedback lives in the full land model, so it
#     comes bundled with the land-model choice, NOT a standalone flag.
#   * convective_cloud: OFF in the YAML. A 30-day C48 A/B (jobs 25929869 ON vs
#     25929870 OFF) showed ON warms the tropics only +0.55 K while pushing albedo
#     0.295->0.370 and OLR 234.7->213.3 (~25 W/m^2 off CERES); SBM-alone is
#     CMIP6-class (albedo 0.295 vs 0.290). The tropical cold bias is a convective-
#     heating / surface-flux issue, not cloud-radiative.
# =============================================================================

# --- Machine-specific paths (Levante defaults; override via the environment) ---
: "${ICON_ROOT:=/pool/data/ICON/grids/public/mpim}"
: "${ERA5_IC:=/scratch/b/b309178/era5_ic_1979-01-01.zarr}"
: "${ETOPO:=/work/bd1083/b309178/diffESM/legoesm_ap/data/bathymetry/etopo_1deg_clean.nc}"
: "${SST:=${ICON_ROOT}/0019/sst_and_seaice/r0001/bc_sst_1979_2016.nc}"
: "${SIC:=${ICON_ROOT}/0019/sst_and_seaice/r0001/bc_sic_1979_2016.nc}"
: "${SOLAR:=${ICON_ROOT}/common/solar_radiation/swflux_14band_cmip6_1850-2299-v3.2.nc}"
: "${GHG:=${ICON_ROOT}/independent/greenhouse_gases/greenhouse_historical_plus.nc}"
: "${OZONE:=${ICON_ROOT}/common/ozone_cmip6_forcing/historical/vmro3_input4MIPs_ozone_CMIP_UReading-CCMI-1-0_gn_195001-199912.nc}"
: "${AEROSOL:=${ICON_ROOT}/common/aerosol_kinne/aeropt_kinne_sw_b14_fin_1979_rast.nc}"
: "${VOLCANIC:=${ICON_ROOT}/common/aerosol_volcanic_cmip6/bc_aeropt_cmip6_volc_lw_b16_sw_b14_1979.nc}"
# CLM surfdata for the multilayer Richards land model (use_multilayer_land): the
# PFT/texture/glacier surface map. Compute nodes have NO internet, so the auto-
# download in clm_surface_map.download_clm_surfdata() FAILS (SSL) — this file
# MUST be staged locally and passed via --clm-surfdata-path (staged 2026-07-02).
: "${CLM_SURFDATA:=/work/bd1083/b309178/diffESM/legoesm_ap/data/clm/surfdata_1.9x2.5_16pfts_CMIP6_simyr2000.nc}"

# --- The machine-specific PATH flags (everything else is in the YAML) ---------
AMIP_PATH_FLAGS=(
  --ic-path "${ERA5_IC}"
  --forcing-path "${SST}"
  --sic-path "${SIC}"
  --solar-file "${SOLAR}"
  --ozone-file "${OZONE}"
  --ghg-file "${GHG}"
  --aerosol-file "${AEROSOL}"
  --volcanic-aerosol-file "${VOLCANIC}"
  --topography "${ETOPO}"
  --clm-surfdata-path "${CLM_SURFDATA}"
)
