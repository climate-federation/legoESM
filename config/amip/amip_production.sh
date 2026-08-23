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
#   * LAND: since 2026-07-07 the YAML runs the FULL multilayer Richards land
#     (use_multilayer_land: true — CLM PFT/texture surfdata, prognostic snow,
#     snow_albedo_feedback: true, stomatal-beta OFF per #741) under tiled
#     COARE3(ocean)/MOST(land) fluxes. Pierre's coupled config uses
#     slab_richards — the land-model choice is the remaining divergence to
#     reconcile, not snow albedo (the old "no snow-albedo feedback" caveat
#     applied to the retired simplified land tile).
#   * convective_cloud: the tracked YAML now ships TRUE (mirror the canonical
#     tuned base config/cmip/cmip_tuned_physics.yaml for parameter-identity with
#     the coupled CMIP slab). Prescribed-SST AMIP SCIENCE runs OVERRIDE it to
#     FALSE (best TOA): a 30-day C48 A/B (jobs 25929869 ON vs 25929870 OFF) showed
#     ON warms the tropics only +0.55 K while pushing albedo 0.295->0.370 and OLR
#     234.7->213.3 (~25 W/m^2 off CERES); SBM-alone is CMIP6-class (albedo 0.295
#     vs 0.290). `--convective-cloud` is a one-directional store_true and cannot
#     flip the YAML's true back off, so the OFF runner (amip_cmip6_chain.sbatch)
#     uses a tiny `include: amip_production.yaml` + `convective_cloud: false`
#     override. The residual tropical cold bias is a convective-heating /
#     surface-flux issue, not cloud-radiative.
# =============================================================================

# --- Machine-specific paths (Levante defaults; override via the environment) ---
: "${ICON_ROOT:=/pool/data/ICON/grids/public/mpim}"
: "${ERA5_IC:=/scratch/b/b309178/era5_ic_1979-01-01.zarr}"
: "${ETOPO:=/work/bd1083/b309178/diffESM/legoesm_ap/data/bathymetry/etopo_1deg_clean.nc}"
# SST/SIC: PCMDI CMIP7 AMIP boundary conditions (the protocol-standard tosbcs/
# siconcbcs mid-month files, 1870-2022) — the dataset every proven MPAS-lane
# run uses (Pierre's climeval_bech day-247+ and the replica chains).  The file
# is in degC -> the PAIRED offset below converts to K; keep SST and
# AMIP_SST_OFFSET consistent when overriding (the legacy ICON bc_sst files are
# already Kelvin and need offset 0).
: "${PCMDI_BC:=/pool/data/INPUT4MIP/data/input4MIPs/CMIP7/CMIP/PCMDI/PCMDI-AMIP-1-1-10}"
: "${SST:=${PCMDI_BC}/ocean/mon/tosbcs/gn/v20250807/tosbcs_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-10_gn_187001-202212.nc}"
: "${SIC:=${PCMDI_BC}/seaIce/mon/siconcbcs/gn/v20250807/siconcbcs_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-10_gn_187001-202212.nc}"
: "${AMIP_SST_OFFSET:=273.15}"
# Legacy alternative (Kelvin; pair with AMIP_SST_OFFSET=0):
#   SST=${ICON_ROOT}/0019/sst_and_seaice/r0001/bc_sst_1979_2016.nc
#   SIC=${ICON_ROOT}/0019/sst_and_seaice/r0001/bc_sic_1979_2016.nc
: "${SOLAR:=${ICON_ROOT}/common/solar_radiation/swflux_14band_cmip6_1850-2299-v3.2.nc}"
: "${GHG:=${ICON_ROOT}/independent/greenhouse_gases/greenhouse_historical_plus.nc}"
# Ozone/aerosol/volcanic default to the merged 1979-2014 transient files (432
# monthly records each, El Chichon + Pinatubo included) so the DEFAULT run is
# CMIP6-compliant for the full AMIP period. The old single-year (Kinne 1979,
# volcanic 1979) and 1950-1999 ozone pool files silently freeze/hold forcing
# past their coverage — only override back to them for deliberate fixed-forcing
# sensitivity runs.
: "${FORCING_1979_2014:=/work/bd1083/b309178/diffESM/legoesm_ap/data/forcing/cmip6_1979-2014}"
: "${OZONE:=${FORCING_1979_2014}/vmro3_input4MIPs_ozone_CMIP_UReading-CCMI-1-0_gn_197901-201412.nc}"
: "${AEROSOL:=${FORCING_1979_2014}/aeropt_kinne_sw_b14_fin_1979-2014_rast.nc}"
: "${VOLCANIC:=${FORCING_1979_2014}/bc_aeropt_cmip6_volc_lw_b16_sw_b14_1979-2014.nc}"
# CLM surfdata for the multilayer Richards land model (use_multilayer_land): the
# PFT/texture/glacier surface map. Compute nodes have NO internet, so the auto-
# download in clm_surface_map.download_clm_surfdata() FAILS (SSL) — this file
# MUST be staged locally and passed via --clm-surfdata-path (staged 2026-07-02).
: "${CLM_SURFDATA:=/work/bd1083/b309178/diffESM/legoesm_ap/data/clm/surfdata_1.9x2.5_16pfts_CMIP6_simyr2000.nc}"
# Subgrid-orography stddev (SSO) for the orographic GWD launch (#1514). With
# gravity_wave_drag=mcfarlane+hines and NO SSO file, McFarlane falls back to
# the scalar h_topo=500 m on EVERY column — a fictional 500-m mountain over
# the open ocean, measured at -0.29 Pa column drag over 40-60S (2x the entire
# observed surface stress): it removed the eddy-driven westerly belt within a
# week of the ERA5 IC and suppressed the storm tracks in BOTH hemispheres.
# File built by scripts/data/prep_subgrid_orography.py from etopo_1deg_clean
# (--fine-res-deg 1.0 --block-deg 2.0; Southern-Ocean mean sgh 500 -> 3.6 m).
# ADOPTING THIS CHANGES THE MODEL CLIMATE (validated 90-d + 274-d restart
# arms, issue #1514): any comparison spanning it is confounded. Opt OUT with
# AMIP_SSO="" (pre-fix reproduction runs only).
: "${AMIP_SSO:=/work/bd1083/b309178/diffESM/legoesm_ap/data/orography/sso_stdh_2deg_from1deg.nc}"
# Land-sea mask (sftlf, fraction) built from the SAME CLM surfdata by
# scripts/data/build_sftlf_from_surfdata.py (2026-07-22). Without it f_land is
# derived from ETOPO elevation>0, and ETOPO's inland-sea BATHYMETRY (Caspian
# -28 m, Aral) classifies those basins as OCEAN -> prescribed nearest-neighbour
# SST -> uncapped potential evaporation (the 2000 W/m^2 hfls / 177 mm prw
# central-Asia hotspots in the 2-yr pilot). The CLM mask keeps the surface
# tiling consistent with the land model's own footprint (Caspian/Aral = land).
# Opt back to the legacy elevation mask with AMIP_LAND_MASK="" (deliberate
# reproduction runs only).
: "${AMIP_LAND_MASK:=/work/bd1083/b309178/diffESM/legoesm_ap/data/clm/sftlf_clm_1.9x2.5.nc}"

# --- The machine-specific PATH flags (everything else is in the YAML) ---------
# Harmonized surfdata: the CANOPY parameter source (per-PFT canopy height,
# roughness ratio, Vcmax25, band albedos).  The default land surface scheme is
# the two-leaf canopy, which is REFUSED without this — the CLM surfdata below
# supplies soil texture and PFT cover, not canopy structure.  Two datasets, two
# jobs; a canopy run needs both staged.
: "${AMIP_SURFDATA:=${REPO:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM}/data/legoesm_surfdata_c260716.nc}"

AMIP_PATH_FLAGS=(
  --surfdata "${AMIP_SURFDATA}"
  --ic-path "${ERA5_IC}"
  --forcing-path "${SST}"
  --sic-path "${SIC}"
  --sst-offset "${AMIP_SST_OFFSET}"
  --solar-file "${SOLAR}"
  --ozone-file "${OZONE}"
  --ghg-file "${GHG}"
  --aerosol-file "${AEROSOL}"
  --volcanic-aerosol-file "${VOLCANIC}"
  --topography "${ETOPO}"
  --clm-surfdata-path "${CLM_SURFDATA}"
)
# Volcanic LONGWAVE absorption (ext_earth band; post-eruption stratospheric
# heating). ON by default for the campaign (decision 2026-07-21): reader
# validated against the merged 1979-2014 file (Pinatubo-peak LW AOD 0.020 vs
# 0.0005 quiet-1979; El Chichon visible). LW AOD ~0 outside 1982-84/1991-93,
# so quiet years are unaffected; the pilot decade exercises El Chichon before
# the full 36-yr. No forward run has exercised the LW channel yet — first
# post-1982 pilot output is the validation gate. Opt OUT (e.g. to reproduce
# pre-2026-07-21 SW-only runs) with AMIP_VOLCANIC_LW=0.
if [[ "${AMIP_VOLCANIC_LW:-1}" == "1" ]]; then
  AMIP_PATH_FLAGS+=( --volcanic-aerosol-lw )
fi
# CLM-derived land mask (see AMIP_LAND_MASK above). Side-effect scope audited
# 2026-07-22: with use_multilayer_land=true the ONLY climate-relevant change is
# f_land itself (the forced slab_land_active is shadowed by the multilayer tile
# branch; surfdata albedo needs --surfdata, which this launcher never passes).
if [[ -n "${AMIP_LAND_MASK}" ]]; then
  AMIP_PATH_FLAGS+=( --land-mask-file "${AMIP_LAND_MASK}" )
fi
# Subgrid orography for the orographic GWD launch (see AMIP_SSO above, #1514).
# Without this flag the run trips the loud scalar-fallback warning (PR #1540)
# and reproduces the pseudo-mountain climate. Opt-out is DELIBERATE only.
if [[ -n "${AMIP_SSO}" ]]; then
  AMIP_PATH_FLAGS+=( --subgrid-orography-file "${AMIP_SSO}" )
fi
