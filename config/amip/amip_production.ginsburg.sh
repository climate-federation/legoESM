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
# IC pre-regridded onto the 720x1440 Gaussian proxy grid the cubed-sphere IC path assumes from
# n_lon (raw WB2 IC is pole-inclusive 721 lat -> size mismatch with the proxy's 720; main uses the
# Gaussian proxy, the actual-nodes fix is on the compare-reanalysis branch).
# + the same analysis's 1..30 hPa levels (scripts/data/add_era5_upper_levels.py, public
# ARCO-ERA5 37-level): the 13-level file stops at 50 hPa and the vertical interpolation holds
# 50 hPa above it, so the L32 top layers (3.6/7.6 hPa) started ~40/20 K cold.  The old
# 13-level IC: ERA5_IC=${DATA}/era5_ic_1979-01-01_gproxy.zarr.
: "${ERA5_IC:=/burg-archive/glab/users/pg2328/legoESM/data/amip_ic_upper/era5_ic_1979-01-01_gproxy_L21.zarr}"
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
# siconcbcs clipped to [0,100]% (raw input4MIPs bcs has non-physical land/edge overshoots
# to ~+/-2500% from Taylor mid-month reconstruction; the loader's pre-clip guard rejects them).
# Make it with: ds['siconcbcs'].clip(0,100) — see data prep notes.
: "${SIC_FILE:=${DATA}/siconcbcs_PCMDI-AMIP-1-1-9_gn_187001-202212_clip.nc}"
: "${SST_VAR:=tosbcs}"
: "${SIC_VAR:=siconcbcs}"
# This input4MIPs file ships tosbcs in degC, so override the YAML's sst_offset=0.0 with the
# Celsius->Kelvin offset (the loader's units guard ENFORCES this — it does not auto-convert).
# Set SST_OFFSET=0 only if you swap to a Kelvin SST file.
: "${SST_OFFSET:=273.15}"

# --- external CMIP6 forcing (synthetic deck by default; drop real input4MIPs in) -
# REAL input4MIPs ozone + solar (2026-10-06).  The synthetic deck's
# ozone_amip_clim.nc has no ozone above 5 hPa and solar_amip_1979-2014.nc
# carries 9.7x the reference flux in the Hartley UV band: together they heated
# the 7.6 hPa layer by +3.5 K/day on every lane and NaN'd the duo CAM6 60-day
# run at day 12.  Staged by scripts/data/download_cmip6_forcing.py +
# adapt_cmip6_solar.py (SOLARIS-HEPPA-3-2 monthly -> TSI + 14-band SSI_frac) +
# a zonal-mean concatenation of UReading-CCMI-1-0 vmro3 1850-2014 (the loader
# zonal-averages anyway; it takes ONE ozone path).  Same set for every grid.
: "${CMIP6_FORCING_DIR:=/burg-archive/glab/users/pg2328/legoESM/data/cmip6_forcing}"
: "${SOLAR:=${CMIP6_FORCING_DIR}/solar_cmip6_SOLARIS-HEPPA-3-2_mon.nc}"
: "${OZONE:=${CMIP6_FORCING_DIR}/vmro3_input4MIPs_ozone_CMIP_UReading-CCMI-1-0_zonalmean_185001-201412.nc}"
# Coupled CMIP lane: the canonical decks (config/cmip/cmip_ocean_*.yaml) now
# carry the forcing SCHEME keys (F40) and take only PATHS from CMIP_PATH_FLAGS
# below.  CMIP6_FORCING_FLAGS (schemes + paths) remains for the CLI-only
# probes under scripts/cluster/cmip6_coupled/ that run without a deck.
CMIP6_FORCING_FLAGS=(
  --ozone-source standard --ozone-forcing external --ozone-file "${OZONE}"
  --solar-source spectral_file --solar-file "${SOLAR}"
  --solar-tsi-var TSI --solar-spectral-var SSI_frac --solar-spectral-band-order rrtmg_sw
  --orbital-insolation --diurnal-cycle   # as every AMIP deck (gridaudit 2026-10-09)
)
# REAL GHG / aerosol / volcanic (2026-10-08), same set as Levante's ICON pool
# files.  The synthetic aerosol_amip_clim.nc (four hand-picked constants, no
# longitude) set cloud droplets at ~768 cm-3 through aerosol_ccn.  Staging +
# provenance: fv3_duo_gaps/forcing_stage/ (deck_ledgers.md 2026-10-08).
#   GHG: input4MIPs UoM-CMIP-1-2-0 global means incl. CFC-22/CCl4/CF4.
#   AEROSOL: Kinne MACv1 fine+coarse (radiation); AEROSOL_CCN: fine mode (CCN).
#   VOLCANIC: ETH CMIP6 stratospheric aerosol v4.0.0, ECHAM6 bands (SW + LW).
: "${GHG:=${CMIP6_FORCING_DIR}/ghg_cmip6_UoM-CMIP-1-2-0_halogens_0000-2014.nc}"
: "${AEROSOL:=${CMIP6_FORCING_DIR}/aeropt_kinne_sw_b14_tot_1979-2014_MACv1.nc}"
: "${AEROSOL_CCN:=${CMIP6_FORCING_DIR}/aeropt_kinne_sw_b14_fin_1979-2014_MACv1.nc}"
: "${VOLCANIC:=${CMIP6_FORCING_DIR}/bc_aeropt_cmip6_volc_lw_b16_sw_b14_1979-2014_IACETH-v4.nc}"
CMIP6_FORCING_FLAGS+=(
  --ghg-forcing external --ghg-file "${GHG}"
  --aerosol-forcing external --aerosol-file "${AEROSOL}" --aerosol-ccn-file "${AEROSOL_CCN}"
  --volcanic-aerosol-file "${VOLCANIC}"
)
# CAM RRTMG-style transported layer above the model top (ozone above p_top
# absorbs; ~2 W/m2 leaves the column budget). The CAM6 AMIP decks set it in
# YAML; the coupled lane has no YAML, so it rides here.
CMIP6_FORCING_FLAGS+=( --rrtmgp-overhead-layer )

# --- CLM surfdata (multilayer Richards land; use_multilayer_land: true) -------
# The production YAML's land needs the CLM surfdata NetCDF staged locally —
# compute nodes have no outbound internet, and the driver's fallback download
# fails there (SSL hostname mismatch on the UCAR svn mirror).
: "${CLM_SURFDATA:=${DATA%/amip}/clm/surfdata_1.9x2.5_16pfts_CMIP6_simyr2000.nc}"

# --- harmonized surfdata (CANOPY parameters) ----------------------------------
# Two different datasets with confusingly similar names. The CLM surfdata above
# carries soil texture and plant-type cover; this one carries canopy structure
# (canopy height, roughness ratio, Vcmax25, band albedos). The default land
# surface scheme is the two-leaf canopy, which REFUSES to run without it rather
# than fall back to generic constants, so a canopy run needs both staged. Twin
# of the same block in config/amip/amip_production.sh.
: "${AMIP_SURFDATA:=${REPO}/data/legoesm_surfdata_c260716.nc}"

# --- subgrid orography for the orographic GWD launch (#1514) -----------------
# amip_production.yaml runs an orographic gravity-wave member.  With
# subgrid_orography_path empty the orographic member launches tau_0 ~ h_topo^2
# from the SCALAR fallback h_topo = 500 m on EVERY column -- a fictional 500 m
# mountain over the open ocean, measured at -0.29 Pa of spurious zonal drag
# over 40-60S, which removes the eddy-driven westerly belt.  The Levante twin
# has wired this since #1514; this launcher did NOT, so every AMIP run started
# here reproduced the pseudo-mountain climate.
# NB the `:=` below matches the Levante twin, and `:=` substitutes on EMPTY as
# well as unset -- so `AMIP_SSO=""` does NOT opt out, it re-selects the default
# (codex review).  Opting out needs `AMIP_SSO=/dev/null`-style explicitness or
# a `-` in place of `:-`; both launchers share this, and it is left as-is here
# rather than silently changing one of them.
# CONSTRUCTION MATCHES THE LEVANTE TWIN (--fine-res-deg 1.0 --block-deg 2.0),
# deliberately, and NOT the finer 0.25-deg-source file that is also staged here.
# Measured 2026-09-04: block size swings the Southern-Ocean launch stress ~33x
# across plausible choices (0.5/1/2/4 deg -> 0.06/0.31/1.00/1.99 relative drag)
# while the SOURCE choice swings it only 2.3x, and nothing in the loader ties
# the block size to the model grid.  So the source is a small effect inside a
# much larger unanchored knob; matching the twin removes a cross-machine
# confound without pretending to have settled the decomposition.
# Acceptance on regeneration: Southern Ocean 40-60S mean sgh 3.968 m against
# the twin's recorded 3.6 m (the residual is the source, our ETOPO regridded to
# 1 deg vs their etopo_1deg_clean), i.e. 1.21x in drag where the previous file
# was 2.31x.
: "${AMIP_SSO:=/burg-archive/glab/users/pg2328/legoESM_chunk/data_pg/amip/sso_stdh_2deg_from1deg.nc}"

# --- machine-specific PATH flags (everything else is in the YAML) -------------
AMIP_PATH_FLAGS=(
  --ic-path "${ERA5_IC}"
  --topography "${ETOPO}"
  --surfdata "${AMIP_SURFDATA}"
  --clm-surfdata-path "${CLM_SURFDATA}"
  --forcing-path "${SST_FILE}" --sst-var "${SST_VAR}" --sst-offset "${SST_OFFSET}"
  --sic-path "${SIC_FILE}"     --sic-var "${SIC_VAR}"
  --solar-file "${SOLAR}"
  --ozone-file "${OZONE}"
  --ghg-file "${GHG}"
  --aerosol-file "${AEROSOL}"
  --aerosol-ccn-file "${AEROSOL_CCN}"
  --volcanic-aerosol-file "${VOLCANIC}"
)
# Volcanic LW (ext_earth) on, as the Levante twin; AMIP_VOLCANIC_LW=0 opts out
# for EVERY lane (AMIP and the coupled CMIP6_FORCING_FLAGS) -- one policy.
if [[ "${AMIP_VOLCANIC_LW:-1}" == "1" ]]; then
  AMIP_PATH_FLAGS+=( --volcanic-aerosol-lw )
  CMIP6_FORCING_FLAGS+=( --volcanic-aerosol-lw )
fi
# Subgrid orography (see AMIP_SSO above, #1514).  Appended like the Levante
# twin so an empty AMIP_SSO is an explicit opt-out rather than a silent drop.
if [[ -n "${AMIP_SSO}" ]]; then
  AMIP_PATH_FLAGS+=( --subgrid-orography-file "${AMIP_SSO}" )
fi

# Coupled CMIP decks (config/cmip/cmip_ocean_*.yaml): machine PATHS only --
# the decks select the forcing schemes (F40).  Terrain + subgrid orography are
# the AMIP lane's (F36); the AMIP_VOLCANIC_LW / AMIP_SSO opt-outs apply as above.
CMIP_PATH_FLAGS=(
  --ozone-file "${OZONE}" --solar-file "${SOLAR}" --ghg-file "${GHG}"
  --aerosol-file "${AEROSOL}" --aerosol-ccn-file "${AEROSOL_CCN}"
  --volcanic-aerosol-file "${VOLCANIC}"
  --topography "${ETOPO}"
)
if [[ "${AMIP_VOLCANIC_LW:-1}" == "1" ]]; then
  CMIP_PATH_FLAGS+=( --volcanic-aerosol-lw )
fi
if [[ -n "${AMIP_SSO}" ]]; then
  CMIP_PATH_FLAGS+=( --subgrid-orography-file "${AMIP_SSO}" )
fi

# Warn (don't fail) on a missing local input so a Stage-0 flat-topo smoke still works.
for _f in "${ETOPO}" "${CLM_SURFDATA}" "${AMIP_SURFDATA}" ${AMIP_SSO:+"${AMIP_SSO}"}; do
  [ -e "${_f}" ] || echo "[ginsburg] NOTE: ${_f} not found — stage it (ETOPO: " \
    "scripts/data/prep_etopo_topography.py; CLM surfdata: data/clm/; harmonized " \
    "surfdata: scripts/data/build_legoesm_surfdata.py), or run a flat-topo " \
    "smoke (--topography flat).  A missing harmonized surfdata is FATAL for " \
    "the default two-leaf canopy, not a warning." >&2
done
export PY AMIP_PATH_FLAGS
