#!/usr/bin/env bash
# =============================================================================
# legoESM AMIP — AUTHORITATIVE production configuration (single source of truth)
# =============================================================================
# Purpose: ONE canonical parameter set so every AMIP run uses the *same*
# configuration. Source this from an sbatch and pass "${AMIP_FLAGS[@]}" to
# scripts/run/run_amip.py. Do NOT fork per-run physics choices — change this
# file (under review) instead, so all runs stay comparable.
#
#   source config/amip/amip_production.sh
#   "${PY}" -u scripts/run/run_amip.py "${AMIP_FLAGS[@]}" \
#       --days "${DAYS}" --output "${OUTDIR}"
#
# Provenance: derived from Pierre's tuned CMIP-realism config
# config/cmip/cmip_ocean_slab.yaml (SBM + Morrison + Sundqvist + COARE3 +
# gustiness + q_c_diagnostic 3e-4), ported to prescribed-SST AMIP. VALIDATED at
# C48/L40 (job 25918469, 10-day): planetary albedo 0.292 (target ~0.29), rsut 99,
# precip 3.2 mm/d (target ~2.8), rlut 234, net TOA +6 — i.e. SBM fixes the
# structural over-bright (~2x) and too-dry (~4x) biases that Bechtold did NOT
# (Bechtold gave albedo 0.57 / precip 0.81). Convection=SBM confirmed by this run
# AND by Pierre's authoritative slab config. Land kept as our tiled COARE3/MOST +
# soil bucket + Jarvis stomatal (Pierre-directed for AMIP, 2026-06-24/25); Louis
# turbulence (required by the tiled surface). Residual: tas ~4 K cold at 10 d
# (spin-up; Pierre's slab equilibrated to 287.9 K over 90 d).
#
# OPEN ITEMS — need Pierre/Veronika to finalize (do NOT silently diverge):
#   * RUNNER: this is the run_amip.py (prescribed-SST) path. Pierre's tuned config
#     is a run_coupled.py --config YAML (cmip_ocean_slab.yaml). Agree on one.
#   * LAND: this uses our tiled COARE3/MOST + bucket + stomatal tile. Pierre's
#     config uses slab_richards (multilayer Richards soil). The simplified AMIP
#     land tile has NO snow-albedo feedback (snow tracked thermodynamically but
#     radiatively invisible); snow_albedo_feedback lives in the full land model,
#     so it comes bundled with the land-model choice, NOT a standalone flag.
#   * convective_cloud: Pierre's config sets it true; the validated AMIP run hit
#     targets WITHOUT it, so it is OFF here. Now exposed via --convective-cloud
#     (run_amip.py) to enable for parity experiments.
#   * q_c_diagnostic = 3e-4 (Pierre's tuned value); rh_crit left at the calibrated
#     default 0.77 (commit 3f376730e), NOT overridden.
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

# --- The canonical flag set (everything except --days / --output) ------------
AMIP_FLAGS=(
  # Grid + dycore: cubed-sphere C48 / L40 hybrid, cd-grid, dt 150 s
  --grid-type cubed_sphere --resolution 48 --nlev 40
  --discretization cdgrid --dt 150
  # Atmospheric physics (SBM convection = the structural-bias fix)
  --radiation rrtmg --rad-update-steps 12      # RRTMGP (rrtmg = rrtmgp builder alias)
  --clouds sundqvist
  --microphysics morrison
  --convection sbm                             # confirmed: fixes albedo+precip (job 25918469)
  --turbulence louis                           # Louis required by the tiled surface
  --gravity-wave-drag hines
  --diurnal-cycle
  # Cloud tuning (rh_crit left at calibrated default 0.77 — NOT overridden)
  --q-c-diagnostic 3e-4                        # Pierre's tuned "thinner cloud" value
  # Tiled surface + interactive land (COARE3 ocean / land-MOST + soil bucket + Jarvis)
  --slab-land-active
  --surface-bulk-scheme coare3 --gustiness-zi 300
  --surface-tiled --surface-z0-land 0.1
  --land-soil-bucket --land-stomatal-beta      # bucket defaults W_max 150 / beta_min 0.1 / w_init 0.5
  # Forcing: prescribed observed SST/SIC + full CMIP6 external forcing
  --ic era5 --ic-path "${ERA5_IC}"
  --dataset custom
  --forcing-path "${SST}" --sst-var tosbcs --sst-offset 0.0
  --sic-path "${SIC}" --sic-var siconcbcs --sic-scale 0.01
  --solar-source file --solar-file "${SOLAR}" --solar-tsi-var TSI
  --ozone-forcing external --ozone-file "${OZONE}"
  --ghg-forcing external --ghg-file "${GHG}"
  --aerosol-forcing external --aerosol-file "${AEROSOL}"
  --volcanic-aerosol-file "${VOLCANIC}"
  --topography "${ETOPO}"
  --start-year 1979
  # Diagnostics / output (CMIP6-protocol CMOR)
  --diag-days 5 --checkpoint-days 30
  --monthly-means --cmip-output --clear-sky-diag
)
