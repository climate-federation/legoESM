#!/usr/bin/env bash
# Full-atmosphere-physics AMIP: MPAS + lat-lon, multilayer land on both.
# User directive 2026-07-30: "all parameterization activated in the atmosphere,
# bechtold for convection, multilayer land for both MPAS and lat lon".
#
# KEY FINDING while building this: config/amip/amip_production*.yaml ALREADY
# specify the full deck ("SOTA multilayer Richards soil + CLM PFT/texture +
# prognostic snow, tiled COARE3/MOST surface fluxes").  The MPAS and lat-lon
# chains had been OVERRIDING it off with --no-use-multilayer-land
# --no-surface-tiled.  So the main change here is DROPPING those two negations
# and letting the production YAML's intended physics run, plus the additional
# realism switches below.
#
# Deliberately LEFT OFF, each with a verified reason (do not "fix" these):
#   --convective-cloud     run_amip help: "recommended for prescribed-SST AMIP,
#                          where conv-cloud is an inert SST-drift compensator
#                          that only adds planetary albedo".
#   --land-stomatal-beta   amip_production.yaml #741: on the multilayer soil it
#                          over-transpires (beta stays near-potential in a moist
#                          root zone), driving an over-evaporation drift; off,
#                          the multilayer uses the realistic beta_soil path.
#   --moisture-advection   cube-lane only, experimental (#771).
#   --held-suarez-forcing, --slab-land-active, physics_parameterization,
#                          carbon_cycle: idealized / superseded / AMIP stub.
#
# RISK, stated up front: no spun-up land IC exists, so the multilayer soil
# COLD-STARTS.  run_amip's own help warns of "the day-0 cold-start shock behind
# the land cold trap"; the YAML mitigates with land_soil_moisture_init_frac=0.25
# (#730).  Watch land hfls/tas over the first ~20 days.
set -euo pipefail

REPO=/work/bd1083/b309178/diffESM/legoesm_pg/legoESM
# BASELINE DECK for the MPAS arm, pinned 2026-09-23.  This script was built to
# the 2026-07-30 directive that names Bechtold convection, and its MPAS arm
# inherited the chain's CONFIG_YAML default, which became the CAM6 suite.  It
# now names the configuration it was measured on.  ARM-SCOPED rather than a
# script-global export (GLM round 3): a global export would also reach any arm
# added later that is meant to follow production, which is the same class of
# silent re-pointing this pin exists to prevent.  The lat-lon arm below sets
# its own CONFIG_YAML and is untouched.
# CONDITIONAL on purpose: the rename has not reached every checkout, and in a
# tree that predates it the production deck IS the right baseline.
BASELINE_DECK="${REPO}/config/amip/amip_sundqvist_l36.yaml"
[[ -f "${BASELINE_DECK}" ]] || BASELINE_DECK="${REPO}/config/amip/amip_production.yaml"

: "${TARGET_DAYS:=1825}"      # 5 yr first; extend once stable

# Physics switches shared by BOTH grids.
SHARED="--convection bechtold --gravity-wave-drag mcfarlane+hines \
--use-multilayer-land --surface-tiled \
--subgrid-autoconversion --dynamic-albedo --orbital-insolation \
--volcanic-aerosol-lw --bechtold-downdraft-transport \
--homogeneous-ice-nucleation --hard-saturation-adjustment --aerosol-ccn \
--no-convective-cloud --sst-offset 273.15 --start-year 1923 \
--checkpoint-days 10 --diag-days 1"

# --- MPAS lane -------------------------------------------------------------
# MPAS-only flags: --hard-sat-ice-curve and every --mpas-* knob.  The lat-lon
# lane REFUSES --hard-sat-ice-curve ("requires an MPAS grid") — that is what
# killed the earlier lat-lon attempt, so it stays out of the lat-lon EXTRA.
# --mpas-land-beta-soil requires --use-multilayer-land: it threads the soil's
# own per-cell root-zone beta into the turbulence surface humidity, REPLACING
# the static --mpas-land-beta 0.6 over land.
MPAS_EXTRA="--grid-type voronoi --discretization mpas --resolution 5 --nlev 30 \
--dt 75 --vertical-coord sigma --rad-update-steps 48 \
--hard-sat-ice-curve --mpas-ice-skin-prognostic \
--mpas-conservative-tracer-clamp --mpas-land-beta-soil \
--mpas-land-lapse-k-per-km 6.5 --mpas-land-beta 0.6 \
--mpas-qv-smooth-del2-m2s 2e5 ${SHARED}"

NAME=fullphys_mpas TARGET_DAYS="${TARGET_DAYS}" CENTURY_DECK=1 START_YEAR=1923 \
CONFIG_YAML="${BASELINE_DECK}" \
EXTRA="${MPAS_EXTRA}" \
sbatch --job-name=fullphys_mpas --time=08:00:00 \
       "${REPO}/scripts/cluster/levante/amip_mpas_gpu_chain.sbatch"

# --- lat-lon lane ----------------------------------------------------------
# CFL at the poles: the latlon24 YAML carries the polar filter; keep its dt.
NAME=fullphys_latlon TARGET_DAYS="${TARGET_DAYS}" CENTURY_DECK=1 START_YEAR=1923 \
CONFIG_YAML="${REPO}/config/amip/amip_production_latlon24.yaml" \
EXTRA="${SHARED}" \
sbatch --job-name=fullphys_latlon --time=08:00:00 \
       "${REPO}/scripts/cluster/levante/amip_mpas_gpu_chain.sbatch"

echo "submitted fullphys_mpas + fullphys_latlon to absolute day ${TARGET_DAYS}"
