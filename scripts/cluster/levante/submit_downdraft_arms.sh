#!/usr/bin/env bash
# Penetrative-downdraft ventilation pair (2026-09-14): does the missing
# sub-cloud DRYING term close the near-surface moist bias?
#
# THE MEASURED CHAIN this tests, all against ERA5/CERES:
#   tropical air at 1000 hPa is 11 RH points too moist (17.56 vs 15.32 g/kg)
#   while its temperature is right, so the sea-air humidity deficit that drives
#   evaporation is only 61% of observed; tropical-ocean evaporation measures
#   0.87 of ERA5; wind is 1.31x ERA5, and 1.31 x 0.61 = 0.80 against 0.87
#   measured, so humidity and wind account for the whole evaporation shortfall.
#   The lowest layer is ~300 m thick and a typical latent flux fills the entire
#   2.24 g/kg excess in under 5 hours -- so the excess is set by REMOVAL being
#   too slow, not by supply.
#
# THE TERM. The convection scheme's downdraft branch does ONLY rain
# re-evaporation, which MOISTENS the sub-cloud layer. The Tiedtke-1989
# penetrative downdraft TRANSPORT -- which advects low-moist-static-energy air
# down from the level of free sinking and DRIES the sub-cloud layer -- is
# implemented, flux-form conservative, and DEFAULT OFF. Its own code comment
# predicts this exact chain: drier sub-cloud layer -> larger sea-air gradient ->
# stronger evaporation -> less boundary-layer cloud -> lower albedo.
#
# PRE-REGISTERED (GLM-5.2, before any number exists):
#   near-surface humidity  -1 to -1.5 g/kg (RH -3 to -7 points)
#   ocean evaporation      0.87 -> 0.92-0.95 of ERA5
#   low cloud fraction     down 5-15%
#   reflected shortwave    down 3-7 W/m2 of the +13
#   precipitation          tropical mean up 0.1-0.2 mm/day, locally down
#   CONFIRMS  -- humidity and shortwave both fall within those bands
#   REFUTES   -- near-surface humidity moves less than 0.3 g/kg, i.e. the term
#                is too weak to matter at alpha=0.3
#   watch for -- over-drying at trade-cumulus margins and over tropical land,
#                and a degenerate origin level where the source air is already
#                saturated (measured NOT to be the case: the level of minimum
#                moist static energy sits at 739 hPa with RH 0.70 and 8.77 g/kg
#                against a boundary layer at 18.37, a 9.6 g/kg contrast).
#   why not offline -- the bias is an equilibrium between surface supply and
#                sub-cloud removal; a single-column call cannot reproduce the
#                convective mass flux that drives the downdraft.
#
# Both arms run the CLUBB diagnostic closure, matching the current production
# deck, so the result transfers. Prognostic CLUBB is NOT used: the shared
# day-80 checkpoint cannot seed its moments (see sc_clubbp's note).
set -euo pipefail
WT=/work/bd1083/b309178/diffESM/legoesm_pg/wt_cloudreview
# BASELINE DECK, pinned 2026-09-23.  config/amip/amip_production.yaml became the
# CAM6 suite on that date and the GPU chain's CONFIG_YAML default follows it.
# This script is a one-variable arm comparison whose baseline is the
# configuration it was measured against, so it names that configuration
# explicitly instead of inheriting whatever production becomes.
# CONDITIONAL on purpose (GLM round 3): the rename has not reached every
# checkout, and in a tree that still predates it the production deck IS the
# right baseline.  So prefer the renamed deck where it exists and fall back to
# production where it does not - correct on both sides of the merge, and it
# never breaks the script the way a hard pin would.
BASELINE_DECK="${WT}/config/amip/amip_sundqvist_l36.yaml"
[[ -f "${BASELINE_DECK}" ]] || BASELINE_DECK="${WT}/config/amip/amip_production.yaml"

MAIN=/work/bd1083/b309178/diffESM/legoesm_pg/legoESM
ROOT=/work/bd1083/b309178/diffESM/legoesm_pg/amip_runs
BASE="--resolution 6 --checkpoint-days 10 \
--subgrid-orography-file /work/bd1083/b309178/diffESM/land_data/extpar_sso_latlon_0p25deg.nc \
--params ${ROOT}/_tools/cc_params_as_rhebc90_r6.yaml \
--land-ic ${MAIN}/data/lmip_soil_ic/soil_ic_mpas6.npz \
--turbulence clubb --no-clubb-prognostic"
declare -A ARMS=(
  [dd_ctl]=""
  [dd_on]="--bechtold-downdraft-transport"
)
for name in "${!ARMS[@]}"; do
  [[ -f ${ROOT}/${name}/checkpoint_day_0080.npz ]] || { echo "missing day-80 checkpoint in ${name}" >&2; exit 2; }
  jid=$(sbatch --account=bd1083 --job-name="${name}" --time=12:00:00 \
        --export="ALL,REPO=${WT},CONFIG_YAML=${BASELINE_DECK},PY=${MAIN}/.venv/bin/python,NAME=${name},TARGET_DAYS=110,MAX_WALLCLOCK_SECONDS=41400,EXTRA=${BASE} ${ARMS[$name]}" \
        "${WT}/scripts/cluster/levante/amip_mpas_gpu_chain.sbatch" | awk '{print $NF}')
  echo "${name} ${jid}"
done
