#!/usr/bin/env bash
# Three-arm stratocumulus factorial (2026-09-13): does the boundary-layer
# closure explain the collapsed Californian deck?
#
# THE DEFECT. Scored against CERES, the three Sc decks disagree in SIGN under
# one closure: Peru is correct (-2.3 W/m2, cover +0.8), Namibia is too cloudy
# (+18.9, +26.3) and California has COLLAPSED (-35.9, -43.5) with a liquid
# water path of 0.01 g/m2 against Peru's 54.8 -- no condensate at all. Its
# lowest levels are DRIER at the surface (70.8% RH vs Peru's 85.8%) and MOISTER
# aloft, i.e. mixed rather than stratified: the inversion is not being held, so
# the column never reaches the cloud scheme's threshold. PLAUSIBLE, not
# confirmed -- that is what this measures.
#
# WHY THESE THREE ARMS. A single CLUBB arm cannot separate the closure change
# from the prognostic higher-order moments, so the factorial splits them. The
# deck supplies everything else identically, including the zenith ocean albedo,
# so each arm differs from its neighbour in ONE field.
#
# PRE-REGISTERED, before any number exists:
#   measures  -- Sc California low cover and reflected SW, days 81-86, per deck.
#   CONFIRMS  -- California cover rises well clear of 10% toward the observed
#                ~48%, and its -35.9 W/m2 shortwave deficit shrinks, while Peru
#                does not degrade.
#   REFUTES   -- California stays under ~10% cover in BOTH CLUBB arms; the
#                closure is then not what ventilates the deck and the next
#                suspect is the surface flux or the vertical resolution across
#                the inversion.
#   why not offline -- the deck is a multi-day equilibrium between surface
#                flux, cloud-top entrainment and radiative cooling, maintained
#                against the run's own subsidence. A single-column call cannot
#                reproduce it.
#
# Every arm restarts from a COPY of cc_ctl's day-80 checkpoint, already placed
# in each arm dir (never a newest-wins pointer: two arms advancing at once make
# "latest" ambiguous).
set -euo pipefail
WT=/work/bd1083/b309178/diffESM/legoesm_pg/wt_cloudreview
MAIN=/work/bd1083/b309178/diffESM/legoesm_pg/legoESM
ROOT=/work/bd1083/b309178/diffESM/legoesm_pg/amip_runs
BASE="--resolution 6 --checkpoint-days 10 \
--subgrid-orography-file /work/bd1083/b309178/diffESM/land_data/extpar_sso_latlon_0p25deg.nc \
--params ${ROOT}/_tools/cc_params_as_rhebc90_r6.yaml \
--land-ic ${MAIN}/data/lmip_soil_ic/soil_ic_mpas6.npz"
# The deck now selects CLUBB with prognostic moments, so the CONTROL states its
# own turbulence explicitly rather than inheriting it -- a control that depends
# on the deck not changing under it is not a control.
declare -A ARMS=(
  [sc_louis]="--turbulence louis --no-clubb-prognostic"
  [sc_clubbd]="--turbulence clubb --no-clubb-prognostic"
  [sc_clubbp]="--turbulence clubb --clubb-prognostic"
)
for name in "${!ARMS[@]}"; do
  [[ -f ${ROOT}/${name}/checkpoint_day_0080.npz ]] || { echo "missing day-80 checkpoint in ${name}" >&2; exit 2; }
  jid=$(sbatch --account=bd1083 --job-name="${name}" --time=06:00:00 \
        --export="ALL,REPO=${WT},PY=${MAIN}/.venv/bin/python,NAME=${name},TARGET_DAYS=86,MAX_WALLCLOCK_SECONDS=19800,EXTRA=${BASE} ${ARMS[$name]}" \
        "${WT}/scripts/cluster/levante/amip_mpas_gpu_chain.sbatch" | awk '{print $NF}')
  echo "${name} ${jid} EXTRA='${BASE} ${ARMS[$name]}'"
done
