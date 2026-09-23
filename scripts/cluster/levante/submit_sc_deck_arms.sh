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
#
# ONE DEVIATION, recorded because the arms are then not byte-identical at day 80:
# the checkpoint was written by a Louis run, so its CLUBB moment slot holds the
# minimal (ncol,1,1) placeholder while prognostic CLUBB needs (ncol,15,nlev+1).
# The driver REFUSES to start rather than silently reseed a prognostic field
# mid-restart, which is correct -- that would branch the trajectory without
# saying so, and it is what killed the first launch of sc_clubbp. The fix is to
# drop ONLY physstate_clubb_moments from that arm's copy, so the reseed is
# confined to the moments and every other carry still resumes. See
# sc_clubbp/WHY_THIS_CHECKPOINT_DIFFERS.txt. Cost: that arm spins its moments up
# over the first hours, so it is not bit-comparable at day 80 the way the other
# two are to each other. Boundary-layer moments equilibrate in hours, so the
# 6-day window absorbs it.
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
        --export="ALL,REPO=${WT},CONFIG_YAML=${BASELINE_DECK},PY=${MAIN}/.venv/bin/python,NAME=${name},TARGET_DAYS=86,MAX_WALLCLOCK_SECONDS=19800,EXTRA=${BASE} ${ARMS[$name]}" \
        "${WT}/scripts/cluster/levante/amip_mpas_gpu_chain.sbatch" | awk '{print $NF}')
  echo "${name} ${jid} EXTRA='${BASE} ${ARMS[$name]}'"
done
