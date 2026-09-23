#!/usr/bin/env bash
# Paired 5-day cloud-cover arms (2026-09-08): every arm restarts from a COPY of
# rhebc90_r6's day-80 checkpoint (already placed in each arm dir), runs the
# production res6 deck from THIS worktree, and stops at day 86.  One variable
# per arm; cc_ctl is the twin (must reproduce rhebc90_r6 days 81-86).
# Score: days 81-85 window means vs cc_ctl (rsut, rlut, clt, layer cover).
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
declare -A ARMS=(
  [cc_ctl]=""
  [cc_cond3e5]="--cloud-cover-condensate-q-ref 3e-5"
  [cc_cond1e4]="--cloud-cover-condensate-q-ref 1e-4"
  [cc_cond1e5]="--cloud-cover-condensate-q-ref 1e-5"
  [cc_xr]="--clouds xu_randall"
  [cc_smix]="--cloud-saturation-scheme mixed_phase"
  [cc_xrmix]="--clouds xu_randall --cloud-saturation-scheme mixed_phase"
)
for name in "${!ARMS[@]}"; do
  [[ -f ${ROOT}/${name}/checkpoint_day_0080.npz ]] || { echo "missing day-80 checkpoint in ${name}" >&2; exit 2; }
  jid=$(sbatch --account=bd1083 --job-name="${name}" --time=06:00:00 \
        --export="ALL,REPO=${WT},CONFIG_YAML=${BASELINE_DECK},PY=${MAIN}/.venv/bin/python,NAME=${name},TARGET_DAYS=86,MAX_WALLCLOCK_SECONDS=19800,EXTRA=${BASE} ${ARMS[$name]}" \
        "${WT}/scripts/cluster/levante/amip_mpas_gpu_chain.sbatch" | awk '{print $NF}')
  echo "${name} ${jid} EXTRA='${BASE} ${ARMS[$name]}'"
done
