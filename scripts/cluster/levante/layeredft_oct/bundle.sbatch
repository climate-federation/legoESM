#!/usr/bin/env bash
#SBATCH --account=bb1596
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --constraint=a100_80
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=/work/bd1083/b309178/diffESM/legoesm_pg/amip_runs/slurm_logs/%x-%j.out
# Several 4-GPU AMIP runs in ONE allocation, one per node; each arm's repo is
# arm.tsv column 3 (arms may share a repo: the chain never writes into it), so
# arms (branch) and controls (base commit) can share a job.
# Stage 1 runs every arm ONE day from its restart copy and applies the day-1
# gate (check_day1.py); only if every arm passes does stage 2 continue the same
# arms, in the same allocation, to their arm.tsv target.
# ORDER: launch the layered arms; their controls are snow-heat's sh2_off_a/b
# (valid only after its 1-day gate passes, and PREREG's resolved-config diff)
# or lf_ctl_a/b:
#   sbatch --nodes=2 --job-name=layeredft --export=ALL,ARMS="lf_on_a lf_on_b" bundle.sbatch
# RESUME (an arm stopped by its wall clock exits 75 and queues nothing):
#   sbatch --nodes=<n> --job-name=layeredft --export=ALL,ARMS="<the same arms>",DAY1_GATE=0 bundle.sbatch
# Each arm restarts from its own newest checkpoint on its own node.
set -uo pipefail
: "${CHAIN:=/work/bd1083/b309178/diffESM/legoesm_pg/wt_layeredft_run/scripts/cluster/levante/amip_mpas_gpu_chain.sbatch}"
[[ -f "${CHAIN}" ]] && grep -q AMIP_STEP_NODE "${CHAIN}" || { echo "REFUSING: ${CHAIN} cannot pin arms to nodes" >&2; exit 64; }
: "${ROOT:=/work/bd1083/b309178/diffESM/legoesm_pg/amip_runs}"
: "${PY:=/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
: "${AMIP_SURFDATA:=/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/data/legoesm_surfdata_c260716.nc}"
: "${ERA5_IC:=/scratch/b/b381103/era5_ic_2001-01-01.zarr}"
COMMON="--start-year 2001 --resolution 6 --checkpoint-days 1 --subgrid-orography-file /work/bd1083/b309178/diffESM/land_data/extpar_sso_latlon_0p25deg.nc --land-ic /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/data/lmip_soil_ic/soil_ic_mpas6.npz"
: "${DAY1_GATE:=1}"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ -f "${HERE}/check_day1.py" ]] || HERE=/work/bd1083/b309178/diffESM/legoesm_pg/wt_layeredft_run/scripts/cluster/levante/layeredft_oct
mkdir -p "${ROOT}/slurm_logs"
read -r -a arms <<< "${ARMS:?set ARMS}"
mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
(( ${#nodes[@]} == ${#arms[@]} )) || { echo "REFUSING: ${#arms[@]} arms on ${#nodes[@]} nodes" >&2; exit 64; }
[[ $(printf '%s\n' "${arms[@]}" | sort | uniq -d) ]] && { echo "REFUSING: duplicate arm names" >&2; exit 64; }
for a in "${arms[@]}"; do
  [[ -s "${ROOT}/${a}/arm.tsv" ]] || { echo "REFUSING: ${ROOT}/${a}/arm.tsv missing" >&2; exit 64; }
  IFS=$'\t' read -r _n _t _r _c _e < "${ROOT}/${a}/arm.tsv"
  [[ "${_n}" == "${a}" && "${_t}" =~ ^[0-9]+$ && "${_r}" == /* && -e "${_r}/.git" ]] \
    || { echo "REFUSING: ${a}/arm.tsv names '${_n}' target '${_t}' repo '${_r}'" >&2; exit 64; }
  # Frozen code: the repo must still be clean (tracked, and no untracked code)
  # and at the commit setup recorded; the restart copy must be the one setup hashed.
  [[ "$(git -c safe.directory='*' -C "${_r}" rev-parse HEAD)" == "${_c}" && \
     -z "$(git -c safe.directory='*' -C "${_r}" status --porcelain -- packages src scripts config)" ]] \
    || { echo "REFUSING: ${_r} moved or is dirty since setup (want ${_c})" >&2; exit 64; }
  ( cd "${ROOT}/${a}" && sha256sum --quiet -c restart.sha256 ) \
    || { echo "REFUSING: ${a} restart checkpoint missing or changed since setup" >&2; exit 64; }
done
export ROOT PY AMIP_SURFDATA ERA5_IC AMIP_N_GPUS=4
newest_day() {   # newest checkpoint day of a run dir (the restart copy at first)
  ls -1 "$1"/checkpoint_day_*.npz 2>/dev/null | sort -V | tail -1 \
    | sed -E 's/.*checkpoint_day_0*([0-9]+)\.npz/\1/'
}
run_stage() {   # $1 = stage label; TGT[name] = target day of each arm
  local -A pid; local i name target repo commit extra rc fail=0
  for i in "${!arms[@]}"; do
    IFS=$'\t' read -r name target repo commit extra < "${ROOT}/${arms[i]}/arm.tsv"
    echo "[bundle] $1 ${name} node=${nodes[i]} repo=${repo} commit=$(git -c safe.directory='*' -C "${repo}" rev-parse HEAD) target=${TGT[${name}]}"
    ( REPO="${repo}" NAME="${name}" TARGET_DAYS="${TGT[${name}]}" EXTRA="${COMMON}${extra}" \
      MAX_WALLCLOCK_SECONDS="${ARM_WALLCLOCK_SECONDS:-39600}" AMIP_STEP_NODE="${nodes[i]}" \
      bash "${CHAIN}" \
      >> "${ROOT}/slurm_logs/${name}-${SLURM_JOB_ID}.out" 2>&1 ) &
    pid[${name}]=$!
  done
  for name in "${!pid[@]}"; do
    wait "${pid[${name}]}"; rc=$?
    echo "[bundle] $1 ${name} exit=${rc} node-log ${ROOT}/slurm_logs/${name}-${SLURM_JOB_ID}.out"
    (( rc == 0 )) || fail=1
  done
  return "${fail}"
}
declare -A TGT START
for a in "${arms[@]}"; do
  IFS=$'\t' read -r _n _t _r _c _e < "${ROOT}/${a}/arm.tsv"
  START[${a}]=$(newest_day "${ROOT}/${a}")
  TGT[${a}]=${_t}
done
if (( DAY1_GATE )); then
  for a in "${arms[@]}"; do TGT[${a}]=$(( START[${a}] + 1 )); done
  run_stage day1 || { echo "[bundle] day-1 stage failed" >&2; exit 1; }
  gate=0
  for a in "${arms[@]}"; do
    IFS=$'\t' read -r _n _t _r _c _e < "${ROOT}/${a}/arm.tsv"
    scheme=bulk; [[ "${_e}" == *"--land-snow-scheme layered"* ]] && scheme=layered
    "${PY}" "${HERE}/check_day1.py" "${ROOT}/${a}" "${START[${a}]}" "${scheme}" \
      "${ROOT}/slurm_logs/${a}-${SLURM_JOB_ID}.out" || gate=1
    TGT[${a}]=${_t}
  done
  (( gate == 0 )) || { echo "[bundle] DAY-1 GATE FAILED: arms stop at day 1" >&2; exit 65; }
fi
run_stage full
