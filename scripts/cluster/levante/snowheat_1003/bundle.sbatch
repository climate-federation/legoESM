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
# Several 4-GPU AMIP runs in ONE allocation, one per node, each with its OWN
# absolute repo (arm.tsv column 3; extra flags last) so arms (branch) and controls (base commit)
# share a job.  Same as campaign_amip_bias_0929/bundle.sbatch otherwise.
#   sbatch --nodes=4 --job-name=snowheat --export=ALL,ARMS="sh_on_a sh_on_b sh_ctl_a sh_ctl_b" bundle.sbatch
set -uo pipefail
: "${ROOT:=/work/bd1083/b309178/diffESM/legoesm_pg/amip_runs}"
: "${PY:=/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
: "${AMIP_SURFDATA:=/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/data/legoesm_surfdata_c260716.nc}"
: "${ERA5_IC:=/scratch/b/b381103/era5_ic_2001-01-01.zarr}"
COMMON="--start-year 2001 --resolution 6 --checkpoint-days 1 --subgrid-orography-file /work/bd1083/b309178/diffESM/land_data/extpar_sso_latlon_0p25deg.nc --land-ic /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/data/lmip_soil_ic/soil_ic_mpas6.npz"
read -r -a arms <<< "${ARMS:?set ARMS}"
mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
(( ${#nodes[@]} == ${#arms[@]} )) || { echo "REFUSING: ${#arms[@]} arms on ${#nodes[@]} nodes" >&2; exit 64; }
[[ $(printf '%s\n' "${arms[@]}" | sort | uniq -d) ]] && { echo "REFUSING: duplicate arm names" >&2; exit 64; }
for a in "${arms[@]}"; do
  [[ -s "${ROOT}/${a}/arm.tsv" ]] || { echo "REFUSING: ${ROOT}/${a}/arm.tsv missing" >&2; exit 64; }
  IFS=$'\t' read -r _n _t _r _e < "${ROOT}/${a}/arm.tsv"
  [[ "${_n}" == "${a}" && "${_t}" =~ ^[0-9]+$ && "${_r}" == /* && -e "${_r}/.git" ]] \
    || { echo "REFUSING: ${a}/arm.tsv names '${_n}' target '${_t}' repo '${_r}'" >&2; exit 64; }
done
export ROOT PY AMIP_SURFDATA ERA5_IC AMIP_N_GPUS=4
declare -A pid
for i in "${!arms[@]}"; do
  IFS=$'\t' read -r name target repo extra < "${ROOT}/${arms[i]}/arm.tsv"
  echo "[bundle] ${name} node=${nodes[i]} repo=${repo} commit=$(git -c safe.directory='*' -C "${repo}" rev-parse HEAD) dirty=$(git -c safe.directory='*' -C "${repo}" status --porcelain --untracked-files=no | wc -l)"
  ( REPO="${repo}" NAME="${name}" TARGET_DAYS="${target}" EXTRA="${COMMON}${extra}" \
    MAX_WALLCLOCK_SECONDS="${ARM_WALLCLOCK_SECONDS:-39600}" AMIP_STEP_NODE="${nodes[i]}" \
    bash "${repo}/scripts/cluster/levante/amip_mpas_gpu_chain.sbatch" \
    > "${ROOT}/slurm_logs/${name}-${SLURM_JOB_ID}.out" 2>&1 ) &
  pid[${name}]=$!
done
fail=0
for name in "${!pid[@]}"; do
  wait "${pid[${name}]}"; rc=$?
  echo "[bundle] ${name} exit=${rc} node-log ${ROOT}/slurm_logs/${name}-${SLURM_JOB_ID}.out"
  (( rc == 0 )) || fail=1
done
exit "${fail}"
