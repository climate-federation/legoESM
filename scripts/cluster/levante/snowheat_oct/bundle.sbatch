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
# ORDER (the OFF arms are the shared 468cef302 control only if step 2 passes):
#   1. sbatch --nodes=3 --job-name=sh2_gate --export=ALL,ARMS="sh2_gate_off sh2_gate_base sh2_gate_base2" bundle.sbatch
#   2. python compare_gate.py $ROOT/sh2_gate_base/checkpoint_day_0276.npz $ROOT/sh2_gate_base2/checkpoint_day_0276.npz
#      (base vs base: the run must be reproducible for the gate to mean anything), then
#      python compare_gate.py $ROOT/sh2_gate_off/checkpoint_day_0276.npz $ROOT/sh2_gate_base/checkpoint_day_0276.npz
#      -> both must print IDENTICAL (exit 0).  Step 3 re-runs both comparisons
#      itself and refuses to launch unless they pass.
#   3. sbatch --nodes=4 --job-name=snowheat2 --export=ALL,ARMS="sh2_on_a sh2_off_a sh2_on_b sh2_off_b" bundle.sbatch
# Chain: ONE script for every arm (base arms too), the frozen run tree's
# amip_mpas_gpu_chain.sbatch, which pins a bundled arm's step to its own node
# (AMIP_STEP_NODE; the 468cef302 chain lacks it and would stack the arms).
set -uo pipefail
: "${CHAIN:=/work/bd1083/b309178/diffESM/legoesm_pg/wt_snowheat2_run/scripts/cluster/levante/amip_mpas_gpu_chain.sbatch}"
[[ -f "${CHAIN}" ]] && grep -q AMIP_STEP_NODE "${CHAIN}" || { echo "REFUSING: ${CHAIN} cannot pin arms to nodes" >&2; exit 64; }
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
  IFS=$'\t' read -r _n _t _r _c _e < "${ROOT}/${a}/arm.tsv"
  [[ "${_n}" == "${a}" && "${_t}" =~ ^[0-9]+$ && "${_r}" == /* && -e "${_r}/.git" ]] \
    || { echo "REFUSING: ${a}/arm.tsv names '${_n}' target '${_t}' repo '${_r}'" >&2; exit 64; }
  # Frozen code: the repo must still be clean and at the commit setup recorded.
  [[ "$(git -c safe.directory='*' -C "${_r}" rev-parse HEAD)" == "${_c}" && \
     -z $(git -c safe.directory='*' -C "${_r}" status --porcelain --untracked-files=no) ]] \
    || { echo "REFUSING: ${_r} moved or is dirty since setup (want ${_c})" >&2; exit 64; }
done
# Long arms only after the 1-day gate, re-checked HERE from the checkpoints
# (not from a note): base vs base2 (the run is reproducible) and switch-off vs
# base (the OFF arms are plain 468cef302) must both be byte-identical.
if [[ " ${arms[*]} " != *" sh2_gate_"* ]]; then
  GATE="$(dirname "${CHAIN}")/snowheat_oct/compare_gate.py"
  g="checkpoint_day_0276.npz"
  "${PY}" "${GATE}" "${ROOT}/sh2_gate_base/${g}" "${ROOT}/sh2_gate_base2/${g}" \
    && "${PY}" "${GATE}" "${ROOT}/sh2_gate_off/${g}" "${ROOT}/sh2_gate_base/${g}" \
    || { echo "REFUSING: the 1-day gate has not passed" >&2; exit 64; }
fi
export ROOT PY AMIP_SURFDATA ERA5_IC AMIP_N_GPUS=4
declare -A pid
for i in "${!arms[@]}"; do
  IFS=$'\t' read -r name target repo commit extra < "${ROOT}/${arms[i]}/arm.tsv"
  echo "[bundle] ${name} node=${nodes[i]} repo=${repo} commit=$(git -c safe.directory='*' -C "${repo}" rev-parse HEAD) dirty=$(git -c safe.directory='*' -C "${repo}" status --porcelain --untracked-files=no | wc -l)"
  ( REPO="${repo}" NAME="${name}" TARGET_DAYS="${target}" EXTRA="${COMMON}${extra}" \
    MAX_WALLCLOCK_SECONDS="${ARM_WALLCLOCK_SECONDS:-39600}" AMIP_STEP_NODE="${nodes[i]}" \
    bash "${CHAIN}" \
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
