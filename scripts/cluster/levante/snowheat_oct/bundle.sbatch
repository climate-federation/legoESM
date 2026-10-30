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
# absolute repo and commit (arm.tsv columns 3-4; extra flags last); bundle refuses
# a repo that moved since setup.  Same as campaign_amip_bias_0929/bundle.sbatch otherwise.
# ORDER (the OFF arms are the plain production default only if step 2 passes):
#   1. sbatch --nodes=7 --job-name=sh2_gate --export=ALL,ARMS="sh2_gate_off sh2_gate_on sh2_gate_on2 sh2_gate_on_2d sh2_gate_off_2d sh2_gate_base sh2_gate_base2" bundle.sbatch
#      then, once sh2_gate_on and sh2_gate_off have written day 276
#      (R=${ROOT:-/work/bd1083/b309178/diffESM/legoesm_pg/amip_runs}):
#      SRC=$R/sh2_gate_on  EXPECT_COMMIT=<same> setup_arms.sh sh2_gate_on_rs
#      SRC=$R/sh2_gate_off EXPECT_COMMIT=<same> setup_arms.sh sh2_gate_off_rs
#      mkdir $R/sh2_gate_on_pert_src && python perturb_snow.py $R/sh2_gate_on/checkpoint_day_0276.npz $R/sh2_gate_on_pert_src/checkpoint_day_0276.npz --delta-K -2
#      SRC=$R/sh2_gate_on_pert_src EXPECT_COMMIT=<same> setup_arms.sh sh2_gate_on_pert
#      sbatch --nodes=3 --job-name=sh2_gate_rs --export=ALL,ARMS="sh2_gate_on_rs sh2_gate_off_rs sh2_gate_on_pert" bundle.sbatch
#   2. python compare_gate.py $ROOT/sh2_gate_base/checkpoint_day_0276.npz $ROOT/sh2_gate_base2/checkpoint_day_0276.npz
#      (base vs base: the run must be reproducible for the gate to mean anything), then
#      python compare_gate.py $ROOT/sh2_gate_off/checkpoint_day_0276.npz $ROOT/sh2_gate_base/checkpoint_day_0276.npz
#      -> both must print IDENTICAL (exit 0); and
#      python compare_gate.py --expect-differ $ROOT/sh2_gate_on/checkpoint_day_0276.npz $ROOT/sh2_gate_off/checkpoint_day_0276.npz
#      -> must print DIFFER (shared) (exit 0); and
#      python compare_gate.py $ROOT/sh2_gate_on/checkpoint_day_0276.npz $ROOT/sh2_gate_on2/checkpoint_day_0276.npz
#      python compare_gate.py $ROOT/sh2_gate_on/checkpoint_day_0276.npz $ROOT/sh2_gate_on_2d/checkpoint_day_0276.npz
#      python compare_gate.py $ROOT/sh2_gate_on_rs/checkpoint_day_0277.npz $ROOT/sh2_gate_on_2d/checkpoint_day_0277.npz
#      python compare_gate.py $ROOT/sh2_gate_off/checkpoint_day_0276.npz $ROOT/sh2_gate_off_2d/checkpoint_day_0276.npz
#      python compare_gate.py $ROOT/sh2_gate_off_rs/checkpoint_day_0277.npz $ROOT/sh2_gate_off_2d/checkpoint_day_0277.npz
#      -> all IDENTICAL (ON reproducible; run length does not change the day-276
#      state; ON and OFF states survive a restart); and
#      python perturb_snow.py --verify $ROOT/sh2_gate_on_pert/checkpoint_day_0277.npz $ROOT/sh2_gate_on_rs/checkpoint_day_0277.npz
#      -> PASS: snow temperature finite, changed, and colder on average over snow
#      columns (the restart reads the saved layer).
#      Step 3 re-runs all nine comparisons itself and refuses to launch unless they pass.
#      ON round-trip fails but OFF passes: snow-layer restart defect; both fail: model.
#   3. sbatch --nodes=4 --job-name=snowheat2 --export=ALL,ARMS="sh2_on_a sh2_off_a sh2_on_b sh2_off_b" bundle.sbatch
#      90 days should fit one job.  If an arm is killed instead (under MPI the
#      model's wall-clock exit is inactive, and checkpoints are written in
#      place), first prove each arm's newest checkpoint loads in full:
#        python compare_gate.py <newest.npz> <newest.npz>   (exit 0 = readable)
#      delete it if not, then submit step 3 again with the same ARMS; each arm
#      resumes from the newest checkpoint in its own directory.
# Chain: ONE script for every arm (base arms too), the frozen run tree's
# amip_mpas_gpu_chain.sbatch, which pins a bundled arm's step to its own node
# (AMIP_STEP_NODE).
set -uo pipefail
: "${CHAIN:=/work/bd1083/b309178/diffESM/legoesm_pg/wt_snowheat_ab_main/scripts/cluster/levante/amip_mpas_gpu_chain.sbatch}"
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
# base (the OFF arms are the plain production default) must both be byte-identical,
# switch-on must differ from switch-off in a shared entry (the switch acts),
# ON must be reproducible, and ON restarted at day 276 must equal ON run
# continuously to day 277.
# Residual, accepted: a tree moved between this check and the model import
# (seconds) is caught only afterwards, by the commit each run log prints.
n_gate=$(printf '%s\n' "${arms[@]}" | grep -c '^sh2_gate_')
(( n_gate == 0 || n_gate == ${#arms[@]} )) \
  || { echo "REFUSING: gate and long arms in one submission" >&2; exit 64; }
if (( n_gate == 0 )); then
  # The gate must have run the SAME code: the switch-off gate arm's recorded
  # commit equals every long arm's, and so does each base gate arm's.
  _gc() { IFS=$'\t' read -r _ _ _ c _ < "${ROOT}/$1/arm.tsv"; echo "${c}"; }
  for a in "${arms[@]}"; do
    [[ "$(_gc "${a}")" == "$(_gc sh2_gate_off)" ]] \
      || { echo "REFUSING: ${a} commit differs from the gate's" >&2; exit 64; }
  done
  [[ "$(_gc sh2_gate_base)" == "$(_gc sh2_gate_off)" && \
     "$(_gc sh2_gate_base2)" == "$(_gc sh2_gate_off)" && \
     "$(_gc sh2_gate_on)" == "$(_gc sh2_gate_off)" && \
     "$(_gc sh2_gate_on2)" == "$(_gc sh2_gate_off)" && \
     "$(_gc sh2_gate_on_2d)" == "$(_gc sh2_gate_off)" && \
     "$(_gc sh2_gate_on_rs)" == "$(_gc sh2_gate_off)" && \
     "$(_gc sh2_gate_off_2d)" == "$(_gc sh2_gate_off)" && \
     "$(_gc sh2_gate_off_rs)" == "$(_gc sh2_gate_off)" && \
     "$(_gc sh2_gate_on_pert)" == "$(_gc sh2_gate_off)" ]] \
    || { echo "REFUSING: base gate arms ran a different commit" >&2; exit 64; }
  GATE="$(dirname "${CHAIN}")/snowheat_oct/compare_gate.py"
  g="checkpoint_day_0276.npz"
  "${PY}" "${GATE}" "${ROOT}/sh2_gate_base/${g}" "${ROOT}/sh2_gate_base2/${g}" \
    && "${PY}" "${GATE}" "${ROOT}/sh2_gate_off/${g}" "${ROOT}/sh2_gate_base/${g}" \
    && "${PY}" "${GATE}" --expect-differ "${ROOT}/sh2_gate_on/${g}" "${ROOT}/sh2_gate_off/${g}" \
    && "${PY}" "${GATE}" "${ROOT}/sh2_gate_on/${g}" "${ROOT}/sh2_gate_on2/${g}" \
    && "${PY}" "${GATE}" "${ROOT}/sh2_gate_on/${g}" "${ROOT}/sh2_gate_on_2d/${g}" \
    && "${PY}" "${GATE}" "${ROOT}/sh2_gate_on_rs/checkpoint_day_0277.npz" "${ROOT}/sh2_gate_on_2d/checkpoint_day_0277.npz" \
    && "${PY}" "${GATE}" "${ROOT}/sh2_gate_off/${g}" "${ROOT}/sh2_gate_off_2d/${g}" \
    && "${PY}" "${GATE}" "${ROOT}/sh2_gate_off_rs/checkpoint_day_0277.npz" "${ROOT}/sh2_gate_off_2d/checkpoint_day_0277.npz" \
    && "${PY}" "$(dirname "${GATE}")/perturb_snow.py" --verify "${ROOT}/sh2_gate_on_pert/checkpoint_day_0277.npz" "${ROOT}/sh2_gate_on_rs/checkpoint_day_0277.npz" \
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
