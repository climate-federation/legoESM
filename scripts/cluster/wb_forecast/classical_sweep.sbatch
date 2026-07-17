#!/bin/bash
#SBATCH --job-name=wb_classical_sweep
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/results/aimip_wbcompare/classical_sweep/logs/%x_%j.out
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=72:00:00
# Classical physics-combo sweep on Ginsburg (1 GPU). For each combo dir under
# config/aimip/wbcompare/classical_sweep/, this:
#   (1) TRAINS the run_aimip CLASSICAL variant with THAT combo's schemes
#       (writes the AIMIP-protocol scorecard to the combo output_dir), then
#   (2) WB2-evals the latest epoch_*.eqx via the eval bridge (reconstructs the
#       training config from the SAME suite, so eval == training byte-for-byte).
# After the loop, plots the WB2 comparison across combos + published SOTA.
#
# run_aimip is single-GPU spectral (NOT the DP scale trainer): combos run
# sequentially, one GPU. rrtmgp dominates walltime -> 72 h.
#
# The combo list is DERIVED from the classical_sweep dir (a bash array over the
# combo subdirs), so dropping a new <combo>/suite.yaml in extends the sweep with
# no edit here.
#
# Env overrides:
#   WB_COMBOS      comma list to subset (default: every combo dir on disk)
#   WB_EVAL_YEAR   WB2 scorecard year (default 2020 — matches the combo suites)
#   WB_LEADS       forecast leads [h] (default 24,72,120,240)
#   WB_SOTA_CSV    published-SOTA CSV (default config/wb/sota/wb2_headline_rmse.csv)
#   WB_SKIP_TRAIN  set 1 to eval+plot existing checkpoints only
#   WB_RESUME      set (any value) to pass --resume to run_aimip
set -uo pipefail

REPO=/burg-archive/glab/users/pg2328/legoESM
cd "$REPO"
PY_GPU="/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python"
_pp="$REPO"
for _d in "$REPO"/packages/*/; do
  [ -d "${_d}legoesm" ] && _pp="${_pp}:${_d}"
done
export PYTHONPATH="${_pp}:${PYTHONPATH:-}"
export MPI4JAX_NO_WARN_JAX_VERSION=1
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export TMPDIR="/local/wb_classical_sweep_${SLURM_JOB_ID:-$$}"

SWEEP_DIR="config/aimip/wbcompare/classical_sweep"
OUT_ROOT="results/aimip_wbcompare/classical_sweep"
EVAL_YEAR="${WB_EVAL_YEAR:-2020}"
LEADS="${WB_LEADS:-24,72,120,240}"
SOTA_CSV="${WB_SOTA_CSV:-config/wb/sota/wb2_headline_rmse.csv}"
mkdir -p "$TMPDIR" "${OUT_ROOT}/logs"

# --- combo list: derived from the sweep dir (subdirs with a suite.yaml) -------
declare -a COMBOS=()
for d in "${SWEEP_DIR}"/*/; do
  [ -f "${d}suite.yaml" ] && COMBOS+=("$(basename "$d")")
done
# Optional subset via WB_COMBOS (comma list); keep only names that exist on disk.
if [ -n "${WB_COMBOS:-}" ]; then
  declare -a SUBSET=()
  IFS=',' read -ra _want <<< "${WB_COMBOS}"
  for w in "${_want[@]}"; do
    w="${w// /}"
    if [ -f "${SWEEP_DIR}/${w}/suite.yaml" ]; then
      SUBSET+=("$w")
    else
      echo "[warn] WB_COMBOS entry '${w}' has no ${SWEEP_DIR}/${w}/suite.yaml — skipping"
    fi
  done
  COMBOS=("${SUBSET[@]}")
fi
if [ "${#COMBOS[@]}" -eq 0 ]; then
  echo "[error] no combos to run (SWEEP_DIR=${SWEEP_DIR}, WB_COMBOS='${WB_COMBOS:-}')"
  exit 1
fi
echo "[$(date)] classical sweep on $(hostname) | combos=${COMBOS[*]} eval_year=${EVAL_YEAR}"
nvidia-smi -L || true

SCORECARD_ARGS=""
for C in "${COMBOS[@]}"; do
  SUITE="${SWEEP_DIR}/${C}/suite.yaml"
  CDIR="${OUT_ROOT}/${C}"
  echo "======================================================================"
  echo "[$(date)] COMBO ${C} | suite=${SUITE}"

  # (1) TRAIN classical variant with this combo's schemes -> AIMIP scorecard.
  if [ "${WB_SKIP_TRAIN:-0}" != "1" ]; then
    "$PY_GPU" -u scripts/run/run_aimip.py --suite "${SUITE}" ${WB_RESUME:+--resume} \
      || { echo "[warn] train failed for ${C} — skipping its WB2 eval"; continue; }
  fi

  # (2) WB2 eval on the latest checkpoint (bridge rebuilds config from SUITE).
  CKPT=$(ls -t "${CDIR}/classical"/epoch_*.eqx 2>/dev/null | head -1 || true)
  if [ -z "${CKPT}" ]; then
    CKPT=$(ls -t "${CDIR}"/epoch_*.eqx 2>/dev/null | head -1 || true)
  fi
  if [ -z "${CKPT}" ]; then
    echo "[warn] no epoch_*.eqx under ${CDIR} — skipping ${C} WB2 scorecard"
    continue
  fi
  echo "[$(date)] EVAL ${C} (ckpt=${CKPT})"
  "$PY_GPU" -u scripts/validate/run_aimip_wb2_eval.py \
    --variant classical --suite "${SUITE}" --checkpoint "${CKPT}" \
    --eval-year "${EVAL_YEAR}" --leads "${LEADS}" \
    --out "${CDIR}/scorecard.json" \
    && SCORECARD_ARGS="${SCORECARD_ARGS} --scorecard ${C}=${CDIR}/scorecard.json" \
    || echo "[warn] WB2 eval failed for ${C}"
done

# --- comparison plot across combos (+ SOTA leaderboard if the CSV exists) -----
if [ -n "${SCORECARD_ARGS}" ]; then
  SOTA_ARG=""
  [ -f "${SOTA_CSV}" ] && SOTA_ARG="--sota-csv ${SOTA_CSV}"
  echo "[$(date)] PLOT -> ${OUT_ROOT}/wb_scorecard.png"
  "$PY_GPU" -u scripts/plot/plot_wb_scorecard.py ${SCORECARD_ARGS} ${SOTA_ARG} \
    --out "${OUT_ROOT}/wb_scorecard.png" || echo "[warn] plot failed"
fi
echo "[$(date)] done -> ${OUT_ROOT} (AIMIP scorecards per combo + wb_scorecard.png)"
rm -rf "$TMPDIR"
