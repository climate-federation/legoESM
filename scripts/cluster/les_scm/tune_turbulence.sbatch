#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --gres=gpu:1
# NOT v100s. The gradient path of the `tke` closure reproducibly aborts with
#   INTERNAL: CUDA error: CUDA_ERROR_ILLEGAL_ADDRESS
# on Tesla V100S (sm_70) -- twice, on two DIFFERENT cards (g049, g050), each
# time entering [tune] tke immediately after louis finished cleanly. The same
# commit completed all nine arms on an A40 (g184), and tke's FORWARD evaluation
# succeeds on V100S (it scores 0.1241), so this is the sm_70 backward pass, not
# the model. Constrain until that is diagnosed separately.
#SBATCH --constraint="a40|a100"
#SBATCH --job-name=les_tune
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/les_tune_%x_%j.out
set +e
# Score and gradient-tune all nine SCM turbulence closures against the matched
# LES reference for $LES_CASE.
#
# Requires the production LES for that case to have finished and written
# profiles/ frames carrying wth (the run's own verify step asserts this).
WT=/burg/glab/users/pg2328/les_scm_wt
cd $WT || exit 1
echo "PINNED sha: $(git rev-parse HEAD)"
echo "host: $(hostname)"; nvidia-smi -L
echo "case: ${LES_CASE:?set LES_CASE}"
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$PYTHONPATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

REPO=/burg-archive/glab/users/pg2328/legoESM
LES_DIR=$REPO/results/les_ref/$LES_CASE
OUT=$REPO/results/scm_les_turbulence/$LES_CASE

if [ ! -d "$LES_DIR/profiles" ]; then
  echo "NO LES REFERENCE at $LES_DIR/profiles — run production_les.sbatch first"
  exit 2
fi

$PY scripts/run/run_scm_les_turbulence_tuning.py \
    --case "$LES_CASE" \
    --les-dir "$LES_DIR" \
    --outdir "$OUT" \
    --nlev "${TUNE_NLEV:-64}" \
    --dt "${TUNE_DT:-60}" \
    --analysis-hours "${TUNE_ANALYSIS_HOURS:-2}" \
    --steps "${TUNE_STEPS:-8}" \
    --tier "${TUNE_TIER:-extended}" \
    ${TUNE_EXTRA_ARGS}
rc=$?
echo "tuning rc=$rc"

if [ -f "$OUT/profiles.npz" ]; then
  $PY scripts/plot/plot_scm_les_turbulence.py "$OUT"
  echo "plot rc=$?"
fi

echo "=== ranking ==="
cat "$OUT/ranking.csv" 2>/dev/null
echo "=== summary ==="
cat "$OUT/summary.md" 2>/dev/null
echo "DONE $(date +%T)"
exit $rc
