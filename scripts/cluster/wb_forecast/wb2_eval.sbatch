#!/bin/bash
#SBATCH --job-name=wb2_eval
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/results/aimip_wbcompare/logs/%x_%j.out
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
# WeatherBench-2 forecast-scorecard eval of ONE run_aimip variant checkpoint via
# the bridge (scripts/validate/run_aimip_wb2_eval.py). Reconstructs the training
# config from the suite so the eval matches training. Current checkout.
#   WB_EVAL_VARIANT   classical | column_nn | sfno_full   (required)
#   WB_EVAL_SUITE     suite YAML the variant was trained with (required)
#   WB_EVAL_CKPT      checkpoint (default: latest epoch_*.eqx in the variant dir)
#   WB_EVAL_YEAR      WB2 scorecard year (default 2020)
#   WB_EVAL_LEADS     forecast leads [h] (default 24,72,120,240)
#   WB_EVAL_OUT       scorecard JSON (default results/aimip_wbcompare/<variant>/scorecard.json)
set -uo pipefail

REPO=/burg-archive/glab/users/pg2328/legoESM
cd "$REPO"
PY_GPU="/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python"
_pp="$REPO"; for _d in "$REPO"/packages/*/; do [ -d "${_d}legoesm" ] && _pp="${_pp}:${_d}"; done
export PYTHONPATH="${_pp}:${PYTHONPATH:-}"
export MPI4JAX_NO_WARN_JAX_VERSION=1
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export TMPDIR="/local/wb2_eval_${SLURM_JOB_ID}"; mkdir -p "$TMPDIR" results/aimip_wbcompare/logs

V="${WB_EVAL_VARIANT:?set WB_EVAL_VARIANT}"
SUITE="${WB_EVAL_SUITE:?set WB_EVAL_SUITE}"
YEAR="${WB_EVAL_YEAR:-2020}"
LEADS="${WB_EVAL_LEADS:-24,72,120,240}"
CKPT="${WB_EVAL_CKPT:-$(ls -t results/aimip_wbcompare/${V}/epoch_*.eqx 2>/dev/null | head -1)}"
OUT="${WB_EVAL_OUT:-results/aimip_wbcompare/${V}/scorecard.json}"
[ -z "$CKPT" ] && { echo "no checkpoint for $V"; exit 1; }
echo "[$(date)] WB2 eval $V | ckpt=$CKPT | suite=$SUITE | year=$YEAR"
"$PY_GPU" -u scripts/validate/run_aimip_wb2_eval.py \
  --variant "$V" --suite "$SUITE" --checkpoint "$CKPT" \
  --eval-year "$YEAR" --leads "$LEADS" --out "$OUT"
echo "[$(date)] done -> $OUT"
rm -rf "$TMPDIR"
