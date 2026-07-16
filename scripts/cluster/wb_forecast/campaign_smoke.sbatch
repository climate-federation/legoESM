#!/bin/bash
#SBATCH --job-name=wb_camp_smoke
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/results/wb_campaign_smoke/slurm_%j.out
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1:30:00
# End-to-end wiring smoke for the WB campaign driver on 1 GPU: train+eval+plot
# for all three families at tiny spectral T21 (config/wb/campaign/spectral_smoke.yaml).
# NOT a science run — catches integration bugs (grid/ERA5 load/rollout/scorecard/
# plot glue) BEFORE the real T63 run. SOTA overlay disabled (--sota-csv '').
#
# Targets the CURRENT checkout (/burg-archive/glab/users/pg2328/legoESM) — NOT the
# stale legoESM_wbforecast worktree; the campaign driver + merged evaluations/
# infra live here. PYTHONPATH = this repo's packages/* namespace (memory python-env).
set -euo pipefail

REPO=/burg-archive/glab/users/pg2328/legoESM
cd "$REPO"
WB_PYTHON="/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python"

_pp="$REPO"
for _d in "$REPO"/packages/*/; do
  [ -d "${_d}legoesm" ] && _pp="${_pp}:${_d}"
done
export PYTHONPATH="${_pp}:${PYTHONPATH:-}"
export MPI4JAX_NO_WARN_JAX_VERSION=1
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1
# Per-job PTX tempdir (avoid the shared /local DeleteFile race across GPU jobs).
export TMPDIR="/local/wb_camp_smoke_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" results/wb_campaign_smoke

echo "[$(date)] WB campaign smoke on $(hostname) | PYTHONPATH head: ${_pp%%:*}"
"$WB_PYTHON" scripts/run/run_weatherbench_campaign.py \
  --config config/wb/campaign/spectral_smoke.yaml \
  --modes physics,neural_gcm,sfno \
  --stages train,eval,plot \
  --training-core spectral \
  --out-root results/wb_campaign_smoke \
  --leads 6 \
  --n-inits 1 \
  --sota-csv ''
echo "[$(date)] smoke done -> results/wb_campaign_smoke/wb_scorecard.png"
rm -rf "$TMPDIR"
