#!/bin/bash
# Retrain the AIMIP suite (classical[winner=edmf] / column_nn / sfno_physics)
# with the ACE2-ERA5 LOSS: residual normalization (std of 6h field change),
# 2-step autoregressive supervision, flat per-var weights, pure MSE. T63 +
# RRTMGP kept. 2-step RRTMGP at T63 is ~2x single-step -> generous walltime;
# aimip_patience=3 early-stops at convergence.
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=aimip_ace2loss
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --output=results/aimip_ace2loss/logs/%x_%j.out
# NOTE: SLURM opens --output at submit time; the log dir must already exist.
# Submit via ./submit.sh (creates it) rather than raw `sbatch <file>`.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/scripts/cluster/scaling_ginsburg/_env.sh"
cd "$REPO"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
mkdir -p results/aimip_ace2loss/logs
nvidia-smi -L
SUITE="${ACE2_SUITE:-config/aimip/ace2/suite.yaml}"
# derive the output dir from the suite (scorecard lives there)
OUT=$("$PY_GPU" -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['output_dir'])" "$SUITE")
echo "[ace2loss] $(date) ACE2-loss retrain — suite=$SUITE out=$OUT"
# ACE2_RESUME=1 -> --resume: load latest epoch_NNNN.eqx and continue; with all
# epochs already done this is an eval-only scorecard regen (e.g. a variant whose
# combined-suite job was cancelled before the scorecard write).
JAX_PLATFORMS=cuda "$PY_GPU" -u scripts/run/run_aimip.py \
  --suite "$SUITE" ${ACE2_RESUME:+--resume}
echo "=== scorecard ==="
python -c "
import json,os
d=json.load(open(os.path.join('$OUT','aimip_scorecard.json')))
v=d.get('variants',d)
for k,val in v.items():
    r=val.get('eval_metrics',{}).get('rmse',{})
    print(f'{k:14s} T={r.get(\"T\",{}).get(\"mean\",\"?\")} T_sfc={r.get(\"T_sfc\",{}).get(\"mean\",\"?\")}')
" 2>/dev/null || true
echo "[ace2loss] done $(date)"
