#!/bin/bash
# Stability fine-tune of the AIMIP classical model for free-running AMIP:
# multi-day prescribed-SST (exact AIMIP forcing) rollout + ERA5-match + drift
# penalty -> the ~58 scheme knobs + 65 spatial coefs. NeuralGCM-style rollout
# curriculum + EMA. Env: FT_CURRICULUM (rd:ep,...) / FT_NSAMPLES / FT_LR /
# FT_WDRIFT / FT_EMA.
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=aimip_amip_ft
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=results/aimip_amip_finetune/logs/%x_%j.out
set -uo pipefail
source "$SLURM_SUBMIT_DIR/scripts/cluster/scaling_ginsburg/_env.sh"
cd "$REPO"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
mkdir -p results/aimip_amip_finetune/logs
echo "[amip-ft] $(date) stability fine-tune (PoC)"
"$PY_GPU" -u scripts/run/run_aimip_amip_finetune.py \
  --variant "${FT_VARIANT:-classical}" \
  --curriculum "${FT_CURRICULUM:-1:4,3:6,5:8}" \
  --n-samples "${FT_NSAMPLES:-6}" \
  --lr "${FT_LR:-3e-4}" \
  --w-drift "${FT_WDRIFT:-3.0}" \
  --ema-decay "${FT_EMA:-0.999}" \
  ${FT_CONV:+--convection-scheme "$FT_CONV"} \
  ${FT_INIT:+--init-ckpt "$FT_INIT"} \
  --out-ckpt "${FT_OUT:-results/aimip_amip_finetune/classical_stable.eqx}"
echo "[amip-ft] done $(date)"
