#!/bin/bash -l
#SBATCH --job-name=aimip_scale_t106
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --output=results/aimip_scale_t106/logs/%x_%j.log
# ===========================================================================
# AIMIP AT SCALE on DKRZ Levante — single A100-80GB, T106 (~1.1 deg), ALL
# training years 1979-2014 (576 IC/target pairs), one variant per job.
#
# AIMIP training is SINGLE-DEVICE by design (spectral semi-implicit solver
# all-gathers levels; SH transforms carry no collectives), so "at scale" =
# a bigger GPU running the full-resolution, all-years, higher-capacity config
# (the 80 GB card is what makes T106 + RRTMGP + 2-step rollout + spatial-
# surface coefs fit), NOT multi-GPU sharding.
#
# Self-CHAINS via run_aimip.py per-epoch checkpoints + --resume: a 15-epoch
# T106 run that exceeds one 8 h walltime resubmits itself until done.
#
# SUBMIT (one per variant; set the account):
#   SBATCH_ACCOUNT=<proj> VARIANT=sfno_physics \
#     sbatch --export=ALL scripts/cluster/aimip_scale/train_aimip_levante.slurm
#   ... VARIANT=classical ... ; ... VARIANT=column_nn ...
# Paths + conda env live in scripts/cluster/scaling_levante/_env.sh (EDIT
# REPO + CONDA_ENV before first run).
# ===========================================================================
set -uo pipefail

VARIANT="${VARIANT:-sfno_physics}"     # classical | column_nn | sfno_physics
CHAIN="${CHAIN:-0}"
CHAIN_MAX="${CHAIN_MAX:-24}"  # v2 curriculum (72h/120h phases) needs more links

source scripts/cluster/scaling_levante/_env.sh    # REPO, PY, conda, JAX env
cd "$REPO"

export JAX_PLATFORMS=cuda JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

SUITE="config/aimip/scale/suite_${VARIANT}.yaml"
OUT="results/aimip_scale_t106/${VARIANT}"
mkdir -p "$OUT/logs"

RESUME_FLAG=""
if ls "$OUT"/*/epoch_*.eqx >/dev/null 2>&1 || [ "${RESUME:-0}" = "1" ]; then
  RESUME_FLAG="--resume"
fi

echo "[aimip-scale] $(date) variant=$VARIANT link=$CHAIN suite=$SUITE resume=${RESUME_FLAG:-no}"
"$PY" -u scripts/run/run_aimip.py --suite "$SUITE" $RESUME_FLAG
RC=$?
echo "[aimip-scale] run_aimip rc=$RC $(date)"

FINAL="$OUT/${VARIANT}/params.eqx"
if [ "$RC" -eq 0 ] && [ ! -f "$FINAL" ] && [ "$CHAIN" -lt "$CHAIN_MAX" ]; then
  echo "[aimip-scale] not complete -> chaining link $((CHAIN+1))/$CHAIN_MAX"
  VARIANT="$VARIANT" CHAIN="$((CHAIN+1))" CHAIN_MAX="$CHAIN_MAX" RESUME=1 \
    sbatch --export=ALL "$SLURM_SUBMIT_DIR/scripts/cluster/aimip_scale/train_aimip_levante.slurm"
elif [ -f "$FINAL" ]; then
  echo "[aimip-scale] DONE: $FINAL"
else
  echo "[aimip-scale] stop (rc=$RC or chain limit $CHAIN_MAX)"
fi
