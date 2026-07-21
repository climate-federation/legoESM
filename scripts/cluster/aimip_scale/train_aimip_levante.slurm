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
# The variant ladder: classical (full physics + RRTMGP radiation),
# column_nn (ALL physics incl. radiation learned by a column MLP),
# sfno_full (SFNO emulates the ENTIRE atmosphere -- no dycore, no
# physics).  sfno_physics (SFNO physics, dycore retained) remains
# available as an optional intermediate rung.
#
# SUBMIT (one per variant; set the account):
#   SBATCH_ACCOUNT=<proj> VARIANT=sfno_full \
#     sbatch --export=ALL scripts/cluster/aimip_scale/train_aimip_levante.slurm
#   ... VARIANT=classical ... ; ... VARIANT=column_nn ...
# Paths + conda env live in scripts/cluster/scaling_levante/_env.sh (EDIT
# REPO + CONDA_ENV before first run).
# ===========================================================================
set -uo pipefail

VARIANT="${VARIANT:-sfno_full}"     # classical | column_nn | sfno_full | sfno_physics
CHAIN="${CHAIN:-0}"
CHAIN_MAX="${CHAIN_MAX:-24}"  # v2 curriculum (72h/120h phases) needs more links

source scripts/cluster/scaling_levante/_env.sh    # REPO, PY, conda, JAX env
cd "$REPO"

# cuda,cpu: expose the CPU backend so chunk_prefetch can host-stage the buffered
# chunk off the GPU (#985); cuda stays the default device, harmless if unused.
export JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

SUITE="config/aimip/scale/suite_${VARIANT}.yaml"
OUT="results/aimip_scale_t106/${VARIANT}"
mkdir -p "$OUT/logs"

# Resume if ANY checkpoint exists: a completed-epoch checkpoint (epoch_*.eqx)
# OR a mid-epoch per-chunk checkpoint (chunk_latest.eqx, written by the chunked
# column_nn / sfno_physics variants; classical and sfno_full checkpoint
# per-epoch only).
RESUME_FLAG=""
if ls "$OUT"/*/epoch_*.eqx >/dev/null 2>&1 \
   || ls "$OUT"/*/chunk_latest.eqx >/dev/null 2>&1 \
   || [ "${RESUME:-0}" = "1" ]; then
  RESUME_FLAG="--resume"
fi

echo "[aimip-scale] $(date) variant=$VARIANT link=$CHAIN suite=$SUITE resume=${RESUME_FLAG:-no}"
# Stop training with slack to run the chain-resubmit below BEFORE Slurm kills
# the job at the walltime cap — a walltime kill terminates this script AND
# python together, so an sbatch placed after the run would never fire on a
# walltime-bound link (same #985 fix as the Derecho launcher).  Keep
# LINK_BUDGET_S in sync with `#SBATCH --time` (8 h) above.
LINK_BUDGET_S=$(( 8*3600 - 30*60 ))   # 27000 s = walltime - 30 min drain
# Chain a budget-killed link ONLY if a resume artifact actually advanced —
# classical/sfno_full have no mid-epoch checkpoint, so blindly chaining a link
# whose epoch never finished would restart epoch 0 forever.
_ckpt_sig() {
  ls -t "$OUT"/*/epoch_*.eqx "$OUT"/*/chunk_latest.eqx 2>/dev/null \
    | head -1 | xargs -r stat -c '%n:%Y' 2>/dev/null
}
SIG_BEFORE="$(_ckpt_sig)"
timeout -k 120 "${LINK_BUDGET_S}s" "$PY" -u scripts/run/run_aimip.py --suite "$SUITE" $RESUME_FLAG
RC=$?
if [ "$RC" -eq 124 ] || [ "$RC" -eq 137 ]; then
  if [ "$(_ckpt_sig)" != "$SIG_BEFORE" ]; then
    echo "[aimip-scale] hit link budget ${LINK_BUDGET_S}s (rc=$RC), checkpoint advanced -> chaining"
    RC=0
  else
    echo "[aimip-scale] hit link budget ${LINK_BUDGET_S}s (rc=$RC) but NO new checkpoint" \
         "-> not mid-epoch resumable (classical/sfno_full have no chunk_latest.eqx" \
         "and the epoch did not finish); NOT chaining. Remedy: chunked variants" \
         "-> chunk_windows>0; classical/sfno_full -> a walltime long enough to" \
         "finish one epoch."
    RC=1
  fi
fi
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
