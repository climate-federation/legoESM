#!/bin/bash
# Prescribed-SST AMIP inference (EXACT ai2cm/AIMIP protocol) for the trained
# classical model. Env: AMIP_START/AMIP_END/AMIP_RECORD_FROM (ISO dates) +
# AMIP_MEMBER (0-4) + AMIP_CKPT (trained params) + AMIP_WALL_LIMIT (hours).
# Default = short forcing-pipeline validation (spin-up + 3 recorded months);
# full run = START 1978-10-01 END 2025-01-01. The 46-yr run exceeds one
# walltime: the driver stops at AMIP_WALL_LIMIT, writes a restart + .RESUME
# marker, and this script SELF-CHAINS (resubmits itself) until the run reaches
# --end or AMIP_CHAIN_MAX links.
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=aimip_amip_infer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=results/aimip_fleet_paper/logs/%x_%j.out
set -uo pipefail
source "$SLURM_SUBMIT_DIR/scripts/cluster/scaling_ginsburg/_env.sh"
cd "$REPO"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
mkdir -p results/aimip_fleet_paper/logs
START="${AMIP_START:-1978-10-01}"
END="${AMIP_END:-1979-04-01}"
RECORD_FROM="${AMIP_RECORD_FROM:-1979-01-01}"
MEMBER="${AMIP_MEMBER:-0}"
VARIANT="${AMIP_VARIANT:-classical}"   # classical | column_nn | sfno_physics
CKPT="${AMIP_CKPT:-results/aimip_ace2loss_assembled/classical/params.eqx}"
WALL_LIMIT="${AMIP_WALL_LIMIT:-11.0}"   # < the 12 h SBATCH walltime
CHAIN="${AMIP_CHAIN_COUNT:-0}"
CHAIN_MAX="${AMIP_CHAIN_MAX:-20}"
REINIT="${AMIP_REINIT_MONTHS:-0}"   # 0=free-run; N>0=hindcast reinit every N months
# The driver appends a _reinitNmo suffix to the out CSV + restart base when
# --reinit-every-months>0, so the chaining marker must match.
if [ "$REINIT" -gt 0 ]; then _RTAG="_reinit${REINIT}mo"; else _RTAG=""; fi
echo "[amip-infer] $(date) $VARIANT AMIP $START..$END record>=$RECORD_FROM r$((MEMBER+1)) reinit=$REINIT ckpt=$CKPT link=$CHAIN"
"$PY_GPU" -u scripts/run/run_aimip_amip_inference.py \
  --ckpt "$CKPT" --variant "$VARIANT" \
  --start "$START" --end "$END" --record-from "$RECORD_FROM" --member "$MEMBER" \
  --wall-limit-hours "$WALL_LIMIT" \
  ${AMIP_CONV:+--convection-scheme "$AMIP_CONV"} \
  $( [ "$REINIT" -gt 0 ] && echo "--reinit-every-months $REINIT" ) \
  --out "results/aimip_fleet_paper/legoesm_${VARIANT}_amip${_RTAG}_r${MEMBER}.csv"
RC=$?
# Must match the driver's variant+reinit-qualified default restart base.
MARKER="results/aimip_fleet_paper/amip_restart_${VARIANT}${_RTAG}_r${MEMBER}.RESUME"
if [ "$RC" -eq 0 ] && [ -f "$MARKER" ]; then
  if [ "$CHAIN" -lt "$CHAIN_MAX" ]; then
    echo "[amip-infer] RESUME marker present -> self-chaining (link $((CHAIN+1))/$CHAIN_MAX)"
    sbatch --export=ALL,AMIP_CHAIN_COUNT=$((CHAIN+1)) \
      "$SLURM_SUBMIT_DIR/scripts/cluster/scaling_ginsburg/aimip_amip_inference.sbatch"
  else
    echo "[amip-infer] chain limit $CHAIN_MAX reached; NOT resubmitting"
  fi
fi
echo "[amip-infer] done $(date) rc=$RC"
