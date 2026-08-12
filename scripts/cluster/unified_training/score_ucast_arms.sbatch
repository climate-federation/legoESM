#!/usr/bin/env bash
#SBATCH --job-name=ucast_score
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=10:00:00
#SBATCH --output=ucast_score_%j.out
# ===========================================================================
# WB2 scorecards for the U-Cast arm comparison, all at ONE epoch.
#
# CONTROLLED COMPARISON — the whole point of this script:
#   * every arm is scored at epoch_0057 (the LAST epoch all four share: the
#     dropout arm stopped at 57 when its link drained, the others reached 59).
#     Scoring one arm at 59 and another at 57 would add a second variable;
#   * EMA weights for every arm (all suites train with aimip_ema_decay 0.999,
#     and ACE2 / U-Cast both evaluate the EMA);
#   * identical eval protocol on every row: eval year 2017, 60 inits at 144 h
#     stride, leads 24/72/120/240 h, WB2 1.5 deg.
# The ONLY thing that differs between rows is the arm's one training knob.
#
# Baseline row = suite_sfno_full_maxdata (dropout 0, adamw, ace2 MSE loss).
# It predates the sfno_arch.json sidecar, so its row logs an UNVERIFIED
# architecture warning; that is fail-soft by design, not a defect.
#
# SUBMIT (from the repo root):
#   sbatch -A <proj> --export=ALL,LEGOESM_REPO=$PWD,\
#LEGOESM_PYTHON=$PWD/.venv/bin/python \
#     scripts/cluster/unified_training/score_ucast_arms.sbatch
# ===========================================================================
set -uo pipefail

source "${SLURM_SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO"

export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# EPOCHS is '+'-separated, NOT comma-separated: `sbatch --export=ALL,K=V` splits
# its list on commas, so `EPOCHS=0055,0057` would arrive as `EPOCHS=0055`.
# Default is the single epoch the U-Cast comparison table was built at.
EPOCHS="${EPOCHS:-${EPOCH:-0057}}"
EPOCHS="${EPOCHS//+/ }"
OUTDIR="${OUTDIR:-results/bench_ucast}"
LEADS="${LEADS:-24,72,120,240}"
N_INITS="${N_INITS:-60}"
STRIDE="${STRIDE:-144}"
EVAL_YEAR="${EVAL_YEAR:-2017}"
mkdir -p "$OUTDIR"

# arm_label : suite : results dir
ARMS=(
  "baseline:config/aimip/scale/suite_sfno_full_maxdata.yaml:results/aimip_maxdata_sfno_full_t63"
  "dropout:config/aimip/scale/suite_sfno_full_ucast_dropout.yaml:results/aimip_ucast_drop_sfno_full_t63"
  "muon:config/aimip/scale/suite_sfno_full_ucast_muon.yaml:results/aimip_ucast_muon_sfno_full_t63"
  "mae:config/aimip/scale/suite_sfno_full_ucast_mae.yaml:results/aimip_ucast_mae_sfno_full_t63"
  # Round 2. muondrop is the deliberate TWO-knob new baseline (muon+dropout,
  # the stage-1 base for the CRPS fine-tune); muonlr is single-knob against
  # the muon arm. Scored at the SAME epoch_0057 as every row above so all six
  # live in ONE table — they also reached epoch 59, but mixing epochs would
  # add a second variable.
  "muondrop:config/aimip/scale/suite_sfno_full_ucast_muondrop.yaml:results/aimip_ucast_muondrop_sfno_full_t63"
  "muonlr:config/aimip/scale/suite_sfno_full_ucast_muonlr.yaml:results/aimip_ucast_muonlr_sfno_full_t63"
)

# Restrict to a subset, e.g. ONLY=mae to redo one row after a walltime kill.
# Measured cost is ~2 h PER ARM at 60 inits x 4 leads on one A100 (the first
# run of this script was sized at 6 h for four arms and lost the last row), so
# a full sweep needs >= 10 h.
#
# WARNING: a MULTI-arm ONLY cannot be passed through `sbatch --export=ALL,...`
# — sbatch splits that list on commas, so `ONLY=a,b` arrives as `ONLY=a` plus
# a stray `b`, and the second arm is silently skipped (hit 2026-08-05).
# Submit one job per arm, which also runs them in parallel.
ONLY="${ONLY:-}"

RC_ALL=0
# CHECKPOINT SELECTION. The last epoch is NOT the best one: on the muonlr arm,
# epoch_0059_ema scores z500 24 h at 20.26 m against epoch_0057_ema's 16.35 m on
# an otherwise identical protocol (+24%). Scanning epochs is how a selection
# rule gets built -- and it must run on the suite's own eval_windows year, NOT
# on 2017, which is the WB2 TEST year (choosing an epoch on it is leakage).
for EPOCH in $EPOCHS; do
for entry in "${ARMS[@]}"; do
  IFS=':' read -r label suite rdir <<< "$entry"
  if [ -n "$ONLY" ] && [[ ",$ONLY," != *",$label,"* ]]; then
    echo "[score] skip $label (ONLY=$ONLY)"
    continue
  fi
  ckpt="$rdir/sfno_full/epoch_${EPOCH}_ema.eqx"
  out="$OUTDIR/${label}_ep${EPOCH}_ema_y${EVAL_YEAR}.json"
  if [ ! -f "$ckpt" ]; then
    # NO SILENT SKIP: a missing arm must be visible in the log, because an
    # absent row reads exactly like "this arm was not worth reporting".
    echo "[score] MISSING $label: $ckpt — row omitted" >&2
    RC_ALL=1
    continue
  fi
  echo "[score] === $label === $ckpt -> $out"
  "$PY" scripts/validate/run_aimip_wb2_eval.py \
    --variant sfno_full --suite "$suite" --checkpoint "$ckpt" \
    --leads "$LEADS" --n-inits "$N_INITS" \
    --init-stride-hours "$STRIDE" --eval-year "$EVAL_YEAR" \
    --out "$out"
  rc=$?
  echo "[score] $label rc=$rc"
  [ "$rc" -ne 0 ] && RC_ALL=1
done
done

echo "[score] done rc=$RC_ALL $(date)"
exit "$RC_ALL"
