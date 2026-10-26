#!/usr/bin/env bash
#SBATCH --job-name=uni_train
#SBATCH --partition=gpu
# The T106 configs are sized for an 80 GB card (base_t106_allyears.yaml says so
# explicitly). Levante's gpu partition mixes 59 a100_80 nodes with 4 a100_40
# nodes and hands out whichever is free, so an unconstrained job dies with
# RESOURCE_EXHAUSTED on the 1-in-16 chance it lands on a 40 GB card — that is
# exactly how the column_nn arm (26721035, l40369) died. Constraining costs
# nothing: it excludes 4 of 63 nodes.
#SBATCH --constraint=a100_80
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=uni_train_%j.out
# ===========================================================================
# Unified WB+AIMIP training on DKRZ Levante (A100-80GB) — one variant/mode
# per job, self-chaining on walltime (shared _chain_body.sh, design D7).
# Set the account with `sbatch -A <proj>` (site policy requires it).
#
# SUBMIT examples:
#   sbatch -A <proj> --export=ALL,CAMPAIGN=aimip,SUITE=config/aimip/scale/suite_classical.yaml,VARIANT=classical \
#     scripts/cluster/unified_training/train_levante.slurm
# Paths + conda env live in scripts/cluster/scaling_levante/_env.sh.
# ===========================================================================
set -uo pipefail

CAMPAIGN="${CAMPAIGN:-aimip}"
VARIANT="${VARIANT:-classical}"
SUITE="${SUITE:?set SUITE=<suite yaml>}"
CHAIN="${CHAIN:-0}"
CHAIN_MAX="${CHAIN_MAX:-24}"

source "${SLURM_SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"  # sets REPO, PY, modules
cd "$REPO"

OUT="${OUT:-results/unified_${CAMPAIGN}/$(basename "${SUITE%.*}")}"

export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
# The WB classical value-and-grad step asks for ONE ~45 GiB scratch arena. The
# default BFC allocator grows its pool in pieces and then cannot serve that
# single block on a card measured at 0.04 GiB in use of 75.29 — pure
# fragmentation, and exactly how job 27093614 died. JAX's own allocator knob is
# this one (TF_GPU_ALLOCATOR is TensorFlow's name and JAX ignores it), and
# cuda_async serves it. PREALLOCATE=true is WORSE: reserving the arena up front
# starves the kernel launches.
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async

mkdir -p "$OUT"
# 12 h walltime minus 45 min drain for the chain resubmit.
LINK_BUDGET_S=$(( 12*3600 - 45*60 ))

resubmit_self() {
  # Export explicitly, then --export=ALL (inherited-ALL precedence over the
  # comma-list form would freeze CHAIN; codex HIGH).
  export CAMPAIGN SUITE VARIANT OUT CHAIN_MAX
  export CHAIN=$((CHAIN+1))
  export RESUME=1
  # This script carries NO ``#SBATCH --account`` (see the header: the account
  # is a submit-time argument), and Levante REJECTS an account-less job. A bare
  # ``sbatch`` here therefore fails at submit while _chain_body.sh has already
  # logged "chaining link N" — the chain dies silently and the run just stops
  # mid-curriculum. Observed 2026-07-28 on the classical arm: link 0 exhausted
  # its budget at epoch 9/11, printed "not complete -> chaining link 1/6", and
  # no job was ever queued.
  #
  # ``SBATCH_ACCOUNT`` / ``SBATCH_CONSTRAINT`` are sbatch's own environment
  # fallbacks, so exporting them is enough — no extra flags to keep in sync.
  # SLURM_JOB_ACCOUNT is set by the running job, which covers the common case
  # of a first submit that passed -A on the command line (that does NOT put
  # SBATCH_ACCOUNT in the job environment).
  export SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-${SLURM_JOB_ACCOUNT:-}}"
  if [ -z "$SBATCH_ACCOUNT" ]; then
    echo "[unified-train] REFUSING to chain: no account resolvable from " \
         "SBATCH_ACCOUNT or SLURM_JOB_ACCOUNT, and Levante rejects an " \
         "account-less job. Resubmit by hand with -A <proj>." >&2
    return 1
  fi
  # Memory-class constraint and GPU COUNT must ride along too. SBATCH_* are
  # sbatch INPUT options, not a record of what the CLI asked for: an initial
  # `sbatch -C a100_80 --gpus=4 ...` sets NEITHER, so a naive resubmit silently
  # drops to the script's `#SBATCH --gpus=1` (disabling the pmap path) and loses
  # the 80 GB constraint (OOM on a 40 GB card). Recover them from what SLURM
  # reports for the RUNNING job, then fall back to the explicit CHAIN_* knobs.
  # (codex review 2026-07-30)
  # Only export when non-empty: an exported EMPTY SBATCH_CONSTRAINT is still a
  # set sbatch input option and would override this script's
  # ``#SBATCH --constraint=a100_80``, silently putting the next link back on a
  # 40 GB card.
  local _constraint="${SBATCH_CONSTRAINT:-${CHAIN_CONSTRAINT:-${SLURM_JOB_CONSTRAINT:-}}}"
  if [ -n "$_constraint" ]; then
    export SBATCH_CONSTRAINT="$_constraint"
  fi
  local _gpus="${CHAIN_GPUS:-}"
  if [ -z "$_gpus" ] && [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
    _gpus="$SLURM_GPUS_ON_NODE"
  fi
  local _args=()
  [ -n "$_gpus" ] && _args+=(--gpus="$_gpus")
  echo "[unified-train] chaining with account=$SBATCH_ACCOUNT " \
       "constraint=${_constraint:-<script default>} gpus=${_gpus:-<script default>}"
  sbatch "${_args[@]}" --export=ALL \
    "$REPO/scripts/cluster/unified_training/train_levante.slurm"
}

source "$REPO/scripts/cluster/unified_training/_chain_body.sh"
