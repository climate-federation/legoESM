#!/usr/bin/env bash
#SBATCH --job-name=uni_train
#SBATCH --partition=gpu
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
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

mkdir -p "$OUT"
# 12 h walltime minus 45 min drain for the chain resubmit.
LINK_BUDGET_S=$(( 12*3600 - 45*60 ))

resubmit_self() {
  sbatch --export=ALL,CAMPAIGN="$CAMPAIGN",SUITE="$SUITE",VARIANT="$VARIANT",OUT="$OUT",CHAIN="$((CHAIN+1))",CHAIN_MAX="$CHAIN_MAX",RESUME=1 \
    "$REPO/scripts/cluster/unified_training/train_levante.slurm"
}

source "$REPO/scripts/cluster/unified_training/_chain_body.sh"
