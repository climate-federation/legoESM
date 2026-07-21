#!/usr/bin/env bash
#SBATCH --account=glab
#SBATCH --partition=burst
#SBATCH --job-name=uni_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=72:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/scripts/cluster/unified_training/logs/train_%j.out
# ===========================================================================
# Unified WB+AIMIP training on Columbia Ginsburg — one variant/mode per job,
# self-chaining on walltime (shared _chain_body.sh, design D7).
#
# SUBMIT examples:
#   sbatch --export=ALL,CAMPAIGN=aimip,SUITE=config/aimip/ace2/suite.yaml,VARIANT=classical \
#     scripts/cluster/unified_training/train_ginsburg.sbatch
#   sbatch --export=ALL,CAMPAIGN=wb,SUITE=config/wb/campaign/spectral_t63.yaml,VARIANT=sfno \
#     scripts/cluster/unified_training/train_ginsburg.sbatch
# ===========================================================================
set -uo pipefail

CAMPAIGN="${CAMPAIGN:-aimip}"
VARIANT="${VARIANT:-classical}"
SUITE="${SUITE:?set SUITE=<suite yaml>}"
OUT="${OUT:-results/unified_${CAMPAIGN}/$(basename "${SUITE%.*}")}"
CHAIN="${CHAIN:-0}"
CHAIN_MAX="${CHAIN_MAX:-12}"

source /burg-archive/glab/users/pg2328/legoESM/scripts/cluster/unified_training/_env.sh
cd "$UT_REPO"
PY="$UT_PYTHON"
REPO="$UT_REPO"

export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

mkdir -p "$OUT"
# 72 h walltime minus 45 min drain for the chain resubmit.
LINK_BUDGET_S=$(( 72*3600 - 45*60 ))

resubmit_self() {
  # Export explicitly, then --export=ALL: on this Slurm, values inherited
  # via ALL take precedence over the comma-list form, so CHAIN would stop
  # advancing once exported (codex HIGH). Exporting the updated values in
  # THIS shell makes them the inherited ones.
  export CAMPAIGN SUITE VARIANT OUT CHAIN_MAX
  export CHAIN=$((CHAIN+1))
  export RESUME=1
  sbatch --export=ALL "$REPO/scripts/cluster/unified_training/train_ginsburg.sbatch"
}

source "$REPO/scripts/cluster/unified_training/_chain_body.sh"
