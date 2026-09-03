#!/bin/bash
# Codex implementation job: shared Newton-root helper (house standard) + adoption
# in the two-leaf canopy closure solver. Runs on a compute node because the login
# node forbids multi-core / long-running work.
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=cxnewton
#SBATCH --time=08:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/.claude/worktrees/pmodel-lmip/scripts/tmp/_codex_newton_%j.log
set -uo pipefail

export CODEX_HOME=/tmp/codex_home_$SLURM_JOB_ID
mkdir -p "$CODEX_HOME"
cp ~/.codex/auth.json "$CODEX_HOME/" 2>/dev/null || true
echo "CODEX_HOME exists=$([ -d "$CODEX_HOME" ] && echo yes || echo NO) auth=$([ -f "$CODEX_HOME/auth.json" ] && echo yes || echo NO)"
# Persist a refreshed token so the next job does not replay a consumed one.
trap 'cp "$CODEX_HOME/auth.json" ~/.codex/auth.json 2>/dev/null || true' EXIT

WT=/burg-archive/glab/users/pg2328/legoESM/.claude/worktrees/pmodel-lmip
cd "$WT" || exit 1
echo "sha:$(git rev-parse HEAD)"

PROMPT=$(cat "$WT/scripts/cluster/land_carbon/codex_newton_helper_prompt.txt")

codex exec --sandbox workspace-write -C "$WT" "$PROMPT" < /dev/null 2>&1 | tail -400
echo "rc=${PIPESTATUS[0]}"
echo CXNEWTON_DONE
