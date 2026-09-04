#!/bin/bash
# Generic codex job: runs the prompt in $PROMPT_FILE against this worktree on a
# compute node (the login node forbids multi-core / long-running work).
#   PROMPT_FILE=path/to/prompt.txt [CODEX_SANDBOX=read-only|workspace-write] \
#     sbatch scripts/cluster/land_carbon/codex_prompt.sbatch
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=cxprompt
#SBATCH --time=06:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/.claude/worktrees/pmodel-lmip/scripts/tmp/_codex_prompt_%j.log
set -uo pipefail

# Node-local /tmp vanished mid-job on g040/g057 (codex's exec backend then
# fails every spawn), so scratch lives under the worktree instead.
SCRATCH=/burg-archive/glab/users/pg2328/legoESM/.claude/worktrees/pmodel-lmip/scripts/tmp/_codex_scratch_$SLURM_JOB_ID
export CODEX_HOME=$SCRATCH/home TMPDIR=$SCRATCH/tmp
mkdir -p "$CODEX_HOME" "$TMPDIR"
cp ~/.codex/auth.json "$CODEX_HOME/" 2>/dev/null || true
echo "CODEX_HOME exists=$([ -d "$CODEX_HOME" ] && echo yes || echo NO) auth=$([ -f "$CODEX_HOME/auth.json" ] && echo yes || echo NO)"
trap 'cp "$CODEX_HOME/auth.json" ~/.codex/auth.json 2>/dev/null; rm -rf "$SCRATCH"' EXIT

WT=/burg-archive/glab/users/pg2328/legoESM/.claude/worktrees/pmodel-lmip
cd "$WT" || exit 1
echo "sha:$(git rev-parse HEAD) prompt:$PROMPT_FILE sandbox:${CODEX_SANDBOX:-read-only}"

PROMPT=$(cat "$PROMPT_FILE")
codex exec --sandbox "${CODEX_SANDBOX:-read-only}" -C "$WT" "$PROMPT" < /dev/null 2>&1 | tail -400
echo "rc=${PIPESTATUS[0]}"
echo CXPROMPT_DONE
