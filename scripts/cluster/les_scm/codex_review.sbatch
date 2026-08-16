#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=les_codex
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/les_codex_%j.out
set +e
# Mandatory adversarial review of the LES -> SCM turbulence-tuning stack.
# CLAUDE.md requires codex review after any substantial numerics/physics change.
# The /codex: slash commands crash, so this drives the CLI directly.
#
# TRAP (recorded from a previous session): a CODEX_HOME pointing somewhere
# unwritable makes codex exit 0 having reviewed NOTHING. Use node-local /tmp
# and check for real output below.
# CODEX_WT and CODEX_PROMPT are REQUIRED. A hardcoded worktree here reviewed
# a tree that did not contain the commits under review -- the same class of
# miss as submitting a campaign from a stale pin. Fail loudly instead.
: "${CODEX_WT:?set CODEX_WT to the worktree holding the commits to review}"
: "${CODEX_PROMPT:?set CODEX_PROMPT to a file holding the review prompt}"
[ -f "$CODEX_PROMPT" ] || { echo "no such prompt file: $CODEX_PROMPT"; exit 1; }
WT=$CODEX_WT
cd $WT || exit 1
echo "reviewing: $CODEX_PROMPT"
echo "PINNED sha: $(git rev-parse HEAD)"
echo "host: $(hostname)"
export CODEX_HOME=/tmp/codex_home_$SLURM_JOB_ID
mkdir -p $CODEX_HOME
cp -r /burg-archive/home/pg2328/.codex/* $CODEX_HOME/ 2>/dev/null
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH

OUT=/burg-archive/glab/users/pg2328/legoESM/logs/les_codex_review_$SLURM_JOB_ID.md

PROMPT=$(cat "$CODEX_PROMPT")

echo "$PROMPT" > $CODEX_HOME/prompt.txt
codex exec --sandbox read-only -C $WT "$PROMPT" 2>&1 | tee $OUT
rc=$?
echo "codex rc=$rc"
# Count only lines that are NOT the prompt codex echoes back and NOT its own
# error/log chatter. Counting the raw file made this guard VACUOUS: a run that
# 401'd on every request still "produced" 124 lines -- the prompt plus the
# errors -- and reported itself reviewed. Same class as a log-scraping guard
# that greps its own prompt for the token it is looking for.
grep -vxFf "$CODEX_HOME/prompt.txt" "$OUT" \
  | grep -vE "^2026-|^ERROR|^warning:|Reconnecting|^$" > $OUT.body
lines=$(wc -l < $OUT.body)
echo "review body lines (prompt echo and error chatter excluded): $lines"
if grep -qE "401 Unauthorized|Missing bearer" "$OUT"; then
  echo "REVIEW FAILED: codex could not authenticate (the token in"
  echo "~/.codex/auth.json is stale; 'codex login' refreshes it). Nothing was"
  echo "reviewed. Treat as UNREVIEWED."
  exit 4
fi
if [ "$lines" -lt 20 ]; then
  echo "REVIEW SUSPECT: fewer than 20 lines of REVIEW (not counting the echoed"
  echo "prompt) — codex may have run with a bad CODEX_HOME and reviewed"
  echo "nothing. Treat as UNREVIEWED."
  exit 3
fi
echo "review written to $OUT"
echo "DONE $(date +%T)"
