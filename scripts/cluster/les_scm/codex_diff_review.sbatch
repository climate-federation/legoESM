#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=cdx_diff
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/cdx_diff_%j.out
set +e
: "${R_PROMPT:?set R_PROMPT (shared, NOT /local)}"
case "$R_PROMPT" in /local/*) echo "REFUSING: $R_PROMPT is node-local"; exit 2;; esac
WT=/burg-archive/glab/users/pg2328/legoESM
cd "$WT" || exit 1
export CODEX_HOME=/burg-archive/home/pg2328/.codex
echo "auth mtime: $(stat -c '%y' $CODEX_HOME/auth.json)"; echo "prompt: $R_PROMPT ($(wc -l < "$R_PROMPT") lines)"
codex exec --sandbox read-only -C "$WT" "$(cat "$R_PROMPT")" 2>&1 | grep -vE "^20[0-9]{2}-.*(refresh|websocket|Failed to)" | tail -90
echo "codex rc=${PIPESTATUS[0]}"; echo "DONE $(date +%T)"
