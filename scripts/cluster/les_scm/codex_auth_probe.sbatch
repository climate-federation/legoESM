#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=cdxprobe
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=0:10:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/cdxprobe_%j.out
set +e
WT=/burg-archive/glab/users/pg2328/legoESM
cd "$WT" || exit 1
export CODEX_HOME=/burg-archive/home/pg2328/.codex
echo "auth mtime: $(stat -c '%y' $CODEX_HOME/auth.json)"
timeout 120 codex exec --sandbox read-only -C "$WT" "Reply with exactly: CODEX_OK" 2>&1 | tail -15
echo "codex rc=$?"
echo "DONE $(date +%T)"
