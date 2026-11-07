#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=cxm4b
#SBATCH --time=00:50:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=8G
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/codex_m4b_%j.log
set -uo pipefail
export CODEX_HOME=/tmp/codex_home_$SLURM_JOB_ID; mkdir -p "$CODEX_HOME"; cp ~/.codex/auth.json "$CODEX_HOME/" 2>/dev/null || true
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3jax; cd "$REPO"
codex exec --sandbox read-only -C "$REPO" "$(cat /burg-archive/glab/users/pg2328/fv3_duo_gaps/codex_m4b_prompt.txt)" 2>&1 | tail -40
echo "rc=${PIPESTATUS[0]}"; echo CXM4B_DONE
