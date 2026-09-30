#!/bin/bash
#SBATCH --account=glab --job-name=codex_impl --partition=short --time=03:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=96G
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/codex_assess_%j.log
# codex AUTHORS here (workspace-write) in its own git worktree; GLM and
# Claude review the diff afterwards.
W=/burg-archive/glab/users/pg2328/legoESM_fv3jax
G=/burg-archive/glab/users/pg2328/fv3_duo_gaps
cd "$W" || exit 1
codex exec -m gpt-5.6-sol --sandbox read-only -C "$W" "$(cat $G/codex_assess.md)" 2>&1
echo "CODEX_RC=$?"

echo DONE codex_assess
