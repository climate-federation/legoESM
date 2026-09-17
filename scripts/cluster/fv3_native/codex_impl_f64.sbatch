#!/bin/bash
#SBATCH --account=glab --job-name=codex_impl --partition=short --time=03:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=96G
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/codex_f64_%j.log
# codex AUTHORS here (workspace-write) in its own git worktree; GLM and
# Claude review the diff afterwards.
W=/burg-archive/glab/users/pg2328/legoESM_codex_f64
G=/burg-archive/glab/users/pg2328/fv3_duo_gaps
cd "$W" || exit 1
codex exec -m gpt-5.6-sol --sandbox workspace-write -C "$W" "$(cat $G/codex_task_f64.md)" 2>&1
echo "CODEX_RC=$?"
cd "$W" && git --no-pager diff HEAD --stat | tail -5
echo DONE codex_f64
