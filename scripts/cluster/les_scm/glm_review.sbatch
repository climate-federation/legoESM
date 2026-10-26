#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=glm_rev
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/glm_rev_%j.out
set +e
# GLM review via the direct Z.AI endpoint. The MCP bridge caps at 120 s and
# timed out on all three attempts at a long generation; this route has no such
# ceiling. Prompt comes from a FILE so no shell quoting can mangle the diff.
: "${G_PROMPT:?set G_PROMPT to the prompt file}"
WT=/burg-archive/glab/users/pg2328/legoESM
cd "$WT" || exit 1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
echo "prompt: $G_PROMPT ($(wc -l < "$G_PROMPT") lines)"
$PY scripts/tmp/_ask_glm.py "$G_PROMPT" "${G_MODEL:-glm-4.6}"
echo "glm rc=$?"
echo "DONE $(date +%T)"
