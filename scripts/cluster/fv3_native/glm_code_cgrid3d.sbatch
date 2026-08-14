#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=glm_code
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/glm/glm_code_%j.log
# GLM-5.3 as AUTHOR (user directive 2026-08-14); codex and Claude review.
set -uo pipefail
if [ -n "${SLURM_JOB_ID:-}" ] && command -v scontrol >/dev/null 2>&1; then
  _SELF=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | tr ' ' '\n' | sed -n 's/^Command=//p' | head -1)
fi
[ -n "${_SELF:-}" ] || _SELF="${BASH_SOURCE[0]}"
REPO=$(cd "$(dirname "$_SELF")/../../.." && pwd)
[ -e "$REPO/.git" ] || { echo "REPO=$REPO not a worktree"; exit 5; }
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
G=/burg-archive/glab/users/pg2328/fv3_duo_gaps/glm
echo "host: $(hostname)"; cd $REPO; echo "PINNED sha: $(git rev-parse HEAD)"
$PY $REPO/scripts/validate/glm_review.py \
    --prompt-file $G/prompt_cgrid3d.txt \
    --model glm-5.3 --max-tokens 64000 \
    --out $G/fv3_cgrid_phase_3d.generated.py
echo "rc=$?"
echo "--- generated ---"; wc -l $G/fv3_cgrid_phase_3d.generated.py 2>/dev/null
$PY -c "import ast,sys; ast.parse(open('$G/fv3_cgrid_phase_3d.generated.py').read()); print('PARSES OK')" 2>&1 | tail -2
echo GLM_CODE_DONE
