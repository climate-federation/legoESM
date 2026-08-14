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
    --model glm-5.3 --max-tokens 32000 --code \
    --out $G/fv3_cgrid_phase_3d.generated.py
echo "rc=$?"
echo "--- generated ---"; wc -l $G/fv3_cgrid_phase_3d.generated.py 2>/dev/null
# A bare ast.parse() on an EMPTY file succeeds -- that vacuous check reported
# "PARSES OK" on the 0-byte first attempt. Assert non-empty AND parseable.
$PY - <<'PYEOF'
import ast, pathlib, sys
f = pathlib.Path('/burg-archive/glab/users/pg2328/fv3_duo_gaps/glm/fv3_cgrid_phase_3d.generated.py')
src = f.read_text() if f.is_file() else ""
if len(src.strip()) < 500:
    print(f"EMPTY-OR-STUB: {len(src)} chars"); sys.exit(1)
ast.parse(src)
print(f"PARSES OK: {len(src)} chars, {src.count(chr(10))} lines, "
      f"{sum(1 for l in src.splitlines() if l.startswith(chr(100)+chr(101)+chr(102)+chr(32)))} defs")
PYEOF
echo GLM_CODE_DONE
