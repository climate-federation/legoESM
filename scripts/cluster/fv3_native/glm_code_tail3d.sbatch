#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=glm_code
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/glm/glm_code_%j.log
# GLM-5.3 as AUTHOR (user directive 2026-08-14); codex and Claude review.
# Module 3 of the 3-D layer: the D-grid tail (d_sw3 -> BARRIER 2 -> d_sw4/5/6,
# the D-grid geopk + one_grad_p, and the NH D-grid pressure tail).
set -uo pipefail
if [ -n "${SLURM_JOB_ID:-}" ] && command -v scontrol >/dev/null 2>&1; then
  _SELF=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | tr ' ' '\n' | sed -n 's/^Command=//p' | head -1)
fi
[ -n "${_SELF:-}" ] || _SELF="${BASH_SOURCE[0]}"
REPO=$(cd "$(dirname "$_SELF")/../../.." && pwd)
[ -e "$REPO/.git" ] || { echo "REPO=$REPO not a worktree"; exit 5; }
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
G=/burg-archive/glab/users/pg2328/fv3_duo_gaps/glm
OUT=$G/fv3_dsw_tail_3d.generated.py
echo "host: $(hostname)"; cd $REPO; echo "PINNED sha: $(git rev-parse HEAD)"
$PY $REPO/scripts/validate/glm_review.py \
    --prompt-file $G/prompt_tail3d.txt \
    --model glm-5.3 --max-tokens 32000 --code --no-stream \
    --out $OUT
echo "rc=$?"
echo "--- generated ---"; wc -l $OUT 2>/dev/null
# A bare ast.parse() on an EMPTY file succeeds -- that vacuous check reported
# "PARSES OK" on the 0-byte first attempt. Assert non-empty AND parseable.
$PY - <<PYEOF
import ast, pathlib, sys
f = pathlib.Path("$OUT")
src = f.read_text() if f.is_file() else ""
if len(src.strip()) < 500:
    print(f"EMPTY-OR-STUB: {len(src)} chars"); sys.exit(1)
ast.parse(src)
defs = [n.name for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)]
print(f"PARSES OK: {len(src)} chars, {src.count(chr(10))} lines")
print("top-level defs:", defs)
want = ["dsw_tail_phase_3d", "dgrid_pressure_phase_3d", "nh_exchanged_area6",
        "dgrid_nh_pressure_phase_3d"]
missing = [w for w in want if w not in defs]
print("MISSING PUBLIC ROUTINES:", missing if missing else "none")
PYEOF
echo GLM_CODE_DONE
