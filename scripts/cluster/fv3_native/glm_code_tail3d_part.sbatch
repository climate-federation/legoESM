#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=glm_tailpart
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/glm/glm_code_%j.log
# GLM-5.3 as AUTHOR, module 3 of the 3-D layer, ONE PART PER JOB.
#
#   sbatch --export=ALL,PART=A glm_code_tail3d_part.sbatch
#
# WHY SPLIT.  The whole-module call died twice with IncompleteRead (jobs
# 9417290 at 10511 bytes, 9417297 at 13407), and --no-stream is NOT optional
# here: thinking={"type":"disabled"} is ignored on the streaming endpoint
# (probe 687d20a82 -- 31991 reasoning tokens, EMPTY content), and with thinking
# on this model spends the whole budget reasoning. So the lever that is left is
# the LENGTH of one completion, and three ~400-line parts each finish inside
# the window that one ~1200-line module did not.
#   A: dsw_tail_phase_3d           (d_sw3 -> BARRIER 2 -> d_sw4/5/6)
#   B: dgrid_pressure_phase_3d     (D-grid geopk + one_grad_p)
#   C: nh_exchanged_area6 + dgrid_nh_pressure_phase_3d (the NH D tail)
# A carries the module docstring and imports; B and C are appended to it.
#
# MAXTOK (default 48000).  MEASURED, job 9417346: thinking={"type":"disabled"}
# is honoured only PARTIALLY on this endpoint -- a 17-line function still spent
# 671 of 812 completion tokens reasoning.  Reasoning scales with the task, so a
# 400-line module at --max-tokens 20000 spent ALL 20000 on reasoning and emitted
# nothing (jobs 9417315/16/17 attempt 1).  The budget therefore has to cover
# reasoning AND content, not content alone.
set -uo pipefail
PART=${PART:?usage: sbatch --export=ALL,PART=A|B|C glm_code_tail3d_part.sbatch}
case "$PART" in A|B|C) ;; *) echo "PART must be A, B or C"; exit 2 ;; esac
if [ -n "${SLURM_JOB_ID:-}" ] && command -v scontrol >/dev/null 2>&1; then
  _SELF=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | tr ' ' '\n' | sed -n 's/^Command=//p' | head -1)
fi
[ -n "${_SELF:-}" ] || _SELF="${BASH_SOURCE[0]}"
REPO=$(cd "$(dirname "$_SELF")/../../.." && pwd)
[ -e "$REPO/.git" ] || { echo "REPO=$REPO not a worktree"; exit 5; }
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
G=/burg-archive/glab/users/pg2328/fv3_duo_gaps/glm
PROMPT=$G/prompt_tail3d_${PART}.txt
OUT=$G/fv3_dsw_tail_3d.part${PART}.py
[ -s "$PROMPT" ] || { echo "missing prompt $PROMPT"; exit 6; }
echo "host: $(hostname)"; cd $REPO; echo "PINNED sha: $(git rev-parse HEAD)"
echo "PART=$PART prompt=$(wc -c < $PROMPT) chars"
rm -f "$OUT"
for attempt in 1 2 3; do
  echo "--- attempt $attempt ---"
  $PY $REPO/scripts/validate/glm_review.py \
      --prompt-file "$PROMPT" \
      --model glm-5.3 --max-tokens ${MAXTOK:-48000} --code --no-stream \
      --out "$OUT"
  echo "rc=$?"
  GOT=$( [ -f "$OUT" ] && wc -c < "$OUT" || echo 0 )
  echo "attempt $attempt produced $GOT chars"
  # The retry gate is on the OUTPUT FILE, never on the exit code: a partial
  # read exits non-zero AND leaves nothing, and a good call can still print
  # warnings.
  [ "$GOT" -gt 5000 ] && break
done
echo "--- generated ---"; wc -l "$OUT" 2>/dev/null
$PY - <<PYEOF
import ast, pathlib, sys
f = pathlib.Path("$OUT")
src = f.read_text() if f.is_file() else ""
if len(src.strip()) < 500:
    print(f"EMPTY-OR-STUB: {len(src)} chars"); sys.exit(1)
try:
    mod = ast.parse(src)
except SyntaxError as exc:
    print(f"SYNTAX ERROR at line {exc.lineno}: {exc.msg}"); sys.exit(1)
defs = [n.name for n in mod.body if isinstance(n, ast.FunctionDef)]
print(f"PARSES OK: {len(src)} chars, {src.count(chr(10))} lines")
print("top-level defs:", defs)
want = {"A": ["dsw_tail_phase_3d"],
        "B": ["dgrid_pressure_phase_3d"],
        "C": ["nh_exchanged_area6", "dgrid_nh_pressure_phase_3d"]}["$PART"]
missing = [w for w in want if w not in defs]
print("MISSING PUBLIC ROUTINES:", missing if missing else "none")
PYEOF
echo GLM_CODE_DONE
