#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=glm_author
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/glm/glm_author_%j.log
# GLM-5.3 authors ONE part of ONE module of the FV3 JAX lane.
#
#   sbatch --export=ALL,PROMPT=<file>,OUT=<file>,WANT="a,b,c" \
#          glm_author_module.sbatch
#
# Generalises glm_code_tail3d_part.sbatch, whose prompt paths and expected
# routine names were baked in. Everything measured there applies here and is
# the reason for the defaults: thinking={"type":"disabled"} is honoured only
# PARTIALLY (671/812 reasoning tokens on a 17-line function), reasoning scales
# with the task, and at a 48k ceiling a 400-line module spent 41-43k on it and
# the code was TRUNCATED. So: stream, budget for reasoning PLUS content, and
# cap the docstring in the prompt.
set -uo pipefail
PROMPT=${PROMPT:?usage: --export=ALL,PROMPT=...,OUT=...,WANT=a,b}
OUT=${OUT:?missing OUT}
WANT=${WANT:?missing WANT (comma-separated top-level defs that must exist)}
[ -s "$PROMPT" ] || { echo "missing prompt $PROMPT"; exit 6; }
if [ -n "${SLURM_JOB_ID:-}" ] && command -v scontrol >/dev/null 2>&1; then
  _SELF=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | tr ' ' '\n' | sed -n 's/^Command=//p' | head -1)
fi
[ -n "${_SELF:-}" ] || _SELF="${BASH_SOURCE[0]}"
REPO=$(cd "$(dirname "$_SELF")/../../.." && pwd)
[ -e "$REPO/.git" ] || { echo "REPO=$REPO not a worktree"; exit 5; }
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
STREAM_FLAG=$([ "${STREAM:-1}" = "0" ] && echo --no-stream || echo "")
echo "host: $(hostname)"; cd $REPO; echo "PINNED sha: $(git rev-parse HEAD)"
echo "PROMPT=$PROMPT ($(wc -c < $PROMPT) chars)  OUT=$OUT  WANT=$WANT"
rm -f "$OUT"
for attempt in 1 2 3; do
  echo "--- attempt $attempt ---"
  $PY $REPO/scripts/validate/glm_review.py --prompt-file "$PROMPT" \
      --model glm-5.3 --max-tokens ${MAXTOK:-96000} --code ${STREAM_FLAG} \
      --out "$OUT"
  echo "rc=$?"
  GOT=$( [ -f "$OUT" ] && wc -c < "$OUT" || echo 0 )
  echo "attempt $attempt produced $GOT chars"
  # The retry gate is on the OUTPUT FILE, never the exit code.
  [ "$GOT" -gt 5000 ] && break
done
echo "--- generated ---"; wc -l "$OUT" 2>/dev/null
WANT="$WANT" OUT="$OUT" $PY - <<'PYEOF'
import ast, os, pathlib, sys
f = pathlib.Path(os.environ["OUT"])
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
want = [w for w in os.environ["WANT"].split(",") if w]
missing = [w for w in want if w not in defs]
print("MISSING PUBLIC ROUTINES:", missing if missing else "none")
# TRUNCATION: a cut can land on a valid expression (`u_lev, v_` is a tuple),
# so ast.parse alone reported "PARSES OK" on a half-written module.
lines = src.rstrip().splitlines()
ok = not missing
last = mod.body[-1] if mod.body else None
if last is None or getattr(last, "end_lineno", 0) < len(lines):
    print(f"TRUNCATED: last top-level node ends at line "
          f"{getattr(last, 'end_lineno', 0)} of {len(lines)}")
    ok = False
if isinstance(last, ast.Expr) and not isinstance(last.value, ast.Constant):
    print(f"TRUNCATED: file ends in a bare expression (line {last.lineno})")
    ok = False
print("COMPLETENESS:", "OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
PYEOF
echo GLM_AUTHOR_DONE
