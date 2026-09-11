#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=fv3batparity
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/fv3_batparity_%j.log
# Score the FACE-BATCHED dynamics arm -- the one the window/SPMD step
# forces -- against the pinned Fortran oracle.
#
# Both arms run IN THIS ONE JOB, back to back, so only `--batched`
# differs: same IC, same oracle files, same derived face map, same
# tendency scales, same tolerances. A separate job per arm would leave a
# drifted input free to masquerade as an arm difference.
#
# The job also runs the harness's OWN dispatch test and then REVERSES the
# forwarding to prove that test can fail. A dispatch test that passes with
# the feature removed certifies nothing.
set -uo pipefail
# --max-rel gates each arm against the oracle. BATCHED_VS_LOOP_MAX_REL
# separately gates the batched arm against the loop arm.
: "${BATCHED_VS_LOOP_MAX_REL:?BATCHED_VS_LOOP_MAX_REL is required; a comparison with no ceiling reports a number and certifies nothing}"
REPO=${REPO:-/burg-archive/glab/users/pg2328/legoESM_codex_m8}
[ -e "$REPO/.git" ] || { echo "PREFLIGHT FAILED: REPO=$REPO"; exit 5; }
DIRTY=$(git -C "$REPO" status --porcelain | wc -l)
[ "$DIRTY" -eq 0 ] || { echo "PREFLIGHT FAILED: dirty ($DIRTY)"; exit 3; }
echo "host: $(hostname)"
echo "PINNED sha: $(git -C "$REPO" rev-parse HEAD)"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH=$REPO/packages/core:$REPO/packages/atmosphere:$REPO/src
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu
cd "$REPO"
H=scripts/validate/fv3_native/full_step_oracle_parity.py
JIT="${JIT:-0}"
JFLAG=""; [ "$JIT" = 1 ] && JFLAG="--jit"

echo "########## 1. the dispatch test, as written"
$PY -m pytest tests/grids/test_fv3_duo_parity_harness.py -q 2>&1 | tail -5

echo "########## 2. NON-VACUITY: delete the forwarding, the test must go red"
cp "$H" /tmp/_h_backup_$$.py
$PY - "$H" <<'PYEOF'
import re, sys
p = sys.argv[1]
s = open(p).read()
n = 0
s2, k = re.subn(r"jctx, _static\.pop\(\"km\"\), batched=batched, \*\*_static",
                "jctx, _static.pop(\"km\"), **_static", s); n += k
s3, k = re.subn(r"jctx, jstate, jpress, q=jq, batched=batched, \*\*kw",
                "jctx, jstate, jpress, q=jq, **kw", s2); n += k
assert n == 2, f"reversal patched {n} sites, expected 2 -- the control is broken"
open(p, "w").write(s3)
print("reversal applied to 2 forwarding sites")
PYEOF
$PY -m pytest tests/grids/test_fv3_duo_parity_harness.py -q 2>&1 | tail -4
echo "(a FAILED line above is the REQUIRED outcome of step 2)"
cp -f /tmp/_h_backup_$$.py "$H"; rm -f /tmp/_h_backup_$$.py
git -C "$REPO" diff --quiet -- "$H" && echo "restored clean" || { echo "RESTORE FAILED"; exit 6; }

ARM="${ARM:-hydro}"
case "$ARM" in
  hydro) EXTRA="--max-rel 2e-9" ;;
  nh)    EXTRA="--nh --max-rel 1e-3" ;;
  moist) EXTRA="--moist --tracers --max-rel 2e-9" ;;
  *) echo "unknown ARM=$ARM"; exit 4 ;;
esac
OUT=/burg-archive/glab/users/pg2328/fv3_duo_gaps/batparity_${ARM}_${SLURM_JOB_ID:-0}
mkdir -p "$OUT"

for BAT in loop batched; do
  FLAG=""; [ "$BAT" = batched ] && FLAG="--batched"
  echo "########## 3. ARM=$ARM jax $BAT"
  timeout 14400 $PY "$H" --backend jax $FLAG $JFLAG $EXTRA \
    --json "$OUT/${BAT}.json" > "$OUT/${BAT}.log" 2>&1
  RC=$?
  echo "rc=$RC"
  grep -E "step arm:" "$OUT/${BAT}.log" | head -2
  sed -n '/one-step residual/,/^$/p' "$OUT/${BAT}.log" | head -25
  grep -E "^(FAIL|PASS|WORST|verdict)" "$OUT/${BAT}.log" | head -5
  [ "$RC" -eq 0 ] || exit "$RC"
done

echo "########## 4. ARM DELTA (batched vs loop, same everything else)"
$PY - "$OUT/loop.json" "$OUT/batched.json" "$JIT" <<'PYEOF'
import json, sys
try:
    a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
except Exception as e:
    print(f"cannot compare: {e}"); raise SystemExit(1)
assert a.get("batched") is False and b.get("batched") is True, (
    "the two runs do not record the arms they claim: "
    f"{a.get('batched')} / {b.get('batched')}")
expected_compiled = sys.argv[3] == "1"
assert (a.get("compiled") is expected_compiled
        and b.get("compiled") is expected_compiled), (
    "the two runs do not record the compiled arm they claim: "
    f"{a.get('compiled')} / {b.get('compiled')} "
    f"(expected {expected_compiled})")
def flat(d, p=""):
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            yield from flat(v, f"{p}{k}.")
        elif (k not in BOOKKEEPING_KEYS
              and isinstance(v, (int, float))
              and not isinstance(v, bool)):
            yield f"{p}{k}", float(v)
BOOKKEEPING_KEYS = {"batched", "compiled", "n_steps"}
fa, fb = dict(flat(a)), dict(flat(b))
keys = [k for k in fa if k in fb]
if not keys:
    print("no shared numeric measurement keys between the two runs")
    raise SystemExit(1)
print(f"{'metric':50s} {'loop':>12s} {'batched':>12s} {'rel':>10s}")
worst = (0.0, None)
for k in sorted(keys):
    x, y = fa[k], fb[k]
    den = max(abs(x), abs(y), 1e-300)
    rel = abs(x - y) / den
    if rel > worst[0]: worst = (rel, k)
    print(f"{k:50s} {x:12.5e} {y:12.5e} {rel:10.2e}")
print(f"\nWORST ARM-TO-ARM rel = {worst[0]:.3e} at {worst[1]}")
import math, os
bound = float(os.environ["BATCHED_VS_LOOP_MAX_REL"])
if not math.isfinite(bound):
    print("FAIL: BATCHED_VS_LOOP_MAX_REL must be finite")
    raise SystemExit(1)
if worst[0] > bound:
    print(f"FAIL: batched-vs-loop {worst[0]:.3e} exceeds "
          f"BATCHED_VS_LOOP_MAX_REL={bound:g} at {worst[1]}")
    raise SystemExit(1)
print(f"PASS: batched-vs-loop within BATCHED_VS_LOOP_MAX_REL={bound:g}")
PYEOF
BKRC=$?
echo "BATPARITY_DONE rc=$BKRC"
exit $BKRC
