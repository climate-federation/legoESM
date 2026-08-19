#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=fv3bkparity
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/fv3_bkparity_%j.log
# THE EXIT CRITERION for phases 2 and 3: the ported JAX lane scored
# against the pinned Zenodo Fortran, through the SAME runner that
# produced the NumPy lane's established numbers.
#
# Both backends run IN THIS ONE JOB, back to back, so the comparison is
# controlled by construction: identical IC, identical oracle files,
# identical derived face map, identical tendency scales and tolerances.
# Only `--backend` differs. Running them in separate jobs would leave
# the door open to a drifted input and turn any difference into a
# confound -- the failure this campaign has paid for more than once.
#
# ARM selects hydro (default) or nh:
#   sbatch --export=ALL,ARM=nh full_step_backend_parity.sbatch
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3jax
[ -e "$REPO/.git" ] || { echo "PREFLIGHT FAILED: REPO=$REPO"; exit 5; }
DIRTY=$(git -C "$REPO" status --porcelain | wc -l)
[ "$DIRTY" -eq 0 ] || { echo "PREFLIGHT FAILED: dirty ($DIRTY)"; exit 3; }
echo "host: $(hostname)"
echo "PINNED sha: $(git -C "$REPO" rev-parse HEAD)"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH=$REPO/packages/core:$REPO/packages/atmosphere:$REPO/src
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu
cd "$REPO"
ARM="${ARM:-hydro}"
EXTRA=""
[ "$ARM" = "nh" ] && EXTRA="--nh"
OUT=/burg-archive/glab/users/pg2328/fv3_duo_gaps/bkparity_${ARM}_${SLURM_JOB_ID:-0}
mkdir -p "$OUT"

for BK in numpy jax; do
  echo "########## ARM=$ARM BACKEND=$BK"
  timeout 10800 $PY scripts/validate/fv3_native/full_step_oracle_parity.py \
    --backend "$BK" $EXTRA --json "$OUT/${BK}.json" > "$OUT/${BK}.log" 2>&1
  RC=$?
  echo "rc=$RC"
  # A backend that DIED is not a backend that agreed. Without this the loop
  # ran on, the comparison below found no shared metrics, and the job still
  # reported success.
  [ "$RC" -eq 0 ] || { echo "FAIL: backend $BK exited $RC"; exit "$RC"; }
  # The decisive lines only: the residual table and the port's own
  # tendency, which is what separates "wrong in detail" from "wrong
  # step". The full log stays in $OUT for either.
  sed -n '/one-step residual/,/^$/p' "$OUT/${BK}.log" | head -25
  grep -E "^(FAIL|PASS|WORST|verdict)" "$OUT/${BK}.log" | head -5
done

echo "########## BACKEND DELTA (jax vs numpy, same everything else)"
$PY - "$OUT/numpy.json" "$OUT/jax.json" <<'EOF'
import json, sys
try:
    a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
except Exception as e:            # a missing json means an arm died
    print(f"cannot compare: {e}")
    raise SystemExit(4)               # NOT success: nothing was compared

def flat(d, p=""):
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            yield from flat(v, f"{p}{k}.")
        elif isinstance(v, (int, float)):
            yield f"{p}{k}", float(v)

fa, fb = dict(flat(a)), dict(flat(b))
keys = [k for k in fa if k in fb]
if not keys:
    print("no shared numeric keys between the two runs")
    raise SystemExit(4)               # NOT success: nothing was compared
print(f"{'metric':50s} {'numpy':>12s} {'jax':>12s} {'rel':>10s}")
worst = (0.0, None)
for k in sorted(keys):
    x, y = fa[k], fb[k]
    den = max(abs(x), abs(y), 1e-300)
    rel = abs(x - y) / den
    if rel > worst[0]:
        worst = (rel, k)
    print(f"{k:50s} {x:12.5e} {y:12.5e} {rel:10.2e}")
print(f"\nWORST BACKEND-TO-BACKEND rel = {worst[0]:.3e} at {worst[1]}")

# THE EXIT CRITERION, enforced. This job is the evidence for the claim that
# the two backends agree; printing a number and exiting zero made it a job
# that could not fail, so a 1e-2 wind error would have been reported as a
# passing parity run. The bound defaults to EXACT because that is what the
# claim says: bit-identical between backends.
import os
bound = float(os.environ.get("MAX_REL", "0.0"))
if worst[0] > bound:
    print(f"FAIL: backend-to-backend {worst[0]:.3e} exceeds MAX_REL={bound:g} "
          f"at {worst[1]}")
    raise SystemExit(1)
print(f"PASS: backend-to-backend within MAX_REL={bound:g}")
EOF
BKRC=$?
echo "BKPARITY_DONE rc=$BKRC"
exit $BKRC
