#!/bin/bash
#SBATCH --job-name=fesomnamefix
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/fesomnamefix_%j.out
# The FESOM adapter called a function that does not exist (_require_fesom_jax,
# left behind by the private-to-public rename in 1c8777e45). Two call sites, one
# of them on the ice-coupling path, which is why a 30-day run died four minutes
# in. This verifies the fix, the NON-VACUITY of the new static test, and that
# nothing else in the ocean suites moved.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu

echo "=== 1. the new test passes on the fixed module ==="
$PY -m pytest -q tests/ocean/unit/test_ocean_model_fesom.py::test_no_undefined_global_names 2>&1 | tail -3

echo
echo "=== 2. NON-VACUITY: with the name the code calls removed, it must FAIL ==="
$PY - <<'PY'
import builtins, dis, types
from legoesm.ocean.dynamics import ocean_model_fesom as mod

# Reproduce the defect's shape: the function the call sites name is absent.
ns = dict(vars(mod))
ns.pop("require_fesom_jax")
known = set(ns) | set(dir(builtins))

def codes(c):
    yield c
    for k in c.co_consts:
        if isinstance(k, types.CodeType):
            yield from codes(k)

missing = set()
for v in vars(mod).values():
    fn = getattr(v, "__func__", v)
    code = getattr(fn, "__code__", None)
    if code is None or getattr(fn, "__module__", None) != mod.__name__:
        continue
    for c in codes(code):
        for ins in dis.get_instructions(c):
            if ins.opname == "LOAD_GLOBAL" and ins.argval not in known:
                missing.add(ins.argval)
assert "require_fesom_jax" in missing, (
    "the checker did NOT notice a missing gate function -- it is vacuous")
print(f"non-vacuity OK: checker flags {sorted(missing)} when the gate is gone")
PY

echo
echo "=== 3. the FESOM adapter suite ==="
$PY -m pytest -q tests/ocean/unit/test_ocean_model_fesom.py 2>&1 | tail -3

echo
echo "=== 4. the OMIP validate suites that read these cards ==="
$PY -m pytest -q tests/validate 2>&1 | tail -3
echo DONE
