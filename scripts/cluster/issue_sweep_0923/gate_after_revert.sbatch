#!/bin/bash
#SBATCH --job-name=gate_revert
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/gate_revert.%j.out
# After retracting the #1783 pin and adding the post-init coordinate check.
#
# The #1783 class is EXPECTED to be red again here: the retraction restores
# the state the issue describes, and the real fix (the class's reference is
# stale) belongs in the test, which is a separate change. Recording that red
# explicitly so it is not mistaken for a regression introduced by the
# coordinate work.
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/_sweep0923_wt
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
cd "$WT"
export PYTHONPATH="$WT/src:$WT/packages/atmosphere:$WT/packages/core:$WT/packages/coupler:$WT/packages/ice:$WT/packages/land:$WT/packages/ml:$WT/packages/ocean:$WT/packages/tools:$WT"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
echo "HEAD: $(git -C "$WT" log --oneline -1)"

echo "########## #1783 class -- expected RED again after the retraction ##########"
"$PY" -m pytest -q tests/unit/test_land_interface_flux.py 2>&1 | tail -10

echo "########## the surface-height tests must be green with the pin gone ##########"
"$PY" -m pytest -q tests/unit/test_surface_reference_height.py \
    tests/unit/test_surface_layer_height_and_saline.py \
    tests/unit/test_aimip_surface_bulk_scheme.py 2>&1 | tail -10

echo "########## #1029 guard: still green with the post-init check added ##########"
"$PY" -m pytest -q tests/unit/test_hybrid_negative_layer_mass.py \
    tests/unit/test_hybrid_level_validity.py 2>&1 | tail -8

echo "########## the sweep's own probe tests ##########"
"$PY" -m pytest -q tests/unit/test_issue_sweep_0923_probes.py 2>&1 | tail -6

echo "########## the post-init check must be REACHED, not dead code ##########"
"$PY" - <<'PYEOF'
import ast, pathlib
src = pathlib.Path("packages/coupler/legoesm/driver/model_driver.py").read_text()
for name in ("_assert_vertical_coordinate_supports_this_orography",
             "_assert_vertical_coordinate_supports_this_state"):
    defs = src.count(f"def {name}")
    calls = src.count(f"self.{name}()")
    print(f"  {name}: {defs} definition, {calls} call site")
    assert defs == 1 and calls == 1, (name, defs, calls)
PYEOF
echo DONE
