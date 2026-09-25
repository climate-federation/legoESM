#!/bin/bash
#SBATCH --job-name=gate_1783
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/gate_1783.%j.out
# #1783 acceptance: the unified land-flux lane pins the lowest-level height
# correction OFF, so the class that regressed goes green WITHOUT the test
# fixture being touched.  The measurement that justified this used a fixture
# override; this run must reproduce it from PRODUCTION config alone.
#
# Also runs every other test that reads the same switch, because a lane-level
# pin is a change to every run that selects that lane.
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/_sweep0923_wt
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
[ -x "$PY" ] || { echo "FATAL: interpreter $PY missing"; exit 2; }
cd "$WT"
export PYTHONPATH="$WT/src:$WT/packages/atmosphere:$WT/packages/core:$WT/packages/coupler:$WT/packages/ice:$WT/packages/land:$WT/packages/ml:$WT/packages/ocean:$WT/packages/tools:$WT"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
echo "HEAD: $(git -C "$WT" log --oneline -1)"

echo "########## the class that regressed, from production config ##########"
"$PY" -m pytest -q tests/unit/test_land_interface_flux.py 2>&1 | tail -20

echo "########## every test that reads the reference-height switch ##########"
"$PY" -m pytest -q tests/unit/test_surface_reference_height.py \
    tests/unit/test_surface_layer_height_and_saline.py \
    tests/unit/test_aimip_surface_bulk_scheme.py 2>&1 | tail -20

echo "########## the AMIP CLI round-trips that pin the switch ##########"
"$PY" -m pytest -q tests/unit/test_run_amip_cli.py -k "z_ref or surface or trained" 2>&1 | tail -15

echo "########## NON-VACUITY: drop the pin, the class must go red again ##########"
cp packages/coupler/legoesm/driver/physics_pipeline.py /tmp/_pp_1783_$$.py
python3 - <<'PINPY'
import pathlib
p = pathlib.Path("packages/coupler/legoesm/driver/physics_pipeline.py")
s = p.read_text()
old = '        if zml is None:\n            zml = False\n'
assert s.count(old) == 1, s.count(old)
p.write_text(s.replace(old, '        if zml is None:\n            pass\n', 1))
print("SABOTAGED: unified lane no longer pins the height correction off")
PINPY
"$PY" -m pytest -q tests/unit/test_land_interface_flux.py 2>&1 | tail -8
cp -f /tmp/_pp_1783_$$.py packages/coupler/legoesm/driver/physics_pipeline.py
rm -f /tmp/_pp_1783_$$.py
echo "RESTORED pin"
"$PY" -m pytest -q tests/unit/test_land_interface_flux.py 2>&1 | tail -4
echo DONE
