#!/bin/bash
#SBATCH --job-name=i1783_zref
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/i1783_zref.%j.out
# Issue #1783, the discriminating measurement named in the issue and never run:
# force surface_z_ref_model_level=False in the TestUnifiedLaneOneFluxLaw fixture
# and re-run the class.
#   green  -> the height switch IS the regression; fix is a lane-level default.
#   red    -> the height switch is not the cause; suspect list reopens.
# Arm A is the untouched baseline on this same tree, so the pair differs in
# exactly one constructor field.
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/_sweep0923_wt
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
[ -x "$PY" ] || { echo "FATAL: interpreter $PY missing"; exit 2; }
cd "$WT"
export PYTHONPATH="$WT/src:$WT/packages/atmosphere:$WT/packages/core:$WT/packages/coupler:$WT/packages/ice:$WT/packages/land:$WT/packages/ml:$WT/packages/ocean:$WT/packages/tools:$WT"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1

T=tests/unit/test_land_interface_flux.py
echo "HEAD: $(git -C "$WT" log --oneline -1)"
git -C "$WT" checkout -q -- "$T"

echo "================= ARM A: baseline (main default, z_ref_model_level=True) ================="
"$PY" -m pytest -q "$T::TestUnifiedLaneOneFluxLaw" 2>&1 | tail -30

echo "================= ARM B: surface_z_ref_model_level=False in the fixture ================="
python3 - "$T" <<'PY'
import sys, re
p = sys.argv[1]
s = open(p).read()
old = 'diurnal_cycle=False, slab_land_active=True, topography="etopo",'
new = ('diurnal_cycle=False, slab_land_active=True, topography="etopo",\n'
       '        surface_z_ref_model_level=False,')
assert s.count(old) == 1, f"anchor count {s.count(old)}"
open(p, "w").write(s.replace(old, new))
print("PATCHED: surface_z_ref_model_level=False injected into _pipeline")
PY
sed -n '61,80p' "$T"
"$PY" -m pytest -q "$T::TestUnifiedLaneOneFluxLaw" 2>&1 | tail -30

echo "================= ARM C: whole file, z_ref off (collateral check) ================="
"$PY" -m pytest -q "$T" 2>&1 | tail -15

git -C "$WT" checkout -q -- "$T"
echo "REVERTED: $(git -C "$WT" status --porcelain "$T" | wc -l) dirty lines for $T"
echo DONE
