#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=regress
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/regress_%j.out
#
# Regression check after merging 651 commits of main (Dhruv's #1645 NEMO-
# faithful bottom-drag time level + control volume, and everything else) into
# the OMIP branch.  The claim to defend: nothing the OMIP campaign relies on
# broke, and the merge did not silently change the run driver's contract.
#
# CPU only: imports + unit tests.  A GPU smoke is the stronger check but the
# two mixing-length arms hold both GPUs; this catches import breaks, API drift
# and contract breaks from 651 commits, which is what a merge most often does.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

echo "=== 1. the run driver and the instrument import cleanly ==="
$PY - <<'PY'
import importlib.util, sys
from pathlib import Path
for rel in ("scripts/run/run_omip_core2.py",
            "scripts/validate/ocean_fidelity/equatorial_diffusivity_oracle.py",
            "scripts/validate/ocean_fidelity/equatorial_thermocline_structure.py"):
    p = Path(rel)
    s = importlib.util.spec_from_file_location(p.stem, p)
    m = importlib.util.module_from_spec(s); sys.modules[p.stem] = m
    s.loader.exec_module(m)
    print("import OK:", rel)
# The model method the instrument depends on must still exist on the class.
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    LatLonCGridOceanModel as M)
assert hasattr(M, "diagnose_vertical_K"), "diagnose_vertical_K vanished in merge"
assert hasattr(M, "tendencies")
print("diagnose_vertical_K present on the model:", True)
PY
rc=$?; [ $rc -eq 0 ] || { echo "FATAL: import/contract check failed rc=$rc"; exit 1; }

echo "=== 2. OMIP instrument + closure tests (this campaign's own) ==="
$PY -m pytest -q \
  tests/unit/test_snapshot_kprofiles.py \
  tests/ocean/unit/test_equatorial_diffusivity_oracle.py \
  tests/ocean/unit/test_equatorial_thermocline_structure.py \
  tests/unit/test_run_omip_core2_eq_visc_shaping.py \
  tests/ocean/unit/test_core2_corrected_cache_default.py \
  tests/ocean/unit/test_tripole_vmix_cli.py \
  2>&1 | tail -4

echo "=== 3. vertical-mixing / tke / bottom-drag (Dhruv's #1645 area) ==="
$PY -m pytest -q \
  tests/ocean/unit/test_zdf_face_control_volume.py \
  tests/unit/test_diagnostic_heat_flux.py \
  tests/ocean/unit/test_no_scheme_duplication.py \
  tests/unit/test_run_omip_core2_restart.py \
  tests/ocean/unit/test_ocean_run_restart.py \
  2>&1 | tail -4

echo "=== 4. dispatch / contract ratchets (merge most often breaks these) ==="
$PY -m pytest -q \
  tests/test_dispatch_hardening.py \
  tests/test_validate_strict_coverage.py \
  tests/unit/test_inert_params.py \
  2>&1 | tail -4
echo DONE
