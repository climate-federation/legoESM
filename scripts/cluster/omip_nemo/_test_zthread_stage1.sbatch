#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=zthread1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/zthread1_%j.out
# Stage-1 gate for the SPMD full-card build: the z_coord/iwm_fields override
# threading must leave the DEFAULT path bit-identical.  Runs lint (F-class)
# plus the ocean unit suites that exercise the latlon/tripole step drivers.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
echo "HEAD+dirty: $(git log --oneline -1); $(git status --porcelain -- packages/ocean | head -2)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
echo "=== ruff F-class on the edited file ==="
$PY -m ruff check --select F,E9 packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py 2>&1 | tail -5
echo "=== ocean unit tests exercising the step drivers ==="
$PY -m pytest -q -x tests/ocean/unit/test_tripole_vmix_cli.py \
  tests/ocean/unit/test_tke_nemo_nn_mxl2.py \
  tests/ocean/unit/test_momentum_diagnostics_closure.py \
  tests/ocean/unit/test_implicit_vmix_face_control_volume.py 2>&1 | tail -4
echo "=== broader ocean unit slice (no -x, count failures) ==="
$PY -m pytest -q tests/ocean/unit -k "tke or implicit or ab2 or leapfrog or mlf or step" 2>&1 | tail -3
echo DONE
