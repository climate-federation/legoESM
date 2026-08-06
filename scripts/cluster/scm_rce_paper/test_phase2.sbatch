#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=scmrce_t2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/scmrce_test_phase2_%j.out
set +e
WT=/burg/glab/users/pg2328/scmrce_wt2
cd $WT
echo "PINNED sha: $(git rev-parse HEAD)"
echo "host: $(hostname)"
export JAX_PLATFORMS=cpu
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$PYTHONPATH
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

echo "=================== NEW THREADING TEST ==================="
JAX_ENABLE_X64=1 $PY -m pytest tests/unit/test_convection_subsidence_solve_threading.py -q 2>&1 | tail -30

echo "=================== REGRESSION: schemes touched ==================="
JAX_ENABLE_X64=1 $PY -m pytest \
  tests/unit/test_tiedtke.py tests/unit/test_emanuel.py \
  tests/unit/test_zhang_mcfarlane.py tests/unit/test_bechtold_implicit_flux.py \
  tests/atmosphere/hydrostatic/unit/test_convection.py \
  -q 2>&1 | tail -25

echo "=================== CI RATCHETS ==================="
JAX_ENABLE_X64=1 $PY -m pytest \
  tests/test_param_specs.py tests/test_no_inline_physics_coeffs.py \
  tests/test_dispatch_hardening.py tests/test_validate_strict_coverage.py \
  tests/test_physics_contracts.py tests/test_no_private_cross_imports.py \
  -q 2>&1 | tail -25

echo "=================== ML PARAMETERIZATION (dt threading) ==================="
JAX_ENABLE_X64=1 $PY -m pytest tests/ -q -k "ml_parameterization" 2>&1 | tail -12
echo ALLDONE
