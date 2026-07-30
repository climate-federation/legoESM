#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=scmrce_t4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/scmrce_test_r4_%j.out
set +e
WT=/burg/glab/users/pg2328/scmrce_wt6
cd $WT
echo "PINNED sha: $(git rev-parse HEAD)"; echo "host: $(hostname)"
export JAX_PLATFORMS=cpu
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$PYTHONPATH
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

echo "=================== THREADING + OVERRIDE ==================="
JAX_ENABLE_X64=1 $PY -m pytest tests/unit/test_convection_subsidence_solve_threading.py \
  tests/unit/test_scm_rce_subsidence_solve_override.py -q 2>&1 | tail -30
echo "=================== REGRESSION (schemes touched) ==================="
JAX_ENABLE_X64=1 $PY -m pytest tests/unit/test_tiedtke.py tests/unit/test_emanuel.py \
  tests/unit/test_zhang_mcfarlane.py tests/unit/test_bechtold.py \
  tests/unit/test_bechtold_implicit_flux.py tests/unit/test_kain_fritsch.py tests/atmosphere/hydrostatic/unit/test_edmf_convection_824.py \
  tests/atmosphere/hydrostatic/unit/test_convection.py -q 2>&1 | tail -25
echo "=================== CI RATCHETS ==================="
JAX_ENABLE_X64=1 $PY -m pytest tests/test_param_specs.py tests/test_no_inline_physics_coeffs.py \
  tests/test_dispatch_hardening.py tests/test_validate_strict_coverage.py \
  tests/test_physics_contracts.py tests/test_no_private_cross_imports.py \
  tests/test_no_hardcoded_constants.py tests/test_no_saturation_reimpl.py -q 2>&1 | tail -25
echo "=================== VALIDATOR (Test 5 kernel sweep) ==================="
JAX_ENABLE_X64=1 $PY scripts/validate/validate_convection_physics.py \
  --json-out /burg-archive/glab/users/pg2328/legoESM/results/scm_rce_paper/kernel_conservation.json 2>&1 | tail -60
echo "validator rc=$?"
echo ALLDONE
