#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=deardtest
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/deardtest_%j.out
set +e
WT=/burg-archive/glab/users/pg2328/legoESM
cd "$WT" || exit 1
export PYTHONPATH=$WT/src:$(ls -d "$WT"/packages/*/ | tr '\n' ':')$PYTHONPATH
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

echo "=========== new Deardorff tests ==========="
$PY -m pytest -q tests/atmosphere/hydrostatic/unit/test_smagorinsky_deardorff.py 2>&1 | tail -30
echo "new rc=${PIPESTATUS[0]}"

echo "=========== the existing Smagorinsky suite (default must be untouched) ==========="
# The config gained a field and the kernel gained a branch, so every existing
# Smagorinsky test is in the blast radius, not only the one just written.
$PY -m pytest -q tests/atmosphere/hydrostatic/unit/test_smagorinsky_faithful.py \
             tests/unit/test_physics_units.py 2>&1 | tail -25
echo "existing rc=${PIPESTATUS[0]}"

echo "=========== guardrail ratchets (new constants + dispatch) ==========="
$PY -m pytest -q tests/test_no_hardcoded_constants.py tests/test_dispatch_hardening.py \
             tests/test_param_specs.py 2>&1 | tail -20
echo "ratchet rc=${PIPESTATUS[0]}"
echo "DONE $(date +%T)"
