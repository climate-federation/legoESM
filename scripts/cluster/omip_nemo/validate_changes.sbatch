#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=omip_val
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/val_%j.out

# Rigorous no-regression check of this session's code changes:
#   - omip2_applicator (tripole branch, wind-stress sign fix, in-step forcing,
#     compute_omip2_surface_forcing)
#   - the smc03 PGF docstring edit (latlon_cgrid_operators)
set +e
cd /burg-archive/glab/users/pg2328/legoESM
export PYTHONPATH=/burg-archive/glab/users/pg2328/legoESM/src:$PYTHONPATH
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
echo "=== HOST ==="; hostname; date
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 $PY -m pytest -q \
  tests/ocean/unit/test_omip2_applicator.py \
  tests/ocean/unit/test_pgf_smc03_phase1.py \
  tests/ocean/unit/test_pgf_smc03_phase2.py \
  tests/ocean/unit/test_pgf_smc03_phase3.py 2>&1 | tail -30
echo "pytest_rc=${PIPESTATUS[0]}"
echo "=== compute_omip2_surface_forcing importable? ==="
$PY -c "from legoesm.ocean.coupler import compute_omip2_surface_forcing; print('import OK')" 2>&1 | tail -2
echo "=== DONE ==="
