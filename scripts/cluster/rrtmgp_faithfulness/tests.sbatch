#!/bin/bash
#SBATCH --account=bb1596
#SBATCH --partition=gpu
#SBATCH --job-name=rrtmgp_tests
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100_80
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/b/b381103/rrtmgp_audit/tests_%j.log
WT=/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/wt_rrtmgp_audit
ROOT=/work/bd1083/b309178/diffESM/legoesm_pg/legoESM
A=/scratch/b/b381103/rrtmgp_audit
export PYTHONPATH="$WT/src:$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT"
export JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
module load gcc/13.4.0-gcc-13.4.0 netcdf-fortran/4.6.2-openmpi-5.0.10-gcc-13.4.0
cd $WT
$ROOT/.venv/bin/python scripts/validate/oracle/rrtmgp_faithfulness/audit.py --oracle $A/build/oracle_driver --data $A/ref/rrtmgp-data --out $A/run_ours --fixture $WT/tests/atmosphere/oracle_fixtures/rrtmgp_rte_clear.npz | tail -2
$ROOT/.venv/bin/python -m pytest -q -p no:cacheprovider --no-header -x --co -q tests/unit/test_two_stream_exact_forms.py | tail -1
$ROOT/.venv/bin/python -m pytest -q -p no:cacheprovider --tb=short \
  tests/unit/test_two_stream_exact_forms.py tests/atmosphere/test_rrtmgp_rte_oracle.py \
  tests/unit/test_two_stream_conservative_limit.py tests/unit/test_two_stream_faithful.py tests/unit/test_two_stream_ssa_clamp.py tests/unit/test_two_stream_top_flux.py \
  tests/atmosphere/test_rrtmgp_lw_sfc_face_oracle.py tests/atmosphere/test_rrtmgp_cloud_optics_oracle.py tests/atmosphere/test_rrtmgp_overhead_layer.py tests/atmosphere/test_rrtmgp_ice_density_tie.py \
  tests/atmosphere/hydrostatic/unit/test_radiation.py tests/atmosphere/hydrostatic/unit/test_rrtmgp_stratosphere.py tests/atmosphere/hydrostatic/unit/test_rrtmgp_minor_loop.py \
  tests/unit/test_amip_production_rrtmgp_batch.py tests/unit/test_amip_production_rrtmgp_scan.py tests/unit/test_rrtmgp_use_scan_lanes.py tests/unit/test_rce_cloud_radiation.py tests/unit/test_orbital_integration_rrtmgp.py \
  tests/test_physics_contracts.py tests/test_no_inline_physics_coeffs.py tests/test_no_hardcoded_constants.py tests/test_no_private_cross_imports.py 2>&1 | tail -30
