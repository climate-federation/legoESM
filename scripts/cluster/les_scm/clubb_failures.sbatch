#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=clubbfail
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/clubbfail_%j.out
set +e
: "${C_WT:?set C_WT}"
WT=$C_WT
cd $WT || exit 1
echo "PINNED sha: $(git rev-parse HEAD)"
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$PYTHONPATH
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
$PY -m pytest -q --no-header -rA --tb=long \
  tests/unit/test_clubb_scheme.py::test_init_clubb_moments_shapes_and_floors \
  tests/unit/test_clubb_scheme.py::test_clubb_prognostic_injected_flux_equals_kinematic_prescription \
  tests/unit/test_clubb_scheme.py::test_prognostic_clubb_convective_bl_physics \
  tests/unit/test_clubb_scheme.py::test_integrate_clubb_column_host_diffusion_controls_grid_noise \
  tests/unit/test_clubb_config.py::test_toc_line_numbers_accurate 2>&1 | tail -140
echo "rc=${PIPESTATUS[0]}"
echo "DONE $(date +%T)"
