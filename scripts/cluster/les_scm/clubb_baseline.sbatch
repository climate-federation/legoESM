#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=clubbbase
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/clubbbase_%j.out
set +e
# BASELINE: the same five tests at the commit BEFORE the CLUBB variance-floor
# fix. A failure is not attributable to that fix until it is shown NOT to fail
# here. test_clubb_variance_floor.py does not exist at this pin and is excluded.
: "${B_WT:?set B_WT}"
WT=$B_WT
cd $WT || exit 1
echo "PINNED sha: $(git rev-parse HEAD)  (pre-fix baseline)"
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$PYTHONPATH
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
$PY -m pytest -q --no-header -rA --tb=line \
  tests/unit/test_clubb_scheme.py::test_init_clubb_moments_shapes_and_floors \
  tests/unit/test_clubb_scheme.py::test_clubb_prognostic_injected_flux_equals_kinematic_prescription \
  tests/unit/test_clubb_scheme.py::test_prognostic_clubb_convective_bl_physics \
  tests/unit/test_clubb_scheme.py::test_integrate_clubb_column_host_diffusion_controls_grid_noise \
  tests/unit/test_clubb_config.py::test_toc_line_numbers_accurate 2>&1 | tail -40
echo "baseline rc=${PIPESTATUS[0]}"
echo "DONE $(date +%T)"
