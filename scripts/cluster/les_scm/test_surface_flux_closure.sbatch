#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=sfcflux
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/sfcflux_%j.out
set +e
: "${SF_WT:?set SF_WT to the pinned worktree}"
WT=$SF_WT
cd $WT || exit 1
echo "PINNED sha: $(git rev-parse HEAD)"; echo "host: $(hostname)"
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$PYTHONPATH
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

echo "=============== prescribed surface flux -> closure ==============="
$PY -m pytest -q tests/atmosphere/test_prescribed_surface_flux_to_closure.py 2>&1 | tail -30
echo "sfcflux rc=${PIPESTATUS[0]}"

echo "=============== regression: turbulence + surface layer + SCM ==============="
# SurfaceLayerConfig gained two fields; every consumer of it is in the blast
# radius, not just the test that was written for the change.
$PY -m pytest -q tests/atmosphere -k "turbulence or surface or scm or ysu or mynn" 2>&1 | tail -20
echo "turb rc=${PIPESTATUS[0]}"

echo "=============== driver CLI + aggregation ==============="
$PY -m pytest -q tests/unit/test_scm_les_joint_aggregation.py 2>&1 | tail -12
echo "cli rc=${PIPESTATUS[0]}"
echo "DONE $(date +%T)"
