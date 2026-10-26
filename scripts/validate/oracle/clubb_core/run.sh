#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
# Regenerate tests/atmosphere/oracle_fixtures/clubb_core_cam.npz.
#   sbatch --export=ALL,WT=<legoESM checkout>,CLUBB_ROOT=<built clubb_release> run.sh
# CLUBB_ROOT must hold a build made by build.sh (same SHA + patch).
set -euo pipefail
: "${WT:?}" "${CLUBB_ROOT:?}"
LEGO_PY=${LEGO_PY:-/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python}
HERE=$WT/scripts/validate/oracle/clubb_core
SET=$(mktemp -d)
PYTHONPATH=$(ls -d $WT/packages/* | tr '\n' ':')$WT JAX_PLATFORMS=cpu \
  $LEGO_PY $HERE/dump_cam_settings.py $SET
cd $CLUBB_ROOT/run_scripts
CLUBB_ROOT=$CLUBB_ROOT $CLUBB_ROOT/.venv-python/bin/python $HERE/gen_clubb_core_fixture.py \
  $SET $WT/tests/atmosphere/oracle_fixtures/clubb_core_cam.npz
