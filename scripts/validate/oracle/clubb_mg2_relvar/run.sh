#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
# Regenerate tests/atmosphere/oracle_fixtures/clubb_mg2_relvar.npz.
#   sbatch --export=ALL,WT=<legoESM checkout>,CLUBB_ROOT=<build_oracle.sh build>,CAM=<CAM cam_cesm2_1_rel_60> run.sh
set -euo pipefail
: "${WT:?}" "${CLUBB_ROOT:?}" "${CAM:?}"
LEGO_PY=${LEGO_PY:-/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python}
TMP=$(mktemp -d)
SET=$TMP/settings
PYTHONPATH=$(ls -d $WT/packages/* | tr '\n' ':')$WT JAX_PLATFORMS=cpu \
  $LEGO_PY $WT/scripts/validate/oracle/clubb_core/dump_cam_settings.py $SET
(cd $CLUBB_ROOT/run_scripts && CLUBB_FIXTURE_PDF=1 CLUBB_ROOT=$CLUBB_ROOT \
  $CLUBB_ROOT/.venv-python/bin/python $WT/scripts/validate/oracle/clubb_core/gen_clubb_core_fixture.py \
  $SET $TMP/clubb_core_pdf.npz)
PYTHONPATH=$(ls -d $WT/packages/* | tr '\n' ':')$WT JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  $LEGO_PY $WT/scripts/validate/oracle/clubb_mg2_relvar/make_fixture.py \
  --pdf $TMP/clubb_core_pdf.npz --cam $CAM --build $TMP/build \
  --out $WT/tests/atmosphere/oracle_fixtures/clubb_mg2_relvar.npz
