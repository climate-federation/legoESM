#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
# Regenerate tests/atmosphere/oracle_fixtures/clubb_glue_cam.npz.
#   sbatch --export=ALL,WT=<legoESM checkout>,ORACLES=<oracles root> run.sh
# ORACLES holds CAM (cam_cesm2_1_rel_60) and rrtmgp-data (v1.8.2).
set -euo pipefail
: "${WT:?}" "${ORACLES:?}"
LEGO_PY=${LEGO_PY:-/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python}
HERE=$WT/scripts/validate/oracle/clubb_glue
WORK=$(mktemp -d)
export PYTHONPATH=$(ls -d $WT/packages/* | tr '\n' ':')$WT JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
RFMIP=$ORACLES/rrtmgp-data/examples/rfmip-clear-sky/inputs/multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc
$LEGO_PY $HERE/build_inputs.py $WORK $RFMIP
bash $HERE/build.sh $ORACLES/CAM $WORK
$LEGO_PY $HERE/gen_clubb_glue_fixture.py $WORK $ORACLES/CAM \
  $WT/tests/atmosphere/oracle_fixtures/clubb_glue_cam.npz
