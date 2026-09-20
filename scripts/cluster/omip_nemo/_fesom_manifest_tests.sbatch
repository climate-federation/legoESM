#!/bin/bash
#SBATCH --job-name=fesmanif
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/fesmanif_%j.out
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/_fluxacc_wt
cd $REPO || exit 1
echo "=== HEAD $(git -C $REPO rev-parse HEAD)"
echo "=== dirty files: $(git -C $REPO status --porcelain | wc -l)"
export PYTHONPATH="$REPO/src:$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
$PY -m pytest tests/ocean/unit/test_run_manifest_ocean.py \
              tests/ocean/unit/test_surface_flux_accumulator.py -q
rc=$?
echo "=== rc=$rc ==="
exit $rc
