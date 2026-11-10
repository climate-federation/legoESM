#!/bin/bash
#SBATCH --job-name=trcregion
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:20:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/trcregion_%j.out
#
# Region box + heat depth bins on the tracer-content probe.
# Runs against the EDITING checkout (_omip_merge), not the run tree.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/_omip_merge
cd $REPO || exit 1
echo "=== HEAD $(git -C $REPO rev-parse HEAD)"
export PYTHONPATH="$REPO/src:$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
$PY -m pytest tests/validate/test_global_tracer_content.py -q
rc=$?
echo "=== rc=$rc ==="
exit $rc
