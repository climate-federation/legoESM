#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=les_bridge
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/les_bridge_%j.out
set +e
# Unit gate for the gSAM-deck -> SCM bridge and the LES frame recorder.
# CPU-only and x64: these are numerics/convention tests, so per CLAUDE.md they
# run in float64 (an fp32 mismatch here would not be a real failure).
WT=/burg/glab/users/pg2328/les_scm_bridge_wt
cd $WT || exit 1
echo "PINNED sha: $(git rev-parse HEAD)"
echo "host: $(hostname)"
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$PYTHONPATH
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

echo "=== les_record (numpy only, no conftest) ==="
$PY -m pytest tests/unit/test_les_record.py -q -p no:cacheprovider --noconftest
echo "les_record rc=$?"

echo "=== sam_case_scm bridge ==="
$PY -m pytest tests/atmosphere/hydrostatic/unit/test_sam_case_scm.py -q \
    -p no:cacheprovider
echo "bridge rc=$?"

echo "=== existing SCM + forcing regression (must stay green) ==="
$PY -m pytest tests/atmosphere/hydrostatic/unit/test_scm.py \
             tests/atmosphere/hydrostatic/unit/test_scm_forcing.py \
             tests/unit/test_sam_case_forcing.py -q -p no:cacheprovider
echo "regression rc=$?"
echo "DONE $(date +%T)"
