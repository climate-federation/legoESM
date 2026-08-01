#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=les_bridge
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/les_bridge_%j.out
set +e
# Unit gate for the whole LES -> SCM turbulence-tuning stack.
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
fail=0

echo "=== les_record (numpy only, no conftest) ==="
$PY -m pytest tests/unit/test_les_record.py -q -p no:cacheprovider --noconftest
rc=$?; echo "les_record rc=$rc"; [ $rc -ne 0 ] && fail=1

echo "=== LES reference loader ==="
$PY -m pytest tests/unit/test_les_reference.py -q -p no:cacheprovider
rc=$?; echo "les_reference rc=$rc"; [ $rc -ne 0 ] && fail=1

echo "=== gSAM deck -> SCM bridge ==="
$PY -m pytest tests/atmosphere/hydrostatic/unit/test_sam_case_scm.py -q \
    -p no:cacheprovider
rc=$?; echo "bridge rc=$rc"; [ $rc -ne 0 ] && fail=1

echo "=== turbulence tuning driver ==="
$PY -m pytest tests/atmosphere/hydrostatic/test_run_scm_les_turbulence_tuning.py \
    -q -p no:cacheprovider
rc=$?; echo "driver rc=$rc"; [ $rc -ne 0 ] && fail=1

echo "=== existing SCM + forcing + metrics regression (must stay green) ==="
$PY -m pytest tests/atmosphere/hydrostatic/unit/test_scm.py \
             tests/atmosphere/hydrostatic/unit/test_scm_forcing.py \
             tests/unit/test_sam_case_forcing.py \
             tests/unit/test_scm_rce_metrics.py -q -p no:cacheprovider
rc=$?; echo "regression rc=$rc"; [ $rc -ne 0 ] && fail=1

echo "=============================================="
echo "GATE $([ $fail -eq 0 ] && echo PASSED || echo FAILED)"
echo "DONE $(date +%T)"
exit $fail
