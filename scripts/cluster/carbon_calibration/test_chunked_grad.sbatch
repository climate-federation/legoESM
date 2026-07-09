#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=carbon_chunk_test
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=05:00:00
#SBATCH --output=results/carbon_calibration/chunk_test_%j.out
# =============================================================================
# Verify the chunked EXACT-gradient accumulation.  Two ISOLATED pytest processes
# so the retained coupled-model XLA/LLVM executables of one group do not stack
# with the next and exhaust compile memory:
#   (1) the cheap/pure tests + the quick single-batch slow/fast smoke paths
#   (2) the heavy chunked==single-batch EXACTNESS gate, ALONE (it compiles the
#       single-batch + chunked coupled reverse-mode over 4 archetypes).
# CPU, x64.  Never the login node.
# =============================================================================
set -uo pipefail

REPO=/burg-archive/glab/users/pg2328/lc_rebase
cd "$REPO" || { echo "cannot cd $REPO"; exit 2; }
mkdir -p results/carbon_calibration

PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH="$REPO:$(ls -d "$REPO"/packages/*/ | sed 's:/$::' | tr '\n' ':')${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
export PYTHONUNBUFFERED=1

echo "=== node $(hostname) | python $($PY --version 2>&1) ==="

echo "=== [1/2] cheap + quick-path trainer tests ==="
$PY -m pytest tests/land/unit/test_train_carbon_params.py -q -s \
    -k "chunk_bounds or weighted_sse or grad_chunk or cover_weighted or loss_registry or nan_target or overrides or som_only or slow_spinup or fast_analytic or quick_dry or resolve_om or input_mode"
R1=$?

echo "=== [2/2] chunked==single-batch EXACTNESS gate (isolated) ==="
$PY -m pytest tests/land/unit/test_train_carbon_params.py -q -s \
    -k "test_chunked_grad_equals_single_batch_slow"
R2=$?

echo "=== RESULTS: cheap_quick=$R1 exactness_gate=$R2 ==="
if [ "$R1" -eq 0 ] && [ "$R2" -eq 0 ]; then
    echo "CHUNK_TEST_OK: chunked gradient == single-batch (exact) + all trainer tests green."
else
    echo "CHUNK_TEST_FAIL: fix before the exact calibration."
fi
