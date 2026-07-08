#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=carbon_fast_validate
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=results/carbon_calibration/validate_fast_%j.out
# =============================================================================
# Stage-B (B2-fast) CHEAP CPU validation of the fast closed-form SOM equilibrium
# BEFORE the real GPU calibration:
#   1. fast_analytic unit tests -- pure cascade + grad + monotonicity, AND the
#      CRUX match (analytic == equilibrate_archetypes at defaults, tiny world)
#   2. B2 trainer unit tests -- fast --quick pipeline (default), slow-path smoke,
#      CLI flag round-trip
# Run: mkdir -p results/carbon_calibration &&
#      sbatch scripts/cluster/carbon_calibration/validate_fast_analytic.sbatch
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
echo "=== [1/2] fast_analytic unit tests (pure closed form + spin-up match) ==="
$PY -m pytest tests/land/unit/test_fast_analytic.py -q -s; R1=$?

echo "=== [2/2] B2 trainer unit tests (fast --quick + slow smoke + CLI) ==="
$PY -m pytest tests/land/unit/test_train_carbon_params.py -q -s; R2=$?

echo "=== RESULTS: fast_analytic=$R1 trainer=$R2 ==="
if [ "$R1" -eq 0 ] && [ "$R2" -eq 0 ]; then
    echo "VALIDATE_FAST_OK: all green -- safe to launch the fast GPU calibration."
else
    echo "VALIDATE_FAST_FAIL: fix before the GPU run."
fi
