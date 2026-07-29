#!/usr/bin/env bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=ut_pytest
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/scripts/cluster/unified_training/logs/ut_pytest_%j.out
# Unified-training unit tests on a compute node (CPU, x64) — main repo.
# Usage: sbatch --export=ALL,UT_PYTEST_TARGET="tests/unit/test_curriculum.py" run_pytest.sbatch
set -euo pipefail
source /burg-archive/glab/users/pg2328/legoESM/scripts/cluster/unified_training/_env.sh
cd "${UT_REPO}"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
echo "python: ${UT_PYTHON}"
echo "target: ${UT_PYTEST_TARGET:-tests/unit/test_curriculum.py}"
# unquoted so UT_PYTEST_TARGET may hold multiple space-separated paths
"${UT_PYTHON}" -m pytest ${UT_PYTEST_TARGET:-tests/unit/test_curriculum.py} -v -p no:cacheprovider
