#!/usr/bin/env bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=wb_pytest
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:20:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM_wbforecast/scripts/cluster/wb_forecast/logs/wb_pytest_%j.out
# WeatherBench Stage-0 unit tests on a compute node (CPU, x64).
# Usage: sbatch --export=ALL,WB_PYTEST_TARGET="tests/evaluations/..." run_pytest.sbatch
set -euo pipefail
source /burg-archive/glab/users/pg2328/legoESM_wbforecast/scripts/cluster/wb_forecast/env.sh
cd "${WB_WORKTREE}"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
echo "python: ${WB_PYTHON}"
echo "target: ${WB_PYTEST_TARGET:-tests/evaluations/}"
# unquoted so WB_PYTEST_TARGET may hold multiple space-separated paths
"${WB_PYTHON}" -m pytest ${WB_PYTEST_TARGET:-tests/evaluations/} -v -p no:cacheprovider
