#!/bin/bash
#SBATCH --job-name=wb_dp_test
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM_wbforecast/scripts/cluster/wb_forecast/logs/dp_test_%j.out
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:20:00
# Data-parallel tests on 2 CPU devices (--xla_force_host_platform_device_count=2)
# validate the cross-device gradient average == serial batch-mean BEFORE any GPU-hours.
set -euo pipefail
source /burg-archive/glab/users/pg2328/legoESM_wbforecast/scripts/cluster/wb_forecast/env.sh
cd "${WB_WORKTREE}"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
export XLA_FLAGS="--xla_force_host_platform_device_count=2"
"${WB_PYTHON}" -m pytest tests/ml/test_data_parallel.py -v -p no:cacheprovider
