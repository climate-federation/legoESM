#!/bin/bash
#SBATCH --job-name=clitest
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/clitest_%j.out
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_arm_literal; cd $ROOT || exit 1
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu
$PY -m pytest -q tests/unit/test_run_omip_core2_step_evaluation_cli.py tests/unit/test_run_omip_core2_buoyancy_sink_cli.py 2>&1 | tail -6
echo "=== card exit: $? ==="
