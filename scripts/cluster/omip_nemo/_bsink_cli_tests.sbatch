#!/bin/bash
#SBATCH --job-name=bsink_t
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/bsink_tests_%j.out
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu
$PY -m pytest -q tests/unit/test_run_omip_core2_buoyancy_sink_cli.py tests/unit/test_run_omip_core2_surface_bc_level_cli.py tests/unit/test_run_omip_core2_kappa_convention_cli.py 2>&1 | tail -15
echo "=== card exit: $? ==="
