#!/bin/bash
#SBATCH --account=glab
#SBATCH --job-name=m4_win_unit
#SBATCH --partition=short
#SBATCH --time=00:20:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/m4_windows_unit_%j.log
set -u
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3jax
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
cd "$REPO" || exit 1
export PYTHONPATH="$REPO/packages/core:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ocean:$REPO/packages/ml:$REPO/packages/tools:$REPO/src"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1
echo "sha: $(git rev-parse HEAD)"; sha256sum packages/core/legoesm/grids/fv3_duo_windows.py tests/grids/test_fv3_duo_windows.py
"$PY" -m pytest -q -p no:cacheprovider tests/grids/test_fv3_duo_windows.py ${EXTRA_TESTS:-} 2>&1 | grep -vE "cuda|jax_plugins|xla_bridge|versions_helpers|for d in range|_check_cuda|plugin_module|^\s+\^|^Traceback|RuntimeError: jaxlib" | tail -40
echo "PYTEST_RC=${PIPESTATUS[0]}"
echo DONE m4_windows_unit
