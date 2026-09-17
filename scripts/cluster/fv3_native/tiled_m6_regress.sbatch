#!/bin/bash
#SBATCH --account=glab
#SBATCH --job-name=m4_regress
#SBATCH --partition=short
#SBATCH --time=03:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=160G
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/m4_regress_%j.log
set -u
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3jax
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
cd "$REPO" || exit 1
export PYTHONPATH="$REPO/packages/core:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ocean:$REPO/packages/ml:$REPO/packages/tools:$REPO/src"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1
echo "sha: $(git rev-parse HEAD)"; sha256sum packages/core/legoesm/grids/fv3_duo_windows.py tests/grids/test_fv3_duo_windows.py
"$PY" -m pytest -q -p no:cacheprovider ${TESTS:-tests/grids/test_fv3_dynamics.py} -q 2>&1 | grep -vE "cuda|jax_plugins|xla_bridge|versions_helpers|for d in range|_check_cuda|plugin_module|^\s+\^|^Traceback|RuntimeError: jaxlib" | tail -80
echo "PYTEST_RC=${PIPESTATUS[0]}"
echo DONE m4_regress
