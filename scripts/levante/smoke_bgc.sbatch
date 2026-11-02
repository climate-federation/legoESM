#!/bin/bash
#SBATCH --job-name=bgc_smoke
#SBATCH --account=bb1596
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/work/bb1596/b384036/diffESM/legoesm_ggn_newcopy/logs/bgc_smoke_%j.log
# Environment + baseline check before the MARBL port: which python, JAX x64, current BGC unit tests.
# Submit from anywhere:  sbatch /work/bb1596/b384036/diffESM/legoesm_ggn_newcopy/scripts/levante/smoke_bgc.sbatch
set -uo pipefail
REPO=/work/bb1596/b384036/diffESM/legoesm_ggn_newcopy
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"   # legoesm from source, no install needed

PY_VENV="$REPO/.venv/bin/python"
PY_GPU="${PY_GPU:-$HOME/micromamba/envs/climateval_gpu/bin/python}"   # override: sbatch --export=ALL,PY_GPU=/path/to/python
PY_MAMBA="$HOME/micromamba/envs/climateval/bin/python"
for PY in "$PY_GPU" "$PY_VENV" "$PY_MAMBA"; do
    echo "=== $PY ==="
    if [[ ! -x "$PY" ]]; then echo "missing"; continue; fi
    JAX_ENABLE_X64=1 "$PY" -c "import jax, jax.numpy as jnp, legoesm; print('jax', jax.__version__, jax.devices(), jnp.ones(1).dtype); print('legoesm', legoesm.__file__)" \
        || echo "import failed"
done

PY=""
for cand in "$PY_GPU" "$PY_VENV" "$PY_MAMBA"; do
    if [[ -x "$cand" ]] && "$cand" -c "import legoesm" 2>/dev/null; then PY="$cand"; break; fi
done
if [[ -z "$PY" ]]; then echo "No python with legoesm importable"; exit 1; fi
echo "=== using $PY ==="

JAX_ENABLE_X64=1 "$PY" -m pytest -q tests/ocean/unit/test_ocean_biogeochemistry.py 2>&1 | tail -30

echo "=== git state of BGC files ==="
git status --short -- src/legoesm/ocean/biogeochemistry scripts/run_omip.py scripts/run_bgc_column.py CLAUDE.md
git diff --stat -- src/legoesm/ocean/biogeochemistry scripts/run_omip.py
