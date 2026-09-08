#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=fv3k2probe
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/k2probe_%j.log
set -uo pipefail
REPO=${REPO:-/burg-archive/glab/users/pg2328/legoESM_fv3jax}
echo "host: $(hostname)"; echo "PINNED sha: $(git -C "$REPO" rev-parse HEAD)"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH=$REPO/packages/core:$REPO/packages/atmosphere:$REPO/src
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu
cd "$REPO"
for ARGS in "--n 4" "--n 4 --clear" "--n 4 --k-split 1"; do
  echo "########## $ARGS"
  timeout 3600 $PY scripts/validate/fv3_native/nh_ksplit2_compile_probe.py $ARGS
  echo "  rc=$?"
done
echo K2PROBE_DONE
