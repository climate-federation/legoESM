#!/bin/bash
#SBATCH --account=glab
#SBATCH --job-name=m6_hlo
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=64G
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/m6_hlo_%j.log
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3jax
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
cd "$REPO" || exit 1
export PYTHONPATH="$REPO/packages/core:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ocean:$REPO/packages/ml:$REPO/packages/tools:$REPO/src"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1 XLA_FLAGS="--xla_force_host_platform_device_count=$((6*${KT:-2}*${KT:-2})) --xla_cpu_max_isa=AVX"
"$PY" scripts/validate/fv3_native/tiled_m6_hlo_census.py ${N:-48} ${KT:-2} ${PACK:-} 2>&1 | grep -vE "cuda|jax_plugins|xla_bridge|versions_helpers|for d in range|_check_cuda|plugin_module|^\s+\^|^Traceback|RuntimeError: jaxlib"
echo DONE
