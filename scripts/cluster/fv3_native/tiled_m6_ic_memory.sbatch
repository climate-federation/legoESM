#!/bin/bash
#SBATCH --account=glab --job-name=m6_icmem --partition=short --time=01:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=120G
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/m6_icmem_%j.log
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3jax
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
cd "$REPO" || exit 1
export PYTHONPATH="$REPO/packages/core:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ocean:$REPO/packages/ml:$REPO/packages/tools:$REPO/src"
N=${N:-96}; KT=${KT:-3}
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1 XLA_FLAGS="--xla_force_host_platform_device_count=$((6*KT*KT)) --xla_cpu_max_isa=AVX"
echo "sha: $(git rev-parse HEAD) dirty: $(git status --porcelain | wc -l) N=$N KT=$KT"
"$PY" scripts/validate/fv3_native/tiled_m6_ic_memory_probe.py $N $KT 2>&1 | grep -vE "cuda|jax_plugins|xla_bridge|versions_helpers|for d in range|_check_cuda|plugin_module|^\s+\^|^Traceback|RuntimeError: jaxlib"
echo DONE
