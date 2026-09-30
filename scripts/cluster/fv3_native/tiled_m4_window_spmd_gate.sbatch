#!/bin/bash
#SBATCH --account=glab
#SBATCH --job-name=m4b_sgate
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=160G
#SBATCH --exclusive
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/m4b_spmd_gate_%j.log
# M4b SPMD window gate on 6*kt*kt virtual CPU devices in one process.
set -u
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3jax
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
cd "$REPO" || exit 1
export PYTHONPATH="$REPO/packages/core:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ocean:$REPO/packages/ml:$REPO/packages/tools:$REPO/src"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
N=${N:-48}; KM=${KM:-10}; KT=${KT:-2}; PAD=${PAD:-11}; NHFLAG=""; [ "${NH:-0}" = 1 ] && NHFLAG="--nh"
NDEV=$((6*KT*KT))
export XLA_FLAGS="--xla_force_host_platform_device_count=$NDEV --xla_cpu_max_isa=AVX"
echo "host: $(hostname)  sha: $(git rev-parse HEAD)  dirty: $(git status --porcelain | wc -l)  XLA_FLAGS=$XLA_FLAGS"
sha256sum packages/core/legoesm/grids/fv3_duo_window_spmd.py packages/core/legoesm/grids/fv3_duo_windows.py packages/core/legoesm/grids/fv3_duo_spmd.py scripts/validate/fv3_native/tiled_m4_window_spmd_gate.py
echo "===== SPMD window gate C$N km=$KM ${NHFLAG:-hydro} kt=$KT pad=$PAD n_split=${NSPLIT:-3} ($NDEV devices) ====="
"$PY" scripts/validate/fv3_native/tiled_m4_window_spmd_gate.py --n "$N" --km "$KM" $NHFLAG --kt "$KT" --pad "$PAD" --n-split "${NSPLIT:-3}" --flat-arm "${FLATARM:-batched}" ${GATE_EXTRA:-} 2>&1 \
  | grep -vE "cuda|jax_plugins|xla_bridge|versions_helpers|for d in range|_check_cuda|plugin_module|^\s+\^$|RuntimeError: jaxlib"
echo "ARM_RC=${PIPESTATUS[0]} (0 bitwise, 1 differs, 2 refused)"
echo DONE m4b_spmd_gate
