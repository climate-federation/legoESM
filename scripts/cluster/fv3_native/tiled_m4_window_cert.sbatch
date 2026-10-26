#!/bin/bash
#SBATCH --account=glab
#SBATCH --job-name=m4_wcert
#SBATCH --partition=short
#SBATCH --time=03:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=120G
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/m4_window_cert_%j.log
# Static pad certification on the WINDOW program (tiled_m4_static_reach.py
# --window KT:PAD): garbage in one interior window's west pad, taint through
# one substep with the real firings, overshoot into owned cells must be 0.
set -u
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3jax
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
cd "$REPO" || exit 1
export PYTHONPATH="$REPO/packages/core:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ocean:$REPO/packages/ml:$REPO/packages/tools:$REPO/src"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
echo "sha: $(git rev-parse HEAD)"; sha256sum scripts/validate/fv3_native/tiled_m4_static_reach.py packages/core/legoesm/grids/fv3_duo_windows.py
ARMS=${ARMS:-"48:hydro:3:8 48:hydro:3:9 48:nh:3:8 48:nh:3:9"}
for ARM in $ARMS; do
  IFS=: read -r N MODE KT PAD <<< "$ARM"; NHFLAG=""; [ "$MODE" = nh ] && NHFLAG="--nh"
  echo "===== window cert C$N km=${KM:-10} $MODE kt=$KT pad=$PAD ====="
  "$PY" scripts/validate/fv3_native/tiled_m4_static_reach.py --n "$N" --km "${KM:-10}" $NHFLAG --window "$KT:$PAD" 2>&1 \
    | grep -vE "cuda|jax_plugins|xla_bridge|versions_helpers|for d in range|_check_cuda|plugin_module|^\s+\^$|RuntimeError: jaxlib|^Traceback"
  echo "ARM_RC=${PIPESTATUS[0]}"
done
echo DONE m4_window_cert
