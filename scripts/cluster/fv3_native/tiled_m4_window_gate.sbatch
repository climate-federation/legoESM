#!/bin/bash
#SBATCH --account=glab
#SBATCH --job-name=m4_wgate
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=120G
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/m4_window_gate_%j.log
# M4 window gate: kernels verbatim on padded windows vs the flat step,
# bitwise.  ARMS="kt:pad kt:pad ..." (env), N, KM, NH (env, optional).
set -u
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3jax
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
cd "$REPO" || exit 1
export PYTHONPATH="$REPO/packages/core:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ocean:$REPO/packages/ml:$REPO/packages/tools:$REPO/src"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
# FMA OFF (M3 protocol): the batched and loop arms already differ at 1e-14
# under default lowering (FMA contraction, job 9631664); bitwise is only a
# meaningful target with the ISA pinned below FMA3.
[ "${FMA:-off}" = off ] && export XLA_FLAGS="--xla_cpu_max_isa=AVX"
echo "XLA_FLAGS=${XLA_FLAGS:-<default lowering>}"
echo "host: $(hostname)  sha: $(git rev-parse HEAD)  dirty: $(git status --porcelain | wc -l)"
sha256sum packages/core/legoesm/grids/fv3_duo_windows.py packages/core/legoesm/grids/fv3_duo_halos.py \
  packages/core/legoesm/core/fv3_acoustic_3d.py packages/core/legoesm/core/fv3_phase3d_common.py \
  scripts/validate/fv3_native/tiled_m4_window_gate.py
N=${N:-48}; KM=${KM:-10}; NHFLAG=""; [ "${NH:-0}" = 1 ] && NHFLAG="--nh"
for ARM in ${ARMS:-"1:3 2:8 2:12"}; do
  KT=${ARM%%:*}; PAD=${ARM##*:}
  echo "===== window gate C$N km=$KM ${NHFLAG:-hydro} kt=$KT pad=$PAD n_split=${NSPLIT:-3} ====="
  "$PY" scripts/validate/fv3_native/tiled_m4_window_gate.py --n "$N" --km "$KM" $NHFLAG \
      --kt "$KT" --pad "$PAD" --n-split "${NSPLIT:-3}" --flat-arm "${FLATARM:-both}" ${GATE_EXTRA:-} 2>&1 \
      | grep -vE "cuda|jax_plugins|xla_bridge|versions_helpers|for d in range|_check_cuda|plugin_module|^\s+\^$|RuntimeError: jaxlib"
  echo "ARM_RC=${PIPESTATUS[0]} (0 bitwise, 1 differs, 2 refused)"
done
echo DONE m4_window_gate
