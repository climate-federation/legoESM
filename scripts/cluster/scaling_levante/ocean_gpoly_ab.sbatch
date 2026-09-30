#!/bin/bash -l
#SBATCH --job-name=ocean_gpoly_ab
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=ocean_gpoly_ab_%j.log
# ===========================================================================
# Global vs block-local polynomial preconditioner, ocean MPAS SPMD weak ladder
# config (20480 cells/GPU, 40 levels, f32, 8 NCCL channels), same halo depth 2.
#   A  poly  M=20   production baseline
#   B  gpoly M=20   price of the global preconditioner alone
#   C  gpoly M=15   the candidate (offline: M=15 global beats M=20 local in
#                   residual on the saved L9 systems, f32 and f64)
# Arms run forward then reverse (two repeats, interleaved).
#
# PRE-REGISTERED: C must cut the 128-GPU step by >= 5% vs A in BOTH repeats
# with finite state; otherwise the option stays opt-in and is not proposed as
# a default. B vs A prices the extra local sweeps on the halo.
#   sbatch -N 8  ... (d8, d32)     sbatch -N 32 ... (adds d128)
# Submit from the worktree:  sbatch --export=ALL,LEGOESM_REPO=$PWD -N <n> <this>
# ===========================================================================
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
export JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=0
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || exit 2
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_shard_autotuning=false"
export NCCL_MIN_NCHANNELS=8 NCCL_MAX_NCHANNELS=8 NCCL_P2P_NET_CHUNKSIZE=131072
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
"$PY" -c 'import legoesm.ocean.dynamics.barotropic_implicit_mpas as m; print("provenance", m.__file__)' || exit 2
OUTDIR="$SCRATCH/legoesm_scaling/ocean_gpoly_ab_j${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$OUTDIR/jitcache"; mkdir -p "$LEGOESM_JIT_CACHE_DIR"
NODES="${SLURM_JOB_NUM_NODES:-1}"; PPN=4
SRUN_DEV=(--gpus-per-node=4 --gpu-bind=none --cpu-bind=cores
          --cpus-per-task="$(( ${SLURM_CPUS_ON_NODE:?} / PPN ))")
rc=0

arm() {  # arm <tag> <nd> <rep> <extra bench flags...>
  local tag="$1" nd="$2" rep="$3"; shift 3
  local nn=$(( (nd + PPN - 1) / PPN ))
  (( nn > NODES )) && { echo "[skip] $tag d$nd needs $nn nodes"; return 0; }
  local out="$OUTDIR/${tag}_d${nd}_r${rep}"
  echo "=== $tag d$nd r$rep $(date '+%T')"
  if ! timeout --kill-after=30 1200 srun --export=ALL --nodes="$nn" --ntasks="$nd" \
        --ntasks-per-node="$PPN" "${SRUN_DEV[@]}" --kill-on-bad-exit=1 \
        "$PY" scripts/bench/bench_ocean_mpas_spmd_scaling.py \
        --multicontroller --n-devices "$nd" --mode weak --cells-per-device 20480 \
        --nlev 40 --lloyd 0 --block-steps 20 --blocks 3 "$@" \
        --out "$out.jsonl" < /dev/null > "$out.log" 2>&1; then
    echo "ARM FAILED $tag d$nd r$rep"; rc=1
  fi
  grep -oE "[0-9.]+ ms/step.*" "$out.log" | tail -1
}

ARMS=(
  "A_poly_M20|"
  "B_gpoly_M20|--pcg-precond gpoly --pcg-fixed-iters 20"
  "C_gpoly_M15|--pcg-precond gpoly --pcg-fixed-iters 15"
)
for nd in 8 32 128; do
  for rep in 1 2; do
    if (( rep == 1 )); then order=(0 1 2); else order=(2 1 0); fi
    for i in "${order[@]}"; do
      IFS='|' read -r tag flags <<< "${ARMS[$i]}"
      # shellcheck disable=SC2086
      arm "$tag" "$nd" "$rep" $flags
    done
  done
done
echo "=== DONE outdir=$OUTDIR rc=$rc"
exit "$rc"
