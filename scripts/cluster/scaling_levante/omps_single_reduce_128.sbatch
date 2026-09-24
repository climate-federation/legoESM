#!/bin/bash -l
#SBATCH --job-name=omps_sr128
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=32
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=omps_sr128_%j.log
# ===========================================================================
# OCEAN MPAS barotropic PCG: one batched global sum per iteration
# (single_reduce, Chronopoulos-Gear) vs two (standard), at 64 and 128 GPUs.
#
# At 4 GPUs single_reduce measured 7.94/7.95 vs 8.18/8.20 ms (-3%). The step
# at 4 GPUs is ~58% collectives, 94 all-reduces per step, and a reduction's
# latency grows with device count, so the lever's size is a 128-GPU question.
# User decision 2026-09-24: measure at 128 before any default moves.
#
# CONFIG = the nature ladder's ocean-MPAS weak row (20480 cells/device, 40
# levels, poly-4 preconditioner, 20 fixed iterations = config defaults,
# 8 NCCL channels, --cpu-bind cores), float32 and float64; ONLY
# --pcg-variant differs. Arms alternate S,R,S,R so drift is visible.
# Tree: the edge-order branch, where the ocean lane keeps the owner edge
# order (production layout).
#
# PRE-REGISTERED: ADOPT-candidate if single_reduce <= 0.95x standard at 128
# in both replicates (both precisions reported). NULL if within the S/S
# replicate spread. Offline cannot answer: it is collective latency at 128.
# ===========================================================================
set -uo pipefail
export LEGOESM_REPO=/work/bd1083/b309178/diffESM/legoesm_pr_edgeorder
export JAX_PLATFORMS=cuda,cpu
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
source "$LEGOESM_REPO/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || exit 2
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_shard_autotuning=false"
export NCCL_MIN_NCHANNELS=8 NCCL_MAX_NCHANNELS=8 NCCL_P2P_NET_CHUNKSIZE=131072
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
OUTDIR="$SCRATCH/legoesm_scaling/omps_sr128_j${SLURM_JOB_ID}"
[ -e "$OUTDIR" ] && { echo "REFUSING: $OUTDIR exists" >&2; exit 1; }
mkdir -p "$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$OUTDIR/jitcache"; mkdir -p "$LEGOESM_JIT_CACHE_DIR"
echo "repo $REPO sha $(git rev-parse --short HEAD) outdir $OUTDIR"

arm() {  # arm <variant> <nd> <x64> <rep>
  local v="$1" nd="$2" x64="$3" rep="$4" nn=$(( ($2 + 3) / 4 ))
  local tag="omps_${v}_d${nd}_x${x64}_r${rep}"
  echo "=== $tag $(date +%T)"
  JAX_ENABLE_X64="$x64" timeout --kill-after=30 1500 srun --export=ALL --nodes="$nn" \
      --ntasks="$nd" --ntasks-per-node=4 --gpus-per-node=4 --cpu-bind=cores \
      --cpus-per-task=64 --kill-on-bad-exit=1 \
      "$PY" scripts/bench/bench_ocean_mpas_spmd_scaling.py --multicontroller \
      --n-devices "$nd" --mode weak --cells-per-device 20480 --nlev 40 --lloyd 0 \
      --block-steps 20 --blocks 3 --pcg-variant "$v" --out "$OUTDIR/$tag.jsonl" \
      < /dev/null > "$OUTDIR/$tag.log" 2>&1 || echo "ARM FAILED $tag rc=$?"
  tail -n 2 "$OUTDIR/$tag.log"
}
# 10 arms (~8-12 min each at 128 incl. compile): f32 at 128 twice, f64 at 128
# once, f32 at 64 once.
for rep in 1 2; do arm standard 128 0 "$rep"; arm single_reduce 128 0 "$rep"; done
arm standard 128 1 1; arm single_reduce 128 1 1
arm standard 64 0 1; arm single_reduce 64 0 1
arm standard 128 0 3
"$PY" - "$OUTDIR" <<'PYEOF'
import glob, json, os, sys
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.jsonl"))):
    for l in open(f):
        d = json.loads(l)
        x = d.get("metadata", {}).get("extra", {})
        print(f"{os.path.basename(f)[:-6]:36s} {d.get('steady_median_ms')} ms  "
              f"pcg={x.get('pcg_variant')} precond={x.get('pcg_precond')}@{x.get('pcg_fixed_iters')} "
              f"allreduces/step={x.get('barotropic_allreduces_per_step')} cells/dev={x.get('cells_per_device')}")
print("OMPS_SR128_DONE")
PYEOF
