#!/bin/bash -l
#SBATCH --job-name=cube_nsys
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:45:00
#SBATCH --exclude=l50100
#SBATCH --output=cube_nsys.%j.log
# WHERE do the closed-loop C768@24's 19.0 ms/step go? The fused-halo A/B
# (job 26828313) refuted collective COUNT as the binding term (XLA's
# combiner already fuses: 101 CPs on GPU; source fusion saved 5 CPs,
# ratio 1.066). The eff-0.553 deficit therefore needs attribution among:
#   (a) NCCL SendRecv time (payload or wait) — comm-bound after all,
#       via per-CP duration, not count;
#   (b) microkernel launch storm (the MPAS signature: ~us kernels,
#       thousands/step);
#   (c) large compute kernels — the deficit is algorithmic.
# (a) NUMBERS this produces: per-step NCCL kernel total + duration
#     median on ranks 0/1 (co-located under 2 rpn — same timebase), and
#     the kernel-duration histogram.
# (b) CONFIRM/REFUTE: NCCL share >=40% of step => comm-bound (then the
#     0/1 pair skew separates wait from payload); median kernel ~2-3 us
#     with thousands/step => launch-bound (upstream XLA class, like
#     MPAS); neither => compute-bound, deficit is algorithmic.
# (c) No offline proxy: needs the real GPU timeline; CPU censuses just
#     proved misleading on this lane.
# Wrapper handles the QUADD_INJECTION_PROXY jax-deadlock and the nsys
# 2023.2.3 flag traps (see _nsys_rank_wrapper.sh).
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_XLA_OVERLAP=0
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/cube_nsys_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$OUTDIR/jitcache"
export NSYS_BIN="$(command -v nsys || echo /usr/local/bin/nsys)"
export NSYS_OUTDIR="$OUTDIR"
export PROFILE_RANKS="0,1"
export PY_BIN="$PY"
rc=0
JAX_ENABLE_X64=0 srun --nodes=12 --ntasks=24 --ntasks-per-node=2 \
    --distribution=block:block --cpu-bind=cores --kill-on-bad-exit=1 \
  bash scripts/cluster/scaling_levante/_nsys_rank_wrapper.sh \
    scripts/bench/bench_cube_tiled_step_scaling.py \
    --kt 2 --resolution 768 --nlev 60 --dt 30 --steps 12 --warmup 3 \
    --closed-loop --multicontroller \
    --out "$OUTDIR/c768_kt2_np24.jsonl" || { echo "capture FAILED"; rc=1; }
echo "=== files ==="; ls -la "$OUTDIR" | head
"$PY" -c "
import json, math
r = json.loads(open('$OUTDIR/c768_kt2_np24.jsonl').readline())
ms = r['steady_median_ms']
assert math.isfinite(ms) and ms > 0
print(f'steady {ms:.2f} ms (unprofiled arms ran at 19.0; nsys overhead visible here)')" || rc=1
echo "DONE rc=$rc"; exit $rc
