#!/bin/bash -l
#SBATCH --job-name=wh_rank0
#SBATCH --account=bd1083_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=00:50:00
#SBATCH --exclude=l50100
#SBATCH --output=wh_rank0.%j.log
# ===========================================================================
# WHERE DOES THE 2.4 ms GO?  Rank-0 timeline of the CURRENT best MPAS step.
#
# Measured budget of the 6.82 ms subdiv-9@64 wide-halo step:
#   local compute      2.00 ms   MEASURED  (same 40 962 cells x 26 levels on
#                                ONE GPU, no collectives, job of 2026-08-13)
#   halo payload       1.85 ms   MEASURED  (wire-payload ballast A/B, job
#                                26919124: +1.845 ms at 2x bytes, linearity
#                                3.36 against a predicted 3.00)
#   halo latency       0.57 ms   fitted    (11 rounds x 52 us, from the
#                                33 -> 11 round A/B)
#   UNACCOUNTED        2.40 ms   35 % of the step, never instrumented
#
# The unaccounted term is bigger than every remaining comm lever put
# together, so it decides what to work on next. Candidates, and what the
# rank-0 timeline says about each:
#   ghost redundancy  -- the depth-9 local region is ~17 % larger than the
#                        owned block, so ~0.34 ms of the 2.00 ms compute is
#                        work on cells that are never kept. Shows up as
#                        kernel time, not as gaps.
#   pack / scatter    -- gathering send rows and scattering received rows is
#                        on-device traffic, not wire. Kernel time, in
#                        kernels adjacent to the NCCL ones.
#   rank skew         -- waiting for the slowest peer. Shows up INSIDE the
#                        NCCL kernel duration, and as gaps before it.
#   launch overhead   -- shows up as gap time with no kernel running.
#
# WHAT TO EXTRACT from the trace, per timed step:
#   N_kernels, sum of kernel durations, sum of NCCL kernel durations,
#   wall, and gap = wall - (sum of all kernel durations).
#
# READING, decided before the run:
#   gap >= 1.2 ms                  -> LAUNCH-BOUND. Fuse the step; comm work
#                                     stays parked.
#   NCCL kernels >= 3.0 ms         -> the wire term is bigger than the
#                                     ballast A/B says, i.e. skew is being
#                                     charged inside the collective; the
#                                     lever is load balance, not bytes.
#   non-NCCL kernels >= 4.0 ms     -> the multi-device compute is 2x the
#                                     single-GPU control at the same load;
#                                     the extra is pack/scatter + ghost, and
#                                     shrinking the halo region is the lever.
#   none of the above              -> the budget does not close; re-derive
#                                     before proposing anything.
#
# Only rank 0 is profiled (~400 MB/rank otherwise). Rank 0 is NOT the
# critical rank -- it sits in fewer colour classes than max_degree -- so its
# NCCL total is a LOWER bound on the slowest rank's. Stated here so the
# number is not read as a global.
#
# SUBMIT: sbatch scripts/cluster/scaling_levante/mpas_wide_rank0_trace.sbatch
# ===========================================================================
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_XLA_OVERLAP=0
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }

export NSYS_OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/wh_rank0_j${SLURM_JOB_ID}}"
mkdir -p "$NSYS_OUTDIR"; echo "outdir=$NSYS_OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$NSYS_OUTDIR/jitcache"
export NSYS_BIN="${NSYS_BIN:-/sw/spack-levante/cuda-12.2.0-2ttufp/nsight-systems-2023.2.3/target-linux-x64/nsys}"
export PROFILE_RANKS="${PROFILE_RANKS:-0}"
export PY_BIN="$PY"

NODES=16; GPN=4; NDEV=$((NODES * GPN))

LEGOESM_MPAS_WIDE_HALO=1 LEGOESM_MPAS_RAGGED_HALO=0 JAX_ENABLE_X64=0 \
  srun --nodes="$NODES" --ntasks="$NDEV" --ntasks-per-node="$GPN" \
    --gpus-per-node="$GPN" --gpu-bind=none --kill-on-bad-exit=1 \
  "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_nsys_rank_wrapper.sh" \
    scripts/bench/bench_mpas_spmd_scaling.py \
    --multicontroller --n-devices "$NDEV" --subdivision 9 --nlev 26 \
    --steps 20 --warmup 5 --lloyd 0 --partition-method sfc \
    --reorder-for "$NDEV" --out "$NSYS_OUTDIR/step.jsonl"
rc=$?

echo "=== RANK-0 TIMELINE ==="
# Reuse the existing kernel-mix analyzer (NCCL vs compute totals, medians
# and the top kernels); the timed loop is 20 steps at ~6.8 ms, so window
# the tail at 150 ms to exclude compile and the setup all-gathers.
"$PY" "${SUBMIT_DIR}/scripts/bench/analyze_nsys_kernel_mix.py" \
  "$NSYS_OUTDIR/rank_0.sqlite" --tail-ms 150 || rc=$?
echo "DONE rc=$rc"
exit $rc
