#!/bin/bash -l
#SBATCH --job-name=mpas_skew
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=mpas_nsys_skew.%j.log
# ===========================================================================
# MULTI-RANK capture: is the MPAS halo collective PAYLOAD or WAIT?
#
# THE QUESTION. s9/np64 spends 5.94 of 9.72 ms per step inside
# ncclDevKernel_SendRecv, median 216 us against a ~30 us wire latency.  The
# single-rank capture (job 26680051) CANNOT decide why: it showed the
# durations are reproducible per schedule slot (most positions CV <= 1.5 %),
# which refutes RANDOM jitter but not a CONSISTENTLY late peer.  Wait is an
# inter-rank quantity; one rank's timeline cannot see it.
#
# THE DISCRIMINATOR.  An NCCL SendRecv kernel starts when ITS rank arrives and
# ends when the transfer completes.  So for ONE collective seen from BOTH
# partners:
#   * both durations ~equal  -> the time is transfer. PAYLOAD-bound. Cutting
#     round count then saves only ~30 us x rounds removed; the bytes still
#     move.  That is the LOW end of a round-depth partitioner's value.
#   * one long, one short    -> the long one arrived early and idled. WAIT-
#     bound, and the fix is imbalance, not round count.
# The gap between those two readings is the entire uncertainty in whether the
# 7-14 day partitioner is worth building.
#
# WHY THESE RANKS.  Timestamps from different nsys processes share a timebase
# only on ONE node (CLOCK_MONOTONIC is per-machine and undisciplined across
# nodes; drift there is tens of ms against a 216 us signal).  So the profiled
# ranks must be partners AND co-located.  Both hold here, and that is
# measured, not assumed: scoring the traced configuration (s9, np64, sfc,
# reorder_target=128 -- see below) gives rank 0 eight rounds with partners
# 1, 2, 12, 13, 49, 50, 61, 63.  Ranks 1 and 2 are on rank 0's node under
# block:block, and rounds 0 and 1 (partners 1 and 2) carry the LARGEST
# payloads of the eight (4809/12217 and 4626/11865 cells/edges), so they are
# also the most informative.  Rank 3 is captured as a same-node NON-partner
# reference: it shares the node's PCIe/NVLink and CPU but exchanges nothing
# with rank 0, which separates "this node is slow" from "this pair is skewed".
#
# WHY reorder-for 128.  The traced run's own launcher was a throwaway and is
# gone, so this was recovered by measurement: with reorder_target=64 the
# schedule gives rank 0 seven rounds, and no multiple of 7 divides the 288
# collectives in that trace; with 128 it gives eight, and 8 x 3 fills = 24
# divides 288 exactly (12 steps).  Matching it is what makes the new capture
# comparable to the old one.
#
# NO --delay/--duration.  The run is ~7.3 s of compile plus 12 steps of
# ~10 ms, so any delay long enough to skip the compile also skips the whole
# steady state.  Capture everything; the analysis drops warmup cycles.
#
# COST: 16 nodes x 1 h wall cap, but the run itself is seconds -- the job is
# dominated by launch and by writing four ~400 MB profiles.
#
# SUBMIT (from the repo root):
#   sbatch scripts/cluster/scaling_levante/mpas_nsys_skew_capture.sbatch
# ===========================================================================
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }

# Absolute path, VERIFIED present (2023.2.3). No module load needed, and
# pinning the version keeps the trace schema stable for the analysis.
NSYS_BIN="${NSYS_BIN:-/sw/spack-levante/cuda-12.2.0-2ttufp/nsight-systems-2023.2.3/target-linux-x64/nsys}"
[ -x "$NSYS_BIN" ] || { echo "nsys not executable: $NSYS_BIN"; exit 2; }

# rank 0 + partners 1,2 + same-node non-partner reference 3.
PROFILE_RANKS="${PROFILE_RANKS:-0,1,2,3}"
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/mpas_skew_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"
echo "outdir=$OUTDIR  nsys=$NSYS_BIN  ranks=$PROFILE_RANKS"
"$NSYS_BIN" --version

WRAPPER="${SUBMIT_DIR}/scripts/cluster/scaling_levante/_nsys_rank_wrapper.sh"
[ -x "$WRAPPER" ] || { echo "wrapper missing/not executable: $WRAPPER"; exit 2; }

export NSYS_BIN NSYS_OUTDIR="$OUTDIR" PROFILE_RANKS PY_BIN="$PY"

# block:block is what puts ranks 0-3 on the first node; the wrapper records
# the rank->node map so that placement is verified from the receipt rather
# than trusted.
JAX_ENABLE_X64=0 srun --ntasks=64 --distribution=block:block \
  --cpu-bind=cores --kill-on-bad-exit=1 \
  "$WRAPPER" scripts/bench/bench_mpas_spmd_scaling.py \
    --n-devices 64 --subdivision 9 --nlev 26 --steps 12 --warmup 3 \
    --lloyd 0 --partition-method sfc --reorder-for 128 --multicontroller \
    --out "$OUTDIR/s9np64.jsonl"
RC=$?
echo "[skew] srun exit=$RC"

# An exit code is not evidence a profile exists. Require one sqlite per
# requested rank, or fail: a job that reports COMPLETED with no usable
# capture is worse than one that reports failure.
missing=""
IFS=',' read -ra WANT <<< "$PROFILE_RANKS"
for r in "${WANT[@]}"; do
  if [ ! -s "$OUTDIR/rank_${r}.sqlite" ]; then
    # --export=sqlite should have produced it; retry explicitly before giving up.
    if [ -s "$OUTDIR/rank_${r}.nsys-rep" ]; then
      echo "[skew] rank $r: exporting sqlite explicitly"
      "$NSYS_BIN" export --type=sqlite --force-overwrite=true \
          --output "$OUTDIR/rank_${r}.sqlite" "$OUTDIR/rank_${r}.nsys-rep" \
          || echo "[skew] rank $r: export FAILED"
    fi
  fi
  [ -s "$OUTDIR/rank_${r}.sqlite" ] || missing="$missing $r"
done

echo "[skew] rank->node map:"; cat "$OUTDIR/_rank_nodes.tsv" 2>/dev/null | sort -t= -k2 | head -8

if [ -n "$missing" ]; then
  echo "[skew] MISSING sqlite for ranks:$missing"
  echo "SKEW_CAPTURE_INCOMPLETE"
  exit 4
fi
if [ "$RC" -ne 0 ]; then
  echo "SKEW_CAPTURE_RUN_FAILED"
  exit "$RC"
fi
echo "SKEW_CAPTURE_DONE"
