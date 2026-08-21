#!/bin/bash -l
# Where does the time go when the shard gets thin?
#
# Two hypotheses for this lane's non-scaling local work are now dead. The
# global mass fixer's all-reduce is present in the no-halo arm exactly as
# predicted and costs nothing measurable at any device count. And the halo rows
# are not recomputed: no operation in the compiled module writes a tensor
# taller than the owned row count.
#
# What the compiled module does show is a fixed-size class of EDGE-SHAPED work.
# Bytes written per step by operations producing a rows-by-longitude-by-level
# tensor, grouped by the output's row count: at 32 devices, 184.6 MB on
# owned-row outputs and 74.3 MB on one- and two-row outputs; at 128 devices,
# 53.0 MB and 76.8 MB. The interior falls with the shard and the edge work does
# not move at all, so it goes from 29% of the bytes to 59%.
#
# Bytes written are not time, and this campaign has twice mistaken a count for a
# time. A one-row output can be a cheap fused write, and the shrinking interior
# can dominate the clock through reads, reductions and occupancy that a byte
# census cannot see. So this traces it.
#
# It also carries the other open question. The exchange count is thirteen at
# every device count and the per-device payload is fixed, yet the communication
# term grows 0.664 -> 1.551 -> 1.665 ms from 32 to 128 devices, and an isolated
# ring exchange on these nodes is flat in rank count. What is left is lost
# overlap as the interior shrinks, placement, serialisation among the thirteen,
# packing charged to the subtraction, or rank skew. A timeline separates them;
# arithmetic on totals cannot.
#
# Falsifiable, written before submit:
#   edge-shaped kernels hold a large and growing share of GPU time  -> the fixed
#     cost is real and the fix is to fuse or batch those operations.
#   edge kernels are a small share and the interior kernels have simply become
#     inefficient at sixteen rows -> the fix is the interior, or the device
#     count is past this resolution's useful limit.
#   the exchanges' first consumers wait longer at 128 than at 32 while the
#     exchanges themselves take the same time -> the communication growth is
#     LOST OVERLAP, not the wire.
#SBATCH --job-name=ll_trshrink
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=32
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --exclude=l50003,l50139,l50027,l50009,l50100
#SBATCH --output=ll_trshrink.%j.log
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
export LEGOESM_XLA_OVERLAP=1
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_shard_autotuning=false --xla_gpu_enable_nccl_comm_splitting=false"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="$SCRATCH/legoesm_scaling/ll_trshrink_j${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$OUTDIR/jitcache"; mkdir -p "$LEGOESM_JIT_CACHE_DIR"
export NCCL_MIN_NCHANNELS=8 NCCL_MAX_NCHANNELS=8 NCCL_P2P_NET_CHUNKSIZE=131072
rc=0
# 32 and 128 devices are the two ends of the trend, and both take a PREFIX of
# the same allocation so the node set is a subset rather than a different draw.
mapfile -t NODES_ALL < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
for NDEV in ${NDEVS:-32 128}; do
 for NC in 0 1; do
  NNODE=$(( NDEV / 4 ))
  LIST=$(IFS=,; echo "${NODES_ALL[*]:0:$NNODE}")
  if [ "$NC" = 1 ]; then ARM=nohalo; else ARM=full; fi
  T="$OUTDIR/traces_${ARM}_n${NDEV}"
  echo "=== trace ndev=$NDEV $ARM nodes=$NNODE $(date +%T) ==="
  LEGOESM_LATLON_PACKED_EXCHANGE=1 LEGOESM_LATLON_SPMD_FUSED_HALO=1 \
  LEGOESM_LATLON_HALO_NOCOMM=$NC JAX_ENABLE_X64=0 \
    timeout 2700 srun --export=ALL --nodelist="$LIST" --nodes="$NNODE" \
      --ntasks="$NDEV" --ntasks-per-node=4 --gpus-per-node=4 \
      --gpu-bind=none --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_atm_latlon_spmd_scaling.py \
      --multicontroller --n-devices "$NDEV" --mode strong \
      --n-lat 2048 --n-lon 4096 --nlev 26 --steps 12 --warmup 3 \
      --profile-dir "$T" \
      --out "$OUTDIR/step_${ARM}_n${NDEV}.jsonl" \
    || { echo "ARM ndev=$NDEV $ARM FAILED"; rc=1; }
  set +e
  "$PY" scripts/bench/analyze_jax_trace_gaps.py --trace-root "$T" \
      --out "$OUTDIR/trace_${ARM}_n${NDEV}.json"
  [ $? -ne 0 ] && { echo "ANALYSIS ndev=$NDEV $ARM FAILED"; rc=1; }
  set -uo pipefail
 done
done
echo "=== SUMMARY ==="
"$PY" - "$OUTDIR" <<'PYSUM'
import json, glob, os, sys
outdir = sys.argv[1]
for f in sorted(glob.glob(os.path.join(outdir, "trace_*.json"))):
    print(f"--- {os.path.basename(f)}")
    try:
        print(json.dumps(json.load(open(f)), indent=2)[:2500])
    except Exception as exc:
        print(f"  unreadable: {exc}")
PYSUM
echo "DONE rc=$rc"; exit $rc
