#!/bin/bash -l
# Would letting the compiler place the halo beat writing it by hand?
#
# The lat-lon lane's halo is hand-written: a shard_map over latitude with
# explicit neighbour exchanges. JAX can instead take globally-shaped arrays with
# a sharding annotation and insert whatever communication the program needs.
# That is less code to own, and the reviews of this campaign disagreed about
# whether it would even produce neighbour exchanges: one expected all-gathers,
# which would be a different program with a different scaling law rather than a
# slower version of the same one.
#
# Nothing in the model changes. The benchmark is a new file, the two arms live
# inside it, and reverting the experiment is deleting it.
#
# The arms compute the SAME VALUES and the benchmark refuses to time them until
# it has checked that, which earned its keep immediately: the first version had
# the neighbour direction backwards on the hand-written side, a program that
# ran, exchanged the right number of bytes, and was wrong.
#
# Falsifiable, written before submit:
#   auto emits all-gathers -> it is a different program, its cost grows with
#     device count, and the hand-written halo stays regardless of timing.
#   auto emits neighbour exchanges and is within noise -> the hand-written halo
#     is code we could stop owning, and that is worth doing on its own.
#   auto emits neighbour exchanges and is FASTER -> the compiler is scheduling
#     the exchange better than the hand-written path, and the interesting
#     question becomes what it does differently.
#   auto is slower -> keep the hand-written halo and record the margin.
#SBATCH --job-name=ll_auto
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=32
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:50:00
#SBATCH --exclude=l50003,l50139,l50027,l50009,l50100
#SBATCH --output=ll_auto.%j.log
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
OUTDIR="$SCRATCH/legoesm_scaling/ll_auto_j${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$OUTDIR/jitcache"; mkdir -p "$LEGOESM_JIT_CACHE_DIR"
export NCCL_MIN_NCHANNELS=8 NCCL_MAX_NCHANNELS=8 NCCL_P2P_NET_CHUNKSIZE=131072
mapfile -t NODES_ALL < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
rc=0
# Smaller counts take a PREFIX of the same allocation, so they run on a subset
# of the larger runs' nodes rather than a different draw.
for NDEV in 32 64 128; do
  NNODE=$(( NDEV / 4 ))
  LIST=$(IFS=,; echo "${NODES_ALL[*]:0:$NNODE}")
  echo "=== ndev=$NDEV nodes=$NNODE $(date +%T) ==="
  JAX_ENABLE_X64=0 \
    timeout 1200 srun --export=ALL --nodelist="$LIST" --nodes="$NNODE" \
      --ntasks="$NDEV" --ntasks-per-node=4 --gpus-per-node=4 \
      --gpu-bind=none --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_latlon_auto_vs_manual_halo.py \
      --multicontroller --n-devices "$NDEV" \
      --n-lat 2048 --n-lon 4096 --nlev 26 \
      --n-iters 10 --reps 20 --warmup 5 \
      --out "$OUTDIR/auto_vs_manual_n${NDEV}.jsonl" \
    || { echo "ARM ndev=$NDEV FAILED"; rc=1; }
done
echo "=== SUMMARY ==="
"$PY" - "$OUTDIR" <<'PYSUM'
import json, glob, os, sys
outdir = sys.argv[1]
rows = []
for f in sorted(glob.glob(os.path.join(outdir, "auto_vs_manual_n*.jsonl"))):
    try:
        rows.append(json.loads(open(f).read().splitlines()[-1]))
    except (OSError, IndexError, ValueError):
        print(f"  {os.path.basename(f)}: unreadable")
if not rows:
    print("NO RECEIPTS"); sys.exit(1)
rows.sort(key=lambda r: r["n_devices"])
print(f"{'devices':>7} {'manual ms':>10} {'auto ms':>10} {'auto/manual':>12}  collectives")
for r in rows:
    m, a = r["arms"]["manual"], r["arms"]["auto"]
    print(f"{r['n_devices']:>7} {m['ms']:>10.4f} {a['ms']:>10.4f} "
          f"{a['ms']/m['ms']:>12.3f}  manual={m['collectives']} "
          f"auto={a['collectives']}  agree_to={r['worst_relative_difference']:.2e}")
PYSUM
echo "DONE rc=$rc"; exit $rc
