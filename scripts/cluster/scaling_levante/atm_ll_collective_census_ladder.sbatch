#!/bin/bash -l
# Why does this lane's communication term GROW with device count when its
# per-device halo cannot?
#
# A device owns a band of latitude rows and exchanges two rows of the full
# longitude circle, so its halo payload is the same at 32, 64 and 128 devices.
# The neighbour ranks are the ones next to it, and with four GPUs to a node
# three of every four neighbour links stay inside a node AT EVERY SCALE. An
# isolated exchange on these nodes is flat in rank count to within 9%.
# Everything about the fabric therefore predicts a FLAT communication term.
# The measurement says 0.664 ms at 32 devices, 1.551 at 64, 1.665 at 128.
#
# Both independent reviews landed on the same objection: that contradiction is
# the most suspicious number in the campaign, and it points at the program
# rather than the machine -- a collective whose size grows with the device
# count (an all-gather-shaped halo hiding among the neighbour exchanges), or
# the collective library changing algorithm as the communicator grows.
#
# This asks the executable rather than the machine. Each device count dumps the
# compiled module from rank 0 and the census counts the collectives and the
# bytes they carry. Same resolution, same flags, same allocation.
#
# Falsifiable, written before submit:
#   Halo bytes per rank CONSTANT across 32/64/128 and the collective count
#     constant -> the growth is in the runtime, and the next move is transport
#     level (explicit neighbour send/recv on separate communicators).
#   Bytes or count GROW with the device count -> there is an O(N) collective in
#     the step, that is the whole effect, and removing it is worth more than
#     any transport work.
#
# The no-communication arm is censused too, and that is not padding. The
# communication term is measured as the FULL step minus that arm, so any
# collective the compiler inserted for its own reasons -- a reshard, a gather
# feeding a reduction -- is present in BOTH arms and cancels out of the
# subtraction. It would then sit inside the "local" number, which at 128
# devices is 45% above its ideal share. That is 0.71 ms, nearly half the size
# of the communication term, and nobody has looked inside it. If the no-comm
# arm turns out to contain collectives, the whole ranking of what to fix next
# changes.
#SBATCH --job-name=ll_census
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=32
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --exclude=l50003,l50139,l50027,l50009,l50100
#SBATCH --output=ll_census.%j.log
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
export LEGOESM_XLA_OVERLAP=1
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="$SCRATCH/legoesm_scaling/ll_census_j${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$OUTDIR/jitcache"; mkdir -p "$LEGOESM_JIT_CACHE_DIR"
export NCCL_MIN_NCHANNELS=8 NCCL_MAX_NCHANNELS=8 NCCL_P2P_NET_CHUNKSIZE=131072

# Only rank 0 dumps; every rank would write the same module 128 times over.
cat > "$OUTDIR/rank_wrapper.sh" <<'WRAP'
#!/usr/bin/env bash
set -uo pipefail
if [ "${SLURM_PROCID:-1}" = "0" ]; then
  export XLA_FLAGS="--xla_dump_to=${DUMP_DIR}${XLA_FLAGS:+ $XLA_FLAGS}"
fi
exec "$PY_BIN" "$@"
WRAP
chmod +x "$OUTDIR/rank_wrapper.sh"
export PY_BIN="$PY"
rc=0
for NDEV in 32 64 128; do
 for NC in 0 1; do
  NODES=$(( NDEV / 4 ))
  if [ "$NC" = 1 ]; then ARM="nocomm"; else ARM="full"; fi
  export DUMP_DIR="$OUTDIR/dump_${ARM}_n${NDEV}"; mkdir -p "$DUMP_DIR"
  echo "=== census arm ndev=$NDEV $ARM $(date +%T) ==="
  LEGOESM_LATLON_PACKED_EXCHANGE=1 LEGOESM_LATLON_SPMD_FUSED_HALO=1 \
  LEGOESM_LATLON_HALO_NOCOMM=$NC JAX_ENABLE_X64=0 \
    timeout 1200 srun --export=ALL --nodes="$NODES" --ntasks="$NDEV" \
      --ntasks-per-node=4 --gpus-per-node=4 --gpu-bind=none \
      --kill-on-bad-exit=1 \
    "$OUTDIR/rank_wrapper.sh" scripts/bench/bench_atm_latlon_spmd_scaling.py \
      --multicontroller --n-devices "$NDEV" --mode strong \
      --n-lat 2048 --n-lon 4096 --nlev 26 \
      --steps 12 --warmup 3 \
      --out "$OUTDIR/step_${ARM}_n${NDEV}.jsonl" \
    || { echo "ARM ndev=$NDEV $ARM FAILED"; rc=1; }
  "$PY" scripts/bench/census_xla_dump.py --dump-dir "$DUMP_DIR" \
      --out "$OUTDIR/census_${ARM}_n${NDEV}.json" \
    || { echo "CENSUS ndev=$NDEV $ARM FAILED"; rc=1; }
 done
done
echo "=== SUMMARY ==="
"$PY" - "$OUTDIR" <<'PYSUM'
import json, sys, os
outdir = sys.argv[1]
rows = {}
for arm in ("full", "nocomm"):
    for n in (32, 64, 128):
        p = f"{outdir}/census_{arm}_n{n}.json"
        if not os.path.exists(p):
            print(f"  {arm} ndev={n}: no census"); continue
        rows[f"{arm}_n{n}"] = json.load(open(p))
if not rows:
    sys.exit(1)
print(json.dumps(rows, indent=2)[:8000])
PYSUM
echo "DONE rc=$rc"; exit $rc
