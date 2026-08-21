#!/bin/bash -l
# Where does the halo exchange's bandwidth actually go when the job gets wider?
#
# A first rank ladder measured 23.8 GB/s at 32 ranks and 10.5 GB/s at 64 and
# 128 -- a halving between 32 and 64 that then stayed flat to three digits. Two
# independent reviews rejected both halves of that reading, and this job answers
# every objection they raised, in one allocation.
#
# Objection 1, placement. The three arms were separate allocations: the 32-rank
# arm sat on eight neighbouring nodes and the 64-rank arm on two widely
# separated groups, so rank count and network placement moved together. Here
# every arm runs inside ONE allocation asked for on a single switch, and each
# arm is pinned to an EXPLICIT prefix of that allocation's node list, so the
# smaller arms are a strict subset of the larger ones rather than whatever
# subset the scheduler picks that minute.
#
# Objection 2, the instrument. A ring in rank order has exactly one off-node
# link per four-GPU node, so its completion time is set by how fast one node can
# push data out, and that does not change when the node count doubles. Its
# flatness is therefore a property of the instrument, not of the fabric. The
# stride sweep fixes that by moving the partner.
#
# Objection 3, the stride sweep's own confound. A stride sharing a common factor
# with the device count splits the ring into several short cycles, and cycle
# structure -- not distance -- can change what the collective library does. So
# the distance sweep uses ODD strides only: at 64 devices every odd stride is
# coprime with 64 and gives one single 64-cycle, holding structure fixed while
# distance goes from 5 ranks to 33. Stride 1 is reported separately and NOT read
# as a distance point, because it differs in kind: three of its four links never
# leave the node.
#
# Objection 4, a measurement that cannot fail. Every arm's exit status is kept,
# every receipt is checked to exist and to record the stride and device count it
# was asked for, and the job exits non-zero if any arm is missing. The benchmark
# itself now refuses to time a pattern until each device has been shown to
# receive its source's index, so "the permutation did not reach the devices" is
# no longer an untestable escape hatch.
#
# Falsifiable, written before submit:
#   Bandwidth flat across 16/32/64 ranks at pinned placement -> the earlier
#     halving was PLACEMENT, and a topology constraint on the job is the lever.
#   Bandwidth still halves at 64 -> rank count or fabric contention is real.
#   Odd strides 5..33 all equal -> distance across the allocation is free, and
#     topology-aware rank ordering is not worth building.
#   Odd strides rising with distance -> distance is a real cost the model pays
#     whenever its neighbour ranks are placed far apart.
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=16
#SBATCH --switches=1@00:20:00
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:50:00
#SBATCH --job-name=pp_switch
#SBATCH --output=pp_switch.%j.log
#SBATCH --exclude=l50003,l50139,l50027,l50009,l50100
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
export LEGOESM_JIT_CACHE_DIR="$SCRATCH/legoesm_jit_cache_j${SLURM_JOB_ID}"
export LEGOESM_XLA_OVERLAP=0
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_shard_autotuning=false --xla_gpu_enable_nccl_comm_splitting=false"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="$SCRATCH/legoesm_scaling/pp_switch_j${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"
export NCCL_MIN_NCHANNELS=8 NCCL_MAX_NCHANNELS=8 NCCL_P2P_NET_CHUNKSIZE=131072

# The allocation's nodes, in one fixed order for the whole job. Every arm takes
# a PREFIX of this list, so a 4-node arm runs on the same four nodes every time
# and is a subset of the 8-node arm, which is a subset of the 16-node arm.
mapfile -t NODES_ALL < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
echo "=== allocation (${#NODES_ALL[@]} nodes): ${NODES_ALL[*]} ==="
ARMS_RUN=(); ARMS_FAILED=()

run_arm() {   # ndev stride tag
  local NDEV=$1 STRIDE=$2 TAG=$3
  local NNODE=$(( (NDEV + 3) / 4 )) NTPN=$(( NDEV < 4 ? NDEV : 4 ))
  local LIST; LIST=$(IFS=,; echo "${NODES_ALL[*]:0:$NNODE}")
  echo "--- $TAG ndev=$NDEV stride=$STRIDE nodes=$LIST $(date +%T)"
  ARMS_RUN+=("$TAG:$NDEV:$STRIDE")
  timeout 900 srun --export=ALL --nodelist="$LIST" --nodes="$NNODE" \
      --ntasks="$NDEV" --ntasks-per-node="$NTPN" --gpus-per-node="$NTPN" \
      --gpu-bind=none --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_ppermute_microbench.py \
      --multicontroller --n-devices "$NDEV" --n-iters 50 --n-reps 64 \
      --ring-stride "$STRIDE" \
      --out "$OUTDIR/${TAG}.jsonl" \
    || { echo "ARM $TAG FAILED"; ARMS_FAILED+=("$TAG"); }
}

# Rank count at pinned placement. The second repetition runs the rank counts in
# the OPPOSITE order, so a drift in the machine over the job's lifetime cancels
# instead of always favouring the arms that ran first.
for NDEV in 4 16 32 64; do run_arm "$NDEV" 1 "n${NDEV}_s1_r1"; done
for NDEV in 64 32 16 4; do run_arm "$NDEV" 1 "n${NDEV}_s1_r2"; done

# Distance at fixed rank count, fixed placement and fixed cycle structure.
# Every stride here is odd, hence coprime with 64, hence one 64-cycle; and every
# one is >= 4, hence every link leaves the node. Distance in nodes: 2, 3, 5, 9.
for STRIDE in 5 9 17 33; do run_arm 64 "$STRIDE" "n64_s${STRIDE}_r1"; done
for STRIDE in 33 17 9 5; do run_arm 64 "$STRIDE" "n64_s${STRIDE}_r2"; done

# What the collective library actually built at the two rank counts that
# straddle the reported halving. These arms use far fewer iterations and turn on
# debug logging, so they are NOT comparable to the timed arms; their receipts
# record n_iters so nothing can glob them into a fit by accident. Kept last: a
# failure here must not cost the timings.
for NDEV in 32 64; do
  NNODE=$(( NDEV / 4 ))
  LIST=$(IFS=,; echo "${NODES_ALL[*]:0:$NNODE}")
  echo "--- NCCL topology dump ndev=$NDEV nodes=$LIST"
  NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV \
    timeout 600 srun --export=ALL --nodelist="$LIST" --nodes="$NNODE" \
      --ntasks="$NDEV" --ntasks-per-node=4 --gpus-per-node=4 --gpu-bind=none \
    "$PY" scripts/bench/bench_ppermute_microbench.py \
      --multicontroller --n-devices "$NDEV" --n-iters 2 --n-reps 8 \
      --out "$OUTDIR/nccldbg_n${NDEV}.jsonl" \
    > "$OUTDIR/nccl_n${NDEV}.log" 2>&1 || echo "NCCL dump ndev=$NDEV FAILED"
done

# A job that reports success with no receipts is worse than one that crashes.
"$PY" - "$OUTDIR" "${ARMS_RUN[@]}" <<'PYCHECK'
import json, sys
outdir, arms = sys.argv[1], sys.argv[2:]
bad = []
for arm in arms:
    tag, ndev, stride = arm.split(":")
    path = f"{outdir}/{tag}.jsonl"
    try:
        rows = [json.loads(l) for l in open(path) if l.strip()]
    except OSError:
        bad.append(f"{tag}: no receipt")
        continue
    if not rows:
        bad.append(f"{tag}: empty receipt")
        continue
    r = rows[-1]
    if r.get("n_devices") != int(ndev) or r.get("ring_stride") != int(stride):
        bad.append(f"{tag}: receipt says n_devices={r.get('n_devices')} "
                   f"ring_stride={r.get('ring_stride')}, asked for {ndev}/{stride}")
    elif not (r.get("bandwidth_gbs", 0) > 0):
        bad.append(f"{tag}: bandwidth {r.get('bandwidth_gbs')} is not positive")
    else:
        print(f"  {tag}: {r['latency_us']:7.2f} us  {r['bandwidth_gbs']:6.2f} GB/s")
if bad:
    print("MISSING OR INVALID ARMS:")
    for b in bad:
        print("  " + b)
    sys.exit(1)
print(f"all {len(arms)} timed arms present")
PYCHECK
rc=$?
if [ "${#ARMS_FAILED[@]}" -gt 0 ]; then
  echo "ARMS THAT FAILED TO RUN: ${ARMS_FAILED[*]}"; rc=1
fi
echo "DONE rc=$rc"; exit $rc
