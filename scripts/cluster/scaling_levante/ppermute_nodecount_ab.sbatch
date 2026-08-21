#!/bin/bash -l
# Is the exchange slower because there are more RANKS, or because they sit on
# more NODES?
#
# With four GPUs per node those two always moved together: 32 ranks meant eight
# nodes and 64 ranks meant sixteen. Pinning the arms to a fixed prefix of one
# allocation removed the scheduler from the picture but not this confound, and
# the collective library's own log showed why it matters -- the sixteen-node
# prefix reaches into a second network subnet that the eight-node prefix does
# not, so the "more ranks" arm is also the "further apart" arm.
#
# Putting fewer processes on each node breaks the tie. Thirty-two ranks spread
# two to a node occupy the SAME sixteen nodes as the sixty-four-rank arm, so
# they see the same network, with half the ranks.
#
# Falsifiable, written before submit. Compare 32 ranks on 8 nodes against 32
# ranks on 16 nodes:
#   Same bandwidth -> RANK COUNT is what costs, and spreading a fixed problem
#     over more nodes is free. Fewer, fatter ranks is then the lever.
#   16-node arm as slow as the 64-rank arm -> NODE COUNT (or the network span
#     it implies) is what costs, and packing the job into as few nodes as
#     possible is the lever, whatever the rank count.
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=16
#SBATCH --switches=1@00:20:00
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --job-name=pp_nodes
#SBATCH --output=pp_nodes.%j.log
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
OUTDIR="$SCRATCH/legoesm_scaling/pp_nodes_j${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"
export NCCL_MIN_NCHANNELS=8 NCCL_MAX_NCHANNELS=8 NCCL_P2P_NET_CHUNKSIZE=131072

mapfile -t NODES_ALL < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
echo "=== allocation (${#NODES_ALL[@]} nodes): ${NODES_ALL[*]} ==="
# The network address of each node, so a subnet boundary inside the allocation
# is on the record rather than discovered later in a debug log.
for h in "${NODES_ALL[@]}"; do echo "  $h $(getent hosts "${h}i" 2>/dev/null | awk '{print $1}')"; done
ARMS_RUN=(); ARMS_FAILED=()

run_arm() {   # ndev ntasks_per_node tag
  local NDEV=$1 NTPN=$2 TAG=$3
  local NNODE=$(( NDEV / NTPN ))
  local LIST; LIST=$(IFS=,; echo "${NODES_ALL[*]:0:$NNODE}")
  echo "--- $TAG ndev=$NDEV ppn=$NTPN nodes=$NNODE ($LIST) $(date +%T)"
  ARMS_RUN+=("$TAG:$NDEV")
  timeout 900 srun --export=ALL --nodelist="$LIST" --nodes="$NNODE" \
      --ntasks="$NDEV" --ntasks-per-node="$NTPN" --gpus-per-node="$NTPN" \
      --gpu-bind=none --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_ppermute_microbench.py \
      --multicontroller --n-devices "$NDEV" --n-iters 50 --n-reps 64 \
      --ring-stride 1 \
      --out "$OUTDIR/${TAG}.jsonl" \
    || { echo "ARM $TAG FAILED"; ARMS_FAILED+=("$TAG"); }
}

# ndev  ppn  nodes   what it isolates
#   16    4     4    small reference, few nodes
#   32    4     8    the fast arm from the previous job
#   32    2    16    SAME rank count, DOUBLE the nodes  <- the discriminator
#   64    4    16    both doubled
for A in "16 4 n16_p4" "32 4 n32_p4" "32 2 n32_p2" "64 4 n64_p4"; do
  run_arm $A"_r1"
done
for A in "64 4 n64_p4" "32 2 n32_p2" "32 4 n32_p4" "16 4 n16_p4"; do
  run_arm $A"_r2"
done

"$PY" - "$OUTDIR" "${ARMS_RUN[@]}" <<'PYCHECK'
import json, sys
outdir, arms = sys.argv[1], sys.argv[2:]
bad = []
for arm in arms:
    tag, ndev = arm.split(":")
    try:
        rows = [json.loads(l) for l in open(f"{outdir}/{tag}.jsonl") if l.strip()]
    except OSError:
        bad.append(f"{tag}: no receipt"); continue
    if not rows:
        bad.append(f"{tag}: empty receipt"); continue
    r = rows[-1]
    if r.get("n_devices") != int(ndev):
        bad.append(f"{tag}: receipt says n_devices={r.get('n_devices')}, asked {ndev}")
    elif not (r.get("bandwidth_gbs", 0) > 0):
        bad.append(f"{tag}: bandwidth {r.get('bandwidth_gbs')} not positive")
    else:
        print(f"  {tag}: {r['latency_us']:7.2f} us  {r['bandwidth_gbs']:6.2f} GB/s")
if bad:
    print("MISSING OR INVALID ARMS:")
    for b in bad: print("  " + b)
    sys.exit(1)
print(f"all {len(arms)} arms present")
PYCHECK
rc=$?
if [ "${#ARMS_FAILED[@]}" -gt 0 ]; then echo "ARMS THAT FAILED: ${ARMS_FAILED[*]}"; rc=1; fi
echo "DONE rc=$rc"; exit $rc
