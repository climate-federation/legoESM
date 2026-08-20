#!/bin/bash -l
# Where does the halo exchange's bandwidth actually go when the job gets wider?
#
# A first rank ladder measured 23.8 GB/s at 32 ranks and 10.5 GB/s at 64 and
# 128 -- a halving between 32 and 64 that then stays flat to three digits. Two
# independent reviews rejected both halves of that reading, for two different
# reasons, and this job answers both objections in one allocation.
#
# Objection 1 (placement): the three arms were separate allocations, and the
# 32-rank arm happened to sit on eight neighbouring nodes while the 64-rank arm
# sat on two widely separated groups. Rank count and network placement moved
# together. Here every rank count runs inside ONE allocation, requested on a
# single switch, interleaved and repeated.
#
# Objection 2 (the instrument): a rank-ordered ring has exactly ONE off-node
# link per four-GPU node, so its completion time is set by per-node egress,
# which does not change when the node count doubles. Such a ring is incapable
# of showing a cost that grows with the width of the job, so its flatness
# proves nothing. The stride sweep fixes that: stride 1 keeps three of every
# four links on NVLink, stride 4 puts every link on the network between
# adjacent nodes, and stride 32 puts every link across the whole allocation.
#
# The stride sweep is also this job's own control. If stride 32 times exactly
# like stride 1, then either the fabric is genuinely flat OR the permutation is
# not reaching the devices I think it is -- and the NCCL topology dump printed
# by the last arm says which.
#
# Falsifiable, written before submit:
#   Bandwidth flat across 16/32/64 ranks at fixed placement -> the earlier
#     halving was PLACEMENT, and a topology constraint on the job is the lever.
#   Bandwidth still halves at 64 -> rank count or fabric contention is real.
#   Stride 32 much slower than stride 1 -> the ring was measuring node egress
#     only, and distance across the allocation is a cost the model pays too.
#   Stride 32 equal to stride 1 -> distance is free here; stop pursuing
#     topology-aware placement.
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=16
#SBATCH --switches=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:45:00
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
echo "=== nodes: $SLURM_JOB_NODELIST ==="

run_arm() {   # ndev stride tag extra_env
  local NDEV=$1 STRIDE=$2 TAG=$3
  local NODES=$(( (NDEV + 3) / 4 )) NTPN=$(( NDEV < 4 ? NDEV : 4 ))
  echo "--- $TAG ndev=$NDEV stride=$STRIDE $(date +%T)"
  timeout 900 srun --export=ALL --nodes="$NODES" --ntasks="$NDEV" \
      --ntasks-per-node="$NTPN" --gpus-per-node="$NTPN" \
      --gpu-bind=none --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_ppermute_microbench.py \
      --multicontroller --n-devices "$NDEV" --n-iters 50 --n-reps 64 \
      --ring-stride "$STRIDE" \
      --out "$OUTDIR/${TAG}.jsonl" || echo "ARM $TAG FAILED"
}

# Rank count at fixed placement, interleaved so a drift in the machine does not
# land entirely on the last arms.
for REP in 1 2; do
  for NDEV in 4 16 32 64; do
    run_arm "$NDEV" 1 "n${NDEV}_s1_r${REP}"
  done
done
# Distance at fixed rank count and fixed placement.
for REP in 1 2; do
  for STRIDE in 4 16 32; do
    run_arm 64 "$STRIDE" "n64_s${STRIDE}_r${REP}"
  done
done
# What the collective library actually built, at the two rank counts that
# straddle the reported halving. Kept last: the debug output is large and a
# failure here must not cost the timings.
for NDEV in 32 64; do
  NODES=$(( NDEV / 4 ))
  echo "--- NCCL topology dump ndev=$NDEV"
  NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV \
    timeout 600 srun --export=ALL --nodes="$NODES" --ntasks="$NDEV" \
      --ntasks-per-node=4 --gpus-per-node=4 --gpu-bind=none \
    "$PY" scripts/bench/bench_ppermute_microbench.py \
      --multicontroller --n-devices "$NDEV" --n-iters 2 --n-reps 8 \
      --out "$OUTDIR/nccldbg_n${NDEV}.jsonl" \
    > "$OUTDIR/nccl_n${NDEV}.log" 2>&1 || echo "NCCL dump ndev=$NDEV FAILED"
done
echo DONE
