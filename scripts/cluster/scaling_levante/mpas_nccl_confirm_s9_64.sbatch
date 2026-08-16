#!/bin/bash -l
#SBATCH --job-name=nccl_confirm
#SBATCH --account=bd1083_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=16
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:50:00
#SBATCH --output=nccl_confirm.%j.log
# ===========================================================================
# NCCL ENV LADDER at s9@64, wide halo — INTERCEPT/SLOPE design (v2, after
# GLM r3 review of v1; assessment 2026-08-16 section c lever i).
#
# v1's flaw (GLM r3): a protocol arm moves BOTH the payload slope and the
# per-op intercept, so "step got faster" cannot be attributed; and a bare
# channels arm is likely INERT for pairwise sendrecv (multi-channel p2p is
# gated by NCCL_P2P_NET_CHUNKSIZE, not by channel count alone).
#
# v2 design: every env runs at ballast 1 AND 2 (bytes x1 / x2, same rounds,
# same schedule), twice interleaved. Two-point fit per env:
#     slope_env     = t(b2) - t(b1)          (payload cost)
#     intercept_env = 2*t(b1) - t(b2)        (compute + per-op overhead)
# The per-op mechanism claim lives in the INTERCEPT delta vs base.
# Envs (each ONE named lever vs base):
#   base   : stack defaults (what production runs)
#   mcp2p  : NCCL_MIN/MAX_NCHANNELS=8 + NCCL_P2P_NET_CHUNKSIZE=131072
#            (the pair that actually enables multi-channel p2p; composite
#            on purpose — channels alone is a known-inert negative control)
#   ll128  : NCCL_PROTO=LL128 (the latency-family protocol lever)
#
# PRE-REGISTERED GATE (written before the run):
#   CONFIRM (per-op overhead is env-movable): some env has
#     intercept_env <= intercept_base - 0.30 ms, with the two-rep spread of
#     each intercept < half that delta.
#   INCONCLUSIVE: best intercept delta in (-0.30, -0.15] ms — expected LL128
#     range is 0.1-0.3 ms (10-20 us/op x 11-14 ops), so this band is named
#     rather than folded into REFUTE (v1 had a REFUTE bias here).
#   REFUTE (these levers do not reach it): all |intercept deltas| < 0.15 ms.
#     Scope: refutes the LEVERS, not the per-op mechanism — if the residual
#     is XLA launch gaps, no NCCL env can move it; the fused-exchange build
#     and (stack-permitting) a gap attribution are the next discriminators.
#   Slope deltas are reported separately as protocol/bandwidth effects and
#   NEVER count toward CONFIRM.
#   Why no cheaper test: the residual exists only at multi-node scale (the
#   4-device colouring is 3 all-NVLink rounds); vader cannot reach it.
# VALIDITY: each env also runs a 5-step arm with NCCL_DEBUG=INFO
# SUBSYS=ENV,TUNING to a per-rank file; the receipt greps that the knob was
# seen by NCCL — an env arm without its validity lines is VOID, not REFUTED.
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
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/nccl_confirm_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
# Multi-shard autotune sync crashed ALL 16 arms of job 26998105 at 64
# ranks (same "No configuration found in cache after synchronizing"
# failure as the 192-rank jobs 26979367/68) — disable it, and use a
# per-job JIT cache in case the shared one carries a poisoned autotune
# entry from those crashes.
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_shard_autotuning=false"
export LEGOESM_JIT_CACHE_DIR="$SCRATCH/legoesm_scaling/_jitcache_j${SLURM_JOB_ID}"
mkdir -p "$LEGOESM_JIT_CACHE_DIR"
rc=0

NODES=16; GPN=4; SUBDIV=9; NDEV=$((NODES * GPN))

ENV_base=""
ENV_mcp2p="NCCL_MIN_NCHANNELS=8 NCCL_MAX_NCHANNELS=8 NCCL_P2P_NET_CHUNKSIZE=131072"
ENV_ll128="NCCL_PROTO=LL128"

run_arm () {  # tag env_name ballast steps [validity]
  local TAG=$1 ENVN=$2 BAL=$3 STEPS=$4 VAL=${5:-0}
  local EXTRA; eval "EXTRA=\$ENV_${ENVN}"
  local DBG=""
  if [ "$VAL" = 1 ]; then
    DBG="NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ENV,INIT,TUNING,P2P NCCL_DEBUG_FILE=$OUTDIR/nccl_${ENVN}.%h.%p"
  fi
  echo "=== s${SUBDIV}@${NDEV} wide=1 [$TAG] ballast=$BAL env: ${EXTRA:-defaults} ==="
  printf '%s\n' "$EXTRA" > "$OUTDIR/${TAG}.env"   # requested-env receipt
  env $EXTRA $DBG \
  LEGOESM_MPAS_WIDE_HALO=1 LEGOESM_MPAS_RAGGED_HALO=0 \
  LEGOESM_MPAS_HALO_BALLAST=$BAL JAX_ENABLE_X64=0 \
    timeout 1800 srun --export=ALL --nodes="$NODES" --ntasks="$NDEV" \
      --ntasks-per-node="$GPN" --gpus-per-node="$GPN" \
      --gpu-bind=none --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
      --multicontroller --n-devices "$NDEV" \
      --subdivision "$SUBDIV" --nlev 26 \
      --steps "$STEPS" --warmup 5 --lloyd 0 --partition-method sfc \
      --reorder-for "$NDEV" \
      --out "$OUTDIR/${TAG}.jsonl" || { echo "${TAG} FAILED"; rc=1; }
  sleep 15
}

# Discarded warm-up arm (node ramp must not alias into the first timed
# arm), then the 12 timed arms in a PALINDROME (codex r3: a plain repeat
# gives each env a different mean time-position, so linear drift biases
# the deltas; the palindrome equalises mean positions).
run_arm warmup_discard base 1 65
# CONFIRM subset for the timed-out 26999539 run: the mcp2p arm read
# 5.91 vs base 6.75 ms at ballast 1 on SINGLE reps. Palindrome pairs.
for ARM in base_b1_a mcp2p_b1_a mcp2p_b1_b base_b1_b; do
  E=${ARM%_b[12]_[ab]}; B=1
  run_arm "$ARM" "$E" "$B" 65
done
run_arm val_mcp2p mcp2p 1 5 1

echo "=== RECEIPTS ==="
"$PY" - "$OUTDIR" <<'EOF'
import glob, json, os, sys
d = sys.argv[1]
EXPECT = sorted(["base_b1_a", "base_b1_b", "mcp2p_b1_a", "mcp2p_b1_b"])
tags = sorted(os.path.basename(f)[:-6]
              for f in glob.glob(os.path.join(d, "*_b1_[ab].jsonl")))
if tags != EXPECT:
    print(f"ARM SET MISMATCH: have {tags} want {EXPECT} — NO VERDICT")
    sys.exit(1)
t = {}
for tag in EXPECT:
    lines = open(os.path.join(d, tag + ".jsonl")).readlines()
    if len(lines) != 1:
        print(f"{tag}: {len(lines)} lines (stale) — NO VERDICT"); sys.exit(1)
    rec = json.loads(lines[0])
    if not rec.get("valid", False):
        print(f"{tag}: invalid — NO VERDICT"); sys.exit(1)
    t.setdefault(tag.rsplit("_", 2)[0], []).append(rec["steady_median_ms"])
for env, v in sorted(t.items()):
    print(f"{env:6s} reps {[round(x, 3) for x in v]}  "
          f"mean {sum(v)/2:6.3f}  spread {abs(v[0]-v[1]):.3f} ms")
delta = sum(t["mcp2p"]) / 2 - sum(t["base"]) / 2
print(f"mcp2p step delta vs base: {delta:+.3f} ms "
      f"(26999539 single-rep read: -0.84)")
EOF
echo "=== ENV VALIDITY (grep NCCL saw the knobs) ==="
for E in mcp2p ll128; do
  n=$(cat "$OUTDIR"/nccl_${E}.* 2>/dev/null | grep -c "NCCL_" || true)
  echo "val_${E}: ${n} NCCL env lines in debug files (0 = arm VOID)"
done
echo "DONE rc=$rc"; exit $rc
