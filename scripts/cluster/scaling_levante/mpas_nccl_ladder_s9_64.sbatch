#!/bin/bash -l
#SBATCH --job-name=nccl_ladder
#SBATCH --account=bd1083_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=16
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:45:00
#SBATCH --output=nccl_ladder.%j.log
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
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/nccl_ladder_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$SCRATCH/legoesm_scaling/_jitcache_shared"
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
for ARM in base_b1_a mcp2p_b1_a ll128_b1_a base_b2_a mcp2p_b2_a ll128_b2_a \
           ll128_b2_b mcp2p_b2_b base_b2_b ll128_b1_b mcp2p_b1_b base_b1_b; do
  E=${ARM%_b[12]_[ab]}; B=${ARM#${E}_b}; B=${B%_[ab]}
  run_arm "$ARM" "$E" "$B" 65
done
# Validity arms (5 steps, INFO logs to files; not timed).
for E in base mcp2p ll128; do
  run_arm "val_${E}" "$E" 1 5 1
done

echo "=== RECEIPTS ==="
"$PY" - "$OUTDIR" <<'EOF'
import glob, json, os, sys
d = sys.argv[1]
EXPECT = sorted(f"{e}_b{b}_{r}" for e in ("base", "mcp2p", "ll128")
                for b in (1, 2) for r in ("a", "b"))
tags = sorted(os.path.basename(f)[:-6]
              for f in glob.glob(os.path.join(d, "*_b[12]_[ab].jsonl")))
if tags != EXPECT:
    print(f"ARM SET MISMATCH: have {tags} want {EXPECT} — NO VERDICT")
    sys.exit(1)
t = {}
for tag in EXPECT:
    lines = open(os.path.join(d, tag + ".jsonl")).readlines()
    if len(lines) != 1:
        print(f"{tag}: {len(lines)} receipt lines (stale/duplicate) — NO VERDICT")
        sys.exit(1)
    rec = json.loads(lines[0])
    if not rec.get("valid", False):
        print(f"{tag}: bench receipt invalid — NO VERDICT"); sys.exit(1)
    env, bal, rep = tag.rsplit("_", 2)
    t.setdefault(env, {}).setdefault("b" + bal[1], []).append(
        rec["steady_median_ms"])
rows = {}
for env, d2 in sorted(t.items()):
    ic = [2 * a - b for a, b in zip(d2["b1"], d2["b2"])]
    sl = [b - a for a, b in zip(d2["b1"], d2["b2"])]
    rows[env] = (sum(ic) / 2, abs(ic[0] - ic[1]), sum(sl) / 2)
    print(f"{env:6s} b1 {sum(d2['b1'])/2:6.3f}  b2 {sum(d2['b2'])/2:6.3f}  "
          f"intercept {rows[env][0]:6.3f} (rep spread {rows[env][1]:.3f} ms)  "
          f"slope {rows[env][2]:6.3f}")
# Mechanical band classification per the PRE-REGISTERED gate in the header.
# Bands only; interpretation stays in the analysis, not the tool.
b0, bspread, _ = rows["base"]
for env in ("mcp2p", "ll128"):
    di = rows[env][0] - b0
    ds = rows[env][2] - rows["base"][2]
    noise = max(bspread, rows[env][1])
    if noise >= 0.5 * abs(di) and abs(di) >= 0.15:
        band = "NOISE-BOUND (spread >= half the delta)"
    elif di <= -0.30:
        band = "CONFIRM band"
    elif di <= -0.15:
        band = "INCONCLUSIVE band"
    else:
        band = "REFUTE band (this lever)"
    print(f"{env}: intercept delta {di:+.3f} ms [{band}]  "
          f"slope delta {ds:+.3f} ms (never counts toward CONFIRM)")
EOF
echo "=== ENV VALIDITY (grep NCCL saw the knobs) ==="
for E in mcp2p ll128; do
  n=$(cat "$OUTDIR"/nccl_${E}.* 2>/dev/null | grep -c "NCCL_" || true)
  echo "val_${E}: ${n} NCCL env lines in debug files (0 = arm VOID)"
done
echo "DONE rc=$rc"; exit $rc
