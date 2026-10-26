#!/bin/bash -l
#SBATCH --job-name=mpas_rag_mcp2p
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=16
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --exclude=l50003,l50139,l50027,l50009,l50100
#SBATCH --output=mpas_rag_mcp2p.%j.log
# ===========================================================================
# Does the mcp2p NCCL pair transfer to the RAGGED exchange (no overlap
# flags)? Coloured+mcp2p = 5.87-5.91 ms is the lane best; ragged plain =
# 6.225 (27041201 off arms). If ragged+mcp2p <= ~5.8 it is the new best.
# Gate: ratio vs ragged-off mean; <=0.95 CONFIRMS transfer, >=0.99 REFUTES
# (the pair was tuned on many-round traffic; 2 fat collectives may not care).
# Superseded header of the template job follows:
# Can XLA latency-hiding OVERLAP the 2-collective ragged halo exchange with
# compute at s9@64 wide?  (Job 27040575: ragged wide = 6.25/6.29 ms vs
# coloured controls 6.70 mean — ratio 0.933, the old 1.220 loss is gone on
# the current stack, no XLA flags needed. 2 collectives/fill is the regime
# where overlap was measured 79% hidden in the microbench; the many-round
# refutation does not apply.)
#
# A PRIORI GATE (written before submit):
#   ratio = overlap_on / mean(overlap_off controls), all arms ragged+wide.
#   CONFIRMS (<= 0.90): overlap hides >= 0.6 ms of the exchange; next step
#     is re-receipting the s9 ladder point with ragged+overlap and folding
#     into the figure.
#   REFUTES  (>= 0.97): latency-hiding does not hide the 2-collective
#     exchange either — the 79% microbench number does not transfer to the
#     full step; drop the overlap lever, keep the plain ragged win.
#   Context anchors: latency-hiding on the COLOURED 14-round exchange was
#   NULL at MPAS (0.0%, 26677669); mcp2p pair won -13..-18% on coloured
#   (26999539/27002564) — its effect on ragged is unmeasured, one arm
#   prices the combination.
# Protocol identical to 27040575 (same mesh, 60 steps, sfc, wide, narrow
# parity gate inherited from that job's green run — not repeated here;
# mcp2p vars unset except the mc arms).
# ===========================================================================
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_XLA_OVERLAP=0
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
export LEGOESM_JIT_CACHE_DIR="$SCRATCH/legoesm_jit_cache_j${SLURM_JOB_ID}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/rag_mcp2p_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
rc=0

NODES=16; GPN=4; NDEV=$((NODES * GPN))
BASE_XLA="--xla_gpu_shard_autotuning=false"
OVL="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_pipelined_p2p=true"
MCP2P=(NCCL_MIN_NCHANNELS=8 NCCL_MAX_NCHANNELS=8 NCCL_P2P_NET_CHUNKSIZE=131072)

run_arm () {  # tag extra_xla extra_env...
  local TAG=$1 EXTRA=$2; shift 2
  echo "=== s9@${NDEV} ragged=1 wide=1 xla='$EXTRA' env='$*' [$TAG] ==="
  rm -f "$OUTDIR/${TAG}.jsonl"
  env -u NCCL_MIN_NCHANNELS -u NCCL_MAX_NCHANNELS -u NCCL_P2P_NET_CHUNKSIZE \
      XLA_FLAGS="$BASE_XLA $EXTRA" \
      LEGOESM_MPAS_WIDE_HALO=1 LEGOESM_MPAS_RAGGED_HALO=1 "$@" \
      JAX_ENABLE_X64=0 \
    timeout 900 srun --export=ALL --nodes="$NODES" --ntasks="$NDEV" \
      --ntasks-per-node="$GPN" --gpus-per-node="$GPN" \
      --gpu-bind=none --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
      --multicontroller --n-devices "$NDEV" \
      --subdivision 9 --nlev 26 \
      --steps 60 --warmup 5 --lloyd 0 --partition-method sfc \
      --halo-strategy ppermute --reorder-for "$NDEV" \
      --out "$OUTDIR/${TAG}.jsonl" || { echo "${TAG} FAILED"; rc=1; }
  sleep 15
}

run_arm warmup_discard ""
run_arm off_a ""
run_arm mc_1 "" "${MCP2P[@]}"
run_arm mc_2 "" "${MCP2P[@]}"
run_arm off_b ""

echo "=== RESULTS ==="
set +e
"$PY" - "$OUTDIR" <<'PYE'
import json, math, sys, pathlib
out = pathlib.Path(sys.argv[1])
TAGS = ["off_a", "mc_1", "mc_2", "off_b"]
t = {}
bad = False
for tag in TAGS:
    try:
        rows = [json.loads(l) for l in open(out / f"{tag}.jsonl")]
    except Exception as e:
        print(f"{tag}: MISSING -> {e}"); bad = True; continue
    if len(rows) != 1:
        print(f"{tag}: {len(rows)} rows"); bad = True; continue
    r = rows[0]
    ok = (r.get("halo_strategy_effective") == "ppermute_ragged"
          and r.get("n_devices") == 64 and r.get("platform") == "gpu"
          and r.get("n_processes") == 64 and r.get("wide_halo") is True
          and math.isfinite(r["steady_median_ms"])
          and r["steady_median_ms"] > 0)
    if not ok:
        print(f"{tag}: binding fail eff={r.get('halo_strategy_effective')} "
              f"ndev={r.get('n_devices')} wide={r.get('wide_halo')}")
        bad = True; continue
    t[tag] = r["steady_median_ms"]
    print(f"{tag}: {t[tag]:9.3f} ms")
if bad or len(t) != len(TAGS):
    sys.exit(1)
ctrl = 0.5 * (t["off_a"] + t["off_b"])
spread = abs(t["off_a"] - t["off_b"]) / ctrl
print(f"off mean {ctrl:.3f} ms, off/off spread {100*spread:.1f}%")
for label, val in (("mcp2p", 0.5*(t["mc_1"]+t["mc_2"])),):
    ratio = val / ctrl
    verdict = ("INCONCLUSIVE (within spread)" if abs(ratio-1.0) <= 2*spread
               else "CONFIRMS" if ratio <= 0.90
               else "REFUTES" if ratio >= 0.97 else "PARTIAL")
    print(f"{label}: {val:.3f} ms, ratio {ratio:.3f} -> {verdict}")
PYE
rcv=$?
[ "$rcv" -ne 0 ] && rc=$rcv
echo "DONE rc=$rc"; exit $rc
