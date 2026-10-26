#!/bin/bash -l
#SBATCH --job-name=mpas_ragged_flags
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=16
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --exclude=l50003,l50139,l50027,l50009,l50100
#SBATCH --output=mpas_ragged_flags.%j.log
# ===========================================================================
# Can XLA's own ragged-all-to-all runtime modes remove the zero-size-slice
# cost that makes the ragged halo a 1.220x LOSS at s9@64?
#
# Context: the ragged (one-collective-per-entity) halo wins 0.686x @16 and
# 0.735x @32 but inverts to 1.220x @64 (jobs 26822138/26824483/26825520/
# 26824688); the inversion is the unpruned zero-size-slice cost, ~12 us per
# slice, confirmed at fixed degree+payload (26825475). At 64 devices each
# collective carries 64 slices of which ~50 are zero-size. Collapsing the
# 14-round coloured exchange to 2 ragged collectives is the PRECONDITION for
# hiding the 1.815 ms payload term under compute (single-collective overlap
# measured 79% hidden; many-round overlap refuted) — so whether the slice
# cost can be killed decides the whole overlap lever.
#
# This jaxlib ships three relevant XLA runtime modes (strings-verified in
# the installed .so; all default OFF):
#   --xla_gpu_unsupported_enable_ragged_all_to_all_multi_host_decomposer
#       split into intra-host + inter-host parts
#   --xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel
#       single-host one-shot kernel (P2P memory, no per-pair send/recv)
#   (the plain ..._decomposer rewrites to a DENSE all-to-all — that is the
#    max-padded pathology, deliberately NOT on this ladder)
#
# A PRIORI GATE (written before submit):
#   Each arm produces steady step ms; the decision number is
#   ratio = ragged_arm / mean(coloured controls).
#   CONFIRMS (some flagged ragged arm <= 1.00): the slice cost is an XLA
#     runtime artifact its own modes fix -> go straight to the overlap A/B
#     on top of that flag set; no code-level pruning needed.
#   PARTIAL  (best arm in 1.00-1.15): flags shave but do not clear ->
#     code-level pruning (neighbor-subgroup collectives) stays on the list.
#   REFUTES  (all ragged arms >= 1.15): the cost is not these modes ->
#     code-level pruning or park the ragged lever.
#   Offline/CPU cannot answer this: XLA:CPU has no ragged-all-to-all thunk
#   (verified 2026-08-09) and the cost is a GPU runtime per-slice overhead.
#
# Protocol: s9 lloyd-0 prewarmed mesh, sfc, reorder-for 64, f32, nlev 26,
# wide halo ON, 60 timed steps, discarded warm-up arm, coloured controls
# first and last, mcp2p vars unset. COMPARABILITY (codex): the historical
# ragged receipts (0.686/0.735/1.220) were NARROW-halo, 12-step, no
# autotuning-off flag — they are context, NOT a baseline for this job.
# Every claim from this ladder is INTERNAL: ragged arms vs the col_a/col_b
# control mean measured in the same job, same env, same wide-halo config.
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
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/ragged_flags_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
rc=0

NODES=16; GPN=4; NDEV=$((NODES * GPN))
BASE_XLA="--xla_gpu_shard_autotuning=false"
# MH decomposer flag REFUTED 27039649: hangs compile at 8 ranks/2 nodes.
OS="--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=true"

run_arm () {  # tag ragged extra_xla_flags
  local TAG=$1 RAG=$2 EXTRA=${3:-}
  echo "=== s9@${NDEV} ragged=$RAG xla='$EXTRA' [$TAG] ==="
  rm -f "$OUTDIR/${TAG}.jsonl"
  env -u NCCL_MIN_NCHANNELS -u NCCL_MAX_NCHANNELS -u NCCL_P2P_NET_CHUNKSIZE \
      XLA_FLAGS="$BASE_XLA $EXTRA" \
      LEGOESM_MPAS_WIDE_HALO=1 LEGOESM_MPAS_RAGGED_HALO="$RAG" \
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

# GPU PARITY GATE first (cheap, 2 nodes, s6@8 = multi-host), NARROW mode.
# Why narrow (jobs 27039649 + 27040145): the bench parity tolerances are
# the EXACT-exchange floor (u x64 1e-5); the wide path is only
# physically-equivalent (documented outer-ring recompute, band 1e-2 —
# test_mpas_wide_halo), so wide configs fail that gate BY CONTRACT
# (u 2.66e-3 at s6@8, ppermute and ragged IDENTICAL — the ragged
# machinery moves the same rows). Narrow ragged multi-host is the real
# correctness gap and the exact tolerances legitimately apply to it.
# The OS flag is included (it compiled+ran in 27039649 arm c); the MH
# decomposer flag HANGS compile (arms b/d) and is excluded everywhere.
echo "=== parity gate: s6@8 wide=0 ragged=1 OS ==="
env -u NCCL_MIN_NCHANNELS -u NCCL_MAX_NCHANNELS -u NCCL_P2P_NET_CHUNKSIZE \
    XLA_FLAGS="$BASE_XLA $OS" \
    LEGOESM_MPAS_WIDE_HALO=0 LEGOESM_MPAS_RAGGED_HALO=1 JAX_ENABLE_X64=0 \
  timeout 900 srun --export=ALL --nodes=2 --ntasks=8 --ntasks-per-node=4 \
    --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
  "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
    --multicontroller --n-devices 8 --subdivision 6 --nlev 8 \
    --steps 4 --warmup 1 --lloyd 0 --partition-method sfc \
    --halo-strategy ppermute --parity-gate \
    --out "$OUTDIR/parity_s6np8.jsonl" \
  || { echo "NARROW RAGGED PARITY GATE FAILED — timing arms not run"; exit 5; }

# Node-ramp warm-up arm, discarded (campaign lesson: lost 3 runs to ramp).
run_arm warmup_discard 0
run_arm col_a 0
run_arm rag_plain 1
run_arm rag_os 1 "$OS"
run_arm col_b 0

echo "=== RESULTS ==="
set +e
"$PY" - "$OUTDIR" <<'EOF'
import json, math, sys, pathlib
out = pathlib.Path(sys.argv[1])
WANT = {"col_a": "ppermute", "rag_plain": "ppermute_ragged",
        "rag_os": "ppermute_ragged", "col_b": "ppermute"}
t = {}
bad = False
for tag, eff in WANT.items():
    try:
        rows = [json.loads(l) for l in open(out / f"{tag}.jsonl")]
    except Exception as e:
        print(f"{tag}: MISSING -> {e}"); bad = True; continue
    if len(rows) != 1:
        print(f"{tag}: {len(rows)} rows"); bad = True; continue
    r = rows[0]
    ok = (r.get("halo_strategy_effective") == eff
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
if bad or len(t) != len(WANT):
    sys.exit(1)
ctrl = 0.5 * (t["col_a"] + t["col_b"])
spread = abs(t["col_a"] - t["col_b"]) / ctrl
print(f"control mean {ctrl:.3f} ms, A/A2 spread {100*spread:.1f}%")
for tag in ("rag_plain", "rag_os"):
    ratio = t[tag] / ctrl
    verdict = ("INCONCLUSIVE (within control spread)"
               if abs(ratio - 1.0) <= 2 * spread else
               "CONFIRMS" if ratio <= 1.00 else
               "PARTIAL" if ratio < 1.15 else "REFUTES")
    print(f"{tag}: ratio {ratio:.3f} -> {verdict}")
EOF
rcv=$?
[ "$rcv" -ne 0 ] && rc=$rcv
echo "DONE rc=$rc"; exit $rc