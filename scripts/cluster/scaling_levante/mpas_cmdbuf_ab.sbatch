#!/bin/bash -l
#SBATCH --job-name=mpas_cmdbuf
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=mpas_cmdbuf.%j.log
# XLA command-buffer (CUDA-graph) A/B on the MPAS GPU lane.
#
# WHY: nsys receipt (PR #1543, cube C768@24 closed loop, analogous regime)
# = 81% of wall is inter-kernel gaps, ~155us launch/sync floor per
# sequential collective; MPAS s9@64 census = 33 collective-permute epochs.
# The scaling benches BYPASS configure_backend, so every launch-floor
# receipt so far ran WITHOUT --xla_gpu_enable_command_buffer=...COLLECTIVES
# (production backend.py sets it; benches never did). GLM-5.2 Rank-1 lever
# (ICON-GPU precedent: CUDA-graph capture of the dynamics loop, 1.5-2x).
#
# Falsifiability, BEFORE submit (compute-discipline gate):
#  (a) numbers produced: steady_median_ms for 4 arms at s7@4 (1 node).
#  (b) CONFIRM if min(B,C)/min(A,A2) <= 0.85 (command buffer swallows a
#      large share of the per-collective launch floor);
#      REFUTE if >= 0.95 (collectives not captured, or floor not
#      launch-bound at this count). A/A2 drift bounds node noise.
#  (c) cannot be answered offline: the launch floor is a GPU runtime
#      property; no checkpoint/CPU test exposes it. 1 node x ~1 h.
#
# Arm binding: capture evidence is the one-off ~100 ms CUDA-graph
# instantiation on the FIRST post-compile step (in the jsonl per_step_ms),
# which discriminates "captured but no win" from "silently not captured".
# First run (job 26873637, vader s6@4): XLA-defaults 2.81/2.80 ms vs
# +COLLECTIVES 3.52 / +min_graph_size 3.48 ms — +25%, capture engaged.
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_XLA_OVERLAP=0
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/mpas_cmdbuf_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
# (This bench never arms the persistent jit cache; all arms cold-compile.)
# Control arms must NOT inherit a submit-shell XLA_FLAGS (codex MAJOR).
unset XLA_FLAGS
rc=0

# Overridable geometry so the same script runs on gpu (1 node x 4 GPU)
# or gpu-devel/vader (2 nodes x 2 GPU, 30-min cap -> s6, cached mesh).
AB_NODES="${AB_NODES:-1}"
AB_GPN="${AB_GPN:-4}"
AB_SUBDIV="${AB_SUBDIV:-7}"
AB_NDEV=$((AB_NODES * AB_GPN))
BENCH_ARGS="--multicontroller --n-devices $AB_NDEV --subdivision $AB_SUBDIV \
  --nlev 26 --steps 12 --warmup 3 --partition-method sfc \
  --reorder-for $AB_NDEV"
RUN="srun --nodes=$AB_NODES --ntasks=$AB_NDEV --ntasks-per-node=$AB_GPN \
  --gpus-per-node=$AB_GPN --gpu-bind=none --kill-on-bad-exit=1"

# Arm flag sets overridable at submit; defaults = the PRODUCTION isolation
# pair (codex P1 on job 26873637: XLA-defaults vs full-set did not isolate
# the backend.py change).  B = OLD production default, C = NEW production
# default; A/A2 = no flags (the bench-receipt baseline).
CB_ARM_B_FLAGS="${CB_ARM_B_FLAGS:---xla_gpu_enable_command_buffer=FUSION,CUSTOM_CALL,COLLECTIVES}"
CB_ARM_C_FLAGS="${CB_ARM_C_FLAGS:---xla_gpu_enable_command_buffer=FUSION,CUSTOM_CALL}"

echo "=== ARM A: baseline (no XLA_FLAGS, matches all prior receipts) ==="
JAX_ENABLE_X64=0 $RUN "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
  $BENCH_ARGS --out "$OUTDIR/cb_off_a.jsonl" || { echo "A FAILED"; rc=1; }

sleep 15
echo "=== ARM B: $CB_ARM_B_FLAGS ==="
XLA_FLAGS="$CB_ARM_B_FLAGS" \
JAX_ENABLE_X64=0 $RUN "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
  $BENCH_ARGS --out "$OUTDIR/cb_on_b.jsonl" || { echo "B FAILED"; rc=1; }

# Back-to-back srun steps can race slurm's memory accounting
# ("Memory required by task is not available", job 26874337 arm C) —
# let the previous step's allocation settle.
sleep 15
echo "=== ARM C: $CB_ARM_C_FLAGS ==="
XLA_FLAGS="$CB_ARM_C_FLAGS" \
JAX_ENABLE_X64=0 $RUN "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
  $BENCH_ARGS --out "$OUTDIR/cb_on_c.jsonl" || { echo "C FAILED"; rc=1; }

sleep 15
echo "=== ARM A2: baseline repeat (drift control) ==="
JAX_ENABLE_X64=0 $RUN "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
  $BENCH_ARGS --out "$OUTDIR/cb_off_a2.jsonl" || { echo "A2 FAILED"; rc=1; }

echo "=== RECEIPTS ==="
set +e
AB_NDEV="$AB_NDEV" "$PY" - "$OUTDIR" <<'EOF'
import json, math, os, sys, pathlib
out = pathlib.Path(sys.argv[1])
t, captured = {}, {}
for tag in ("cb_off_a", "cb_on_b", "cb_on_c", "cb_off_a2"):
    try:
        rows = [json.loads(l) for l in open(out / f"{tag}.jsonl")]
    except Exception as e:
        print(f"{tag}: MISSING -> {e}"); sys.exit(1)
    if len(rows) != 1:
        print(f"{tag}: {len(rows)} rows"); sys.exit(1)
    r = rows[0]
    # JAX 0.10 CUDA plugin backend name is "cuda" (codex BLOCKER).
    nd = int(os.environ["AB_NDEV"])
    ok = (r.get("n_devices") == nd and r.get("platform") in ("gpu", "cuda")
          and r.get("n_processes") == nd
          and math.isfinite(r["steady_median_ms"])
          and r["steady_median_ms"] > 0)
    if not ok:
        print(f"{tag}: binding fail: nd={r.get('n_devices')} "
              f"platform={r.get('platform')} nproc={r.get('n_processes')}")
        sys.exit(1)
    t[tag] = r["steady_median_ms"]
    # Capture evidence, in-band (codex P2: vmodule line counts prove the
    # scheduling pass logged, not that a graph was built): CUDA-graph
    # instantiation shows as a one-off ~100 ms FIRST post-compile step
    # (job 26873637: 108.6/110.0 ms captured vs 3.2/3.3 ms uncaptured).
    steps = r.get("per_step_ms") or []
    inst = (len(steps) >= 3 and steps[1] > 10 * max(steps[2], 1e-9))
    captured[tag] = inst
    print(f"{tag}: {t[tag]:9.2f} ms  (CPs={r.get('hlo_collective_permutes')}, "
          f"graph_instantiation_step1={inst})")
base = min(t["cb_off_a"], t["cb_off_a2"])
drift = abs(t["cb_off_a"] - t["cb_off_a2"]) / base
rb, rc_ = t["cb_on_b"] / base, t["cb_on_c"] / base
print(f"A/A2 drift {100*drift:.1f}%  |  B/off {rb:.3f}  C/off {rc_:.3f}")
if drift > 0.05:
    print("VERDICT: INCONCLUSIVE (A/A2 drift > 5% — node noise unbounded)")
    sys.exit(3)
ratio = min(rb, rc_)
if ratio <= 0.85:
    print("VERDICT: CONFIRMED (command buffer <= 0.85x at this config)")
elif ratio >= 0.95:
    if not (captured["cb_on_b"] or captured["cb_on_c"]):
        # No instantiation signature: flags did not engage — NOT a
        # refutation of the lever, a failed capture.
        print("VERDICT: NOT-CAPTURED (>= 0.95x AND no step-1 graph "
              "instantiation signature in either flagged arm)")
        sys.exit(4)
    print("VERDICT: REFUTED-at-this-config (>= 0.95x with capture "
          "evidence; does not bind other scales/fabrics)")
    sys.exit(2)
else:
    print("VERDICT: INCONCLUSIVE (0.85 < ratio < 0.95)"); sys.exit(3)
EOF
rcv=$?
[ "$rcv" -ne 0 ] && rc=$rcv
echo "DONE rc=$rc"; exit $rc
