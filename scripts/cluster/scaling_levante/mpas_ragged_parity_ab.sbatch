#!/bin/bash -l
#SBATCH --job-name=mpas_ragged
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:30:00
#SBATCH --output=mpas_ragged.%j.log
# PRODUCTION acceptance for the ragged halo fill
# (LEGOESM_MPAS_RAGGED_HALO=1 -> _ragged_halo_fill: 2 ragged_all_to_all
# per fill instead of max_degree coloured ppermute rounds).
# Receipt so far: synthetic microbench 0.475x per-fill at 16 GPUs
# (job 26818265); schedule proven a row-identical re-grouping of the
# ppermute schedule by tests/parallel/test_ragged_halo_schedule.py.
# Falsifiability, BEFORE submit:
#  ARM 1 (parity, s6@4, 1 node): --parity-gate with env ON must be
#    GREEN and the receipt must record
#    halo_strategy_effective=ppermute_ragged (anti-masquerade — an env
#    typo silently running ppermute would 'pass' parity trivially).
#    A parity FAIL kills the lever (executor bug), regardless of speed.
#  ARM 2/3 (A/B timing, s8@16, 4 nodes, sfc, reorder-for 16): env ON vs
#    OFF, byte-identical otherwise. CONFIRMS if ON <= ~0.9x OFF
#    steady_median (microbench says the halo is a large step fraction at
#    these tiles); REFUTES the production win if ON >= OFF (fill gain
#    swallowed by gather/scatter restructuring or scheduler effects).
#  ARM 4 (A2 drift control): env OFF repeat; |A - A2| bounds node noise.
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_XLA_OVERLAP=0
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/mpas_ragged_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
# Fresh per-job jit/autotune cache: job 26824291 died in BOTH arms on
# "Autotuning failed ... No configuration found in cache after
# synchronizing results across all shards" — the shared persistent
# cache (poisoned by the timed-out 26822138?) desyncs XLA's sharded
# autotuning. Isolate.
export LEGOESM_JIT_CACHE_DIR="$OUTDIR/jitcache"
rc=0

# ARM 1 (parity) BANKED GREEN in job 26821453: u/T/p_s parity OK
# with halo_strategy_effective=ppermute_ragged at s6@4 — not rerun
# (45-min walltime was eaten by per-arm mesh+reorder+compile).
cp -f "$SCRATCH/legoesm_scaling/mpas_ragged_j26821453/parity_on.jsonl" \
  "$OUTDIR/parity_on.jsonl" || { echo "banked parity receipt missing"; rc=1; }

# ARM 2 (ppermute baseline A) BANKED from job 26822138: 7.08 ms
# steady_median at s8@16 (each s8@16 arm costs ~50 min, dominated
# by the host-side reorder — 3 arms do not fit 90 min).
cp -f "$SCRATCH/legoesm_scaling/mpas_ragged_j26822138/timing_off_a.jsonl" \
  "$OUTDIR/timing_off_a.jsonl" || { echo "banked A receipt missing"; rc=1; }

echo "=== ARM 3: timing, ragged ON (B), s8@16 ==="
LEGOESM_MPAS_RAGGED_HALO=1 JAX_ENABLE_X64=0 \
  srun --nodes=4 --ntasks=16 --ntasks-per-node=4 --gpus-per-node=4 \
    --gpu-bind=none --kill-on-bad-exit=1 \
  "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
    --multicontroller --n-devices 16 --subdivision 8 --nlev 26 \
    --steps 12 --warmup 3 --partition-method sfc --reorder-for 16 \
    --out "$OUTDIR/timing_on_b.jsonl" || { echo "B FAILED"; rc=1; }

echo "=== ARM 4: timing, ragged OFF (A2, drift control) ==="
LEGOESM_MPAS_RAGGED_HALO=0 JAX_ENABLE_X64=0 \
  srun --nodes=4 --ntasks=16 --ntasks-per-node=4 --gpus-per-node=4 \
    --gpu-bind=none --kill-on-bad-exit=1 \
  "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
    --multicontroller --n-devices 16 --subdivision 8 --nlev 26 \
    --steps 12 --warmup 3 --partition-method sfc --reorder-for 16 \
    --out "$OUTDIR/timing_off_a2.jsonl" || { echo "A2 FAILED"; rc=1; }

echo "=== RECEIPTS ==="
set +e
"$PY" - "$OUTDIR" <<'EOF'
import json, math, sys, pathlib
out = pathlib.Path(sys.argv[1])
WANT = {"parity_on": ("ppermute_ragged", 4),
        "timing_off_a": ("ppermute", 16),
        "timing_on_b": ("ppermute_ragged", 16),
        "timing_off_a2": ("ppermute", 16)}
t, cp = {}, {}
for tag, (eff, nd) in WANT.items():
    try:
        rows = [json.loads(l) for l in open(out / f"{tag}.jsonl")]
    except Exception as e:
        print(f"{tag}: MISSING -> {e}"); sys.exit(1)
    if len(rows) != 1:
        print(f"{tag}: {len(rows)} rows"); sys.exit(1)
    r = rows[0]
    ok = (r.get("halo_strategy_effective") == eff
          and r.get("n_devices") == nd
          and r.get("platform") == "gpu"
          and r.get("n_processes") == nd
          and math.isfinite(r["steady_median_ms"])
          and r["steady_median_ms"] > 0)
    if not ok:
        print(f"{tag}: binding fail: eff={r.get('halo_strategy_effective')} "
              f"nd={r.get('n_devices')} platform={r.get('platform')} "
              f"nproc={r.get('n_processes')}"); sys.exit(1)
    t[tag] = r["steady_median_ms"]
    cp[tag] = r.get("hlo_collective_permutes")
    print(f"{tag}: {t[tag]:9.2f} ms  (effective={eff}, CPs={cp[tag]})")
# Independent HLO evidence beyond the closure label (codex r1 MINOR):
# the ragged arm's compiled module must hold FEWER collective-permutes
# than the ppermute arm (ragged lowers to a custom call, not a CP).
if cp["timing_on_b"] is not None and cp["timing_off_a"] is not None:
    if not cp["timing_on_b"] < cp["timing_off_a"]:
        print(f"HLO census does not show the CP cut: on={cp['timing_on_b']} "
              f"off={cp['timing_off_a']} — env label unsupported"); sys.exit(1)
drift = abs(t["timing_off_a"] - t["timing_off_a2"]) / t["timing_off_a"]
ratio = t["timing_on_b"] / min(t["timing_off_a"], t["timing_off_a2"])
print(f"A/A2 drift {100*drift:.1f}%  |  ragged/off ratio {ratio:.3f}")
if ratio <= 0.9:
    print("VERDICT: CONFIRMED (ragged <= 0.9x)")
elif ratio >= 1.0:
    print("VERDICT: REFUTED (ragged >= off)"); sys.exit(2)
else:
    print("VERDICT: INCONCLUSIVE (0.9 < ratio < 1.0)"); sys.exit(3)
EOF
rcv=$?
[ "$rcv" -ne 0 ] && rc=$rcv
echo "DONE rc=$rc"; exit $rc
