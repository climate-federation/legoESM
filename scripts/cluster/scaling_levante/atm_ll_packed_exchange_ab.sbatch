#!/bin/bash -l
#SBATCH --job-name=ll_packed
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=16
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --exclude=l50100
#SBATCH --output=ll_packed.%j.log
# ===========================================================================
# PRODUCTION A/B for the packed stage-entry halo exchange (PR #1582).
#
# Lever: LEGOESM_LATLON_PACKED_EXCHANGE=1 — the three stage-entry
# exchange epochs ([B|ln p] fold h1, wall (T,u,dp[,p_s]) h1, PPM-T fold
# h2) ride ONE ppermute pair per dtype group. CPU-virtual-device census
# receipt: 25 -> 13 collective-permutes/step, step outputs bit-identical
# (max diff 0.0, x64, nd 2/4/8), grad parity 1e-12.
#
# The banked comparison point is the campaign's fused+overlap best,
# which this run reproduces as its own control arm (never compared
# across protocols): LL2048@64 5.278 ms with FUSED_HALO=1 + XLA_OVERLAP=1.
#
# Falsifiability BEFORE submit:
#  (a) number produced: steady_median_ms at LL2048@64, 4 arms.
#  (b) CONFIRM if ON/OFF <= 0.85 (the 12 removed CPs are ~46% of the
#      count at the receipted ~88 us/CP sweep price, i.e. ~1.0 ms of a
#      5.3 ms step — a >=15% cut is the conservative read).
#      REFUTE if >= 0.98 (the count cut does not convert: either the
#      payload/pack cost eats it or the scheduler was already hiding
#      those epochs). Intermediate = real but shallower, reprice.
#      Arm spread > half the effect => INCONCLUSIVE regardless (the
#      ratio is not resolvable against the noise it is read off).
#  (c) no cheaper test: the CP-count-to-wall conversion is a GPU/NCCL
#      property; the CPU census already ran and cannot price it.
#
# Protocol: byte-identical arms except LEGOESM_LATLON_PACKED_EXCHANGE.
# Both the fused-halo and the XLA-overlap envs stay ON in every arm
# (the production lat-lon deck) so the packed lever is the ONLY
# variable.
# ===========================================================================
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
# AB_BACKEND=gpu (default, NCCL) | cpu (gloo, `compute` partition) — the
# packed exchange is a shard_map/ppermute change, so the CPU multicontroller
# lane exercises the SAME code path at rank counts the GPU partition cannot
# reach.
AB_BACKEND="${AB_BACKEND:-gpu}"
if [ "$AB_BACKEND" = "cpu" ]; then
  export JAX_PLATFORMS=cpu
elif [ "$AB_BACKEND" = "gpu" ]; then
  export JAX_PLATFORMS=cuda,cpu
else
  echo "AB_BACKEND=$AB_BACKEND: must be 'gpu' or 'cpu'"; exit 2
fi
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/ll_packed_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
unset XLA_FLAGS
# The production lat-lon deck (held FIXED across arms).
export LEGOESM_LATLON_SPMD_FUSED_HALO=1
export LEGOESM_XLA_OVERLAP=1
rc=0

# Geometry overridable at submit so the SAME script runs the production
# receipt (defaults: LL2048@64 on 16 gpu nodes) or a cheap early-signal
# arm on the usually-idle gpu-devel/vader nodes (2 nodes x 2 A100_40,
# 30-min cap).  A small-scale arm CONFIRMS or REFUTES only at its own
# scale — it never stands in for the production row.
AB_NODES="${AB_NODES:-16}"
AB_GPN="${AB_GPN:-4}"
AB_NLAT="${AB_NLAT:-2048}"
AB_NLON="${AB_NLON:-4096}"
# Timed-sample count. Job 26880368 (vader, LL512@4, steps 12 -> ~9 timed
# samples) put 7.7% spread on two IDENTICAL control arms — the median of
# ~9 samples is not stable enough to resolve a 5% effect. More samples
# per arm shrinks the median's own variance; it costs seconds.
AB_STEPS="${AB_STEPS:-40}"
AB_WARMUP="${AB_WARMUP:-10}"
AB_NDEV=$((AB_NODES * AB_GPN))

run_arm () {  # tag packed_env
  echo "=== LL${AB_NLAT}@${AB_NDEV} f32 arm=$1 PACKED=$2 ==="
  sleep 15
  if [ "$AB_BACKEND" = "cpu" ]; then
    SRUN_DEV=(--cpu-bind=cores --distribution=block:cyclic)
  else
    SRUN_DEV=(--gpus-per-node="$AB_GPN" --gpu-bind=none)
  fi
  LEGOESM_LATLON_PACKED_EXCHANGE="$2" JAX_ENABLE_X64=0 \
    srun --nodes="$AB_NODES" --ntasks="$AB_NDEV" \
      --ntasks-per-node="$AB_GPN" "${SRUN_DEV[@]}" \
      --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_atm_latlon_spmd_scaling.py \
      --multicontroller --n-devices "$AB_NDEV" --mode strong \
      --n-lat "$AB_NLAT" --n-lon "$AB_NLON" --nlev 26 \
      --steps "$AB_STEPS" --warmup "$AB_WARMUP" \
      --out "$OUTDIR/$1.jsonl" || { echo "$1 FAILED"; rc=1; }
}

# ARM ORDER: a discarded warm-up arm first, then INTERLEAVED A/B/A/B.
# Job 26880294 (vader, LL512@4) showed the three arms falling monotonically
# with submission order — 7.083 / 6.669 / 6.577 ms — so the two IDENTICAL
# control arms differed by 7.7% while the effect under test was ~1%: a node
# warm-up ramp, not a lever. Sequential A,B,A2 aliases that ramp straight
# into the ratio. Interleaving cancels a monotone trend to first order; the
# discarded arm absorbs the steepest part of it.
run_arm warmup_discard 0
run_arm off_a 0
run_arm on_b 1
run_arm off_a2 0
run_arm on_b2 1

echo "=== RECEIPTS ==="
set +e
AB_NDEV="$AB_NDEV" AB_BACKEND="$AB_BACKEND" "$PY" - "$OUTDIR" <<'EOF'
import json, math, os, sys, pathlib
out = pathlib.Path(sys.argv[1])
t = {}
for tag in ("off_a", "on_b", "off_a2", "on_b2"):
    try:
        rows = [json.loads(l) for l in open(out / f"{tag}.jsonl")]
    except Exception as e:
        print(f"{tag}: MISSING -> {e}"); sys.exit(1)
    if len(rows) != 1:
        print(f"{tag}: {len(rows)} rows"); sys.exit(1)
    r = rows[0]
    nd = int(os.environ["AB_NDEV"])
    want = ({"cpu"} if os.environ.get("AB_BACKEND") == "cpu"
            else {"gpu", "cuda"})
    ok = (r.get("n_devices") == nd and r.get("platform") in want
          and r.get("n_processes") == nd
          and math.isfinite(r["steady_median_ms"])
          and r["steady_median_ms"] > 0)
    if not ok:
        print(f"{tag}: binding fail nd={r.get('n_devices')} "
              f"platform={r.get('platform')} nproc={r.get('n_processes')}")
        sys.exit(1)
    t[tag] = r["steady_median_ms"]
    print(f"{tag}: {t[tag]:9.3f} ms")
# Two arms per condition, interleaved: compare the MEDIANS of like arms,
# and let the spread WITHIN each condition bound the noise.
off = sorted((t["off_a"], t["off_a2"]))
on = sorted((t["on_b"], t["on_b2"]))
base = sum(off) / 2.0
drift = (off[1] - off[0]) / off[0]
on_spread = (on[1] - on[0]) / on[0]
ratio = (sum(on) / 2.0) / base
print(f"off {off[0]:.3f}/{off[1]:.3f}  on {on[0]:.3f}/{on[1]:.3f}")
print(f"within-off spread {100*drift:.1f}%  within-on spread "
      f"{100*on_spread:.1f}%  |  packed/off ratio {ratio:.3f}")
# The effect must clear the noise it is measured against, not just the
# threshold: a 1% ratio read off 8%-spread arms is not a measurement.
if max(drift, on_spread) > 2.0 * abs(1.0 - ratio):
    print("VERDICT: INCONCLUSIVE (arm spread exceeds half the effect — "
          "the ratio is not resolvable at this noise level)")
    sys.exit(3)
if ratio <= 0.85:
    print("VERDICT: CONFIRMED — the 25->13 CP cut converts to wall time")
elif ratio >= 0.98:
    print("VERDICT: REFUTED — count cut does not convert at this scale")
    sys.exit(2)
else:
    print("VERDICT: intermediate — real but shallower; reprice the bound")
    sys.exit(3)
EOF
rcv=$?
[ "$rcv" -ne 0 ] && rc=$rcv
echo "DONE rc=$rc"; exit $rc
