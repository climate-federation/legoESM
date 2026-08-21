#!/bin/bash -l
# DOES SERIALISING THE HALO'S EDGE ROWS LEVEL-FIRST BUY WALL-CLOCK AT 128 GPUs?
#
# The compiled step held 28 transposes, none of them free, converting between
# (level, lat, lon) and (lat, lon, level): the state is stored lat-leading and
# XLA:GPU pipelines level-leading. Twenty-six of the 28 acted on halo edge
# slices of one or two latitude rows, whose shape does not follow the shard,
# and a kernel census of two device traces put 0.617 of the step's 0.707 ms
# per-step FIXED cost in the transpose class.
#
# LEGOESM_LATLON_HALO_LEVEL_LEADING=1 serialises three-dimensional edge
# payloads level-first on the wire and applies the derived inverse on receipt.
# Same widths, same offsets, same dtype grouping, same collective count, same
# payload bytes, and the answer is bit-identical (tests/parallel/
# test_latlon_halo_level_leading.py, which fails on a broken inverse).
#
# ALREADY MEASURED, on the GPU's own post-optimisation HLO (job 27132642, two
# A100s): 28 transposes -> 2, and 4267 instructions -> 4118. The 26 edge
# conversions are gone and the two full-shard carry conversions remain. So the
# compiler did not merely relocate them. What is NOT known is whether that
# converts to wall-clock at the device count where the lane plateaus, which is
# the only question this job asks.
#
# Falsifiable, written before submit:
#   CONFIRM: the flag-on step is at least 5% faster than flag-off at 128 GPUs,
#     with the two-rep spread of each arm below half that difference.
#   REFUTE: |difference| < 2%. Then the 26 removed conversions were not on the
#     critical path, the transpose fixed cost measured in the trace belongs to
#     something else, and the layout reading of the plateau is wrong.
#   INCONCLUSIVE in between, or if the arm spread is not below half the effect.
# A bit-identical change cannot make the model wrong, so there is no accuracy
# gate here; the receipt records the step time only.
#SBATCH --job-name=ll_llead
#SBATCH --account=bd1083_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=32
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:15:00
#SBATCH --exclude=l50003,l50139,l50027,l50009,l50100
#SBATCH --output=ll_llead.%j.log
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
export LEGOESM_XLA_OVERLAP=1
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_shard_autotuning=false"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="$SCRATCH/legoesm_scaling/ll_llead_j${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$OUTDIR/jitcache"; mkdir -p "$LEGOESM_JIT_CACHE_DIR"

NODES="${SLURM_JOB_NUM_NODES:-32}"; GPN=4; NDEV=$((NODES * GPN))
echo "level-leading A/B at ${NDEV} GPUs on ${NODES} nodes"
rc=0

run_arm () {  # level_leading tag
  local LL=$1 TAG=$2
  echo "--- $TAG level_leading=$LL $(date +%T)"
  # Every halo knob is PINNED, not merely left alone: an inherited no-comm
  # or ballast setting would give a wrong-answer or confounded run a
  # performance verdict, and the filename would not show it.
  printf 'LEGOESM_LATLON_HALO_LEVEL_LEADING=%s NOCOMM=0 BALLAST=1\n' "$LL" \
    > "$OUTDIR/${TAG}.env"
  LEGOESM_LATLON_PACKED_EXCHANGE=1 LEGOESM_LATLON_SPMD_FUSED_HALO=1 \
  LEGOESM_LATLON_HALO_NOCOMM=0 LEGOESM_LATLON_HALO_BALLAST=1 \
  LEGOESM_LATLON_HALO_LEVEL_LEADING=$LL JAX_ENABLE_X64=0 \
    timeout --kill-after=30 600 srun --export=ALL --nodes="$NODES" --ntasks="$NDEV" \
      --ntasks-per-node="$GPN" --gpus-per-node="$GPN" --gpu-bind=none \
      --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_atm_latlon_spmd_scaling.py \
      --multicontroller --n-devices "$NDEV" --mode strong \
      --n-lat 2048 --n-lon 4096 --nlev 26 --steps 60 --warmup 10 \
      --out "$OUTDIR/${TAG}.jsonl" || { echo "ARM $TAG FAILED"; rc=1; }
  sleep 10
}

# A discarded warm-up arm, then a palindrome so a drift on the allocation does
# not land on whichever setting ran first.
run_arm 0 warmup_discard
for A in "0 off_a" "1 on_a" "1 on_b" "0 off_b"; do run_arm $A; done

"$PY" - "$OUTDIR" "$NDEV" <<'PYEOF'
import json, os, statistics, sys
outdir, ndev = sys.argv[1], int(sys.argv[2])
problems = []


def load(tag):
    path = os.path.join(outdir, tag + ".jsonl")
    if not os.path.isfile(path):
        problems.append("MISSING: %s.jsonl" % tag)
        return None
    lines = [l for l in open(path).read().splitlines() if l.strip()]
    if not lines:
        problems.append("EMPTY: %s.jsonl" % tag)
        return None
    rec = json.loads(lines[-1])
    knob = rec["metadata"]["extra"].get("halo_knobs", {})
    want = "1" if tag.startswith("on") else "0"
    # A filename is not evidence the knob took. If the benchmark does not
    # record this knob yet, say so rather than assuming either way.
    bind = [("level-leading", knob.get("LEGOESM_LATLON_HALO_LEVEL_LEADING"), want),
            ("packed exchange", knob.get("LEGOESM_LATLON_PACKED_EXCHANGE"), "1"),
            ("fused halo", knob.get("LEGOESM_LATLON_SPMD_FUSED_HALO"), "1"),
            ("no-comm", knob.get("LEGOESM_LATLON_HALO_NOCOMM"), "0"),
            ("ballast", knob.get("LEGOESM_LATLON_HALO_BALLAST"), "1")]
    for name, got_v, want_v in bind:
        if got_v is None:
            problems.append("UNBOUND: %s -- the receipt does not record %s, "
                            "so the arm is not self-evidencing" % (tag, name))
            return None
        if got_v != want_v:
            problems.append("WRONG ARM: %s ran with %s at %r, expected %r"
                            % (tag, name, got_v, want_v))
            return None
    if rec.get("n_devices") != ndev:
        problems.append("WRONG ARM: %s ran on %r devices" % (tag, rec.get("n_devices")))
        return None
    return float(rec["steady_median_ms"])


arms = {k: [load(k + "_a"), load(k + "_b")] for k in ("off", "on")}
med = {}
for k, v in arms.items():
    good = [x for x in v if x is not None]
    if not good:
        problems.append("NO DATA: %s" % k)
        continue
    med[k] = statistics.median(good)
    spread = 100.0 * (max(good) - min(good)) / min(good) if len(good) > 1 else float("nan")
    print("%-4s %8.4f ms   reps %s   spread %.2f%%"
          % (k, med[k], ["%.4f" % x for x in good], spread))

if "off" in med and "on" in med:
    d = 100.0 * (med["on"] - med["off"]) / med["off"]
    spreads = []
    for k, v in arms.items():
        good = [x for x in v if x is not None]
        if len(good) > 1:
            spreads.append(100.0 * (max(good) - min(good)) / min(good))
    print("\nlevel-leading vs default at %d GPUs: %+.2f%%" % (ndev, d))
    if spreads and max(spreads) >= abs(d) / 2.0:
        print("  INCONCLUSIVE: an arm's own spread (%.2f%%) is not below half "
              "the effect. The pre-registered gate refuses this." % max(spreads))
    elif d <= -5.0:
        print("  CONFIRM: at or beyond the pre-registered 5% bar.")
    elif abs(d) < 2.0:
        print("  REFUTE: under the 2% bar; the removed conversions were not "
              "on the critical path.")
    else:
        print("  INCONCLUSIVE: between the 2% and 5% bars.")
if problems:
    print("\nRECEIPT PROBLEMS -- the numbers above are not a verdict:")
    for p in problems:
        print("  " + p)
    sys.exit(3)
PYEOF
SUMRC=$?
if [ "$rc" -ne 0 ] || [ "$SUMRC" -ne 0 ]; then echo "A/B FAILED (arm=$rc summary=$SUMRC)"; exit 1; fi
echo DONE
