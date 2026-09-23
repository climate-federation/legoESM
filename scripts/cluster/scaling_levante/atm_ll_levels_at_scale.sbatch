#!/bin/bash -l
#SBATCH --job-name=ll_lev128
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=32
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --exclude=l50003,l50139,l50027,l50009,l50100
#SBATCH --output=ll_lev128.%j.log
# ===========================================================================
# IS THE LAT-LON STEP LATENCY-BOUND AT 128 GPUs? ASK IT WITH VERTICAL LEVELS.
#
# What is already measured, and is not in question here:
#   - 2048x4096x26. One GPU 201.48 ms. 128 GPUs 3.963 ms. Perfect scaling
#     from one GPU would be 1.574 ms, so the lane is 2.5x off.
#   - With the halo collectives replaced by identity, 128 GPUs still takes
#     2.295-2.429 ms. LOCAL WORK ALONE EXCEEDS THE IDEAL WHOLE STEP, so
#     communication is not this lane's ceiling.
#   - The kernel trace says why, descriptively: the step emits the SAME
#     number of kernels at 32 and at 128 devices (179 vs 181) and their mean
#     duration collapses from 35 us to 13 us, with the idle gap share
#     rising from 2.9% to 5-7%.
#   - At ONE GPU, cost per level is flat (1.79-2.00 ms per level from 16 to
#     40 levels), so on this lane adding levels costs proportionally.
#
# THE QUESTION THIS JOB ANSWERS. If the step is limited by a per-kernel floor
# -- launch overhead plus the occupancy a sixteen-row shard cannot fill --
# rather than by arithmetic, then making each kernel do MORE work should be
# nearly free at a high device count and NOT free at a low one. Vertical
# levels make each kernel do more work.
#
# THE FIRST DRAFT OF THIS ARM WAS CONFOUNDED, AND THE CHECK THAT CAUGHT IT IS
# WHY THIS ONE IS BUILT DIFFERENTLY. GLM's review asked whether the compiled
# program's shape is really invariant to the level count. It is NOT. A free
# compile-only census (job 27159780) counted, on the same lane:
#     nlev 26 -> 4293 instructions, 182 fusions
#     nlev 28 -> 4317 instructions, 183 fusions
#     nlev 40 -> 4135 instructions, 163 fusions
# So 40 levels is not merely more work in the same kernels; it is also
# NINETEEN FEWER kernels. Timing 40 against 28 at one device count therefore
# cannot separate "the extra work was free" from "there were fewer launches".
#
# HOW THIS DESIGN REMOVES THAT CONFOUND. The same two programs -- 28 levels
# and 40 levels -- are timed at THREE device counts inside ONE allocation.
# The shape difference between them is a property of the programs, identical
# at every device count, so it cancels when the ratios are COMPARED ACROSS
# DEVICE COUNTS. What is read is not r_scale itself but how r_scale MOVES:
#     r_scale(d) = t(40 levels, d) / t(28 levels, d)
# A step whose time tracks its arithmetic gives the same r_scale everywhere.
# A step sitting on a per-kernel floor gives r_scale falling towards 1 as the
# shard thins, because the extra arithmetic drops into time the step was
# already spending.
#
# WHY 24 AND 40 AND NOT 26. Both are divisible by EIGHT. On the icosahedral
# lane divisibility by four is worth a factor of three in cost per level, and
# although that cliff was measured ABSENT at one GPU on this lane, at a
# sixteen-row shard the backend's vectorisation along the level axis could
# newly depend on it. Codex's review of the previous draft pointed out that
# 28 and 40 agree modulo four but NOT modulo eight, which leaves a tile-width
# effect free to move the answer without changing any instruction count; 24
# and 40 agree modulo eight and close that. The production count, 26, is not
# timed here; this arm is about the mechanism, not a configuration choice.
#
# THE COMMUNICATION CONTROL, and it is the reason the arm count doubled.
# Codex: a small r_scale could come from extra arithmetic hiding under a
# fixed communication term rather than from a per-kernel floor, and the level
# count changes the halo payload too. So every point is measured TWICE, once
# with the wire and once with the halo collectives replaced by identity. The
# VERDICT is read off the halo-off ratio, which contains no communication at
# all; the full-step ratio is reported beside it.
#
# WHY 8, 64 AND 128 DEVICES. At 8 GPUs the shard is 256 latitude rows and the
# kernels are fat -- that is the reference where time should track work. At
# 64 it is 32 rows, at 128 it is 16. The three points give a trend rather
# than a single contrast, and all three run in the same allocation on the
# same binary. One-device arms are deliberately NOT run: GLM pointed out that
# a 2048x4096x40 single-device array risks an allocator artefact that would
# inflate the reference for a reason having nothing to do with levels.
#
# A PRIORI GATE, written before the run:
#   CONFIRMS a per-kernel floor: r_scale(8) >= 1.30 AND r_scale(128) <= 1.15.
#     Then thin shards leave the step paying for kernels rather than for
#     arithmetic, extra work per device is close to free at 128, and the
#     local ceiling this lane hits is a floor per kernel.
#   REFUTES it: r_scale(128) >= 0.9 * r_scale(8), i.e. the response to work
#     is the same however thin the shard. Then the kernels are not starved
#     and the local excess needs another explanation.
#   Anything else is PARTIAL: report the three ratios, claim no mechanism.
#   A ratio that lands within its own replicate error of a gate boundary
#   returns NO VERDICT rather than a coin flip.
#
# SUBMIT: sbatch scripts/cluster/scaling_levante/atm_ll_levels_at_scale.sbatch
# ===========================================================================
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
OUTDIR="$SCRATCH/legoesm_scaling/ll_lev128_j${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$OUTDIR/jitcache"; mkdir -p "$LEGOESM_JIT_CACHE_DIR"

NODES="${SLURM_JOB_NUM_NODES:-32}"; GPN=4; NDEV=$((NODES * GPN))
NLAT="${NLAT:-2048}"; NLON="${NLON:-4096}"
LEVELS="${LEVELS:-24 40}"       # both divisible by EIGHT; 40/24 = 1.667x work
DEVSETS="${DEVSETS:-8 64 128}"  # 256, 32 and 16 latitude rows per device
# Sixteen channels, because eight was measured to be the worst setting
# available on this lane and an arm run at a starved channel count prices a
# tuning artefact rather than the model.
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-16}"
export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-16}"
echo "levels ${LEVELS} at ${DEVSETS} GPUs, LL${NLAT}x${NLON}"
rc=0

run_arm () {  # ndev nlev nocomm tag
  local ND=$1 NL=$2 NC=$3 TAG=$4
  local NN=$(( (ND + GPN - 1) / GPN ))
  [ "$NN" -gt "$NODES" ] && { echo "SKIP $TAG: needs $NN nodes, have $NODES"; return 0; }
  echo "--- $TAG  ndev=$ND nlev=$NL nocomm=$NC $(date +%T)"
  # Every halo knob is PINNED, not inherited, and the receipt records the
  # channel counts as well: codex pointed out that an inherited NCCL tuning
  # override could move a payload-sensitive ratio invisibly.
  printf 'ndev=%s nlev=%s nocomm=%s nccl_channels=%s..%s byte levers OFF\n' \
    "$ND" "$NL" "$NC" "$NCCL_MIN_NCHANNELS" "$NCCL_MAX_NCHANNELS" \
    > "$OUTDIR/${TAG}.env"
  LEGOESM_LATLON_PACKED_EXCHANGE=1 LEGOESM_LATLON_SPMD_FUSED_HALO=1 \
  LEGOESM_LATLON_HALO_NOCOMM=$NC LEGOESM_LATLON_HALO_BALLAST=1 \
  LEGOESM_LATLON_HALO_LEVEL_LEADING=0 LEGOESM_LATLON_HALO_DEDUP=0 \
  JAX_ENABLE_X64=0 \
    timeout --kill-after=30 900 srun --export=ALL --nodes="$NN" --ntasks="$ND" \
      --ntasks-per-node="$GPN" --gpus-per-node="$GPN" --gpu-bind=none \
      --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_atm_latlon_spmd_scaling.py \
      --multicontroller --n-devices "$ND" --mode strong \
      --n-lat "$NLAT" --n-lon "$NLON" --nlev "$NL" --steps 60 --warmup 10 \
      --out "$OUTDIR/${TAG}.jsonl" || { echo "ARM $TAG FAILED"; rc=1; }
  sleep 10
}

# A discarded warm-up arm first, then every point forwards and backwards, so
# a drift on the allocation does not land on whichever setting ran first.
# (The settling really is this discarded arm: the fused timing lane does not
# execute the --warmup count, as codex noted, so the header must not claim
# it does.)
run_arm "$NDEV" 24 0 warmup_discard
for ND in $DEVSETS; do
  for NC in 0 1; do
    for NL in $LEVELS; do run_arm "$ND" "$NL" "$NC" "d${ND}_l${NL}_c${NC}_a"; done
  done
done
for ND in $(echo $DEVSETS | tr ' ' '\n' | tac | tr '\n' ' '); do
  for NC in 1 0; do
    for NL in $(echo $LEVELS | tr ' ' '\n' | tac | tr '\n' ' '); do
      run_arm "$ND" "$NL" "$NC" "d${ND}_l${NL}_c${NC}_b"
    done
  done
done

echo "=== RECEIPTS ==="
LEVELS="$LEVELS" DEVSETS="$DEVSETS" \
  "$PY" - "$OUTDIR" <<'PYEOF'
import json, os, sys

outdir = sys.argv[1]
levels = [int(x) for x in os.environ["LEVELS"].split()]
devsets = [int(x) for x in os.environ["DEVSETS"].split()]
lo, hi = min(levels), max(levels)
problems = []


def med(tag):
    path = os.path.join(outdir, tag + ".jsonl")
    if not os.path.isfile(path):
        return None
    recs = [json.loads(l) for l in open(path) if l.startswith("{")]
    recs = [r for r in recs if r.get("steady_median_ms")]
    return float(recs[-1]["steady_median_ms"]) if recs else None


rows = {}
for nd in devsets:
    for nc in (0, 1):
        for nl in levels:
            a = med(f"d{nd}_l{nl}_c{nc}_a")
            b = med(f"d{nd}_l{nl}_c{nc}_b")
            if a is None or b is None:
                problems.append(f"d{nd}_l{nl}_c{nc}: incomplete pair")
                continue
            m = 0.5 * (a + b)
            rows[(nd, nc, nl)] = (m, abs(a - b) / m)

print(f"{'GPUs':>6} {'wire':>6} {'nlev':>5} {'ms/step':>10} {'spread':>8}")
for (nd, nc, nl), (m, spread) in sorted(rows.items()):
    print(f"{nd:>6} {'off' if nc else 'on':>6} {nl:>5} {m:10.4f} "
          f"{spread:8.2%}")


def ratio(nd, nc):
    if (nd, nc, lo) not in rows or (nd, nc, hi) not in rows:
        return None
    r = rows[(nd, nc, hi)][0] / rows[(nd, nc, lo)][0]
    # Two replicates give a PEAK-TO-PEAK spread, so the first-order
    # uncertainty of a ratio of two means is the AVERAGE of the two relative
    # spreads, not their sum. The previous draft's void band was twice too
    # wide; codex caught it.
    err = 0.5 * (rows[(nd, nc, hi)][1] + rows[(nd, nc, lo)][1])
    return r, err


print(f"\n{hi} levels over {lo} is {hi / lo:.4f}x the arithmetic. The two "
      f"programs also differ in kernel count, identically at every device "
      f"count, which is why the TREND across device counts is read and not "
      f"any single ratio.")
def fmt(r):
    return f"{r[0]:9.4f} {r[1]:6.2%}" if r else f"{'-':>9} {'-':>6}"


print(f"{'GPUs':>6} {'r local':>9} {'+/-':>6} {'r full':>9} {'+/-':>6}")
for nd in devsets:
    print(f"{nd:>6} {fmt(ratio(nd, 1))} {fmt(ratio(nd, 0))}")

need = [(devsets[0], 1), (devsets[-1], 1)]
if problems or any(ratio(nd, nc) is None for nd, nc in need):
    print("\nNO VERDICT:")
    for p in problems:
        print("  " + p)
    for nd, nc in need:
        if ratio(nd, nc) is None:
            print(f"  no halo-off ratio at {nd} GPUs")
    sys.exit(4)

r_thick, e_thick = ratio(devsets[0], 1)
r_thin, e_thin = ratio(devsets[-1], 1)
# A verdict inside its own error bar is not a verdict.
if (abs(r_thin - 1.15) < e_thin * r_thin
        or abs(r_thick - 1.30) < e_thick * r_thick):
    print("\nVERDICT: NONE -- a ratio sits within its replicate error of a "
          "gate boundary; rerun before reading it either way")
    sys.exit(3)
if r_thick >= 1.30 and r_thin <= 1.15:
    print(f"\nVERDICT: CONFIRMS a per-kernel floor in LOCAL work. With the "
          f"wire deleted, {devsets[0]} GPUs pay {r_thick:.3f}x for "
          f"{hi / lo:.3f}x the arithmetic and {devsets[-1]} GPUs pay "
          f"{r_thin:.3f}x for the same work. No communication is involved in "
          "either number, so thin shards leave the step paying for kernels "
          "rather than for arithmetic.")
elif r_thin >= 0.9 * r_thick:
    print("\nVERDICT: REFUTES -- with the wire deleted the response to extra "
          "work is the same however thin the shard, so the kernels are not "
          "starved and the local excess needs another explanation.")
    sys.exit(6)
else:
    print("\nVERDICT: PARTIAL -- report the ratios, claim no mechanism.")
    sys.exit(3)
PYEOF
exit "$rc"
