#!/bin/bash -l
# TWENTY-SIX LEVELS AGAINST THIRTY-TWO, AT THE DEVICE COUNT WHERE IT MATTERS.
#
# A single-GPU scan on this lane measured the cost per cell per level varying
# THREEFOLD with the level count in no pattern a rule can express, with 26
# among the expensive counts and 32 among the cheap ones: 22.6 ms at 30 levels
# and 8.5 ms at 32 on the same mesh — more work, a third of the time. The
# cause was not established; byte alignment and cache capacity were both
# tested and both refuted. Every scaling arm this campaign has run used 26.
#
# That is a physics decision, not a performance knob, so nothing here changes
# a default. What is missing is a number at the device count the model plateaus
# at, taken with the channel count that lane actually wants, so the decision
# can be made on evidence rather than on a single-GPU table.
#
# THIS IS NOT A LIKE-FOR-LIKE SPEEDUP AND MUST NOT BE REPORTED AS ONE. Thirty-
# two levels is 23% MORE work. The receipt prints the step time AND the cost
# per level, and the honest claim is about work done per unit time, not about
# the same problem running faster.
#
# One job, both level counts, so the two arms share an allocation: a node-set
# difference between them would be a hardware change inside the comparison.
# The halo-off arm runs at each level count too, so the local and
# communication terms are separable at both.
#
# Falsifiable, written before submit:
#   CONFIRMS the single-GPU table at scale: 32 levels is no slower than 26 per
#     STEP despite 23% more work, i.e. cost per level falls by at least 20%.
#   REFUTES it at scale: cost per level within 5% between the two, meaning the
#     single-GPU effect does not survive sharding and 26 is not a trap here.
#   Either way the arm spread must be under half the effect or it is
#     INCONCLUSIVE.
#SBATCH --job-name=mpas_lev
#SBATCH --account=bd1083_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=32
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --exclude=l50003,l50139,l50027,l50009,l50100
#SBATCH --output=mpas_lev.%j.log
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_XLA_OVERLAP=0
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_shard_autotuning=false"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="$SCRATCH/legoesm_scaling/mpas_lev_j${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$OUTDIR/jitcache"; mkdir -p "$LEGOESM_JIT_CACHE_DIR"

NODES="${SLURM_JOB_NUM_NODES:-32}"; GPN=4; NDEV=$((NODES * GPN))
SUBDIV="${SUBDIV:-9}"; CHANNELS="${CHANNELS:-16}"; STEPS="${STEPS:-65}"
LEVELS="${LEVELS:-26 32}"
echo "level A/B at ${NDEV} GPUs, subdivision ${SUBDIV}, ${CHANNELS} channels"
rc=0

run_arm () {  # nlev nocomm tag
  local NL=$1 NC=$2 TAG=$3
  echo "--- $TAG nlev=$NL nocomm=$NC $(date +%T)"
  printf 'NLEV=%s NOCOMM=%s CHANNELS=%s\n' "$NL" "$NC" "$CHANNELS" \
    > "$OUTDIR/${TAG}.env"
  LEGOESM_MPAS_WIDE_HALO=1 LEGOESM_MPAS_RAGGED_HALO=0 \
  LEGOESM_MPAS_HALO_BALLAST=1 LEGOESM_MPAS_HALO_NOSTAGE=0 \
  LEGOESM_MPAS_HALO_NOCOMM=$NC JAX_ENABLE_X64=0 \
  NCCL_MIN_NCHANNELS=$CHANNELS NCCL_MAX_NCHANNELS=$CHANNELS \
  NCCL_P2P_NET_CHUNKSIZE=131072 \
    timeout --kill-after=30 1500 srun --export=ALL --nodes="$NODES" \
      --ntasks="$NDEV" --ntasks-per-node="$GPN" --gpus-per-node="$GPN" \
      --gpu-bind=none --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
      --multicontroller --n-devices "$NDEV" --subdivision "$SUBDIV" \
      --nlev "$NL" --steps "$STEPS" --warmup 5 --lloyd 0 \
      --partition-method sfc --reorder-for "$NDEV" \
      --out "$OUTDIR/${TAG}.jsonl" || { echo "ARM $TAG FAILED"; rc=1; }
  sleep 15
}

set -- $LEVELS
LO=$1; HI=$2
run_arm "$LO" 0 warmup_discard
# Palindrome across the level counts, so a drift on the allocation does not
# land on whichever ran first.
run_arm "$LO" 0 "l${LO}_a"
run_arm "$HI" 0 "l${HI}_a"
run_arm "$HI" 1 "l${HI}nc_a"
run_arm "$LO" 1 "l${LO}nc_a"
run_arm "$LO" 1 "l${LO}nc_b"
run_arm "$HI" 1 "l${HI}nc_b"
run_arm "$HI" 0 "l${HI}_b"
run_arm "$LO" 0 "l${LO}_b"

LEVELS="$LEVELS" "$PY" - "$OUTDIR" "$NDEV" <<'PYEOF'
import json, os, statistics, sys
outdir, ndev = sys.argv[1], int(sys.argv[2])
lo, hi = [int(x) for x in os.environ["LEVELS"].split()]
problems = []


def load(tag, want_nlev, want_nocomm):
    path = os.path.join(outdir, tag + ".jsonl")
    if not os.path.isfile(path):
        problems.append("MISSING: %s.jsonl" % tag)
        return None
    lines = [l for l in open(path).read().splitlines() if l.strip()]
    if not lines:
        problems.append("EMPTY: %s.jsonl" % tag)
        return None
    rec = json.loads(lines[-1])
    knobs = rec["metadata"]["extra"]["halo_knobs"]
    # A filename is not evidence of which arm ran.
    for name, got, want in (("levels", rec.get("nlev"), want_nlev),
                            ("devices", rec.get("n_devices"), ndev),
                            ("platform", rec.get("platform"), "gpu"),
                            ("halo switch",
                             knobs.get("LEGOESM_MPAS_HALO_NOCOMM"),
                             want_nocomm)):
        if got != want:
            problems.append("WRONG ARM: %s ran with %s = %r, expected %r"
                            % (tag, name, got, want))
            return None
    return float(rec["steady_median_ms"])


def med(prefix, nlev, nocomm):
    vals = [v for v in (load("%s_%s" % (prefix, r), nlev, nocomm)
                        for r in ("a", "b")) if v is not None]
    return (statistics.median(vals), vals) if vals else (None, [])


print("subdivision arm at %d GPUs, %d levels against %d\n" % (ndev, lo, hi))
print("%-8s %10s %10s %12s %14s" % ("levels", "step", "halo off", "comm",
                                    "ms per level"))
rows = {}
for nl in (lo, hi):
    full, fv = med("l%d" % nl, nl, "0")
    nc, nv = med("l%dnc" % nl, nl, "1")
    rows[nl] = (full, nc)
    if full is None:
        print("%-8d %10s" % (nl, "NO DATA"))
        continue
    print("%-8d %10.4f %10s %12s %14.5f   reps %s / %s"
          % (nl, full, "%.4f" % nc if nc else "-",
             "%.4f" % (full - nc) if nc else "-", full / nl, fv, nv))

if rows.get(lo, (None,))[0] and rows.get(hi, (None,))[0]:
    f_lo, f_hi = rows[lo][0], rows[hi][0]
    per_lo, per_hi = f_lo / lo, f_hi / hi
    drop = 100.0 * (per_hi - per_lo) / per_lo
    print("\ncost per level: %.5f at %d, %.5f at %d  -> %+.2f%%"
          % (per_lo, lo, per_hi, hi, drop))
    print("step time:      %.4f at %d, %.4f at %d  -> %+.2f%%   "
          "(%d levels is %.0f%% MORE work)"
          % (f_lo, lo, f_hi, hi, 100.0 * (f_hi - f_lo) / f_lo, hi,
             100.0 * (hi - lo) / lo))
    if drop <= -20.0:
        print("  CONFIRMS the single-GPU table at scale.")
    elif abs(drop) < 5.0:
        print("  REFUTES it at scale: cost per level is the same either way.")
    else:
        print("  INCONCLUSIVE: between the bars.")
if problems:
    print("\nRECEIPT PROBLEMS -- not a verdict:")
    for p in problems:
        print("  " + p)
    sys.exit(3)
PYEOF
SUMRC=$?
if [ "$rc" -ne 0 ] || [ "$SUMRC" -ne 0 ]; then echo "LEVEL A/B FAILED"; exit 1; fi
echo DONE
