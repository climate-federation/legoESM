#!/bin/bash -l
#SBATCH --job-name=ll_budget
#SBATCH --account=bd1083_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=32
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:30:00
#SBATCH --exclude=l50003,l50139,l50027,l50009,l50100
#SBATCH --output=ll_budget.%j.log
# ===========================================================================
# HOW MUCH OF THE LAT-LON STEP IS COMMUNICATION, AT 32 / 64 / 128 GPUs?
#
# The MPAS lane has a fully split step budget (compute / staging / wire /
# per-op residual).  The lat-lon lane has NONE: its 2^n ladder shows the
# curve bending away from ideal above 32 GPUs (LL2048: 90 % parallel
# efficiency at 32, 66 % at 64, 40 % at 128) and nothing says whether the
# lost time is communication or per-device compute becoming inefficient as
# the band thins.  Those two diagnoses point at opposite levers, so the
# split has to be measured before any more lat-lon work is priced.
#
# Instrument: LEGOESM_LATLON_HALO_NOCOMM=1 replaces every lat-band halo
# collective with an optimization barrier.  The pack, the split, the pole
# fold, the wall constants, the shapes and the kernel count are unchanged;
# only the collective is gone.  It BREAKS THE ANSWER (each band keeps its
# own edge rows as ghosts) and exists only because the profiler on this
# stack does not record the halo collectives at all (receipt 2026-08-13).
#
#   full - nocomm = HALO wire time + waiting for the slowest peer
#                   + any local work that differs because the two arms are
#                     no longer integrating the same fields
#
# THAT THIRD TERM IS REAL, AND IT CANNOT BE BOUNDED BY WATCHING THE NO-COMM
# ARM WOBBLE.  The no-comm arm keeps each band's own edge rows as ghosts, so
# from the second step on the two arms carry different states; if the wrong
# fields make a limiter or a solver take a CONSTANT amount less work, the
# contamination is steady and every drift check reads clean.
#
# So this job runs a third arm, LEGOESM_LATLON_HALO_NOCOMM=wire, which
# performs the real exchange and then keeps the local rows anyway.  It
# answers bit-for-bit as the no-comm arm does -- same wrong trajectory, same
# local work -- while paying the full communication cost:
#
#   wire - nocomm = HALO wire time + waiting for the slowest peer, with the
#                   trajectory identical on both sides by construction
#
# That is the number the verdict is taken on.  ``full - nocomm`` is still
# printed as the upper bound it always was, and the gap between the two IS
# the trajectory contamination, measured rather than argued.  The no-comm
# arm's block-to-block drift is printed alongside, as a weak sanity check
# only -- it can be small while the contamination is large.
#
# SCOPE, verified by HLO census on 4 CPU devices (LL32x64 L8, same flags):
# the knob removes ALL 52 collective-permutes from the compiled step and
# leaves the 4 all-reduces untouched.  Those all-reduces are the mass
# fixer's global sums, which run in BOTH arms -- so this receipt prices the
# halo exchange only, and any mass-fixer reduction cost sits inside the
# nocomm arm, not in the difference.
#
# The remaining nocomm time is per-device compute + on-device halo staging
# + the mass-fixer reductions; comparing it across device counts at FIXED
# global size shows directly whether the local work stops shrinking.
#
# A PRIORI GATE (written before submit), on LL2048 at 128 GPUs.  Measured
# ladder points: 1 GPU 201.48 ms, so ideal at 128 is 1.574 ms and the
# measured 3.956 ms carries 2.38 ms of excess over ideal.
#   CONFIRMS comm-bound: (full - nocomm) >= 1.2 ms, i.e. communication is
#     at least half the excess -> comm levers (fewer/bigger messages,
#     2-D decomposition to cut the perimeter) are the right target.
#   REFUTES comm-bound: (full - nocomm) <= 0.4 ms -> the excess is local
#     work that stopped shrinking (thin bands, launch-bound kernels), and
#     every comm lever on the roadmap is aimed at the wrong term.
#   Intermediate 0.4-1.2 ms: both terms matter; report both, pick by size.
# Cheaper offline test: none.  The HLO census gives collective COUNT and
# BYTES, not time, and the profiler does not record these collectives.
#
# Protocol matches the 2^n ladder byte for byte except the knob: LL2048
# f32, nlev 26, packed stage-entry exchange + fused SPMD halo + XLA
# overlap + the NCCL multi-channel p2p pair, two fused blocks of 60 steps
# each after the compile call and the latency probe, a whole discarded
# warm-up arm, then INTERLEAVED arms (12-step arms gave 7.7 % spread on
# identical controls in this campaign; 60-step arms gave 0.3 %).  The
# fused lane has no per-step warm-up window -- it separates the compile,
# the probe and the timed blocks instead -- so nothing here relies on one.
#
# SUBMIT: sbatch scripts/cluster/scaling_levante/atm_ll_budget_split.sbatch
# ===========================================================================
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_XLA_OVERLAP=1
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_shard_autotuning=false --xla_gpu_enable_nccl_comm_splitting=false"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/ll_budget_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$SCRATCH/legoesm_jit_cache_j${SLURM_JOB_ID}"
mkdir -p "$LEGOESM_JIT_CACHE_DIR"
export NCCL_MIN_NCHANNELS=8 NCCL_MAX_NCHANNELS=8 NCCL_P2P_NET_CHUNKSIZE=131072
rc=0

NLAT="${NLAT:-2048}"; NLON=$((2 * NLAT)); GPN=4
# The ideal line and every gate threshold below are the LL2048 single-device
# receipt. Another resolution needs its own baseline, so refuse rather than
# score a different problem against this one.
if [ "$NLAT" -ne 2048 ]; then
  echo "NLAT=$NLAT: this job's ideal baseline and gate thresholds are the"
  echo "LL2048 single-device receipt (201.48 ms, job 27051113). Supply a"
  echo "matching baseline before running another resolution."
  exit 6
fi
# Device counts to split, largest first. Overridable so the ladder can be
# cut into smaller allocations that clear the queue sooner (a 32-node job
# waits behind everything; two 8/16-node jobs do not).
DEVS="${DEVS:-128 64 32}"
FIRST_DEV="${DEVS%% *}"

# Deterministic node subset per device count: an arm that lands on a
# different set of nodes than its pair puts a hardware change inside the
# difference being measured. The first N hosts of the allocation, always.
ALLOC_HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort)
pin_hosts () {  # nodes -> comma-separated first N allocation hosts
  echo "$ALLOC_HOSTS" | head -n "$1" | paste -sd,
}

run_arm () {  # ndev nocomm tag [ballast]
  local NDEV=$1 NC=$2 TAG=$3 BAL=${4:-1}
  local NODES=$(( (NDEV + GPN - 1) / GPN ))
  local NTPN=$(( NDEV < GPN ? NDEV : GPN ))
  local HOSTS
  HOSTS=$(pin_hosts "$NODES")
  echo "=== LL${NLAT}@${NDEV} nocomm=$NC ballast=$BAL [$TAG] "\
       "$(date +%T) hosts=$HOSTS ==="
  LEGOESM_LATLON_PACKED_EXCHANGE=1 LEGOESM_LATLON_SPMD_FUSED_HALO=1 \
  LEGOESM_LATLON_HALO_NOCOMM=$NC LEGOESM_LATLON_HALO_BALLAST=$BAL \
  JAX_ENABLE_X64=0 \
    timeout 1500 srun --export=ALL --nodelist="$HOSTS" \
      --nodes="$NODES" --ntasks="$NDEV" \
      --ntasks-per-node="$NTPN" --gpus-per-node="$NTPN" \
      --gpu-bind=none --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_atm_latlon_spmd_scaling.py \
      --multicontroller --n-devices "$NDEV" --mode strong \
      --n-lat "$NLAT" --n-lon "$NLON" --nlev 26 \
      --steps 60 \
      --out "$OUTDIR/${TAG}.jsonl" || { echo "${TAG} FAILED"; rc=1; }
  sleep 15
}

# Discarded warm-up arm: the node ramp must not alias into the first timed
# arm (this campaign lost three runs to exactly that).
run_arm "$FIRST_DEV" 0 warmup_discard
# BALLAST_ARMS=1 adds a doubled-payload pair per device count. Bytes on
# the wire double; the collective count, the schedule, the arithmetic and
# the answer do not. That separates the two ways communication can cost:
# payload, which a smaller halo would cut, and per-operation cost, which it
# would not. The lat-band decomposition gives every device the SAME halo
# bytes at every device count, yet the measured communication time doubled
# from 32 to 64 devices -- so this is the question that decides whether a
# two-dimensional decomposition is worth building.
BALLAST_ARMS="${BALLAST_ARMS:-0}"
for ND in $DEVS; do
  run_arm "$ND" 0 "full_${ND}_a"
  run_arm "$ND" 1 "nc_${ND}_a"
  run_arm "$ND" wire "wire_${ND}_a"
  run_arm "$ND" 0 "full_${ND}_b"
  run_arm "$ND" 1 "nc_${ND}_b"
  run_arm "$ND" wire "wire_${ND}_b"
  if [ "$BALLAST_ARMS" = "1" ]; then
    run_arm "$ND" 0 "b2_${ND}_a" 2
    run_arm "$ND" 0 "b2_${ND}_b" 2
    # Packing-only control: the doubled payload is BUILT and then not sent.
    # Without it the payload delta contains the device-side concatenate the
    # multiplier adds, and that is not a small term -- the raw delta came
    # out larger than the entire communication term, which can only happen
    # if packing is being counted as wire time.
    run_arm "$ND" 1 "b2nc_${ND}_a" 2
    run_arm "$ND" 1 "b2nc_${ND}_b" 2
  fi
done

echo "=== RECEIPTS ==="
# An arm that failed leaves a stale or missing receipt; scoring on top of
# that reports a number from a run that did not happen.
if [ "$rc" -ne 0 ]; then
  echo "ABORT: an arm failed (rc=$rc); no verdict"
  exit "$rc"
fi
BALLAST_ARMS="$BALLAST_ARMS" "$PY" - "$OUTDIR" "$DEVS" <<'EOF'
import json, os, sys

outdir = sys.argv[1]
DEVS = tuple(sorted(int(d) for d in sys.argv[2].split()))
BALLAST = os.environ.get("BALLAST_ARMS", "0") == "1"
# Single-GPU LL2048 ladder point (job 27051113), same build and flags.
T1 = 201.4807


def _record(tag):
    path = os.path.join(outdir, tag + ".jsonl")
    lines = [l for l in open(path).read().strip().split("\n")
             if l.startswith("{")]
    return json.loads(lines[-1])


def med(tag):
    return _record(tag)["steady_median_ms"]


def block_drift_ms(tag):
    """How much this arm's per-step time moved between its timed blocks.

    The no-comm arm answers wrongly by construction, so its fields drift away
    from the full arm's as the run proceeds and its local work can drift with
    them.  A receipt already carries the per-block wall times of the slowest
    process, so the size of that drift is free to read: if it is small next to
    the full-minus-nocomm difference, the difference is not being paid for by
    the diverging trajectory.  Returns None when the lane recorded no blocks.
    """
    rec = _record(tag)
    blocks = rec.get("parallel_block_ms") or rec.get("block_ms")
    steps = rec.get("extra", {}).get("steps")
    if not blocks or len(blocks) < 2 or not steps:
        return None
    per_step = [b / steps for b in blocks]
    return max(per_step) - min(per_step)


rows = {}
missing = []
for nd in DEVS:
    for kind in (("full", "nc", "wire", "b2", "b2nc") if BALLAST
                 else ("full", "nc", "wire")):
        for rep in ("a", "b"):
            tag = f"{kind}_{nd}_{rep}"
            try:
                rows[tag] = med(tag)
            except Exception as exc:
                missing.append(f"{tag} ({exc})")
if missing:
    # A gate that reports a number from a partial run is not a gate.
    print(f"ABORT: arms missing {missing}; no verdict")
    sys.exit(4)

print(f"{'ndev':>5} {'full':>8} {'nocomm':>8} {'wire':>8} {'comm':>8} "
      f"{'upper':>8} {'ideal':>8} {'excess':>8} {'comm/excess':>12} "
      f"{'spread':>8}")
print("  comm  = wire - nocomm   (same trajectory on both sides)")
print("  upper = full - nocomm   (also carries the trajectory difference)")
verdict_row = None
for nd in DEVS:
    full = (rows[f"full_{nd}_a"] + rows[f"full_{nd}_b"]) / 2
    nc = (rows[f"nc_{nd}_a"] + rows[f"nc_{nd}_b"]) / 2
    wire = (rows[f"wire_{nd}_a"] + rows[f"wire_{nd}_b"]) / 2
    spread = max(abs(rows[f"full_{nd}_a"] - rows[f"full_{nd}_b"]) / full,
                 abs(rows[f"nc_{nd}_a"] - rows[f"nc_{nd}_b"]) / nc,
                 abs(rows[f"wire_{nd}_a"] - rows[f"wire_{nd}_b"]) / wire)
    # The wire arm answers exactly as the no-comm arm does, so this
    # difference is the collective and nothing else.
    comm = wire - nc
    # What the old two-arm subtraction gave: the same term PLUS whatever the
    # two different trajectories cost differently.
    upper = full - nc
    ideal = T1 / nd
    excess = full - ideal
    frac = comm / excess if excess > 0 else float("nan")
    print(f"{nd:>5} {full:8.3f} {nc:8.3f} {wire:8.3f} {comm:8.3f} "
          f"{upper:8.3f} {ideal:8.3f} {excess:8.3f} {frac:11.2f} "
          f"{spread:7.1%}")
    print(f"      ndev={nd}: the trajectory difference between the full and "
          f"no-comm arms is worth {upper - comm:+.3f} ms, which the old "
          f"two-arm subtraction counted as communication")
    # A difference smaller than half the arm spread is not a result.
    if abs(comm) < 0.5 * spread * full:
        print(f"      ndev={nd}: INCONCLUSIVE — comm term is inside the "
              f"arm spread")
    # Weak sanity check only: block-to-block movement inside the no-comm arm.
    # A steady difference in local work has zero drift, so this cannot bound
    # the contamination -- the wire arm above is what does that.
    drifts = [d for d in (block_drift_ms(f"nc_{nd}_a"),
                          block_drift_ms(f"nc_{nd}_b")) if d is not None]
    if drifts:
        drift = max(drifts)
        print(f"      ndev={nd}: the no-comm arm's per-step time moved "
              f"{drift:.3f} ms between its own blocks "
              f"({100 * drift / abs(comm) if comm else float('nan'):.0f}% of "
              f"the {comm:.3f} ms difference)")
        if abs(comm) > 0 and drift > 0.5 * abs(comm):
            print(f"      ndev={nd}: the no-comm arm is also unsteady within "
                  f"itself by more than half the communication term; treat "
                  f"every number in this row as provisional")
    else:
        print(f"      ndev={nd}: no per-block times in the receipt — the "
              f"trajectory-drift bound could not be computed")
    if BALLAST:
        b2 = (rows[f"b2_{nd}_a"] + rows[f"b2_{nd}_b"]) / 2
        b2nc = (rows[f"b2nc_{nd}_a"] + rows[f"b2nc_{nd}_b"]) / 2
        raw = b2 - full          # payload AND the packing it adds
        packing = b2nc - nc      # the packing alone, no wire
        wire = raw - packing     # what the extra bytes cost on the wire
        per_op = comm - wire
        print(f"      ndev={nd}: doubling the payload costs {raw:+.3f} ms, of "
              f"which {packing:+.3f} is device-side packing, so the extra "
              f"bytes cost {wire:+.3f} on the wire")
        print(f"      ndev={nd}: of {comm:.3f} ms of communication, "
              f"{wire:.3f} is payload and {per_op:.3f} is per-operation")
        if packing > 0.25 * raw:
            print(f"      ndev={nd}: packing is {100*packing/raw:.0f}% of the "
                  f"raw delta — an uncontrolled payload arm would have "
                  f"overstated the payload term by that much")
    if nd == 128:
        verdict_row = (comm, excess, spread, full)

if verdict_row is None:
    print("NO VERDICT: the gate thresholds below were written for the "
          "128-device row, which this run did not measure. The table above "
          "stands on its own; do not read a verdict into it.")
    sys.exit(0)
comm, excess, spread, full = verdict_row
if abs(comm) < 0.5 * spread * full:
    print("NO VERDICT: the 128-device communication term is inside its own "
          "arm spread")
    sys.exit(3)
if comm >= 1.2:
    print("VERDICT: CONFIRMS comm-bound at 128 devices — communication is at least "
          "half the excess over ideal; comm levers are the right target")
elif comm <= 0.4:
    print("VERDICT: REFUTES comm-bound at 128 devices — the excess is local work "
          "that stopped shrinking; comm levers aim at the wrong term")
    sys.exit(6)
else:
    print("VERDICT: intermediate — both terms matter; rank by size, do not "
          "assume the comm lever wins")
EOF
gate_rc=$?
# The gate's own exit code carried the abort and refute signals and was
# being thrown away, so a job whose gate said ABORT still exited 0.
if [ "$gate_rc" -ne 0 ]; then
  echo "RECEIPT GATE exited $gate_rc"
  rc=$gate_rc
fi
echo "DONE rc=$rc"
exit $rc
