#!/bin/bash -l
# MPAS (icosahedral) lane: sweep NCCL channel count at fixed node count.
#
# Background: on the lat-lon lane (64 GPUs, 2048x4096x26) the NCCL channel
# count was a large free win and the campaign default (8) was the WORST value
# (full step 4.73 / 4.91 / 4.31 / 4.30 ms at c4/c8/c16/c32; halo-minus-nohalo
# communication 1.36 / 1.54 / 0.95 / 0.93 ms). The MPAS lane's halo is a
# different animal: unstructured neighbour lookup, edge-coloured into 11-14
# collective-permute rounds per step, and a separate earlier result found that
# "8 channels + NCCL_P2P_NET_CHUNKSIZE=131072" cut its 64-GPU step by 13-18%.
# Nobody has ever swept the channel count on THIS lane. This script does
# exactly that, with NCCL_P2P_NET_CHUNKSIZE pinned to the known-good 131072 so
# the channel count is the only NCCL knob moving.
#
# Arms (each = one 65-step timed run, warmup 5, subdivision 9, nlev 26):
#   - 1 discarded warm-up arm (halo on, 8 channels) to absorb one-time
#     JIT/compile and mesh-cache effects; its receipt is not summarised.
#   - 1 final five-step validity arm with NCCL_DEBUG=INFO, whose log is
#     grepped for the channel count NCCL actually chose. An arm whose knob
#     NCCL never saw is VOID, not REFUTED.
#   - 4 channel counts {4,8,16,32} x halo {on,off} x 2 reps. Within one rep
#     the channel counts are visited in one order and the two halo settings
#     are visited back to back at each count, and the next rep reverses both,
#     so neither the channel order nor the halo order accumulates node drift.
#   LEGOESM_MPAS_HALO_NOCOMM=1 replaces every halo ppermute with the identity.
#   It BREAKS THE ANSWER (each device scatters its own rows into its halo
#   slots) and exists only to price the exchange: the profiler on this stack
#   does not record these collectives at all. It leaves the kernels, the local
#   region, the staging and the masking untouched, so a channel count that
#   moves the halo-off arm is moving something other than communication and
#   the arm is a drift detector as well as a control.
#
# Level count is 26 -- the value every prior receipt on this lane used and the
# value the model runs today -- so this sweep can be read against the 6.75 ms
# baseline directly. NLEV is an override, not a decision taken here.
#
# Falsifiable, written before submit:
#   - FLAT curve (all four counts within rep-to-rep noise): channel count is
#     not the lever on the collective-permute halo; the earlier 8+chunksize
#     win was chunksize-driven, not channel-driven. Keep 8 and stop spending
#     node-hours on this knob here.
#   - MONOTONE SATURATING curve (improves 4->8->16, flat 16->32): the lane is
#     channel-starved exactly like the lat-lon lane; the saturation point is
#     the setting to propose as the lane default -- another free win at the
#     device counts where the model plateaus.
#   - MINIMUM AT 8 (U-shape): 8 is genuinely optimal for ~11-14 small rounds
#     per step; the earlier "8 + chunksize 131072" result was a joint optimum,
#     not an accident. Keep 8, close the question.
#   - The halo-off arm MOVES with the channel count: the knob is not confined
#     to communication on this lane and no arm of this sweep is interpretable
#     as a communication effect. Report it as VOID, not as a win.
#     On the lat-lon lane that control was flat to 0.3%.
#SBATCH --job-name=mpas_chan
#SBATCH --account=bd1179_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=16
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:30:00
#SBATCH --exclude=l50003,l50139,l50027,l50009,l50100
#SBATCH --output=mpas_chan.%j.log
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
export LEGOESM_XLA_OVERLAP=0
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_shard_autotuning=false --xla_gpu_enable_nccl_comm_splitting=false"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="$SCRATCH/legoesm_scaling/mpas_chan${SLURM_JOB_NUM_NODES}n_j${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$OUTDIR/jitcache"; mkdir -p "$LEGOESM_JIT_CACHE_DIR"

# Node count comes from the allocation, so `sbatch --nodes=N` of this same
# file re-runs the whole sweep at N*4 GPUs with nothing else edited.
NODES="${SLURM_JOB_NUM_NODES:-16}"; GPN=4; SUBDIV="${SUBDIV:-9}"; NLEV="${NLEV:-26}"
NDEV=$((NODES * GPN))
echo "sweep at ${NDEV} GPUs on ${NODES} nodes (subdivision ${SUBDIV}, nlev ${NLEV})"

# An arm that dies leaves a missing or stale receipt, and a median taken over
# what survived is a number from a run that did not happen. Track it and say so.
rc=0

# Accepted arms on this lane land about five minutes apart, so 300 s would
# kill the cold first arm and 900 s x 17 does not fit any walltime this job
# would be scheduled with. 480 s per arm x 17 arms plus the sleeps is 2.4 h,
# inside the three-hour request.
ARM_TIMEOUT="${ARM_TIMEOUT:-480}"

run_arm () {  # channels nocomm tag [steps] [debug]
  local CH=$1 NC=$2 TAG=$3 STEPS=${4:-65} DBGON=${5:-0}
  local DBG="" TMO="$ARM_TIMEOUT"
  # A twelve-step validity arm cannot need the full timed-arm budget, and
  # four of them at the full budget would not fit the allocation.
  if [ "$STEPS" -le 20 ]; then TMO=240; fi
  if [ "$DBGON" = 1 ]; then
    DBG="NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ENV,INIT,GRAPH,P2P,TUNING NCCL_DEBUG_FILE=$OUTDIR/nccl_${TAG}.%h.%p"
  fi
  echo "--- $TAG channels=$CH nocomm=$NC steps=$STEPS $(date +%T)"
  printf 'NCCL_MIN_NCHANNELS=%s NCCL_MAX_NCHANNELS=%s NCCL_P2P_NET_CHUNKSIZE=131072 LEGOESM_MPAS_HALO_NOCOMM=%s\n' \
    "$CH" "$CH" "$NC" > "$OUTDIR/${TAG}.env"
  # Every halo knob is PINNED, not merely left alone: one inherited
  # LEGOESM_MPAS_HALO_NOSTAGE=1 in the submitting shell would skip halo
  # staging on every rank and the receipts would still look ordinary.
  env $DBG \
  LEGOESM_MPAS_WIDE_HALO=1 LEGOESM_MPAS_WIDE_HALO_STRIDE= \
  LEGOESM_MPAS_RAGGED_HALO=0 LEGOESM_MPAS_HALO_BALLAST=1 \
  LEGOESM_MPAS_HALO_NOSTAGE=0 \
  LEGOESM_MPAS_HALO_NOCOMM=$NC JAX_ENABLE_X64=0 \
  NCCL_MIN_NCHANNELS=$CH NCCL_MAX_NCHANNELS=$CH \
  NCCL_P2P_NET_CHUNKSIZE=131072 \
    timeout --kill-after=30 "$TMO" srun --export=ALL --nodes="$NODES" --ntasks="$NDEV" \
      --ntasks-per-node="$GPN" --gpus-per-node="$GPN" \
      --gpu-bind=none --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
      --multicontroller --n-devices "$NDEV" \
      --subdivision "$SUBDIV" --nlev "$NLEV" \
      --steps "$STEPS" --warmup 5 --lloyd 0 --partition-method sfc \
      --reorder-for "$NDEV" \
      --out "$OUTDIR/${TAG}.jsonl" || { echo "ARM $TAG FAILED"; rc=1; }
  sleep 15
}

run_arm 8 0 warmup_discard
# Rep 1 walks the channel counts up and takes halo-on before halo-off at each;
# rep 2 walks down and takes halo-off first. Reversing BOTH orders is what
# keeps a drift on the node out of the channel effect AND out of the
# halo-on-minus-halo-off difference, which a block of one then a block of the
# other does not.
for CH in 4 8 16 32; do
  run_arm "$CH" 0 "halo_c${CH}_r1"
  run_arm "$CH" 1 "nohalo_c${CH}_r1"
done
for CH in 32 16 8 4; do
  run_arm "$CH" 1 "nohalo_c${CH}_r2"
  run_arm "$CH" 0 "halo_c${CH}_r2"
done
# Validity, one arm PER CHANNEL COUNT: a curve point whose channel count NCCL
# silently clamped is not a curve point. The benchmark refuses warmup >= steps,
# so these cannot be five-step arms.
for CH in 4 8 16 32; do run_arm "$CH" 0 "val_c${CH}" 12 1; done
echo "=== NCCL validity: the channel count NCCL actually built, per arm ==="
for CH in 4 8 16 32; do
  # NCCL prints "Channel 00/04", so the count is ZERO PADDED. Comparing the
  # padded string against the requested count fails for 4 and 8 and would
  # void exactly the two arms this sweep exists to compare.
  # NCCL prints "Channel 00/04", so the count is ZERO PADDED — comparing the
  # padded string against the requested count fails for 4 and 8 and would
  # void exactly the arms this sweep exists to compare. It ALSO prints
  # lines whose denominator is 00 (communicators built with no channels of
  # their own); those are not a channel count and requiring the built set to
  # equal the requested one without dropping them voids every arm, which is
  # what the first version of this gate did.
  SEEN=$(grep -h -oE "Channel [0-9]+/[0-9]+" "$OUTDIR"/nccl_val_c${CH}.* 2>/dev/null \
         | sed 's:.*/::' | while read -r n; do echo $((10#$n)); done \
         | grep -v '^0$' | sort -un | paste -sd,)
  CHUNK=$(grep -h -oE "P2P Chunksize set to [0-9]+" "$OUTDIR"/nccl_val_c${CH}.* 2>/dev/null \
          | grep -oE "[0-9]+$" | sort -u | paste -sd,)
  echo "  requested ${CH}: NCCL built ${SEEN:-<no debug output>} channels, chunksize ${CHUNK:-<none>}"
  # A knob NCCL never saw makes that curve point VOID, not REFUTED. The count
  # must be the ONLY one NCCL built: a mixed "4,8" means some communicator
  # ignored the request, and then the arm is not a single-channel-count point.
  if [ "$SEEN" != "$CH" ]; then
    echo "    VOID: asked for ${CH} channels, NCCL built '${SEEN:-nothing}'."
    rc=1
  fi
  # The chunk size is held fixed across the whole sweep, so a communicator
  # that took a DIFFERENT one makes its arm incomparable. Absence of the line
  # is a question about what this NCCL logs, not evidence of a wrong value —
  # voiding on absence is how a validity gate starts failing valid runs.
  if [ -n "$CHUNK" ] && [ "$CHUNK" != "131072" ]; then
    echo "    VOID: chunk size was '$CHUNK', not the pinned 131072."
    rc=1
  elif [ -z "$CHUNK" ]; then
    echo "    note: this NCCL did not log a chunk size; the channel count did bind."
  fi
done

"$PY" - "$OUTDIR" "$NDEV" "$SUBDIV" "$NLEV" <<'PYEOF'
import json, os, statistics, sys

outdir, ndev, subdiv, nlev = (sys.argv[1], int(sys.argv[2]),
                             int(sys.argv[3]), int(sys.argv[4]))
CHS = [4, 8, 16, 32]
ARMS = [("halo", "0"), ("nohalo", "1")]
problems = []


permutes = {}


def load_arm(tag, want_nocomm):
    """Steady median per-step ms from one arm's receipt, or None.

    Every field the arm's identity depends on is checked against the
    receipt, because a filename is not evidence that the knob took: a
    run on the wrong device count, on the CPU backend, or with the halo
    switch unset produces a perfectly plausible number.
    """
    path = os.path.join(outdir, tag + ".jsonl")
    if not os.path.isfile(path):
        problems.append("MISSING: %s.jsonl -- arm produced no receipt" % tag)
        return None
    try:
        lines = [ln for ln in open(path).read().splitlines() if ln.strip()]
    except OSError as exc:
        problems.append("UNREADABLE: %s.jsonl (%s)" % (tag, exc))
        return None
    if not lines:
        problems.append("UNREADABLE: %s.jsonl (empty)" % tag)
        return None
    try:
        rec = json.loads(lines[-1])
        value = rec["steady_median_ms"]
        meta = rec["metadata"]
        knobs = meta["extra"]["halo_knobs"]
        permutes[tag] = rec["hlo_collective_permutes"]
    except (ValueError, KeyError, TypeError) as exc:
        problems.append("UNREADABLE: %s.jsonl (%s)" % (tag, exc))
        return None
    if not isinstance(value, (int, float)):
        problems.append("UNREADABLE: %s.jsonl (steady_median_ms is %r)" % (tag, value))
        return None
    bind = [
        ("device count", rec.get("n_devices"), ndev),
        ("platform", rec.get("platform"), "gpu"),
        ("transport", meta.get("transport"), "nccl"),
        ("multi-controller", rec.get("multicontroller"), True),
        ("subdivision", rec.get("subdivision"), subdiv),
        ("levels", rec.get("nlev"), nlev),
        ("partitioner", rec.get("partition_method"), "sfc"),
        ("halo switch", knobs.get("LEGOESM_MPAS_HALO_NOCOMM"), want_nocomm),
        ("wide halo", knobs.get("LEGOESM_MPAS_WIDE_HALO"), "1"),
        ("wide-halo stride", knobs.get("LEGOESM_MPAS_WIDE_HALO_STRIDE"), ""),
        ("ragged halo", knobs.get("LEGOESM_MPAS_RAGGED_HALO"), "0"),
        ("ballast", knobs.get("LEGOESM_MPAS_HALO_BALLAST"), "1"),
        ("no-stage", knobs.get("LEGOESM_MPAS_HALO_NOSTAGE"), "0"),
    ]
    for name, got, want in bind:
        if got != want:
            problems.append("WRONG ARM: %s ran with %s = %r, expected %r"
                            % (tag, name, got, want))
            return None
    return float(value)


cells = {}   # (arm, channels) -> median over the reps that survived
for arm, nocomm in ARMS:
    for ch in CHS:
        got, lost, vals = [], [], []
        for rep in (1, 2):
            v = load_arm("%s_c%d_r%d" % (arm, ch, rep), nocomm)
            if v is None:
                lost.append(rep)
            else:
                got.append(rep)
                vals.append(v)
        if not vals:
            problems.append("NO DATA: %s c%d -- both reps missing/unreadable" % (arm, ch))
            continue
        cells[(arm, ch)] = statistics.median(vals)
        if lost:
            problems.append(
                "PARTIAL: %s c%d from rep %s only (rep %s missing/unreadable)"
                % (arm, ch, "+".join(map(str, got)), "+".join(map(str, lost))))

print("MPAS channel sweep @ %d GPUs -- median per-step ms over reps" % ndev)
for arm, _ in ARMS:
    ref = cells.get((arm, 8))
    print("")
    print("%s (%s):" % (arm, "the real run" if arm == "halo"
                        else "exchange replaced by the identity -- WRONG ANSWERS, control only"))
    if ref is None:
        print("  reference arm (8 channels) missing; percentages not available")
    else:
        print("  reference: 8 channels = %.4f ms" % ref)
    for ch in CHS:
        v = cells.get((arm, ch))
        if v is None:
            print("  channels=%-3d  NO USABLE RECEIPT" % ch)
        elif ref is None:
            print("  channels=%-3d  %.4f ms" % (ch, v))
        else:
            print("  channels=%-3d  %.4f ms   %+.2f%% vs 8-channel"
                  % (ch, v, 100.0 * (v - ref) / ref))

print("")
print("communication per channel count: halo - nohalo (ms/step)")
for ch in CHS:
    on, off = cells.get(("halo", ch)), cells.get(("nohalo", ch))
    if on is None or off is None:
        print("  channels=%-3d  INCOMPLETE (halo=%s, nohalo=%s)" % (ch, on, off))
    else:
        print("  channels=%-3d  %.4f ms   (%.1f%% of the step)"
              % (ch, on - off, 100.0 * (on - off) / on))

census_absent = set()
# The halo-off arm must actually compile FEWER collective-permutes than the
# halo-on arm. Without this the control could be running the full exchange
# and the difference between them would be noise wearing a label.
for ch in CHS:
    for rep in (1, 2):
        on = permutes.get("halo_c%d_r%d" % (ch, rep))
        off = permutes.get("nohalo_c%d_r%d" % (ch, rep))
        if on is None or off is None:
            # The in-process HLO census cannot run under multiple controllers
            # -- the step closes over sharded schedule arrays and JAX refuses
            # to re-lower it -- so EVERY GPU arm records null here. That is a
            # missing instrument, not a failed control, and voiding on it
            # would refuse every multi-node run this sweep exists for. The
            # halo switch is bound from the receipt instead, which is what
            # actually says which arm ran.
            census_absent.add((ch, rep))
            continue
        if not (isinstance(on, int) and isinstance(off, int) and off < on):
            problems.append(
                "CONTROL INOPERATIVE: c%d rep %d compiled %r collective-permutes "
                "with the halo on and %r with it off" % (ch, rep, on, off))

nh = [cells[("nohalo", c)] for c in CHS if ("nohalo", c) in cells]
if len(nh) == len(CHS):
    spread = 100.0 * (max(nh) - min(nh)) / min(nh)
    print("")
    print("control spread across channel counts (halo removed): %.2f%%" % spread)
    if spread > 1.0:
        problems.append(
            "VOID: the halo-off control moved %.2f%% across channel counts. "
            "The knob is not confined to communication on this lane, or the "
            "node drifted; no arm is interpretable as a communication effect."
            % spread)

if census_absent:
    print("")
    print("note: the compiled-collective census is unavailable in %d of %d "
          "arm pairs (it cannot run under multiple controllers); the halo "
          "switch is bound from the receipt instead."
          % (len(census_absent), len(CHS) * 2))

print("")
if problems:
    print("SWEEP INCOMPLETE -- the numbers above are NOT a verdict.")
    print("RECEIPT PROBLEMS (%d) -- stated here, NOT silently skipped:" % len(problems))
    for p in problems:
        print("  " + p)
    sys.exit(3)
print("all %d sweep receipts present, readable and bound to the arm they claim"
      % (len(CHS) * len(ARMS) * 2))
PYEOF
SUMRC=$?
if [ "$rc" -ne 0 ] || [ "$SUMRC" -ne 0 ]; then
  echo "SWEEP FAILED (arm rc=$rc, summary rc=$SUMRC) -- no verdict"
  exit 1
fi
echo DONE
