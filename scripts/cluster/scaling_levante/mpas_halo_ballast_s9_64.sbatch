#!/bin/bash -l
#SBATCH --job-name=wh_ballast
#SBATCH --account=bd1083_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=16
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --exclude=l50100
#SBATCH --output=wh_ballast.%j.log
# ===========================================================================
# HOW MUCH OF THE MPAS STEP IS HALO BANDWIDTH?  One variable, measured.
#
# The campaign's cost model splits the subdiv-9@64 wide-halo step (6.98 ms)
# into ~0.57 ms of per-round latency and ~2.64 ms of halo payload, leaving
# ~3.78 ms of compute.  That split is FITTED from three separate A/B
# receipts and it rests on a compute time imported from an OLDER nsys trace
# of a DIFFERENT (narrow-halo) configuration.  Every bytes-axis lever now
# under consideration -- mixed-precision halo, per-field halo depth,
# de-padding the coloured schedule -- is priced against that unmeasured
# number.  So measure it.
#
# Second knob in the same arm: LEGOESM_MPAS_HALO_NOCOMM=1 replaces every
# halo ppermute with the identity, keeping the gather, the concatenate,
# the scatter, the schedule shape and the enlarged local region. It
# BREAKS THE ANSWER and exists only to time on-device halo staging apart
# from wire time and peer skew -- the profiler on this stack records none
# of the halo collectives, so there is no other way to split them.
#   b1 - nocomm     = wire + skew
#   nocomm - 2.00   = staging + ghost-cell work (2.00 ms = same per-device
#                     load on ONE GPU, measured)
#   wire alone      = the ballast delta below, so skew falls out.
#
# LEGOESM_MPAS_HALO_BALLAST=N repeats each packed halo message N times on
# the wire and discards the copies on receipt.  Bytes scale by N; the round
# count, the schedule, the arithmetic and the answer are unchanged (bit
# identity and round-count invariance are unit-tested in
# tests/parallel/test_mpas_halo_ballast.py).  The step's response to N IS
# the bandwidth term.
#
# A PRIORI GATE (written before submit):
#   Predicted  step(N=2) - step(N=1) = +2.64 ms  (the fitted payload term).
#   CONFIRMS   delta in [2.3, 3.0] ms  -> the split is sound, the exchange
#              is bandwidth-dominated 4.6:1, and the bytes levers are worth
#              their predicted ~15-20 % each.
#   REFUTES    delta <= 1.8 ms -> payload is a SMALLER share than fitted;
#              the compute term is bigger than 3.78 ms and the next lever
#              is compute, not bytes.
#   REFUTES    delta >= 3.5 ms -> payload is LARGER than fitted; bytes
#              levers are worth more than predicted and mixed-precision
#              halo jumps to the top of the list.
#   N=4 is included as the linearity check: a bandwidth term must give
#   delta(4) ~= 3 x delta(2).  If it does not, the knob is measuring
#   something other than wire time (buffer allocation, kernel count) and
#   NO conclusion may be drawn from the N=2 row.
#
# Protocol matches the s9 ladder byte for byte except the env knob:
# s9 lloyd-0 prewarmed mesh, sfc, reorder-for 64, f32, nlev 26, wide halo
# ON (the current best), 60 timed steps after 5 warmup, discarded warm-up
# arm then INTERLEAVED arms -- 12-step arms gave 7.7 % spread on identical
# controls in this campaign, 60-step arms gave 0.3 %.
#
# SUBMIT: sbatch scripts/cluster/scaling_levante/mpas_halo_ballast_s9_64.sbatch
# ===========================================================================
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_XLA_OVERLAP=0
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/ballast_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$SCRATCH/legoesm_scaling/_jitcache_shared"
mkdir -p "$LEGOESM_JIT_CACHE_DIR"
# Match the 2^n ladder's environment so a split measured here can be
# compared against a ladder point: the NCCL multi-channel p2p pair
# (-13..-18 % at s9@64) and the shard-autotuning disable that this
# jaxlib needs at >= 64 ranks.
export NCCL_MIN_NCHANNELS=8 NCCL_MAX_NCHANNELS=8 NCCL_P2P_NET_CHUNKSIZE=131072
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_shard_autotuning=false"
rc=0

NODES="${NODES:-16}"; GPN="${GPN:-4}"; SUBDIV="${SUBDIV:-9}"
# Halo variant under test. The 2^n ladder runs RAGGED=1 (the current
# lane best, 5.73 ms at s9@64); the original ballast receipt ran the
# coloured schedule, so the variant has to be selectable or the split
# belongs to a configuration nothing else is measured in.
RAGGED="${RAGGED:-0}"
NDEV=$((NODES * GPN))

run_arm () {  # ballast tag [nocomm] [nostage]
  local BAL=$1 TAG=$2 NC=${3:-0} NS=${4:-0}
  echo "=== s${SUBDIV}@${NDEV} wide=1 ragged=$RAGGED ballast=$BAL nocomm=$NC nostage=$NS [$TAG] ==="
  LEGOESM_MPAS_WIDE_HALO=1 LEGOESM_MPAS_RAGGED_HALO="$RAGGED" \
  LEGOESM_MPAS_HALO_BALLAST=$BAL LEGOESM_MPAS_HALO_NOCOMM=$NC \
  LEGOESM_MPAS_HALO_NOSTAGE=$NS JAX_ENABLE_X64=0 \
    timeout 1800 srun --nodes="$NODES" --ntasks="$NDEV" \
      --ntasks-per-node="$GPN" --gpus-per-node="$GPN" \
      --gpu-bind=none --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
      --multicontroller --n-devices "$NDEV" \
      --subdivision "$SUBDIV" --nlev 26 \
      --steps 60 --warmup 5 --lloyd 0 --partition-method sfc \
      --reorder-for "$NDEV" \
      --out "$OUTDIR/${TAG}.jsonl" || { echo "${TAG} FAILED"; rc=1; }
  sleep 15
}

# Discarded warm-up arm: the node ramp must not alias into the first
# timed arm (this campaign lost three runs to exactly that).
run_arm 1 warmup_discard
# Interleaved, so drift is bounded by one arm rather than by arm order.
run_arm 1 b1_a
run_arm 1 nc_a 1
run_arm 1 ns_a 1 1
run_arm 1 b1_b
run_arm 1 nc_b 1
run_arm 1 ns_b 1 1
run_arm 2 b2_a
run_arm 2 b2_b

echo "=== RECEIPTS ==="
"$PY" - "$OUTDIR" "$NDEV" <<'EOF'
import json, os, sys
d = sys.argv[1]
ndev = sys.argv[2]
med = {}
for tag in ("b1_a", "b1_b", "nc_a", "nc_b", "ns_a", "ns_b",
            "b2_a", "b2_b"):
    p = os.path.join(d, tag + ".jsonl")
    try:
        lines = [l for l in open(p).read().strip().split("\n")
                 if l.startswith("{")]
        med[tag] = json.loads(lines[-1])["steady_median_ms"]
    except Exception as e:
        print(f"{tag}: MISSING -> {e}")
missing = [t for t in ("b1_a", "b1_b", "nc_a", "nc_b", "ns_a", "ns_b",
                      "b2_a", "b2_b") if t not in med]
if missing:
    # A gate that reports a number from a partial run is not a gate.
    print(f"ABORT: arms missing {missing}; no verdict")
    sys.exit(4)
for t, v in med.items():
    print(f"{t:>8}: {v:7.3f} ms")
if {"b1_a", "b1_b", "b2_a", "b2_b"} <= med.keys():
    b1 = (med["b1_a"] + med["b1_b"]) / 2
    b2 = (med["b2_a"] + med["b2_b"]) / 2
    spread1 = abs(med["b1_a"] - med["b1_b"]) / b1
    spread2 = abs(med["b2_a"] - med["b2_b"]) / b2
    d2 = b2 - b1
    print(f"b1 mean {b1:.3f}  b2 mean {b2:.3f}  delta(2) {d2:+.3f} ms")
    print(f"arm spreads: b1 {spread1:.1%}  b2 {spread2:.1%}")
    if "nc_a" in med and "nc_b" in med:
        nc = (med["nc_a"] + med["nc_b"]) / 2
        # nocomm keeps the gather/scatter staging and the enlarged local
        # region but deletes every collective, so:
        #   b1 - nocomm = wire time + waiting for the slowest peer
        #   nocomm - 2.00 = on-device halo staging + ghost-cell work
        #                   (2.00 ms = the same load on ONE GPU)
        print(f"nocomm mean {nc:.3f}  spread "
              f"{abs(med['nc_a'] - med['nc_b']) / nc:.1%}")
        print(f"  wire + skew          = b1 - nocomm = {b1 - nc:+.3f} ms")
        print(f"  staging + ghost      = nocomm - 2.000 = {nc - 2.0:+.3f} ms")
        print(f"  of which wire (ballast) = {d2:.3f} ms  "
              f"=> skew = {b1 - nc - d2:+.3f} ms")
    if "ns_a" in med and "ns_b" in med and "nc_a" in med:
        ns = (med["ns_a"] + med["ns_b"]) / 2
        nc = (med["nc_a"] + med["nc_b"]) / 2
        # nostage deletes the halo entirely while keeping the kernel, the
        # local region and the masking, so it is the SAME program at 4
        # and at 64 devices. The difference between the two scales is
        # therefore whatever grows with device count that is NOT the
        # halo: the mass-fix allreduce across nodes, and jitter.
        print(f"  staging (nocomm - nostage) = {nc - ns:+.3f} ms")
        print(f"  nostage@{ndev} = {ns:.3f} ms   (compare the 4-device "
              f"nostage arm: 2.725 ms at the SAME per-device load; the "
              f"difference is the residual that grows with device count)")
    # A ratio smaller than half the arm spread is not a result.
    if abs(d2) < 0.5 * max(spread1, spread2) * b1:
        print("INCONCLUSIVE: delta is inside the arm spread")
    if "b4_a" in med:
        d4 = med["b4_a"] - b1
        lin = d4 / d2 if d2 else float("nan")
        print(f"delta(4) {d4:+.3f} ms   linearity delta(4)/delta(2) "
              f"{lin:.2f}  (a bandwidth term predicts 3.00)")
        # Enforce the a-priori gate written in the header, so the
        # verdict is the SCRIPT's not the reader's.
        if not (2.0 <= lin <= 4.0):
            print("VERDICT: NON-LINEAR in payload -- the knob is not "
                  "measuring wire time; the delta(2) row means nothing")
            sys.exit(5)
        if 2.3 <= d2 <= 3.0:
            print("VERDICT: CONFIRMS the fitted 2.64 ms payload term")
        elif d2 <= 1.8:
            print("VERDICT: REFUTES -- payload is a SMALLER share than "
                  "fitted; compute is the bigger term")
            sys.exit(6)
        elif d2 >= 3.5:
            print("VERDICT: REFUTES -- payload is LARGER than fitted; "
                  "bytes levers are worth more than predicted")
            sys.exit(7)
        else:
            print("VERDICT: intermediate -- reprice before acting")
EOF
gate_rc=$?
# A crashed gate must not report the job COMPLETED: this block exited
# non-zero on a NameError once while the job still exited 0.
if [ "$gate_rc" -ne 0 ]; then
  echo "RECEIPT GATE exited $gate_rc"
  rc=$gate_rc
fi
echo "DONE rc=$rc"
exit $rc
