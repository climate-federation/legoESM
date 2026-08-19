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
# overlap + the NCCL multi-channel p2p pair, 60 timed steps after 10
# warmup, a discarded warm-up arm, then INTERLEAVED arms (12-step arms
# gave 7.7 % spread on identical controls in this campaign; 60-step arms
# gave 0.3 %).
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

run_arm () {  # ndev nocomm tag
  local NDEV=$1 NC=$2 TAG=$3
  local NODES=$(( (NDEV + GPN - 1) / GPN ))
  local NTPN=$(( NDEV < GPN ? NDEV : GPN ))
  local HOSTS
  HOSTS=$(pin_hosts "$NODES")
  echo "=== LL${NLAT}@${NDEV} nocomm=$NC [$TAG] $(date +%T) hosts=$HOSTS ==="
  LEGOESM_LATLON_PACKED_EXCHANGE=1 LEGOESM_LATLON_SPMD_FUSED_HALO=1 \
  LEGOESM_LATLON_HALO_NOCOMM=$NC JAX_ENABLE_X64=0 \
    timeout 1500 srun --export=ALL --nodelist="$HOSTS" \
      --nodes="$NODES" --ntasks="$NDEV" \
      --ntasks-per-node="$NTPN" --gpus-per-node="$NTPN" \
      --gpu-bind=none --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_atm_latlon_spmd_scaling.py \
      --multicontroller --n-devices "$NDEV" --mode strong \
      --n-lat "$NLAT" --n-lon "$NLON" --nlev 26 \
      --steps 60 --warmup 10 \
      --out "$OUTDIR/${TAG}.jsonl" || { echo "${TAG} FAILED"; rc=1; }
  sleep 15
}

# Discarded warm-up arm: the node ramp must not alias into the first timed
# arm (this campaign lost three runs to exactly that).
run_arm "$FIRST_DEV" 0 warmup_discard
for ND in $DEVS; do
  run_arm "$ND" 0 "full_${ND}_a"
  run_arm "$ND" 1 "nc_${ND}_a"
  run_arm "$ND" 0 "full_${ND}_b"
  run_arm "$ND" 1 "nc_${ND}_b"
done

echo "=== RECEIPTS ==="
# An arm that failed leaves a stale or missing receipt; scoring on top of
# that reports a number from a run that did not happen.
if [ "$rc" -ne 0 ]; then
  echo "ABORT: an arm failed (rc=$rc); no verdict"
  exit "$rc"
fi
"$PY" - "$OUTDIR" "$DEVS" <<'EOF'
import json, os, sys

outdir = sys.argv[1]
DEVS = tuple(sorted(int(d) for d in sys.argv[2].split()))
# Single-GPU LL2048 ladder point (job 27051113), same build and flags.
T1 = 201.4807


def med(tag):
    path = os.path.join(outdir, tag + ".jsonl")
    lines = [l for l in open(path).read().strip().split("\n")
             if l.startswith("{")]
    return json.loads(lines[-1])["steady_median_ms"]


rows = {}
missing = []
for nd in DEVS:
    for kind in ("full", "nc"):
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

print(f"{'ndev':>5} {'full':>8} {'nocomm':>8} {'comm':>8} {'comm%':>7} "
      f"{'ideal':>8} {'excess':>8} {'comm/excess':>12} {'spread':>8}")
verdict_row = None
for nd in DEVS:
    full = (rows[f"full_{nd}_a"] + rows[f"full_{nd}_b"]) / 2
    nc = (rows[f"nc_{nd}_a"] + rows[f"nc_{nd}_b"]) / 2
    spread = max(abs(rows[f"full_{nd}_a"] - rows[f"full_{nd}_b"]) / full,
                 abs(rows[f"nc_{nd}_a"] - rows[f"nc_{nd}_b"]) / nc)
    comm = full - nc
    ideal = T1 / nd
    excess = full - ideal
    frac = comm / excess if excess > 0 else float("nan")
    print(f"{nd:>5} {full:8.3f} {nc:8.3f} {comm:8.3f} {100*comm/full:6.1f}% "
          f"{ideal:8.3f} {excess:8.3f} {frac:11.2f} {spread:7.1%}")
    # A difference smaller than half the arm spread is not a result.
    if abs(comm) < 0.5 * spread * full:
        print(f"      ndev={nd}: INCONCLUSIVE — comm term is inside the "
              f"arm spread")
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
echo "DONE rc=$rc"
exit $rc
