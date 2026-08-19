#!/bin/bash -l
#SBATCH --job-name=mpas_kern_diff
#SBATCH --account=bd1179_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --exclude=l50003,l50139,l50027,l50009,l50100
#SBATCH --output=mpas_kern_diff.%j.log
# ===========================================================================
# WHICH KERNEL COSTS 2.5x PER CELL AT ONE PROBLEM SHAPE?
#
# Measured on a single GPU, 60 timed steps, 0.0% spread (job 27070658):
#
#   cells       ms/step   ns/cell
#    40,962       2.000      48.8
#   163,842      20.245     123.6   <- 2.53x its cheaper neighbour
#   655,362      50.170      76.6
# 2,621,442     173.750      66.3
#
# This is the whole 16-GPU scaling stall: at 16 GPUs each device holds
# 163,841 cells and the step takes 18.9 ms, which is the single-GPU time
# at that size. Communication measures zero on that lane, the ghost ring
# and the exchange schedule are on trend, and changing the partitioner
# does not fix it -- all four are receipted. So it is the kernels.
#
# Since it reproduces on ONE GPU with no collectives, a profile is now a
# clean instrument rather than the one that failed on this campaign
# before: the earlier capture's blind spot was collectives, and there are
# none here.
#
# Method: capture the two sizes that bracket the penalty, 163,842 and
# 655,362 cells, and compare kernels normalised per cell per step. The
# guilty kernel is the one whose per-cell cost RISES at the smaller
# problem -- the opposite of every kernel that merely has fixed overhead.
#
# Trap this job is written to avoid (campaign, 2026-08-13): the timed loop
# is a small tail of the trace because compilation dominates the wall, so
# a fraction-based window lands on autotuning instead of the loop. Steps
# are raised so the loop is seconds long, and the window is taken as the
# last N milliseconds, computed from the measured step time and the step
# count rather than guessed.
#
# A PRIORI GATE: the report names the kernels whose per-cell-per-step cost
# is at least 1.5x higher at 163,842 cells than at 655,362, ranked by how
# much absolute time that difference accounts for. If the top entry
# covers less than half the 47 ns/cell gap, say so -- the penalty is then
# spread across the whole kernel mix and points at a global effect
# (occupancy, cache residency) rather than one kernel.
#
# SUBMIT: sbatch scripts/cluster/scaling_levante/mpas_1gpu_kernel_diff.sbatch
# ===========================================================================
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_XLA_OVERLAP=0
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/kern_diff_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$SCRATCH/legoesm_jit_cache_j${SLURM_JOB_ID}"
mkdir -p "$LEGOESM_JIT_CACHE_DIR"
NSYS_BIN="${NSYS_BIN:-$(command -v nsys)}"
[ -n "$NSYS_BIN" ] || { echo "nsys not on PATH; cannot profile"; exit 3; }
rc=0
STEPS="${STEPS:-300}"

capture () {  # subdivision tag
  local SD=$1 TAG=$2
  echo "=== capture subdivision $SD [$TAG] $(date +%T) ==="
  JAX_ENABLE_X64=0 \
    timeout 3000 srun --export=ALL --nodes=1 --ntasks=1 \
      --ntasks-per-node=1 --gpus-per-node=1 --gpu-bind=none \
      --kill-on-bad-exit=1 \
    "$NSYS_BIN" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
      --force-overwrite=true -o "$OUTDIR/${TAG}" \
    "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
      --multicontroller --n-devices 1 --subdivision "$SD" --nlev 26 \
      --steps "$STEPS" --warmup 5 --lloyd 0 --partition-method sfc \
      --reorder-for 1 \
      --out "$OUTDIR/${TAG}.jsonl" || { echo "${TAG} FAILED"; rc=1; }
  "$NSYS_BIN" export --type sqlite --force-overwrite=true \
      --output "$OUTDIR/${TAG}.sqlite" "$OUTDIR/${TAG}.nsys-rep" \
    || { echo "${TAG} export FAILED"; rc=1; }
}

capture 7 s7
capture 8 s8

if [ "$rc" -ne 0 ]; then
  echo "ABORT: a capture failed (rc=$rc); no report"
  exit "$rc"
fi

echo "=== KERNEL DIFF ==="
STEPS="$STEPS" "$PY" - "$OUTDIR" <<'EOF'
import json, os, sqlite3, sys

outdir = sys.argv[1]
steps = int(os.environ["STEPS"])
CASES = {"s7": 163842, "s8": 655362}


def kernel_table(tag):
    """Total GPU ns per kernel name inside the timed loop.

    The loop is the LAST steps*ms of the trace: compilation dominates the
    wall here, so a fraction-based window would land on autotuning. The
    window is computed from the run's own measured step time, not guessed.
    """
    with open(os.path.join(outdir, tag + ".jsonl")) as fh:
        rec = json.loads([l for l in fh if l.startswith("{")][-1])
    ms = rec["steady_median_ms"]
    window_ns = int(steps * ms * 1e6 * 0.9)   # 90% of the loop, safely inside

    con = sqlite3.connect(os.path.join(outdir, tag + ".sqlite"))
    end = con.execute(
        "SELECT MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
    if end is None:
        raise RuntimeError(f"{tag}: no kernel rows in the trace")
    start = end - window_ns
    rows = con.execute(
        "SELECT s.value, SUM(k.end - k.start), COUNT(*) "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON s.id = k.demangledName "
        "WHERE k.start >= ? GROUP BY s.value", (start,)).fetchall()
    con.close()
    if not rows:
        raise RuntimeError(f"{tag}: window caught no kernels")
    return ms, {name: (total, n) for name, total, n in rows}, window_ns


out = {}
for tag, cells in CASES.items():
    ms, tbl, win = kernel_table(tag)
    span_steps = win / (ms * 1e6)
    # ns per cell per step for each kernel
    out[tag] = {name: total / span_steps / cells
                for name, (total, n) in tbl.items()}
    busy = sum(total for total, _ in tbl.values()) / span_steps / 1e6
    print(f"{tag}: {cells:,} cells, {ms:.3f} ms/step, "
          f"{len(tbl)} kernel names, {busy:.3f} ms/step on GPU "
          f"({100*busy/ms:.0f}% busy)")

names = set(out["s7"]) | set(out["s8"])
rank = []
for name in names:
    a = out["s7"].get(name, 0.0)
    b = out["s8"].get(name, 0.0)
    rank.append((a - b, a, b, name))
rank.sort(reverse=True)

gap = sum(out["s7"].values()) - sum(out["s8"].values())
print(f"\ntotal per-cell gap accounted for by kernels: {gap:.1f} ns/cell")
print(f"{'delta':>8} {'s7 ns/cell':>11} {'s8 ns/cell':>11} {'ratio':>7}  kernel")
top = 0.0
for d, a, b, name in rank[:20]:
    ratio = (a / b) if b > 0 else float("inf")
    if d > 0:
        top += d
    print(f"{d:8.2f} {a:11.2f} {b:11.2f} {ratio:7.2f}  {name[:88]}")

if rank and gap > 0:
    lead = rank[0][0]
    print(f"\nlargest single kernel covers {100*lead/gap:.0f}% of the gap")
    if lead >= 0.5 * gap:
        print("VERDICT: one kernel carries the penalty — name it and fix it")
    else:
        print("VERDICT: the penalty is spread across the kernel mix, which "
              "points at a global effect (occupancy, cache residency) "
              "rather than a single kernel")
EOF
gate_rc=$?
[ "$gate_rc" -ne 0 ] && rc=$gate_rc
echo "DONE rc=$rc"; exit $rc
