#!/bin/bash -l
#SBATCH --job-name=cpu_cap_test
#SBATCH --account=bb1596
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:50:00
#SBATCH --output=cpu_pool_cap_%j.log
# SUBMIT: sbatch --export=ALL,LEGOESM_REPO=$PWD,\
#   LEGOESM_PYTHON=/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python \
#   scripts/cluster/scaling_levante/cpu_pool_cap_vs_partition.sbatch
# ===========================================================================
# IS THE FLAT ~34 BUSY PROCESSORS A FIXED WORK SPLIT, OR A CAP NOBODY LIFTED?
#
# Job 27686824 measured occupancy flat at 33-37 busy processors across a 16x
# change in local work, and both reviewers refused the causal reading: a capped
# worker pool, or a control-group CPU quota of about that size, produces exactly
# the same flatness with 590 threads present but never runnable. The earlier
# three-flag sweep measuring +-2% is itself a hint that something is capped.
#
# Three checks, cheapest first, all on one node with one rank and no
# collectives:
#   1. RECORD THE QUOTA. cpu.max and cpu.stat of this job's control group and
#      its ancestors. A quota near 34 cores, or a non-zero throttle count, ends
#      the argument immediately.
#   2. COUNT THE BUSY THREADS. Per-thread CPU deltas during the timed phase.
#      Exactly K recurring busy threads means a pool cap; a broad, shifting set
#      means the work is being split that way.
#   3. LIFT THE POOL AND RE-RUN. The same arm with the thread pool set
#      explicitly above the observed width. Busy rising above ~34 falsifies the
#      fixed-split reading outright.
#
# PRE-REGISTERED READING, on the 1024 arm (chosen over 2048 to fit the window;
# its occupancy was 36.59 and its per-cell cost matches 2048's to 0.1%):
#   quota < 64 cores or nr_throttled > 0            -> CAP. Reading settled.
#   pool lifted and busy > 45                       -> CAP, and liftable.
#   pool lifted and busy within +-25% of 36.6       -> not a pool cap; the fixed
#                                                      split survives this test
#   fewer than 8 in-window samples on any arm       -> refuse the arm
# The denominator is also corrected here: the affinity mask is 64 PHYSICAL cores
# plus their hyperthread siblings, so occupancy is reported against 64, not 128.
# ===========================================================================
set -u
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
export JAX_PLATFORMS=cpu
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
export LD_LIBRARY_PATH="/sw/spack-levante/gcc-13.4.0-c4mgfk/lib64:/sw/spack-levante/openmpi-4.1.2-mnmady/lib:${LD_LIBRARY_PATH:-}"
OUTDIR="/scratch/b/$USER/legoesm_scaling/pool_cap_${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"

echo "=== node $(hostname) $(date)"
echo "--- 1. control-group CPU quota and throttling, this task's cgroup and ancestors"
srun --overlap -n1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --cpu-bind=cores bash -c '
  cg=$(awk -F: "/^0::/{print \$3}" /proc/self/cgroup)
  echo "[cgroup] $cg"
  d=/sys/fs/cgroup$cg
  while [ "$d" != "/sys/fs/cgroup" ] && [ -n "$d" ]; do
    for f in cpu.max cpu.stat cpuset.cpus.effective; do
      [ -r "$d/$f" ] && echo "[quota] $d/$f: $(tr "\n" " " < "$d/$f")"
    done
    d=$(dirname "$d")
  done
  echo "[affinity] nproc=$(nproc) taskset=$(taskset -cp $$ | sed "s/.*: //")"
  echo "[physical] $(lscpu | grep -E "^Core|^Socket|^Thread")"'

run_arm () {  # $1 = label, rest = env assignments
  local label="$1"; shift
  echo "=== arm ${label}  env: $*  $(date +%T)"
  "$PY" scripts/bench/sample_cpu_cores.py --pattern run_cpu_mpi_scaling.py \
        --out "$OUTDIR/cores_${label}.jsonl" --interval 2 --duration 2400 --per-thread &
  local sampler=$!
  env "$@" srun --overlap -n1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --cpu-bind=cores \
      "$PY" scripts/bench/run_cpu_mpi_scaling.py --grid latlon --mode strong \
      --resolution 1024 --n-levels 40 --precision float32 --physics none \
      --n-warmup 3 --n-timing 20 --output-dir "$OUTDIR" \
      > "$OUTDIR/run_${label}.log" 2>&1
  local rc=$?
  kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null
  [ "$rc" -ne 0 ] && { echo "ARM ${label} FAILED rc=${rc}"; tail -5 "$OUTDIR/run_${label}.log"; return 1; }
  "$PY" - "$OUTDIR/run_${label}.log" "$OUTDIR/cores_${label}.jsonl" "$label" <<'PYEOF'
import json, re, sys
log, samples, label = sys.argv[1:4]
text = open(log).read()
m0 = re.search(r"\[phase\] timing-start rank=0 epoch=([\d.]+)", text)
m1 = re.search(r"\[phase\] timing-end rank=0 epoch=([\d.]+)", text)
if not (m0 and m1):
    raise SystemExit(f"{label}: NO PHASE MARKERS — cannot window")
t0, t1 = float(m0.group(1)), float(m1.group(1))
rows = [json.loads(l) for l in open(samples)]
inside = [r for r in rows if t0 <= r["epoch"] <= t1]
if len(inside) < 8:
    raise SystemExit(f"{label}: only {len(inside)} in-window samples — REFUSING")
busy = sorted(r["cores_busy"] for r in inside)
med = busy[len(busy) // 2]
q1, q3 = busy[len(busy) // 4], busy[(3 * len(busy)) // 4]
over = sorted(r.get("tid_busy_over_10pct", 0) for r in inside)
overmed = over[len(over) // 2]
step = re.search(r"Timing:\s*([\d.]+)\s*ms/step", text)
print(f"RESULT {label} busy_median={med:.2f} iqr=[{q1:.2f},{q3:.2f}] n={len(inside)} "
      f"of_64_physical={100 * med / 64:.0f}% threads_over_10pct={overmed} "
      f"step_ms={step.group(1) if step else 'NOT FOUND'}")
PYEOF
}

fail=0
run_arm baseline || fail=1
# Lift every pool this runtime might be capped by, well above the observed width.
run_arm pool192 XLA_FLAGS=--xla_cpu_parallel_codegen_split_count=192 \
        OMP_NUM_THREADS=128 EIGEN_MAX_THREADS=128 \
        TF_NUM_INTEROP_THREADS=128 TF_NUM_INTRAOP_THREADS=128 || fail=1
[ "$fail" -ne 0 ] && { echo "=== POOL_CAP_INCOMPLETE"; exit 3; }
echo "=== POOL_CAP_DONE $(date)"; echo "OUTDIR=$OUTDIR"
