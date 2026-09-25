#!/bin/bash -l
#SBATCH --job-name=cpu_thr_bw
#SBATCH --account=bb1596
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --output=cpu_threads_bw_%j.log
# SUBMIT: sbatch --export=ALL,LEGOESM_REPO=$PWD,\
#   LEGOESM_PYTHON=/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python \
#   scripts/cluster/scaling_levante/cpu_threads_and_bandwidth.sbatch
# ===========================================================================
# WHY DO 128 THREADS SUM TO ONLY 35 BUSY PROCESSORS?
#
# Three explanations have now died on this lane. Work-size: occupancy is flat
# across a 16x change in local problem (job 27686824). Control-group quota:
# cpu.max is unlimited at every level and nothing was ever throttled. Worker-pool
# cap: lifting the codegen split, the OpenMP count, the Eigen maximum and both
# TensorFlow pools moved occupancy 0.5% and step time 0.7% (job 27687266).
#
# What that job also showed is the shape nobody had looked at: 128 threads are
# EACH more than 10% busy and together they sum to ~35 cores, i.e. the whole
# pool runs at about 27% duty. Threads get work and then wait. Two candidates
# remain, and they call for opposite work:
#
#   (i)  MEMORY SYSTEM. The lane is near the node's bandwidth ceiling, so extra
#        threads cannot help and waiting is unavoidable. Fix: arithmetic
#        intensity -- fusion, blocking, layout -- not parallelism.
#   (ii) SYNCHRONISATION. Many short parallel regions with a barrier between
#        them; threads idle at the barriers. Fix: fewer, longer kernels.
#
# THE TWO MEASUREMENTS, both at FIXED work (1024x2048x40):
#   A. Thread sweep 16/32/64/128, the driver takes its count from
#      --cpus-per-task. Parallel efficiency, measured directly.
#   B. DRAM traffic from hardware counters during the timed phase, against a
#      STREAM triad ceiling measured on THIS node in the same job -- a
#      published bandwidth figure is not a denominator.
#
# PRE-REGISTERED READING:
#   speedup 16 -> 128 threads >= 5x                  -> parallelism is fine, and
#                                                       the 35-busy figure is a
#                                                       duty-cycle artifact, not
#                                                       a ceiling
#   speedup <= 2x AND traffic >= 50% of STREAM       -> (i) memory system
#   speedup <= 2x AND traffic <= 20% of STREAM       -> (ii) synchronisation
#   anything else                                    -> UNRESOLVED, say so
# Both numbers are reported whatever happens; the counters are a cross-check on
# the sweep, not a substitute for it.
#
# Counters can be unavailable on a shared cluster. If perf cannot open them the
# job says so and still reports the sweep -- it does not print a zero and let it
# read as "no traffic".
# ===========================================================================
set -u
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
export JAX_PLATFORMS=cpu
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
export LD_LIBRARY_PATH="/sw/spack-levante/gcc-13.4.0-c4mgfk/lib64:/sw/spack-levante/openmpi-4.1.2-mnmady/lib:${LD_LIBRARY_PATH:-}"
OUTDIR="/scratch/b/$USER/legoesm_scaling/threads_bw_${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"
echo "=== node $(hostname) $(date)"

echo "--- B0. the node's own bandwidth ceiling"
cc -O3 -march=native -fopenmp scripts/bench/microbench/stream_triad.c \
   -o "$OUTDIR/stream_triad" || { echo "STREAM COMPILE FAILED"; exit 2; }
srun --overlap -n1 --cpus-per-task=128 --cpu-bind=cores \
     env OMP_NUM_THREADS=128 OMP_PROC_BIND=spread OMP_PLACES=cores \
     "$OUTDIR/stream_triad" 200 | tee "$OUTDIR/stream.txt"

fail=0
for T in 16 32 64 128; do
  echo "=== A. threads=${T} at fixed work $(date +%T)"
  "$PY" scripts/bench/sample_cpu_cores.py --pattern run_cpu_mpi_scaling.py \
        --out "$OUTDIR/cores_t${T}.jsonl" --interval 2 --duration 3000 --per-thread &
  sampler=$!
  srun --overlap -n1 --cpus-per-task="$T" --cpu-bind=cores \
       "$PY" scripts/bench/run_cpu_mpi_scaling.py --grid latlon --mode strong \
       --resolution 1024 --n-levels 40 --precision float32 --physics none \
       --n-warmup 3 --n-timing 20 --output-dir "$OUTDIR" \
       > "$OUTDIR/run_t${T}.log" 2>&1
  rc=$?; kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null
  [ "$rc" -ne 0 ] && { echo "ARM t=${T} FAILED rc=${rc}"; tail -5 "$OUTDIR/run_t${T}.log"; fail=1; continue; }
  "$PY" - "$OUTDIR/run_t${T}.log" "$OUTDIR/cores_t${T}.jsonl" "$T" <<'PYEOF'
import json, re, sys
log, samples, label = sys.argv[1:4]
text = open(log).read()
m0 = re.search(r"\[phase\] timing-start rank=0 epoch=([\d.]+)", text)
m1 = re.search(r"\[phase\] timing-end rank=0 epoch=([\d.]+)", text)
if not (m0 and m1):
    raise SystemExit(f"t={label}: NO PHASE MARKERS")
t0, t1 = float(m0.group(1)), float(m1.group(1))
inside = [r for r in (json.loads(l) for l in open(samples)) if t0 <= r["epoch"] <= t1]
if len(inside) < 5:
    raise SystemExit(f"t={label}: only {len(inside)} in-window samples — REFUSING")
busy = sorted(r["cores_busy"] for r in inside)
step = re.search(r"Timing:\s*([\d.]+)\s*ms/step", text)
print(f"RESULT threads={label} busy_median={busy[len(busy)//2]:.2f} n={len(inside)} "
      f"step_ms={step.group(1) if step else 'NOT FOUND'}")
PYEOF
  [ $? -ne 0 ] && fail=1
done

echo "=== B. DRAM traffic at 128 threads, same work"
if srun --overlap -n1 --cpus-per-task=128 --cpu-bind=cores \
     perf stat -e cache-misses,cache-references,instructions,cycles -x, -o "$OUTDIR/perf.csv" \
     "$PY" scripts/bench/run_cpu_mpi_scaling.py --grid latlon --mode strong \
     --resolution 1024 --n-levels 40 --precision float32 --physics none \
     --n-warmup 3 --n-timing 20 --output-dir "$OUTDIR" > "$OUTDIR/run_perf.log" 2>&1; then
  "$PY" - "$OUTDIR/perf.csv" "$OUTDIR/run_perf.log" "$OUTDIR/stream.txt" <<'PYEOF'
import re, sys
perf, log, stream = sys.argv[1:4]
rows = {}
for line in open(perf):
    parts = line.split(",")
    if len(parts) > 2 and parts[2]:
        try:
            rows[parts[2].strip()] = float(parts[0])
        except ValueError:
            pass   # "<not supported>" / "<not counted>"
misses = rows.get("cache-misses")
if misses is None:
    raise SystemExit("perf did not deliver cache-misses — counters unavailable, "
                     "the sweep above stands alone")
text = open(log).read()
m0 = re.search(r"\[phase\] timing-start rank=0 epoch=([\d.]+)", text)
m1 = re.search(r"\[phase\] timing-end rank=0 epoch=([\d.]+)", text)
# perf counts the WHOLE process -- compile and warmup included -- so the total
# is an UPPER bound on bytes and the implied rate over the timed window is an
# upper bound too. Said out loud rather than quietly divided.
ceiling = float(re.search(r"STREAM_TRIAD_GBs ([\d.]+)", open(stream).read()).group(1))
gb = misses * 64 / 1e9
print(f"PERF cache_misses={misses:.3g} bytes_at_64B={gb:.1f} GB over the WHOLE run "
      f"(compile+warmup included, so the implied timed-phase rate is an UPPER bound) "
      f"stream_ceiling={ceiling:.1f} GB/s")
if m0 and m1:
    print(f"PERF timed_window={float(m1.group(1)) - float(m0.group(1)):.1f} s; "
          f"if all traffic were inside it the rate would be "
          f"{gb / (float(m1.group(1)) - float(m0.group(1))):.1f} GB/s = "
          f"{100 * gb / (float(m1.group(1)) - float(m0.group(1))) / ceiling:.0f}% of STREAM")
PYEOF
else
  echo "PERF UNAVAILABLE — counters could not be opened; the thread sweep stands alone"
fi
[ "$fail" -ne 0 ] && { echo "=== THREADS_BW_INCOMPLETE"; exit 3; }
echo "=== THREADS_BW_DONE $(date)"; echo "OUTDIR=$OUTDIR"
