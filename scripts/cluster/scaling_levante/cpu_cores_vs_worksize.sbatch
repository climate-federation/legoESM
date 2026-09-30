#!/bin/bash -l
#SBATCH --job-name=cpu_cores_size
#SBATCH --account=bb1596
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=cpu_cores_vs_worksize_%j.log
# ===========================================================================
# DOES A BIGGER LOCAL PROBLEM USE MORE CORES, OR IS THE OCCUPANCY A CEILING?
#
# The top-ranked CPU bottleneck is that a rank keeps only a few of its allowed
# cores busy in steady state, on both lanes and at every rank count, with the
# remaining threads asleep rather than stalled. Two explanations survive and
# they call for opposite work:
#
#   (a) WORK-SIZE LIMITED. The per-op tiles are too small for XLA's CPU runtime
#       to hand out, so more local work would occupy more threads. The fix is
#       then at the decomposition: fewer, fatter ranks.
#   (b) PARTITION-LIMITED. XLA splits each op a fixed way regardless of size,
#       so occupancy is flat in problem size. The fix is then in the emitter or
#       its flags, and the ranks-per-node choice cannot help.
#
# The existing evidence cannot separate them: the earlier sweep varied THREADS
# at fixed work (32 -> 128 cpus gave 2.27x), which both explanations predict.
# This varies WORK at fixed threads, one rank, one node, no collectives, so
# nothing here is confounded by transport.
#
# PRE-REGISTERED READING, adjacent pairs (each is exactly 4x the cells of the
# one before; 512 -> 2048 is 16x, so the pair-wise statement is the one that
# holds):
#   occupancy rises >= 1.5x on BOTH adjacent steps   -> (a) work-size limited
#   occupancy within +-25% across all three          -> (b) partition-limited
#   anything else                                    -> UNRESOLVED, say so
# Time per step is reported beside it: under (a) it should grow sub-linearly in
# the 4x work, under (b) roughly 4x.
#
# WHAT THIS CANNOT SETTLE, stated up front (both reviewers): occupancy counts
# scheduled CPU time, so a thread spinning on a memory stall counts as busy;
# a NUMA or bandwidth effect can therefore move the number without either
# explanation being right. And one rank takes the model's own step function
# while multi-rank runs take the MPI-decomposed one, so this measures the
# single-rank code path, not the multi-rank lane directly.
#
# INSTRUMENT. Occupancy comes from scripts/bench/sample_cpu_cores.py, which is
# unit-tested against a real busy loop. Every earlier CPU-occupancy number on
# these lanes came from a shell one-liner reading the WRONG /proc fields
# (cutime/cstime, i.e. reaped children, identically zero for a threads-only JAX
# process); both reviewers caught it independently, and any pre-2026-09-25
# occupancy figure from that probe is unverified. Samples are windowed on the
# benchmark's own "[phase] timing-start/-end" markers, not on a fixed sleep:
# arm durations differ 16x here, so a fixed skip lands inside compile for the
# small arm and inside steady state for the large one -- which would manufacture
# reading (a) on its own.
# SUBMIT LINE (this worktree has no .venv of its own, so the bare python on a
# compute node has no jax and the job dies in 8 seconds):
#   sbatch --export=ALL,LEGOESM_REPO=$PWD,\
#     LEGOESM_PYTHON=/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python \
#     scripts/cluster/scaling_levante/cpu_cores_vs_worksize.sbatch
# ===========================================================================
set -u
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
export JAX_PLATFORMS=cpu
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
export LD_LIBRARY_PATH="/sw/spack-levante/gcc-13.4.0-c4mgfk/lib64:/sw/spack-levante/openmpi-4.1.2-mnmady/lib:${LD_LIBRARY_PATH:-}"
OUTDIR="/scratch/b/$USER/legoesm_scaling/cores_vs_size_${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"

echo "=== node $(hostname)  $(date)  cpus_per_task=${SLURM_CPUS_PER_TASK}"
echo "=== memory: $(free -g | awk '/^Mem:/{print $2" GiB total, "$7" GiB available"}')"
# What the rank really gets: affinity width and the thread pools, recorded
# rather than assumed (a pool left at 32 would silently re-run the old lane).
srun --overlap -n1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --cpu-bind=cores bash -c '
  echo "[affinity] nproc=$(nproc) taskset=$(taskset -cp $$ 2>/dev/null | sed "s/.*: //")"
  echo "[threads] OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset} XLA_FLAGS=${XLA_FLAGS:-unset}"'

fail=0
for R in 512 1024 2048; do
  CELLS=$(( R * 2 * R * 40 ))
  echo "=== resolution ${R}x$((2*R))x40 = ${CELLS} cells  $(date +%T)"
  "$PY" scripts/bench/sample_cpu_cores.py --pattern run_cpu_mpi_scaling.py \
        --out "$OUTDIR/cores_${R}.jsonl" --interval 2 --duration 3000 &
  sampler=$!
  srun --overlap -n1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --cpu-bind=cores \
       "$PY" scripts/bench/run_cpu_mpi_scaling.py --grid latlon --mode strong \
       --resolution "$R" --n-levels 40 --precision float32 --physics none \
       --n-warmup 3 --n-timing 20 --output-dir "$OUTDIR" \
       > "$OUTDIR/run_${R}.log" 2>&1
  rc=$?
  kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null
  if [ "$rc" -ne 0 ]; then
    echo "ARM ${R} FAILED rc=${rc} — last lines:"; tail -5 "$OUTDIR/run_${R}.log"
    fail=1; continue
  fi
  # Window the samples on the benchmark's own timing phase and refuse a verdict
  # from too few of them.
  "$PY" - "$OUTDIR/run_${R}.log" "$OUTDIR/cores_${R}.jsonl" "$R" <<'PYEOF'
import json, re, sys
log, samples, R = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(log).read()
m0 = re.search(r"\[phase\] timing-start rank=0 epoch=([\d.]+)", text)
m1 = re.search(r"\[phase\] timing-end rank=0 epoch=([\d.]+)", text)
if not (m0 and m1):
    raise SystemExit(f"R={R}: NO PHASE MARKERS in the run log — cannot window; "
                     "the benchmark is older than the marker patch")
t0, t1 = float(m0.group(1)), float(m1.group(1))
rows = [json.loads(l) for l in open(samples)]
inside = [r["cores_busy"] for r in rows if t0 <= r["epoch"] <= t1]
if len(inside) < 5:
    raise SystemExit(f"R={R}: only {len(inside)} samples inside the {t1 - t0:.1f}s "
                     "timing window — REFUSING to report an occupancy")
inside.sort()
med = inside[len(inside) // 2]
if not 0.0 < med <= 256.0:
    raise SystemExit(f"R={R}: occupancy {med} is outside any physical range — "
                     "the instrument is wrong, not the model")
# The driver prints "Timing: 868.14 ms/step", not a json key; the earlier
# pattern matched nothing and printed NOT FOUND beside a real occupancy.
step = re.search(r"Timing:\s*([\d.]+)\s*ms/step", text)
print(f"RESULT R={R} cores_busy_median={med:.2f} n={len(inside)} "
      f"window={t1 - t0:.1f}s threads={rows[-1]['threads']} "
      f"step_ms={step.group(1) if step else 'NOT FOUND'}")
PYEOF
  [ $? -ne 0 ] && fail=1
done
if [ "$fail" -ne 0 ]; then
  echo "=== CORES_VS_SIZE_INCOMPLETE — at least one arm failed or refused"; exit 3
fi
echo "=== CORES_VS_SIZE_DONE $(date)"
echo "OUTDIR=$OUTDIR"
