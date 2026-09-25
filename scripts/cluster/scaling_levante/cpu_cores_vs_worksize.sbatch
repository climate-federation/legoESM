#!/bin/bash -l
#SBATCH --job-name=cpu_cores_size
#SBATCH --account=bb1596
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:45:00
#SBATCH --output=cpu_cores_vs_worksize_%j.log
# ===========================================================================
# DOES A BIGGER LOCAL PROBLEM USE MORE CORES, OR IS 4.5 A CEILING?
#
# The top-ranked CPU bottleneck is that each rank keeps ~4.5 of its 32 allowed
# cores busy in steady state (measured 2026-09-24, both lanes, every rank
# count; 27 threads asleep rather than stalled). Two explanations survive and
# they call for opposite work:
#
#   (a) WORK-SIZE LIMITED. The per-op tiles are too small for XLA's CPU
#       runtime to hand out, so more local work would occupy more threads.
#       Then the fix is at the decomposition: fewer, fatter ranks.
#   (b) PARTITION-LIMITED. XLA splits each op a fixed way regardless of size,
#       so cores-busy is flat. Then the fix is in the emitter/flags, and the
#       ranks-per-node choice cannot help.
#
# This separates them the cheapest way available: ONE rank, ONE node, all 128
# cpus, the same lane and code path, with the local problem grown 4x twice.
# No collectives are involved, so nothing here is confounded by transport.
#
# PRE-REGISTERED READING (write it down before the run, per compute
# discipline). Cores busy is (utime+stime) delta over wall delta from
# /proc/<pid>/stat, steady state only, sampled every 5 s:
#   cores busy rises by >= 1.5x from R=512 to R=2048  -> (a) work-size limited
#   cores busy stays within +-25% across the three     -> (b) partition-limited
#   anything else                                      -> report as unresolved
# Time per step is recorded next to it: under (a) the step should grow
# sub-linearly in the 4x work, under (b) it should grow ~4x.
#
# A cheaper test cannot answer it: the existing thread-count sweep varied the
# THREADS at fixed work and found more threads help (2.27x from 32 to 128
# cpus), which is consistent with BOTH explanations. Varying the WORK at fixed
# threads is the discriminating direction and has never been run.
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
CLK=$(getconf CLK_TCK)

echo "=== node $(hostname)  $(date)  cpus=${SLURM_CPUS_PER_TASK}"
for R in 512 1024 2048; do
  echo "=== resolution ${R}x$((2*R))x40  $(date +%T)"
  ( srun --overlap -n1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --cpu-bind=cores \
        "$PY" scripts/bench/run_cpu_mpi_scaling.py --grid latlon --mode strong \
        --resolution "$R" --n-levels 40 --precision f32 --physics none \
        --n-warmup 3 --n-timing 20 --output-dir "$OUTDIR" \
        > "$OUTDIR/run_${R}.log" 2>&1 ) &
  runpid=$!
  # steady-state core sampling: skip the first 90 s (import, trace, compile)
  sleep 90
  prev=""; : > "$OUTDIR/cores_${R}.txt"
  while kill -0 $runpid 2>/dev/null; do
    for p in $(pgrep -u "$USER" -f "run_cpu_mpi_scaling.py"); do
      read -r _ _ _ _ _ _ _ _ _ _ _ _ _ ut st _ < <(sed 's/^.*) //' "/proc/$p/stat" 2>/dev/null) || continue
      now=$(date +%s); cpu=$(( (ut+st)*1000/CLK ))
      if [ -n "$prev" ]; then
        dt=$(( now - ${prev%%:*} )); dc=$(( cpu - ${prev##*:} ))
        [ "$dt" -gt 0 ] && echo "$now cores_busy $(( dc / dt / 10 )).$(( (dc*10/dt) % 10 ))" \
            >> "$OUTDIR/cores_${R}.txt"
      fi
      prev="$now:$cpu"
    done
    sleep 5
  done
  wait $runpid || echo "ARM ${R} FAILED rc=$?"
  echo "--- cores busy samples (R=$R), median of the last 10:"
  tail -10 "$OUTDIR/cores_${R}.txt" | awk '{print $3}' | sort -n | awk '{a[NR]=$1} END{print (NR? a[int((NR+1)/2)] : "none")}'
  grep -hE "ms/step|per_step|step_ms" "$OUTDIR/run_${R}.log" | tail -2
done
echo "=== CORES_VS_SIZE_DONE $(date)"
echo "OUTDIR=$OUTDIR"
