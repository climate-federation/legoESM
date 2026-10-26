#!/bin/bash -l
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=tripole_solver_ab_%j.log

# Tripole barotropic-solver A/B on the nature-ladder tripole arm
# (synthetic ORCA-fold grid 3072x4352, 75 levels, FLAT 4000 m bottom,
# implicit-CN free surface, fixed-iteration PCG). Every arm passes the same
# flags and environment as nature_ladder.sbatch's tripole lane (16 NCCL
# channels, same env resets, same STEPS/WARMUP values) plus the solver flags
# under test.
#
# PHASE=accuracy: multicontroller at ACC_ND GPUs (default 16 = 4 nodes),
#   same grid and 75 levels, --record-true-residual: ACC_STEPS untimed solves
#   from the initial state, before timing; each one's TRUE global relative residual
#   ||rhs - A eta||/||rhs|| of the returned eta (same unweighted norm for
#   every arm, working precision) lands in the receipt as
#   true_residual{median,max,min,n_solves}. Its timings are perturbed by the
#   probe and are never used. (The old single-process zero-forcing probe ran
#   out of GPU memory on this grid.)
# PHASE=timing: multicontroller at ND_LIST, arms interleaved, run forward then
#   reverse so drift shows as a forward/reverse gap.
#
# PRE-REGISTERED (step 1, single_reduce vs standard, both Jacobi @ 60):
#   number   : steady_median_ms per arm, f32 and f64, at 64/128/192 GPUs,
#              REPS x (forward, reverse) passes.
#   BINDING  : 192 GPUs (largest latency share). 64/128 are reported only.
#   CONFIRMS : single_reduce >= 3% faster at 192 in every pass, both
#              precisions; accuracy-phase probe residual within 2x of
#              standard's AND both within 10x of the converged reference
#              (true_residual median, accuracy phase).
#   REFUTES  : < 1.5% faster at 192 (median over passes), or accuracy fails.
#   1.5-3%   : indeterminate; report the spread, decide nothing.
#   why GPUs : the saved term is 60 allreduce latencies per step across
#              nodes; the 4-GPU A/B (<=2.5%) cannot see it.
#
# ARMS: newline-separated "variant precond degree iters" (degree 0 = not
# chebyshev). X64_LIST: precisions per arm (default "0 1", as the ladder).
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
export JAX_PLATFORMS=cuda,cpu
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh" || exit 1
cd "$REPO" || exit 1
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_shard_autotuning=false"
export NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=16
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
PHASE="${PHASE:?set PHASE=accuracy|timing}"
[[ "$PHASE" == accuracy || "$PHASE" == timing ]] || { echo "PHASE=$PHASE: must be accuracy or timing" >&2; exit 1; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/tripole_solver_ab_${PHASE}_j${SLURM_JOB_ID:-local}}"; mkdir -p "$OUTDIR" || exit 1
export LEGOESM_JIT_CACHE_DIR="$OUTDIR/jitcache_$PHASE"; mkdir -p "$LEGOESM_JIT_CACHE_DIR" || exit 1
NODES="${SLURM_JOB_NUM_NODES:-1}"; PPN=4
ARMS="${ARMS-standard jacobi 0 60
single_reduce jacobi 0 60}"
X64_LIST="${X64_LIST-0 1}"
for x in $X64_LIST; do [[ "$x" == 0 || "$x" == 1 ]] || { echo "X64_LIST entry $x: must be 0 or 1" >&2; exit 1; }; done
[[ -n "${X64_LIST// }" && -n "${ARMS//[[:space:]]}" ]] || { echo "empty ARMS or X64_LIST" >&2; exit 1; }
STEPS="${STEPS:-60}"; WARMUP="${WARMUP:-10}"; REPS="${REPS:-2}"; n_run=0
echo "phase=$PHASE nodes=$NODES outdir=$OUTDIR sha=$(git rev-parse --short HEAD)"
rc=0

live_steps() {  # job steps other than batch/extern; fails if squeue fails
  local out
  out=$(squeue -h -s -j "${SLURM_JOB_ID:?}" -o %i) || return 1
  grep -vE '\.(batch|extern)$' <<< "$out" || true
}
reap_steps() {  # this allocation runs only harness steps
  local s steps t=0
  steps=$(live_steps) || steps="?"
  for s in $steps; do [[ "$s" == "?" ]] || { echo "[reap] scancel -s KILL $s"; scancel -s KILL "$s"; }; done
  while :; do
    steps=$(live_steps) || steps="squeue-failed"
    [[ -z "$steps" ]] && { sleep 15; return 0; }  # slurmctld view lags
    (( t >= 300 )) && { echo "!!! steps still alive after 300 s: $(tr '\n' ' ' <<< "$steps") - stopping harness"; exit 1; }
    for s in $steps; do [[ "$s" == squeue-failed ]] || scancel -s KILL "$s"; done
    sleep 10; t=$((t + 10))
  done
}

arm() {  # arm <nd> <variant> <precond> <degree> <iters> <x64> <pass>
  local nd="$1" var="$2" pre="$3" deg="$4" it="$5" x64="$6" pass="$7"
  local nn=$(( (nd + PPN - 1) / PPN ))
  (( nn > NODES )) && { echo "[skip] d$nd needs $nn nodes"; return 0; }
  local tag="${PHASE}_d${nd}_${var}_${pre}${deg}_M${it}_x${x64}_s${STEPS}_${pass}"
  rm -f "$OUTDIR/${tag}.jsonl"
  export JAX_ENABLE_X64="$x64"
  unset NCCL_P2P_NET_CHUNKSIZE \
        LEGOESM_LATLON_PACKED_EXCHANGE LEGOESM_LATLON_SPMD_FUSED_HALO LEGOESM_MESH_CACHE_DIR \
        LEGOESM_OCEAN_MIXED LEGOESM_VMIX_F32_SOLVE LEGOESM_BAROCLINIC_F32 LEGOESM_ENABLE_TF32
  export NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=16
  local a=(--tripole --mode strong --n-lat 3072 --n-lon 4352 --n-devices "$nd"
           --pcg-fixed-iters "$it" --pcg-variant "$var" --pcg-precond "$pre"
           --out "$OUTDIR/${tag}.jsonl")
  (( deg > 0 )) && a+=(--cheb-degree "$deg")
  local launch
  a+=(--multicontroller --nlev 75)
  # Pinned A_h=1e4 is explicitly unstable on 1.6 km polar cells at dt 600 s
  # (NaN by step 9); mesh-scaled viscosity, user decision 2026-10-19.
  a+=(--A-h-from-mesh)
  if [[ "$PHASE" == accuracy ]]; then
    a+=(--record-true-residual --steps "${ACC_STEPS:-10}" --warmup 2)
  else
    a+=(--steps "$STEPS" --warmup "$WARMUP")
  fi
  launch=(srun --export=ALL --nodes="$nn" --ntasks="$nd" --ntasks-per-node="$PPN"
          --gpus-per-node=4 --gpu-bind=none --cpu-bind=cores
          --cpus-per-task="$(( ${SLURM_CPUS_ON_NODE:?} / PPN ))" --kill-on-bad-exit=1)
  echo "=== arm $tag nn=$nn $(date '+%F %T')"; n_run=$((n_run + 1))
  if ! timeout --kill-after=30 "${ARM_TIMEOUT:-1200}" "${launch[@]}" \
        "$PY" scripts/bench/bench_ocean_latlon_spmd_scaling.py "${a[@]}" < /dev/null; then
    echo "!!! FAILED $tag"; rc=1
    [[ -e "$OUTDIR/${tag}.jsonl" ]] && mv -f "$OUTDIR/${tag}.jsonl" "$OUTDIR/${tag}.failed.jsonl"
    # Killing the srun client leaves a hung step holding its nodes, which
    # blocks every later wider step ("Requested nodes are busy").
    reap_steps
  fi
  sleep 5
}

mapfile -t ARM_LINES < <(grep -v '^[[:space:]]*$' <<< "$ARMS")
for l in "${ARM_LINES[@]}"; do
  read -r v pc dg it extra <<< "$l"
  [[ "$v" =~ ^(standard|single_reduce)$ && -n "$pc" && "$dg" =~ ^[0-9]+$ && "$it" =~ ^[0-9]+$ && -z "$extra" ]] \
    || { echo "bad ARMS line '$l': want 'variant precond degree iters'" >&2; exit 1; }
done
if [[ "$PHASE" == accuracy ]]; then
  for x64 in $X64_LIST; do for l in "${ARM_LINES[@]}"; do arm "${ACC_ND:-16}" $l "$x64" p1; done; done
else
  for ND in ${ND_LIST:-64 128 192}; do
    for x64 in $X64_LIST; do
      for ((r=1; r<=REPS; r++)); do
        for l in "${ARM_LINES[@]}"; do arm "$ND" $l "$x64" "fwd$r"; done
        for ((i=${#ARM_LINES[@]}-1; i>=0; i--)); do arm "$ND" ${ARM_LINES[i]} "$x64" "rev$r"; done
      done
    done
  done
fi
(( n_run == 0 )) && { echo "!!! no arm ran (nodes=$NODES)"; rc=1; }
echo "=== DONE rc=$rc arms=$n_run outdir=$OUTDIR ==="
exit "$rc"
