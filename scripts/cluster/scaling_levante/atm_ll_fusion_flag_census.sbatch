#!/bin/bash -l
# CAN THE COMPILER BE ASKED FOR FEWER, BIGGER KERNELS?
#
# Above 32 devices this lane is granularity-bound: it runs the same ~180
# kernels per step whatever the shard size, their mean duration falls from 35
# to 13 microseconds between 32 and 128 GPUs, and the idle share between them
# doubles. Removing the halo's 26 layout conversions took the step's
# instruction count from 4267 to 4118; the remaining count is the lever.
#
# An earlier arm tried eight XLA COMMUNICATION flags one at a time and found
# them all no-ops or regressions. Fusion is a different subsystem and has not
# been tried on this lane.
#
# This job compiles only. For each flag it dumps the step's post-optimisation
# module and counts instructions and fusions -- deterministic properties of
# the executable, obtained in seconds on two devices, where a timing arm at
# the device count that matters costs a 32-node allocation. A flag that does
# not move the count cannot move the time, so this is the cheap filter that
# decides which flags are worth an allocation at all.
#
# Falsifiable, written before submit:
#   WORTH TIMING: some flag cuts fusion count by >= 10% with no increase in
#     instruction count. That is a real change in granularity.
#   NOT WORTH TIMING: every flag moves fusion count by < 3%. Then the ~180
#     kernels are the program's shape, not a compiler setting, and the lever
#     is model code -- merging stages, not asking XLA more nicely.
#   A flag that CRASHES or refuses is recorded as such, not silently skipped:
#     an unsettable flag is a fact about this jaxlib worth writing down, and
#     this campaign has already mistaken three of those for measurements.
#SBATCH --job-name=ll_fuse
#SBATCH --account=bd1179_gpu
#SBATCH --partition=gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --time=00:28:00
#SBATCH --output=ll_fuse.%j.log
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="$SCRATCH/legoesm_scaling/ll_fuse_j${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
export LEGOESM_JIT_CACHE_DIR=""   # a cached compile does not re-dump

# One flag at a time against the same baseline. The level-leading halo is ON
# in every arm, because that is the program whose granularity is in question.
FLAGS_base=""
FLAGS_prio="--xla_gpu_enable_priority_fusion=true"
FLAGS_noprio="--xla_gpu_enable_priority_fusion=false"
FLAGS_multiout="--xla_gpu_enable_multi_output_fusion=true"
FLAGS_horiz="--xla_gpu_enable_horizontal_loop_fusion=true"
FLAGS_alltoall="--xla_gpu_all_reduce_combine_threshold_bytes=33554432"
FLAGS_cmdbuf="--xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUSTOM_CALL"
FLAGS_dot="--xla_gpu_enable_triton_gemm=true"

run_arm () {  # tag
  local TAG=$1 EXTRA
  eval "EXTRA=\$FLAGS_${TAG}"
  local DUMP="$OUTDIR/hlo_$TAG"
  mkdir -p "$DUMP"
  echo "--- $TAG flags='${EXTRA:-<none>}' $(date +%T)"
  printf '%s\n' "${EXTRA:-<none>}" > "$OUTDIR/${TAG}.flags"
  LEGOESM_LATLON_PACKED_EXCHANGE=1 LEGOESM_LATLON_SPMD_FUSED_HALO=1 \
  LEGOESM_LATLON_HALO_LEVEL_LEADING=1 LEGOESM_LATLON_HALO_NOCOMM=1 \
  LEGOESM_LATLON_HALO_BALLAST=1 JAX_ENABLE_X64=0 \
  XLA_FLAGS="--xla_gpu_shard_autotuning=false --xla_dump_to=$DUMP $EXTRA" \
    timeout --kill-after=30 600 srun --export=ALL --nodes=1 --ntasks=1 \
      --gpus-per-node=2 --gpu-bind=none --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_atm_latlon_spmd_scaling.py \
      --n-devices 2 --mode strong --n-lat 128 --n-lon 256 --nlev 26 \
      --steps 3 --warmup 1 --out "$OUTDIR/${TAG}.jsonl" \
      > "$OUTDIR/${TAG}.log" 2>&1
  local ARM_RC=$?
  if [ "$ARM_RC" -ne 0 ]; then
    echo "  REFUSED or CRASHED (rc=$ARM_RC): $(grep -oiE 'unknown flag[^\"]*|Unrecognized[^\"]*' "$OUTDIR/${TAG}.log" | head -1)"
    echo "REFUSED rc=$ARM_RC" > "$OUTDIR/${TAG}.status"
    return 0
  fi
  echo "OK" > "$OUTDIR/${TAG}.status"
}

for T in base prio noprio multiout horiz alltoall cmdbuf dot; do run_arm "$T"; done

echo "=== instructions and fusions per flag ==="
"$PY" - "$OUTDIR" <<'PYEOF'
import glob, os, re, sys
outdir = sys.argv[1]
rows = []
for tag in ("base", "prio", "noprio", "multiout", "horiz", "alltoall",
            "cmdbuf", "dot"):
    status = os.path.join(outdir, tag + ".status")
    st = open(status).read().strip() if os.path.isfile(status) else "NO STATUS"
    mods = [p for p in glob.glob(os.path.join(outdir, "hlo_" + tag,
                                              "*after_optimizations.txt"))
            if "jit__scan_run" in p]
    if not mods:
        rows.append((tag, st, None, None))
        continue
    text = open(sorted(mods)[-1]).read()
    n_instr = len(re.findall(r"^\s*%?[\w.\-]+\s*=\s", text, re.MULTILINE))
    n_fus = len(re.findall(r"=\s*[^=\n]*?\bfusion\(", text))
    rows.append((tag, st, n_instr, n_fus))
base = next((r for r in rows if r[0] == "base"), None)
print("%-10s %-14s %12s %10s %10s" % ("flag", "status", "instructions",
                                      "fusions", "vs base"))
for tag, st, n_i, n_f in rows:
    if n_i is None:
        print("%-10s %-14s %12s %10s %10s" % (tag, st, "-", "-", "-"))
        continue
    d = ("%+.1f%%" % (100.0 * (n_f - base[3]) / base[3])
         if base and base[3] else "-")
    print("%-10s %-14s %12d %10d %10s" % (tag, st, n_i, n_f, d))
print("\nA flag that does not move the fusion count cannot move the time.")
PYEOF
echo DONE
