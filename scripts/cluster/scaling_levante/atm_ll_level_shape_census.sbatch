#!/bin/bash -l
#SBATCH --job-name=ll_levshape
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --time=00:25:00
#SBATCH --output=ll_levshape.%j.log
# ===========================================================================
# DOES THE LAT-LON STEP EMIT THE SAME KERNELS AT 26, 28 AND 40 LEVELS?
#
# This is the PREMISE of the levels-at-scale timing arm, not a result of it.
# That arm argues: vertical levels make each kernel do more work WITHOUT
# changing how many kernels the step launches, so if extra levels are nearly
# free at a high device count the step is limited by a per-kernel floor.
# GLM's review named the hole: nobody had checked the premise. If the
# compiled program's shape changes with the level count, then either verdict
# of the timing arm is explainable after the fact, which makes it
# unfalsifiable.
#
# Compiling is free -- minutes on two GPUs at a toy grid -- and the
# icosahedral lane's own level scan already showed that a level count changes
# EXTENTS without changing structure there (2292 against 2271 instructions,
# 108 fusions either way). This asks the same question on this lane, and adds
# the count that matters for the alignment control.
#
# A PRIORI GATE:
#   PREMISE HOLDS: the fusion count is identical across 26, 28 and 40, and
#     the instruction count moves by under 2%. Then the timing arm is
#     measuring work per kernel at a fixed kernel count, as it claims.
#   PREMISE FAILS: the fusion count differs. Then the timing arm compares two
#     different programs and its gate must be rewritten before it is read.
#
# SUBMIT: sbatch scripts/cluster/scaling_levante/atm_ll_level_shape_census.sbatch
# ===========================================================================
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="$SCRATCH/legoesm_scaling/ll_levshape_j${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
export LEGOESM_JIT_CACHE_DIR=""   # a cached compile does not re-dump

LEVELS="${LEVELS:-26 28 40}"

run_arm () {  # nlev
  local NL=$1 DUMP="$OUTDIR/hlo_$1"
  mkdir -p "$DUMP"
  echo "--- nlev=$NL $(date +%T)"
  # The halo knobs are pinned to the SAME setting the timing arm runs, so the
  # program counted here is the program timed there.
  LEGOESM_LATLON_PACKED_EXCHANGE=1 LEGOESM_LATLON_SPMD_FUSED_HALO=1 \
  LEGOESM_LATLON_HALO_LEVEL_LEADING=0 LEGOESM_LATLON_HALO_DEDUP=0 \
  LEGOESM_LATLON_HALO_NOCOMM=0 LEGOESM_LATLON_HALO_BALLAST=1 \
  JAX_ENABLE_X64=0 \
  XLA_FLAGS="--xla_gpu_shard_autotuning=false --xla_dump_to=$DUMP" \
    timeout --kill-after=30 600 srun --export=ALL --nodes=1 --ntasks=1 \
      --gpus-per-node=2 --gpu-bind=none --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_atm_latlon_spmd_scaling.py \
      --n-devices 2 --mode strong --n-lat 128 --n-lon 256 --nlev "$NL" \
      --steps 3 --warmup 1 --out "$OUTDIR/l${NL}.jsonl" \
      > "$OUTDIR/l${NL}.log" 2>&1
  local ARM_RC=$?
  echo "rc=$ARM_RC" > "$OUTDIR/l${NL}.status"
  [ "$ARM_RC" -ne 0 ] && echo "  ARM FAILED rc=$ARM_RC (see l${NL}.log)"
  return 0
}

for NL in $LEVELS; do run_arm "$NL"; done

echo "=== compiled shape per level count ==="
LEVELS="$LEVELS" "$PY" - "$OUTDIR" <<'PYEOF'
import glob, os, re, sys
outdir = sys.argv[1]
levels = [int(x) for x in os.environ["LEVELS"].split()]
rows = []
for nl in levels:
    mods = [p for p in glob.glob(os.path.join(outdir, f"hlo_{nl}",
                                              "*after_optimizations.txt"))
            if "jit__scan_run" in p]
    if not mods:
        rows.append((nl, None, None, None))
        continue
    text = open(sorted(mods)[-1]).read()
    n_instr = len(re.findall(r"^\s*%?[\w.\-]+\s*=\s", text, re.MULTILINE))
    n_fus = len(re.findall(r"=\s*[^=\n]*?\bfusion\(", text))
    # The transpose family is counted with the ROOT prefix allowed: an
    # earlier census in this campaign silently skipped every transpose that
    # is the root of its computation, which is most of them.
    n_tr = len(re.findall(r"^\s*(?:ROOT\s+)?%?[\w.\-]+\s*=\s[^=\n]*?"
                          r"\btranspose\(", text, re.MULTILINE))
    rows.append((nl, n_instr, n_fus, n_tr))

print("%6s %14s %10s %12s" % ("nlev", "instructions", "fusions", "transposes"))
for nl, n_i, n_f, n_t in rows:
    print("%6d %14s %10s %12s" % (nl, n_i if n_i is not None else "-",
                                  n_f if n_f is not None else "-",
                                  n_t if n_t is not None else "-"))

good = [r for r in rows if r[1] is not None]
if len(good) != len(levels):
    print("\nNO VERDICT: a level count produced no dumped module")
    sys.exit(4)
fus = {r[0]: r[2] for r in good}
ins = {r[0]: r[1] for r in good}
spread_i = (max(ins.values()) - min(ins.values())) / min(ins.values())
print(f"\nfusion counts {fus}")
print(f"instruction spread across level counts: {spread_i:.2%}")
if len(set(fus.values())) == 1 and spread_i < 0.02:
    print("\nVERDICT: PREMISE HOLDS -- the level count changes extents, not "
          "the program's shape, so a levels arm measures work per kernel at "
          "a fixed kernel count.")
else:
    print("\nVERDICT: PREMISE FAILS -- the compiled program's shape depends "
          "on the level count, so the levels timing arm compares two "
          "different programs. Rewrite its gate before reading it.")
    sys.exit(6)
PYEOF
