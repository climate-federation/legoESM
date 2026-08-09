#!/bin/bash -l
#SBATCH --job-name=cube_bound
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=14
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=cube_bound.%j.log
#SBATCH --exclude=l50100
# ===========================================================================
# Price the TILED cubed-sphere lane -- the only cube path that goes past 6
# GPUs -- on the PRODUCTION assembly.
#
# WHY THIS IS A RERUN. The first attempt (job 26772775) omitted --closed-loop
# and so measured the single-shot adapter lane: dynamics-only, state
# re-materialised from pristine input every step. It reported C768/L60 @24 =
# 67.54 ms against the campaign's 14.09 ms for a nominally identical case, a
# 4.8x gap that is a LANE difference, not a regression. --closed-loop is the
# np>6 production assembly (state stays tile-sharded across steps,
# make_tiled_cc_loop, in-stage telescoping mass fixer). Every arm here sets it.
#
# ARM 0 IS A KNOWN-ANSWER CHECK, and it gates the rest. The tiled bench's own
# receipt is C384/L60 kt=2 @24 = 9.01 ms/step (job 26450938). Reproduce that
# before quoting any new cube number: the previous run's headline was wrong
# precisely because nothing checked the harness against a value already known.
#
# ARMS 1-2 then measure C768/L60 at kt=2 (24 GPUs) and kt=3 (54 GPUs) in the
# SAME job on the SAME node set, so the 24->54 ratio is internally controlled
# -- a ladder assembled from jobs weeks apart is this campaign's documented
# confound.
#
# (a) NUMBER: ms/step for all three arms, one executable, one allocation.
# (b) CONFIRMS the harness if arm 0 lands near 9.01 ms; anything far from it
#     means the lane is still wrong and arms 1-2 must not be quoted.
# (c) WHY NOT CHEAPER: these are GPU step times on the production assembly.
#     No offline proxy exists -- and the CPU proxy is known to differ, since
#     GPU XLA combines collective-permutes ~3.2x.
#
# ALSO SETTLED BEFORE SUBMITTING, so it is not this job's question: the halo
# is NOT where this lane's time goes. The compiled module holds 123
# collective-permutes at kt=2; x 29.7 us = 3.65 ms, a few percent of the step.
# Removing every collective could not pay for the work.
#
# 14 nodes = 56 GPUs, enough for the 54-GPU rung; kt is validated only for
# {2,3} (#1360 guard).
#
# SUBMIT:  sbatch scripts/cluster/scaling_levante/cube_bound_anchor.sbatch
# ===========================================================================
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/cube_bound_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
rc=0

echo "=== [cube] arm 0/3: KNOWN-ANSWER CHECK — C384/L60 kt=2 @24, closed loop ==="
# The tiled lane's OWN receipt is 9.01 ms/step here (job 26450938). Reproduce
# it before any new cube number is quoted: the first version of this job
# omitted --closed-loop and measured the single-shot adapter lane instead of
# the np>6 production assembly, which is a 4.8x different path.
JAX_ENABLE_X64=0 srun --nodes=6 --ntasks=24 --ntasks-per-node=4 \
  --distribution=block:block --cpu-bind=cores --kill-on-bad-exit=1 \
  "$PY" scripts/bench/bench_cube_tiled_step_scaling.py \
    --kt 2 --resolution 384 --nlev 60 --steps 12 --warmup 3 \
    --closed-loop --multicontroller --out "$OUTDIR/c384_kt2_np24_check.jsonl" \
  || { echo "[cube] known-answer check FAILED"; rc=1; }

echo "=== [cube] arm 1/3: kt=2 tiled, 24 GPUs, CLOSED LOOP ==="
JAX_ENABLE_X64=0 srun --nodes=6 --ntasks=24 --ntasks-per-node=4 \
  --distribution=block:block --cpu-bind=cores --kill-on-bad-exit=1 \
  "$PY" scripts/bench/bench_cube_tiled_step_scaling.py \
    --kt 2 --resolution 768 --nlev 60 --steps 12 --warmup 3 \
    --closed-loop --multicontroller --out "$OUTDIR/c768_kt2_np24.jsonl" \
  || { echo "[cube] kt2 FAILED"; rc=1; }

# ARM 2 (kt=3 @54, C768) DISABLED 2026-08-09: host OOM — the bench
# builds global C768 state per rank (the #1370-class residency wall,
# fixed for the ocean lane, NOT ported to the cube tiled bench).
# Re-enable after a cube-side jax.default_device(cpu)/sharded-build fix.
echo "=== [cube] receipts ==="
"$PY" - "$OUTDIR" <<'PYEOF'
import glob, json, os, sys
base = sys.argv[1]
rows = []
for path in sorted(glob.glob(os.path.join(base, "*.jsonl"))):
    with open(path) as f:
        line = f.readline().strip()
    if not line:
        print(f"{os.path.basename(path)}: EMPTY -- no receipt")
        continue
    d = json.loads(line)
    ms = d.get("steady_median_ms") or d.get("time_per_step_ms")
    rows.append((os.path.basename(path), ms, d.get("n_devices")))
    print(f"{os.path.basename(path):28s} nd={d.get('n_devices')} "
          f"{ms:8.3f} ms/step")
rows = [r for r in rows if "check" not in r[0]]
if len(rows) == 2:
    (n1, m1, d1), (n2, m2, d2) = rows
    print(f"\nscale {d1}->{d2}: {m1/m2:.3f}x on {d2/d1:.2f}x devices "
          f"= efficiency {(m1/m2)/(d2/d1):.3f}")
else:
    print("\nboth rungs are required for the comparison; not computing a ratio")
PYEOF
echo "DONE rc=$rc"; exit $rc
