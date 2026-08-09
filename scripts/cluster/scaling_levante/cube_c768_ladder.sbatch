#!/bin/bash -l
#SBATCH --job-name=cube_c768
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=27
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --exclude=l50100
#SBATCH --output=cube_c768.%j.log
# C768/L60 closed-loop 24->54 ladder — the LAST open cube measurement.
# History: 26804520 arms crashed (coordination); 26825926/26826088 OOM:
# the closed-loop lane costs 46.5 GB HOST per task at C768 (sacct
# 26826088.1; single-shot was 6.7, C384 closed 15.3) — 4 tasks/node
# exceeds node RAM. Fix here: 2 ranks/node (93 GB/node), BOTH rungs, so
# the 24->54 ratio is internally controlled at matched placement.
# DO NOT cross-compare these against the pinned-4rpn C384 anchor
# (8.91-9.01 ms) — placement is a measured 1.5x confound on this lane.
# Falsifiability, BEFORE submit: 24->54 = 2.25x devices at 147k->65k
# cols/GPU (both far above the ~30k floor).
#   Healthy strong scaling: ratio >= ~1.8x (eff >= 0.8).
#   The single-shot lane's 0.568 eff reproduced on the PRODUCTION
#   assembly: ratio ~1.28 -> a real cube scale-out deficit to attack.
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_XLA_OVERLAP=0
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/cube_c768_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$OUTDIR/jitcache"
rc=0

echo "=== C768 kt=2 @24 closed loop (12 nodes x 2 rpn) ==="
JAX_ENABLE_X64=0 srun --nodes=12 --ntasks=24 --ntasks-per-node=2 \
    --distribution=block:block --cpu-bind=cores --kill-on-bad-exit=1 \
  "$PY" scripts/bench/bench_cube_tiled_step_scaling.py \
    --kt 2 --resolution 768 --nlev 60 --dt 30 --steps 12 --warmup 3 \
    --closed-loop --multicontroller \
    --out "$OUTDIR/c768_kt2_np24.jsonl" || { echo "kt2 FAILED"; rc=1; }

echo "=== C768 kt=3 @54 closed loop (27 nodes x 2 rpn) ==="
JAX_ENABLE_X64=0 srun --nodes=27 --ntasks=54 --ntasks-per-node=2 \
    --distribution=block:block --cpu-bind=cores --kill-on-bad-exit=1 \
  "$PY" scripts/bench/bench_cube_tiled_step_scaling.py \
    --kt 3 --resolution 768 --nlev 60 --dt 30 --steps 12 --warmup 3 \
    --closed-loop --multicontroller \
    --out "$OUTDIR/c768_kt3_np54.jsonl" || { echo "kt3 FAILED"; rc=1; }

"$PY" - "$OUTDIR" <<'EOF' || rc=1
import json, math, sys, pathlib
out = pathlib.Path(sys.argv[1])
t = {}
for tag, nd in (("c768_kt2_np24", 24), ("c768_kt3_np54", 54)):
    rows = [json.loads(l) for l in open(out / f"{tag}.jsonl")]
    r = rows[-1]
    assert r["n_devices"] == nd and r.get("closed_loop") is True
    assert math.isfinite(r["steady_median_ms"]) and r["steady_median_ms"] > 0
    t[tag] = r["steady_median_ms"]
    print(f"{tag}: {t[tag]:8.2f} ms/step")
ratio = t["c768_kt2_np24"] / t["c768_kt3_np54"]
print(f"24->54 speedup {ratio:.3f}x on 2.25x devices "
      f"(eff {ratio/2.25:.3f})")
EOF
echo "DONE rc=$rc"; exit $rc
