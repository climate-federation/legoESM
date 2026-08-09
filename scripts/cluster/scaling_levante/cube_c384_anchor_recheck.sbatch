#!/bin/bash -l
#SBATCH --job-name=cube_recheck
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:40:00
#SBATCH --output=cube_recheck.%j.log
# Cube tiled-lane known-answer DISCRIMINATOR. History:
#   2026-07-24 job 26452632 (clean 9051c5126):  C384/L60 kt2 @24 closed = 9.01 ms
#   2026-08-08 job 26804520 (DIRTY 6fc6be9ea): same protocol           = 13.19 ms
# Two candidate causes; this run discriminates them:
#   (a) real ~46% regression merged into the tiled lane since 07-24
#       -> CLEAN main still lands ~13 ms  => bisect next
#   (b) the dropped session's uncommitted (dirty) state slowed it
#       -> CLEAN main returns ~9 ms       => no regression, close item
# (c) cheaper test impossible: GPU step time on the production assembly
#     has no offline proxy (CPU proxy differs; GPU XLA combines CPs 3.2x).
# One arm, one number. C768 arms NOT rerun here (they crashed on
# coordination errors in 26804520; separate issue, do not conflate).
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
if [ -n "$(git status --porcelain 2>/dev/null | grep -v '^??')" ]; then
  echo "REFUSING to run: tree is dirty — the dirty-vs-clean question is the point"
  git status --porcelain | head; exit 2
fi
echo "git_sha=$(git rev-parse --short HEAD) (clean)"
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/cube_recheck_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
JAX_ENABLE_X64=0 srun --ntasks=24 --distribution=block:block \
  --cpu-bind=cores --kill-on-bad-exit=1 \
  "$PY" scripts/bench/bench_cube_tiled_step_scaling.py \
    --kt 2 --resolution 384 --nlev 60 --steps 12 --warmup 3 \
    --closed-loop --multicontroller \
    --out "$OUTDIR/c384_kt2_np24_clean.jsonl" \
  || { echo "recheck FAILED"; exit 1; }
"$PY" - "$OUTDIR/c384_kt2_np24_clean.jsonl" <<'EOF'
import json, sys
d = json.loads(open(sys.argv[1]).readline())
ms = d["steady_median_ms"]
print(f"CLEAN C384/L60 kt2 @24 closed-loop: {ms:.2f} ms/step "
      f"(2026-07-24 anchor 9.01; dirty 2026-08-08 13.19)")
print("VERDICT-DATA: " + ("~anchor -> dirty-state artifact, no regression"
      if ms < 10.5 else "~dirty -> REAL regression merged since 07-24, bisect"
      if ms > 12.0 else "intermediate — neither hypothesis clean, investigate"))
EOF
echo "DONE"
