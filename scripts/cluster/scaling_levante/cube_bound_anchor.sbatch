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
# ===========================================================================
# Price the CUBED-SPHERE lane. It is the one atmosphere grid with strong
# receipts and NO distance-to-limit number: C768 6->24 GPUs came out at
# efficiency 1.04 and 15.1 GC/s (job 26453782), which says it scales, not how
# much is left. The lat-lon and MPAS lanes both carry measured/bound ratios
# (1.83-1.96 and 3.16-4.47); the cube carries none, so "close to the
# theoretical limit on each grid" is currently unanswerable here.
#
# WHAT A BOUND NEEDS, and what is missing. The lat-lon bench takes a
# single-device same-tile step time plus the measured fabric constants and
# reports measured/bound. The cube bench has no such flags, so the anchor is
# produced here and the ratio is computed from it:
#
#   bound = compute(same tile, 1 GPU) + comm(census x fabric)
#
# ARM 1 is that anchor: C768 at kt=1 on 6 GPUs is the face-sharded lane with
# ONE tile per face -- the smallest same-path configuration -- and its step
# time at a matched per-device tile is the compute term, because a face-local
# step still pays its own halo but no tiling exchange.
# ARM 2 is the 24-GPU tiled rung (kt=2) that the 1.04 receipt came from, rerun
# here so both numbers come from ONE job on ONE node set: the campaign's own
# lesson is that a ratio built from two jobs weeks apart is a confound.
#
# (a) NUMBER: ms/step for both rungs, from the same executable and allocation.
# (b) CONFIRMS the cube is near its limit if the 24-GPU step is within ~1.5x
#     of (anchor/4 + the known 6.65 ms halo term); REFUTES it if the gap is
#     >2.5x, which would put the cube in MPAS territory and make it the next
#     target rather than lat-lon.
# (c) WHY NOT CHEAPER: the compute term is a GPU step time at a real tile.
#     There is no offline substitute; the mesh geometry and the fused step are
#     what set it.
#
# 14 nodes = 56 GPUs, enough for the 24-GPU rung with room; kt is validated
# only for {2,3} (#1360 guard), so kt=2 is the tiled rung and 6-GPU face
# sharding is the anchor.
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

echo "=== [cube] arm 1/2: kt=2 tiled, 24 GPUs (the 1.04-efficiency rung) ==="
JAX_ENABLE_X64=0 srun --ntasks=24 --distribution=block:block \
  --cpu-bind=cores --kill-on-bad-exit=1 \
  "$PY" scripts/bench/bench_cube_tiled_step_scaling.py \
    --kt 2 --resolution 768 --nlev 60 --steps 12 --warmup 3 \
    --multicontroller --out "$OUTDIR/c768_kt2_np24.jsonl" \
  || { echo "[cube] kt2 FAILED"; rc=1; }

echo "=== [cube] arm 2/2: kt=3 tiled, 54 GPUs (the tile-floor rung) ==="
JAX_ENABLE_X64=0 srun --ntasks=54 --distribution=block:block \
  --cpu-bind=cores --kill-on-bad-exit=1 \
  "$PY" scripts/bench/bench_cube_tiled_step_scaling.py \
    --kt 3 --resolution 768 --nlev 60 --steps 12 --warmup 3 \
    --multicontroller --out "$OUTDIR/c768_kt3_np54.jsonl" \
  || { echo "[cube] kt3 FAILED"; rc=1; }

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
if len(rows) == 2:
    (n1, m1, d1), (n2, m2, d2) = rows
    print(f"\nscale {d1}->{d2}: {m1/m2:.3f}x on {d2/d1:.2f}x devices "
          f"= efficiency {(m1/m2)/(d2/d1):.3f}")
else:
    print("\nboth rungs are required for the comparison; not computing a ratio")
PYEOF
echo "DONE rc=$rc"; exit $rc
