#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=autotune_ab
#SBATCH --time=01:30:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G --gres=gpu:1
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/autotune_ab_%j.log
# The autotuner-off fix (--xla_gpu_autotune_level=0, the only setting that
# compiles on several GPUs) has an unmeasured runtime cost and may move bits
# against single-GPU tuned kernels. ONE GPU, ONE allocation, the same
# single-process timing arm twice, XLA_FLAGS the only difference; the two
# final states are diffed bitwise.
set -uo pipefail
REPO=${REPO:-/burg-archive/glab/users/pg2328/legoESM_codex_m8}
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH=$REPO/packages/core:$REPO/packages/atmosphere:$REPO/src
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false LEGOESM_JIT_CACHE_DIR=""
cd "$REPO"
N=${N:-48}; KM=${KM:-10}; NT=${NT:-20}
OUT=/burg-archive/glab/users/pg2328/fv3_duo_gaps/autotune_ab_${SLURM_JOB_ID}; mkdir -p "$OUT"
echo "host $(hostname) gpu $(nvidia-smi -L | head -1) sha $(git rev-parse HEAD) dirty $(git status --porcelain | wc -l) C$N km=$KM n_timed=$NT"
for ARM in default level0; do
  F=""; [ $ARM = level0 ] && F="--xla_gpu_autotune_level=0"
  echo "=== ARM $ARM XLA_FLAGS='$F' ==="
  XLA_FLAGS="$F" $PY scripts/bench/bench_fv3_duo_spmd_scaling.py --mode single --resolution $N --km $KM --n-timed $NT --dump-state "$OUT/$ARM.npz" 2>&1 | grep -E '\[SCALING\]|FAILED|dumped'
done
$PY - "$OUT/default.npz" "$OUT/level0.npz" <<'PYEOF'
import numpy as np, sys
a = np.load(sys.argv[1]); b = np.load(sys.argv[2])
worst = 0.0; n_diff = 0
for k in a.files:
    x, y = a[k], b[k]
    if x.shape != y.shape: print("SHAPE", k); continue
    d = np.abs(x - y); n = int((d > 0).sum()); n_diff += n
    if n: 
        rel = d.max() / max(np.abs(x).max(), 1e-300); worst = max(worst, rel)
        print(f"  {k}: {n} cells differ, max abs {d.max():.3e}, max rel-to-peak {rel:.3e}")
print(f"BITWISE between autotune default and level 0: {'YES' if n_diff == 0 else 'NO'} ({n_diff} cells differ; worst rel-to-peak {worst:.3e})")
PYEOF
echo AUTOTUNE_AB_DONE
