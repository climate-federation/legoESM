#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=w2_ab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=11:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/w2_backend_ab_%j.log
# Williamson-2 duo stepper, NumPy arm vs JAX arm, ONE VARIABLE: --backend.
#
# Everything else is held byte-identical: same n, same dt, same days, same
# k2e_nord, same --ext-bundle, same --oracle-conventions, and d_ext = 0
# (every duo deck resolves D_EXT = 0.0; a nonzero value has already produced
# one retracted claim in this campaign).
#
# Then both npz are scored by the EXISTING calibrated gate,
# w2_duo_oracle_gate.py, which is referenced to
# C48.sw.case2.alpha0.duo.hord6 at day 5.0 with envelope max|v| = 0.0236 and
# rms(v) = 0.0096 m/s, scaled (48/N)^2.  The gate is NOT modified: the whole
# design of this campaign is that it scores the JAX arm unchanged.
set -uo pipefail

REPO=/burg-archive/glab/users/pg2328/_fv3jax_run6
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
OUT=/burg-archive/glab/users/pg2328/fv3_duo_gaps/w2_ab_$SLURM_JOB_ID
mkdir -p "$OUT"

export PYTHONPATH=$REPO/packages/core:$REPO/packages/atmosphere:$REPO/src:$REPO/scripts/validate
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1

echo "host: $(hostname)"
cd "$REPO"
echo "PINNED sha: $(git rev-parse HEAD)"

N=24
DT=450
DAYS=5
COMMON="--n $N --dt $DT --days $DAYS --d-ext 0.0 --k2e-nord 2 --ext-bundle --oracle-conventions"
echo "COMMON FLAGS: $COMMON   (the ONLY difference between arms is --backend)"

for BK in numpy jax; do
  echo "=============== ARM: $BK ==============="
  /usr/bin/time -v $PY $REPO/scripts/validate/fv3_native/run_duo_stepper_w2.py \
      $COMMON --backend $BK --out "$OUT/w2_${BK}.npz" 2>&1 | tail -40
  echo "arm $BK rc=$?"
done

echo "=============== A/B: the two arms against each other ==============="
$PY - "$OUT/w2_numpy.npz" "$OUT/w2_jax.npz" <<'PYEOF'
import sys, numpy as np
a = np.load(sys.argv[1]); b = np.load(sys.argv[2])
print("keys:", sorted(set(a.files) & set(b.files)))
for k in ("backend", "d_ext", "k2e_nord", "git_sha"):
    if k in a.files:
        print(f"  {k}: numpy={a[k]}  jax={b[k]}")
va, vb = a["v"], b["v"]
assert va.shape == vb.shape, (va.shape, vb.shape)
for i, t in enumerate(a["times_days"]):
    d = np.abs(va[i] - vb[i])
    s = max(np.abs(va[i]).max(), 1e-30)
    print(f"  day {float(t):5.2f}  max|dv| {d.max():.6e}   rel {d.max()/s:.6e}"
          f"   numpy max|v| {np.abs(va[i]).max():.6e}"
          f"   jax max|v| {np.abs(vb[i]).max():.6e}")
PYEOF

echo "=============== EXISTING CALIBRATED GATE, both arms ==============="
for BK in numpy jax; do
  echo "--- gate: $BK"
  $PY $REPO/scripts/validate/fv3_native/w2_duo_oracle_gate.py \
      "$OUT/w2_${BK}.npz" --res $N 2>&1 | tail -20
  echo "gate $BK rc=$?"
done

echo "OUT=$OUT"
echo W2_AB_DONE
