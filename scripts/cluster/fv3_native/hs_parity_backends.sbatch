#!/bin/bash
# Held-Suarez 1-step parity, BOTH backends, gated + cross-checked.
# numpy = the established SPECIFICATION score; jax = the same step with the
# three per-face physics kernels routed through their JAX twins (c2l_ord4,
# held_suarez_tend, fv_update_phys). Same IC / oracle / face map / tolerance,
# so the two numbers are comparable by construction. A final probe asserts the
# jax arm's port fields DIFFER from numpy's (the twins genuinely ran, not a
# silent fallback) yet stay far below the gate -- the 1e-7 gate alone cannot
# tell those apart.
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --job-name=fv3hs_backends
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/hs_backends_%j.log
set -uo pipefail
REPO=${REPO:-/burg-archive/glab/users/pg2328/legoESM_fv3jax}
echo "PINNED sha: $(git -C "$REPO" rev-parse HEAD)"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH=$REPO/packages/core:$REPO/packages/atmosphere:$REPO/src
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu
OUT=/burg-archive/glab/users/pg2328/fv3_duo_gaps/hs_backends_$SLURM_JOB_ID
mkdir -p "$OUT"/{numpy,jax,maps_numpy,maps_jax}; cd "$REPO"
for ARM in numpy jax; do
  echo "===== $ARM ====="
  $PY scripts/validate/fv3_native/full_step_oracle_parity.py --physics held_suarez \
    --backend $ARM --max-rel 1e-7 --save-fields "$OUT/$ARM/hs_fields.npz"
  echo "  $ARM parity rc=$?"
done
$PY scripts/plot/fv3_duo_face_maps.py "$OUT/jax/hs_fields.npz" --out-dir "$OUT/maps_jax" | tail -2
# cross-check: jax port fields differ from numpy at fp scale (<1e-7), never 0
$PY - "$OUT/numpy/hs_fields.npz" "$OUT/jax/hs_fields.npz" <<'PYEOF'
import sys, numpy as np
a=np.load(sys.argv[1]); b=np.load(sys.argv[2])
worst=0.0; moved=False
for k in a.files:
    if not k.startswith("port_"): continue
    x,y=np.asarray(a[k],float),np.asarray(b[k],float)
    m=np.isfinite(x)&np.isfinite(y); d=np.abs(x-y)[m]
    if d.size==0: continue
    sc=np.abs(x[m]).max(); rel=d.max()/sc if sc>0 else 0.0
    worst=max(worst,rel)
    if d.max()>0: moved=True
assert moved, "FAIL: jax port fields BIT-IDENTICAL to numpy -> twins did not run"
assert worst<1e-7, f"FAIL: jax vs numpy port fields too far apart: {worst:.3e}"
print(f"JAX-vs-NUMPY port-field cross-check PASS: worst rel {worst:.3e} (nonzero, <1e-7)")
PYEOF
echo FV3_HS_BACKENDS_DONE
