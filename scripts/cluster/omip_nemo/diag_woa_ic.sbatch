#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/diagic_%j.out

# The blowup seed is a step-1 spurious PGF (~12 m/s) at ~75 S. Is the WOA IC
# T/S field bad there (interpolation artifact -> spurious horizontal rho
# gradient) = a SETUP bug, or is the IC clean = a core-dynamics PGF bug?
set +e
cd /burg-archive/glab/users/pg2328/legoESM
export PYTHONPATH=/burg-archive/glab/users/pg2328/legoESM/src:$PYTHONPATH
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
echo "=== HOST ==="; hostname; date
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 $PY - <<'EOF'
import numpy as np, jax; jax.config.update("jax_enable_x64", True)
import sys; sys.path.insert(0, "scripts")
from run_omip_core2 import build_tripole
g, z, model, st, H = build_tripole(20, 5500.0, "data/grids/eORCA1.2_mesh_mask.nc",
    woa_init=True, woa_t="data/woa18/woa18_decav_t00_01.nc",
    woa_s="data/woa18/woa18_decav_s00_01.nc")
T = np.asarray(st.T.data)[..., 0]; S = np.asarray(st.S.data)[..., 0]
lat = np.rad2deg(np.asarray(g.lat_T)); m = np.asarray(st.land_mask.data) > 0.5
print("surface T range over ocean [%.2f, %.2f] C" % (T[m].min(), T[m].max()))
print("surface S range over ocean [%.2f, %.2f] PSU" % (S[m].min(), S[m].max()))
# band stats
for lo, hi, name in [(-90,-70,"60-90S"), (-70,-40,"40-70S"), (-10,10,"equator"), (40,70,"40-70N")]:
    b = m & (lat >= lo) & (lat < hi)
    if b.any():
        print("  %-9s: T[%.1f,%.1f] S[%.2f,%.2f] meanS=%.2f n=%d" % (
            name, T[b].min(), T[b].max(), S[b].min(), S[b].max(), S[b].mean(), b.sum()))
# horizontal surface-S gradient magnitude (proxy for PGF seed); find max location
dSx = np.zeros_like(S); dSy = np.zeros_like(S)
dSx[:, 1:-1] = (S[:, 2:] - S[:, :-2]) * 0.5
dSy[1:-1, :] = (S[2:, :] - S[:-2, :]) * 0.5
gradS = np.hypot(dSx, dSy)
gradS = np.where(m, gradS, 0.0)
j, i = np.unravel_index(np.argmax(gradS), gradS.shape)
print("max |grad S| (per-cell) = %.3f PSU at lat %.1f (j=%d,i=%d); S nbhd: %s" % (
    gradS[j, i], lat[j, i], j, i, np.round(S[max(0,j-1):j+2, max(0,i-1):i+2].ravel(), 2)))
# also T gradient max
dTx = np.zeros_like(T); dTy = np.zeros_like(T)
dTx[:, 1:-1] = (T[:, 2:] - T[:, :-2]) * 0.5; dTy[1:-1, :] = (T[2:, :] - T[:-2, :]) * 0.5
gradT = np.where(m, np.hypot(dTx, dTy), 0.0)
jt, it = np.unravel_index(np.argmax(gradT), gradT.shape)
print("max |grad T| = %.2f C at lat %.1f" % (gradT[jt, it], lat[jt, it]))
EOF
echo "rc=$?"; echo "=== DONE ==="
