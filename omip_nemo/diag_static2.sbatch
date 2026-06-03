#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/diagstat2_%j.out

# Is the WOA-on-20-levels initial DENSITY statically stable, or does the coarse
# vertical interpolation create inversions (dense-over-light) -> a bad-IC /
# convection trigger rather than a dycore bug?
set +e
cd /burg-archive/glab/users/pg2328/legoESM
export PYTHONPATH=/burg-archive/glab/users/pg2328/legoESM/src:$PYTHONPATH
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
echo "=== HOST ==="; hostname; date
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 $PY - <<'EOF'
import numpy as np, jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, sys; sys.path.insert(0, "scripts")
from run_omip_core2 import build_tripole
from legoesm.ocean.eos import compute_ocean_rho
g, z, model, st, H = build_tripole(20, 5500.0, "data/grids/eORCA1.2_mesh_mask.nc",
    woa_init=True, woa_t="data/woa18/woa18_decav_t00_01.nc",
    woa_s="data/woa18/woa18_decav_s00_01.nc")
m = np.asarray(st.land_mask.data) > 0.5
J = jnp.ones(np.asarray(st.land_mask.data).shape)  # 2-D z* Jacobian ~ 1 at rest
rho = np.asarray(compute_ocean_rho(st, z, J))
print("rho shape", rho.shape, "range [%.3f, %.3f] kg/m3" % (np.nanmin(rho), np.nanmax(rho)))
drho = rho[..., 1:] - rho[..., :-1]            # downward diff; >0 = stable
wet = m[..., None] & np.isfinite(drho)
inv = (drho < -1e-3) & wet                     # inversion: lighter below
nwet = max(1, wet.sum())
print("static INVERSIONS (drho<-1e-3 over wet interfaces): %d = %.2f%%" % (inv.sum(), 100*inv.sum()/nwet))
if inv.any():
    print("worst inversion drho = %.3f kg/m3" % np.nanmin(np.where(inv, drho, np.nan)))
    # how many COLUMNS have >=1 inversion
    col_inv = inv.any(axis=-1) & m
    print("ocean columns with >=1 inversion: %d / %d = %.1f%%" % (col_inv.sum(), m.sum(), 100*col_inv.sum()/max(1,m.sum())))
# sample stable column profile (mid-latitude ocean)
print("N2-proxy (drho per level, one column):", np.round(drho[m][0][:8], 4))
EOF
echo "rc=$?"; echo "=== DONE ==="
