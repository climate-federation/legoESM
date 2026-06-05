#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/diagwoa_%j.out

# Is the WOA-on-20-levels initial state statically UNSTABLE (density inversions
# -> convective blowup) or statically stable (=> the day-1 blowup is baroclinic
# PGF / dynamics)? Decides the fix direction.
set +e
cd /burg-archive/glab/users/pg2328/legoESM
export PYTHONPATH=/burg-archive/glab/users/pg2328/legoESM/src:$PYTHONPATH
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
echo "=== HOST ==="; hostname; date
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 $PY - <<'EOF'
import numpy as np, sys
sys.path.insert(0, "scripts")
from run_omip_core2 import build_tripole
import jax; jax.config.update("jax_enable_x64", True)
g, z, model, st, H = build_tripole(20, 5500.0, "data/grids/eORCA1.2_mesh_mask.nc",
                                   woa_init=True,
                                   woa_t="data/woa18/woa18_decav_t00_01.nc",
                                   woa_s="data/woa18/woa18_decav_s00_01.nc")
T = np.asarray(st.T.data); S = np.asarray(st.S.data); m = np.asarray(st.land_mask.data) > 0.5
print("T finite:", np.isfinite(T).all(), "S finite:", np.isfinite(S).all())
print(f"T range [{np.nanmin(T):.2f},{np.nanmax(T):.2f}] C; S [{np.nanmin(S):.2f},{np.nanmax(S):.2f}] PSU")
# density per level via ocean EOS; check static stability (rho increasing down)
from legoesm.ocean.eos import compute_ocean_rho
zc = np.asarray(z.z_ref) if hasattr(z, "z_ref") else None
try:
    rho = np.asarray(compute_ocean_rho(st.T.data, st.S.data, z))
except Exception as e:
    print("compute_ocean_rho(T,S,z) failed:", e)
    import inspect; print("sig:", inspect.signature(compute_ocean_rho)); rho = None
if rho is not None:
    drho = rho[..., 1:] - rho[..., :-1]      # downward difference (level k+1 - k)
    inv = (drho < -1e-6) & m[..., None]       # denser over lighter = inversion
    print(f"rho range [{np.nanmin(rho):.2f},{np.nanmax(rho):.2f}]")
    print(f"static INVERSIONS (rho decreasing downward) over ocean: "
          f"{inv.sum()} cells = {100*inv.sum()/max(1,(m[...,None]*np.ones_like(drho)).sum()):.2f}% of wet interfaces")
    # worst inversion magnitude
    if inv.any(): print(f"worst inversion drho = {np.nanmin(np.where(inv, drho, np.nan)):.3f} kg/m3")
EOF
echo "rc=$?"; echo "=== DONE ==="
