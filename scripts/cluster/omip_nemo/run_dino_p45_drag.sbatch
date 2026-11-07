#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=dino_p45
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/dino_p45_%j.out

# DINO A/B of the NEMO drag law (P4): the DINO reference runs zdfdrg
# ln_non_lin, so --bottom-drag-scheme nemo_quadratic is its faithful
# drag.  2-yr lat-lon spin-ups, legacy vs nemo_quadratic, with the
# yearly ACC transport as the headline metric (NEMO: 62/66 Sv; the
# P1-P3 campaign measured visbeck 56/92, treguier 55/89 with the
# legacy drag).  MPAS gets the same A/B at 90 d (stability + drift).
set +e
WT=/burg-archive/glab/users/pg2328/legoESM_worktrees/nemo-gaps-p4p5; cd $WT
export PYTHONPATH="$WT/src:$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT:$WT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
echo "=== HOST ==="; hostname; nvidia-smi -L 2>/dev/null | head -1; date
echo "=== worktree ==="; git -C $WT log --oneline -1

run_acc () {  # $1 = bottom_drag_scheme
  JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 GM_CONFIRM=0 $PY - "$1" <<'PYEOF' 2>&1 | grep -vE "FutureWarning|warnings.warn|XLA|cuda|CUDA|Plugin"
import sys, dataclasses, numpy as np, jax
jax.config.update("jax_enable_x64", True)
scheme = sys.argv[1]
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.experiments.dino import (DINOConfig, create_dino_z_star,
    dino_lat_lon_grid, dino_lat_lon_state, dino_lat_lon_model_config,
    dino_lat_lon_surface_forcing_arrays, apply_dino_lat_lon_surface_forcing)
from legoesm.ocean.diagnostics_streamfunction import barotropic_streamfunction
from legoesm.ocean.diagnostics_climate import acc_transport
base = DINOConfig(bottom_drag_scheme=scheme, gm_kappa_scheme="treguier")
z = create_dino_z_star(base); g = dino_lat_lon_grid(base, n_lon=50)
dz = np.asarray(z.dz_ref); lat = np.degrees(np.asarray(g.lat))
def ph(H):
    zt = np.concatenate([[0.0], np.cumsum(dz)[:-1]]); zb = np.cumsum(dz)
    return np.clip(np.minimum(zb[None,None,:], H[...,None]) - zt[None,None,:], 0, dz[None,None,:])
def acc(st):
    u = np.asarray(st.u.data)
    if not np.isfinite(u).all(): return float("nan")
    psi = np.asarray(barotropic_streamfunction(u, ph(np.asarray(st.H_bathy.data)),
                                               np.asarray(st.land_mask.data), g))
    return acc_transport(psi*1e6, lat, drake_lat_south=base.channel_lat_south_deg,
                         drake_lat_north=base.channel_lat_north_deg).transport_Sv
st = dino_lat_lon_state(g, z, base)
model = LatLonCGridOceanModel(g, z, dino_lat_lon_model_config(g, base, physics=True)[0])
frc = dino_lat_lon_surface_forcing_arrays(g, base); dt = base.dt
snap = int(round(365*86400/dt)); accs = []
for k in range(int(round(2*365*86400/dt))):
    st = apply_dino_lat_lon_surface_forcing(st, frc, z, base, dt)
    st = model.step(st, dt=dt)
    if (k+1) % snap == 0:
        accs.append(acc(st))
        if not np.isfinite(accs[-1]): break
print("drag=%-15s ACC/yr: %s" % (scheme,
      " ".join("%.0f" % a if np.isfinite(a) else "NaN" for a in accs)))
PYEOF
}

echo "=== DINO lat-lon 2yr ACC: legacy vs nemo_quadratic (NEMO: 62/66) ==="
run_acc legacy
run_acc nemo_quadratic
echo "=== DONE ==="; date
