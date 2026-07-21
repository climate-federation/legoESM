"""Controlled comparison run (v2 — WIND FIX): legoESM nemo_dino_kamm, 180-day mean.

v1 bug: wind_through_step=True recipes SKIP the wind in
apply_dino_lat_lon_surface_forcing, expecting it through model.step(surface_forcing=).
The v1 harness never passed surface_forcing -> NO WIND for 180d. This version builds
the step surface forcing (dino_step_surface_forcing) and threads it into model.step.
Everything else byte-identical to kamm_run180.py.

Usage: kamm_run180_v2.py <nsteps> <out.npz>   (nsteps=5760 for full 180d)
"""
import sys, dataclasses, numpy as np, jax, jax.numpy as jnp
from legoesm.ocean.fidelity.nemo_io import read_nemo_mesh_mask, read_nemo_restart
from legoesm.ocean.fidelity.nemo_state_bridge import bridge_nemo_to_legoesm_topo
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.experiments.dino import (dino_config_for_recipe, dino_lat_lon_model_config,
    dino_lat_lon_state, dino_lat_lon_surface_forcing_arrays, apply_dino_lat_lon_surface_forcing,
    dino_step_surface_forcing)

RECIPE = sys.argv[1]; OUT = sys.argv[2]; DT = 2700.0
NSTEPS = 2880; ACC0 = 1920  # 90-day screen
RUN = "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_TRAJ"
g = read_nemo_mesh_mask(f"{RUN}/mesh_mask.nc", nn_hls=2)
s = read_nemo_restart(f"{RUN}/DINO_00000320_restart.nc", nn_hls=2)  # geometry donor only
br = bridge_nemo_to_legoesm_topo(g, s, periodic_i=True, full_step=True)
ALPHA = float(sys.argv[3]) if len(sys.argv) > 3 else None
cfg = dataclasses.replace(dino_config_for_recipe(RECIPE),
    lon_west_deg=1.0, lon_east_deg=49.0, sill_lon_m_deg=1.0)   # bridge-frame lon fix
st = dino_lat_lon_state(br.geometry, br.z_coord, cfg, land_mask_override=br.land_mask)
# --- HARD topo census gate: legoESM wet cells must equal NEMO tmask exactly ---
import netCDF4 as _nc
_mm = _nc.Dataset(f"{RUN}/mesh_mask.nc"); _H = 2
_tm = (np.moveaxis(np.asarray(_mm["tmask"][0]).squeeze(), 0, -1)[_H:-_H, _H:-_H] > 0.5)
_wet = np.asarray(br.z_coord.is_active) & (np.asarray(st.land_mask.data) > 0.5)[:, :, None]
if not np.array_equal(_wet, _tm):
    _d = int(np.sum(_wet != _tm))
    raise SystemExit(f"TOPO CENSUS FAIL: {_d} cells differ from NEMO tmask — refusing to run")
print(f"topo census OK: {int(_tm.sum())} wet cells == NEMO tmask")          # analytic IC == NEMO usrdef_istate
mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
forcing = dino_lat_lon_surface_forcing_arrays(br.geometry, cfg)
sf = dino_step_surface_forcing(forcing)   # WIND: tau_x/taum into the dycore external-tau block
print(f"slope_scheme={mc.gm_redi.slope_scheme} kappa_GM_max={float(jnp.max(jnp.abs(mc.gm_redi.kappa_GM))):.1f}")
print(f"tau_x[Pa] min/max = {float(jnp.min(sf.tau_x)):.3f}/{float(jnp.max(sf.tau_x)):.3f}")

dyn = jax.jit(lambda st: model.step(st, DT, surface_forcing=sf))  # sf constant (annual tau)

acc = {k: jnp.zeros_like(getattr(st, k).data) for k in ("T", "S", "eta", "u", "v")}
for k in range(NSTEPS):
    st = apply_dino_lat_lon_surface_forcing(st, forcing, br.z_coord, cfg, DT, t_seconds=(k + 1) * DT)
    st = dyn(st)
    if k >= ACC0:
        for f in acc: acc[f] = acc[f] + getattr(st, f).data
    if (k + 1) % 960 == 0:
        Td = np.asarray(st.T.data); m = np.asarray(st.land_mask.data) > 0.5
        us = np.asarray(st.u.data[..., 0])
        print(f"  day {(k+1)*DT/86400:5.1f}  T[{Td[m].min():.1f},{Td[m].max():.1f}] "
              f"usurf[{us.min():.3f},{us.max():.3f}] finite={np.isfinite(Td[m]).all()}", flush=True)
mean = {f: np.asarray(acc[f]) / (NSTEPS - ACC0) for f in acc}
mean["land_mask"] = np.asarray(st.land_mask.data)
np.savez(OUT, **mean)
Td = np.asarray(st.T.data); m = mean["land_mask"] > 0.5
print(f"DONE nsteps={NSTEPS} ({NSTEPS*DT/86400:.1f}d) STABLE={np.isfinite(Td[m]).all() and Td[m].max()<45} -> {OUT}")
