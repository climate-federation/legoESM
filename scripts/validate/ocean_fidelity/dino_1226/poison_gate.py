"""GATE: wet-cell tracer tendencies must NOT respond to dry-cell T."""
import dataclasses, numpy as np, jax.numpy as jnp
from legoesm.ocean.fidelity.nemo_io import read_nemo_mesh_mask, read_nemo_restart
from legoesm.ocean.fidelity.nemo_state_bridge import bridge_nemo_to_legoesm_topo
from legoesm.ocean.fidelity.tendency_probe import probe_latlon_cgrid
from legoesm.ocean.experiments.dino import (dino_config_for_recipe, dino_lat_lon_model_config,
    dino_lat_lon_surface_forcing_arrays, dino_step_surface_forcing)
RUN = "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_STEPDUMP"
g = read_nemo_mesh_mask(f"{RUN}/mesh_mask.nc", nn_hls=0)
s = read_nemo_restart(f"{RUN}/DINO_00005760_restart.nc", nn_hls=0)
br = bridge_nemo_to_legoesm_topo(g, s, periodic_i=True, full_step=True)
cfg = dataclasses.replace(dino_config_for_recipe("nemo_dino_kamm_mlf"),
    lon_west_deg=1.0, lon_east_deg=49.0, sill_lon_m_deg=1.0)
mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
sf = dino_step_surface_forcing(dino_lat_lon_surface_forcing_arrays(br.geometry, cfg))
act = np.asarray(br.z_coord.is_active)
pr0 = probe_latlon_cgrid(br.state, br.geometry, br.z_coord, mc, surface_forcing=sf, dt=2700.0)
T999 = jnp.where(jnp.asarray(act), br.state.T.data, 999.0)
S999 = jnp.where(jnp.asarray(act), br.state.S.data, 999.0)
stP = br.state._replace(T=br.state.T.replace(data=T999), S=br.state.S.replace(data=S999))
pr1 = probe_latlon_cgrid(stP, br.geometry, br.z_coord, mc, surface_forcing=sf, dt=2700.0)
worst = 0.0; worst_f = ""
for f in pr0._fields:
    a0, a1 = getattr(pr0, f), getattr(pr1, f)
    if a0 is None or not hasattr(a0, "shape") or np.asarray(a0).shape != act.shape: continue
    d = float(np.max(np.abs(np.where(act, np.asarray(a1) - np.asarray(a0), 0.0))))
    if d > worst: worst, worst_f = d, f
print(f"POISON GATE: worst wet-cell response = {worst:.3e} ({worst_f or 'none'}) -> {'PASS' if worst < 1e-14 else 'FAIL'}")
