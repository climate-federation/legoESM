"""1-year DINO with paper-exact viscosity (no MOM6 stability knobs)."""
import dataclasses, time, numpy as np

from legoesm.ocean.experiments.dino import (
    DINOConfig, create_dino_z_star, dino_lat_lon_grid,
    dino_lat_lon_state, dino_lat_lon_model_config,
    dino_lat_lon_surface_forcing_arrays, apply_dino_lat_lon_surface_forcing,
)
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel

# Paper-exact config: no MOM6 stability knobs
cfg = dataclasses.replace(DINOConfig(), A_h_floor=0.0, A_h_eq_boost=1.0)
print(f"A_h_floor={cfg.A_h_floor}, A_h_eq_boost={cfg.A_h_eq_boost}")

z = create_dino_z_star(cfg)
grid = dino_lat_lon_grid(cfg, n_lon=50)
state = dino_lat_lon_state(grid, z, cfg)
model_cfg, _ = dino_lat_lon_model_config(grid, cfg, physics=True)
model = LatLonCGridOceanModel(grid, z, model_cfg)
forcing = dino_lat_lon_surface_forcing_arrays(grid, cfg)

dt = cfg.dt
n_steps = int(round(365.0 * 86400.0 / dt))
snap_every = int(round(30.0 * 86400.0 / dt))

print(f"DINO paper-exact: {grid.n_lat}x{grid.n_lon}, {z.n_levels} levels, dt={dt}s")
print(f"{n_steps} steps = 365 days, snapshot every {snap_every} steps")
header = f"{'step':>6} {'day':>7} {'|u|':>10} {'|v|':>10} {'|eta|':>10} {'T_max':>7} {'T_min':>7}"
print(header)

t0 = time.time()
for k in range(n_steps):
    state = apply_dino_lat_lon_surface_forcing(state, forcing, z, cfg, dt)
    state = model.step(state, dt=dt)
    if (k + 1) % snap_every == 0 or k == n_steps - 1:
        u = np.asarray(state.u.data)
        v = np.asarray(state.v.data)
        eta = np.asarray(state.eta.data)
        T = np.asarray(state.T.data)
        mask = np.asarray(state.land_mask.data) > 0.5
        t_s = (k + 1) * dt
        print(
            f"{k+1:6d} {t_s/86400:7.2f} "
            f"{np.max(np.abs(u)):10.4e} {np.max(np.abs(v)):10.4e} "
            f"{np.max(np.abs(eta)):10.4e} "
            f"{np.max(T[mask,:]):7.2f} {np.min(T[mask,:]):7.2f}"
        )

print(f"Done. Wall time: {time.time()-t0:.1f}s")
