"""Baroclinic-adjustment comparison: legoESM vs Oceananigans (the §5 precursor).

A 3D STRATIFIED baroclinic-instability twin: an identical buoyancy front
(b = N^2 z + db*ramp(y), thermal-wind balanced) goes unstable and forms eddies.
SAME physics as §5 but CLEAN (no restoring forcing, no wall stress, 48x48x8). Tests
the baroclinic wiring (stratification + hydrostatic PGF + vertical advection +
baroclinic instability) the single-layer cases couldn't. Surface b/u/v compared
vs the reference (chaotic -> early growth + statistical eddy metrics).

Reference: scripts/data/generate_oceananigans_baroclinic_adjustment_reference.jl.

Run::

    LEGOESM_OCEAN_FIDELITY_OCEANANIGANS_REF=/tmp/ocn_fidelity_ref \
      JAX_ENABLE_X64=1 .venv/bin/python \
      scripts/validate/ocean_fidelity/compare_oceananigans_baroclinic_adjustment.py [stop_days]
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax
import jax.numpy as jnp
from netCDF4 import Dataset

from legoesm import constants
from legoesm.grids.latlon import create_regional_latlon_grid
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.eos import LinearEOSConfig
from legoesm.ocean.fidelity.oceananigans_recipe import oceananigans_canonical_ocean_config
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.vertical import create_ocean_z_star

# Oceananigans baroclinic_adjustment.jl parameters.
LX = LY = 1.0e6          # 1000 km
LZ = 1.0e3              # 1 km
N2 = 1e-5
M2 = 1e-7
DY = 1.0e5             # 100 km front width
DB = DY * M2          # 1e-2
EPS_B = 1e-2 * DB
LATC = -45.0          # BetaPlane latitude
NX = NY = 48
NZ = 8
ALPHA_T = 2.0e-4
T_REF_C = 10.0
G = constants.g
RHO0 = 1000.0


def _ramp(y, dy):
    return np.minimum(np.maximum(0.0, y / dy + 0.5), 1.0)


def build_setup():
    # lat/lon spans giving ~LX x LY at LATC.
    R = constants.R_earth
    dlat_deg = np.degrees(LY / R)
    dlon_deg = np.degrees(LX / (R * np.cos(np.radians(LATC))))
    grid, wall = create_regional_latlon_grid(
        NY, NX, LATC - dlat_deg / 2, LATC + dlat_deg / 2,
        lon_west=-dlon_deg / 2, lon_east=dlon_deg / 2, periodic_x=True)
    z = create_ocean_z_star(n_levels=NZ, H_max=LZ)
    cfg = oceananigans_canonical_ocean_config(
        eos_linear=LinearEOSConfig(alpha_T=ALPHA_T, beta_S=0.0),
        g=G, rho_0=RHO0, A_h=0.0, K_h=0.0,
        momentum_advection=os.environ.get("MOM_ADV", "weno5"),
        barotropic_solver=os.environ.get("BARO_SOLVER", "implicit_cn"),
        bottom_drag_r=0.0, tracer_advection="weno5", weno_smoothness="split")
    cfg = cfg._replace(barotropic_implicit_theta_eta=1.0,
                       barotropic_implicit_theta_pgf=1.0)
    state = rest_state_latlon_cgrid_ocean(grid, z, land_mask_override=wall, H_max=LZ,
                                          T_water_init_C=T_REF_C, T_deep=T_REF_C)
    model = LatLonCGridOceanModel(grid, z, cfg)
    return grid, wall, z, state, model


def set_ic(grid, z, state):
    """IC: b = N^2 z + db*ramp(y) (+noise) -> T = T_ref + b/(g a); thermal-wind u."""
    R = constants.R_earth
    lat = np.asarray(grid.lat)                         # (n_lat,)
    y = R * (lat - np.radians(LATC))                   # metres from the front centre
    z_full = np.asarray(z.z_full_ref)                  # (nlev,) <=0
    mask = np.asarray(state.land_mask.data)            # (n_lat, n_lon)
    n_lat, n_lon = mask.shape
    nlev = len(z_full)
    rng = np.random.default_rng(8675309)
    T = np.empty((n_lat, n_lon, nlev))
    ramp = _ramp(y, DY)                                # (n_lat,)
    for k in range(nlev):
        b_col = N2 * z_full[k] + DB * ramp             # (n_lat,)
        T[:, :, k] = T_REF_C + b_col[:, None] / (G * ALPHA_T)
    T = (T + EPS_B / (G * ALPHA_T) * rng.standard_normal(T.shape)) * mask[:, :, None]
    # Thermal wind: f du/dz = -db/dy ; u(z=-H)=0. db/dy = DB*d(ramp)/dy.
    dbdy = DB * np.gradient(ramp, y)                   # (n_lat,)
    f_lat = 2.0 * constants.Omega * np.sin(lat)
    f_safe = np.where(np.abs(f_lat) < 1e-12, 1e-12, f_lat)
    u = np.zeros((n_lat, n_lon + 1, nlev))
    for k in range(nlev):
        u[:, :, k] = (-(1.0 / f_safe) * dbdy * (z_full[k] + LZ))[:, None]
    uface = np.minimum(mask, np.roll(mask, 1, axis=1))
    uface = np.concatenate([uface, uface[:, :1]], axis=1)
    u = u * uface[:, :, None]
    return state._replace(
        T=state.T.replace(data=jnp.asarray(T)),
        u=state.u.replace(data=jnp.asarray(u)))


def main():
    stop_days = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    grid, wall, z, state, model = build_setup()
    state = set_ic(grid, z, state)
    dt = float(os.environ.get("DT", "600.0"))          # 10 min like the oracle wizard start
    nsteps = int(round(stop_days * 86400.0 / dt))
    print(f"[baro-adjust] {NY}x{NX}x{NZ} dt={dt}s stop={stop_days}d ({nsteps} steps); "
          f"IC max|u|={float(jnp.max(jnp.abs(state.u.data))):.4f}", flush=True)

    res = Dataset(os.path.join(os.environ["LEGOESM_OCEAN_FIDELITY_OCEANANIGANS_REF"],
                               "baroclinic_adjustment", "baroclinic_adjustment.nc"))
    o_t = np.asarray(res.variables["times_s"][:])
    o_u = np.asarray(res.variables["u"][:])            # (lat, lon, t) after Julia transpose
    print(f"[oracle] times(d)={np.round(o_t/86400.0,1)} max|u_final|={np.abs(o_u[...,-1]).max():.3e}",
          flush=True)

    step = jax.jit(lambda s: model.step(s, dt, surface_forcing=None))
    print("\n  day | lego max|u| (surface) | oracle max|u| | finite", flush=True)
    o_days = o_t / 86400.0
    t = 0.0
    for it in range(1, nsteps + 1):
        state = step(state)
        t += dt / 86400.0
        if any(abs(t - od) < dt / 86400.0 / 2 for od in o_days):
            su = np.asarray(state.u.data)[:, :, -1]    # surface level
            fin = bool(jnp.all(jnp.isfinite(state.u.data)))
            oi = int(np.argmin([abs(t - od) for od in o_days]))
            print(f"  {t:4.0f} | {np.abs(su).max():.4e}            | "
                  f"{np.abs(o_u[..., oi]).max():.4e}  | {fin}", flush=True)
            if not fin:
                print("  >>> legoESM blew", flush=True); break


if __name__ == "__main__":
    main()
