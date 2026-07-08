"""Spherical baroclinic-adjustment: legoESM lat-lon C-grid vs Oceananigans LatitudeLongitudeGrid.

The SPHERE analog of compare_oceananigans_baroclinic_adjustment.py: identical buoyancy-front
physics, but on a real lat-lon grid (cos(lat) metric) instead of the Cartesian beta-plane.
Tests whether the full-velocity vertical-advection fix (which closed the Cartesian precursor)
ALSO holds on legoESM's lat-lon C-grid in the baroclinic-eddy regime — the spherical-metric
test the Cartesian case could not do. Region: ~1000 km patch centred at φ₀=-45°, periodic in
λ, walls in φ. Coriolis = full f (matsuno_split, the matched vertex-f = HydrostaticSpherical
Coriolis EnstrophyConserving).

Reference: scripts/data/generate_oceananigans_spherical_baroclinic_reference.jl.

Run::

    LEGOESM_OCEAN_FIDELITY_OCEANANIGANS_REF=/tmp/ocn_fidelity_ref JAX_ENABLE_X64=1 \
      .venv/bin/python scripts/validate/ocean_fidelity/compare_oceananigans_spherical_baroclinic.py [stop_days]
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

PHI0 = -45.0
DPHI = 9.0                              # ~1000 km in latitude
DLAMBDA = DPHI / np.cos(np.radians(45.0))  # ~1000 km in lon at φ₀
LZ = 1.0e3
NZ = 8
NX = NY = 48
N2 = 1e-5
M2 = 1e-7
DY = 1.0e5
DB = DY * M2
EPS_B = 1e-2 * DB
ALPHA_T = 2.0e-4
T_REF_C = 10.0
G = constants.g
RHO0 = 1000.0


def _ramp(y, dy):
    return np.minimum(np.maximum(0.0, y / dy + 0.5), 1.0)


def build_setup():
    grid, wall = create_regional_latlon_grid(
        NY, NX, PHI0 - DPHI / 2, PHI0 + DPHI / 2,
        lon_west=-DLAMBDA / 2, lon_east=DLAMBDA / 2, periodic_x=True)
    z = create_ocean_z_star(n_levels=NZ, H_max=LZ)
    cfg = oceananigans_canonical_ocean_config(
        eos_linear=LinearEOSConfig(alpha_T=ALPHA_T, beta_S=0.0),
        g=G, rho_0=RHO0, A_h=0.0, K_h=0.0,
        momentum_advection=os.environ.get("MOM_ADV", "weno9"),
        barotropic_solver=os.environ.get("BARO_SOLVER", "implicit_cn"),
        # matched vertex-f Coriolis = HydrostaticSphericalCoriolis EnstrophyConserving.
        coriolis_scheme=os.environ.get("CORIOLIS_SCHEME", "matsuno_split"),
        bottom_drag_r=0.0, tracer_advection="weno7", weno_smoothness="split")
    cfg = cfg._replace(
        barotropic=cfg.barotropic._replace(barotropic_implicit_theta_eta=1.0, barotropic_implicit_theta_pgf=1.0),
        # The §5 closure: full-velocity vertical momentum advection (default True via the
        # recipe; the env hook lets us A/B the perturbation form on the sphere).
        weno_vertadv_full_velocity=os.environ.get("VERTADV_FULL", "1") == "1")
    state = rest_state_latlon_cgrid_ocean(grid, z, land_mask_override=wall, H_max=LZ,
                                          T_water_init_C=T_REF_C, T_deep=T_REF_C)
    model = LatLonCGridOceanModel(grid, z, cfg)
    return grid, wall, z, state, model


def set_ic(grid, z, state):
    """Same b-front IC as the Cartesian case, from REST; y = R·(lat − φ₀)."""
    lat = np.asarray(grid.lat)
    y = constants.R_earth * (lat - np.radians(PHI0))   # metres from the front centre
    z_full = np.asarray(z.z_full_ref)
    mask = np.asarray(state.land_mask.data)
    n_lat, n_lon = mask.shape
    nlev = len(z_full)
    rng = np.random.default_rng(8675309)
    T = np.empty((n_lat, n_lon, nlev))
    ramp = _ramp(y, DY)
    for k in range(nlev):
        b_col = N2 * z_full[k] + DB * ramp
        T[:, :, k] = T_REF_C + b_col[:, None] / (G * ALPHA_T)
    T = (T + EPS_B / (G * ALPHA_T) * rng.standard_normal(T.shape)) * mask[:, :, None]
    return state._replace(T=state.T.replace(data=jnp.asarray(T)))


def main():
    stop_days = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    grid, wall, z, state, model = build_setup()
    state = set_ic(grid, z, state)
    dt = float(os.environ.get("DT", "600.0"))
    # Seed F_slow prev pair (factory defaults barotropic_slow_forcing_ab2 on
    # for the explicit_ab2 x implicit_cn x ab2 combo; no-op otherwise).
    state = model.seed_scan_carry(state, dt)
    nsteps = int(round(stop_days * 86400.0 / dt))
    print(f"[sph-baro] {NY}x{NX}x{NZ} lat-lon φ₀={PHI0} dt={dt}s stop={stop_days}d "
          f"vertadv_full={model.config.weno_vertadv_full_velocity}", flush=True)

    res = Dataset(os.path.join(os.environ["LEGOESM_OCEAN_FIDELITY_OCEANANIGANS_REF"],
                               "spherical_baroclinic", "spherical_baroclinic.nc"))
    o_t = np.asarray(res.variables["times_s"][:])
    o_u = np.asarray(res.variables["u"][:])
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
            su = np.asarray(state.u.data)[:, :, 0]
            fin = bool(jnp.all(jnp.isfinite(state.u.data)))
            oi = int(np.argmin([abs(t - od) for od in o_days]))
            print(f"  {t:4.0f} | {np.abs(su).max():.4e}            | "
                  f"{np.abs(o_u[..., oi]).max():.4e}  | {fin}", flush=True)
            if not fin:
                print("  >>> legoESM blew", flush=True); break


if __name__ == "__main__":
    main()
