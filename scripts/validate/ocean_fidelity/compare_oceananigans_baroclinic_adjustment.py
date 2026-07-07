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
from legoesm.grids.latlon import create_beta_plane_cgrid_geometry
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
# Resolution env-overridable so the precursor can be pushed to eddy-RESOLVING (to
# reproduce the §5 over-energization, which needs better-resolved eddies than 48x48x8).
NX = NY = int(os.environ.get("BARO_N", "48"))
NZ = int(os.environ.get("BARO_NZ", "8"))
ALPHA_T = 2.0e-4
T_REF_C = 10.0
G = constants.g
RHO0 = 1000.0
# BetaPlane(latitude=-45): f(y) = f0 + beta*y, exactly as Oceananigans builds it
# from the Cartesian RectilinearGrid + BetaPlane (NOT a spherical sin(lat) grid).
F0 = 2.0 * constants.Omega * np.sin(np.radians(LATC))
BETA = 2.0 * constants.Omega * np.cos(np.radians(LATC)) / constants.R_earth
DX_M = LX / NX                                         # square cells, 1000km/48
Y_ORIGIN = -LY / 2.0                                   # front centred at domain mid (y=0)


def _ramp(y, dy):
    return np.minimum(np.maximum(0.0, y / dy + 0.5), 1.0)


def build_setup():
    # FAITHFUL grid: Oceananigans baroclinic_adjustment uses a Cartesian
    # RectilinearGrid (x,y in metres) + BetaPlane(-45) -- NOT a lat-lon spherical
    # grid. The beta-plane C-grid matches it exactly: uniform Cartesian metric
    # (cos_lat==1, no spherical convergence) and f=f0+beta*y at each stagger point.
    # Topology (Periodic, Bounded, Bounded): zonally re-entrant (no E/W walls) +
    # N/S v-faces auto-walled by the barotropic solver. This removes the spherical
    # metric terms that feed the 2dx grid-scale Coriolis runaway on the latlon grid.
    grid = create_beta_plane_cgrid_geometry(
        NY, NX, dx_m=DX_M, dy_m=DX_M, f0=F0, beta=BETA,
        y_origin_m=Y_ORIGIN, x_origin_m=0.0, cartesian_pseudo_lat=True)
    z = create_ocean_z_star(n_levels=NZ, H_max=LZ)
    wall = jnp.ones((NY, NX), dtype=jnp.asarray(grid.cos_lat).dtype)   # all ocean
    cfg = oceananigans_canonical_ocean_config(
        eos_linear=LinearEOSConfig(alpha_T=ALPHA_T, beta_S=0.0),
        g=G, rho_0=RHO0, A_h=0.0, K_h=0.0,
        # weno9 momentum + weno7 tracer = W9V, matching the regenerated oracle
        # (WENOVectorInvariant(vorticity_order=9) + WENO(order=7)).
        momentum_advection=os.environ.get("MOM_ADV", "weno9"),
        barotropic_solver=os.environ.get("BARO_SOLVER", "implicit_cn"),
        # FAITHFUL defaults to Oceananigans WENOVectorInvariant + BetaPlane +
        # ImplicitFreeSurface: matsuno_split = HydrostaticSphericalCoriolis
        # EnstrophyConserving (f & ζ co-located at the FF vertex); split vorticity
        # = VelocityStencil; standard divergence = OnlySelfUpwinding.
        coriolis_scheme=os.environ.get("CORIOLIS_SCHEME", "matsuno_split"),
        bottom_drag_r=0.0, tracer_advection="weno7",
        weno_smoothness=os.environ.get("WENO_SMOOTH", "split"),
        # Faithful no-backstop eddy stack knobs (default = recipe behaviour). With
        # BARO_SOLVER=explicit_substep these route via from_flat into BarotropicConfig.
        barotropic_slow_forcing_ab2=os.environ.get("BARO_SLOW_AB2", "0") == "1",
        barotropic_diffusion_alpha=float(os.environ.get("BARO_ALPHA", "0.05")),
        barotropic_time_filter=(os.environ.get("BARO_FILTER") or None))
    cfg = cfg._replace(
        barotropic=cfg.barotropic._replace(barotropic_implicit_theta_eta=1.0, barotropic_implicit_theta_pgf=1.0),
        weno_divergence_smoothness=os.environ.get("WENO_DIV_SMOOTH", "standard") or None,
        vortcor_reconstruct_zeta=os.environ.get("VORTCOR_ZETA", "0") == "1",
        vortcor_enstrophy_metric=os.environ.get("VORTCOR_METRIC", "0") == "1",
        weno_vertadv_full_velocity=os.environ.get("VERTADV_FULL", "1") == "1")
    state = rest_state_latlon_cgrid_ocean(grid, z, land_mask_override=wall, H_max=LZ,
                                          T_water_init_C=T_REF_C, T_deep=T_REF_C)
    model = LatLonCGridOceanModel(grid, z, cfg)
    return grid, wall, z, state, model


def set_ic(grid, z, state):
    """IC matching Oceananigans `set!(model, b=bᵢ)` EXACTLY: b = N^2 z + db*ramp(y)
    (+noise) -> T = T_ref + b/(g a), and START FROM REST (u=v=0). The jet spins up
    by geostrophic adjustment during the run, like the oracle -- imposing the
    thermal-wind jet at t=0 instead would start legoESM with MORE energy than the
    oracle ever has (0.97 m/s vs the oracle's 0.72 m/s day-6 jet) and inflate the
    over-energization measurement. On the Cartesian beta-plane grid `lat` is pinned
    to 0, so y is recovered from the cell index: y_c = y_origin + (j+1/2) dy."""
    z_full = np.asarray(z.z_full_ref)                  # (nlev,) <=0
    mask = np.asarray(state.land_mask.data)            # (n_lat, n_lon)
    n_lat, n_lon = mask.shape
    nlev = len(z_full)
    y = Y_ORIGIN + (np.arange(n_lat) + 0.5) * DX_M     # metres from front centre (y=0)
    rng = np.random.default_rng(8675309)
    T = np.empty((n_lat, n_lon, nlev))
    ramp = _ramp(y, DY)                                # (n_lat,)
    for k in range(nlev):
        b_col = N2 * z_full[k] + DB * ramp             # (n_lat,)
        T[:, :, k] = T_REF_C + b_col[:, None] / (G * ALPHA_T)
    T = (T + EPS_B / (G * ALPHA_T) * rng.standard_normal(T.shape)) * mask[:, :, None]
    # Rest start (u=v=0) -- the oracle sets ONLY b; the jet adjusts up in the run.
    return state._replace(T=state.T.replace(data=jnp.asarray(T)))


def main():
    stop_days = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    grid, wall, z, state, model = build_setup()
    state = set_ic(grid, z, state)
    dt = float(os.environ.get("DT", "600.0"))          # 10 min like the oracle wizard start
    # Seed F_slow prev pair (factory defaults barotropic_slow_forcing_ab2 on
    # for the explicit_ab2 x implicit_cn x ab2 combo; no-op otherwise).
    state = model.seed_scan_carry(state, dt)
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
            su = np.asarray(state.u.data)[:, :, 0]     # surface level (legoESM k=0 = surface)
            fin = bool(jnp.all(jnp.isfinite(state.u.data)))
            oi = int(np.argmin([abs(t - od) for od in o_days]))
            print(f"  {t:4.0f} | {np.abs(su).max():.4e}            | "
                  f"{np.abs(o_u[..., oi]).max():.4e}  | {fin}", flush=True)
            if not fin:
                print("  >>> legoESM blew", flush=True); break


if __name__ == "__main__":
    main()
