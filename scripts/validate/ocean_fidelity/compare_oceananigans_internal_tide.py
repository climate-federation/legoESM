"""Internal tide: legoESM vs Oceananigans (the first TOPOGRAPHY + tidal-FORCING case).

A barotropic M2 tide (uniform oscillating body force on u, A₂·sin(ω₂t)) drives a stratified
(constant N²) hydrostatic flow over a Gaussian ridge → radiates internal tides (baroclinic
internal waves). legoESM runs the SAME setup on a Cartesian f-plane C-grid (beta=0) with the
bump as bathymetry (H_bathy_override) and the tidal forcing injected each step; the radiated
vertical-velocity field w(x,z) is pattern-matched to the oracle. 2D x-z (thin y).

Reference: scripts/data/generate_oceananigans_internal_tide_reference.jl.

Run::

    LEGOESM_OCEAN_FIDELITY_OCEANANIGANS_REF=/tmp/ocn_fidelity_ref JAX_ENABLE_X64=1 \
      .venv/bin/python scripts/validate/ocean_fidelity/compare_oceananigans_internal_tide.py [stop_days]
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
from legoesm.ocean.vertical import create_ocean_z_star, create_partial_cell_coordinate

LX = 2.0e6                  # x ∈ (-L, L), L=1000 km → 2000 km total
H = 2.0e3                   # depth [m]
NX = 256
NY = 8                      # thin y (the flow is x-z, y-uniform)
NZ = 64
H0 = 250.0                  # ridge height [m]
WIDTH = 2.0e4               # ridge width [m]
N2 = 1e-4                   # stratification
LATC = -45.0               # f-plane latitude
ALPHA_T = 2.0e-4
T_REF_C = 10.0
G = constants.g
RHO0 = 1000.0
F0 = 2.0 * constants.Omega * np.sin(np.radians(LATC))
DX_M = LX / NX
T2 = 12.421 * 3600.0        # M2 period [s]
OMEGA2 = 2.0 * np.pi / T2
EPS = 0.1
U2 = EPS * OMEGA2 * WIDTH
A2 = U2 * (OMEGA2 ** 2 - F0 ** 2) / OMEGA2


def _hill(x):
    return H0 * np.exp(-x ** 2 / (2.0 * WIDTH ** 2))


def build_setup():
    grid = create_beta_plane_cgrid_geometry(
        NY, NX, dx_m=DX_M, dy_m=DX_M, f0=F0, beta=0.0,    # beta=0 → f-plane
        y_origin_m=-NY * DX_M / 2, x_origin_m=-LX / 2, cartesian_pseudo_lat=True)
    z_star = create_ocean_z_star(n_levels=NZ, H_max=H)
    # Bump bathymetry: depth = H − hill(x), y-uniform.
    xc = -LX / 2 + (np.arange(NX) + 0.5) * DX_M
    Hb = (H - _hill(xc))[None, :] * np.ones((NY, 1))      # (NY, NX)
    # PARTIAL CELLS (default) — bottom-concentrated topography matching Oceananigans'
    # PartialCellBottom (z-LEVEL upper cells + a partial bottom cell); the flow over the
    # ridge displaces isopycnals → internal tide. Pure z-STAR (PARTIAL_CELL=0) is
    # terrain-following (levels follow H_bathy → uniform compression → NO tide).
    if os.environ.get("PARTIAL_CELL", "1") == "1":
        z = create_partial_cell_coordinate(z_star, jnp.asarray(Hb))
    else:
        z = z_star
    cfg = oceananigans_canonical_ocean_config(
        eos_linear=LinearEOSConfig(alpha_T=ALPHA_T, beta_S=0.0),
        g=G, rho_0=RHO0, A_h=0.0, K_h=0.0,
        momentum_advection=os.environ.get("MOM_ADV", "weno5"),
        barotropic_solver="implicit_cn", coriolis_scheme="explicit_ab2",
        bottom_drag_r=0.0, tracer_advection="weno5", weno_smoothness="split")
    cfg = cfg._replace(barotropic_implicit_theta_eta=1.0, barotropic_implicit_theta_pgf=1.0)
    _pgf = os.environ.get("PGF_SCHEME")            # "adcroft" (default) | "smc03"
    if _pgf:
        cfg = cfg._replace(pgf_scheme=_pgf)
    wall = jnp.ones((NY, NX), dtype=jnp.asarray(grid.cos_lat).dtype)
    state = rest_state_latlon_cgrid_ocean(grid, z, land_mask_override=wall,
                                          H_bathy_override=jnp.asarray(Hb),
                                          T_water_init_C=T_REF_C, T_deep=T_REF_C)
    model = LatLonCGridOceanModel(grid, z, cfg)
    return grid, wall, z, state, model


def set_ic(grid, z, state):
    """b = N²z → T = T_ref + b/(g α); u = U₂ (initial barotropic tide); v = 0."""
    z_full = np.asarray(z.z_full_ref)
    mask = np.asarray(state.land_mask.data)
    n_lat, n_lon = mask.shape
    nlev = len(z_full)
    T = np.empty((n_lat, n_lon, nlev))
    for k in range(nlev):
        T[:, :, k] = T_REF_C + (N2 * z_full[k]) / (G * ALPHA_T)
    T = T * mask[:, :, None]
    u = np.full((n_lat, n_lon + 1, nlev), U2)
    uface = np.minimum(mask, np.roll(mask, 1, axis=1))
    uface = np.concatenate([uface, uface[:, :1]], axis=1)
    u = u * uface[:, :, None]
    return state._replace(T=state.T.replace(data=jnp.asarray(T)),
                          u=state.u.replace(data=jnp.asarray(u)))


def amp2dx(a, b):
    """centred pattern correlation of two (x,z) fields."""
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    return float(np.sum(a * b) / d) if d > 0 else float("nan")


def main():
    stop_days = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
    grid, wall, z, state, model = build_setup()
    state = set_ic(grid, z, state)
    dt = float(os.environ.get("DT", "300.0"))
    print(f"[int-tide] {NX}x{NY}x{NZ} f-plane dt={dt}s stop={stop_days}d U2={U2:.4e} "
          f"A2={A2:.4e} T2={T2/3600:.2f}h", flush=True)

    ds = Dataset(os.path.join(os.environ["LEGOESM_OCEAN_FIDELITY_OCEANANIGANS_REF"],
                              "internal_tide",
                              os.environ.get("IT_REF", "internal_tide_zlevel.nc")))
    o_t = np.asarray(ds.variables["times_s"][:])
    # Julia writes (t,x,z) column-major → netCDF4 reads it reversed as (z,x,t).
    # Compare the BUOYANCY ANOMALY b'=b−N²z (isopycnal displacement = the internal-tide
    # signature). This is coordinate-INDEPENDENT, unlike w (legoESM's state.w is the
    # z-star COORDINATE vertical velocity, ≈0 for a barotropic flow over a z-star bump).
    o_bp = np.asarray(ds.variables["bprime"][:])        # (z, x, t)
    zc = np.asarray(ds.variables["z"][:])               # (z,)
    print(f"[oracle] {len(o_t)} snaps, b' shape {o_bp.shape}, "
          f"max|b'_final|={np.abs(o_bp[:, :, -1]).max():.3e}", flush=True)

    umask = np.asarray(state.u.data) != 0.0             # wet u-faces (from IC)
    umask = jnp.asarray(umask.astype(np.asarray(state.u.data).dtype))
    z_full = np.asarray(z.z_full_ref)
    jrow = NY // 2

    @jax.jit
    def step_and_force(s, tt):
        s = model.step(s, dt, surface_forcing=None)
        # M2 tidal body force on u (operator-split, masked to wet faces).
        f = A2 * jnp.sin(OMEGA2 * tt)
        return s._replace(u=s.u.replace(data=(s.u.data + f * dt) * umask))

    o_days = o_t / 86400.0
    nsteps = int(round(stop_days * 86400.0 / dt))
    Hb = np.asarray(state.H_bathy.data)[jrow]           # (n_lon,) local depth
    # WET-cell mask: cells whose reference depth is above the local seafloor (partial
    # cells / z-level: z_ref IS the physical depth). Masks the dry/below-ridge cells whose
    # b' is a static artifact in BOTH codes — leaving the radiated internal-tide signal.
    wet = (z_full[None, :] > -Hb[:, None])              # (n_lon, nlev)
    print("\n  day | lego max|b'|(wet) | b' pattern_corr(wet) | finite", flush=True)
    t = 0.0
    for it in range(1, nsteps + 1):
        state = step_and_force(state, t)
        t += dt
        td = t / 86400.0
        if any(abs(td - od) < dt / 86400.0 / 2 for od in o_days):
            oi = int(np.argmin([abs(td - od) for od in o_days]))
            T = np.asarray(state.T.data)[jrow]          # (n_lon, nlev)
            bp = G * ALPHA_T * (T - T_REF_C) - N2 * z_full[None, :]   # b' = b − N²z_ref
            fin = bool(np.all(np.isfinite(bp[wet])))
            o_bp_oi = o_bp[:, :, oi].T                   # (z,x) → (x, z)
            nxc = min(bp.shape[0], o_bp_oi.shape[0])
            o_bi = np.array([np.interp(-z_full, -zc, o_bp_oi[ix, :]) for ix in range(nxc)])
            m = wet[:nxc]
            corr = amp2dx(bp[:nxc][m], o_bi[m]) if fin else float("nan")
            print(f"  {td:4.1f} | {np.abs(bp[wet]).max():.3e}      | {corr:+.4f}             | {fin}",
                  flush=True)
            if not fin:
                print("  >>> legoESM blew", flush=True); break


if __name__ == "__main__":
    main()
