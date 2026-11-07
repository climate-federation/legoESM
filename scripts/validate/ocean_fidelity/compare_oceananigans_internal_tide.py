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
H0 = float(os.environ.get("IT_H0", "250.0"))   # ridge height [m] (0 → flat bottom)
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
    # FLAT_Y (default ON): Oceananigans uses topology=(Periodic, Flat, Bounded) —
    # a TRUE 2-D x–z domain with NO meridional dimension. legoESM's `Flat`-y analog
    # (set_meridionally_flat) zeros every meridional difference (∂/∂y≡0), matching
    # the oracle exactly: it removes the closed-basin geostrophic adjustment (the
    # walled grid otherwise locks the barotropic tide) AND forbids the spurious 2Δy
    # mode. Required for fidelity; FLAT_Y=0 reverts to the walled (closed-basin) grid.
    from legoesm.grids.halo_latlon import set_meridionally_flat
    set_meridionally_flat(os.environ.get("FLAT_Y", "1") == "1")
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
    # Match the oracle's NO-closure setup: Oceananigans internal_tide uses
    # WENO advection and NO explicit diffusivity. The canonical config defaults
    # to K_v=1e-4 / A_v=1e-3 background mixing, which (with zero-flux BCs) slowly
    # relaxes the stratification toward uniform — over topography that relaxation
    # is horizontally uneven and spins up a spurious flow from rest. Setting
    # K_v=A_v=0 matches the oracle and is the physically correct rest state.
    # NOTE: K_v=0 currently NaNs the implicit tracer solve (a zero-diffusivity
    # degeneracy, tracked separately); default to the canonical background until
    # that is fixed, so the case is at least stable. The oracle truly has none.
    _kv = float(os.environ.get("IT_K_V", "1e-4"))
    _av = float(os.environ.get("IT_A_V", "1e-3"))
    cfg = oceananigans_canonical_ocean_config(
        eos_linear=LinearEOSConfig(alpha_T=ALPHA_T, beta_S=0.0),
        g=G, rho_0=RHO0, A_h=0.0, K_h=0.0, A_v=_av, K_v=_kv,
        # FAITHFUL scheme = flux_form: the oracle's `momentum_advection = WENO()`
        # on a RectilinearGrid is FLUX-FORM (div_𝐯u), NOT WENOVectorInvariant.
        # (Earlier weno5/vector-invariant runs were the WRONG scheme — they spun
        # up a spurious y-velocity, max|v| 2.7 vs 0.70 for flux_form.) flux_form
        # is stable and ~7× closer in rms; the RESIDUAL is grid-scale (2Δx) noise
        # at the partial-cell staircase, not the momentum scheme.
        momentum_advection=os.environ.get("MOM_ADV", "flux_form"),
        # 3rd-order upwind momentum flux (vs the 1st-order "upwind" default): the
        # oracle uses WENO5 momentum, whose LOW DISPERSION the 1st-order scheme
        # badly misses → the internal-tide beam dephases (day-2 b' corr 0.44→0.54
        # at dt=300, →0.61 with dt=150). UP3 is the closest STABLE legoESM
        # match (vector-invariant weno5 momentum blows up here).  The ORACLE-
        # referenced arm: Oceananigans upwind_biased_advective_fluxes.jl:18-24
        # upwind-biases by the sign of the TRANSPORT (not NEMO's advected-
        # velocity pair) -- see UP3_REFERENCE_SELECTOR.
        momentum_flux_scheme=os.environ.get("MOM_FLUX", "oceananigans_up3"),
        barotropic_solver="implicit_cn", coriolis_scheme="explicit_ab2",
        bottom_drag_r=0.0, tracer_advection="weno5", weno_smoothness="split",
        # Oceananigans `Flat`-y topology (record on the config; the global was set
        # above before the rest state so the face masks are flat-aware too).
        meridionally_flat=os.environ.get("FLAT_Y", "1") == "1")
    cfg = cfg._replace(barotropic=cfg.barotropic._replace(barotropic_implicit_theta_eta=1.0, barotropic_implicit_theta_pgf=1.0))
    _pgf = os.environ.get("PGF_SCHEME")            # "adcroft" (default) | "smc03"
    if _pgf:
        cfg = cfg._replace(pgf_scheme=_pgf)
    # FAITHFUL vector-invariant form. The oracle is Oceananigans WENOVectorInvariant,
    # which reconstructs the RELATIVE VORTICITY ζ and multiplies by the transport
    # velocity (flux = v̂·ζᴿ); legoESM's DEFAULT is the potential-vorticity form
    # (reconstruct q=ζ/h ×h·v). reconstruct_zeta=True matches the oracle AND damps
    # the spurious internal-tide growth ~2.4× (1.1e-3→4.6e-4) and localizes it onto
    # the bump's partial-cell staircase flanks. Default ON (faithful); RECON_ZETA=0
    # reverts to the PV form for the probe. The RESIDUAL (rms still ~14× oracle,
    # pattern_corr~0) = spurious vorticity at the partial-cell STAIRCASE STEPS on the
    # bump flanks (curl_vertex_cgrid + 2D vertex mask blind to per-level steps);
    # full fidelity needs Oceananigans' immersed-boundary vorticity over the staircase.
    if os.environ.get("RECON_ZETA", "1") == "1":
        cfg = cfg._replace(vortcor_reconstruct_zeta=True)
    if os.environ.get("ENSTROPHY_METRIC") == "1":
        cfg = cfg._replace(vortcor_enstrophy_metric=True)
    _divs = os.environ.get("DIV_SMOOTH")           # "standard" (Oceananigans) | "split"
    if _divs:
        cfg = cfg._replace(weno_divergence_smoothness=_divs)
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
    # Seed F_slow prev pair (factory defaults barotropic_slow_forcing_ab2 on
    # for the explicit_ab2 x implicit_cn x ab2 combo; no-op otherwise).
    state = model.seed_scan_carry(state, dt)
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
    o_x = np.asarray(ds.variables["x"][:])              # oracle x-centres
    o_bp0 = o_bp[:, :, 0].T                             # (x, z) oracle REST b'

    def _bp_lego(st):
        T = np.asarray(st.T.data)[jrow]
        return G * ALPHA_T * (T - T_REF_C) - N2 * z_full[None, :]

    # The internal tide = the TIME-VARYING anomaly b'(t)−b'(0): subtracting the rest
    # field removes the STATIC partial-cell coordinate offset (which dominates the raw
    # b' and is NOT the tide). Compare on the ORACLE's native z-grid (interpolate
    # legoESM ONTO it) so the oracle's deep near-bump signal is preserved (interpolating
    # the oracle DOWN onto the coarse legoESM levels destroys it), restricted to the
    # bump region |x|<200 km where the radiated tide lives.
    bp0_lego = _bp_lego(state)
    print("\n  day | lego max|Δb'| | oracle max|Δb'| | Δb' pattern_corr(bump) | finite",
          flush=True)
    t = 0.0
    for it in range(1, nsteps + 1):
        state = step_and_force(state, t)
        t += dt
        td = t / 86400.0
        if any(abs(td - od) < dt / 86400.0 / 2 for od in o_days):
            oi = int(np.argmin([abs(td - od) for od in o_days]))
            dl = _bp_lego(state) - bp0_lego              # (n_lon, nlev) lego tidal anomaly
            do = o_bp[:, :, oi].T - o_bp0                # (x, zc) oracle tidal anomaly
            nxc = min(dl.shape[0], do.shape[0])
            fin = bool(np.all(np.isfinite(dl)))
            # interpolate legoESM onto the oracle z-grid (preserve the oracle signal)
            lego_on = np.array([np.interp(zc, z_full[::-1], dl[ix][::-1])
                                for ix in range(nxc)])
            bump = np.abs(o_x[:nxc]) < 2.0e5
            wetb = (zc[None, :] > -Hb[:nxc, None]) & bump[:, None]
            corr = amp2dx(lego_on[wetb], do[:nxc][wetb]) if fin else float("nan")
            print(f"  {td:4.1f} | {np.abs(lego_on[wetb]).max() if fin else -1:.3e}    | "
                  f"{np.abs(do[:nxc][wetb]).max():.3e}     | {corr:+.4f}            | {fin}",
                  flush=True)
            if not fin:
                print("  >>> legoESM blew", flush=True); break


if __name__ == "__main__":
    main()
