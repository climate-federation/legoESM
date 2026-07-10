"""Reference boundary-layer LES on the plane NH dycore: GABLS1 + Wangara.

Two canonical ABL intercomparison cases, run as 3D large-eddy simulation on the
compressible plane non-hydrostatic dycore with the **dynamic Smagorinsky** SGS
closure (Germano 1991 / Lilly 1992; ``smagorinsky_dynamic=True``):

* **GABLS1** (Beare et al. 2006, BLM 118; Cuxart et al. 2006) — stable boundary
  layer. Neutral θ=265 K mixed layer below 100 m, capping inversion 0.01 K/m
  above; geostrophic ug=8 m/s, f=1.39e-4 (73°N); surface cooled at 0.25 K/hr
  (prescribed T_s, bulk neutral-log drag). The reference SBL equilibrates to a
  ~150-200 m depth with a super-geostrophic low-level jet near its top.

* **Wangara** Day 33 (Clarke 1971; Yamada & Mellor 1975) — convective boundary
  layer. θ≈277 K, prescribed cosine surface sensible-heat flux peaking 0.216 K
  m/s at 13:00 LT (+ moisture flux), f for −34.5°S. The CBL grows to ~1-1.5 km
  by mid-afternoon with a well-mixed θ and convective velocity w*~1-2 m/s.

Vertical transport is operator-split into an IMPLICIT mixing-length PBL column
(`_apply_pbl_column`): each step the dycore runs first (advection + acoustic +
horizontal dynamic-Smagorinsky SGS + Coriolis), then a jitted, time-traced
backward-Euler Thomas solve applies vertical diffusion of u/v/θ'/q_v with the
surface flux as the bottom boundary condition (bulk neutral-log drag + bulk T_s
flux for GABLS, prescribed cosine fluxes for Wangara). K_v is a Louis-type
`l²·|∂U/∂z|·f(Ri)` closure with a small floor (damps the 2Δz vertical mode) and
a modest cap (so it does NOT parametrise away the resolved CBL convection — that
is left to the dycore's resolved eddies + 3D dynamic SGS). The implicit solve is
unconditionally stable, replacing an earlier explicit single-level source that
injected a 2Δz mode (GABLS) and blew up under strong convective heating (Wangara).

Usage
-----
.. code-block:: bash

   JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 .venv/bin/python \\
       scripts/run/run_les_plane.py --case gabls1 \\
       --nx 32 --ny 32 --nlev 64 --dx 12.5 --H 400 --dz-sfc 6.0 \\
       --dt 0.1 --hours 2 --output results/les_gabls1

   (dz_sfc<H/nlev so the stretched grid builds; dt keeps the horizontal
   acoustic CFL c·dt/(n·dx)<1 at the default --n-acoustic-substeps 8 — raise
   dt only if you also raise n. These are also the per-case ``gabls1`` defaults,
   so a bare ``--case gabls1`` runs without overrides.)
"""
from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.atmosphere.dynamics.compressible_euler import CompressibleEulerConfig
from legoesm.atmosphere.dynamics.compressible_euler_plane import (
    PlaneCompressibleEulerModel,
    make_flat_plane_terrain_metric,
    make_rest_state,
)
from legoesm.core.field import Field
from legoesm.timestepping.tridiagonal import thomas_solve
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_stretched_height_coordinate

# x64 is enabled in main() unless --f32 is given (production GPU LES runs in
# float32 — RTX 50xx fp64 is ~1/64 of fp32 — and float32 keeps grid+hc+state in
# ONE dtype so the time loop can be a single jitted lax.scan; see build()).
_DEFAULT_X64 = "--f32" not in sys.argv
jax.config.update("jax_enable_x64", _DEFAULT_X64)


def _cast_height_coord(hc, dtype):
    """Cast every floating array field of a HeightCoordinate to ``dtype`` so the
    vertical coordinate matches the grid + state dtype (the dycore otherwise
    silently upcasts the whole state to the hc dtype, defeating float32)."""
    fields = {}
    for name in hc._fields:
        val = getattr(hc, name)
        if isinstance(val, jnp.ndarray) and jnp.issubdtype(val.dtype, jnp.floating):
            fields[name] = val.astype(dtype)
        else:
            fields[name] = val
    return hc._replace(**fields)


# --------------------------------------------------------------------------- #
# Case definitions: θ_ref(z), geostrophic wind, Coriolis, surface forcing.    #
# --------------------------------------------------------------------------- #
_KAPPA_VK = constants.kappa_von_karman


def gabls1_theta_ref(z):
    """GABLS1 initial θ(z): 265 K below 100 m, +0.01 K/m capping inversion."""
    return jnp.where(z > 100.0, 265.0 + 0.01 * (z - 100.0), 265.0)


def neutral_theta_ref(z):
    """Neutral ABL: well-mixed θ=290 K capped by a strong inversion above 0.7 of
    the domain (the lid limits BL growth so the shear-driven turbulence reaches a
    quasi-steady, resolution-converged state — the cleanest test of resolved SGS
    transport, no buoyancy production/destruction)."""
    return jnp.where(z > 700.0, 290.0 + 0.02 * (z - 700.0), 290.0)


def ekman_theta_ref(z):
    """Ekman (Andren et al. 1994) initial θ(z): neutral (well-mixed θ=273.15 K)
    capped by an inversion above 0.6 of the domain so the wind-driven neutral BL
    reaches a quasi-steady depth (the classic neutral Ekman-spiral test) rather
    than growing without bound."""
    return jnp.where(z > 600.0, 273.15 + 0.01 * (z - 600.0), 273.15)


def wangara_theta_ref(z):
    """Wangara Day-33 initial θ(z): ~277 K mixed layer, capping inversion above
    ~1 km (simplified sounding — the convective growth is flux-driven)."""
    return jnp.where(z > 1000.0, 277.0 + 0.008 * (z - 1000.0), 277.0)


_CASES = {
    "neutral": dict(
        theta_ref_fn=neutral_theta_ref,
        f_c=1.0e-4, ug=10.0, vg=0.0, z0=0.1,
        moist=False,
        # 1 km domain, dx=20 m (well-resolved neutral ABL ~ Δ/h ~ 0.02).
        nx=32, ny=32, nlev=50, dx=20.0, H=1000.0, dz_sfc=10.0,
        # dt=0.05 s: the LES regime runs with the acoustic w-damping OFF
        # (--off-centering 0 --si-w-filter 0), so the vertical acoustic CFL needs
        # a smaller step. With the damping ON the resolved w' is annihilated each
        # substep and the BL turbulence decays to laminar (max|w|~mm/s); with it
        # OFF, w' grows and sustains (the resolved eddies the LES is meant to
        # carry). See docs/les_plane_turbulence_notes.md.
        dt=0.05, hours=1.0, log_wind=True,
    ),
    "ekman": dict(
        theta_ref_fn=ekman_theta_ref,
        f_c=1.0e-4, ug=10.0, vg=0.0, z0=0.1,
        moist=False,
        # Andren (1994) neutral Ekman BL: f=1e-4 (~45°N), ug=10 m/s, z0=0.1 m.
        # 1 km domain; like the neutral case it runs with the acoustic w-damping
        # OFF and a log-wind IC so resolved shear production starts at t=0.
        nx=32, ny=32, nlev=64, dx=20.0, H=1000.0, dz_sfc=10.0,
        dt=0.05, hours=1.0, log_wind=True,
    ),
    "gabls1": dict(
        theta_ref_fn=gabls1_theta_ref,
        f_c=1.39e-4, ug=8.0, vg=0.0, z0=0.1,
        moist=False,
        # defaults (Beare 2006 domain is 400³ m @ dx=6.25 m; coarsened here).
        # dz_sfc=6.0 not 6.25: create_stretched_height_coordinate requires
        # dz_sfc·nlev < H (room to stretch); 6.25·64=400=H exactly is rejected,
        # so 6.0·64=384<400 gives a near-uniform 6 m surface layer.
        # dt=0.1 not 0.5: horizontal acoustic CFL = c·(dt/n)/dx with the default
        # n_acoustic_substeps=8 and dx=12.5 ⇒ 340·0.5/(8·12.5)=1.70>1 (step-1
        # blow-up); 0.1 ⇒ CFL=0.34. Raise dt only with a matching --n bump.
        nx=32, ny=32, nlev=64, dx=12.5, H=400.0, dz_sfc=6.0,
        dt=0.1, hours=2.0,
    ),
    "wangara": dict(
        theta_ref_fn=wangara_theta_ref,
        f_c=2.0 * constants.Omega * float(np.sin(np.deg2rad(-34.5))),
        ug=-8.0, vg=0.0, z0=0.1,
        moist=True,
        nx=48, ny=48, nlev=50, dx=100.0, H=2500.0, dz_sfc=20.0,
        # dt=0.5 not 1.0: dz_sfc=20 m is the tight direction (dx=100 m is loose).
        # At dt=1.0 the vertical acoustic CFL c·(dt/n)/dz_sfc ≈ 340·(1.0/8)/20 ≈
        # 2.1 > 1 at the default n_acoustic_substeps=8 → step-1 blow-up
        # (max|w|~316). dt=0.5 runs clean (verified f32 GPU); raise dt only with a
        # matching --n-acoustic-substeps bump.
        dt=0.5, hours=4.0,
    ),
}

# Cases with a surface-flux branch in ``_apply_pbl_column``.  Kept in sync
# with ``_CASES`` (enforced by tests/unit/test_run_les_plane_cli.py) so a new
# case cannot silently fall through to the Wangara flux branch: an unknown
# selection raises ValueError at the column's entry (dispatch doctrine).
_PBL_COLUMN_CASES = ("neutral", "ekman", "gabls1", "wangara")


def _implicit_vertical_diffusion(phi_asc, K_iface, dz_asc, dzc, dt,
                                 flux_sfc=None, drag_sfc=None):
    """Backward-Euler implicit vertical diffusion of a column field.

    ``phi_asc`` ascending in z (index 0 = surface), shape (...,n). ``K_iface``
    diffusivity at the n-1 interior interfaces (index k = between k, k+1).
    Surface (k=0 bottom) BC: a prescribed flux ``flux_sfc`` [φ·m/s, +ve into the
    column] and/or an implicit drag ``drag_sfc`` [m/s] (bottom flux = −drag·φ_0).
    No-flux top. Unconditionally stable ⇒ no CFL limit on the vertical mixing /
    surface forcing (fixes the explicit single-level blow-up). Thomas solve."""
    n = phi_asc.shape[-1]
    # Interface coupling coefficient r_k = dt·K_k / (dz_k_center) for k=0..n-2.
    r = dt * K_iface / dzc                       # (..., n-1)
    z = jnp.zeros(phi_asc.shape[:-1] + (1,))
    # Sub-diagonal a (coupling to k-1): a_k = -r_{k-1}/dz_k for k>=1.
    a = -jnp.concatenate([z, r], axis=-1) / dz_asc
    # Super-diagonal c (coupling to k+1): c_k = -r_k/dz_k for k<=n-2.
    c = -jnp.concatenate([r, z], axis=-1) / dz_asc
    b = 1.0 - a - c
    d = phi_asc
    if drag_sfc is not None:                     # implicit surface drag on φ_0
        add0 = dt * drag_sfc / dz_asc[..., 0]
        b = b.at[..., 0].add(add0)
    if flux_sfc is not None:                      # prescribed surface flux into φ_0
        d = d.at[..., 0].add(dt * flux_sfc / dz_asc[..., 0])
    return thomas_solve(a, b, c, d)


def _apply_pbl_column(state, t, *, case, hc, z0, dt, ug, vg):
    """Implicit mixing-length PBL column: vertical diffusion of u, v, θ', q_v
    with the surface flux as the bottom boundary condition.

    Replaces the crude single-level explicit surface source (which injected an
    undamped 2Δz vertical mode and blew up under strong convective heating).
    K_v is a Louis-type mixing-length closure ``l²·|∂U/∂z|·f(Ri)`` with a small
    floor; the implicit Thomas solve is unconditionally stable. The dynamic
    Smagorinsky still provides the resolved HORIZONTAL SGS inside the dycore."""
    # ``case`` is a static Python string (partial-bound), so this guard runs at
    # trace time — an unknown case raises instead of silently running the
    # Wangara prescribed-flux branch (dispatch doctrine: raise on unknown).
    if case not in _PBL_COLUMN_CASES:
        raise ValueError(
            f"Unknown LES case {case!r} for the PBL surface-flux column; "
            f"expected one of {_PBL_COLUMN_CASES}.")
    g = constants.g
    # Ascending-z views (state stores top-down: index 0 = top).
    u = state.u.data[..., ::-1]
    v = state.v.data[..., ::-1]
    thp = state.theta_prime.data[..., ::-1]
    theta_ref = hc.theta_ref[::-1]
    z_full = hc.z_full[::-1]                       # ascending (index 0 = surface)
    dz = hc.dz[::-1]
    theta = theta_ref + thp                        # full θ, ascending

    uc, vc = u, v                                  # A-grid approx for the column
    # Interface (k between k,k+1) gradients + Richardson number.
    dzc = z_full[1:] - z_full[:-1]                 # (n-1,) centre spacing
    dUdz = jnp.sqrt(
        ((uc[..., 1:] - uc[..., :-1]) / dzc) ** 2
        + ((vc[..., 1:] - vc[..., :-1]) / dzc) ** 2) + 1e-6
    dthdz = (theta[..., 1:] - theta[..., :-1]) / dzc
    theta_i = 0.5 * (theta[..., 1:] + theta[..., :-1])
    Ri = g / theta_i * dthdz / (dUdz ** 2)
    # Louis-type stability function: stable (Ri>0) shuts mixing off by Ri_c=0.25;
    # unstable (Ri<0) enhances. Mixing length l = κz/(1+κz/λ), λ=40 m.
    z_i = 0.5 * (z_full[1:] + z_full[:-1])
    lmix = _KAPPA_VK * z_i / (1.0 + _KAPPA_VK * z_i / 40.0)
    # Stable (Ri>0): Ri_c=0.25 shut-off (sustains the SBL gradient). Unstable
    # (Ri<0): keep f_stab≈1 (NOT the big √(1−16Ri) enhancement) so the column
    # provides only baseline surface-layer mixing — super-adiabatic instability
    # is then left to trigger RESOLVED convection + the dycore 3D dynamic SGS,
    # rather than being parametrised away (which killed the Wangara CBL eddies).
    f_stab = jnp.where(
        Ri > 0.0, jnp.clip(1.0 - Ri / 0.25, 0.0, 1.0) ** 2, 1.0)

    z1 = z_full[0]
    Umag = jnp.sqrt(uc[..., 0] ** 2 + vc[..., 0] ** 2 + 0.01)
    Cd = (_KAPPA_VK / jnp.log(z1 / z0)) ** 2
    drag = Cd * Umag                               # implicit momentum drag [m/s]
    u_star = jnp.sqrt(jnp.maximum(Cd, 0.0)) * Umag  # friction velocity (τ=u*²)

    # SURFACE-LAYER-ONLY mixing (the LES fix for laminar collapse).  The dycore
    # 3D SGS (``sgs_vertical_diffusion=True``) already carries the RESOLVED
    # vertical SGS flux; a SECOND full-column Louis K here double-counted the
    # mixing, homogenised the column and killed the resolved eddies (max|w|~cm/s,
    # wvar~1e-4 — a 1D column, not an LES).  Confine the column's role to its
    # ONLY irreplaceable job: depositing the surface stress / heat / moisture
    # flux through the UNDER-RESOLVED surface layer (the lowest O(10) cells),
    # decaying to ~0 above it so resolved convection + the dycore 3D SGS — not a
    # K-profile — set the mixed-layer structure.  K_sl = κ u_* z f(Ri)·e^(−z/h_sl)
    # with a small near-surface floor for the 2Δz mode; NO 20 m²/s cap.
    h_sl = 40.0                                    # surface-layer decay scale [m]
    # THIN surface-layer coupling: K_sl deposits the surface stress/flux through
    # only the under-resolved lowest cells (decay e^(−z/40 m)), so it does NOT
    # spread the surface drag through a deep layer and spin the whole column down
    # — the RESOLVED eddies carry momentum/heat aloft (that is the LES). A deep
    # K_sl (h_sl=100 m) collapsed the geostrophic wind to <1 m/s.
    K_sl = _KAPPA_VK * u_star[..., None] * z_i * f_stab * jnp.exp(-z_i / h_sl)
    # Small UNIFORM background K: damps the column 2Δz mode (timescale dz²/2K ~
    # 5-10 s) without homogenising the resolved large eddies, which mix on the
    # turnover time ~h/w* ≫ h²/2K for this K. The implicit Thomas solve is
    # unconditionally stable; this floor only needs to suppress the 2Δz growth
    # the dycore 3D SGS (explicit, CFL-limited) cannot. Capped well below the
    # old 20 m²/s that flattened the whole column into a laminar 1D profile.
    K_floor = 0.3
    K_iface = jnp.minimum(K_sl + K_floor, 10.0)

    new_u = _implicit_vertical_diffusion(u, K_iface, dz, dzc, dt, drag_sfc=drag)
    new_v = _implicit_vertical_diffusion(v, K_iface, dz, dzc, dt, drag_sfc=drag)

    if case in ("neutral", "ekman"):
        # Pure shear-driven ABL: ZERO surface buoyancy flux. θ' is only mixed
        # (no surface source), so turbulence is generated solely by the resolved
        # surface shear — the cleanest validation of the SGS momentum transport.
        new_th = _implicit_vertical_diffusion(thp, K_iface, dz, dzc, dt,
                                              flux_sfc=0.0)
        upd = dict(theta_prime=state.theta_prime.replace(data=new_th[..., ::-1]))
    elif case == "gabls1":
        theta_s = 265.0 - 0.25 * t / 3600.0
        F_th = Cd * Umag * (theta_s - theta[..., 0])      # bulk kinematic w'θ'
        new_th = _implicit_vertical_diffusion(thp, K_iface, dz, dzc, dt,
                                              flux_sfc=F_th)
        upd = dict(theta_prime=state.theta_prime.replace(data=new_th[..., ::-1]))
    elif case == "wangara":                        # Wangara prescribed fluxes
        t_hr = 9.0 + t / 3600.0
        cos_t = jnp.cos((t_hr - 13.0) / 11.0 * jnp.pi)
        F_th = 0.216 * cos_t
        F_qv = 2.29e-5 * cos_t
        new_th = _implicit_vertical_diffusion(thp, K_iface, dz, dzc, dt,
                                              flux_sfc=F_th)
        qv = state.tracers.data[..., 0][..., ::-1]
        new_qv = _implicit_vertical_diffusion(qv, K_iface, dz, dzc, dt,
                                              flux_sfc=F_qv)
        new_tr = state.tracers.data.at[..., 0].set(
            jnp.clip(new_qv[..., ::-1], 0.0, None))
        upd = dict(theta_prime=state.theta_prime.replace(data=new_th[..., ::-1]),
                   tracers=state.tracers.replace(data=new_tr))
    else:
        # A case admitted by the entry guard but lacking a surface-flux branch
        # (i.e. added to _PBL_COLUMN_CASES without implementing its fluxes).
        raise ValueError(
            f"LES case {case!r} has no surface-flux branch in "
            "_apply_pbl_column; implement its surface forcing.")
    return state._replace(
        u=state.u.replace(data=new_u[..., ::-1]),
        v=state.v.replace(data=new_v[..., ::-1]),
        **upd)


def build(args):
    spec = _CASES[args.case]
    nx, ny, nlev = args.nx, args.ny, args.nlev
    dx, H, dz_sfc = args.dx, args.H, args.dz_sfc
    f_c = spec["f_c"]
    ug, vg, z0 = spec["ug"], spec["vg"], spec["z0"]

    # Single consistent dtype across grid + hc + state so the dycore does not
    # upcast (and so the time loop can be one jitted lax.scan): float32 for
    # production GPU runs (--f32), float64 for the bit-exact science default.
    dtype = jnp.float32 if getattr(args, "f32", False) else jnp.float64
    grid = create_plane_grid(nx=nx, ny=ny, nlev=nlev, dx=dx, dy=dx,
                             coriolis_mode="f_plane", f0=f_c, dtype=dtype)
    hc = create_stretched_height_coordinate(
        nlev, H=H, dz_sfc=dz_sfc, theta_ref_fn=spec["theta_ref_fn"],
        p_sfc=1.0e5)
    hc = _cast_height_coord(hc, dtype)
    # Geostrophic reference wind for the SAM-form Coriolis (f·(u−ug0)).
    ug0 = jnp.full(nlev, ug, dtype=dtype)
    vg0 = jnp.full(nlev, vg, dtype=dtype)
    hc = hc._replace(u_geo0=ug0, v_geo0=vg0)
    tm = make_flat_plane_terrain_metric(grid, hc)

    cfg = CompressibleEulerConfig(
        use_coriolis=True,
        semi_implicit_acoustic=True,
        substep_horizontal_acoustic=True,
        n_acoustic_substeps=args.n_acoustic_substeps,
        # Off-centre the SI vertical acoustic solve to damp the vertical 2Δz
        # (Nyquist) computational mode — without it (β=0, fully centred) the
        # single-level surface-flux forcing piles into an undamped 2Δz column
        # mode once the stable BL turbulence collapses. β=0.2 ≈ the RCE value.
        acoustic_off_centering=args.off_centering,
        si_w_vertical_filter_nu=args.si_w_filter,
        # LES dynamic Smagorinsky.
        smagorinsky_cs=0.17,                 # fallback / initial value
        smagorinsky_dynamic=not args.static_sgs,
        # Bou-Zeid et al. (2005) scale-dependent dynamic (LASD) — the jax-alfa
        # oracle closure. cs_max=1.0 to match the oracle's [0,1] C_s² mask.
        smagorinsky_scale_dependent=args.scale_dependent,
        smagorinsky_dynamic_cs_max=1.0 if args.scale_dependent else 0.3,
        smagorinsky_prandtl=1.0,
        # Keep the Mason κz mixing-length cap ON even for LASD. It is a NUMERICAL
        # necessity here: the dycore's 3D SGS vertical flux (_vertical_K_diffusion_full)
        # is EXPLICIT forward-Euler, so it is CFL-limited (K < 0.5·dz²/dt ≈ 45 m²/s
        # near the surface for dz≈3 m, dt=0.1 s). Without the cap the uncapped
        # near-wall LASD K (l_m=Cs·Δ in the high-shear first cell) overshoots that
        # limit and the explicit vertical SGS blows up at ~0.08 h. The κz cap is
        # only active in the lowest O(1) cells where κz < Cs·Δ; LASD's β scale
        # correction still governs the resolved SGS through the rest of the column.
        smagorinsky_wall_damping=True,
        smagorinsky_delta_max=1.0e30,        # fine grid: no horizontal Δ cap
        sgs_vertical_diffusion=True,   # 3D dynamic SGS for resolved eddies; column adds surface coupling
        # weak biharmonic for the 2Δ acoustic mode; small sponge at the top.
        hyperdiff_coeff=args.hyperdiff,
        hyperdiff_rho_coeff=args.hyperdiff,
        hyperdiff_w_coeff=args.hyperdiff,
        sponge_coeff=0.05, sponge_width=0.2 * H, sponge_w_only=True,
        sponge_profile_shape="sam_rational",
        fix_mass=True, anchor_mass_to_initial=True,
    )
    model = PlaneCompressibleEulerModel(grid, hc, tm, cfg)

    # IC: rest state (hydrostatic θ_ref) + geostrophic mean wind + θ' seed.
    dtype = grid.area_T.dtype
    state = make_rest_state(grid, hc, dtype=dtype)
    # IC perturbations in the bottom half: θ' AND velocity. Seeding velocity
    # (u', v', w') — not θ' alone — gives the resolved shear/convective eddies a
    # direct kick so turbulence spins up in O(10) eddy-turnovers instead of
    # waiting for buoyancy to convert a θ' seed (the θ'-only seed in a stable BL
    # barely grew, contributing to the laminar collapse).
    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    zmask = (hc.z_full < 0.5 * H).astype(dtype)

    def _lowpass(noise, frac=6):
        """Keep only LARGE horizontal scales (|k| < N/frac). A white-noise seed
        is dominated by 2Δ energy that the biharmonic hyperdiff annihilates in a
        few steps before it can organise; a smooth, large-scale seed survives to
        be amplified by the mean shear (resolved production)."""
        nyy, nxx = noise.shape[0], noise.shape[1]
        fh = jnp.fft.rfft2(noise, axes=(0, 1))
        cy = max(1, nyy // frac)
        cx = max(1, nxx // frac)
        iy = jnp.arange(nyy)
        my = (jnp.minimum(iy, nyy - iy) < cy)[:, None, None]
        mx = (jnp.arange(fh.shape[1]) < cx)[None, :, None]
        out = jnp.fft.irfft2(jnp.where(my & mx, fh, 0.0), axes=(0, 1),
                             s=(nyy, nxx))
        return out / (jnp.std(out) + 1e-12)        # unit-std large-scale field

    th_seed = 0.1 * _lowpass(
        jax.random.normal(k1, (ny, nx, nlev), dtype=dtype)) * zmask
    # Large-scale velocity kick (~5% of the mean): seeds eddies at scales the
    # hyperdiff does not erase, so the mean shear amplifies them.
    u_amp = 0.05 * abs(ug)
    # Mean wind: a LOG-LAW profile (shear present from t=0 ⇒ immediate resolved
    # shear production) for the neutral case; uniform geostrophic otherwise (the
    # SBL/CBL cases balance the geostrophic wind via Coriolis, not a log IC).
    if spec.get("log_wind", False):
        z_f = hc.z_full
        u_mean = ug * jnp.log(jnp.clip(z_f, z0, None) / z0) / jnp.log(H / z0)
        u_mean = jnp.clip(u_mean, -abs(ug), abs(ug))
        u_base = jnp.broadcast_to(u_mean, (ny, nx, nlev))
    else:
        u_base = jnp.full((ny, nx, nlev), ug, dtype=dtype)
    u0 = u_base + u_amp * _lowpass(
        jax.random.normal(k2, (ny, nx, nlev), dtype=dtype)) * zmask
    v0 = jnp.full((ny, nx, nlev), vg, dtype=dtype) + (
        u_amp * _lowpass(jax.random.normal(k3, (ny, nx, nlev), dtype=dtype))
        * zmask)
    w0 = (0.5 * u_amp
          * _lowpass(jax.random.normal(k4, (ny, nx, nlev + 1), dtype=dtype))
          * (hc.z_half < 0.5 * H).astype(dtype))
    state = state._replace(
        u=state.u.replace(data=u0),
        v=state.v.replace(data=v0),
        w=state.w.replace(data=w0),
        theta_prime=state.theta_prime.replace(data=th_seed),
    )
    if spec["moist"]:
        tracers = jnp.zeros((ny, nx, nlev, 1), dtype=dtype)
        state = state._replace(
            tracers=state.tracers.replace(data=tracers))

    surf = jax.jit(partial(
        _apply_pbl_column, case=args.case, hc=hc, z0=z0,
        dt=args.dt, ug=ug, vg=vg))
    return model, state, surf, grid, hc, spec


def _diagnostics(state, hc, spec):
    """Domain-mean BL diagnostics from the current state."""
    u = np.asarray(state.u.data)
    v = np.asarray(state.v.data)
    w = np.asarray(state.w.data)
    thp = np.asarray(state.theta_prime.data)
    z = np.asarray(hc.z_full)
    theta = np.asarray(hc.theta_ref) + thp.mean(axis=(0, 1))
    wvar = (w[..., :-1] ** 2).mean(axis=(0, 1))          # ~w'² (incl mean≈0)
    umean = u.mean(axis=(0, 1))
    vmean = v.mean(axis=(0, 1))
    spd = np.sqrt(umean ** 2 + vmean ** 2)
    # BL depth: height of max vertical θ gradient (inversion base), ascending z.
    order = np.argsort(z)
    zs, ths = z[order], theta[order]
    dthdz = np.gradient(ths, zs)
    h_bl = zs[3 + int(np.argmax(dthdz[3:]))] if len(zs) > 6 else np.nan
    return dict(
        max_w=float(np.abs(w).max()),
        wvar_max=float(wvar.max()),
        h_bl=float(h_bl),
        spd_max=float(spd.max()),
        spd_max_z=float(z[np.argmax(spd)]),
        theta_sfc=float(theta[order][0]),
    )


def _resolved_profiles(state, hc, spec):
    """Planar-mean profiles + resolved second moments (the LES turbulence
    statistics): θ, u, v, |U|, TKE, variances, kinematic momentum fluxes and the
    surface friction velocity. Shared by the per-frame recorder AND the final
    dump so the profile diagnostics are computed in exactly one place."""
    z = np.asarray(hc.z_full)                              # top-down (idx 0 = top)
    u3 = np.asarray(state.u.data)
    v3 = np.asarray(state.v.data)
    w3 = np.asarray(state.w.data)[..., :-1]               # full-level w (drop top)
    um = u3.mean(axis=(0, 1)); vm = v3.mean(axis=(0, 1)); wm = w3.mean(axis=(0, 1))
    up, vp, wp = u3 - um, v3 - vm, w3 - wm                # resolved fluctuations
    uw = (up * wp).mean(axis=(0, 1))
    vw = (vp * wp).mean(axis=(0, 1))
    uu = (up * up).mean(axis=(0, 1))
    vv = (vp * vp).mean(axis=(0, 1))
    ww = (wp * wp).mean(axis=(0, 1))
    tke = 0.5 * (uu + vv + ww)
    theta = np.asarray(hc.theta_ref) + np.asarray(
        state.theta_prime.data).mean(axis=(0, 1))
    spd = np.sqrt(um ** 2 + vm ** 2)
    u_star = float((uw[np.argmin(z)] ** 2 + vw[np.argmin(z)] ** 2) ** 0.25)
    return dict(z=z, theta=theta, u=um, v=vm, spd=spd, wvar=ww,
                uu=uu, vv=vv, ww=ww, tke=tke, uw=uw, vw=vw,
                u_star=u_star, z0=spec["z0"], case=spec.get("case", ""))


def _select_height_indices(hc, H):
    """Indices into the (top-down) vertical axis for the surface (lowest model
    level) + four heights spanning the BL at 0.1/0.25/0.5/0.8·H. Returns
    (indices, heights[m]) ascending in z."""
    z = np.asarray(hc.z_full)
    surf = int(np.argmin(z))
    targets = np.array([0.10, 0.25, 0.50, 0.80]) * float(H)
    idx = [surf] + [int(np.argmin(np.abs(z - t))) for t in targets]
    # de-duplicate while preserving the surface-first, ascending order
    seen, out = set(), []
    for k in sorted(idx, key=lambda kk: z[kk]):
        if k not in seen:
            seen.add(k); out.append(k)
    out = np.array(out, dtype=int)
    return out, z[out]


def _height_slices(state, hc, h_idx):
    """Horizontal (x,y) cross-sections of w, θ, u, v at the chosen level indices.
    Each returned array is (n_heights, ny, nx). w is averaged to full levels."""
    w = np.asarray(state.w.data)
    w_full = 0.5 * (w[..., :-1] + w[..., 1:])             # half -> full level
    u = np.asarray(state.u.data)
    v = np.asarray(state.v.data)
    theta = np.asarray(hc.theta_ref) + np.asarray(state.theta_prime.data)
    return dict(
        w=np.stack([w_full[:, :, k] for k in h_idx]),
        theta=np.stack([theta[:, :, k] for k in h_idx]),
        u=np.stack([u[:, :, k] for k in h_idx]),
        v=np.stack([v[:, :, k] for k in h_idx]),
    )


def _record_frame(state, hc, spec, args, t, frame, h_idx, h_z,
                  snap_dir, prof_dir):
    """Write one snapshot npz (height cross-sections) + one profile npz."""
    t_hours = t / 3600.0
    sl = _height_slices(state, hc, h_idx)
    np.savez(snap_dir / f"snap_{frame:03d}.npz",
             t_hours=t_hours, case=args.case, heights=h_z,
             dx=args.dx, Lx=args.nx * args.dx, Ly=args.ny * args.dx, **sl)
    prof = _resolved_profiles(state, hc, spec)
    np.savez(prof_dir / f"prof_{frame:03d}.npz", t_hours=t_hours, **prof)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=list(_CASES), required=True)
    p.add_argument("--nx", type=int)
    p.add_argument("--ny", type=int)
    p.add_argument("--nlev", type=int)
    p.add_argument("--dx", type=float)
    p.add_argument("--H", type=float)
    p.add_argument("--dz-sfc", type=float)
    p.add_argument("--dt", type=float)
    p.add_argument("--hours", type=float)
    p.add_argument("--off-centering", type=float, default=0.2,
                   help="SI vertical-acoustic off-centring β (damps the 2Δz "
                        "vertical mode). 0 = centred (undamped). LES NOTE: this "
                        "damping ALSO annihilates the resolved w' each acoustic "
                        "substep — set 0 (with --si-w-filter 0 and a smaller dt) "
                        "to let BL turbulence sustain; nonzero ⇒ laminar collapse.")
    p.add_argument("--si-w-filter", type=float, default=0.2,
                   help="Vertical Laplacian filter ν on w inside the SI solve "
                        "(damps 2Δz w noise).")
    p.add_argument("--f32", action="store_true",
                   help="Run in float32 (production GPU LES). Keeps grid+hc+state "
                        "in one dtype (no dycore upcast) so the run is fast on "
                        "consumer GPUs and the time loop can be a single scan. "
                        "Default float64 (bit-exact science).")
    p.add_argument("--static-sgs", action="store_true",
                   help="Use fixed-C_s Smagorinsky (diagnostic) instead of the "
                        "dynamic coefficient.")
    p.add_argument("--scale-dependent", action="store_true",
                   help="Use the Bou-Zeid et al. (2005) scale-dependent dynamic "
                        "(LASD) SGS closure — the jax-alfa oracle model — "
                        "instead of the standard (scale-invariant) Germano "
                        "dynamic coefficient. Requires the dynamic path "
                        "(i.e. not --static-sgs).")
    p.add_argument("--n-acoustic-substeps", type=int, default=8,
                   help="Horizontal acoustic substeps per dt. Need "
                        "c·(dt/n)/dx < 1 (c≈340): for dx=25 m, dt=0.5 s ⇒ n≳7.")
    p.add_argument("--hyperdiff", type=float, default=1.0e-2)
    p.add_argument("--print-every", type=int, default=200)
    p.add_argument("--record-frames", type=int, default=20,
                   help="Number of evenly-spaced frames to save (snapshot "
                        "cross-sections at surface+4 heights and mean profiles) "
                        "for the publication diagnostics. 0 disables recording.")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--research-only", action="store_true",
                   help="Acknowledge that the compressible core CANNOT sustain LES "
                        "turbulence (relaminarises; some cases NaN) and run anyway. "
                        "Required — this driver is research-only; use run_spectral_les.py "
                        "or run_pseudo_les.py for real turbulent LES.")
    args = p.parse_args()
    # This core does NOT sustain resolved ABL LES turbulence: the numerical
    # dissipation (acoustic off-centring + biharmonic hyperdiff) caps the
    # effective Re below transition, so the BL relaminarises (u*→~0), and some
    # cases hit a destructive acoustic burst → NaN. Confirmed + documented in
    # docs/physics-notes/les_crossgrid_regression_2026-07.md. Gate it behind an
    # explicit opt-in so its (laminar / NaN) output is never mistaken for LES.
    if not args.research_only:
        sys.stderr.write(
            "\n*** run_les_plane.py drives the COMPRESSIBLE plane core, which does "
            "NOT sustain resolved LES turbulence — it relaminarises (u*→~0) and some "
            "cases hit an acoustic burst → NaN. RESEARCH-ONLY / non-standard.\n"
            "    For real turbulent LES use the spectral core (run_spectral_les.py / "
            "run_spectral_sbl.py / run_spectral_cbl.py) or the pseudo-incompressible "
            "core (run_pseudo_les.py).\n"
            "    Re-run with --research-only to proceed anyway. "
            "See docs/physics-notes/les_crossgrid_regression_2026-07.md ***\n\n")
        raise SystemExit(2)
    sys.stderr.write("*** COMPRESSIBLE LES (--research-only): relaminarises / may "
                     "NaN; output is NOT validated turbulent LES. ***\n")
    # Fill unset args from the per-case defaults.
    spec = _CASES[args.case]
    spec["case"] = args.case
    for k in ("nx", "ny", "nlev", "dx", "H", "dz_sfc", "dt", "hours"):
        if getattr(args, k) is None:
            setattr(args, k, spec[k])
    args.output = args.output or Path(f"results/les_{args.case}")
    args.output.mkdir(parents=True, exist_ok=True)

    model, state, surf, grid, hc, spec = build(args)
    nsteps = int(args.hours * 3600.0 / args.dt)
    print(f"[LES {args.case}] nx={args.nx} ny={args.ny} nlev={args.nlev} "
          f"dx={args.dx} m H={args.H} m dz_sfc={args.dz_sfc} m")
    print(f"  dt={args.dt} s  steps={nsteps}  hours={args.hours}  "
          f"f_c={spec['f_c']:.3e}  ug={spec['ug']}  dynamic-Smagorinsky")
    print(f"{'step':>7} {'t[h]':>6} {'max|w|':>9} {'wvar_max':>9} "
          f"{'h_bl[m]':>8} {'spd_max':>8} {'@z[m]':>7} {'th_sfc':>8}")

    # Frame recorder: even spacing across the run (snapshots + profiles).
    record = args.record_frames > 0
    if record:
        snap_dir = args.output / "snapshots"; snap_dir.mkdir(exist_ok=True)
        prof_dir = args.output / "profiles"; prof_dir.mkdir(exist_ok=True)
        h_idx, h_z = _select_height_indices(hc, args.H)
        record_every = max(1, nsteps // args.record_frames)
        frame = 0
        _record_frame(state, hc, spec, args, 0.0, frame, h_idx, h_z,
                      snap_dir, prof_dir)  # t=0 initial frame
        frame += 1
        print(f"  recording {args.record_frames} frames every {record_every} "
              f"steps; heights[m]={np.round(h_z, 1)}")

    import time
    t0 = time.time()
    for i in range(nsteps):
        t = i * args.dt
        state = model.step(state, dt=args.dt, physics_fn=None)
        state = surf(state, jnp.asarray(t))
        if record and (i + 1) % record_every == 0:
            _record_frame(state, hc, spec, args, (i + 1) * args.dt, frame,
                          h_idx, h_z, snap_dir, prof_dir)
            frame += 1
        if (i + 1) % args.print_every == 0 or i == 0:
            d = _diagnostics(state, hc, spec)
            mw = d["max_w"]
            if not np.isfinite(mw) or mw > 100.0:
                print(f"[BLOWUP] step {i+1} max|w|={mw}")
                return 1
            print(f"{i+1:7d} {(i+1)*args.dt/3600:6.3f} {d['max_w']:9.3f} "
                  f"{d['wvar_max']:9.4f} {d['h_bl']:8.1f} {d['spd_max']:8.3f} "
                  f"{d['spd_max_z']:7.1f} {d['theta_sfc']:8.3f}")
    wall = time.time() - t0
    d = _diagnostics(state, hc, spec)
    print(f"\n[DONE] wall={wall:.1f}s  {nsteps/wall:.1f} steps/s")
    print(f"  final: BL depth={d['h_bl']:.1f} m  max|w|={d['max_w']:.3f}  "
          f"jet spd={d['spd_max']:.3f} @ z={d['spd_max_z']:.1f} m  "
          f"θ_sfc={d['theta_sfc']:.3f} K")
    # Dump final mean profiles + resolved turbulence statistics for the oracle
    # (Monin-Obukhov similarity) validation — see scripts/validate/validate_les_vs_oracle.py.
    prof = _resolved_profiles(state, hc, spec)
    np.savez(args.output / "final_profiles.npz", **prof)
    if record:                                            # final-time frame too
        _record_frame(state, hc, spec, args, nsteps * args.dt, frame,
                      h_idx, h_z, snap_dir, prof_dir)
    print(f"  profiles -> {args.output}/final_profiles.npz  "
          f"(u*≈{prof['u_star']:.3f} m/s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
