"""Emit LES reference artifacts for the LES-truth suite (registry-driven).

Turns a registry :class:`LESCase` into a finished LES run and writes the
self-describing :class:`LESReferenceArtifact` (``bridge.save_artifact``) the SCM
tuner consumes — horizontal-mean truth profiles (θ, u, v), the resolved AND SGS
turbulent fluxes, and the exact forcing the LES received (LES_SUITE.md §5).

The horizontal-mean + flux reductions live in the CPU-testable
``legoesm.atmosphere.les_suite.emit``; this driver only builds the regime's IC +
config, runs the spectral truth core on GPU, and stacks the reductions per output
time. The SGS heat flux uses the closure's own ``ν_t``
(``spectral_les_plane.eddy_viscosity``) → ``<w'θ'>_sgs = -<(ν_t/Pr) ∂θ/∂z>``.

GPU-gated (a 96³ sim-hour is ~7 min on a V100S). The **dry free-convective CBL**
(incl. the sheared --Ug axis) and the **dry stably-stratified SBL** (GABLS1, with a
Rayleigh sponge and the published prescribed surface cooling through the core's
Monin–Obukhov surface layer) regimes are wired end-to-end; the MOIST regimes need their own IC
builders + the D9 cloud scheme and are an explicit follow-up — the driver raises rather
than silently emitting a wrong-regime artifact.

Each frame also records the momentum fluxes (resolved + SGS), variances, resolved
TKE, ``<w'w'θ'>``, the pressure correlations (from the instantaneous kinematic
pressure, ``spectral_les_plane.kinematic_pressure``), the background-viscosity
floor's flux share, and the realised surface flux/stress; the run ends with the
column θ/u/v budget residuals (sources integrated stage-by-stage by the core).
``--refine N`` is the N× convergence run.

A case carrying a full :class:`~legoesm.atmosphere.les_suite.registry.LESSetup`
(the published cases in ``catalog.PUBLISHED_CASES``) runs through one generic dry
builder: every input — sounding, height-varying geostrophic wind, latitude,
z0/z0h, MO-similarity surface drag, surface flux / cooling rate / surface-θ
series (DEPHY files read by ``dephy_scm.load_dephy_les_forcing``), sponge, seed
noise, dt — comes from the case, and a CLI flag for any of them is refused. A
setup with ``pending`` (unapproved) values emits only ``--smoke`` artifacts.

``--checkpoint PATH --wall-limit-s S`` makes a run resumable across jobs: the
state is written at every frame and at the wall limit (exit 75, which
``scripts/cluster/les_suite/derecho_les_case.pbs`` turns into the next link), and
a resumed run is bit-identical to an uninterrupted one.

Usage::

    JAX_PLATFORMS=cuda .venv/bin/python scripts/run/run_les_suite.py \\
        --case cbl_nieuwstadt --hours 2.0 --frames 12 \\
        --output results/les_suite/artifacts
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

_F32 = "--f32" in sys.argv
import jax  # noqa: E402

if not _F32:
    jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import legoesm  # noqa: E402
from legoesm.atmosphere.dynamics.les import spectral_les_plane as sl  # noqa: E402
from legoesm.atmosphere.les_suite import (  # noqa: E402
    get_case,
    list_cases,
    register_default_catalog,
)
from legoesm.atmosphere.les_suite.bridge import save_artifact  # noqa: E402
from legoesm.atmosphere.les_suite.emit import (  # noqa: E402
    build_reference_artifact,
    column_share,
    covariance,
    horizontal_mean,
    resolved_vertical_flux,
    sgs_flux_from_face_stress,
    sgs_vertical_scalar_flux_mean,
    third_moment,
)
from legoesm.timestepping.split_explicit import select_dt  # noqa: E402

# Regimes whose IC/forcing builder is wired here. The rest raise (honest dispatch,
# NOT a silent wrong-regime emission) until their builders land. Moist regimes
# (shallow_cumulus, stratocumulus) still need their IC builders + the D9 cloud scheme.
# dry_neutral runs only through a full LESSetup (no legacy builder).
_WIRED_REGIMES = ("dry_convective", "dry_stable", "dry_neutral")


def _coriolis_f(lat_deg: float) -> float:
    """Coriolis parameter f = 2Ω sin(φ) [1/s] at latitude ``lat_deg`` (Ω from
    ``legoesm.constants``). A sheared CBL needs f≠0 for U_g to drive a geostrophic/Ekman
    balance (the core's ``f_cor*(v−vg)`` / ``−f_cor*(u−ug)`` terms)."""
    from legoesm import constants  # noqa: PLC0415
    return float(2.0 * constants.Omega * np.sin(np.radians(lat_deg)))


def _sgs_les_config(sgs: str) -> dict:
    """SpectralLESConfig overrides selecting the LES SGS closure (the D7 σ_LES spread).

    ``lasd`` = Bou-Zeid scale-dependent dynamic Smagorinsky (``smagorinsky_dynamic``);
    ``smagorinsky`` = static Mason-capped |S|-Smagorinsky; ``vreman`` = Vreman 2004.
    The emit SGS-flux reconstruction reads the SAME ``eddy_viscosity`` the core
    integrates, so it stays bit-consistent for every variant. Raises on an unknown SGS
    (dispatch hardening) — never a silent wrong-closure emission.
    """
    if sgs == "lasd":
        return {"smagorinsky_dynamic": True, "sgs_model": "smagorinsky"}
    if sgs == "smagorinsky":
        return {"smagorinsky_dynamic": False, "sgs_model": "smagorinsky"}
    if sgs == "vreman":
        return {"smagorinsky_dynamic": False, "sgs_model": "vreman"}
    raise SystemExit(
        f"unknown --sgs {sgs!r}; supported: lasd, smagorinsky, vreman")


def frame_step_schedule(n_steps: int, frames: int) -> list[int]:
    """Step indices (after the t=0 IC) at which to record ``frames`` snapshots.

    Returns up to ``frames-1`` strictly-increasing DISTINCT step indices in
    ``[1, n_steps]`` (the last is always ``n_steps`` → final frame ≈ T). Sampling by
    index — not by crossing continuous time targets — guarantees distinct snapshot
    times and no overshoot past ``n_steps``. If ``dt`` is too coarse to resolve
    ``frames`` distinct snapshots (``n_steps < frames-1``), fewer are returned (the
    caller reports it); never a silent duplicate/drop.
    """
    n_out = min(frames - 1, max(1, n_steps))
    return sorted({int(round(k * n_steps / n_out)) for k in range(1, n_out + 1)})


def _build_cbl(case, args, dtype, sgs="lasd", u_geo_mag=0.0):
    """Dry CBL IC + config (mirrors run_spectral_cbl.build).

    ``sgs`` selects the SGS closure (the D7 σ_LES spread) via :func:`_sgs_les_config`.
    ``u_geo_mag`` (|U_g|, m/s) is the geostrophic wind for a SHEARED CBL: the wind is
    initialised to (U_g, 0) so the column starts in geostrophic balance and the surface
    drag builds the Ekman spiral. ``0`` reproduces the free-convective IC (u=v=0).
    """
    theta0 = args.theta0
    cfg = sl.SpectralLESConfig(
        nx=case.grid.nx, ny=case.grid.ny, nz=case.grid.nz,
        Lx=case.grid.Lx_m, Ly=case.grid.Ly_m, Lz=case.grid.Lz_m,
        z0=args.z0, dealias=True, **_sgs_les_config(sgs),
        time_scheme="rk3",
        buoyancy=True, theta_ref0=theta0, pr_sgs=args.pr_sgs,
        # sgs_buoyancy MUST stay off: emit.sgs_vertical_scalar_flux_mean reconstructs
        # the SGS heat flux from eddy_viscosity() with K_h=ν_t/Pr, matching the core's
        # scalar_rhs. If sgs_buoyancy were on, the core rescales ν_t by the Lilly
        # factor (spectral_les_plane rhs) and the emitted SGS flux would no longer
        # equal the flux the model integrated.
        sgs_buoyancy=False)
    g = sl.make_grid(cfg, dtype=dtype)
    z = g.z_c
    th = jnp.where(z > args.zi0, theta0 + args.gamma * (z - args.zi0),
                   theta0).astype(dtype)
    key = jax.random.PRNGKey(0)
    th3 = jnp.broadcast_to(th, (cfg.ny, cfg.nx, cfg.nz)).astype(dtype) + (
        0.1 * jax.random.normal(key, (cfg.ny, cfg.nx, cfg.nz), dtype=dtype)
        * (z < args.zi0).astype(dtype))
    # Sheared CBL starts in geostrophic balance (u=U_g, v=0); free-convective is u=v=0.
    u = jnp.full((cfg.ny, cfg.nx, cfg.nz), dtype(u_geo_mag), dtype)
    v = jnp.zeros((cfg.ny, cfg.nx, cfg.nz), dtype)
    w = jnp.zeros((cfg.ny, cfg.nx, cfg.nz + 1), dtype)
    st = sl.SpectralLESState(
        u=u, v=v, w=w, rhs_u_prev=jnp.zeros_like(u), rhs_v_prev=jnp.zeros_like(v),
        rhs_w_prev=jnp.zeros_like(w), theta=th3, rhs_theta_prev=jnp.zeros_like(th3))
    # --q0 overrides the case's surface flux (the Q1 buoyancy-axis sweep); else the
    # case default. The artifact records the flux actually applied, so the tuner's
    # controlled comparison stays consistent.
    Q0 = args.q0 if args.q0 is not None else case.surface_theta_flux_K_m_s
    if Q0 is None:
        raise SystemExit(f"{case.name}: dry CBL requires surface_theta_flux_K_m_s")
    return g, st, float(Q0)


# GABLS1 stably-stratified sounding (Beare et al. 2006), mirrored from run_spectral_sbl:
# a well-mixed layer capped by a +0.01 K/m inversion, SMOOTHED over ~25 m (an unresolved
# ∂θ/∂z step seeds a Gibbs/KH instability where the dynamic SGS shuts off).
_SBL_ZI_M = 100.0        # mixed-layer depth
_SBL_DELTA_M = 25.0      # inversion smoothing scale
_SBL_GAMMA_K_M = 0.01    # capping lapse rate [K/m]
_SBL_TAU_SPONGE_S = 50.0  # Rayleigh sponge timescale in the top 25% of the domain
_SBL_STATIC_CS = 0.18     # static-Smagorinsky C_s for the SBL (run_spectral_sbl default)
_PR_SGS = 1.0             # turbulent Prandtl number (suite default)
_NU_FLOOR_M2_S = 0.01     # stable-regime background viscosity (user-approved for the SBLs)
# Regime-appropriate defaults (the LESCase carries flux + U_g, not θ0 / latitude).
_CBL_THETA0_K, _GABLS1_THETA0_K = 300.0, 265.0
_CBL_LAT_DEG, _GABLS1_LAT_DEG = 45.0, 73.0  # 73° → f=1.39e-4 (GABLS1)


def _build_sbl(case, args, dtype, sgs="vreman", u_geo_mag=8.0):
    """Dry stably-stratified sheared SBL IC + config (mirrors run_spectral_sbl.build).

    GABLS1: a stratified θ sounding, a geostrophic wind ``U_g`` + Coriolis driving an
    Ekman SBL, and surface COOLING (a negative kinematic heat flux). Needs a background
    ``nu_floor`` + the Rayleigh sponge (applied in the emit step) — without them the
    low-level jet piles at the rigid lid and the run blows up (~0.9 h). ``sgs`` selects
    the closure via :func:`_sgs_les_config` (LASD can destabilise the anisotropic SBL
    grid → smagorinsky/vreman are the stable spread members).
    """
    theta0 = args.theta0
    cfg = sl.SpectralLESConfig(
        nx=case.grid.nx, ny=case.grid.ny, nz=case.grid.nz,
        Lx=case.grid.Lx_m, Ly=case.grid.Ly_m, Lz=case.grid.Lz_m,
        z0=args.z0, dealias=True, **_sgs_les_config(sgs), c_s=_SBL_STATIC_CS,
        time_scheme="rk3", buoyancy=True, theta_ref0=theta0, pr_sgs=args.pr_sgs,
        nu_floor=args.nu_floor, sgs_buoyancy=False)
    g = sl.make_grid(cfg, dtype=dtype)
    z = g.z_c
    zi, di, gam = _SBL_ZI_M, _SBL_DELTA_M, _SBL_GAMMA_K_M
    th = (theta0 + 0.5 * gam * ((z - zi) + di * jnp.log(jnp.cosh((z - zi) / di))
                                + di * jnp.log(2.0))).astype(dtype)
    seed = (z < 0.5 * cfg.Lz).astype(dtype)  # perturb only the (turbulent) lower half
    th3 = jnp.broadcast_to(th, (cfg.ny, cfg.nx, cfg.nz)).astype(dtype) + (
        0.1 * jax.random.normal(jax.random.PRNGKey(0), (cfg.ny, cfg.nx, cfg.nz), dtype)
        * seed)
    u = jnp.full((cfg.ny, cfg.nx, cfg.nz), dtype(u_geo_mag), dtype) + (
        0.1 * jax.random.normal(jax.random.PRNGKey(1), (cfg.ny, cfg.nx, cfg.nz), dtype)
        * seed)
    v = 0.1 * jax.random.normal(jax.random.PRNGKey(2), (cfg.ny, cfg.nx, cfg.nz), dtype) * seed
    w = jnp.zeros((cfg.ny, cfg.nx, cfg.nz + 1), dtype)
    dt0 = float(args.dt) if args.dt is not None else float(case.grid.dt_s)
    u, v, w = sl.project(u, v, w, dt=dt0, g=g)
    st = sl.SpectralLESState(
        u=u, v=v, w=w, rhs_u_prev=jnp.zeros_like(u), rhs_v_prev=jnp.zeros_like(v),
        rhs_w_prev=jnp.zeros_like(w), theta=th3, rhs_theta_prev=jnp.zeros_like(th3))
    if case.surface_cooling_rate_K_per_h is not None:
        # Published prescribed-cooling BC: the flux is derived each RK stage by the
        # core's MOST surface layer from T_s(t) (see _surface_temperature).
        if args.q0 is not None:
            raise SystemExit(f"{case.name}: --q0 conflicts with the case's prescribed "
                             "surface cooling rate")
        return g, st, None
    Q0 = args.q0 if args.q0 is not None else case.surface_theta_flux_K_m_s
    if Q0 is None:
        raise SystemExit(f"{case.name}: SBL requires a surface cooling rate or "
                         "surface_theta_flux_K_m_s")
    return g, st, float(Q0)


def _surface_temperature(case, theta0, t_s):
    """Prescribed surface temperature T_s(t) = θ₀ − rate·t [K] for a cooling-forced
    case (Beare et al. 2006 GABLS1: θ₀ = 265 K, 0.25 K/h), held over each step from
    its start time ``t_s`` [s] like ``run_spectral_sbl``. ``None`` = flux-forced."""
    rate = case.surface_cooling_rate_K_per_h
    if rate is None:
        return None
    return theta0 - rate * (t_s / 3600.0)


def _knots(z, knots, col):
    """Column ``col`` of piecewise-linear ``knots`` (first entry = height) at ``z``;
    held constant beyond the end knots."""
    zk = np.asarray([k[0] for k in knots], dtype=np.float64)
    return np.interp(np.asarray(z, dtype=np.float64), zk,
                     np.asarray([k[col] for k in knots], dtype=np.float64))


def _resolve_setup(case, z_c):
    """Everything a setup case defines, on the LES centre levels ``z_c``.

    Returns a dict: ``theta, u0, v0, ug, vg`` (``(nz,)`` profiles), ``lat_deg``,
    ``z0``, ``z0h``, ``theta_ref``, and the surface forcing as exactly one of
    ``q0`` (constant kinematic θ flux) / ``cooling_rate`` (K/h) / ``theta_s``
    (``(times_s, θ_s)`` series). A DEPHY case is read from its file and must agree
    with the latitude/roughness its LESSetup declares — two sources disagreeing
    about one parameter is refused, not resolved."""
    s = case.setup
    z_c = np.asarray(z_c, dtype=np.float64)
    out = dict(lat_deg=s.lat_deg, z0=s.z0_m, z0h=s.z0h_m, theta_ref=s.theta_ref_K,
               q0=None, cooling_rate=None, theta_s=None)
    if s.dephy_case is not None:
        from legoesm.atmosphere.forcing.scm.dephy_scm import (  # noqa: PLC0415
            load_dephy_les_forcing,
        )
        d = load_dephy_les_forcing(s.dephy_case)
        for label, declared, got in (("lat_deg", s.lat_deg, d.lat_deg),
                                     ("z0_m", s.z0_m, d.z0_m),
                                     ("z0h_m", s.z0h_m, d.z0h_m)):
            if (declared is None) != (got is None) or (
                    declared is not None and not math.isclose(declared, got,
                                                               rel_tol=1e-6)):
                raise SystemExit(f"{case.name}: setup {label}={declared} disagrees "
                                 f"with the DEPHY file's {got}")
        if z_c[-1] > d.z_m[-1]:
            raise SystemExit(f"{case.name}: LES top {z_c[-1]} m is above the DEPHY "
                             f"profiles' top {d.z_m[-1]} m")
        prof = {k: np.interp(z_c, d.z_m, v) for k, v in (
            ("theta", d.theta_K), ("u0", d.u_m_s), ("v0", d.v_m_s),
            ("ug", d.ug_m_s), ("vg", d.vg_m_s))}
        out.update(prof)
        if d.w_theta_s is not None:
            if np.max(np.abs(d.w_theta_s - d.w_theta_s[0])) > 0.0:
                raise SystemExit(f"{case.name}: time-varying prescribed surface flux "
                                 "is not wired (the LES step takes a constant Q0)")
            out["q0"] = float(d.w_theta_s[0])
        else:
            out["theta_s"] = (np.asarray(d.time_s - d.time_s[0]), np.asarray(d.theta_s))
        return out
    out["theta"] = _knots(z_c, s.theta_knots, 1)
    out["ug"] = _knots(z_c, s.geostrophic_knots, 1)
    out["vg"] = _knots(z_c, s.geostrophic_knots, 2)
    if s.wind_init_knots:
        out["u0"] = _knots(z_c, s.wind_init_knots, 1)
        out["v0"] = _knots(z_c, s.wind_init_knots, 2)
    else:
        out["u0"], out["v0"] = out["ug"].copy(), out["vg"].copy()
    if case.surface_theta_flux_K_m_s is not None:
        out["q0"] = float(case.surface_theta_flux_K_m_s)
    elif case.surface_cooling_rate_K_per_h is not None:
        out["cooling_rate"] = float(case.surface_cooling_rate_K_per_h)
    else:
        out["theta_s"] = (np.asarray([k[0] for k in s.surface_theta_knots]),
                          np.asarray([k[1] for k in s.surface_theta_knots]))
    return out


def _build_setup(case, args, dtype, sgs):
    """IC + config for a case carrying a full :class:`LESSetup` (the generic dry
    builder). Returns ``(g, st, Q0, res)`` with ``res`` from :func:`_resolve_setup`
    (``Q0`` is ``None`` unless the surface is flux-forced). Regime conventions of
    the suite are kept: a stable case gets the background viscosity floor and the
    SBL static C_s; θ seed noise only, from the setup's amplitude and depth."""
    s = case.setup
    stable = case.regime == "dry_stable"
    extra = dict(nu_floor=args.nu_floor, c_s=_SBL_STATIC_CS) if stable else {}
    gr = case.grid
    # z_c is needed before the config can carry the setup's θ_ref: build a probe
    # grid only for the level heights (z depends on Lz, nz only).
    z_c = (np.arange(gr.nz) + 0.5) * (gr.Lz_m / gr.nz)
    res = _resolve_setup(case, z_c)
    cfg = sl.SpectralLESConfig(
        nx=gr.nx, ny=gr.ny, nz=gr.nz, Lx=gr.Lx_m, Ly=gr.Ly_m, Lz=gr.Lz_m,
        z0=res["z0"], z0h=res["z0h"], sfc_flux_drag=s.surface_drag, dealias=True,
        **_sgs_les_config(sgs), time_scheme="rk3", buoyancy=True,
        theta_ref0=res["theta_ref"], pr_sgs=args.pr_sgs, sgs_buoyancy=False, **extra)
    g = sl.make_grid(cfg, dtype=dtype)
    np.testing.assert_allclose(np.asarray(g.z_c, np.float64), z_c,
                               rtol=8 * np.finfo(np.dtype(dtype)).eps, atol=0)
    shape = (cfg.ny, cfg.nx, cfg.nz)
    seed = (g.z_c < s.perturb_top_m).astype(dtype)
    th3 = jnp.broadcast_to(jnp.asarray(res["theta"], dtype), shape) + (
        s.perturb_K * jax.random.normal(jax.random.PRNGKey(0), shape, dtype) * seed)
    u = jnp.broadcast_to(jnp.asarray(res["u0"], dtype), shape)
    v = jnp.broadcast_to(jnp.asarray(res["v0"], dtype), shape)
    w = jnp.zeros((cfg.ny, cfg.nx, cfg.nz + 1), dtype)
    st = sl.SpectralLESState(
        u=u, v=v, w=w, rhs_u_prev=jnp.zeros_like(u), rhs_v_prev=jnp.zeros_like(v),
        rhs_w_prev=jnp.zeros_like(w), theta=th3, rhs_theta_prev=jnp.zeros_like(th3))
    return g, st, res["q0"], res


def _make_emit_step(g, case, u_geo, f_cor, Q0, dtype):
    """The jitted per-step map for one regime:
    ``step(state, dt, t_sfc) -> (state, u_*, budget)``.

    ``budget`` = the step's time-integrated :data:`sl.BUDGET_TERMS` (from the RK
    stages) followed by the sponge's change of the column-integrated mean θ, u, v.
    Dry CBL = plain ``sl.step``; SBL adds the Rayleigh sponge in the top 25% (+ a
    re-projection to restore ∇·u=0 after the sponge) so the jet/gravity-waves
    reaching the rigid lid are absorbed, not reflected. ``t_sfc`` is the
    prescribed surface temperature (MOST BC) or ``None`` for the flux ``Q0``."""
    if case.regime not in ("dry_convective", "dry_stable"):
        raise SystemExit(f"no emit step wired for regime {case.regime!r}")
    z_sp = case.sponge_base_m(case.grid.Lz_m)       # the catalog's (None = no sponge)
    return _make_step(g, u_geo, f_cor, Q0, dtype,
                      sponge=None if z_sp is None else (z_sp, _SBL_TAU_SPONGE_S))


def _make_step(g, u_geo, f_cor, Q0, dtype, sponge=None):
    """Jitted ``step(state, dt, t_sfc) -> (state, u_*, budget)`` with an optional
    Rayleigh sponge ``(z_bottom [m], tau [s])``: a cosine ramp from ``z_bottom`` to
    the lid that relaxes the horizontal DEVIATIONS of u, v, θ and all of w, then
    re-projects (the sponge breaks ∇·u=0). It changes no horizontal mean, so its
    column-budget entries (the last three of ``budget``) are reported, not hidden.
    ``u_geo`` is ``(u_g, v_g)`` as scalars or ``(nz,)`` profiles."""
    q0 = 0.0 if Q0 is None else Q0

    def _col(f):
        return g.dz * jnp.sum(sl.planar_mean(f, g))

    if sponge is None:
        @jax.jit
        def _plain_step(state, dt, t_sfc=None):
            state, us, b = sl.step(state, g=g, dt=dt, u_geo=u_geo, f_cor=f_cor,
                                   force=(0.0, 0.0), sfc_theta_flux=q0, t_sfc=t_sfc,
                                   return_budget=True)
            return state, us, jnp.concatenate([b, jnp.zeros(3, b.dtype)])
        return _plain_step
    z_sp, tau = sponge
    Lz = g.cfg.Lz
    if not (0.0 <= z_sp < Lz) or not tau > 0.0:
        raise SystemExit(f"sponge needs 0 <= z_bottom < Lz={Lz} and tau > 0, got "
                         f"{sponge}")
    zc, zf = g.z_c, g.z_f
    spc = jnp.where(zc > z_sp, 0.5 * (1.0 - jnp.cos(
        jnp.pi * (zc - z_sp) / (Lz - z_sp))), 0.0).astype(dtype)
    spf = jnp.where(zf > z_sp, 0.5 * (1.0 - jnp.cos(
        jnp.pi * (zf - z_sp) / (Lz - z_sp))), 0.0).astype(dtype)

    @jax.jit
    def _sponge_step(state, dt, t_sfc=None):
        state, us, b = sl.step(state, g=g, dt=dt, u_geo=u_geo, f_cor=f_cor,
                               force=(0.0, 0.0), sfc_theta_flux=q0, t_sfc=t_sfc,
                               return_budget=True)
        rc = (dt / tau) * spc
        rf = (dt / tau) * spf
        u = state.u - rc * (state.u - sl.planar_mean(state.u, g, keepdims=True))
        v = state.v - rc * (state.v - sl.planar_mean(state.v, g, keepdims=True))
        w = state.w - rf * state.w
        th = state.theta - rc * (state.theta
                                 - sl.planar_mean(state.theta, g, keepdims=True))
        u, v, w = sl.project(u, v, w, dt=dt, g=g)  # sponge broke ∇·u=0 → re-project
        sp = jnp.stack([_col(th) - _col(state.theta), _col(u) - _col(state.u),
                        _col(v) - _col(state.v)])
        return (state._replace(u=u, v=v, w=w, theta=th), us,
                jnp.concatenate([b, sp.astype(b.dtype)]))
    return _sponge_step


def _frame_fields_fn(g, u_geo, f_cor, Q0):
    """Jitted per-frame field set (device): ν_t, the applied surface BC, the SGS
    momentum stress on faces, and the kinematic pressure with its gradients.

    The SGS fluxes are rebuilt from ``eddy_viscosity`` — what ``rhs`` applies only
    while ``sgs_buoyancy`` is off (with it on, ``rhs`` rescales ``ν_t`` by the Lilly
    factor first), so that combination is refused rather than emitted wrong."""
    if g.cfg.sgs_buoyancy:
        raise SystemExit("emitted SGS fluxes assume sgs_buoyancy=False (rhs would "
                         "rescale ν_t by the Lilly factor before applying it)")
    q0 = 0.0 if Q0 is None else Q0

    @jax.jit
    def _fields(st, t_sfc=None):
        cd_surf, q, most_resid = sl.surface_boundary(st.u, st.v, st.theta, g, q0,
                                                     t_sfc, return_residual=True)
        nu_t = sl.eddy_viscosity(st.u, st.v, st.w, g)
        tau13, tau23, spd1, cd = sl.sgs_vertical_momentum_flux_faces(
            st.u, st.v, st.w, nu_t, g, cd_surf=cd_surf)
        p = sl.kinematic_pressure(st, g, u_geo, f_cor, sfc_theta_flux=q0, t_sfc=t_sfc)
        dpdz = sl.f2c(jnp.pad(sl.ddz_c2f(p, g.dz), ((0, 0), (0, 0), (1, 1))))
        floor = jnp.full_like(nu_t, g.cfg.nu_floor)
        fl13, fl23, _s, _c = sl.sgs_vertical_momentum_flux_faces(
            st.u, st.v, st.w, floor, g, cd_surf=cd_surf)
        fl13 = fl13.at[..., 0].set(0.0)   # the wall stress is not the floor's
        fl23 = fl23.at[..., 0].set(0.0)
        return dict(
            q=jnp.asarray(q, st.u.dtype), nu_t=nu_t, tau13=tau13, tau23=tau23,
            tau_x_s=-cd * spd1 * sl.planar_mean(st.u[..., 0], g),
            tau_y_s=-cd * spd1 * sl.planar_mean(st.v[..., 0], g),
            p=p, dpdx=sl.ddx(p, g), dpdy=sl.ddy(p, g), dpdz=dpdz,
            dudx=sl.ddx(st.u, g), dvdy=sl.ddy(st.v, g),
            dwdz=sl.ddz_f2c(st.w, g.dz), wc=sl.f2c(st.w),
            fl13=fl13, fl23=fl23, most_resid=jnp.asarray(most_resid, st.u.dtype))
    return _fields


def _frame_stats(st, g, fields, pr_sgs):
    """Horizontal-mean truth profiles + turbulence statistics for one snapshot.

    The SGS heat flux uses the core's exact face discretization (via emit) with
    the surface flux the core APPLIED at this state (prescribed ``Q0`` or the MOST
    flux under prescribed cooling) — so the emitted ``<w'θ'>_sgs`` equals the flux
    ``scalar_rhs`` integrates. Momentum SGS fluxes come from the core's own face
    stress (``sgs_vertical_momentum_flux_faces``)."""
    a = {k: np.asarray(v) for k, v in fields.items()}
    u, v, th, wc = np.asarray(st.u), np.asarray(st.v), np.asarray(st.theta), a["wc"]
    wf = np.asarray(st.w)
    q = float(a["q"])
    ww_f = covariance(wf, wf)                 # faces (nz+1) via the same reducer
    out = dict(
        theta=horizontal_mean(th), u=horizontal_mean(u), v=horizontal_mean(v),
        wtheta_resolved=resolved_vertical_flux(wc, th),
        wtheta_sgs=sgs_vertical_scalar_flux_mean(th, a["nu_t"], float(g.dz),
                                                 pr_sgs=pr_sgs, surface_flux=q),
        uw_resolved=covariance(u, wc), vw_resolved=covariance(v, wc),
        uw_sgs=sgs_flux_from_face_stress(a["tau13"]),
        vw_sgs=sgs_flux_from_face_stress(a["tau23"]),
        uu=covariance(u, u), vv=covariance(v, v),
        ww=0.5 * (ww_f[:-1] + ww_f[1:]), thth=covariance(th, th),
        wwtheta=third_moment(wc, wc, th),
        wp=covariance(wc, a["p"]), theta_dpdz=covariance(th, a["dpdz"]),
        pstrain_11=covariance(a["p"], a["dudx"]),
        pstrain_22=covariance(a["p"], a["dvdy"]),
        pstrain_33=covariance(a["p"], a["dwdz"]),
        pres_uw=covariance(u, a["dpdz"]) + covariance(wc, a["dpdx"]),
        pres_vw=covariance(v, a["dpdz"]) + covariance(wc, a["dpdy"]),
        wtheta_floor=sgs_vertical_scalar_flux_mean(
            th, np.full_like(th, g.cfg.nu_floor), float(g.dz), pr_sgs=pr_sgs,
            surface_flux=0.0),
        uw_floor=sgs_flux_from_face_stress(a["fl13"]),
        vw_floor=sgs_flux_from_face_stress(a["fl23"]),
    )
    out["tke_resolved"] = 0.5 * (out["uu"] + out["vv"] + out["ww"])
    surface = dict(w_theta_s=q, tau_x_s=float(a["tau_x_s"]),
                   tau_y_s=float(a["tau_y_s"]), most_resid=float(a["most_resid"]))
    return out, surface


def _floor_shares(prof):
    """Column share of the background-viscosity floor in the total (resolved+SGS)
    heat and momentum flux profiles."""
    return (
        column_share(prof["wtheta_floor"], prof["wtheta_resolved"] + prof["wtheta_sgs"]),
        column_share(prof["uw_floor"], prof["uw_resolved"] + prof["uw_sgs"]),
        column_share(prof["vw_floor"], prof["vw_resolved"] + prof["vw_sgs"]),
    )


def _refined_case(case, n):
    """``case`` on an ``n``× finer grid over the same domain, ``dt_s/n``."""
    gr = case.grid
    return dataclasses.replace(case, grid=dataclasses.replace(
        gr, nx=gr.nx * n, ny=gr.ny * n, nz=gr.nz * n, dt_s=gr.dt_s / n))


# Exit status of a run that stopped at its wall-clock limit after writing a
# checkpoint (EX_TEMPFAIL): the job script resubmits on it.
EXIT_CHECKPOINTED = 75
_PROFILE_KEYS_SKIP = ("theta", "u", "v", "wtheta_resolved", "wtheta_sgs")


def _save_checkpoint(path, st, acc, k, col0, profs, sfc, cum, times, fingerprint):
    """Write the full resumable run state atomically (temp file, then rename)."""
    arrs = {f"state__{f}": np.asarray(getattr(st, f)) for f in ("u", "v", "w", "theta")}
    arrs.update({f"prof__{k_}": np.stack(v) for k_, v in profs.items()})
    arrs.update({f"sfc__{k_}": np.asarray(v, dtype=np.float64) for k_, v in sfc.items()})
    arrs.update({f"cum__{k_}": np.asarray(v, dtype=np.float64) for k_, v in cum.items()})
    arrs.update(acc=np.asarray(acc), col0=np.asarray(col0), times=np.asarray(times),
                k_done=np.asarray(k), fingerprint=np.asarray(fingerprint))
    tmp = path.with_name(path.name + ".partial")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrs)
    os.replace(tmp, path)


def _load_checkpoint(path, st, fingerprint, dtype):
    """Restore what :func:`_save_checkpoint` wrote. A checkpoint from a run with a
    different configuration is refused, never silently continued."""
    with np.load(path) as z:
        got = str(z["fingerprint"])
        if got != fingerprint:
            raise SystemExit(f"checkpoint {path} is from a different run "
                             f"configuration:\n  checkpoint: {got}\n  this run:   "
                             f"{fingerprint}")
        fields = {f: jnp.asarray(z[f"state__{f}"], dtype)
                  for f in ("u", "v", "w", "theta")}
        st = st._replace(**fields)
        profs = {k[len("prof__"):]: list(z[k]) for k in z.files if k.startswith("prof__")}
        sfc = {k[len("sfc__"):]: [float(x) for x in z[k]]
               for k in z.files if k.startswith("sfc__")}
        cum = {k[len("cum__"):]: [float(x) for x in z[k]]
               for k in z.files if k.startswith("cum__")}
        return (st, jnp.asarray(z["acc"], dtype), int(z["k_done"]), np.asarray(z["col0"]),
                profs, sfc, cum, [float(t) for t in z["times"]])


def _fingerprint(case, args, sgs, dt0, n_steps, rec_steps, hours, cfg, inputs):
    """Everything that makes two runs the same run, as one stable string: the
    case, run options, the RESOLVED core config (defaults included), a digest of
    the resolved initial state + forcing (a changed forcing file or builder), and
    a digest of the source (every ``legoesm`` .py file + this driver), so a
    checkpoint never resumes under edited code."""
    h = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(inputs):
        a = np.ascontiguousarray(np.asarray(leaf))
        h.update(str((a.dtype, a.shape)).encode())
        h.update(a.tobytes())
    src = hashlib.sha256(Path(__file__).read_bytes())
    for root in sorted(legoesm.__path__):
        for f in sorted(Path(root).rglob("*.py")):
            src.update(str(f.relative_to(root)).encode())
            src.update(f.read_bytes())
    rec = dict(case=dataclasses.asdict(case), sgs=sgs, dt=dt0, n_steps=n_steps,
               rec_steps=list(rec_steps), hours=hours, cfg=cfg._asdict(),
               inputs=h.hexdigest(), source=src.hexdigest(),
               args={k: getattr(args, k) for k in (
                   "pr_sgs", "theta0", "gamma", "zi0", "q0", "Ug", "lat", "z0",
                   "nu_floor", "refine", "f32", "smoke", "label")})
    return json.dumps(rec, sort_keys=True, default=str)


def _distribute(g, st, comm, dtype):
    """Serial grid + GLOBAL initial state -> the y-slab MPI grid + this rank's slab.

    The IC is built (and projected) globally on every rank exactly as in a serial
    run, then sliced, so the distributed initial state is bit-identical to the
    serial one. ``comm=None`` (serial) returns its inputs unchanged."""
    if comm is None:
        return g, st
    n, r = comm.Get_size(), comm.Get_rank()
    if getattr(st, "tracers", None) is not None:
        raise SystemExit("--mpi: tracer-carrying states are not gathered for output "
                         "(dry cases only)")
    ny = g.cfg.ny
    if ny % n:
        raise SystemExit(f"--mpi: ny={ny} must divide by the {n} ranks (y-slab)")
    if g.cfg.dealias and (ny // n) % 2:
        raise SystemExit(f"--mpi: the 3/2-rule de-aliasing needs an even ny per rank; "
                         f"ny={ny} over {n} ranks gives {ny // n}")
    layout = sl.SpectralLESLayout(rank=r, n_ranks=n, ny_global=ny, nx=g.cfg.nx,
                                  comm=comm)
    gd = sl.make_grid(g.cfg, dtype=dtype, layout=layout)
    nyl = ny // n
    rows = slice(r * nyl, (r + 1) * nyl)
    return gd, jax.tree_util.tree_map(lambda a: a[rows], st)  # every leaf is (ny, ...)


def _gather_y(comm, a):
    """Rank 0: the y-slabs of ``a`` stacked into the global array; other ranks:
    ``None``. Arrays of rank < 3 (planar-mean scalars, identical on every rank) and
    serial runs pass through."""
    a = np.asarray(a)
    if comm is None:
        return a
    if a.ndim < 3:
        # only rank-invariant planar-mean scalars may skip the gather; a 2-D slab
        # field passed through would silently stand for the whole plane
        if a.size != 1:
            raise SystemExit(f"_gather_y: a {a.shape} field is neither a (y,x,z) slab "
                             "nor a scalar; gather it explicitly")
        return a
    parts = comm.gather(a, root=0)
    return None if parts is None else np.concatenate(parts, axis=0)


def _all_ok(hcomm, fn, what):
    """Run ``fn`` on every rank and fail on EVERY rank if it failed on any (a
    rank-local exception must not leave the others waiting in a collective)."""
    if hcomm is None:
        return fn()
    out, err = None, None
    try:
        out = fn()
    except (Exception, SystemExit) as exc:  # noqa: BLE001 -- re-raised collectively
        err = f"{type(exc).__name__}: {exc}"
    errs = hcomm.allgather(err)
    bad = [(r, e) for r, e in enumerate(errs) if e is not None]
    if bad:
        raise SystemExit(f"{what} failed on rank(s) {[r for r, _ in bad]}: {bad[0][1]}")
    return out


def _stray_checkpoints(base, n_ranks):
    """Checkpoint files for ``base`` written by a run over a DIFFERENT rank count
    (or serial vs MPI): resuming would silently restart from t=0, so they are
    refused instead."""
    if base is None:
        return []
    found = [q for q in base.parent.glob(base.name + ".r*of*")
             if not q.name.endswith(".partial")
             and not (n_ranks and q.name.endswith(f"of{n_ranks}"))]
    if n_ranks and base.exists():
        found.append(base)
    return sorted(found)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", required=True, help="registry LESCase name")
    p.add_argument("--hours", type=float, default=None,
                   help="integration length [h] (default: the case duration)")
    p.add_argument("--frames", type=int, default=12,
                   help="number of output snapshots recorded into the artifact")
    p.add_argument("--dt", type=float, default=None,
                   help="fixed dt [s] (default: adaptive CFL)")
    p.add_argument("--cfl", type=float, default=0.8)
    p.add_argument("--dt-max", type=float, default=1.0)
    p.add_argument("--max-wind", type=float, default=20.0)
    p.add_argument("--pr-sgs", type=float, default=None,
                   help=f"turbulent Prandtl number (legacy cases; default {_PR_SGS}). "
                        "Refused for a setup case.")
    p.add_argument("--theta0", type=float, default=None,
                   help="reference θ [K] (legacy cases; default by regime: 300 "
                        "convective / 265 GABLS1 stable). Refused for a setup case.")
    p.add_argument("--gamma", type=float, default=None,
                   help="legacy CBL inversion dθ/dz [K/m] (default 0.008). Refused for "
                        "a setup case.")
    p.add_argument("--zi0", type=float, default=None,
                   help="legacy CBL initial inversion height [m] (default 800). "
                        "Refused for a setup case.")
    p.add_argument("--q0", type=float, default=None,
                   help="override the case surface kinematic heat flux [K m/s] "
                        "(the Q1 buoyancy-axis sweep); default = the case value")
    p.add_argument("--Ug", type=float, default=None,
                   help="geostrophic wind |U_g| [m/s] for a SHEARED CBL (the dry-grid "
                        "U_g axis). Default: the case's geostrophic_wind_m_s (0 = free "
                        "convection). U_g>0 initialises u=U_g and drives f-balance.")
    p.add_argument("--lat", type=float, default=None,
                   help="latitude [deg] for the Coriolis f=2Ω sin(φ) (used when U_g>0; "
                        "default by regime: 45° convective / 73° GABLS1 stable → "
                        "f=1.39e-4).")
    p.add_argument("--sgs", type=str, default=None,
                   help="LES SGS closure (the D7 σ_LES spread): one of the case's "
                        "sgs_variants (lasd/smagorinsky/vreman). Default: the case's "
                        "first variant. Sets the artifact's sgs tag + filename.")
    p.add_argument("--label", type=str, default=None,
                   help="extra tag in the artifact filename (e.g. q0 value) to keep "
                        "a flux sweep from overwriting")
    p.add_argument("--z0", type=float, default=None,
                   help="legacy-case roughness [m] (default 0.1). Refused for a "
                        "setup case.")
    p.add_argument("--nu-floor", type=float, default=None,
                   help=f"background eddy-viscosity floor [m^2/s] (SBL stability; "
                        f"legacy cases, default {_NU_FLOOR_M2_S}; refused for a setup "
                        "case); "
                        "its share of the total heat/momentum flux is printed per "
                        "frame and stored in the artifact")
    p.add_argument("--refine", type=int, default=1,
                   help="convergence run: nx, ny, nz x N over the same domain, the "
                        "default dt / N (an explicit --dt is used as given); "
                        "artifact tag __r<N>")
    p.add_argument("--smoke", action="store_true",
                   help="a short test run: allowed for a case whose setup still has "
                        "unapproved (pending) values, artifact tagged __smoke")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="resumable-state file: resumed from when it exists, written "
                        "at every output frame and at --wall-limit-s, removed once "
                        "the artifact is written")
    p.add_argument("--wall-limit-s", type=float, default=None,
                   help="stop after this many wall-clock seconds of stepping, write "
                        f"--checkpoint and exit {EXIT_CHECKPOINTED} (the job script "
                        "resubmits)")
    p.add_argument("--f32", action="store_true")
    p.add_argument("--mpi", action="store_true",
                   help="run the y-slab MPI decomposition over MPI.COMM_WORLD "
                        "(launch under mpiexec); rank 0 writes the artifact")
    p.add_argument("--output", type=Path, default=Path("results/les_suite/artifacts"))
    args = p.parse_args(argv)

    if args.frames < 2:
        raise SystemExit("--frames must be >= 2 (an initial condition + >=1 later)")
    if args.hours is not None and not (args.hours > 0):
        raise SystemExit("--hours must be > 0")
    if args.dt is not None and not (args.dt > 0):
        raise SystemExit("--dt must be > 0")
    if args.wall_limit_s is not None and args.checkpoint is None:
        raise SystemExit("--wall-limit-s needs --checkpoint (somewhere to stop to)")
    if args.smoke and args.hours is None:
        raise SystemExit("--smoke needs --hours (the short window to test)")
    if not list_cases():  # idempotent: only populate an empty registry
        register_default_catalog()
    case = get_case(args.case)
    if case.regime not in _WIRED_REGIMES:
        raise SystemExit(
            f"{case.name}: regime {case.regime!r} emission is not wired yet "
            f"(wired: {list(_WIRED_REGIMES)}). The MOIST IC builders + the D9 cloud "
            "scheme are a follow-up; refusing to emit a wrong-regime artifact.")
    if case.regime == "dry_neutral" and case.setup is None:
        raise SystemExit(f"{case.name}: a dry_neutral case needs a full LESSetup")
    if case.setup is not None and args.hours is not None and not args.smoke:
        raise SystemExit(f"{case.name}: a setup case runs its published duration "
                         f"({case.duration_hours} h); only a --smoke run may set --hours")
    if case.setup is not None and case.setup.pending and not args.smoke:
        raise SystemExit(
            f"{case.name}: its setup has values the user has not approved — a "
            "production artifact is refused until they are (run --smoke to test):\n  "
            + "\n  ".join(case.setup.pending))
    # Resolve the SGS closure (D7 σ_LES spread). Must be one of the case's declared
    # variants — a stray SGS is a hard error (dispatch hardening), never a silent
    # emission of a closure the case never claimed.
    sgs_name = args.sgs if args.sgs is not None else case.sgs_variants[0]
    if sgs_name not in case.sgs_variants:
        raise SystemExit(
            f"{case.name}: --sgs {sgs_name!r} is not one of the case's sgs_variants "
            f"{list(case.sgs_variants)}")
    if args.refine < 1:
        raise SystemExit(f"--refine must be >= 1, got {args.refine}")
    base_dx = case.grid.Lx_m / case.grid.nx
    if args.refine > 1:
        case = _refined_case(case, args.refine)
    dtype = jnp.float32 if args.f32 else jnp.float64
    args.output.mkdir(parents=True, exist_ok=True)
    comm = hcomm = MPI = None
    if args.mpi:
        from mpi4py import MPI  # noqa: PLC0415
        comm = MPI.COMM_WORLD
        # Host-side (mpi4py) collectives use their OWN communicator, so they can
        # never match a pending mpi4jax collective of the jitted step on ``comm``.
        hcomm = comm.Dup()
    rank = 0 if comm is None else comm.Get_rank()

    res = None
    if case.setup is not None:
        # The setup is the record of every scientific input: a CLI value for any
        # of them would be a second, unrecorded source for the same parameter.
        clash = [f for f in ("theta0", "gamma", "zi0", "q0", "Ug", "lat", "z0", "dt",
                             "pr_sgs", "nu_floor")
                 if getattr(args, f) is not None]
        if clash:
            raise SystemExit(f"{case.name}: --{', --'.join(clash)} conflict with the "
                             "case's LESSetup, which defines them")
        args.pr_sgs, args.nu_floor = _PR_SGS, _NU_FLOOR_M2_S
        g, st, Q0, res = _build_setup(case, args, dtype, sgs_name)
        st_ic = st
        g, st = _distribute(g, st, comm, dtype)
        args.theta0 = res["theta_ref"]
        u_geo = (jnp.asarray(res["ug"], dtype), jnp.asarray(res["vg"], dtype))
        u_geo_mag = float(np.max(np.hypot(res["ug"], res["vg"])))
        f_cor = _coriolis_f(res["lat_deg"])
        s = case.setup
        sponge = (None if s.sponge_bottom_m is None
                  else (s.sponge_bottom_m, s.sponge_tau_s))
        step = _make_step(g, u_geo, f_cor, Q0, dtype, sponge=sponge)
        fields_fn = _frame_fields_fn(g, u_geo, f_cor, Q0)
    else:
        # Legacy anchors / dry grid: values not on the case come from the CLI with
        # these documented defaults (recorded in the artifact's les_setup).
        args.gamma = 0.008 if args.gamma is None else args.gamma
        args.zi0 = 800.0 if args.zi0 is None else args.zi0
        args.z0 = 0.1 if args.z0 is None else args.z0
        args.pr_sgs = _PR_SGS if args.pr_sgs is None else args.pr_sgs
        args.nu_floor = _NU_FLOOR_M2_S if args.nu_floor is None else args.nu_floor
        # Regime-appropriate θ0 / latitude when not overridden (the case has flux +
        # U_g only, NOT the sounding baseline or latitude — a bare GABLS1 run must
        # still get 265 K/73°).
        if args.theta0 is None:
            args.theta0 = (_GABLS1_THETA0_K if case.regime == "dry_stable"
                           else _CBL_THETA0_K)
        lat = args.lat if args.lat is not None else (
            _GABLS1_LAT_DEG if case.regime == "dry_stable" else _CBL_LAT_DEG)
        u_geo_mag = args.Ug if args.Ug is not None else (case.geostrophic_wind_m_s
                                                          or 0.0)
        if not math.isfinite(u_geo_mag) or u_geo_mag < 0.0:
            raise SystemExit(f"--Ug must be finite and >= 0, got {u_geo_mag}")
        f_cor = 0.0
        if u_geo_mag > 0.0:
            if not math.isfinite(lat) or not (-90.0 <= lat <= 90.0):
                raise SystemExit(f"--lat must be finite in [-90, 90], got {lat}")
            f_cor = _coriolis_f(lat)
        if case.regime == "dry_stable":
            g, st, Q0 = _build_sbl(case, args, dtype, sgs=sgs_name, u_geo_mag=u_geo_mag)
        else:
            g, st, Q0 = _build_cbl(case, args, dtype, sgs=sgs_name, u_geo_mag=u_geo_mag)
        st_ic = st
        g, st = _distribute(g, st, comm, dtype)
        u_geo = (u_geo_mag, 0.0)
        step = _make_emit_step(g, case, u_geo, f_cor, Q0, dtype)
        fields_fn = _frame_fields_fn(g, u_geo, f_cor, Q0)
    hours = args.hours if args.hours is not None else case.duration_hours
    T = hours * 3600.0
    if args.dt is not None:
        dt0 = float(args.dt)
    elif case.setup is not None or case.regime == "dry_stable":
        dt0 = float(case.grid.dt_s)  # the case's own dt (already / refine)
    else:
        # the base-grid CFL dt, divided by the refinement factor (exactly dt/N)
        dt0 = select_dt(base_dx, max_wind_safe=args.max_wind, cfl_safe=args.cfl,
                        dt_cap=args.dt_max) / args.refine
    if dt0 >= T:
        raise SystemExit(
            f"dt={dt0:.3f}s >= integration length T={T:.1f}s: a single step would "
            "overshoot the whole run. Use a smaller --dt or a longer --hours.")
    dt = jnp.asarray(dt0, dtype)

    # Frame schedule by STEP INDEX (robust to any dt): integrate n_steps ≈ T/dt fixed
    # steps and record at evenly-spaced step indices. Sampling by index (not by
    # crossing continuous targets) guarantees strictly-increasing, DISTINCT snapshot
    # times, the final frame at t=n_steps·dt≈T, and no overshoot past T+dt. If dt is
    # too coarse to resolve `frames` distinct snapshots, we record as many as there
    # are steps and say so (never a silent drop).
    n_steps = max(1, int(round(T / dt0)))
    rec_steps = frame_step_schedule(n_steps, args.frames)
    if len(rec_steps) < args.frames - 1:
        print(f"[warn] dt={dt0:.3f}s over {n_steps} steps cannot resolve "
              f"{args.frames} frames; recording {len(rec_steps) + 1}")

    rate = case.surface_cooling_rate_K_per_h
    series = None if res is None else res["theta_s"]
    if series is not None and n_steps * dt0 > float(series[0][-1]) + 1e-9:
        raise SystemExit(f"{case.name}: the run ({n_steps * dt0:.0f} s) outlasts its "
                         f"surface-temperature series ({float(series[0][-1]):.0f} s)")
    if series is not None:
        sfc_bc, sfc_val = "surface_theta_series", None
        bc_label = (f"theta_s=series[{float(series[1][0]):.2f}..{float(series[1][-1]):.2f}"
                    "K](MOST)")
    elif rate is not None:
        sfc_bc, sfc_val = "cooling_K_per_h", float(rate)
        bc_label = f"cooling={rate}K/h(MOST)"
    else:
        sfc_bc, sfc_val = "flux_K_m_s", float(Q0)
        bc_label = f"Q0={Q0}"
        if res is not None:
            bc_label += f"(drag={g.cfg.sfc_flux_drag})"
    # the prescribed-surface-temperature MOST solver runs (and reports a residual)
    most_path = sfc_bc in ("surface_theta_series", "cooling_K_per_h")
    # cooling base: the case's initial surface θ (setup) or θ0 (GABLS1 anchors)
    t_s0 = (args.theta0 if res is None else float(res["theta"][0]))
    print(f"[les-suite emit] case={case.name} regime={case.regime} "
          f"grid={case.grid.label} {bc_label} Ug={u_geo_mag} f={f_cor:.3e} "
          f"hours={hours} dt={dt0:.4f} n_steps={n_steps} refine={args.refine} "
          f"nu_floor={g.cfg.nu_floor} frames={len(rec_steps) + 1} sgs={sgs_name}"
          f"{' SMOKE' if args.smoke else ''}")

    profs: dict[str, list] = {}
    sfc: dict[str, list] = {"w_theta_s": [], "tau_x_s": [], "tau_y_s": [],
                            "most_resid": []}
    cum = {"w_theta_s_cum": [], "tau_x_s_cum": [], "tau_y_s_cum": []}
    times = []
    n_core = len(sl.BUDGET_TERMS)
    acc = jnp.zeros(n_core + 3, dtype)       # time-integrated budget terms

    def _t_sfc(t_s):
        if series is not None:
            return jnp.asarray(float(np.interp(t_s, series[0], series[1])), dtype)
        ts = _surface_temperature(case, t_s0, t_s)
        return None if ts is None else jnp.asarray(ts, dtype)

    def _col_means(state):
        return np.array([float(g.dz * jnp.sum(sl.planar_mean(f, g)))
                         for f in (state.theta, state.u, state.v)])

    dz_min = min(float(g.dz), case.grid.Lx_m / case.grid.nx,
                 case.grid.Ly_m / case.grid.ny)

    def _record(t):
        # finish every device collective before any host-side MPI call
        fields = jax.block_until_ready(fields_fn(st, _t_sfc(t)))
        vmax = max(float(jnp.max(jnp.abs(f))) for f in (st.u, st.v, st.w))
        if comm is not None:
            vmax = hcomm.allreduce(vmax, op=MPI.MAX)
            st_g = types.SimpleNamespace(**{f: _gather_y(hcomm, getattr(st, f))
                                            for f in ("u", "v", "w", "theta")})
            fields = {k: _gather_y(hcomm, v) for k, v in fields.items()}
            if rank != 0:
                return
        else:
            st_g = st
        prof, srf = _frame_stats(st_g, g, fields, args.pr_sgs)
        for k, val in prof.items():
            profs.setdefault(k, []).append(val)
        for k, val in srf.items():
            sfc[k].append(val)
        a_np = np.asarray(acc)
        cum["w_theta_s_cum"].append(float(a_np[0]))
        cum["tau_x_s_cum"].append(float(a_np[1]))
        cum["tau_y_s_cum"].append(float(a_np[2]))
        times.append(float(t))
        sh, su, sv = _floor_shares(prof)
        print(f"  frame t={t / 3600:.3f}h  nu_floor share of total flux: "
              f"heat={sh:.4f} u-mom={su:.4f} v-mom={sv:.4f}  "
              f"w'θ'_0={srf['w_theta_s']:+.5f} u'w'_0={srf['tau_x_s']:+.5f}  "
              f"CFL<={vmax * dt0 / dz_min:.3f}"
              + (f"  MOST-resid={srf['most_resid']:.1e}" if most_path else "")
              + ("" if clock["t0"] is None
                 else f"  wall={time.time() - clock['t0']:.1f}s"))

    clock = {"t0": None}      # stepping start, for per-frame wall time (cost)
    if args.checkpoint is not None and g.cfg.time_scheme != "rk3":
        # the checkpoint holds u, v, w, θ only — a multistep scheme's previous RHS
        # would be lost on resume
        raise SystemExit(f"--checkpoint supports time_scheme='rk3' only, got "
                         f"{g.cfg.time_scheme!r}")
    fp_inputs = (st_ic, u_geo, Q0, f_cor, res)
    if comm is not None:   # a run over a different rank count is a different run
        fp_inputs = fp_inputs + (np.asarray(comm.Get_size()),)
    fp = _fingerprint(case, args, sgs_name, dt0, n_steps, rec_steps, hours, g.cfg,
                      fp_inputs)
    del st_ic, fp_inputs
    ckpt = args.checkpoint
    n_ranks = 0 if comm is None else comm.Get_size()
    stray = _stray_checkpoints(args.checkpoint, n_ranks)
    if comm is not None:   # one decision for all ranks (rank 0's view of the dir)
        stray = hcomm.bcast(stray, root=0)
    if stray:
        raise SystemExit(f"checkpoint(s) from a run over a different rank count: "
                         f"{[str(q) for q in stray]}; refusing to restart from t=0")
    if ckpt is not None and comm is not None:
        ckpt = ckpt.with_name(f"{ckpt.name}.r{rank}of{n_ranks}")
    have = ckpt is not None and ckpt.exists()
    if comm is not None:
        if hcomm.allreduce(have, op=MPI.LOR) != hcomm.allreduce(have, op=MPI.LAND):
            raise SystemExit(f"rank {rank}: per-rank checkpoints {args.checkpoint}.r* "
                             "exist on some ranks only; refusing a partial resume")
    k_start = 1
    if have:
        st, acc, k_done, col0, profs, sfc, cum, times = _all_ok(
            hcomm, lambda: _load_checkpoint(ckpt, st, fp, dtype), "checkpoint load")
        if comm is not None:
            steps = hcomm.allgather(k_done)
            if len(set(steps)) != 1:
                raise SystemExit("per-rank checkpoints are at different steps "
                                 f"({sorted(set(steps))}); refusing")
        k_start = k_done + 1
        print(f"[RESUME] {args.checkpoint} at step {k_done}/{n_steps}, "
              f"{len(times)} frames recorded")
    else:
        # Record the TRUE initial condition at t=0 (before any step) — the
        # prognostic score initialises the SCM to LES(t=0), so the IC must be the
        # real t=0 state, not a once-stepped state mislabeled t=0.
        col0 = _col_means(st)
        _record(0.0)
    rec_set = set(rec_steps)
    t0 = clock["t0"] = time.time()
    for k in range(k_start, n_steps + 1):
        st, _, b = step(st, dt, _t_sfc((k - 1) * dt0))
        acc = acc + b
        if comm is not None:
            # the step's mpi4jax collectives complete before any host MPI call
            jax.block_until_ready((st, acc))
        if args.wall_limit_s is not None:
            # dispatch is asynchronous: without this the clock would time the
            # enqueue, and the checkpoint write could land past the walltime
            acc.block_until_ready()
        if k in rec_set:
            mw = float(jnp.max(jnp.abs(st.w)))
            bad = not np.isfinite(mw) or mw > 1e3
            if comm is not None:
                bad = hcomm.allreduce(bad, op=MPI.LOR)
            if bad:
                print(f"[BLOWUP] step {k} t={k * dt0:.0f}s max|w|={mw} (rank {rank})")
                return 1
            _record(k * dt0)
            if ckpt is not None:
                _all_ok(hcomm, lambda: _save_checkpoint(
                    ckpt, st, acc, k, col0, profs, sfc, cum, times, fp),
                    "checkpoint write")
        stop = (args.wall_limit_s is not None and k < n_steps
                and time.time() - t0 >= args.wall_limit_s)
        if comm is not None and args.wall_limit_s is not None:
            stop = hcomm.bcast(stop, root=0)   # every rank stops at the same step
        if stop:
            _all_ok(hcomm, lambda: _save_checkpoint(
                ckpt, st, acc, k, col0, profs, sfc, cum, times, fp), "checkpoint write")
            print(f"[CHECKPOINT] wall limit {args.wall_limit_s:.0f}s reached at step "
                  f"{k}/{n_steps} -> {args.checkpoint}; exit {EXIT_CHECKPOINTED}")
            return EXIT_CHECKPOINTED
    print(f"[DONE] wall={time.time()-t0:.1f}s  {len(times)} frames")
    d_col = _col_means(st) - col0          # collective: every rank takes part
    col1 = _col_means(st)
    if rank != 0:
        hcomm.Barrier()                    # rank 0 has saved the artifact
        if ckpt is not None and ckpt.exists():
            ckpt.unlink()
        return 0

    # RULE 0 column budgets: change of the column-integrated mean = boundary +
    # body sources the RK stages applied + the sponge (reported separately).
    a_np = np.asarray(acc)
    names = dict(zip(sl.BUDGET_TERMS, a_np[:n_core]))
    src_th = names["sfc_heat_flux"] + names["lid_heat_flux"]
    src_u = names["sfc_stress_x"] + names["coriolis_u"] + names["body_force_u"]
    src_v = names["sfc_stress_y"] + names["coriolis_v"] + names["body_force_v"]
    # Round-off is set by the column CONTENT (e.g. ∫θdz ~ 1e5 K m), so the residual
    # is printed next to it as well as next to the change.
    for i, (lab, src, sp) in enumerate((("theta", src_th, a_np[n_core]),
                                        ("u", src_u, a_np[n_core + 1]),
                                        ("v", src_v, a_np[n_core + 2]))):
        dc = d_col[i]
        print(f"  budget {lab}: d(column)={dc:+.6e} sources={src:+.6e} "
              f"sponge={sp:+.6e} residual={dc - src - sp:+.3e} "
              f"column={col1[i]:+.6e}")

    # Times are strictly increasing + distinct by construction (index sampling).
    times_arr = np.asarray(times)
    recs_z = np.asarray(g.z_c)
    stats = {k: np.stack(v) for k, v in profs.items() if k not in _PROFILE_KEYS_SKIP}
    if res is not None:
        u_geo_art, v_geo_art = np.asarray(res["ug"]), np.asarray(res["vg"])
    elif u_geo_mag > 0.0:
        u_geo_art, v_geo_art = np.full_like(recs_z, u_geo_mag), np.zeros_like(recs_z)
    else:
        u_geo_art = v_geo_art = None
    les_setup = {
        "surface_bc": sfc_bc,
        "surface_bc_value": sfc_val,
        "nu_floor_m2_s": float(g.cfg.nu_floor),
        "refine": int(args.refine),
        "dt_s": float(dt0),
        "grid": case.grid.label,
        "domain_m": [case.grid.Lx_m, case.grid.Ly_m, case.grid.Lz_m],
        "theta0_K": float(args.theta0),
        "u_geo_m_s": float(u_geo_mag),
        "pr_sgs": float(args.pr_sgs),
        "z0_m": float(g.cfg.z0),
        "z0h_m": (None if g.cfg.z0h is None else float(g.cfg.z0h)),
        "sfc_flux_drag": g.cfg.sfc_flux_drag,
        "held_out": bool(case.held_out),
        "smoke": bool(args.smoke),
        "f32": bool(args.f32),
        "mpi_ranks": 1 if comm is None else int(comm.Get_size()),
        # prescribed-surface-temperature MOST solver: worst convergence residual over
        # the output frames (None on the prescribed-flux paths)
        # (and how many frames' residuals were not finite — a failed solve is
        # counted, never dropped)
        "most_residual_max": (
            None if not most_path or not np.any(np.isfinite(sfc["most_resid"]))
            else float(np.max(np.asarray(sfc["most_resid"])[
                np.isfinite(sfc["most_resid"])]))),
        "most_residual_nonfinite_frames": (
            None if not most_path
            else int(np.sum(~np.isfinite(sfc["most_resid"])))),
    }
    if case.setup is not None:
        les_setup["setup"] = json.loads(json.dumps(dataclasses.asdict(case.setup)))
    artifact = build_reference_artifact(
        case_name=case.name,
        sgs=sgs_name,
        z=recs_z,
        times_s=times_arr,
        theta=np.stack(profs["theta"]),
        u=np.stack(profs["u"]),
        v=np.stack(profs["v"]),
        wtheta_resolved=np.stack(profs["wtheta_resolved"]),
        wtheta_sgs=np.stack(profs["wtheta_sgs"]),
        subsidence_w=np.zeros_like(recs_z),   # dry: no large-scale subsidence
        prescribe="fluxes",
        # the surface heat flux the LES APPLIED at each output time (the prescribed
        # Q0, or the MOST flux under prescribed cooling / a θ_s series) — the SCM
        # gets this series.
        w_theta_s=np.asarray(sfc["w_theta_s"]),
        # geostrophic forcing the SCM must receive too: the bridge rebuilds SCMForcing
        # from these so the SCM arm sees the SAME U_g(z)/f. Free-convective legacy
        # cases (U_g=0) leave u_geo/v_geo absent (None).
        u_geo=u_geo_art,
        v_geo=v_geo_art,
        f_c=f_cor,
        tau_x_s=np.asarray(sfc["tau_x_s"]),
        tau_y_s=np.asarray(sfc["tau_y_s"]),
        **{k: np.asarray(v) for k, v in cum.items()},
        **stats,
        les_setup=les_setup,
    )
    # Sheared legacy runs auto-tag with U_g so they never overwrite the
    # free-convective ``{case}__{sgs}.npz``; --label adds any further distinction.
    shear_tag = (f"__ug{u_geo_mag:g}" if res is None and u_geo_mag > 0.0 else "")
    refine_tag = f"__r{args.refine}" if args.refine > 1 else ""
    label_tag = f"__{args.label}" if args.label else ""
    smoke_tag = "__smoke" if args.smoke else ""
    out = args.output / (f"{case.name}__{sgs_name}{shear_tag}{refine_tag}{label_tag}"
                         f"{smoke_tag}.npz")
    save_artifact(artifact, out)
    print(f"  artifact -> {out}  (nt={artifact.nt}, nz={artifact.nz})")
    if comm is not None:
        hcomm.Barrier()
    if ckpt is not None and ckpt.exists():
        ckpt.unlink()
    return 0


def _run_cli() -> None:
    """Under ``--mpi`` any failure on one rank aborts the whole job: a rank that
    exits (an exception, a rank-local refusal, a failed artifact write on rank 0)
    must never leave the others waiting in a collective until the walltime."""
    if "--mpi" in sys.argv:
        from mpi4py import MPI  # noqa: PLC0415
        if MPI.COMM_WORLD.Get_rank() != 0:   # one log: rank 0's (errors stay on stderr)
            sys.stdout = open(os.devnull, "w")  # noqa: SIM115
    try:
        rc = main()
    except BaseException as exc:
        if "--mpi" not in sys.argv:
            raise
        import traceback  # noqa: PLC0415
        traceback.print_exc()
        code = exc.code if isinstance(exc, SystemExit) else 1
        if code in (0, None):
            raise
        from mpi4py import MPI  # noqa: PLC0415
        MPI.COMM_WORLD.Abort(code if isinstance(code, int) else 1)
        raise
    sys.exit(rc)


if __name__ == "__main__":
    _run_cli()
