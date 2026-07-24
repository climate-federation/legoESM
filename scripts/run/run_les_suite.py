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
Rayleigh sponge) regimes are wired end-to-end; the MOIST regimes need their own IC
builders + the D9 cloud scheme and are an explicit follow-up — the driver raises rather
than silently emitting a wrong-regime artifact.

Usage::

    JAX_PLATFORMS=cuda .venv/bin/python scripts/run/run_les_suite.py \\
        --case cbl_nieuwstadt --hours 2.0 --frames 12 \\
        --output results/les_suite/artifacts
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

_F32 = "--f32" in sys.argv
import jax  # noqa: E402

if not _F32:
    jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from legoesm.atmosphere.dynamics.les import spectral_les_plane as sl  # noqa: E402
from legoesm.atmosphere.les_suite import (  # noqa: E402
    get_case,
    list_cases,
    register_default_catalog,
)
from legoesm.atmosphere.les_suite.bridge import save_artifact  # noqa: E402
from legoesm.atmosphere.les_suite.emit import (  # noqa: E402
    build_reference_artifact,
    horizontal_mean,
    resolved_vertical_flux,
    sgs_vertical_scalar_flux_mean,
)
from legoesm.timestepping.split_explicit import select_dt  # noqa: E402

# Regimes whose IC/forcing builder is wired here. The rest raise (honest dispatch,
# NOT a silent wrong-regime emission) until their builders land. Moist regimes
# (shallow_cumulus, stratocumulus) still need their IC builders + the D9 cloud scheme.
_WIRED_REGIMES = ("dry_convective", "dry_stable")


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
    Q0 = args.q0 if args.q0 is not None else case.surface_theta_flux_K_m_s
    if Q0 is None:
        raise SystemExit(f"{case.name}: SBL requires surface_theta_flux_K_m_s (cooling)")
    return g, st, float(Q0)


def _make_emit_step(g, regime, u_geo, f_cor, Q0, Lz, dtype):
    """The jitted per-step map for one regime. Dry CBL = plain ``sl.step``; SBL adds the
    Rayleigh sponge in the top 25% (+ a re-projection to restore ∇·u=0 after the sponge)
    so the jet/gravity-waves reaching the rigid lid are absorbed, not reflected."""
    if regime == "dry_convective":
        return jax.jit(partial(sl.step, g=g, u_geo=u_geo, f_cor=f_cor,
                               force=(0.0, 0.0), sfc_theta_flux=Q0),
                       static_argnames=("first",))
    if regime == "dry_stable":
        zc, zf, z_sp = g.z_c, g.z_f, 0.75 * Lz
        spc = jnp.where(zc > z_sp, 0.5 * (1.0 - jnp.cos(
            jnp.pi * (zc - z_sp) / (Lz - z_sp))), 0.0).astype(dtype)
        spf = jnp.where(zf > z_sp, 0.5 * (1.0 - jnp.cos(
            jnp.pi * (zf - z_sp) / (Lz - z_sp))), 0.0).astype(dtype)

        @partial(jax.jit, static_argnames=("first",))
        def _sbl_step(state, dt, first=False):
            state, us = sl.step(state, g=g, dt=dt, u_geo=u_geo, f_cor=f_cor,
                                force=(0.0, 0.0), sfc_theta_flux=Q0, first=first)
            rc = (dt / _SBL_TAU_SPONGE_S) * spc
            rf = (dt / _SBL_TAU_SPONGE_S) * spf
            u = state.u - rc * (state.u - state.u.mean((0, 1), keepdims=True))
            v = state.v - rc * (state.v - state.v.mean((0, 1), keepdims=True))
            w = state.w - rf * state.w
            th = state.theta - rc * (state.theta - state.theta.mean((0, 1), keepdims=True))
            u, v, w = sl.project(u, v, w, dt=dt, g=g)  # sponge broke ∇·u=0 → re-project
            return state._replace(u=u, v=v, w=w, theta=th), us
        return _sbl_step
    raise SystemExit(f"no emit step wired for regime {regime!r}")


def _mean_profiles(st, g, pr_sgs, Q0):
    """Horizontal-mean (θ, u, v) + resolved & SGS heat flux for one snapshot.

    The SGS heat flux uses the core's exact face discretization (via emit) with the
    prescribed surface kinematic flux ``Q0`` on the surface face — so the emitted
    ``<w'θ'>_sgs`` equals the flux ``scalar_rhs`` integrated.
    """
    u = np.asarray(st.u)
    v = np.asarray(st.v)
    wc = np.asarray(sl.f2c(st.w))
    th = np.asarray(st.theta)
    z = np.asarray(g.z_c)
    nu_t = np.asarray(sl.eddy_viscosity(st.u, st.v, st.w, g))
    theta_m = horizontal_mean(th)
    u_m = horizontal_mean(u)
    v_m = horizontal_mean(v)
    wth_res = resolved_vertical_flux(wc, th)
    wth_sgs = sgs_vertical_scalar_flux_mean(
        th, nu_t, float(g.dz), pr_sgs=pr_sgs, surface_flux=Q0)
    return z, theta_m, u_m, v_m, wth_res, wth_sgs


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
    p.add_argument("--pr-sgs", type=float, default=1.0, help="turbulent Prandtl number")
    p.add_argument("--theta0", type=float, default=None,
                   help="reference θ [K] (default by regime: 300 convective / 265 "
                        "GABLS1 stable)")
    p.add_argument("--gamma", type=float, default=0.008, help="inversion dθ/dz [K/m]")
    p.add_argument("--zi0", type=float, default=800.0, help="initial inversion height [m]")
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
    p.add_argument("--z0", type=float, default=0.1)
    p.add_argument("--nu-floor", type=float, default=0.05,
                   help="background eddy-viscosity floor [m^2/s] (SBL stability)")
    p.add_argument("--f32", action="store_true")
    p.add_argument("--output", type=Path, default=Path("results/les_suite/artifacts"))
    args = p.parse_args(argv)

    if args.frames < 2:
        raise SystemExit("--frames must be >= 2 (an initial condition + >=1 later)")
    if args.hours is not None and not (args.hours > 0):
        raise SystemExit("--hours must be > 0")
    if args.dt is not None and not (args.dt > 0):
        raise SystemExit("--dt must be > 0")
    if not list_cases():  # idempotent: only populate an empty registry
        register_default_catalog()
    case = get_case(args.case)
    if case.regime not in _WIRED_REGIMES:
        raise SystemExit(
            f"{case.name}: regime {case.regime!r} emission is not wired yet "
            f"(wired: {list(_WIRED_REGIMES)}). The MOIST IC builders + the D9 cloud "
            "scheme are a follow-up; refusing to emit a wrong-regime artifact.")
    # Resolve the SGS closure (D7 σ_LES spread). Must be one of the case's declared
    # variants — a stray SGS is a hard error (dispatch hardening), never a silent
    # emission of a closure the case never claimed.
    sgs_name = args.sgs if args.sgs is not None else case.sgs_variants[0]
    if sgs_name not in case.sgs_variants:
        raise SystemExit(
            f"{case.name}: --sgs {sgs_name!r} is not one of the case's sgs_variants "
            f"{list(case.sgs_variants)}")
    # Geostrophic wind (U_g axis) + its Coriolis f. U_g=0 ⇒ free-convective (f=0, u=0),
    # dynamically unchanged from the pre-shear path; U_g>0 ⇒ sheared CBL (u₀=U_g, f from
    # --lat). NaN/inf must be rejected (a NaN comparison is False → would slip through).
    # Regime-appropriate θ0 / latitude when not overridden (the case has flux + U_g only,
    # NOT the sounding baseline or latitude — a bare GABLS1 run must still get 265 K/73°).
    if args.theta0 is None:
        args.theta0 = _GABLS1_THETA0_K if case.regime == "dry_stable" else _CBL_THETA0_K
    lat = args.lat if args.lat is not None else (
        _GABLS1_LAT_DEG if case.regime == "dry_stable" else _CBL_LAT_DEG)
    u_geo_mag = args.Ug if args.Ug is not None else (case.geostrophic_wind_m_s or 0.0)
    if not math.isfinite(u_geo_mag) or u_geo_mag < 0.0:
        raise SystemExit(f"--Ug must be finite and >= 0, got {u_geo_mag}")
    f_cor = 0.0
    if u_geo_mag > 0.0:
        if not math.isfinite(lat) or not (-90.0 <= lat <= 90.0):
            raise SystemExit(f"--lat must be finite in [-90, 90], got {lat}")
        f_cor = _coriolis_f(lat)
    dtype = jnp.float32 if args.f32 else jnp.float64
    args.output.mkdir(parents=True, exist_ok=True)

    if case.regime == "dry_stable":
        g, st, Q0 = _build_sbl(case, args, dtype, sgs=sgs_name, u_geo_mag=u_geo_mag)
    else:
        g, st, Q0 = _build_cbl(case, args, dtype, sgs=sgs_name, u_geo_mag=u_geo_mag)
    hours = args.hours if args.hours is not None else case.duration_hours
    T = hours * 3600.0
    step = _make_emit_step(g, case.regime, (u_geo_mag, 0.0), f_cor, Q0,
                           case.grid.Lz_m, dtype)
    if args.dt is not None:
        dt0 = float(args.dt)
    elif case.regime == "dry_stable":
        dt0 = float(case.grid.dt_s)  # SBL is stiffer (stratification) — use the case dt
    else:
        dt0 = select_dt(g.dx, max_wind_safe=args.max_wind, cfl_safe=args.cfl,
                        dt_cap=args.dt_max)
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

    print(f"[les-suite emit] case={case.name} regime={case.regime} "
          f"grid={case.grid.label} Q0={Q0} Ug={u_geo_mag} f={f_cor:.3e} "
          f"hours={hours} dt={dt0:.3f} n_steps={n_steps} "
          f"frames={len(rec_steps) + 1} sgs={sgs_name}")

    recs_z = None
    theta_s, u_s, v_s, wthr_s, wths_s, times = [], [], [], [], [], []

    def _record(t):
        nonlocal recs_z
        z, thm, um, vm, wr, ws = _mean_profiles(st, g, args.pr_sgs, Q0)
        recs_z = z
        theta_s.append(thm)
        u_s.append(um)
        v_s.append(vm)
        wthr_s.append(wr)
        wths_s.append(ws)
        times.append(float(t))

    # Record the TRUE initial condition at t=0 (before any step) — the prognostic
    # score initialises the SCM to LES(t=0), so the IC must be the real t=0 state,
    # not a once-stepped state mislabeled t=0.
    _record(0.0)
    rec_set = set(rec_steps)
    first = True
    t0 = time.time()
    for k in range(1, n_steps + 1):
        st, _ = step(st, dt=dt, first=first)
        first = False
        if k in rec_set:
            mw = float(jnp.max(jnp.abs(st.w)))
            if not np.isfinite(mw) or mw > 1e3:
                print(f"[BLOWUP] step {k} t={k * dt0:.0f}s max|w|={mw}")
                return 1
            _record(k * dt0)
    print(f"[DONE] wall={time.time()-t0:.1f}s  {len(times)} frames")

    # Times are strictly increasing + distinct by construction (index sampling).
    times_arr = np.asarray(times)
    artifact = build_reference_artifact(
        case_name=case.name,
        sgs=sgs_name,
        z=recs_z,
        times_s=times_arr,
        theta=np.stack(theta_s),
        u=np.stack(u_s),
        v=np.stack(v_s),
        wtheta_resolved=np.stack(wthr_s),
        wtheta_sgs=np.stack(wths_s),
        subsidence_w=np.zeros_like(recs_z),   # dry CBL: no large-scale subsidence
        prescribe="fluxes",
        w_theta_s=np.full(times_arr.shape[0], Q0),
        # geostrophic forcing the SCM must receive too: the bridge rebuilds SCMForcing
        # from these so the SCM arm sees the SAME U_g/f. Free-convective (U_g=0) leaves
        # u_geo/v_geo absent (None) — byte-identical to the pre-shear artifacts.
        u_geo=(np.full_like(recs_z, u_geo_mag) if u_geo_mag > 0.0 else None),
        v_geo=(np.zeros_like(recs_z) if u_geo_mag > 0.0 else None),
        f_c=f_cor,
    )
    # Sheared runs auto-tag with U_g so they never overwrite the free-convective
    # ``{case}__{sgs}.npz`` (or a different-shear run); --label adds any further distinction.
    shear_tag = f"__ug{u_geo_mag:g}" if u_geo_mag > 0.0 else ""
    label_tag = f"__{args.label}" if args.label else ""
    out = args.output / f"{case.name}__{sgs_name}{shear_tag}{label_tag}.npz"
    save_artifact(artifact, out)
    print(f"  artifact -> {out}  (nt={artifact.nt}, nz={artifact.nz})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
