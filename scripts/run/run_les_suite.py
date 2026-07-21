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

GPU-gated (a 96³ sim-hour is ~7 min on a V100S). Currently the **dry free-convective
CBL** regime is wired end-to-end (the gate-0-validated path); the stably-stratified
and moist regimes need their own IC builders and are an explicit follow-up — the
driver raises rather than silently emitting a wrong-regime artifact.

Usage::

    JAX_PLATFORMS=cuda .venv/bin/python scripts/run/run_les_suite.py \\
        --case cbl_nieuwstadt --hours 2.0 --frames 12 \\
        --output results/les_suite/artifacts
"""
from __future__ import annotations

import argparse
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
# NOT a silent wrong-regime emission) until their builders land.
_WIRED_REGIMES = ("dry_convective",)


def _build_cbl(case, args, dtype):
    """Dry free-convective CBL IC + config (mirrors run_spectral_cbl.build)."""
    theta0 = args.theta0
    cfg = sl.SpectralLESConfig(
        nx=case.grid.nx, ny=case.grid.ny, nz=case.grid.nz,
        Lx=case.grid.Lx_m, Ly=case.grid.Ly_m, Lz=case.grid.Lz_m,
        z0=args.z0, dealias=True, smagorinsky_dynamic=True,
        sgs_model="smagorinsky", time_scheme="rk3",
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
    u = jnp.zeros((cfg.ny, cfg.nx, cfg.nz), dtype)
    v = jnp.zeros((cfg.ny, cfg.nx, cfg.nz), dtype)
    w = jnp.zeros((cfg.ny, cfg.nx, cfg.nz + 1), dtype)
    st = sl.SpectralLESState(
        u=u, v=v, w=w, rhs_u_prev=jnp.zeros_like(u), rhs_v_prev=jnp.zeros_like(v),
        rhs_w_prev=jnp.zeros_like(w), theta=th3, rhs_theta_prev=jnp.zeros_like(th3))
    Q0 = case.surface_theta_flux_K_m_s
    if Q0 is None:
        raise SystemExit(f"{case.name}: dry CBL requires surface_theta_flux_K_m_s")
    return g, st, float(Q0)


def _mean_profiles(st, g, pr_sgs):
    """Horizontal-mean (θ, u, v) + resolved & SGS heat flux for one snapshot."""
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
    wth_sgs = sgs_vertical_scalar_flux_mean(th, nu_t, z, pr_sgs=pr_sgs)
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
    p.add_argument("--theta0", type=float, default=300.0)
    p.add_argument("--gamma", type=float, default=0.008, help="inversion dθ/dz [K/m]")
    p.add_argument("--zi0", type=float, default=800.0, help="initial inversion height [m]")
    p.add_argument("--z0", type=float, default=0.1)
    p.add_argument("--f32", action="store_true")
    p.add_argument("--output", type=Path, default=Path("results/les_suite/artifacts"))
    args = p.parse_args(argv)

    if not list_cases():  # idempotent: only populate an empty registry
        register_default_catalog()
    case = get_case(args.case)
    if case.regime not in _WIRED_REGIMES:
        raise SystemExit(
            f"{case.name}: regime {case.regime!r} emission is not wired yet "
            f"(wired: {list(_WIRED_REGIMES)}). The stable/moist IC builders are a "
            "follow-up; refusing to emit a wrong-regime artifact.")
    dtype = jnp.float32 if args.f32 else jnp.float64
    args.output.mkdir(parents=True, exist_ok=True)

    g, st, Q0 = _build_cbl(case, args, dtype)
    hours = args.hours if args.hours is not None else case.duration_hours
    T = hours * 3600.0
    step = jax.jit(partial(sl.step, g=g, u_geo=(0.0, 0.0), f_cor=0.0,
                           force=(0.0, 0.0), sfc_theta_flux=Q0),
                   static_argnames=("first",))
    dt0 = (float(args.dt) if args.dt is not None else
           select_dt(g.dx, max_wind_safe=args.max_wind, cfl_safe=args.cfl,
                     dt_cap=args.dt_max))
    dt = jnp.asarray(dt0, dtype)
    frame_times = np.linspace(0.0, T, args.frames)

    sgs_name = case.sgs_variants[0]
    print(f"[les-suite emit] case={case.name} regime={case.regime} "
          f"grid={case.grid.label} Q0={Q0} hours={hours} dt={dt0:.3f} sgs={sgs_name}")

    # record t=0, then integrate, snapshotting nearest each frame time.
    recs_z = None
    theta_s, u_s, v_s, wthr_s, wths_s, times = [], [], [], [], [], []

    def _record(t):
        nonlocal recs_z
        z, thm, um, vm, wr, ws = _mean_profiles(st, g, args.pr_sgs)
        recs_z = z
        theta_s.append(thm)
        u_s.append(um)
        v_s.append(vm)
        wthr_s.append(wr)
        wths_s.append(ws)
        times.append(float(t))

    st, _ = step(st, dt=dt, first=True)
    _record(0.0)
    t = float(dt)
    fi = 1
    t0 = time.time()
    while t < T and fi < len(frame_times):
        target = frame_times[fi]
        while t < target:
            st, _ = step(st, dt=dt, first=False)
            t += float(dt)
        mw = float(jnp.max(jnp.abs(st.w)))
        if not np.isfinite(mw) or mw > 1e3:
            print(f"[BLOWUP] t={t:.0f}s max|w|={mw}")
            return 1
        _record(t)
        fi += 1
    print(f"[DONE] wall={time.time()-t0:.1f}s  {len(times)} frames")

    # de-duplicate strictly-increasing times (the t=0 + first-frame may coincide)
    times_arr = np.asarray(times)
    keep = np.concatenate([[True], np.diff(times_arr) > 0])
    sel = np.where(keep)[0]

    artifact = build_reference_artifact(
        case_name=case.name,
        sgs=sgs_name,
        z=recs_z,
        times_s=times_arr[sel],
        theta=np.stack(theta_s)[sel],
        u=np.stack(u_s)[sel],
        v=np.stack(v_s)[sel],
        wtheta_resolved=np.stack(wthr_s)[sel],
        wtheta_sgs=np.stack(wths_s)[sel],
        subsidence_w=np.zeros_like(recs_z),   # free-convective CBL: no subsidence
        prescribe="fluxes",
        w_theta_s=np.full(sel.shape[0], Q0),
        f_c=0.0,
    )
    out = args.output / f"{case.name}__{sgs_name}.npz"
    save_artifact(artifact, out)
    print(f"  artifact -> {out}  (nt={artifact.nt}, nz={artifact.nz})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
