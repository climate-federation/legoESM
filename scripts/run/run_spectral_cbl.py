"""Convective boundary layer (CBL) on the pseudo-spectral incompressible core.

Buoyancy-driven free-convection CBL with a prescribed surface kinematic heat flux
``Q0`` and a capping inversion (the Nieuwstadt et al. 1993 / oracle ``CBL_N91``
intercomparison setup). Validates the UNIVERSAL convective scaling the jax-alfa
oracle and every faithful CBL LES reproduce:

* convective velocity  ``w_* = (g/θ · Q0 · z_i)^{1/3}`` and ``max σ_w/w_* ≈ 0.6``
* a WELL-MIXED layer (∂θ/∂z ≈ 0 through the bulk of the CBL)
* the canonical heat-flux profile ``⟨w'θ'⟩/Q0`` decreasing ~linearly from 1 at
  the surface to ≈ −0.2 at the inversion ``z_i`` (entrainment), then → 0.

Uses the dynamic Bou-Zeid LASD SGS. Saves ``cbl_profiles.npz`` for assessment.

Usage
-----
.. code-block:: bash

   JAX_PLATFORMS=cuda .venv/bin/python scripts/run/run_spectral_cbl.py \\
       --nx 96 --ny 96 --nz 96 --Lx 3200 --Ly 3200 --Lz 1600 \\
       --Q0 0.06 --zi0 800 --hours 1.0 --f32
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

from legoesm import constants  # noqa: E402
from legoesm.atmosphere.dynamics.les import spectral_les_plane as sl  # noqa: E402
from legoesm.timestepping.split_explicit import select_dt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import les_record  # noqa: E402


def build(args, dtype):
    cfg = sl.SpectralLESConfig(
        nx=args.nx, ny=args.ny, nz=args.nz, Lx=args.Lx, Ly=args.Ly, Lz=args.Lz,
        z0=args.z0, dealias=True, c_s=args.cs, nu_floor=args.nu_floor,
        smagorinsky_dynamic=not args.static, sgs_model=args.sgs_model,
        time_scheme=args.time_scheme,
        buoyancy=True, theta_ref0=args.theta0, pr_sgs=1.0)
    g = sl.make_grid(cfg, dtype=dtype)
    z = g.z_c
    # Mixed layer at θ0 below z_i0, capping inversion +γ above.
    th = jnp.where(z > args.zi0, args.theta0 + args.gamma * (z - args.zi0),
                   args.theta0).astype(dtype)
    key = jax.random.PRNGKey(0)
    th3 = jnp.broadcast_to(th, (args.ny, args.nx, args.nz)).astype(dtype) + (
        0.1 * jax.random.normal(key, (args.ny, args.nx, args.nz), dtype=dtype)
        * (z < args.zi0).astype(dtype))
    u = jnp.zeros((args.ny, args.nx, args.nz), dtype)
    v = jnp.zeros((args.ny, args.nx, args.nz), dtype)
    w = jnp.zeros((args.ny, args.nx, args.nz + 1), dtype)
    st = sl.SpectralLESState(
        u=u, v=v, w=w, rhs_u_prev=jnp.zeros_like(u), rhs_v_prev=jnp.zeros_like(v),
        rhs_w_prev=jnp.zeros_like(w), theta=th3, rhs_theta_prev=jnp.zeros_like(th3))
    return g, st


def diagnose(st, g, args):
    u = np.asarray(st.u); v = np.asarray(st.v)
    wc = np.asarray(sl.f2c(st.w)); th = np.asarray(st.theta)
    z = np.asarray(g.z_c)
    thm = th.mean((0, 1))
    wp = wc - wc.mean((0, 1)); thp = th - thm
    ww = (wp * wp).mean((0, 1))
    wth = (wp * thp).mean((0, 1))                       # resolved kinematic heat flux
    # inversion height z_i = height of max ∂θ/∂z.
    dthdz = np.gradient(thm, z)
    zi = float(z[3 + int(np.argmax(dthdz[3:]))])
    wstar = (constants.g / args.theta0 * args.Q0 * zi) ** (1.0 / 3.0)
    return z, thm, ww, wth, zi, wstar


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nx", type=int, default=96)
    p.add_argument("--ny", type=int, default=96)
    p.add_argument("--nz", type=int, default=96)
    p.add_argument("--Lx", type=float, default=3200.0)
    p.add_argument("--Ly", type=float, default=3200.0)
    p.add_argument("--Lz", type=float, default=1600.0)
    p.add_argument("--z0", type=float, default=0.1)
    p.add_argument("--theta0", type=float, default=300.0)
    p.add_argument("--gamma", type=float, default=0.008, help="inversion dθ/dz [K/m]")
    p.add_argument("--zi0", type=float, default=800.0, help="initial inversion height [m]")
    p.add_argument("--Q0", type=float, default=0.06, help="surface heat flux [K m/s]")
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--hours", type=float, default=1.0)
    p.add_argument("--f32", action="store_true")
    p.add_argument("--static", action="store_true",
                   help="static Smagorinsky (C_s) instead of dynamic LASD — "
                        "needed on anisotropic dx<dz grids where LASD destabilises.")
    p.add_argument("--cs", type=float, default=0.18, help="static Smagorinsky C_s")
    p.add_argument("--nu-floor", type=float, default=0.0,
                   help="background eddy-viscosity floor [m²/s] (damps residual "
                        "high-k energy where ν_t vanishes, e.g. above the inversion).")
    p.add_argument("--sgs-model", choices=["smagorinsky", "vreman"],
                   default="vreman", help="static SGS closure (ignored if not --static)")
    p.add_argument("--time-scheme", choices=["rk3", "ab2"], default="rk3")
    p.add_argument("--adaptive-dt", action="store_true",
                   help="CFL-adaptive dt (recomputed every --dt-recompute steps)")
    p.add_argument("--cfl", type=float, default=0.8, help="advective CFL target")
    p.add_argument("--dt-max", type=float, default=1.0, help="cap on adaptive dt [s]")
    p.add_argument("--max-wind", type=float, default=20.0,
                   help="conservative max resolved speed [m/s] for the static-CFL dt")
    p.add_argument("--dt-recompute", type=int, default=25)
    p.add_argument("--print-every", type=int, default=1000)
    p.add_argument("--record-frames", type=int, default=20,
                   help="evenly-spaced frames (snapshots + profiles); 0 disables.")
    p.add_argument("--case-label", type=str, default="wangara")
    p.add_argument("--output", type=Path, default=Path("results/spectral_cbl"))
    args = p.parse_args()
    dtype = jnp.float32 if args.f32 else jnp.float64
    args.output.mkdir(parents=True, exist_ok=True)

    g, st = build(args, dtype)
    step = jax.jit(partial(sl.step, g=g, u_geo=(0.0, 0.0), f_cor=0.0,
                           force=(0.0, 0.0), sfc_theta_flux=args.Q0),
                   static_argnames=("first",))
    wstar0 = (constants.g / args.theta0 * args.Q0 * args.zi0) ** (1.0 / 3.0)
    tstar = args.zi0 / wstar0
    print(f"[spectral-CBL] {args.nx}x{args.ny}x{args.nz} Lz={args.Lz}m Q0={args.Q0} "
          f"zi0={args.zi0}m {args.time_scheme} "
          f"sgs={'LASD' if not args.static else args.sgs_model} "
          f"dt={'adaptive cfl='+str(args.cfl) if args.adaptive_dt else args.dt} "
          f"w*0~{wstar0:.2f} t*~{tstar:.0f}s")
    rec = args.record_frames > 0
    if rec:
        zc_np = np.asarray(g.z_c)
        h_idx, h_z = les_record.select_heights(zc_np, args.Lz)
        frame = 0
        _last_rec_h = [None]

        def _save(t_hours):
            nonlocal frame
            les_record.record_frame(
                args.output, frame, t_hours, args.case_label, zc_np,
                np.asarray(st.u), np.asarray(st.v), np.asarray(sl.f2c(st.w)),
                np.asarray(st.theta), args.Lx, args.Ly, h_idx, h_z, args.z0)
            frame += 1
            _last_rec_h[0] = t_hours
        print(f"  recording {args.record_frames} frames; heights[m]={np.round(h_z,1)}")

    # STATIC trace-time CFL dt (reuses split_explicit.select_dt; differentiable /
    # scan-friendly — not a per-step re-pin).
    dt0 = (select_dt(g.dx, max_wind_safe=args.max_wind, cfl_safe=args.cfl,
                     dt_cap=args.dt_max) if args.adaptive_dt else float(args.dt))
    dt = jnp.asarray(dt0, dtype)
    T = args.hours * 3600.0
    print(f"  dt={dt0:.3f}s ({'static-CFL' if args.adaptive_dt else 'fixed'})")
    st, us = step(st, dt=dt, first=True)
    if rec:
        _save(0.0)
    t = float(dt); i = 1
    next_rec = T / args.record_frames if rec else np.inf
    blk = max(1, args.dt_recompute)
    t0 = time.time()
    while t < T:
        dth = float(dt)
        for _ in range(blk):
            st, us = step(st, dt=dt, first=False)
        t += blk * dth; i += blk
        mw = float(jnp.max(jnp.abs(st.w)))
        if not np.isfinite(mw) or mw > 1e3:
            print(f"[BLOWUP] step {i} t={t:.0f}s max|w|={mw}"); return 1
        if rec and t >= next_rec and frame < args.record_frames:
            _save(t / 3600.0); next_rec += T / args.record_frames
        if (i // blk) % max(1, args.print_every // blk) == 0:
            z, thm, ww, wth, zi, ws = diagnose(st, g, args)
            print(f"{i:7d} {t:7.0f}s dt={dth:.3f} max|w|={mw:5.2f} "
                  f"sigw/w*={np.sqrt(ww.max())/ws:.2f} zi={zi:.0f}m th_sfc={thm[0]:.2f}")
    # ALWAYS record the terminal state. The initial-state frame consumes one
    # slot of --record-frames, so the in-loop cadence stops one step short of
    # t = T and the old `frame < record_frames` guard was already exhausted
    # here -- every reference silently ended one cadence step early (measured:
    # --record-frames 3 over 0.25 h gave t = 0, 0.083, 0.167 h, never 0.25).
    # Guarded on the time, not the count, so it cannot emit a duplicate frame.
    if rec and (_last_rec_h[0] is None or t / 3600.0 > _last_rec_h[0] + 1.0e-9):
        _save(t / 3600.0)
    print(f"[DONE] wall={time.time()-t0:.1f}s  {i/(time.time()-t0):.1f} steps/s ({i} steps)")
    z, thm, ww, wth, zi, ws = diagnose(st, g, args)
    np.savez(args.output / "cbl_profiles.npz", z=z, theta=thm, ww=ww, wth=wth,
             zi=zi, wstar=ws, Q0=args.Q0)
    # Convective-scaling assessment.
    print(f"  z_i={zi:.0f} m  w_*={ws:.2f} m/s  max σ_w/w_*={np.sqrt(ww.max())/ws:.2f}"
          f" (target ~0.6)")
    kzi = int(np.argmin(np.abs(z - zi)))
    print(f"  ⟨w'θ'⟩/Q0 at surface={wth[1]/args.Q0:+.2f} (target ~1), "
          f"at z_i={wth[kzi]/args.Q0:+.2f} (target ~−0.2, entrainment)")
    sl_lo, sl_hi = int(0.2*kzi), int(0.8*kzi)
    dthdz_ml = float(np.gradient(thm, z)[sl_lo:sl_hi].mean())
    print(f"  mixed-layer ∂θ/∂z={dthdz_ml*1000:.3f} mK/m (target ~0, well-mixed)")
    print(f"  profiles -> {args.output}/cbl_profiles.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
