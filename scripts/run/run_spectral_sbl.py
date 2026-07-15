"""Stable boundary layer (SBL) on the pseudo-spectral incompressible core.

GABLS1-style (Beare et al. 2006 / Cuxart et al. 2006; oracle GABLS3 analog):
geostrophic wind ``Ug`` + Coriolis ``f`` driving an Ekman SBL, a stable initial
sounding, and a prescribed NEGATIVE surface kinematic heat flux (surface
cooling). The dynamic Bou-Zeid LASD SGS shuts mixing off in the strongly-stable
layers. Validates the canonical SBL features every faithful LES reproduces:

* a SHALLOW, statically-STABLE boundary layer (∂θ/∂z > 0 maintained, not mixed
  away) of depth O(150-250 m);
* SUSTAINED (not collapsed) weak turbulence — u_* in O(0.2-0.4 m/s);
* a super-geostrophic LOW-LEVEL JET near the SBL top (inertial oscillation).

Saves ``sbl_profiles.npz``.

Usage
-----
.. code-block:: bash

   JAX_PLATFORMS=cuda .venv/bin/python scripts/run/run_spectral_sbl.py \\
       --nx 64 --ny 64 --nz 96 --Lx 400 --Ly 400 --Lz 400 \\
       --Ug 8 --Q0 -0.012 --hours 4 --f32
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
        z0=args.z0, dealias=True, c_s=args.cs,
        smagorinsky_dynamic=not args.static, sgs_model=args.sgs_model,
        time_scheme=args.time_scheme,
        buoyancy=True, theta_ref0=args.theta0, pr_sgs=1.0, nu_floor=args.nu_floor)
    g = sl.make_grid(cfg, dtype=dtype)
    z = g.z_c
    # GABLS1 sounding: θ0 below 100 m, +0.01 K/m capping inversion above —
    # SMOOTHED over ~25 m (a step in ∂θ/∂z is unresolved and seeds a Gibbs/KH
    # instability at the inversion where the dynamic SGS shuts off). The smooth
    # ramp is the integral of 0.5·γ·(1+tanh((z−zi)/Δi)).
    zi, di, gam = 100.0, 25.0, 0.01
    th = (args.theta0 + 0.5 * gam * ((z - zi) + di * jnp.log(
        jnp.cosh((z - zi) / di)) + di * jnp.log(2.0))).astype(dtype)
    key = jax.random.PRNGKey(0)
    seed = (z < 0.5 * args.Lz).astype(dtype)
    th3 = jnp.broadcast_to(th, (args.ny, args.nx, args.nz)).astype(dtype) + (
        0.1 * jax.random.normal(key, (args.ny, args.nx, args.nz), dtype=dtype) * seed)
    # Geostrophic mean wind + small perturbation to trigger turbulence.
    u = jnp.full((args.ny, args.nx, args.nz), args.Ug, dtype) + (
        0.1 * jax.random.normal(jax.random.PRNGKey(1),
                                (args.ny, args.nx, args.nz), dtype) * seed)
    v = 0.1 * jax.random.normal(jax.random.PRNGKey(2),
                                (args.ny, args.nx, args.nz), dtype) * seed
    w = jnp.zeros((args.ny, args.nx, args.nz + 1), dtype)
    u, v, w = sl.project(u, v, w, dt=args.dt, g=g)
    st = sl.SpectralLESState(
        u=u, v=v, w=w, rhs_u_prev=jnp.zeros_like(u), rhs_v_prev=jnp.zeros_like(v),
        rhs_w_prev=jnp.zeros_like(w), theta=th3, rhs_theta_prev=jnp.zeros_like(th3))
    return g, st


def diagnose(st, g):
    u = np.asarray(st.u); v = np.asarray(st.v)
    wc = np.asarray(sl.f2c(st.w)); th = np.asarray(st.theta)
    z = np.asarray(g.z_c)
    um = u.mean((0, 1)); vm = v.mean((0, 1)); thm = th.mean((0, 1))
    spd = np.sqrt(um ** 2 + vm ** 2)
    up = u - um; vp = v - vm; wp = wc - wc.mean((0, 1))
    uw = (up * wp).mean((0, 1)); vw = (vp * wp).mean((0, 1))
    # Resolved-stress u_* PROXY at k=1 (k=0 is wall-damped: SGS carries the
    # near-wall stress). The authoritative surface u_* is the MOST wall value
    # (printed at run end); this is a secondary resolved-turbulence check.
    ustar = float((uw[1] ** 2 + vw[1] ** 2) ** 0.25)
    # SBL depth: where the stress falls to 5% of its surface value.
    tau = np.sqrt(uw ** 2 + vw ** 2)
    h = float(z[np.where(tau > 0.05 * tau[1])[0][-1]]) if tau[1] > 1e-6 else 0.0
    jet = float(spd.max()); jetz = float(z[np.argmax(spd)])
    return z, um, vm, thm, spd, ustar, h, jet, jetz


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nx", type=int, default=64)
    p.add_argument("--ny", type=int, default=64)
    p.add_argument("--nz", type=int, default=96)
    p.add_argument("--Lx", type=float, default=400.0)
    p.add_argument("--Ly", type=float, default=400.0)
    p.add_argument("--Lz", type=float, default=400.0)
    p.add_argument("--z0", type=float, default=0.1)
    p.add_argument("--theta0", type=float, default=265.0)
    p.add_argument("--Ug", type=float, default=8.0)
    p.add_argument("--fcor", type=float, default=1.39e-4)
    p.add_argument("--nu-floor", type=float, default=0.05, help="background eddy-viscosity floor [m2/s]")
    p.add_argument("--Q0", type=float, default=-0.005,
                   help="PRESCRIBED surface kinematic heat flux [K m/s] "
                        "(negative = cooling); used only when --cooling-rate==0.")
    p.add_argument("--cooling-rate", type=float, default=0.0,
                   help="GABLS1-canonical surface COOLING rate [K/hr]: the surface "
                        "temperature is cooled T_sfc(t)=theta0 - rate*t and the "
                        "surface heat flux + stability-corrected drag are derived "
                        "from the coupled stable MOST surface layer. 0 disables "
                        "(falls back to the prescribed --Q0 flux). GABLS1 uses 0.25.")
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--hours", type=float, default=4.0)
    p.add_argument("--f32", action="store_true")
    p.add_argument("--static", action="store_true",
                   help="static Smagorinsky (C_s) instead of dynamic LASD — "
                        "needed on anisotropic dx<dz grids where LASD destabilises.")
    p.add_argument("--cs", type=float, default=0.18, help="static Smagorinsky C_s")
    p.add_argument("--sgs-model", choices=["smagorinsky", "vreman"],
                   default="vreman", help="static SGS closure (ignored if not --static)")
    p.add_argument("--time-scheme", choices=["rk3", "ab2"], default="rk3")
    p.add_argument("--adaptive-dt", action="store_true",
                   help="CFL-adaptive dt (recomputed every --dt-recompute steps)")
    p.add_argument("--cfl", type=float, default=0.8, help="advective CFL target")
    p.add_argument("--dt-max", type=float, default=0.5, help="cap on adaptive dt [s]")
    p.add_argument("--max-wind", type=float, default=20.0,
                   help="conservative max resolved speed [m/s] for the static-CFL dt")
    p.add_argument("--dt-recompute", type=int, default=25)
    p.add_argument("--print-every", type=int, default=2000)
    p.add_argument("--record-frames", type=int, default=20,
                   help="evenly-spaced frames (snapshots + profiles); 0 disables.")
    p.add_argument("--case-label", type=str, default="gabls1")
    p.add_argument("--output", type=Path, default=Path("results/spectral_sbl"))
    args = p.parse_args()
    dtype = jnp.float32 if args.f32 else jnp.float64
    args.output.mkdir(parents=True, exist_ok=True)

    g, st = build(args, dtype)
    # Rayleigh sponge in the top 25%: absorb the resolved fluctuations + w (and
    # relax θ to its horizontal mean) so the low-level jet / GWs reaching the
    # stress-free rigid lid are damped rather than reflected (without it the SBL
    # jet piles at the lid and the run blows up ~0.9 h — Beare et al. 2006 use a
    # sponge here too). Mirrors the neutral channel driver's sponge.
    zc = g.z_c; zf = g.z_f; z_sp = 0.75 * args.Lz
    spc = jnp.where(zc > z_sp,
                    0.5 * (1.0 - jnp.cos(jnp.pi * (zc - z_sp) / (args.Lz - z_sp))),
                    0.0).astype(dtype)
    spf = jnp.where(zf > z_sp,
                    0.5 * (1.0 - jnp.cos(jnp.pi * (zf - z_sp) / (args.Lz - z_sp))),
                    0.0).astype(dtype)
    tau_sp = 50.0

    cooling = args.cooling_rate > 0.0   # GABLS1 prescribed-cooling MOST surface BC

    @partial(jax.jit, static_argnames=("first",))
    def step(state, dt, t_sfc, first=False):
        state, us = sl.step(state, g=g, dt=dt, u_geo=(args.Ug, 0.0),
                            f_cor=args.fcor, force=(0.0, 0.0),
                            sfc_theta_flux=args.Q0, t_sfc=t_sfc, first=first)
        rc = (dt / tau_sp) * spc                            # sponge from live dt
        rf = (dt / tau_sp) * spf
        u = state.u - rc * (state.u - state.u.mean((0, 1), keepdims=True))
        v = state.v - rc * (state.v - state.v.mean((0, 1), keepdims=True))
        w = state.w - rf * state.w
        th = state.theta - rc * (state.theta - state.theta.mean((0, 1), keepdims=True))
        # Sponge damping breaks ∇·u=0; re-project before the next step (codex
        # physics review 2026-06-09). θ is a scalar — unaffected by projection.
        u, v, w = sl.project(u, v, w, dt=dt, g=g)
        return state._replace(u=u, v=v, w=w, theta=th), us
    sfc_bc = (f"cooling={args.cooling_rate}K/hr (MOST)" if cooling
              else f"Q0={args.Q0}")
    print(f"[spectral-SBL] {args.nx}x{args.ny}x{args.nz} Lz={args.Lz}m Ug={args.Ug} "
          f"f={args.fcor:.2e} {sfc_bc} {args.time_scheme} "
          f"sgs={'LASD' if not args.static else args.sgs_model} "
          f"dt={'adaptive cfl='+str(args.cfl) if args.adaptive_dt else args.dt}")

    def t_sfc_at(t_sec):
        """Prescribed (cooled) surface temperature for the MOST BC; None disables
        the coupled surface layer (prescribed-flux fallback)."""
        if not cooling:
            return None
        return jnp.asarray(args.theta0 - args.cooling_rate * (t_sec / 3600.0), dtype)
    rec = args.record_frames > 0
    if rec:
        zc_np = np.asarray(g.z_c)
        h_idx, h_z = les_record.select_heights(zc_np, args.Lz)
        frame = 0

        def _save(t_hours):
            nonlocal frame
            les_record.record_frame(
                args.output, frame, t_hours, args.case_label, zc_np,
                np.asarray(st.u), np.asarray(st.v), np.asarray(sl.f2c(st.w)),
                np.asarray(st.theta), args.Lx, args.Ly, h_idx, h_z, args.z0)
            frame += 1
        print(f"  recording {args.record_frames} frames; heights[m]={np.round(h_z,1)}")

    # STATIC trace-time CFL dt (reuses split_explicit.select_dt; differentiable /
    # scan-friendly — not a per-step re-pin).
    dt0 = (select_dt(g.dx, max_wind_safe=args.max_wind, cfl_safe=args.cfl,
                     dt_cap=args.dt_max) if args.adaptive_dt else float(args.dt))
    dt = jnp.asarray(dt0, dtype)
    T = args.hours * 3600.0
    print(f"  dt={dt0:.3f}s ({'static-CFL' if args.adaptive_dt else 'fixed'})")
    st, us = step(st, dt, t_sfc_at(0.0), first=True)
    if rec:
        _save(0.0)
    t = float(dt); i = 1
    next_rec = T / args.record_frames if rec else np.inf
    blk = max(1, args.dt_recompute)
    t0 = time.time()
    while t < T:
        dth = float(dt)
        ts = t_sfc_at(t)               # T_sfc held over the block (drift ~3e-4 K)
        for _ in range(blk):
            st, us = step(st, dt, ts, first=False)
        t += blk * dth; i += blk
        mw = float(jnp.max(jnp.abs(st.w)))
        if not np.isfinite(mw) or mw > 1e3:
            print(f"[BLOWUP] step {i} t={t:.0f}s max|w|={mw}"); return 1
        if rec and t >= next_rec and frame < args.record_frames:
            _save(t / 3600.0); next_rec += T / args.record_frames
        if (i // blk) % max(1, args.print_every // blk) == 0:
            z, um, vm, thm, spd, ustar, h, jet, jetz = diagnose(st, g)
            print(f"{i:7d} {t/3600:5.2f}h dt={dth:.3f} max|w|={mw:5.2f} u*={ustar:.3f}"
                  f" h_sbl={h:.0f}m jet={jet:.2f}@{jetz:.0f}m")
    if rec and frame < args.record_frames:
        _save(t / 3600.0)
    print(f"[DONE] wall={time.time()-t0:.1f}s  {i/(time.time()-t0):.1f} steps/s ({i} steps)")
    z, um, vm, thm, spd, ustar, h, jet, jetz = diagnose(st, g)
    np.savez(args.output / "sbl_profiles.npz", z=z, u=um, v=vm, theta=thm,
             spd=spd, u_star=ustar, h_sbl=h, jet=jet, jetz=jetz)
    print(f"  SBL depth h={h:.0f} m (target ~150-250)  u_*={ustar:.3f} m/s "
          f"(target ~0.2-0.4, >0 = not collapsed)")
    print(f"  low-level jet={jet:.2f} m/s @ {jetz:.0f} m (super-geostrophic > Ug={args.Ug})")
    print(f"  stable? θ(top)-θ(sfc)={thm[int(args.nz*0.6)]-thm[0]:+.2f} K (>0 = stable)")
    if cooling:
        spd1 = float(np.sqrt(um[0] ** 2 + vm[0] ** 2))
        u_s, th_s, q0, _cd = sl.most_surface_flux(
            spd1, float(thm[0]), float(args.theta0 - args.cooling_rate * (t / 3600.0)),
            float(g.z_c[0]), args.z0, args.theta0)
        print(f"  MOST surface: T_sfc={float(args.theta0 - args.cooling_rate*(t/3600.0)):.2f} K"
              f"  w'θ'_0={float(q0):+.4f} K m/s (GABLS1 ref ~-0.012)  "
              f"u*_MOST={float(u_s):.3f} (ref ~0.27)")
    print(f"  profiles -> {args.output}/sbl_profiles.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
