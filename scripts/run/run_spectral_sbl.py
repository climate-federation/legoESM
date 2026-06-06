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
from legoesm.atmosphere.dynamics import spectral_les_plane as sl  # noqa: E402


def build(args, dtype):
    cfg = sl.SpectralLESConfig(
        nx=args.nx, ny=args.ny, nz=args.nz, Lx=args.Lx, Ly=args.Ly, Lz=args.Lz,
        z0=args.z0, dealias=True, smagorinsky_dynamic=True,
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
                   help="surface heat flux [K m/s] (negative = cooling)")
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--hours", type=float, default=4.0)
    p.add_argument("--f32", action="store_true")
    p.add_argument("--print-every", type=int, default=2000)
    p.add_argument("--output", type=Path, default=Path("results/spectral_sbl"))
    args = p.parse_args()
    dtype = jnp.float32 if args.f32 else jnp.float64
    args.output.mkdir(parents=True, exist_ok=True)

    g, st = build(args, dtype)
    step = jax.jit(partial(sl.step, g=g, dt=args.dt, u_geo=(args.Ug, 0.0),
                           f_cor=args.fcor, force=(0.0, 0.0),
                           sfc_theta_flux=args.Q0), static_argnames=("first",))
    nsteps = int(args.hours * 3600.0 / args.dt)
    print(f"[spectral-SBL] {args.nx}x{args.ny}x{args.nz} Lz={args.Lz}m Ug={args.Ug} "
          f"f={args.fcor:.2e} Q0={args.Q0} steps={nsteps} dt={args.dt}")
    st, us = step(st, first=True)
    t0 = time.time()
    for i in range(1, nsteps + 1):
        st, us = step(st, first=False)
        if i % args.print_every == 0:
            z, um, vm, thm, spd, ustar, h, jet, jetz = diagnose(st, g)
            mw = float(jnp.max(jnp.abs(st.w)))
            if not np.isfinite(mw) or mw > 1e3:
                print(f"[BLOWUP] step {i} max|w|={mw}"); return 1
            print(f"{i:7d} {i*args.dt/3600:5.2f}h max|w|={mw:5.2f} u*={ustar:.3f}"
                  f" h_sbl={h:.0f}m jet={jet:.2f}@{jetz:.0f}m"
                  f" dθ(sfc-top)={thm[0]-thm[int(args.nz*0.6)]:.2f}K")
    print(f"[DONE] wall={time.time()-t0:.1f}s  {nsteps/(time.time()-t0):.1f} steps/s")
    z, um, vm, thm, spd, ustar, h, jet, jetz = diagnose(st, g)
    np.savez(args.output / "sbl_profiles.npz", z=z, u=um, v=vm, theta=thm,
             spd=spd, u_star=ustar, h_sbl=h, jet=jet, jetz=jetz)
    print(f"  SBL depth h={h:.0f} m (target ~150-250)  u_*={ustar:.3f} m/s "
          f"(target ~0.2-0.4, >0 = not collapsed)")
    print(f"  low-level jet={jet:.2f} m/s @ {jetz:.0f} m (super-geostrophic > Ug={args.Ug})")
    print(f"  stable? θ(top)-θ(sfc)={thm[int(args.nz*0.6)]-thm[0]:+.2f} K (>0 = stable)")
    print(f"  profiles -> {args.output}/sbl_profiles.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
