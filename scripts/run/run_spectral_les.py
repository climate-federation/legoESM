"""Neutral ABL on the pseudo-spectral INCOMPRESSIBLE plane-LES core.

Pressure-gradient-driven neutral channel (constant body force ``f_x = u_*²/Lz``
⇒ a target surface friction velocity and a Monin–Obukhov log-law equilibrium in
a few eddy turnovers). This is the clean validation case against the jax-alfa
oracle's physical target — the log law, ``φ_m→1`` and the resolved-variance
similarity that any faithful ABL LES must reproduce. Unlike the compressible
plane dycore, this core SUSTAINS the resolved turbulence
(``les_plane_turbulence_notes.md``).

Saves ``final_profiles.npz`` (mean U, resolved uu/vv/ww/tke, uw/vw flux, u_*) in
the SAME format as ``run_les_plane.py`` so
``scripts/validate/validate_les_vs_oracle.py`` compares it to MOST directly.

Usage
-----
.. code-block:: bash

   JAX_PLATFORMS=cuda .venv/bin/python scripts/run/run_spectral_les.py \\
       --nx 96 --ny 96 --nz 96 --Lx 2000 --Ly 2000 --Lz 1000 \\
       --ustar 0.45 --dt 0.4 --hours 1.5 --f32
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

_KAPPA = constants.kappa_von_karman


def build(args, dtype):
    cfg = sl.SpectralLESConfig(
        nx=args.nx, ny=args.ny, nz=args.nz, Lx=args.Lx, Ly=args.Ly, Lz=args.Lz,
        z0=args.z0, c_s=args.cs, wall_damping=True, dealias=True,
        smagorinsky_dynamic=args.dynamic)
    g = sl.make_grid(cfg, dtype=dtype)
    # Log-law mean IC (shear from t=0) + divergence-free small perturbations.
    z = g.z_c
    u_tar = args.ustar / _KAPPA * jnp.log(jnp.clip(z, args.z0, None) / args.z0)
    key = jax.random.PRNGKey(0)
    amp = args.ic_amp * args.ustar / _KAPPA   # IC perturbation (fraction of bulk)
    u = jnp.broadcast_to(u_tar, (args.ny, args.nx, args.nz)).astype(dtype) + (
        amp * jax.random.normal(key, (args.ny, args.nx, args.nz), dtype=dtype))
    v = amp * jax.random.normal(jax.random.PRNGKey(1),
                                (args.ny, args.nx, args.nz), dtype=dtype)
    w = jnp.zeros((args.ny, args.nx, args.nz + 1), dtype=dtype)
    w = w.at[..., 1:args.nz].set(
        0.5 * amp * jax.random.normal(jax.random.PRNGKey(2),
                                      (args.ny, args.nx, args.nz - 1), dtype=dtype))
    u, v, w = sl.project(u, v, w, dt=args.dt, g=g)          # divergence-free IC
    st = sl.SpectralLESState(u=u, v=v, w=w, rhs_u_prev=jnp.zeros_like(u),
                             rhs_v_prev=jnp.zeros_like(v), rhs_w_prev=jnp.zeros_like(w))
    force = (args.ustar ** 2 / args.Lz, 0.0)               # constant PG body force
    return g, st, force


def profiles(st, g):
    u = np.asarray(st.u); v = np.asarray(st.v)
    wc = np.asarray(sl.f2c(st.w))
    um = u.mean((0, 1)); vm = v.mean((0, 1)); wm = wc.mean((0, 1))
    up, vp, wp = u - um, v - vm, wc - wm
    uw = (up * wp).mean((0, 1)); vw = (vp * wp).mean((0, 1))
    uu = (up * up).mean((0, 1)); vv = (vp * vp).mean((0, 1)); ww = (wp * wp).mean((0, 1))
    return um, vm, uu, vv, ww, uw, vw


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nx", type=int, default=96)
    p.add_argument("--ny", type=int, default=96)
    p.add_argument("--nz", type=int, default=96)
    p.add_argument("--Lx", type=float, default=2000.0)
    p.add_argument("--Ly", type=float, default=2000.0)
    p.add_argument("--Lz", type=float, default=1000.0)
    p.add_argument("--z0", type=float, default=0.1)
    p.add_argument("--cs", type=float, default=0.25,
                   help="Smagorinsky C_s. ~0.25-0.30 + 3/2 dealiasing reproduces MOST; the Bou-Zeid LASD dynamic coefficient is the proper fix.")
    p.add_argument("--ustar", type=float, default=0.45, help="target u_* [m/s]")
    p.add_argument("--dt", type=float, default=0.4)
    p.add_argument("--hours", type=float, default=1.5)
    p.add_argument("--f32", action="store_true")
    p.add_argument("--dynamic", action="store_true", help="Bou-Zeid LASD scale-dependent dynamic C_s(x,y,z)")
    p.add_argument("--ic-amp", type=float, default=0.05, help="IC perturbation as fraction of bulk wind")
    p.add_argument("--tau-bulk", type=float, default=100.0, help="bulk-relax timescale [s]")
    p.add_argument("--print-every", type=int, default=1000)
    p.add_argument("--output", type=Path, default=Path("results/spectral_neutral"))
    args = p.parse_args()
    dtype = jnp.float32 if args.f32 else jnp.float64
    args.output.mkdir(parents=True, exist_ok=True)

    g, st, force = build(args, dtype)
    # CONSTANT-MASS-FLUX forcing: pin the bulk ⟨u⟩ to the log-law target each step
    # so the surface drag cannot collapse the mean wind (standard channel-LES
    # forcing). u_* then emerges from the resolved+SGS surface stress at that
    # bulk velocity and the BL equilibrates to Monin–Obukhov in a few turnovers,
    # rather than the ~14-turnover spin-up of a fixed body force.
    z = g.z_c
    u_bulk_target = float(jnp.mean(
        args.ustar / _KAPPA * jnp.log(jnp.clip(z, args.z0, None) / args.z0)))

    # Rayleigh sponge in the top 25%: relax the resolved fluctuations toward
    # their horizontal mean and damp w, so turbulent momentum/energy transported
    # to the stress-free rigid lid is absorbed rather than accumulating there
    # (standard ABL-LES; without it the half-channel piles momentum at the lid).
    zc = g.z_c
    z_sp = 0.75 * args.Lz
    spc = jnp.where(zc > z_sp,
                    0.5 * (1.0 - jnp.cos(jnp.pi * (zc - z_sp) / (args.Lz - z_sp))),
                    0.0).astype(dtype)               # (nz,) 0→1 ramp
    zf = g.z_f
    spf = jnp.where(zf > z_sp,
                    0.5 * (1.0 - jnp.cos(jnp.pi * (zf - z_sp) / (args.Lz - z_sp))),
                    0.0).astype(dtype)
    tau_sp = 50.0                                    # sponge timescale [s]
    rc = (args.dt / tau_sp) * spc
    rf = (args.dt / tau_sp) * spf

    # INTEGRAL-CONTROLLED body force: a uniform fx(t) adapts SLOWLY to hold the
    # bulk ⟨u⟩ at target. Unlike the exact per-step velocity re-pin (which
    # re-injected the turbulent ⟨u⟩-fluctuation energy each step → wvar ~5× too
    # high), the force integrates the bulk error on a slow timescale tau_bulk, so
    # it tracks only the MEAN drag, not the fast fluctuations. fx is carried
    # across steps as an explicit state.
    fx0 = jnp.asarray(args.ustar ** 2 / args.Lz, dtype=dtype)
    gain = 1.0 / args.tau_bulk

    @partial(jax.jit, static_argnames=("first",))
    def step(state, fx, first=False):
        state, us = sl.step(state, g=g, dt=args.dt, u_geo=(0.0, 0.0),
                            f_cor=0.0, first=first, force=(fx, 0.0))
        u, v, w = state.u, state.v, state.w
        u = u - rc * (u - u.mean((0, 1), keepdims=True))   # sponge: damp fluctuations
        v = v - rc * (v - v.mean((0, 1), keepdims=True))
        w = w - rf * w                                       # sponge: damp w toward 0
        fx = fx + gain * (u_bulk_target - jnp.mean(u))      # slow integral control
        return state._replace(u=u, v=v, w=w), fx, us
    nsteps = int(args.hours * 3600.0 / args.dt)
    tau = args.Lz / args.ustar                              # eddy turnover [s]
    print(f"[spectral-LES neutral] {args.nx}x{args.ny}x{args.nz} "
          f"L=({args.Lx},{args.Ly},{args.Lz}) m  u*_tar={args.ustar}  dt={args.dt}s "
          f"steps={nsteps}  turnover~{tau:.0f}s  dtype={dtype.__name__}")
    fx = fx0
    st, fx, us = step(st, fx, first=True)
    t0 = time.time()
    for i in range(1, nsteps + 1):
        st, fx, us = step(st, fx, first=False)
        if i % args.print_every == 0:
            wc = sl.f2c(st.w)
            wv = float(jnp.mean(wc ** 2)); mw = float(jnp.max(jnp.abs(st.w)))
            spd = float(jnp.mean(jnp.sqrt(st.u ** 2 + st.v ** 2)))
            if not np.isfinite(mw) or mw > 1e3:
                print(f"[BLOWUP] step {i} max|w|={mw}"); return 1
            print(f"{i:7d} {i*args.dt:8.0f}s  max|w|={mw:6.3f}  wvar={wv:7.4f}  "
                  f"u*={float(us):.3f}  <spd>={spd:5.2f}")
    wall = time.time() - t0
    print(f"[DONE] wall={wall:.1f}s  {nsteps/wall:.1f} steps/s")
    um, vm, uu, vv, ww, uw, vw = profiles(st, g)
    u_star_res = float((uw[0] ** 2 + vw[0] ** 2) ** 0.25)   # RESOLVED stress only
    u_star = float(us)   # TOTAL surface stress (wall model = resolved+SGS) — the
    #                      physically-correct MOST u_* (the resolved part alone is
    #                      low because the near-wall surface layer is under-resolved
    #                      on a uniform grid and the SGS carries the rest).
    z = np.asarray(g.z_c)
    np.savez(args.output / "final_profiles.npz",
             z=z, theta=np.full_like(z, 290.0), u=um, v=vm,
             wvar=ww, uu=uu, vv=vv, ww=ww, tke=0.5 * (uu + vv + ww),
             uw=uw, vw=vw, u_star=u_star, u_star_resolved=u_star_res,
             z0=args.z0, case="neutral_spectral")
    print(f"  profiles -> {args.output}/final_profiles.npz  (u*_total≈{u_star:.3f}"
          f", u*_resolved≈{u_star_res:.3f}, target {args.ustar})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
