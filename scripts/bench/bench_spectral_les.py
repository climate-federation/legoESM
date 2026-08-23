"""Wall-clock benchmark for the pseudo-spectral incompressible LES core.

Measures the per-step cost of :func:`legoesm.atmosphere.dynamics.spectral_les_plane.step`
in the production CBL configuration (dynamic Bou-Zeid LASD SGS + Boussinesq
buoyancy + 3/2 dealiasing + SSP-RK3), so the ``LES_SUITE`` run-count budget
(``docs/atmosphere/les_suite/LES_SUITE.md`` §D8) can be converted into real GPU
hours on the target hardware before any ensemble is launched.

This is a TIMING harness, not a physics run: the initial condition is a mixed
layer + capping inversion + small noise (the CBL shape), which exercises the same
operators/shapes as a production run but is not integrated long enough to be
physically meaningful. For the science run use ``scripts/run/run_spectral_cbl.py``.

The reported "sim-hours per wall-hour" uses the supplied ``--dt`` as the model
step; with the CFL-adaptive controller a production run picks its own dt, so pass
the dt you intend to run at (default 0.5 s, the ``run_spectral_cbl`` default).

Usage
-----
.. code-block:: bash

   JAX_PLATFORMS=cuda .venv/bin/python scripts/bench/bench_spectral_les.py \\
       --nx 96 --ny 96 --nz 96 --steps 200 --dt 0.5
   # sweep a few sizes:
   JAX_PLATFORMS=cuda .venv/bin/python scripts/bench/bench_spectral_les.py \\
       --sizes 64 96 128 --steps 100
"""
from __future__ import annotations

import argparse
import sys
import time
from functools import partial

import numpy as np

_F32 = "--f32" in sys.argv
import jax  # noqa: E402

if not _F32:
    jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from legoesm.atmosphere.dynamics.les import spectral_les_plane as sl  # noqa: E402


def build_state(n: int, cfg: sl.SpectralLESConfig, dtype):
    """CBL-shaped IC (mixed layer + inversion + noise) at resolution ``n``³-ish."""
    g = sl.make_grid(cfg, dtype=dtype)
    z = g.z_c
    theta0, zi0, gamma = 300.0, cfg.Lz * 0.5, 0.008
    th = jnp.where(z > zi0, theta0 + gamma * (z - zi0), theta0).astype(dtype)
    key = jax.random.PRNGKey(0)
    th3 = jnp.broadcast_to(th, (cfg.ny, cfg.nx, cfg.nz)).astype(dtype) + (
        0.1 * jax.random.normal(key, (cfg.ny, cfg.nx, cfg.nz), dtype=dtype)
        * (z < zi0).astype(dtype))
    u = jnp.zeros((cfg.ny, cfg.nx, cfg.nz), dtype)
    v = jnp.zeros((cfg.ny, cfg.nx, cfg.nz), dtype)
    w = jnp.zeros((cfg.ny, cfg.nx, cfg.nz + 1), dtype)
    st = sl.SpectralLESState(
        u=u, v=v, w=w, rhs_u_prev=jnp.zeros_like(u), rhs_v_prev=jnp.zeros_like(v),
        rhs_w_prev=jnp.zeros_like(w), theta=th3, rhs_theta_prev=jnp.zeros_like(th3))
    return g, st


def bench_one(nx: int, ny: int, nz: int, *, Lx: float, Ly: float, Lz: float,
              steps: int, dt: float, dtype, sgs_dynamic: bool) -> dict:
    cfg = sl.SpectralLESConfig(
        nx=nx, ny=ny, nz=nz, Lx=Lx, Ly=Ly, Lz=Lz, z0=0.1, dealias=True,
        smagorinsky_dynamic=sgs_dynamic, sgs_model="vreman",
        time_scheme="rk3", buoyancy=True, theta_ref0=300.0, pr_sgs=1.0)
    g, st = build_state(nz, cfg, dtype)
    step = jax.jit(partial(sl.step, g=g, u_geo=(0.0, 0.0), f_cor=0.0,
                           force=(0.0, 0.0), sfc_theta_flux=0.06),
                   static_argnames=("first",))
    dt_a = jnp.asarray(dt, dtype)

    # Compile + warm up (first=True primes the AB2/RK carry; block to force it).
    t_c0 = time.perf_counter()
    st, _ = step(st, dt=dt_a, first=True)
    jax.block_until_ready(st)
    compile_s = time.perf_counter() - t_c0
    for _ in range(3):  # a few untimed steps to settle the steady kernel
        st, _ = step(st, dt=dt_a, first=False)
    jax.block_until_ready(st)

    t0 = time.perf_counter()
    for _ in range(steps):
        st, _ = step(st, dt=dt_a, first=False)
    jax.block_until_ready(st)
    wall = time.perf_counter() - t0

    ncell = nx * ny * nz
    per_step = wall / steps
    steps_per_s = steps / wall
    # sim seconds advanced per wall second = steps/s * dt; the sim-h-per-wall-h
    # ratio is identical (the hour units cancel).
    sim_h_per_wall_h = steps_per_s * dt
    wall_h_per_sim_h = (3600.0 / dt) / steps_per_s / 3600.0
    finite = bool(np.isfinite(float(jnp.max(jnp.abs(st.w)))))
    return dict(nx=nx, ny=ny, nz=nz, ncell=ncell, steps=steps, dt=dt,
                compile_s=compile_s, per_step_ms=per_step * 1e3,
                steps_per_s=steps_per_s, ns_per_cell_step=per_step / ncell * 1e9,
                sim_h_per_wall_h=sim_h_per_wall_h, wall_h_per_sim_h=wall_h_per_sim_h,
                finite=finite)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nx", type=int, default=96)
    p.add_argument("--ny", type=int, default=96)
    p.add_argument("--nz", type=int, default=96)
    p.add_argument("--sizes", type=int, nargs="*", default=None,
                   help="cubic sizes to sweep (overrides --nx/ny/nz); e.g. 64 96 128")
    p.add_argument("--Lx", type=float, default=3200.0)
    p.add_argument("--Ly", type=float, default=3200.0)
    p.add_argument("--Lz", type=float, default=1600.0)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--f32", action="store_true")
    p.add_argument("--static-sgs", action="store_true",
                   help="static SGS instead of dynamic LASD (cheaper; not the prod path)")
    args = p.parse_args()
    dtype = jnp.float32 if args.f32 else jnp.float64

    dev = jax.devices()[0]
    print(f"[bench] device={dev.device_kind} ({dev.platform}) "
          f"dtype={'f32' if args.f32 else 'f64'} sgs={'LASD' if not args.static_sgs else 'static'} "
          f"dt={args.dt}s steps={args.steps}")
    sizes = [(n, n, n) for n in args.sizes] if args.sizes else [(args.nx, args.ny, args.nz)]
    hdr = (f"{'grid':>14} {'Mcell':>7} {'compile':>8} {'ms/step':>9} "
           f"{'steps/s':>8} {'ns/cell':>8} {'simh/wallh':>10} {'wallh/simh':>10}")
    print(hdr)
    for (nx, ny, nz) in sizes:
        r = bench_one(nx, ny, nz, Lx=args.Lx, Ly=args.Ly, Lz=args.Lz,
                      steps=args.steps, dt=args.dt, dtype=dtype,
                      sgs_dynamic=not args.static_sgs)
        warn = "" if r["finite"] else "  [NONFINITE]"
        print(f"{nx}x{ny}x{nz:>4} {r['ncell']/1e6:7.2f} {r['compile_s']:7.1f}s "
              f"{r['per_step_ms']:8.2f} {r['steps_per_s']:8.1f} {r['ns_per_cell_step']:8.2f} "
              f"{r['sim_h_per_wall_h']:10.2f} {r['wall_h_per_sim_h']:10.3f}{warn}")
    print("[bench] sim-h/wall-h = model hours simulated per wall-clock hour at the given dt.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
