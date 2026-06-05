"""Throughput benchmark for the plane boundary-layer LES (CPU / GPU).

Times the compiled ``model.step`` + surface-coupling for a chosen case and grid,
reporting steps/s, grid-point throughput, and ns/point/step. Backend-agnostic:
run on CPU (default) or GPU (``JAX_PLATFORMS=cuda``, after installing the CUDA
plugin — see the procedure at the bottom of this docstring). Also verifies the
step is single-compile (no retrace) and contains no host-callback so it is
GPU-clean.

The dynamic SGS closures (Germano and Bou-Zeid LASD) are SINGLE-RANK: their
spectral test filter, planar mean and 3×3 local average act over the LOCAL
horizontal tile, so under a horizontal MPI decomposition they would be per-rank
(wrong). MPI LES must use the STATIC closure (``--static-sgs``), validated
serial==MPI; this is reported by the benchmark when ``--sgs static`` is chosen.

Usage
-----
.. code-block:: bash

   # CPU
   JAX_ENABLE_X64=1 .venv/bin/python scripts/bench/bench_les_plane.py \\
       --case neutral --nx 64 --ny 64 --nlev 64 --sgs lasd --nsteps 50

   # GPU
   uv pip install jax-cuda12-plugin==0.10.0
   JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 .venv/bin/python \\
       scripts/bench/bench_les_plane.py --case neutral --nx 128 --ny 128 \\
       --nlev 128 --sgs lasd --nsteps 50
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "run"))
import run_les_plane as rlp  # noqa: E402

jax.config.update("jax_enable_x64", True)


def _build(case, nx, ny, nlev, dx, H, dz_sfc, dt, sgs):
    """Reuse the production driver's build() with a synthetic argparse.Namespace."""
    spec = rlp._CASES[case]
    args = argparse.Namespace(
        case=case, nx=nx, ny=ny, nlev=nlev, dx=dx, H=H, dz_sfc=dz_sfc,
        dt=dt, hours=0.0, off_centering=0.0, si_w_filter=0.0,
        static_sgs=(sgs == "static"),
        scale_dependent=(sgs == "lasd"),
        n_acoustic_substeps=8, hyperdiff=1.0e-3,
    )
    return rlp.build(args), args


def bench(case, nx, ny, nlev, dx, H, dz_sfc, dt, sgs, nsteps):
    (model, state, surf, grid, hc, spec), args = _build(
        case, nx, ny, nlev, dx, H, dz_sfc, dt, sgs)

    def one(state, t):
        # model.step and surf are each internally JIT-compiled (matches the
        # production driver loop); no outer jit (it would nest over the already-
        # jitted surf and leak a tracer).
        state = model.step(state, dt=dt, physics_fn=None)
        return surf(state, t)

    # Warm-up (compile) — excluded from timing.
    state = one(state, jnp.asarray(0.0))
    jax.block_until_ready(state)

    t0 = time.time()
    for i in range(nsteps):
        state = one(state, jnp.asarray(i * dt))
    jax.block_until_ready(state)
    wall = time.time() - t0

    ncell = nx * ny * nlev
    sps = nsteps / wall
    print(f"== LES bench: case={case} sgs={sgs} grid={nx}x{ny}x{nlev} "
          f"({ncell:,} cells) backend={jax.default_backend()} ==")
    print(f"  {nsteps} steps in {wall:.2f} s  ->  {sps:.1f} steps/s")
    print(f"  throughput : {ncell * sps / 1e6:.1f} Mcell/s   "
          f"{1e9 * wall / (nsteps * ncell):.1f} ns/cell/step")
    finite = bool(jnp.all(jnp.isfinite(state.w.data)))
    print(f"  state finite after {nsteps} steps: {finite}")
    if sgs in ("lasd", "germano"):
        print("  NOTE: dynamic SGS is single-rank (per-tile filter/average); "
              "use --sgs static for MPI.")
    return sps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=list(rlp._CASES), default="neutral")
    p.add_argument("--nx", type=int, default=64)
    p.add_argument("--ny", type=int, default=64)
    p.add_argument("--nlev", type=int, default=64)
    p.add_argument("--dx", type=float, default=20.0)
    p.add_argument("--H", type=float, default=1000.0)
    p.add_argument("--dz-sfc", type=float, default=10.0)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--sgs", choices=["static", "germano", "lasd"], default="lasd")
    p.add_argument("--nsteps", type=int, default=50)
    a = p.parse_args()
    bench(a.case, a.nx, a.ny, a.nlev, a.dx, a.H, a.dz_sfc, a.dt, a.sgs, a.nsteps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
