"""Per-step throughput + compile-time benchmark for the plane NH dycore.

Measures:

1. JIT compile time (first ``step`` call, traced + lowered + compiled).
2. Per-step wall time over ``--n-warmup`` warmup + ``--n-bench``
   timed steps.
3. Steps/second + cells/second throughput.
4. State memory footprint (sum of leaf array bytes).

GPU optimization notes
----------------------
The plane dycore is pure JAX with no host callbacks, ``jax.lax.scan``
+ ``fori_loop`` for the acoustic substep loop, and ``jnp.roll`` /
``jnp.pad(mode='wrap')`` for periodic halos. XLA fuses the per-
direction stencil applications, the Smagorinsky strain magnitude,
and the variable-K diffusion flux into single kernels per RK stage.

The MPI 2D pencil halo (PR5,
:mod:`legoesm.parallel.plane_mpi`) overlays cleanly: each rank's
local domain is a smaller ``PlaneGrid`` and the per-rank wall time
should scale ~linearly with cell count up to the halo-communication
overhead floor.

Single-GPU performance levers (apply via CLI):

* Increase ``--n-bench`` to amortise the first-step compile cost.
* Set ``--dtype float32`` to halve memory bandwidth on GPUs; the
  plane dycore is x64-safe but x32 typically runs 2x faster.
* For multi-GPU runs use the MPI pencil layout (one rank per GPU)
  with ``mpirun -np N`` (lands with the OpenMPI-installed CI job).

Usage
-----
.. code-block:: bash

   JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python \\
       scripts/bench_plane_dycore.py \\
       --nx 32 --ny 32 --nlev 20 --n-warmup 5 --n-bench 50
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
    CompressibleEulerConfig,
)
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    PlaneCompressibleEulerModel,
    make_flat_plane_terrain_metric,
    make_rest_state,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate


def parse_args():
    p = argparse.ArgumentParser(
        description="Plane NH dycore throughput benchmark.",
    )
    p.add_argument("--nx", type=int, default=32)
    p.add_argument("--ny", type=int, default=32)
    p.add_argument("--nlev", type=int, default=20)
    p.add_argument("--dx", type=float, default=1_000.0)
    p.add_argument("--H", type=float, default=20_000.0)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--n-warmup", type=int, default=5)
    p.add_argument("--n-bench", type=int, default=50)
    p.add_argument("--dtype", choices=["float32", "float64"],
                   default="float64")
    p.add_argument("--smag-cs", type=float, default=0.2)
    p.add_argument("--hyperdiff", type=float, default=1.0e6)
    return p.parse_args()


def _measure_step_time(model, state, dt, n_steps):
    """Time ``n_steps`` ``model.step`` calls; block on the final
    state via ``.block_until_ready()`` so async dispatch doesn't
    skew the timing."""
    state = model.step(state, dt=dt)
    jax.tree_util.tree_leaves(state)[0].block_until_ready()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        state = model.step(state, dt=dt)
    jax.tree_util.tree_leaves(state)[0].block_until_ready()
    t1 = time.perf_counter()
    return state, (t1 - t0) / n_steps


def _state_footprint_bytes(state) -> int:
    """Sum of leaf array nbytes (does not include duplicate device buffers)."""
    return sum(
        leaf.nbytes
        for leaf in jax.tree_util.tree_leaves(state)
        if hasattr(leaf, "nbytes")
    )


def main():
    args = parse_args()
    if args.dtype == "float64":
        jax.config.update("jax_enable_x64", True)
        dtype = jnp.float64
    else:
        dtype = jnp.float32

    n_cells = args.nx * args.ny * args.nlev
    print(f"Plane NH dycore benchmark")
    print(f"  grid: nx={args.nx} ny={args.ny} nlev={args.nlev} "
          f"({n_cells:,} cells)")
    print(f"  dx={args.dx} m, Lz={args.H} m, dt={args.dt} s, "
          f"dtype={args.dtype}")
    print(f"  warmup={args.n_warmup} bench={args.n_bench}")

    grid = create_plane_grid(
        nx=args.nx, ny=args.ny, nlev=args.nlev,
        dx=args.dx, dy=args.dx, dtype=dtype,
    )
    hc = create_height_coordinate(args.nlev, H=args.H)
    tm = make_flat_plane_terrain_metric(grid, hc)
    cfg = CompressibleEulerConfig(
        sponge_coeff=0.05, sponge_width=2_000.0,
        hyperdiff_coeff=args.hyperdiff,
        hyperdiff_rho_coeff=args.hyperdiff,
        hyperdiff_w_coeff=args.hyperdiff,
        smagorinsky_cs=args.smag_cs, smagorinsky_prandtl=1.0,
        fix_mass=True, anchor_mass_to_initial=True,
        use_coriolis=False,
    )
    model = PlaneCompressibleEulerModel(grid, hc, tm, cfg)
    state = make_rest_state(grid, hc, dtype=dtype)

    footprint = _state_footprint_bytes(state)
    print(f"\nState footprint: {footprint / 1024:.1f} KiB "
          f"({footprint / (n_cells * 4):.1f} bytes/cell at fp32-equiv)")

    # First call → JIT trace + lower + compile.
    t_compile_0 = time.perf_counter()
    state = model.step(state, dt=args.dt)
    jax.tree_util.tree_leaves(state)[0].block_until_ready()
    t_compile_1 = time.perf_counter()
    print(f"JIT compile time (first step): "
          f"{(t_compile_1 - t_compile_0):.2f} s")

    # Warmup (compiled cache primed).
    for _ in range(args.n_warmup):
        state = model.step(state, dt=args.dt)
    jax.tree_util.tree_leaves(state)[0].block_until_ready()

    # Timed benchmark.
    state, t_per_step = _measure_step_time(
        model, state, args.dt, args.n_bench,
    )
    steps_per_sec = 1.0 / t_per_step
    cells_per_sec = steps_per_sec * n_cells
    print(f"\nThroughput:")
    print(f"  {t_per_step * 1000:8.2f} ms/step")
    print(f"  {steps_per_sec:8.2f} steps/s")
    print(f"  {cells_per_sec / 1e6:8.2f} M cells/s")

    # Simulated-time-per-wall-time ratio.
    sypd = (args.dt * 86_400) / (t_per_step * 365.25 * 86_400)
    print(f"  {sypd:8.2e} SYPD (simulated years per wall day)")


if __name__ == "__main__":
    main()
