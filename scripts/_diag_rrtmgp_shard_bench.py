#!/usr/bin/env python
"""RRTMGP column-sharding benchmark + correctness check on the MPAS path.

Builds the MPAS rrtmgp radiation physics_fn twice — single-device and
column-mesh-sharded across all local devices — calls both on an identical
MPAS column state, and reports (1) heating-rate AGREEMENT (sharded must equal
single-device to round-off) and (2) the SPEEDUP.  This is the systematic
bottleneck test for the rrtmgp-on-MPAS perf fix (issue #273 column_mesh wired
into ``_run_mpas``).  Run on a multi-GPU node.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from legoesm.core.field import Field  # noqa: E402
from legoesm.core.state import MPASHydrostaticState  # noqa: E402
from legoesm.grids.voronoi import create_voronoi_mesh  # noqa: E402
from legoesm.grids.vertical import create_sigma_coordinate  # noqa: E402
from legoesm.atmosphere.physics.radiation.config import (  # noqa: E402
    RadiationConfig, RRTMGPConfig,
)
from legoesm.atmosphere.physics.radiation.integration import (  # noqa: E402
    make_radiation_physics,
)
from legoesm.parallel.column_shard import create_column_mesh  # noqa: E402

LEVEL = 4
NLEV = 30


def make_state(mesh, nlev):
    nCells = mesh.nCells
    # A non-isothermal column (decreasing T with height) so radiation has a
    # non-trivial profile; p_s uniform 1e5.
    sigma_f = jnp.linspace(0.02, 0.99, nlev)
    T = jnp.broadcast_to(300.0 - 50.0 * (1.0 - sigma_f)[None, :], (nCells, nlev))
    return MPASHydrostaticState(
        u=Field(data=jnp.zeros((mesh.nEdges, nlev), dtype=jnp.float64),
                name="u", dims=("nEdges", "nlev"), units="m/s"),
        T=Field(data=jnp.asarray(T, dtype=jnp.float64), name="T",
                dims=("nCells", "nlev"), units="K"),
        p_s=Field(data=jnp.full((nCells,), 1.0e5, dtype=jnp.float64),
                  name="p_s", dims=("nCells",), units="Pa"),
        phis=Field(data=jnp.zeros((nCells,), dtype=jnp.float64),
                   name="phis", dims=("nCells",), units="m2/s2"),
    )


def time_fn(fn, state, mesh, sigma, label, n=5):
    # Close over mesh+sigma (compile-time constants — mirrors the real MPAS
    # step where they are static attributes of ``self``); jit over the state
    # only.  jit'ing over sigma would make ``sigma.n_levels`` a tracer and
    # break the reshape.
    # The radiation physics_fn returns a HydrostaticTendencies (a NamedTuple,
    # which IS a tuple) — detect the (tend, phys_state) 2-tuple form by the
    # absence of a .dT_dt attribute rather than isinstance(tuple).
    def _tend(r):
        return r if hasattr(r, "dT_dt") else r[0]
    jfn = jax.jit(lambda s: fn(s, mesh, sigma))
    heat = _tend(jfn(state)).dT_dt.data
    heat.block_until_ready()
    t0 = time.time()
    for _ in range(n):
        _tend(jfn(state)).dT_dt.data.block_until_ready()
    dt = (time.time() - t0) / n
    print(f"[bench] {label:18s} {dt*1000:8.1f} ms/call  heat[min,max]="
          f"[{float(jnp.min(heat))*86400:.2f},{float(jnp.max(heat))*86400:.2f}] K/day",
          flush=True)
    return heat, dt


def main():
    devs = jax.devices()
    n_dev = len(devs)
    print(f"[bench] devices={n_dev} {devs}", flush=True)
    mesh = create_voronoi_mesh(LEVEL, lloyd_iterations=5)
    sigma = create_sigma_coordinate(NLEV)
    state = make_state(mesh, NLEV)
    print(f"[bench] L{LEVEL} nCells={mesh.nCells} nlev={NLEV}", flush=True)

    rad_cfg = RadiationConfig(scheme="rrtmgp", rrtmgp=RRTMGPConfig())

    fn1 = make_radiation_physics(rad_cfg, "mpas", column_mesh=None)
    heat1, t1 = time_fn(fn1, state, mesh, sigma, "single-device")

    # compute_fp32 cast (the production mechanism: fp32 optics+solve under x64).
    fn_fp32 = make_radiation_physics(
        RadiationConfig(scheme="rrtmgp", rrtmgp=RRTMGPConfig(compute_fp32=True)),
        "mpas", column_mesh=None)
    heat_fp32, t_fp32 = time_fn(fn_fp32, state, mesh, sigma, "fp32-cast")
    _rel = float(jnp.max(jnp.abs(heat1 - heat_fp32))) / (
        float(jnp.max(jnp.abs(heat1))) + 1e-30)
    print(f"[bench] fp32-CAST vs fp64: {t1 / t_fp32:.2f}x faster, "
          f"heating rel-diff={_rel:.2e} -> "
          f"{'OK' if _rel < 1e-2 else 'CHECK'}", flush=True)

    if n_dev > 1 and mesh.nCells % n_dev == 0:
        cmesh = create_column_mesh(n_dev)
        fn2 = make_radiation_physics(rad_cfg, "mpas", column_mesh=cmesh)
        heat2, t2 = time_fn(fn2, state, mesh, sigma, f"sharded x{n_dev}")
        max_abs = float(jnp.max(jnp.abs(heat1 - heat2)))
        rel = max_abs / (float(jnp.max(jnp.abs(heat1))) + 1e-30)
        print(f"[bench] CORRECTNESS max|Δheat|={max_abs*86400:.3e} K/day "
              f"(rel={rel:.2e}) -> {'MATCH' if rel < 1e-6 else 'MISMATCH'}",
              flush=True)
        print(f"[bench] SPEEDUP {t1/t2:.2f}x ({t1*1000:.0f} -> {t2*1000:.0f} ms)",
              flush=True)
    else:
        print(f"[bench] no shard test: n_dev={n_dev} nCells%n_dev="
              f"{mesh.nCells % n_dev}", flush=True)
    print("[bench] done", flush=True)


if __name__ == "__main__":
    main()
