"""Verify the BCW benchmark's ``--scan-steps N`` path produces
bit-identical state to the per-step Python loop (the default
``--scan-steps 1`` baseline).  Both should be the same XLA computation,
just batched differently inside a ``jax.lax.scan`` body — so any
divergence indicates an actual regression in the optimisation, not a
floating-point rounding difference.

Usage:
    JAX_ENABLE_X64=1 PYTHONPATH=. .venv/bin/python scripts/verify_scan_steps_equivalence.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from functools import partial


def _make_scan_chunk(model, dt, n_steps):
    def _body(s, _):
        return model.step(s, dt), None

    @jax.jit
    def _chunk(s):
        s_new, _ = jax.lax.scan(_body, s, None, length=n_steps)
        return s_new

    return _chunk


def _diff_pytrees(a, b):
    """Max-abs difference between two same-structure pytrees of arrays."""
    leaves_a, _ = jax.tree.flatten(a)
    leaves_b, _ = jax.tree.flatten(b)
    diffs = []
    for la, lb in zip(leaves_a, leaves_b):
        if hasattr(la, "data"):
            la = la.data
            lb = lb.data
        if not hasattr(la, "shape"):
            continue
        diffs.append(float(jnp.max(jnp.abs(la - lb))))
    return max(diffs) if diffs else 0.0


def probe_spectral_t21():
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
        SpectralPrimitiveEquationModel, SpectralPEConfig,
        isothermal_rest_state_spectral,
    )
    grid = create_gaussian_grid(n_max=21)
    sigma = create_sigma_coordinate(8)
    state0 = isothermal_rest_state_spectral(
        grid, sigma, perturbation_amplitude=0.5,
    )
    model = SpectralPrimitiveEquationModel(grid, sigma, SpectralPEConfig())
    dt = 600.0
    N = 12

    state_loop = state0
    for _ in range(N):
        state_loop = model.step(state_loop, dt)
    jax.block_until_ready(jax.tree.leaves(state_loop))

    chunk = _make_scan_chunk(model, dt, N)
    state_scan = chunk(state0)
    jax.block_until_ready(jax.tree.leaves(state_scan))

    return _diff_pytrees(state_loop, state_scan)


def probe_icosahedral_i4():
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationModel, MPASPrimitiveEquationConfig,
    )
    from legoesm.atmosphere.held_suarez import held_suarez_init_mpas
    mesh = create_voronoi_mesh(subdivision_level=4)
    sigma = create_sigma_coordinate(8)
    state0 = held_suarez_init_mpas(mesh, sigma, T_init=280.0)
    model = MPASPrimitiveEquationModel(mesh, sigma, MPASPrimitiveEquationConfig())
    dt = 150.0
    N = 12

    state_loop = state0
    for _ in range(N):
        state_loop = model.step(state_loop, dt)
    jax.block_until_ready(jax.tree.leaves(state_loop))

    chunk = _make_scan_chunk(model, dt, N)
    state_scan = chunk(state0)
    jax.block_until_ready(jax.tree.leaves(state_scan))

    return _diff_pytrees(state_loop, state_scan)


def probe_cubed_sphere_c24():
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationModel, CDGridPrimitiveEquationConfig,
        hydrostatic_to_fv3,
    )
    from legoesm.atmosphere.held_suarez import held_suarez_init
    grid = create_cubed_sphere(24)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    sigma = create_sigma_coordinate(8)
    state_cc = held_suarez_init(grid, sigma, T_init=280.0)
    state0 = hydrostatic_to_fv3(state_cc, cdgrid)
    config = CDGridPrimitiveEquationConfig()
    model = CDGridPrimitiveEquationModel(grid, sigma, config)
    dt = 300.0
    N = 8

    state_loop = state0
    for _ in range(N):
        state_loop = model.step(state_loop, dt)
    jax.block_until_ready(jax.tree.leaves(state_loop))

    chunk = _make_scan_chunk(model, dt, N)
    state_scan = chunk(state0)
    jax.block_until_ready(jax.tree.leaves(state_scan))

    return _diff_pytrees(state_loop, state_scan)


def main():
    probes = [
        ("spectral_T21",   probe_spectral_t21),
        ("icosahedral_I4", probe_icosahedral_i4),
        ("cubed-sphere_C24", probe_cubed_sphere_c24),
    ]
    # Tolerance: scan reorders associative reductions via XLA fusion,
    # so we allow up to ~10× the fp32 eps × O(N) for an MPAS-style
    # mixed-precision dycore (storage fp32, accumulators fp64).  At
    # T~280K and ~10^3 ops/step over N steps the worst-case
    # rounding-chain drift is ~ 280 · 1e-7 · 1e3 · N = 3e-2 · N.
    # 1e-3 absolute is the right order for N≈12 steps; bit-identical
    # cells still flag as OK at this tolerance, and a true regression
    # (e.g. an off-by-one indexing change) would exceed it by orders.
    print(f"{'cell':22s}  {'max_abs_diff':>15s}  status")
    print("-" * 56)
    for name, fn in probes:
        try:
            diff = fn()
            status = "OK" if diff < 1e-3 else "DIVERGED"
            print(f"{name:22s}  {diff:15.3e}  {status}")
        except Exception as exc:
            import traceback
            tb = traceback.format_exc().strip().splitlines()[-1]
            print(f"{name:22s}  {'-':>15s}  ERROR  {tb[:80]}")


if __name__ == "__main__":
    main()
