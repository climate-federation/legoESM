"""Direct contract tests for shared NEMO source-statement rounding."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.source_rounding import nemo_source_round


@pytest.fixture(autouse=True)
def _enable_x64():
    """Run these binary64 bit-preservation checks in float64 (restored after).

    Without this, the jnp.float64 inputs below are silently truncated to
    float32 before nemo_source_round ever sees them, so the bit-preservation
    assertions would pass vacuously on already-rounded-off float32 bits.
    """
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _bits(values: jax.Array) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).view(np.uint64)


def test_nemo_source_round_preserves_binary64_bits_and_jit_parity() -> None:
    subnormal = np.nextafter(np.float64(0.0), np.float64(1.0))
    values_np = np.array(
        [-np.inf, -2.0, -subnormal, -0.0, subnormal, 3.0, np.inf],
        dtype=np.float64,
    )
    values = jnp.asarray(values_np, dtype=jnp.float64)

    eager = nemo_source_round(values)
    compiled = jax.jit(nemo_source_round)(values)

    # Ground truth computed purely in NumPy (never touches JAX/x64), so a
    # float64->float32->float64 round-trip on `values` itself cannot also
    # corrupt this comparison target. The subnormal (~4.94e-324) underflows
    # to exactly 0.0 under float32, so a truncation bug flips its bit
    # pattern away from expected_bits and this assertion catches it.
    expected_bits = values_np.view(np.uint64)
    np.testing.assert_array_equal(_bits(eager), expected_bits)
    np.testing.assert_array_equal(_bits(compiled), expected_bits)
    np.testing.assert_array_equal(_bits(compiled), _bits(eager))


def test_nemo_source_round_has_identity_gradient_for_finite_inputs() -> None:
    subnormal = np.nextafter(np.float64(0.0), np.float64(1.0))
    values = jnp.asarray([-2.0, -subnormal, -0.0, subnormal, 3.0], dtype=jnp.float64)

    gradient = jax.grad(lambda x: jnp.sum(nemo_source_round(x)))(values)

    np.testing.assert_array_equal(np.asarray(gradient), np.ones(5, dtype=np.float64))


def _microbench():
    """The round-351 store microbench: its density and its validated copy counter."""
    import importlib.util
    from pathlib import Path

    script = (Path(__file__).resolve().parents[2]
              / "scripts/bench/orca2_round351_eos_store_microbench.py")
    spec = importlib.util.spec_from_file_location("orca2_round351_eos_store_microbench", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nemo_source_store_keeps_bits_stores_one_copy_and_differentiates() -> None:
    from legoesm.core.source_rounding import nemo_source_store

    mb = _microbench()
    t, s, depth, tmask, _ = mb.make_inputs((12, 10, 6))

    def reads(rho):  # the slope's reads: i+1 and j+1 stencils, vertical difference
        return (jnp.roll(rho, -1, axis=1) - rho, jnp.roll(rho, -1, axis=0) - rho,
                rho[..., 1:] - rho[..., :-1])

    def plain(*a):
        return reads(mb.density(*a))

    def stored(*a):
        return reads(nemo_source_store(mb.density, *a))

    def rho_copies(fn):
        text = jax.jit(fn).lower(t, s, depth, tmask).compile().as_text()
        return mb.copy_summary([text], t.size)["rho"]

    # The known case duplicates without the store (else the next assertion is vacuous).
    assert rho_copies(plain)["copies"] >= 2
    got = rho_copies(stored)
    assert (got["copies"], got["partial_kernels"], got["multiplies"]) == (1, 0, mb.POLY["rho"][1])
    assert got["dead_branch_copies"] == 1  # the identical other branch, never executed

    reference = jax.jit(plain)(t, s, depth, tmask)
    for out in (jax.jit(stored)(t, s, depth, tmask), stored(t, s, depth, tmask)):
        for a, b in zip(out, reference, strict=True):
            np.testing.assert_array_equal(_bits(a), _bits(b))

    def loss(fn):
        return lambda t_, s_: sum(jnp.sum(r * r) for r in fn(t_, s_, depth, tmask))

    g_stored = jax.jit(jax.grad(loss(stored), argnums=(0, 1)))(t, s)
    g_plain = jax.jit(jax.grad(loss(plain), argnums=(0, 1)))(t, s)
    for a, b in zip(g_stored, g_plain, strict=True):
        assert np.all(np.isfinite(np.asarray(a))) and float(np.abs(np.asarray(a)).max()) > 0.0
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12 * float(np.abs(b).max()))


def test_pressure_density_of_every_rk3_stage_is_stored_once() -> None:
    # compute_frozen_geom_density is what the ORCA2 step calls for each RK3 stage's density.
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import compute_frozen_geom_density
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.vertical import create_ocean_z_star

    mb = _microbench()
    grid, z_coord = create_latlon_grid(12, 24), create_ocean_z_star(n_levels=6, H_max=4000.0)
    cfg = LatLonCGridOceanConfig.from_flat(eos="nemo_eos80", eos_depth="geometric")
    state = rest_state_latlon_cgrid_ocean(grid, z_coord)
    text = jax.jit(lambda st: compute_frozen_geom_density(st, grid, z_coord, cfg)[2:]).lower(
        state).compile().as_text()
    got = mb.copy_summary([text], state.T.data.size)["rho"]
    # One executed copy, and its twin in the store's never-taken branch.
    assert (got["copies"], got["partial_kernels"], got["dead_branch_copies"]) == (1, 0, 1)
