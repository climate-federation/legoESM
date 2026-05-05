"""Smoke test for `clamp_and_redistribute` (mass-conserving eta floor).

Tests issue #176: barotropic eta floor that redistributes mass injected
by the floor clamp over cells with headroom, so total mass is preserved
(serial path; the MPI reduction is exercised in distributed tests).

Run with:

    JAX_ENABLE_X64=1 .venv/bin/python3.14 -m pytest \
        tests/ocean/unit/test_eta_floor.py -v
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.ocean.dynamics.eta_floor import clamp_and_redistribute


@pytest.fixture(autouse=True)
def _x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


def _setup(n=8):
    """Build a small test field with one cell that violates the floor."""
    eta = jnp.zeros((n, n))
    # One cell well below floor; rest are above zero.
    eta = eta.at[0, 0].set(-2.0)
    eta = eta.at[2:, :].set(1.0)
    floor = jnp.full((n, n), -0.5)
    mask = jnp.ones((n, n))
    area = jnp.full((n, n), 100.0)  # uniform cell area
    return eta, floor, mask, area


def test_returns_correct_shape():
    eta, floor, mask, area = _setup()
    out = clamp_and_redistribute(eta, floor, mask, area, n_iter=3)
    assert out.shape == eta.shape
    assert out.dtype == eta.dtype
    assert jnp.all(jnp.isfinite(out))


def test_floor_is_respected():
    eta, floor, mask, area = _setup()
    out = clamp_and_redistribute(eta, floor, mask, area, n_iter=3)
    # Final eta must be at or above floor everywhere active.
    assert jnp.all(out >= floor - 1e-12)


def test_mass_conservation_serial():
    """Serial path: the floor clamp should redistribute its mass injection.

    Total mass change = sum((eta_new - eta_old) * area * mask). After
    redistribution this should be much smaller than the raw clamp's
    injected mass.
    """
    eta, floor, mask, area = _setup()
    raw_clamp = jnp.maximum(eta, floor) * mask
    raw_injected = float(jnp.sum((raw_clamp - eta) * area * mask))
    assert raw_injected > 0.0  # there is injection to redistribute

    out = clamp_and_redistribute(eta, floor, mask, area, n_iter=5)
    final_drift = float(jnp.sum((out - eta) * area * mask))
    # After redistribution drift should be a small fraction of the raw
    # injection — exact zero is not guaranteed because cells already at the
    # floor block further reduction.
    assert abs(final_drift) < 0.1 * raw_injected + 1e-9


def test_mask_zero_cells_stay_zero():
    eta, floor, mask, area = _setup()
    # Mask out a strip; values stay multiplied by mask = 0.
    mask = mask.at[0, :].set(0.0)
    out = clamp_and_redistribute(eta, floor, mask, area, n_iter=3)
    assert jnp.allclose(out[0, :], 0.0)


def test_jit_compiles():
    eta, floor, mask, area = _setup()
    fn = jax.jit(lambda e, f, m, a: clamp_and_redistribute(e, f, m, a, n_iter=2))
    out = fn(eta, floor, mask, area)
    assert out.shape == eta.shape
    assert jnp.all(jnp.isfinite(out))
