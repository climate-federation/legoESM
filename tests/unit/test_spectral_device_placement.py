"""Unit tests for ``legoesm.parallel.metal.place_spectral_grid``.

Single home for the Metal spectral-on-CPU routing block previously
copy-pasted across the four spectral dycores (atmosphere spectral_sw/pe/nh,
ocean spectral_ocean_pe). Behavior contract:

- non-Metal backend: grid untouched, no routing flags, fp64 support checked
  via ``check_spectral_backend`` (honoring ``allow_unsupported``);
- Metal backend: grid transferred to CPU, routing flags set, NO
  ``check_spectral_backend`` call (the routing IS the mitigation).

The Metal path is exercised by monkeypatching the ``get_backend`` symbol on
the metal module (it is imported by name), since CI has no Metal device.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.parallel import metal
from legoesm.parallel.metal import SpectralDevicePlacement, place_spectral_grid


def test_non_metal_grid_untouched_and_flags_off(monkeypatch):
    # Pin the backend so the test is correct even on an actual Metal host
    # (codex review LOW: without this, a Metal machine takes the other branch).
    monkeypatch.setattr(metal, "get_backend", lambda: "cpu")
    grid = jnp.arange(4.0)
    placement = place_spectral_grid(grid, allow_unsupported=True)
    assert isinstance(placement, SpectralDevicePlacement)
    assert placement.grid is grid  # identity: no copy/transfer off Metal
    assert placement.use_cpu_for_spectral is False
    assert placement.cpu_device is None
    assert placement.default_device is None


def test_non_metal_calls_spectral_backend_check(monkeypatch):
    monkeypatch.setattr(metal, "get_backend", lambda: "cpu")
    calls: list[bool] = []
    monkeypatch.setattr(
        metal,
        "check_spectral_backend",
        lambda *, allow_unsupported=False: calls.append(allow_unsupported),
    )
    place_spectral_grid(jnp.zeros(2), allow_unsupported=True)
    place_spectral_grid(jnp.zeros(2))
    assert calls == [True, False]  # threaded through, defaulting False


def test_non_metal_propagates_backend_check_failure(monkeypatch):
    monkeypatch.setattr(metal, "get_backend", lambda: "gpu")

    def _boom(*, allow_unsupported=False):
        raise RuntimeError("no fp64 on this backend")

    monkeypatch.setattr(metal, "check_spectral_backend", _boom)
    with pytest.raises(RuntimeError, match="no fp64"):
        place_spectral_grid(jnp.zeros(2))


def test_metal_routes_grid_to_cpu_and_skips_backend_check(monkeypatch):
    # CI has no Metal device, so jax.devices()[0] is CPU here: this verifies
    # branch selection, flag plumbing, and the check_spectral_backend skip —
    # NOT a real Metal->CPU transfer (default_device == cpu_device on CPU CI).
    # The true-Metal invariant is only checkable on Apple hardware (codex
    # review LOW: acknowledged coverage gap).
    monkeypatch.setattr(metal, "get_backend", lambda: "metal")

    def _must_not_run(*, allow_unsupported=False):
        raise AssertionError("check_spectral_backend must not run on Metal")

    monkeypatch.setattr(metal, "check_spectral_backend", _must_not_run)
    grid = jnp.arange(3.0)
    placement = place_spectral_grid(grid)
    assert placement.use_cpu_for_spectral is True
    cpu = jax.devices("cpu")[0]
    assert placement.cpu_device == cpu
    assert placement.default_device == jax.devices()[0]
    assert cpu in placement.grid.devices()
    # Values survive the transfer.
    assert jnp.array_equal(placement.grid, grid)
