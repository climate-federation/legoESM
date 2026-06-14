"""FV3_3D iter 595: hord parameter plumbing through _ppm_1d/_xppm/_yppm.

Adds ``hord`` kwarg (default 12 = preserves pre-iter-595 behavior)
to _ppm_1d, _xppm, _yppm.  Dispatches to the appropriate limiter
(iord=8/9/10/11/12 → iter 585/pert_ppm/iter 593/iter 592/_pert_ppm_iv0).

Tests
-----

1. ``test_hord_default_12_preserves_baseline`` — hord=12 = current.
2. ``test_hord_8_differs_from_12`` — switching to hord=8 changes flux.
3. ``test_hord_invalid_raises``.
4. ``test_xppm_yppm_accept_hord_kwarg``.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.fv_tp_2d import _ppm_1d, _xppm, _yppm


def _build_q(n=12, seed=595):
    """Build a 1D field with face shape (6, n+4, 1) suitable for _ppm_1d."""
    rng = np.random.default_rng(seed=seed)
    q = rng.uniform(0.0, 5.0, size=(6, n + 4, 1)).astype(np.float64)
    # Add a sharp transition for interesting limiter behavior
    q[:, n // 2:n // 2 + 2, :] += 2.0
    return jnp.asarray(q)


def test_hord_default_12_preserves_baseline():
    """hord=12 must be the default and match calling without kwarg."""
    n = 12
    q = _build_q(n=n)
    bl_d, br_d, qc_d = _ppm_1d(q, n)
    bl_e, br_e, qc_e = _ppm_1d(q, n, hord=12)
    assert float(jnp.abs(bl_d - bl_e).max()) < 1e-12
    assert float(jnp.abs(br_d - br_e).max()) < 1e-12


def test_hord_8_differs_from_12():
    """hord=8 should differ from hord=12 on a field with sharp transitions."""
    n = 12
    q = _build_q(n=n, seed=596)
    bl_12, br_12, _ = _ppm_1d(q, n, hord=12)
    bl_8, br_8, _ = _ppm_1d(q, n, hord=8)
    diff = float(jnp.abs(bl_12 - bl_8).max() + jnp.abs(br_12 - br_8).max())
    assert diff > 1e-4, (
        f"hord=8 vs hord=12 should differ on sharp field; got {diff:.3e}"
    )


def test_hord_invalid_raises():
    n = 8
    q = _build_q(n=n)
    with pytest.raises(ValueError, match="hord must be one of"):
        _ppm_1d(q, n, hord=7)


def test_xppm_yppm_accept_hord_kwarg():
    """_xppm and _yppm should accept hord and propagate it."""
    n = 8
    q = _build_q(n=n)
    crx = jnp.full((6, n + 1, 1), 0.3)  # uniform Courant
    fx_12 = _xppm(q, crx, n, hord=12)
    fx_8 = _xppm(q, crx, n, hord=8)
    assert jnp.all(jnp.isfinite(fx_12))
    assert jnp.all(jnp.isfinite(fx_8))
    # On the sharp field, these should differ
    diff = float(jnp.abs(fx_12 - fx_8).max())
    assert diff > 1e-4, f"_xppm hord variants should differ; got {diff:.3e}"

    cry = jnp.full((6, n, n + 1), 0.3)
    q_yppm = _build_q(n=n)
    # _yppm wants shape (6, M, n+4); reshape q accordingly
    fy_12 = _yppm(jnp.swapaxes(q_yppm, 1, 2), cry, n, hord=12)
    fy_8 = _yppm(jnp.swapaxes(q_yppm, 1, 2), cry, n, hord=8)
    assert jnp.all(jnp.isfinite(fy_12))
    assert jnp.all(jnp.isfinite(fy_8))
