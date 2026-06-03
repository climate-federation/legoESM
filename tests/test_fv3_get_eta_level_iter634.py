"""FV3_3D iter 634: get_eta_level port.

Faithful JAX port of FV3 ``get_eta_level`` (tools/fv_eta.F90:
1859-1890).  Hybrid (ak, bk) coord → (pf, ph) with FV3 log-mean
full-level pressure formula.

Tests
-----

1. ``test_get_eta_level_shapes``.
2. ``test_get_eta_level_ph_formula``.
3. ``test_get_eta_level_pf_log_mean``.
4. ``test_get_eta_level_top_kappa_branch``.
5. ``test_get_eta_level_pscale``.
6. ``test_get_eta_level_batch_p_s``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.vertical import get_eta_level


def _make_ak_bk(npz, ak0_positive=True):
    # Simple monotonic hybrid coord
    sigma_half = jnp.linspace(0.0, 1.0, npz + 1)
    p_top = 100.0 if ak0_positive else 0.0
    p_ref = 1.0e5
    bk = sigma_half ** 2  # increases from 0 to 1 (sigma-like)
    ak = p_top * (1.0 - sigma_half) + (1.0 - bk) * 0.0  # ak[0] = p_top
    return ak, bk


def test_get_eta_level_shapes():
    """Shapes (..., npz) for pf, (..., npz+1) for ph."""
    npz = 20
    ak, bk = _make_ak_bk(npz)
    p_s = jnp.asarray(1.0e5)
    pf, ph = get_eta_level(ak, bk, p_s)
    assert pf.shape == (npz,)
    assert ph.shape == (npz + 1,)


def test_get_eta_level_ph_formula():
    """ph[k] = ak[k] + bk[k]·p_s for k>=1; ph[0] = ak[0]."""
    npz = 10
    ak, bk = _make_ak_bk(npz)
    p_s = jnp.asarray(8.0e4)
    _, ph = get_eta_level(ak, bk, p_s)
    # ph[0]
    assert abs(float(ph[0]) - float(ak[0])) < 1e-10
    # ph[k] = ak[k] + bk[k] * p_s for k >= 1
    expected = ak[1:] + bk[1:] * p_s
    assert jnp.allclose(ph[1:], expected, atol=1e-10)


def test_get_eta_level_pf_log_mean():
    """pf[k] = (ph[k+1] - ph[k]) / log(ph[k+1]/ph[k]) for k >= 1."""
    npz = 10
    ak, bk = _make_ak_bk(npz)
    p_s = jnp.asarray(1.0e5)
    pf, ph = get_eta_level(ak, bk, p_s)
    for k in range(1, npz):
        expected = float((ph[k + 1] - ph[k]) / jnp.log(ph[k + 1] / ph[k]))
        assert abs(float(pf[k]) - expected) / max(abs(expected), 1.0) < 1e-10, (
            f"k={k}: pf={float(pf[k])}, expected={expected}"
        )


def test_get_eta_level_top_kappa_branch():
    """When ak[0] = 0, pf[0] = (ph[1] - ph[0]) · kappa/(kappa+1).

    FV3 lines 1882-1884: avoids log(0) at the model top.
    """
    npz = 8
    sigma_half = jnp.linspace(0.0, 1.0, npz + 1)
    bk = sigma_half
    ak = jnp.zeros_like(sigma_half)  # top has ak = 0 → kappa branch
    p_s = jnp.asarray(1.0e5)
    pf, ph = get_eta_level(ak, bk, p_s)
    kappa = constants.kappa
    expected = float((ph[1] - ph[0]) * kappa / (kappa + 1.0))
    assert abs(float(pf[0]) - expected) / max(abs(expected), 1.0) < 1e-10


def test_get_eta_level_pscale():
    """pscale multiplies ph (and hence pf scales linearly)."""
    npz = 6
    ak, bk = _make_ak_bk(npz)
    p_s = jnp.asarray(1.0e5)
    pf1, ph1 = get_eta_level(ak, bk, p_s, pscale=None)
    pf2, ph2 = get_eta_level(ak, bk, p_s, pscale=2.0)
    assert jnp.allclose(ph2, 2.0 * ph1, atol=1e-6)
    # pf is also doubled (log-mean is homogeneous of degree 1)
    assert jnp.allclose(pf2, 2.0 * pf1, atol=1e-6)


def test_get_eta_level_batch_p_s():
    """Batched p_s input → batched (pf, ph) output."""
    npz = 8
    ak, bk = _make_ak_bk(npz)
    p_s = jnp.asarray([[1.0e5, 8.0e4], [9.0e4, 1.1e5]])
    pf, ph = get_eta_level(ak, bk, p_s)
    assert pf.shape == (2, 2, npz)
    assert ph.shape == (2, 2, npz + 1)
    assert jnp.all(jnp.isfinite(pf))
    assert jnp.all(jnp.isfinite(ph))
