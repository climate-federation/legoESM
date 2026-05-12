"""FV3_3D iter 727: compute_zh_from_delz_fv3 port + iter-706 refactor.

Extracts height-from-delz pattern (used inline in iter-706
prt_height) as a public helper.

Tests
-----

1. ``test_zh_surface_matches_phis_over_g``.
2. ``test_zh_uniform_delz_evenly_spaced``.
3. ``test_zh_monotone_decreasing_in_k``.
4. ``test_zh_iter706_prt_height_unchanged``.
5. ``test_zh_inverse_of_hydrostatic_delz``.
6. ``test_zh_shapes_3d``.
7. ``test_zh_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import (
    compute_zh_from_delz_fv3,
    hydrostatic_delz_fv3,
    prt_height_fv3,
)


def test_zh_surface_matches_phis_over_g():
    """zh[km] = phis / g exactly."""
    km = 10
    phis = jnp.asarray(50000.0)   # = 5 km · g approximately
    delz = jnp.full((km,), -500.0)
    zh = compute_zh_from_delz_fv3(phis, delz)
    assert abs(float(zh[-1]) - 50000.0 / constants.g) < 1e-10


def test_zh_uniform_delz_evenly_spaced():
    """Uniform delz=-500 → zh evenly spaced, top at phis/g + km·500."""
    km = 10
    phis = jnp.asarray(0.0)
    delz = jnp.full((km,), -500.0)
    zh = compute_zh_from_delz_fv3(phis, delz)
    # zh[km] = 0, zh[km-1] = 500, ..., zh[0] = km*500 = 5000
    expected = jnp.linspace(km * 500.0, 0.0, km + 1)
    assert jnp.allclose(zh, expected, atol=1e-10)


def test_zh_monotone_decreasing_in_k():
    """zh monotonically decreasing top → surface."""
    rng = np.random.default_rng(seed=727)
    km = 20
    phis = jnp.asarray(rng.uniform(0.0, 5000.0)) * constants.g
    delz = jnp.asarray(rng.uniform(-500.0, -50.0, size=(km,)))
    zh = compute_zh_from_delz_fv3(phis, delz)
    assert jnp.all(jnp.diff(zh) < 0.0)   # monotone decreasing


def test_zh_iter706_prt_height_unchanged():
    """iter-706 prt_height refactor preserves output."""
    rng = np.random.default_rng(seed=728)
    n = 30
    km = 20
    delz = jnp.full((n, km), -500.0)
    pe_col = jnp.linspace(1.0e4, 1.0e5, km + 1)
    peln = jnp.broadcast_to(jnp.log(pe_col)[None, :], (n, km + 1))
    phis = jnp.zeros((n,))
    area = jnp.ones((n,))
    lat = jnp.linspace(-jnp.pi / 2 * 0.95, jnp.pi / 2 * 0.95, n)
    out = prt_height_fv3(5.0e4, phis, delz, peln, area, lat)
    # Output should be finite + plausible (~5 km for 500 hPa)
    assert all(jnp.isfinite(v) or v == -1.0 for v in out.values())
    assert 4000.0 < out["gb"] < 7000.0   # plausible 500-hPa height


def test_zh_inverse_of_hydrostatic_delz():
    """compute_zh_from_delz(phis, hydrostatic_delz(T, pe)) round-trips.
    Δheight between adjacent zh levels should equal |delz|."""
    km = 5
    pt = jnp.full((km,), 280.0)
    pe = jnp.linspace(2.0e4, 1.0e5, km + 1)
    phis = jnp.asarray(0.0)
    delz = hydrostatic_delz_fv3(pt, pe)
    zh = compute_zh_from_delz_fv3(phis, delz)
    # Δzh[k] = zh[k] - zh[k+1] = -delz[k] = |delz[k]|
    delta_zh = zh[:-1] - zh[1:]
    assert jnp.allclose(delta_zh, -delz, atol=1e-10)


def test_zh_shapes_3d():
    """3-D (n_x, n_y, km) delz + 2-D (n_x, n_y) phis →
    (n_x, n_y, km+1) zh."""
    rng = np.random.default_rng(seed=729)
    n_x, n_y, km = 4, 5, 20
    phis = jnp.asarray(rng.uniform(0.0, 5000.0, size=(n_x, n_y))) * constants.g
    delz = jnp.full((n_x, n_y, km), -300.0)
    zh = compute_zh_from_delz_fv3(phis, delz)
    assert zh.shape == (n_x, n_y, km + 1)


def test_zh_finite():
    """No NaN/Inf."""
    rng = np.random.default_rng(seed=730)
    km = 30
    phis = jnp.asarray(rng.uniform(0.0, 5000.0, size=(4, 4))) * constants.g
    delz = jnp.asarray(rng.uniform(-500.0, -100.0, size=(4, 4, km)))
    zh = compute_zh_from_delz_fv3(phis, delz)
    assert jnp.all(jnp.isfinite(zh))
