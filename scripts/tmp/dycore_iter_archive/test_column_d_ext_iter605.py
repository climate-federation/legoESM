"""FV3_3D iter 605: column_mass_weighted_mean + column_d_ext_field utilities.

Faithful port of FV3 dyn_core.F90:1310-1326.

Tests
-----

1. ``test_column_mass_weighted_mean_constant_field``.
2. ``test_column_mass_weighted_mean_zero_mass_safe``.
3. ``test_column_d_ext_disabled_returns_zero``.
4. ``test_column_d_ext_formula``.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.diagnostics import (
    column_d_ext_field,
    column_mass_weighted_mean,
)


def test_column_mass_weighted_mean_constant_field():
    """Constant field along levels → mean = constant."""
    field = jnp.ones((6, 8, 8, 5)) * 3.7
    mass = jnp.asarray(np.random.default_rng(605).uniform(1, 5, size=(6, 8, 8, 5)))
    out = column_mass_weighted_mean(field, mass)
    assert out.shape == (6, 8, 8)
    assert jnp.allclose(out, 3.7), f"constant field → mean = 3.7; got max {out.max()}"


def test_column_mass_weighted_mean_zero_mass_safe():
    """Zero mass → returns 0 (no NaN/Inf)."""
    field = jnp.ones((4, 5)) * 2.0
    mass = jnp.zeros((4, 5))
    out = column_mass_weighted_mean(field, mass)
    assert jnp.all(out == 0.0), f"expected all zeros; got {out}"
    assert jnp.all(jnp.isfinite(out))


def test_column_d_ext_disabled_returns_zero():
    """d_ext = 0 → divg2 = 0 (FV3 line 1329)."""
    vt = jnp.ones((6, 8, 8, 5))
    delp = jnp.ones((6, 8, 8, 5)) * 100.0
    out = column_d_ext_field(vt, delp, d_ext=0.0, da_min_c=1e10)
    assert jnp.all(out == 0.0)


def test_column_d_ext_formula():
    """divg2 = d_ext · da_min_c · column_mass_mean(vt).

    With vt = const c, column_mass_mean = c, so divg2 = d_ext·da_min_c·c.
    """
    c = 2.5
    vt = jnp.ones((6, 8, 8, 5)) * c
    delp = jnp.ones((6, 8, 8, 5)) * 100.0
    d_ext = 0.02
    da_min_c = 1e10
    out = column_d_ext_field(vt, delp, d_ext=d_ext, da_min_c=da_min_c)
    expected = d_ext * da_min_c * c
    assert jnp.allclose(out, expected), (
        f"expected divg2 = {expected:.3e}; got max {float(out.max()):.3e}"
    )
