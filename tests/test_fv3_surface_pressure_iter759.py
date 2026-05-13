"""FV3_3D iter 759: surface_pressure_from_delp_fv3 port.

ps = p_top + sum(delp).

Tests
-----

1. ``test_ps_known_value``.
2. ``test_ps_zero_delp``.
3. ``test_ps_p_top_offset``.
4. ``test_ps_matches_iter731_pe_last``.
5. ``test_ps_shapes_3d``.
6. ``test_ps_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    compute_pe_from_delp_fv3,
    surface_pressure_from_delp_fv3,
)


def test_ps_known_value():
    """Σ delp = 1e5, p_top=0 → ps = 1e5."""
    delp = jnp.full((10,), 1.0e4)
    ps = surface_pressure_from_delp_fv3(delp)
    assert abs(float(ps) - 1.0e5) < 1e-8


def test_ps_zero_delp():
    """Zero delp → ps = p_top."""
    delp = jnp.zeros((5,))
    ps = surface_pressure_from_delp_fv3(delp, p_top=1000.0)
    assert abs(float(ps) - 1000.0) < 1e-10


def test_ps_p_top_offset():
    """ps offset by p_top correctly."""
    delp = jnp.full((5,), 2.0e4)
    ps = surface_pressure_from_delp_fv3(delp, p_top=500.0)
    expected = 500.0 + 5.0 * 2.0e4
    assert abs(float(ps) - expected) < 1e-8


def test_ps_matches_iter731_pe_last():
    """ps == pe[..., -1] when computed via iter-731."""
    rng = np.random.default_rng(seed=759)
    delp = jnp.asarray(rng.uniform(500.0, 2000.0, size=(10,)))
    p_top = 100.0
    ps_via_iter759 = surface_pressure_from_delp_fv3(delp, p_top=p_top)
    pe = compute_pe_from_delp_fv3(delp, p_top=p_top)
    assert abs(float(ps_via_iter759) - float(pe[-1])) < 1e-8


def test_ps_shapes_3d():
    """3-D → 2-D output."""
    rng = np.random.default_rng(seed=760)
    n_x, n_y, km = 4, 5, 20
    delp = jnp.asarray(rng.uniform(500.0, 2000.0, size=(n_x, n_y, km)))
    ps = surface_pressure_from_delp_fv3(delp)
    assert ps.shape == (n_x, n_y)


def test_ps_finite():
    """No NaN/Inf."""
    rng = np.random.default_rng(seed=761)
    delp = jnp.asarray(rng.uniform(100.0, 3000.0, size=(4, 4, 30)))
    ps = surface_pressure_from_delp_fv3(delp)
    assert jnp.all(jnp.isfinite(ps))
