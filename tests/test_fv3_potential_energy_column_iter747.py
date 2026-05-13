"""FV3_3D iter 747: potential_energy_column_fv3 port.

PE_col = sum(delp * 0.5*(phi[k]+phi[k+1])) / g.

Tests
-----

1. ``test_pe_zero_phi``.
2. ``test_pe_uniform_phi_known_value``.
3. ``test_pe_with_hydrostatic_phi``.
4. ``test_pe_shapes_3d``.
5. ``test_pe_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import (
    geopotential_from_T_peln_fv3,
    potential_energy_column_fv3,
)


def test_pe_zero_phi():
    """phi=0 everywhere → PE=0."""
    km = 5
    phi = jnp.zeros((km + 1,))
    delp = jnp.full((km,), 1.0e4)
    pe = potential_energy_column_fv3(phi, delp)
    assert abs(float(pe)) < 1e-15


def test_pe_uniform_phi_known_value():
    """Uniform phi=Φ → phi_avg=Φ → PE = Φ·p_s/g."""
    km = 10
    Phi_val = 5.0e4   # ~5 km · g
    phi = jnp.full((km + 1,), Phi_val)
    delp = jnp.full((km,), 1.0e4)
    pe = potential_energy_column_fv3(phi, delp)
    expected = Phi_val * 1.0e5 / constants.g
    assert abs(float(pe) - expected) / expected < 1e-10


def test_pe_with_hydrostatic_phi():
    """Combine iter-738 phi + iter-747 PE → finite, sensible value."""
    km = 10
    pt = jnp.full((km,), 280.0)
    pe_face = jnp.linspace(1.0e4, 1.0e5, km + 1)
    peln = jnp.log(pe_face)
    phis = jnp.asarray(0.0)
    phi = geopotential_from_T_peln_fv3(pt, peln, phis)
    delp = pe_face[1:] - pe_face[:-1]
    pe = potential_energy_column_fv3(phi, delp)
    # PE > 0 for non-trivial column
    assert float(pe) > 0.0
    # Physical scale: ~few × 1e8 J/m²
    assert 1e6 < float(pe) < 1e10


def test_pe_shapes_3d():
    """3-D → 2-D output."""
    rng = np.random.default_rng(seed=747)
    n_x, n_y, km = 4, 5, 20
    phi = jnp.asarray(rng.uniform(0.0, 1.0e5, size=(n_x, n_y, km + 1)))
    delp = jnp.full((n_x, n_y, km), 1000.0)
    pe = potential_energy_column_fv3(phi, delp)
    assert pe.shape == (n_x, n_y)


def test_pe_finite():
    """No NaN/Inf."""
    rng = np.random.default_rng(seed=748)
    phi = jnp.asarray(rng.uniform(0.0, 2.0e5, size=(4, 4, 31)))
    delp = jnp.asarray(rng.uniform(500.0, 2000.0, size=(4, 4, 30)))
    pe = potential_energy_column_fv3(phi, delp)
    assert jnp.all(jnp.isfinite(pe))
