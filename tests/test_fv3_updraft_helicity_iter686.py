"""FV3_3D iter 686: updraft_helicity_fv3 port.

Faithful JAX port of FV3 ``updraft_helicity`` (tools/
fv_diagnostics.F90:5048-5108).

Tests
-----

1. ``test_uh_zero_vort``.
2. ``test_uh_zero_w``.
3. ``test_uh_uniform_in_window``.
4. ``test_uh_outside_window_zero``.
5. ``test_uh_hydrostatic_requires_args``.
6. ``test_uh_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import updraft_helicity_fv3


def _column_setup(km=20):
    """Uniform 500-m layers."""
    delz = jnp.full((km,), -500.0)
    vort = jnp.zeros((km,))
    w = jnp.zeros((km,))
    return delz, vort, w


def test_uh_zero_vort():
    """Zero vort → uh = 0."""
    km = 20
    delz, _, w = _column_setup(km)
    w = jnp.full((km,), 5.0)
    uh = updraft_helicity_fv3(jnp.zeros((km,)), w, delz=delz)
    assert abs(float(uh)) < 1e-12


def test_uh_zero_w():
    """Zero w → uh = 0."""
    km = 20
    delz, vort, _ = _column_setup(km)
    vort = jnp.full((km,), 0.01)
    uh = updraft_helicity_fv3(vort, jnp.zeros((km,)), delz=delz)
    assert abs(float(uh)) < 1e-12


def test_uh_uniform_in_window():
    """Uniform vort=Ω, w=W in 2-5 km window: uh = Ω·W·(5000-2000) = 3000·Ω·W."""
    km = 20  # 500-m layers; bottom at k=km-1, top at k=0
    delz = jnp.full((km,), -500.0)
    Omega = 0.01
    W = 5.0
    vort = jnp.full((km,), Omega)
    w_arr = jnp.full((km,), W)
    uh = updraft_helicity_fv3(vort, w_arr, delz=delz, z_bot=2000.0, z_top=5000.0)
    expected = Omega * W * (5000.0 - 2000.0)
    assert abs(float(uh) - expected) / expected < 1e-12


def test_uh_outside_window_zero():
    """If z_top ≤ z_bot → uh = 0."""
    km = 20
    delz, _, _ = _column_setup(km)
    vort = jnp.full((km,), 0.01)
    w_arr = jnp.full((km,), 5.0)
    uh = updraft_helicity_fv3(vort, w_arr, delz=delz, z_bot=2000.0, z_top=2000.0)
    assert abs(float(uh)) < 1e-12


def test_uh_hydrostatic_requires_args():
    """hydrostatic=True without pt/q/peln raises ValueError."""
    km = 10
    vort = jnp.zeros((km,))
    w = jnp.zeros((km,))
    with pytest.raises(ValueError):
        updraft_helicity_fv3(vort, w, hydrostatic=True)


def test_uh_finite():
    """No NaN/Inf on random inputs."""
    rng = np.random.default_rng(seed=686)
    km = 30
    vort = jnp.asarray(rng.uniform(-0.05, 0.05, size=(4, 4, km)))
    w = jnp.asarray(rng.uniform(-5, 10, size=(4, 4, km)))
    delz = jnp.full((4, 4, km), -300.0)
    uh = updraft_helicity_fv3(vort, w, delz=delz)
    assert jnp.all(jnp.isfinite(uh))
