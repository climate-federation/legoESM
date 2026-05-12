"""FV3_3D iter 726: hydrostatic_delz_fv3 port.

Faithful JAX port of FV3 hydrostatic delz initialization
(fv_mapz.F90:3402, 3411).

Tests
-----

1. ``test_hydro_delz_dry_isothermal``.
2. ``test_hydro_delz_moist_uses_Tv``.
3. ``test_hydro_delz_sign_convention``.
4. ``test_hydro_delz_thick_layer``.
5. ``test_hydro_delz_shapes_3d``.
6. ``test_hydro_delz_finite``.
7. ``test_hydro_delz_moist_requires_q``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import hydrostatic_delz_fv3


def test_hydro_delz_dry_isothermal():
    """Single isothermal layer:
        delz = (R_d/g) · T · (p_top − p_bot)
    With T=280, p_top=9e4, p_bot=1e5:
        delz = R_d/g · 280 · (-1e4)
    """
    pt = jnp.array([280.0])
    pe = jnp.array([9.0e4, 1.0e5])
    delz = hydrostatic_delz_fv3(pt, pe)
    expected = constants.R_d / constants.g * 280.0 * (-1.0e4)
    assert abs(float(delz[0]) - expected) / abs(expected) < 1e-10


def test_hydro_delz_moist_uses_Tv():
    """Moist branch: delz = (R_d/g)·T·(1+zvir·q)·Δpe.
    Moist atmosphere should give thicker (more negative) layer
    than dry."""
    pt = jnp.array([290.0])
    pe = jnp.array([9.0e4, 1.0e5])
    q = jnp.array([0.015])
    delz_dry = hydrostatic_delz_fv3(pt, pe)
    delz_moist = hydrostatic_delz_fv3(pt, pe, q=q, moist=True)
    # |delz_moist| > |delz_dry| (Tv > T)
    assert abs(float(delz_moist[0])) > abs(float(delz_dry[0]))
    zvir = constants.R_v / constants.R_d - 1.0
    expected = constants.R_d / constants.g * 290.0 * (1.0 + zvir * 0.015) * (-1.0e4)
    assert abs(float(delz_moist[0]) - expected) / abs(expected) < 1e-10


def test_hydro_delz_sign_convention():
    """FV3 convention: pe[k] < pe[k+1] (top to bottom) → delz < 0."""
    pt = jnp.full((5,), 280.0)
    pe = jnp.linspace(1.0e4, 1.0e5, 6)  # monotone increasing
    delz = hydrostatic_delz_fv3(pt, pe)
    assert jnp.all(delz < 0.0)


def test_hydro_delz_thick_layer():
    """Layer near top is thicker (lower density) than at surface."""
    pe = jnp.array([5.0e3, 1.0e4, 5.0e4, 1.0e5])
    pt = jnp.full((3,), 250.0)
    delz = hydrostatic_delz_fv3(pt, pe)
    # |delz[0]| corresponds to (1e4 - 5e3) = 5e3 → smaller in magnitude
    # |delz[1]| corresponds to (5e4 - 1e4) = 4e4 → larger
    # |delz[2]| corresponds to (1e5 - 5e4) = 5e4 → largest
    assert abs(float(delz[2])) > abs(float(delz[1]))
    assert abs(float(delz[1])) > abs(float(delz[0]))


def test_hydro_delz_shapes_3d():
    """3-D input → 3-D output."""
    rng = np.random.default_rng(seed=726)
    n_x, n_y, km = 4, 5, 20
    pe_col = jnp.linspace(1.0e4, 1.0e5, km + 1)
    pe = jnp.broadcast_to(pe_col[None, None, :], (n_x, n_y, km + 1))
    pt = jnp.asarray(rng.uniform(220.0, 300.0, size=(n_x, n_y, km)))
    delz = hydrostatic_delz_fv3(pt, pe)
    assert delz.shape == (n_x, n_y, km)


def test_hydro_delz_finite():
    """No NaN/Inf."""
    rng = np.random.default_rng(seed=727)
    km = 30
    pe_col = jnp.linspace(1.0e4, 1.0e5, km + 1)
    pe = jnp.broadcast_to(pe_col[None, None, :], (4, 4, km + 1))
    pt = jnp.asarray(rng.uniform(200.0, 320.0, size=(4, 4, km)))
    q = jnp.asarray(rng.uniform(0.0, 0.025, size=(4, 4, km)))
    delz = hydrostatic_delz_fv3(pt, pe, q=q, moist=True)
    assert jnp.all(jnp.isfinite(delz))


def test_hydro_delz_moist_requires_q():
    """moist=True without q raises ValueError."""
    pt = jnp.full((5,), 280.0)
    pe = jnp.linspace(1.0e4, 1.0e5, 6)
    with pytest.raises(ValueError):
        hydrostatic_delz_fv3(pt, pe, moist=True)
