"""FV3_3D iter 663: u_jet_fv3 + gh_jet_fv3 ports.

Faithful JAX ports of FV3 ``u_jet`` and ``gh_jet``
(tools/test_cases.F90:4297-4348, 4349-4360) for Galewsky-like
barotropic-instability test.

Tests
-----

1. ``test_u_jet_zero_outside_band``.
2. ``test_u_jet_finite_inside``.
3. ``test_u_jet_centered_peak_near_umax``.
4. ``test_gh_jet_shape``.
5. ``test_gh_jet_south_pole``.
6. ``test_gh_jet_monotonic_outside_band``.
7. ``test_gh_jet_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import gh_jet_fv3, u_jet_fv3


def test_u_jet_zero_outside_band():
    """u_jet = 0 at equator and poles."""
    assert abs(float(u_jet_fv3(jnp.asarray(0.0)))) < 1e-14
    assert abs(float(u_jet_fv3(jnp.asarray(-jnp.pi / 2 + 0.01)))) < 1e-14
    assert abs(float(u_jet_fv3(jnp.asarray(jnp.pi / 2 - 0.01)))) < 1e-14


def test_u_jet_finite_inside():
    """u_jet finite inside jet band."""
    lats = jnp.linspace(jnp.pi / 7 + 0.01, jnp.pi / 2 - jnp.pi / 7 - 0.01, 20)
    u = u_jet_fv3(lats)
    assert jnp.all(jnp.isfinite(u))
    assert jnp.all(u > 0)


def test_u_jet_centered_peak_near_umax():
    """Peak of u_jet is approximately umax at lat = (ph0+ph1)/2."""
    ph0 = jnp.pi / 7
    ph1 = jnp.pi / 2 - jnp.pi / 7
    center = (ph0 + ph1) / 2
    umax = 80.0
    peak = float(u_jet_fv3(jnp.asarray(center), umax=umax))
    # At center, (lat-ph0)*(lat-ph1) = -((ph1-ph0)/2)² so
    # u_jet = (umax/en)·exp(-4/(ph1-ph0)²) = umax exactly.
    assert abs(peak - umax) < 1e-10


def test_gh_jet_shape():
    """gh_jet output shape matches input."""
    lats = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, 50)
    gh = gh_jet_fv3(lats)
    assert gh.shape == (50,)


def test_gh_jet_south_pole():
    """At lat = -π/2 (south pole), gh ≈ g·h0."""
    gh_sp = float(gh_jet_fv3(jnp.asarray(-jnp.pi / 2)))
    expected = constants.g * 10157.946867
    assert abs(gh_sp - expected) / expected < 1e-3


def test_gh_jet_monotonic_outside_band():
    """Outside the jet band (no acceleration), gh is constant."""
    lats = jnp.linspace(-jnp.pi / 2 + 0.01, jnp.pi / 8, 30)  # south of ph0
    gh = gh_jet_fv3(lats)
    # In the south of band, u_jet=0 so gh stays constant
    # (depends on integration starting south).  Just verify monotonic
    # or nearly-constant
    diffs = jnp.diff(gh)
    # Outside band, diff should be ~0
    assert float(jnp.max(jnp.abs(diffs))) < 1.0


def test_gh_jet_finite():
    """gh_jet finite across full lat range."""
    lats = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, 100)
    gh = gh_jet_fv3(lats)
    assert jnp.all(jnp.isfinite(gh))
