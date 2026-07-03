"""Byte-identity test for the shared richardson_number helper (#518 §5).

richardson.py and kpp.py computed the gradient Richardson number Ri = N^2/S^2
from byte-identical inline blocks (differing only in the negative-Ri clip).
Pins the factored helper to that exact formula — including the floor-on-the-
SQUARED-denominator form (which differs deliberately from
_shared.vertical_shear_squared's linear-floor form).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from legoesm.ocean.physics.vertical_mixing._shared import (
    richardson_number,
    vertical_shear_squared,
)


def _reference(N2, u, v, dz_actual, eps, clip_negative):
    """Exact pre-#518 inline formula from richardson.py / kpp.py."""
    dz_half = 0.5 * (dz_actual[..., :-1] + dz_actual[..., 1:])
    du = u[..., :-1] - u[..., 1:]
    dv = v[..., :-1] - v[..., 1:]
    S2 = (du**2 + dv**2) / jnp.maximum(dz_half**2, eps)
    Ri = N2 / jnp.maximum(S2, eps)
    if clip_negative:
        Ri = jnp.maximum(Ri, 0.0)
    return Ri


def _fields(seed=0):
    rng = np.random.default_rng(seed)
    shape = (8, 12)  # (..., nlev)
    u = jnp.asarray(rng.standard_normal(shape) * 0.2, dtype=jnp.float64)
    v = jnp.asarray(rng.standard_normal(shape) * 0.2, dtype=jnp.float64)
    dz = jnp.asarray(5.0 + 50.0 * rng.random(shape), dtype=jnp.float64)
    N2 = jnp.asarray(rng.standard_normal((8, 11)) * 1e-4, dtype=jnp.float64)
    return N2, u, v, dz


def test_byte_identical_both_clip_modes():
    N2, u, v, dz = _fields()
    eps = float(jnp.finfo(jnp.float32).eps)
    for clip in (True, False):
        got = richardson_number(N2, u, v, dz, eps=eps, clip_negative=clip)
        ref = _reference(N2, u, v, dz, eps, clip)
        assert np.array_equal(np.asarray(got), np.asarray(ref)), clip


def test_clip_makes_ri_nonnegative():
    # Negative N2 (unstable) -> negative Ri unless clipped.
    N2 = -jnp.ones((4, 5), dtype=jnp.float64) * 1e-4
    u = jnp.zeros((4, 6), dtype=jnp.float64).at[..., 0].set(0.3)
    v = jnp.zeros((4, 6), dtype=jnp.float64)
    dz = jnp.full((4, 6), 10.0, dtype=jnp.float64)
    eps = 1e-12
    ri_clip = richardson_number(N2, u, v, dz, eps=eps, clip_negative=True)
    ri_raw = richardson_number(N2, u, v, dz, eps=eps, clip_negative=False)
    assert np.all(np.asarray(ri_clip) >= 0.0)
    assert np.any(np.asarray(ri_raw) < 0.0)


def test_squared_floor_differs_from_linear_floor_in_thin_layers():
    """Document the deliberate non-merge: the helper's squared-denominator
    floor diverges from vertical_shear_squared's linear floor when dz_half is
    sub-sqrt(eps)-thin (this is why they are kept separate)."""
    eps = float(jnp.finfo(jnp.float32).eps)
    dz_thin = jnp.full((1, 2), 1e-4, dtype=jnp.float64)  # dz_half = 1e-4
    u = jnp.asarray([[0.0, 0.1]], dtype=jnp.float64)
    v = jnp.zeros((1, 2), dtype=jnp.float64)
    dz_half = 0.5 * (dz_thin[..., :-1] + dz_thin[..., 1:])
    s2_linear = vertical_shear_squared(u, v, dz_half)            # floor on dz_half
    du = u[..., :-1] - u[..., 1:]
    s2_squared = (du**2) / jnp.maximum(dz_half**2, eps)          # floor on dz_half^2
    # In this thin layer the two floor conventions give different S2.
    assert not np.allclose(np.asarray(s2_linear), np.asarray(s2_squared))
