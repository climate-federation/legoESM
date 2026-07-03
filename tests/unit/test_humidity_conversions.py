"""Direct unit tests for the canonical specific-humidity ↔ mixing-ratio helpers
in :mod:`legoesm.thermo`.

These conversions are the boundary between SPECIFIC humidity ``q = m_v/(m_v+m_d)``
(ERA5 / DEPHY / reanalysis convention) and the MASS MIXING RATIO ``r = m_v/m_d``
that the legoESM physics path consumes.  The four ERA5→grid carry functions
(`era5_to_state.py`) route their moisture through ``specific_humidity_to_mixing_ratio``
in FLOAT32, so the helper's divide-by-zero guard must actually hold at float32
precision — a clip to ``1 - 1e-12`` alone does NOT (``1 - 1e-12`` rounds to ``1.0``
in float32), which previously produced ``inf`` at ``q ≥ 1``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from legoesm.thermo import (
    mixing_ratio_to_specific_humidity,
    specific_humidity_to_mixing_ratio,
)


def test_basic_conversion_matches_formula():
    q = jnp.array([0.0, 5.0e-3, 0.02, 0.04])
    r = specific_humidity_to_mixing_ratio(q)
    np.testing.assert_allclose(np.asarray(r), np.asarray(q) / (1.0 - np.asarray(q)),
                               rtol=1e-7)
    # mixing ratio strictly exceeds specific humidity for q > 0
    assert np.all(np.asarray(r)[1:] > np.asarray(q)[1:])


def test_round_trip_specific_mixing_specific():
    q = jnp.array([0.0, 1.0e-4, 0.01, 0.03, 0.05])
    q_back = mixing_ratio_to_specific_humidity(specific_humidity_to_mixing_ratio(q))
    np.testing.assert_allclose(np.asarray(q_back), np.asarray(q), rtol=1e-6, atol=1e-12)


def test_negative_input_floored_to_zero():
    """Interpolation undershoot (q < 0) floors to r = 0, never negative."""
    r = specific_humidity_to_mixing_ratio(jnp.array([-1.0e-3, -0.5, 0.0]))
    np.testing.assert_allclose(np.asarray(r), 0.0)


def test_float32_divide_safety_no_inf():
    """The guard must hold at float32 precision: corrupt q ≥ 1 in float32 (where
    1 - 1e-12 rounds to 1.0) must give a large-but-FINITE mixing ratio, not inf.
    This is the exact hazard the four ERA5 carries (float32) rely on."""
    q32 = jnp.asarray(np.array([1.0, 1.0 - 1e-12, 2.0], dtype=np.float32))
    r = specific_humidity_to_mixing_ratio(q32)
    assert np.all(np.isfinite(np.asarray(r))), np.asarray(r)
    assert np.all(np.asarray(r) > 0.0)


def test_jit_and_grad_safe():
    import jax

    f = jax.jit(lambda q: specific_humidity_to_mixing_ratio(q).sum())
    g = jax.grad(f)(jnp.array([0.005, 0.02, 0.04]))
    assert np.all(np.isfinite(np.asarray(g)))
    # dr/dq = 1/(1-q)^2 > 0
    assert np.all(np.asarray(g) > 0.0)
