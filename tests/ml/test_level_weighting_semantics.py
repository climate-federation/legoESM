"""Pin what ``level_weighting="pressure"`` actually IS (#1464).

The WeatherBench campaign configs select ``level_weighting: "pressure"``. The
name reads as mass/pressure weighting — monotonic toward the surface, the
GraphCast convention. The implementation is a GAUSSIAN BUMP centred on
``p_ref_Pa`` that downweights BOTH the surface and the top; the monotonic
option is ``"density"``. #1464's degradation pattern ranks almost monotonically
with this weight profile, so the semantics are load-bearing: anyone reading
the config word "pressure" and assuming surface emphasis is training a
different objective than they think.

These tests pin the SHAPE, not the values, so a deliberate retuning of
``p_ref_Pa``/``p_scale_Pa`` passes while a silent change of the option's
character (e.g. someone "fixing" pressure to be monotonic without renaming it,
which would invalidate every result trained under the old meaning) goes red.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from legoesm.training.losses import LossConfig, level_weights

P_S = 101325.0


def _w(option):
    sigma = jnp.linspace(0.02, 0.995, 40)
    return np.asarray(level_weights(
        sigma, P_S, LossConfig(level_weighting=option))), np.asarray(sigma)


def test_pressure_is_a_bump_not_monotonic():
    """"pressure" peaks in the MID troposphere and downweights BOTH ends."""
    w, sigma = _w("pressure")
    k = int(np.argmax(w))
    assert 0 < k < len(w) - 1, "peak sits at an endpoint — no longer a bump"
    # Both ends materially below the peak (the #1464 signature: near-surface
    # fields trained at a fraction of the mid-troposphere weight).
    assert w[-1] < 0.5 * w[k], (w[-1], w[k])
    assert w[0] < 0.5 * w[k], (w[0], w[k])
    p_peak = float(sigma[k]) * P_S
    assert 3.0e4 < p_peak < 7.0e4, f"bump centre moved to {p_peak:.0f} Pa"


def test_density_is_the_monotonic_surface_weighted_option():
    """"density" (the GraphCast-style choice) increases toward the surface."""
    w, _ = _w("density")
    assert np.all(np.diff(w) > 0), "density weighting is no longer monotonic"
    assert w[-1] > 5.0 * w[0]


def test_the_two_options_are_genuinely_different_objectives():
    """Non-vacuity: the surface/mid ratio differs by more than a factor 3."""
    wp, _ = _w("pressure")
    wd, _ = _w("density")
    ratio_p = wp[-1] / wp.max()
    ratio_d = wd[-1] / wd.max()
    assert ratio_d > 3.0 * ratio_p, (ratio_p, ratio_d)
