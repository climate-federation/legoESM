"""RULE 5 tests for the grid-point segment-tail saturation adjustment.

``saturation_adjustment_increment`` runs on the compiled and per-step
segment tails when ``microphysics == "none"``.  Its switch onto the
saturation curve is smooth by default (``SEGMENT_SAT_ADJUST_RH_WIDTH``);
``rh_width=0.0`` is the exact enthalpy-conserving adjustment.  Three tests:

1. hard limit (width 0) matches an independent exact solve;
2. default width: the per-case gap to the hard limit, stored below;
3. ``check_grads`` through the switch at the default width.

float64 explicitly (x64 enabled at import): the stored gaps are fp64 numbers.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from legoesm import constants
from legoesm.atmosphere.physics.microphysics._warm_rain import (
    SEGMENT_SAT_ADJUST_RH_WIDTH, saturation_adjustment_increment)
from legoesm.thermo import saturation_specific_humidity

jax.config.update("jax_enable_x64", True)

DT = 600.0
L_OVER_CP = constants.L_v / constants.c_pd

# (T [K], p [Pa], RH = q_v / q_sat_specific).  No case hits the 5 K cap.
CASES = [
    (300.0, 1.0e5, 0.97),
    (300.0, 1.0e5, 1.001),
    (300.0, 1.0e5, 1.02),
    (285.0, 8.5e4, 1.05),
    (260.0, 6.0e4, 1.10),
    (230.0, 3.0e4, 1.30),
]

# Default-width condensed vapour minus hard-limit condensed vapour [kg/kg],
# per case, measured 2026-11-06 at rh_width = SEGMENT_SAT_ADJUST_RH_WIDTH
# (1/30).  Re-measure (print_gaps below) on any change to the smooth form or
# its default width.
STORED_GAPS = [
    0.0,
    -2.5456800321101202e-06,
    -3.6449905860038e-05,
    -3.435945599861075e-05,
    -7.4425862709914445e-06,
    -9.699790301111286e-09,
]


def _state():
    T = jnp.array([c[0] for c in CASES])
    p = jnp.array([c[1] for c in CASES])
    rh = jnp.array([c[2] for c in CASES])
    return T, rh * saturation_specific_humidity(T, p), p


def _exact_reference(T, q_v, p):
    """Independent exact adjustment: 200-step bisection on
    q_eq = q_sat(T + L_v/c_pd (q_v - q_eq), p), zero where sub-saturated."""
    out = []
    for t, q, pp in zip(np.asarray(T), np.asarray(q_v), np.asarray(p)):
        if q <= float(saturation_specific_humidity(t, pp)):
            out.append(0.0)
            continue
        lo, hi = 0.0, float(q)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            f = mid - float(saturation_specific_humidity(
                t + L_OVER_CP * (q - mid), pp))
            lo, hi = (lo, mid) if f > 0.0 else (mid, hi)
        out.append(float(q) - 0.5 * (lo + hi))
    return np.array(out)


def test_hard_limit_matches_exact_adjustment():
    T, q_v, p = _state()
    dq = np.asarray(saturation_adjustment_increment(T, q_v, p, DT,
                                                    rh_width=0.0))
    ref = _exact_reference(T, q_v, p)
    np.testing.assert_allclose(dq, ref, rtol=1e-9, atol=1e-15)
    # Lands on the curve: post-adjustment state saturated where it condensed.
    T_new = T + L_OVER_CP * dq
    rh_new = (q_v - dq) / saturation_specific_humidity(T_new, p)
    wet = ref > 0.0
    np.testing.assert_allclose(np.asarray(rh_new)[wet], 1.0, rtol=1e-9)


def _default_gaps():
    T, q_v, p = _state()
    soft = saturation_adjustment_increment(
        T, q_v, p, DT, rh_width=SEGMENT_SAT_ADJUST_RH_WIDTH)
    hard = saturation_adjustment_increment(T, q_v, p, DT, rh_width=0.0)
    return np.asarray(soft - hard)


@pytest.mark.parametrize("i", range(len(CASES)))
def test_default_width_gap_is_recorded(i):
    gap = _default_gaps()[i]
    np.testing.assert_allclose(gap, STORED_GAPS[i], rtol=1e-6, atol=1e-12)


def test_gradient_through_switch_at_default():
    T, q_v, p = _state()

    def f(T, q_v):
        return saturation_adjustment_increment(
            T, q_v, p, DT, rh_width=SEGMENT_SAT_ADJUST_RH_WIDTH)

    check_grads(f, (T, q_v), order=1, modes=("rev",), eps=1e-6,
                atol=1e-5, rtol=1e-5)
    g = jax.grad(lambda q: jnp.sum(f(T, q)))(q_v)
    assert np.all(np.isfinite(np.asarray(g)))
    assert np.any(np.asarray(g) != 0.0)


if __name__ == "__main__":   # print_gaps: re-measure STORED_GAPS
    print(repr([float(x) for x in _default_gaps()]))
