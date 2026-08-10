"""Prognostic CLUBB must not manufacture a 3685 K^2 temperature variance.

The maximum-correlation floor ``thlp2 >= wpthlp**2 / (wp2 * 0.99**2)`` divided
by the RAW ``wp2``. The reference gets a floored denominator for free from its
call order -- ``advance_wp2_wp3`` floors ``wp2`` at ``w_tol**2`` and runs first
-- but this port runs the scalar-variance solve first and so, on step 1, saw
the seeded ``wp2 = tke_min = 1e-6``, four hundred times below that floor.

It was invisible for as long as the surface flux was zero, because the
numerator is ``wpthlp**2`` and ``0 / 1e-6`` is 0. The moment a prescribed-flux
case actually delivered its surface heat flux to the closure, the dry
convective column got ``thlp2 = 3685 K^2`` -- a 61 K RMS temperature
fluctuation -- and drifted linearly by ~4.5 K per step, scoring 772 against
0.043 with the flux withheld.

These tests pin the arithmetic of the floor, both ends of the fix, and the
property that made it hide.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from legoesm.atmosphere.physics.turbulence.clubb import (  # noqa: E402
    _MAX_MAG_CORRELATION_FLUX,
    CLUBBConfig,
    init_clubb_moments,
)

# The surface kinematic heat flux the dry convective deck delivers:
# 70.03 W/m^2 / (rho c_pd exner) at the surface.
CBL_WPTHLP_SFC = 0.0601


def _floor(wpthlp, wp2, max_corr=_MAX_MAG_CORRELATION_FLUX):
    """The maximum-correlation variance floor, as the solver applies it."""
    return wpthlp ** 2 / (wp2 * max_corr ** 2)


def test_the_unfloored_arithmetic_reproduces_the_observed_blow_up():
    """The number the probe actually measured, from the formula alone.

    This is the diagnosis, not the fix: with the OLD seed the floor evaluates
    to the 3690 K^2 the side-by-side column probe printed, every step.
    """
    got = _floor(CBL_WPTHLP_SFC, 1.0e-6)
    assert got == pytest.approx(3685.0, rel=0.01)
    # a 61 K RMS temperature fluctuation, i.e. not a temperature fluctuation
    assert np.sqrt(got) > 60.0


def test_flooring_the_denominator_bounds_the_variance():
    """w_tol**2 is 400x tke_min, so the floor drops by the same factor."""
    cfg = CLUBBConfig()
    w_tol_sqd = float(cfg.w_tol) ** 2
    unfloored = _floor(CBL_WPTHLP_SFC, 1.0e-6)
    floored = _floor(CBL_WPTHLP_SFC, w_tol_sqd)
    assert unfloored / floored == pytest.approx(w_tol_sqd / 1.0e-6, rel=1e-9)
    assert floored < 10.0, "a few K RMS, not sixty"


def test_zero_surface_flux_hides_it_entirely():
    """Why this survived until a prescribed flux was delivered to the closure.

    The numerator is squared, so it is also why the SIGN does not matter: a
    stable case with a cooling flux is hit exactly as hard as a convective one.
    """
    assert _floor(0.0, 1.0e-6) == 0.0
    warming = _floor(+CBL_WPTHLP_SFC, 1.0e-6)
    cooling = _floor(-CBL_WPTHLP_SFC, 1.0e-6)
    assert warming == pytest.approx(cooling)


def test_seed_is_not_below_the_floor_the_core_enforces():
    """The other end of the fix: init must not seed below w_tol**2."""
    cfg = CLUBBConfig()
    st = init_clubb_moments(1, 24, cfg, dtype=jnp.float64)
    w_tol_sqd = float(cfg.w_tol) ** 2
    for name in ("wp2", "up2", "vp2"):
        arr = np.asarray(getattr(st, name))
        assert np.all(arr >= w_tol_sqd - 1e-18), (
            f"{name} seeded at {arr.min():.3e}, below the w_tol**2 floor "
            f"{w_tol_sqd:.3e} that advance_wp2_wp3 enforces from its first "
            "advance")


def test_seeded_state_cannot_produce_an_absurd_variance():
    """End to end on the seed: the floor evaluated on the seeded wp2 is sane."""
    cfg = CLUBBConfig()
    st = init_clubb_moments(1, 24, cfg, dtype=jnp.float64)
    wp2_seed = float(np.asarray(st.wp2).min())
    got = _floor(CBL_WPTHLP_SFC, wp2_seed)
    assert got < 10.0, f"seeded floor still {got:.1f} K^2"
    assert np.sqrt(got) < 4.0, "RMS temperature fluctuation must stay plausible"


def test_other_variances_keep_their_own_tolerance_floors():
    """The scalar variances must still start at their tolerance-squared."""
    cfg = CLUBBConfig()
    st = init_clubb_moments(1, 24, cfg, dtype=jnp.float64)
    assert np.allclose(np.asarray(st.thlp2), float(cfg.thl_tol) ** 2)
    assert np.allclose(np.asarray(st.rtp2), float(cfg.rt_tol) ** 2)
    # fluxes and wp3 start at rest
    for name in ("wprtp", "wpthlp", "upwp", "vpwp", "wp3", "rtpthlp"):
        assert np.allclose(np.asarray(getattr(st, name)), 0.0)
