"""Controls for the SCM-RCE T/q objective probe.

The probe exists to decide which humidity metric the tuning campaign will
minimise, so it is an INSTRUMENT: a bug here picks the wrong objective and the
whole campaign optimises the wrong thing without ever erroring.  Every test
below is a case whose answer is known analytically before the probe runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.validate.scm_rce_tq_objective_probe import (
    SPLIT_HEIGHTS_KM,
    _relative_perturbation_shares,
)


def _linear_grid(n=50, top_km=20.0):
    z_km = np.linspace(0.0, top_km, n)
    w = np.full(n, 1.0 / n)
    return z_km, w


def test_uniform_profile_share_equals_mass_share():
    """A profile that is CONSTANT in the vertical must weight the metric
    exactly like the mass weights — anything else means the probe is inventing
    a height bias that the metric does not have."""
    z_km, w = _linear_grid()
    values = np.full(z_km.size, 5.0e-3)
    out = _relative_perturbation_shares(
        values, w, z_km, transform="identity", relative_error=0.1)
    for h in SPLIT_HEIGHTS_KM:
        assert out[f"share_above_{h:g}km"] == pytest.approx(
            out[f"mass_share_above_{h:g}km"], rel=1e-12)


def test_decaying_profile_is_bottom_heavy():
    """The measured claim: a q_v-like profile decaying three decades over the
    troposphere makes an ABSOLUTE metric blind above 5 km."""
    z_km, w = _linear_grid()
    qv = 18.0e-3 * 10.0 ** (-3.0 * z_km / 15.0)
    out = _relative_perturbation_shares(
        qv, w, z_km, transform="identity", relative_error=0.1)
    share = out["share_above_5km"]
    mass = out["mass_share_above_5km"]
    assert share < 0.05 * mass, (
        f"expected the absolute metric to be far more bottom-heavy than the "
        f"mass distribution; got share={share:.4g} vs mass={mass:.4g}")


def test_log_transform_recovers_the_mass_distribution():
    """``d log q`` is the same at every level under a uniform relative error, so
    the log metric looks exactly where the mass is, whatever the profile."""
    z_km, w = _linear_grid()
    qv = 18.0e-3 * 10.0 ** (-3.0 * z_km / 15.0)
    out = _relative_perturbation_shares(
        qv, w, z_km, transform="log", relative_error=0.1)
    for h in SPLIT_HEIGHTS_KM:
        assert out[f"share_above_{h:g}km"] == pytest.approx(
            out[f"mass_share_above_{h:g}km"], rel=1e-12)


def test_shares_are_independent_of_the_perturbation_size():
    """The printed table is only meaningful if the SHARES do not depend on the
    arbitrary 10 % — verify rather than assert it in a docstring."""
    z_km, w = _linear_grid()
    qv = 18.0e-3 * 10.0 ** (-3.0 * z_km / 15.0)
    a = _relative_perturbation_shares(
        qv, w, z_km, transform="identity", relative_error=0.01)
    b = _relative_perturbation_shares(
        qv, w, z_km, transform="identity", relative_error=0.50)
    for h in SPLIT_HEIGHTS_KM:
        assert a[f"share_above_{h:g}km"] == pytest.approx(
            b[f"share_above_{h:g}km"], rel=1e-12)


def test_unknown_transform_raises():
    z_km, w = _linear_grid()
    with pytest.raises(ValueError, match="unknown transform"):
        _relative_perturbation_shares(
            np.ones_like(z_km), w, z_km, transform="sqrt", relative_error=0.1)


def test_degenerate_total_raises_instead_of_dividing_by_zero():
    """An all-zero profile would make every share 0/0.  A NaN share that prints
    as a plausible blank is exactly the failure this repo keeps hitting, so it
    must be a hard error."""
    z_km, w = _linear_grid()
    with pytest.raises(ValueError, match="degenerate total"):
        _relative_perturbation_shares(
            np.zeros_like(z_km), w, z_km,
            transform="identity", relative_error=0.1)
