"""IFS/SAM homogeneous-freezing ice-supersaturation allowance (``rh_homo``).

Reference: gSAM 1.8.7 ``SRC/MICRO_SAM1MOM/cloud.f90`` (Khairoutdinov 2023,
"Modeled after IFS model")::

    if(tabs.lt.235.and.qci.lt.1.e-8) then
      rh_homo = 2.583 - tabs/207.8
    else
      rh_homo = 1.
    end if
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.thermo import (
    homogeneous_freezing_rh_factor,
    saturation_mixing_ratio_ice,
)


def test_matches_the_cloud_f90_ramp_below_235K():
    T = jnp.asarray([200.0, 210.0, 220.0, 230.0, 234.9])
    got = np.asarray(homogeneous_freezing_rh_factor(T))
    want = 2.583 - np.asarray(T) / 207.8
    np.testing.assert_allclose(got, want, rtol=1e-12)
    # bracketing values quoted in the source comment
    assert 1.44 < want[-1] < 1.46      # ~1.45 at 235 K
    assert 1.61 < want[0] < 1.63       # ~1.62 at 200 K


def test_no_allowance_at_or_above_235K():
    T = jnp.asarray([235.0, 250.0, constants.T_freeze, 300.0])
    np.testing.assert_array_equal(
        np.asarray(homogeneous_freezing_rh_factor(T)), np.ones(4),
    )


def test_allowance_withdrawn_once_ice_exists():
    """cloud.f90: "if ice already exists - do as usual, that is no
    supersaturation over ice"."""
    T = jnp.full((3,), 210.0)
    q_ice = jnp.asarray([0.0, 1.0e-9, 1.0e-7])  # below, below, ABOVE 1e-8
    got = np.asarray(homogeneous_freezing_rh_factor(T, q_ice))
    assert got[0] > 1.3 and got[1] > 1.3   # pristine -> allowance
    assert got[2] == 1.0                   # ice present -> withdrawn


def test_disabled_switch_returns_unity_everywhere():
    T = jnp.asarray([190.0, 210.0, 234.0])
    np.testing.assert_array_equal(
        np.asarray(homogeneous_freezing_rh_factor(T, enabled=False)),
        np.ones(3),
    )


def test_factor_never_reduces_the_saturation_target():
    """A multiplier < 1 would make cold air condense EARLIER than plain ice
    saturation — the ramp crosses 1.0 at 236.6 K, above the 235 K gate, but
    pin the floor so a future constant edit cannot invert the sign."""
    T = jnp.linspace(150.0, 320.0, 400)
    assert float(jnp.min(homogeneous_freezing_rh_factor(T))) >= 1.0


@pytest.mark.parametrize("scheme", ["morrison", "thompson", "p3"])
def test_schemes_default_to_the_allowance_on(scheme):
    from legoesm.atmosphere.physics.microphysics import config as cfg
    sub = {"morrison": cfg.MorrisonConfig, "thompson": cfg.ThompsonConfig,
           "p3": cfg.P3Config}[scheme]()
    assert sub.homogeneous_ice_supersaturation is True
    assert sub._replace(
        homogeneous_ice_supersaturation=False,
    ).homogeneous_ice_supersaturation is False


def test_raises_the_ice_target_where_it_applies():
    """The point of the factor: the deposition target in pristine cold air is
    the homogeneous-freezing threshold, not plain ice saturation."""
    T = jnp.asarray([210.0])
    p = jnp.asarray([25_000.0])
    q_ice = jnp.asarray([0.0])
    plain = saturation_mixing_ratio_ice(T, p)
    raised = plain * homogeneous_freezing_rh_factor(T, q_ice)
    assert float((raised / plain)[0]) == pytest.approx(2.583 - 210.0 / 207.8)
