"""NEMO's nn_eice attenuation of the Langmuir / wave-breaking TKE sources.

ORCA1 selects nn_eice = 3 (namelist_cfg:453), which saturates at a quarter
ice cover, while legoESM had only the linear form hardcoded — NEMO's
nn_eice = 2. The two agree at zero and full ice and differ most in between,
so a test that only checks the endpoints passes for both and proves nothing;
`test_two_and_three_differ_at_intermediate_cover` is the one that bites.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.physics.vertical_mixing.tke import (
    nemo_ice_tke_attenuation,
)


FR = jnp.asarray([0.0, 0.1, 0.25, 0.5, 0.75, 1.0])


def test_linear_is_one_minus_fraction():
    got = nemo_ice_tke_attenuation(FR, 2)
    assert np.allclose(np.asarray(got), 1.0 - np.asarray(FR))


def test_saturating_reaches_zero_at_quarter_cover():
    """zice_fra = MIN(4 fr, 1), so the factor is 0 for every fr >= 0.25."""
    got = np.asarray(nemo_ice_tke_attenuation(FR, 3))
    assert got[0] == pytest.approx(1.0)
    assert got[1] == pytest.approx(0.6)          # 1 - 4*0.1
    assert np.allclose(got[2:], 0.0)


def test_two_and_three_differ_at_intermediate_cover():
    """The endpoints agree; the middle is where the choice matters.

    This is the non-vacuity control for the pair: both forms give 1.0 at
    fr = 0 and 0.0 at fr = 1, so endpoint-only assertions cannot tell them
    apart. At half cover the linear form still passes half the TKE through
    and the ORCA1 form passes none.
    """
    lin = np.asarray(nemo_ice_tke_attenuation(FR, 2))
    sat = np.asarray(nemo_ice_tke_attenuation(FR, 3))
    assert lin[0] == pytest.approx(sat[0])
    assert lin[-1] == pytest.approx(sat[-1])
    assert lin[3] == pytest.approx(0.5)
    assert sat[3] == pytest.approx(0.0)
    assert float(np.max(lin - sat)) > 0.4


def test_tanh_form_matches_nemo():
    """atol is 1e-7, not the allclose default, and the reason is numerical.

    1 - tanh(10 fr) cancels catastrophically as fr grows: at fr = 0.75 the
    true value is 6.12e-07 and float32 returns 5.96e-07, a 2.6% relative
    error that is inherent to evaluating the difference in single precision
    rather than any defect here. The largest absolute gap across this grid is
    2.03e-08, which clears the default atol of 1e-8 and fails a test that is
    otherwise checking the right thing. Measured, not guessed.
    """
    got = np.asarray(nemo_ice_tke_attenuation(FR, 1))
    assert np.allclose(got, 1.0 - np.tanh(np.asarray(FR) * 10.0), atol=1e-7)


def test_zero_means_no_attenuation():
    assert nemo_ice_tke_attenuation(FR, 0) is None


def test_none_ice_returns_none_for_every_choice():
    for c in (0, 1, 2, 3):
        assert nemo_ice_tke_attenuation(None, c) is None


def test_unknown_choice_raises():
    """Dispatch hardening: an unknown selector must not silently pick one."""
    with pytest.raises(ValueError, match="tke_nn_eice"):
        nemo_ice_tke_attenuation(FR, 4)


def test_factor_never_negative():
    for c in (1, 2, 3):
        got = np.asarray(nemo_ice_tke_attenuation(
            jnp.asarray([0.0, 0.5, 1.0, 1.5]), c))
        assert (got >= 0.0).all()


def test_config_default_is_the_legacy_linear_form():
    """Default must keep existing runs unchanged until the choice is made."""
    from legoesm.ocean.physics.vertical_mixing.config import TKEConfig
    assert TKEConfig().tke_nn_eice == 2
