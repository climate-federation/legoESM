from __future__ import annotations

import pytest

from legoesm.ocean.fidelity.nemo_testcase_recipe import build_gyre_zco_card
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round237_bt_alpha_gate as gate,
)


def test_required_card_alpha_has_no_physical_fallback():
    card = build_gyre_zco_card()
    gate.validate_card_alpha(card, 0.07)
    missing = gate._replace_card_alpha(card, None)
    with pytest.raises(ValueError, match="must state its deck's rn_bt_alpha"):
        gate.validate_card_alpha(missing, 0.07)


def test_wrong_card_alpha_is_refused():
    card = gate._replace_card_alpha(build_gyre_zco_card(), 0.09)
    with pytest.raises(ValueError, match="does not match the executed deck"):
        gate.validate_card_alpha(card, 0.07)


def test_alpha009_known_answer_is_exact():
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        nemo_ab3am4_coeff_arrays,
    )

    _, weights = nemo_ab3am4_coeff_arrays(3, alpha=0.09, ramp=True)
    import numpy as np
    np.testing.assert_array_equal(np.asarray(weights)[2], gate.EXPECTED_WEIGHTS_009)
