"""Cooper (1986) base ice-crystal number ``N_i0`` — P3 and Thompson must use
the canonical base 0.005/L = 5/m^3 (= ``MorrisonConfig``), not 5e3.

Physics-review fix (2026-06-29): ``P3Config.N_i0`` and ``ThompsonConfig.N_i0``
defaulted to ``5e3`` m^-3 — 1000x the Cooper (1986) base — which pinned all
clouds colder than ~-15 C at the ``N_i_nuc_max`` cap.  ``MorrisonConfig`` already
used the correct 5.0, with a comment noting 5e3 over-nucleates by 1000x.  All
three schemes apply the identical Cooper exponential
``N_i0 * exp(cooper_a * max(T_freeze - T, 0))``.
"""
from __future__ import annotations

import numpy as np

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.config import (
    MorrisonConfig,
    ThompsonConfig,
    P3Config,
    __param_spec__,
)


def test_p3_thompson_n_i0_match_cooper_base():
    """All three Cooper-nucleation schemes use the canonical 5/m^3 base."""
    assert MorrisonConfig().N_i0 == 5.0   # already-correct sibling (reference)
    assert ThompsonConfig().N_i0 == 5.0
    assert P3Config().N_i0 == 5.0


def test_n_i0_spec_bounds_bracket_cooper_base():
    """Trainable bounds bracket the canonical base, not the 1000x-too-high band."""
    for name in ("MorrisonConfig", "ThompsonConfig", "P3Config"):
        bounds = tuple(__param_spec__[name]["params"]["N_i0"]["bounds"])
        assert bounds == (1.0, 50.0), f"{name} N_i0 bounds = {bounds}"


def test_cooper_number_at_minus20C_is_order_per_litre():
    """Cooper ``N_i = N_i0 * exp(cooper_a*(T_f - T))``; at T = -20 C (dT = 20 K)
    with the correct base this is order 1/L, not order 1000/L."""
    cfg = P3Config()
    dT = 20.0  # supercooling [K]
    N_i = cfg.N_i0 * np.exp(cfg.cooper_a * dT)  # [1/m^3]
    per_litre = N_i / 1000.0
    assert 0.5 < per_litre < 20.0, per_litre  # the Cooper supercooled regime
    # the old 5e3 base would have given ~2300/L here
    assert per_litre < 100.0
    # sanity: T_freeze is the curve's zero-supercooling anchor
    assert cfg.N_i0 == cfg.N_i0 * np.exp(cfg.cooper_a * max(constants.T_freeze
                                                            - constants.T_freeze, 0.0))
