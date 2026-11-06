"""Kessler config canaries and dispatch hardening.

The process-level pins that used to live here checked the pre-port surrogate
against a NumPy copy of its own formulas (surrogate rain evaporation, constant
fall speed, donor clamps); they passed while the scheme produced no surface
rain.  The scheme is now the faithful port of ESCOMP atmos_phys0_31_000
``kessler_run`` and is checked against the RUNNING oracle in
``tests/atmosphere/microphysics/unit/test_kessler_oracle.py``.  What remains
here: the config defaults equal the oracle's hard-coded coefficients, and the
dispatch raises on an unknown scheme.
"""

from __future__ import annotations

import pytest
from legoesm.atmosphere.physics.microphysics.config import (
    KesslerConfig,
    MicrophysicsConfig,
)
from legoesm.atmosphere.physics.microphysics.integration import get_microphysics_fn
from legoesm.atmosphere.physics.microphysics.kessler import _KESSLER_ACCR_EXP


def test_faithful_kessler_config_constants():
    """The config defaults are the oracle's hard-coded KW78 coefficients
    (kessler.F90: ``.001*(qc-.001)``, ``2.2*qr**.875``)."""
    c = KesslerConfig()
    assert c.autoconversion_rate == 1.0e-3        # k1 [1/s]
    assert c.autoconversion_threshold == 1.0e-3   # a [kg/kg]
    assert c.accretion_coeff == 2.2               # k2
    assert _KESSLER_ACCR_EXP == 0.875             # accretion q_r exponent


def test_dispatch_unknown_scheme_raises():
    """Dispatch hardening: an unknown microphysics scheme raises ValueError."""
    with pytest.raises(ValueError, match="[Uu]nknown microphysics scheme"):
        get_microphysics_fn(MicrophysicsConfig(scheme="not_a_scheme"))
