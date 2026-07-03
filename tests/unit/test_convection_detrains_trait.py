"""TOA-drift fix: the `detrains_to_cloud` convection-scheme trait decides
whether convective condensate is true detrainment into the q_c cloud bucket
(plume / mass-flux schemes) or convective PRECIPITATION (adjustment schemes).

Routing an adjustment scheme's (sbm Betts-Miller) column-net drying into q_c let
q_c accumulate ~100x -> opaque clouds, ~0.85 planetary albedo, runaway cold
drift / OLR collapse. The trait gates physics_pipeline so adjustment-scheme
condensate precipitates directly instead.
"""
from __future__ import annotations

import pytest

from legoesm.atmosphere.physics.convection.integration import (
    convection_scheme_traits,
)

# Plume / mass-flux schemes: genuine condensate detrainment into q_c.
_DETRAINING = [
    "mass_flux", "edmf", "zhang_mcfarlane", "kain_fritsch",
    "emanuel", "tiedtke", "bechtold",
]
# Adjustment / moisture-convergence schemes: their drying is convective precip.
_ADJUSTMENT = ["sbm", "dca", "kuo", "none"]


@pytest.mark.parametrize("scheme", _DETRAINING)
def test_massflux_schemes_detrain_to_cloud(scheme):
    assert convection_scheme_traits(scheme).detrains_to_cloud is True


@pytest.mark.parametrize("scheme", _ADJUSTMENT)
def test_adjustment_schemes_do_not_detrain_to_cloud(scheme):
    """sbm / dca / kuo (and the no-op) must NOT feed q_c — their condensate is
    convective precipitation (this is the TOA-drift fix)."""
    assert convection_scheme_traits(scheme).detrains_to_cloud is False


def test_trait_is_static_bool():
    """Static Python bool (scheme is a compile-time constant) so the pipeline's
    branch is a feature gate, not traced control flow."""
    t = convection_scheme_traits("sbm")
    assert isinstance(t.detrains_to_cloud, bool)
    # every other scheme also yields a plain bool
    for s in _DETRAINING + _ADJUSTMENT:
        assert isinstance(convection_scheme_traits(s).detrains_to_cloud, bool)
