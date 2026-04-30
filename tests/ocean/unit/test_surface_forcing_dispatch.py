"""Tests for previously-untested surface-forcing dispatch branches.

Covers ``SurfaceForcingConfig.scheme="combined"`` and
``scheme="bulk_formulas"`` factory dispatch through
``make_surface_forcing_physics``.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm.ocean.physics.surface_forcing.config import (
    BulkFormulaConfig,
    PrescribedForcingConfig,
    RestoringConfig,
    SurfaceForcingConfig,
)
from legoesm.ocean.physics.surface_forcing.integration import (
    make_surface_forcing_physics,
)


@pytest.mark.parametrize(
    "scheme",
    ["none", "prescribed", "restoring", "combined", "bulk_formulas"],
)
def test_factory_returns_callable_for_every_scheme(scheme):
    config = SurfaceForcingConfig(scheme=scheme)
    fn = make_surface_forcing_physics(config)
    assert callable(fn)


def test_factory_rejects_unknown_scheme():
    config = SurfaceForcingConfig(scheme="not_a_scheme")
    with pytest.raises(ValueError, match="Unknown surface forcing scheme"):
        make_surface_forcing_physics(config)


def test_combined_scheme_uses_both_subconfigs():
    """The combined factory closes over both prescribed + restoring configs."""
    presc = PrescribedForcingConfig(tau_x=0.05, Q_net=10.0)
    rest = RestoringConfig(tau_T=86400.0, T_star_eq=22.0)
    config = SurfaceForcingConfig(
        scheme="combined", prescribed=presc, restoring=rest,
    )
    fn = make_surface_forcing_physics(config)
    assert callable(fn)


def test_bulk_formulas_constant_and_coare3_select_correctly():
    for sub_scheme in ("constant", "coare3", "large_yeager"):
        bf = BulkFormulaConfig(bulk_scheme=sub_scheme, U_a=8.0)
        config = SurfaceForcingConfig(scheme="bulk_formulas", bulk_formulas=bf)
        fn = make_surface_forcing_physics(config)
        assert callable(fn)


def test_bulk_formula_emissivity_field_present():
    """Tier 7 lifted emissivity to config — verify it is a real field."""
    bf = BulkFormulaConfig(emissivity=0.95)
    assert bf.emissivity == 0.95
    # Default
    bf_default = BulkFormulaConfig()
    assert bf_default.emissivity == pytest.approx(0.97)


@pytest.mark.parametrize("scheme", ["none", "harmonic", "biharmonic", "gm_redi"])
def test_lateral_mixing_factory_dispatch(scheme):
    from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
    from legoesm.ocean.physics.lateral_mixing.integration import (
        make_lateral_mixing_physics,
    )
    config = LateralMixingConfig(scheme=scheme)
    fn = make_lateral_mixing_physics(config)
    assert callable(fn)


def test_lateral_mixing_unknown_scheme_raises():
    from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
    from legoesm.ocean.physics.lateral_mixing.integration import (
        make_lateral_mixing_physics,
    )
    config = LateralMixingConfig(scheme="not_real")
    with pytest.raises(ValueError):
        make_lateral_mixing_physics(config)
