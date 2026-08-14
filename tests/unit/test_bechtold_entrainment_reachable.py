"""The deep entrainment/detrainment base rates must be settable AND reach the scheme.

Exposed 2026-08-14: `epsilon_deep` sets the ITCZ width and tropical rain
concentration (Pierre Gentine's w7_eps_hi arm: 1.75e-3 -> 3.0e-3 narrowed the
ITCZ half-rain width from 20 deg to the observed 15 deg). It was previously
hard-coded and listed in BechtoldConfig's `__param_spec__` EXCLUDED block, so
it could not be varied from any driver.

These tests fail if the parameter is un-wired at ANY of the four sites: the
spec, the flat ExperimentConfig field, the --params scalar map, or the
pipeline threading.
"""
import pytest

QUALIFIED = "atm.conv.BechtoldConfig.epsilon_deep"
IFS_DEEP_EPSILON = 1.75e-3
IFS_DEEP_DELTA = 0.75e-4


def _bechtold_spec():
    from legoesm.atmosphere.physics.convection.config import __param_spec__ as spec
    for blk in spec.values():
        if blk.get("scheme_key") == "atm.conv.BechtoldConfig":
            return blk
    raise AssertionError("no BechtoldConfig block in __param_spec__")


@pytest.mark.parametrize("field", ["epsilon_deep", "delta_deep"])
def test_spec_lists_them_as_tunable_not_excluded(field):
    blk = _bechtold_spec()
    assert field in blk["params"], f"{field} missing from __param_spec__ params"
    assert field not in blk.get("excluded", {}), (
        f"{field} is BOTH tunable and excluded — the audit reads `excluded` "
        f"and would drop it")
    meta = blk["params"][field]
    lo, hi = meta["bounds"]
    default = IFS_DEEP_EPSILON if field == "epsilon_deep" else IFS_DEEP_DELTA
    assert lo < default < hi, f"{field} bounds {lo,hi} must bracket the IFS default {default}"


def test_params_route_resolves_to_a_flat_field():
    from legoesm.driver.run_config_yaml import build_atm_scalar_param_map
    assert build_atm_scalar_param_map().get(QUALIFIED) == "bechtold_epsilon_deep"


def test_default_is_byte_identical_to_the_previous_hardcoded_value():
    """Exposing a knob must not move the model."""
    import legoesm.atmosphere.physics.convection.config as cc
    import legoesm.driver.physics_pipeline as pp
    from legoesm.driver.config import ExperimentConfig
    seen = {}
    real = cc.BechtoldConfig
    cc.BechtoldConfig = lambda *a, **kw: (seen.update(kw), real(*a, **kw))[1]
    try:
        pp._resolve_convection(ExperimentConfig()._replace(convection="bechtold"))
    finally:
        cc.BechtoldConfig = real
    assert seen["epsilon_deep"] == IFS_DEEP_EPSILON
    assert seen["delta_deep"] == IFS_DEEP_DELTA


def test_set_value_reaches_the_scheme_constructor():
    """The failure this guards: a map entry that resolves but never arrives."""
    import legoesm.atmosphere.physics.convection.config as cc
    import legoesm.driver.physics_pipeline as pp
    from legoesm.driver.config import ExperimentConfig
    seen = {}
    real = cc.BechtoldConfig
    cc.BechtoldConfig = lambda *a, **kw: (seen.update(kw), real(*a, **kw))[1]
    try:
        pp._resolve_convection(ExperimentConfig()._replace(
            convection="bechtold", bechtold_epsilon_deep=3.0e-3,
            bechtold_delta_deep=1.8e-4))
    finally:
        cc.BechtoldConfig = real
    assert seen["epsilon_deep"] == 3.0e-3
    assert seen["delta_deep"] == 1.8e-4


@pytest.mark.parametrize("field,bad", [("bechtold_epsilon_deep", 1.0),
                                       ("bechtold_delta_deep", 1.0)])
def test_out_of_range_is_refused(field, bad):
    from legoesm.driver.config import ExperimentConfig
    cfg = ExperimentConfig()._replace(**{field: bad})
    with pytest.raises(Exception):
        cfg.validate_strict()
