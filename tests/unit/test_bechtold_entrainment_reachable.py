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


def test_exposing_bechtolds_rate_does_not_expose_tiedtkes():
    """The two schemes have a field of the same name; only one is tunable.

    Tiedtke's deep base rate is still held fixed in-scheme, so it must stay in
    that scheme's `excluded` block, and the driver route must be qualified to
    Bechtold at every hop: the registry key, the flat field name, and the
    constructor that actually receives it.
    """
    import legoesm.atmosphere.physics.convection.config as cc
    from legoesm.driver.run_config_yaml import build_atm_scalar_param_map

    tiedtke = next(b for b in cc.__param_spec__.values()
                   if b.get("scheme_key") == "atm.conv.TiedtkeConfig")
    assert "epsilon_deep" in tiedtke["excluded"]
    assert "epsilon_deep" not in tiedtke["params"]

    routes = build_atm_scalar_param_map()
    for field in ("epsilon_deep", "delta_deep"):
        assert f"atm.conv.TiedtkeConfig.{field}" not in routes
        # The flat field the Bechtold route lands on is Bechtold-prefixed, so
        # a Tiedtke run cannot pick it up by accident.
        assert routes[f"atm.conv.BechtoldConfig.{field}"].startswith("bechtold_")

    # The behavioural half: resolving the TIEDTKE lane with the Bechtold knobs
    # set must leave Tiedtke's own rates at their in-scheme defaults.  Asserted
    # on the RESOLVED config, not on a patched constructor: the pipeline reads
    # a pre-built `ConvectionConfig().tiedtke`, so a constructor spy would
    # never fire and would pass no matter what the knob did.
    import legoesm.driver.physics_pipeline as pp
    from legoesm.driver.config import ExperimentConfig
    _kernel, resolved = pp._resolve_convection(ExperimentConfig()._replace(
        convection="tiedtke", bechtold_epsilon_deep=3.0e-3,
        bechtold_delta_deep=1.8e-4))
    defaults = cc.TiedtkeConfig()
    for field in ("epsilon_deep", "delta_deep"):
        assert getattr(resolved, field) == getattr(defaults, field), (
            f"the Bechtold {field} knob reached the Tiedtke config: "
            f"{getattr(resolved, field)} != {getattr(defaults, field)}")
    # And the values it would have carried are genuinely different, or the
    # assertion above is satisfied by a knob that does nothing anywhere.
    assert defaults.epsilon_deep != 3.0e-3
    assert defaults.delta_deep != 1.8e-4


def test_the_unread_cape_sink_fields_are_gone():
    """`cape_relaxation_sink` and the ratio it scaled had no consumer, so they
    were deleted rather than left as a lever that moves nothing (a
    zero-gradient leaf for an optimiser, an inert switch for a deck).  Scanned
    across the whole convection package so a reintroduction anywhere trips.
    """
    import pathlib

    import legoesm.atmosphere.physics.convection.bechtold as bechtold
    import legoesm.atmosphere.physics.convection.config as cc

    blk = _bechtold_spec()
    for name in ("cape_relaxation_sink", "cape_sink_heating_ratio"):
        assert not hasattr(cc.BechtoldConfig(), name)
        assert name not in blk["params"] and name not in blk["excluded"]
    pkg = pathlib.Path(bechtold.__file__).parent
    for path in sorted(pkg.rglob("*.py")):
        src = path.read_text(errors="replace")
        assert "cape_relaxation_sink" not in src, path.name
        assert "cape_sink_heating_ratio" not in src, path.name


def test_the_unread_polar_cap_flag_is_gone():
    """The PR's own production change: a cap nothing applied, defaulting True.

    A reader who sees the field assumes the cap is on. It is not implemented,
    so it must not be advertised; it comes back with its consumer.
    """
    import legoesm.atmosphere.physics.convection.config as cc
    assert "parcel_theta_cap" not in cc.BechtoldConfig._fields
