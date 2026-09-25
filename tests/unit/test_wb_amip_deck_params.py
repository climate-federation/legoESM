"""Deck-supplied starting values and pinned settings must reach the model.

WHY THIS FILE EXISTS. The WeatherBench classical arm is being trained in the
PRODUCTION AMIP configuration, which differs from library defaults in a dozen
places: a higher critical humidity, a much smaller diagnostic cloud water,
maximum-random cloud overlap, eight sub-columns, prognostic higher-order
turbulence. Two mechanisms carry those across, and each has a way of failing
silently — a starting value that lands on a parameter the selection does not
carry, and a pinned setting that quietly erases every TRAINED value of the
same config. Both are asserted here.
"""

from __future__ import annotations

import pytest

from legoesm.training.aimip_params import (
    aimip_legacy_owned_fields,
    aimip_scheme_keys_for,
)
from legoesm.training.param_collector import build_trainable_params

_SCHEMES = dict(convection="bechtold", turbulence="clubb", gwd="mcfarlane",
                microphysics="morrison", radiation="rrtmgp", cloud="sundqvist")


def _active():
    return aimip_scheme_keys_for(**_SCHEMES)


def _build(init_values, pinned=None):
    """The trainable bundle, built the way scale_build builds it.

    ``pinned`` mirrors the trainer's exclusion of deck-pinned fields: a pinned
    field must not also be a trained leaf, or the pin silently wins over a
    parameter the optimizer is still moving.
    """
    from legoesm.training.scale_build import _pinned_trainable_names

    excluded = set(aimip_legacy_owned_fields(cloud_scheme=_SCHEMES["cloud"]))
    excluded |= _pinned_trainable_names(pinned or {})
    return build_trainable_params(
        active_scheme_keys=_active(), tier="extended",
        init_values=init_values, exclude=tuple(sorted(excluded)))


def test_a_starting_value_is_what_the_run_starts_from():
    """The production values, not the library defaults."""
    want = {"atm.clouds.CloudConfig.rh_crit": 0.85,
            "atm.clouds.CloudConfig.q_c_diagnostic": 5.0e-6}
    got = _build(want).as_dict()
    for k, v in want.items():
        assert float(got[k]) == pytest.approx(v, rel=1e-6)
    # ...and it really moved: the defaults are 0.77 and 1e-3.
    assert float(got["atm.clouds.CloudConfig.rh_crit"]) != pytest.approx(0.77)


def test_a_starting_value_that_reaches_nothing_is_refused():
    """Otherwise the deck asks for a tuned start and the run ignores it."""
    with pytest.raises(ValueError, match="reached no trainable"):
        _build({"atm.conv.TiedtkeConfig.M_b_max": 0.05})   # scheme not active


def test_a_starting_value_at_a_bound_is_refused_not_clipped():
    """A value at the bound seeds a different number on a saturated leaf."""
    with pytest.raises(ValueError, match="clamp of the edge"):
        _build({"atm.clouds.CloudConfig.cloud_fsd": 1.0})


def test_a_pinned_setting_does_not_erase_the_trained_ones():
    """Both address a config by the same key; the merge must be PER FIELD.

    A shallow merge dropped every trained field of any config the deck also
    pinned a value on — the cloud config being exactly that case in the AMIP
    deck (pinned overlap and sub-columns beside a trained critical humidity).
    Asserted on the BUILT configuration, not on the source text.
    """
    import yaml

    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.training.aimip_params import (
        AIMIPTrainableBundle,
        make_aimip_classical_spectral_physics,
    )
    from legoesm.training.model_registry import build_variant

    deck = yaml.safe_load(open("config/wb/campaign/spectral_t63_amip.yaml"))
    cl = deck["classical"]
    grid = create_gaussian_grid(8)
    params = _build(dict(cl["param_init"]), cl["param_fixed"])
    bundle = AIMIPTrainableBundle(
        classical=build_variant(
            "classical", nlev=8,
            overrides={"spatial_surface": False, "spatial_init_std": 0.0,
                       "spatial_seed": 0}),
        schemes=params)
    non_rad, rad = make_aimip_classical_spectral_physics(
        bundle, grid, 1800.0, radiation="rrtmgp", split_rad=True,
        rad_update_interval_steps=int(cl["rad_update_interval_steps"]),
        rrtmgp_gpoint_batch_size=int(cl["rrtmgp_gpoint_batch_size"]),
        param_overrides={k: dict(v) for k, v in cl["param_fixed"].items()},
        orbital_insolation=bool(cl["orbital_insolation"]),
        convective_rain_to_surface=bool(cl["convective_rain_to_surface"]),
        clubb_top_press=(None if cl.get("clubb_top_press_hpa") is None
                         else float(cl["clubb_top_press_hpa"]) * 100.0),
        convection_scheme=cl["convection"], turbulence_scheme=cl["turbulence"],
        gwd_scheme=cl["gwd"], microphysics_scheme=cl["microphysics"],
        cloud_scheme=cl["cloud"], surface_bulk_scheme=cl["surface_bulk"])

    cloud = rad.radiation_config.cloud_config
    # The pinned settings reached the model...
    assert cloud.cloud_vertical_overlap_optics == "max_random"
    assert int(cloud.cloud_n_subcolumns) == 8
    assert float(cloud.cloud_fsd) == pytest.approx(1.0)
    # ...and the TRAINED value on the same config survived the merge.
    assert float(cloud.rh_crit) == pytest.approx(0.85, rel=1e-6)


def test_the_production_settings_that_are_not_registry_parameters_arrive():
    """Orbit and convective-rain routing live on a family config.

    Neither is addressable by a registry key, so both would sit at their
    library defaults — a circular orbit and rain detrained into the resolved
    field — while the deck claimed to match production.
    """
    import yaml

    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.training.aimip_params import (
        AIMIPTrainableBundle,
        make_aimip_classical_spectral_physics,
    )
    from legoesm.training.model_registry import build_variant

    deck = yaml.safe_load(open("config/wb/campaign/spectral_t63_amip.yaml"))
    cl = deck["classical"]
    assert cl["orbital_insolation"] is True
    assert cl["convective_rain_to_surface"] is True

    grid = create_gaussian_grid(8)
    bundle = AIMIPTrainableBundle(
        classical=build_variant(
            "classical", nlev=8,
            overrides={"spatial_surface": False, "spatial_init_std": 0.0,
                       "spatial_seed": 0}),
        schemes=_build(dict(cl["param_init"]), cl["param_fixed"]))
    non_rad, rad = make_aimip_classical_spectral_physics(
        bundle, grid, 1800.0, radiation="rrtmgp", split_rad=True,
        rad_update_interval_steps=6, rrtmgp_gpoint_batch_size=8,
        param_overrides={k: dict(v) for k, v in cl["param_fixed"].items()},
        orbital_insolation=True, convective_rain_to_surface=True,
        clubb_top_press=(None if cl.get("clubb_top_press_hpa") is None
                         else float(cl["clubb_top_press_hpa"]) * 100.0),
        convection_scheme=cl["convection"], turbulence_scheme=cl["turbulence"],
        gwd_scheme=cl["gwd"], microphysics_scheme=cl["microphysics"],
        cloud_scheme=cl["cloud"], surface_bulk_scheme=cl["surface_bulk"])

    assert rad.radiation_config.orbit is not None, "still a circular orbit"
    assert non_rad.physics_config.convection.rain_to_surface is True
    # The background gravity-wave source is present alongside the orographic
    # one, with production's launch amplitude.
    gwd = non_rad.physics_config.gravity_wave_drag
    assert "e3sm_cam" in gwd.scheme and "mcfarlane" in gwd.scheme
    assert float(gwd.e3sm_cam.frontal.taubgnd) == pytest.approx(0.7e-3)


def test_a_pinned_field_is_not_also_a_trained_leaf():
    """A field cannot be both: the pin always wins, so the leaf is dead.

    GLM's finding. A dead leaf costs optimizer state and, worse, appears in the
    trained-parameter report as though it were learned.
    """
    import yaml

    deck = yaml.safe_load(open("config/wb/campaign/spectral_t63_amip.yaml"))
    cl = deck["classical"]
    trained = set(_build(dict(cl["param_init"]), cl["param_fixed"]).raw_values)
    pinned = {f"{key}.{field}"
              for key, fields in cl["param_fixed"].items() for field in fields}
    assert not (trained & pinned), (
        f"pinned AND trainable: {sorted(trained & pinned)} — the pin wins, so "
        f"these leaves would train against nothing")


def test_the_radiation_column_block_size_is_a_deck_key_that_reaches_the_solver():
    """A deck can bound the backward pass's radiation scratch.

    Max-random overlap hands the solver ``n_sub * ncol`` sub-columns, and the
    gradient of the un-blocked solve over them asked for 490 GiB at T63/L32.
    The block size has to be settable from the deck AND arrive on the solver's
    config: it was briefly readable in the builder while still missing from
    the deck validator, so a deck that set it was rejected outright.

    Non-vacuity: drop ``rrtmgp_column_chunk_size`` from ``_WB_CLASSICAL_KEYS``
    and the first half fails; stop forwarding it in
    ``make_aimip_classical_spectral_physics`` and the second half fails.
    """
    import yaml
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.training.aimip_params import (
        AIMIPTrainableBundle,
        make_aimip_classical_spectral_physics,
    )
    from legoesm.training.model_registry import build_variant
    from legoesm.training.scale_build import validate_wb_campaign_yaml

    deck = yaml.safe_load(open("config/wb/campaign/spectral_t63_amip.yaml"))
    deck["classical"]["rrtmgp_column_chunk_size"] = 4608
    validate_wb_campaign_yaml(deck)          # the deck key is accepted...

    cl = deck["classical"]
    bundle = AIMIPTrainableBundle(
        classical=build_variant(
            "classical", nlev=8,
            overrides={"spatial_surface": False, "spatial_init_std": 0.0,
                       "spatial_seed": 0}),
        schemes=_build(dict(cl["param_init"]), cl["param_fixed"]))
    _, rad = make_aimip_classical_spectral_physics(
        bundle, create_gaussian_grid(8), 1800.0, radiation="rrtmgp",
        split_rad=True, rad_update_interval_steps=6,
        rrtmgp_gpoint_batch_size=8, rrtmgp_column_chunk_size=4608,
        param_overrides={k: dict(v) for k, v in cl["param_fixed"].items()},
        convection_scheme=cl["convection"], turbulence_scheme=cl["turbulence"],
        gwd_scheme=cl["gwd"], microphysics_scheme=cl["microphysics"],
        cloud_scheme=cl["cloud"], surface_bulk_scheme=cl["surface_bulk"])

    # ...and it is what radiation will actually block the column axis by.
    assert int(rad.radiation_config.rrtmgp.column_chunk_size) == 4608
