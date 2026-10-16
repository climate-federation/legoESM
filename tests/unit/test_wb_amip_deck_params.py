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


def test_the_cam6_deck_pins_the_condensate_floor_off_and_builds():
    """Owner 2026-09-27: q_c_diagnostic is PINNED at 0.0 (floor off, as in
    the production deck), not trained -- 0.0 is outside its trainable spec
    bounds, so seeding it as a starting value killed the build."""
    import copy
    import pathlib
    from types import SimpleNamespace

    import jax
    import yaml

    from legoesm.training.scale_build import build_mode_components

    repo = pathlib.Path(__file__).resolve().parents[2]
    deck = yaml.safe_load((repo / "config" / "wb" / "campaign"
                           / "spectral_t63_cam6.yaml").read_text())
    cl = deck["classical"]
    assert cl["param_fixed"]["atm.clouds.CloudConfig"]["q_c_diagnostic"] == 0.0
    assert "atm.clouds.CloudConfig.q_c_diagnostic" not in cl.get(
        "param_init", {})
    yml = copy.deepcopy(deck)
    yml["spectral"] = dict(yml["spectral"], n_max=10)   # build, not resolution
    cfg = SimpleNamespace(mode="physics", training_core="spectral",
                          smoke=False, multi_step_hours=(6,), n_days=1)
    *_, params, mk, _loss, _dt = build_mode_components(cfg, yml)
    names = params.schemes.as_dict() if hasattr(params, "schemes") else {}
    assert "atm.clouds.CloudConfig.q_c_diagnostic" not in names
    mk(params)
    assert jax.tree.leaves(params)


def test_the_amip_radiation_lid_and_fp32_are_deck_keys_the_solver_runs():
    """AMIP production runs RRTMGP with the lid layer above the model top
    and in fp32.  The radiation config is not a registry parameter, so the
    WB deck carries both as classical keys: accepted by the validator,
    forwarded onto the built RRTMGPConfig, and CONSUMED by the solver built
    from it (fp32 output; a lid that changes the fluxes).

    Non-vacuity: drop either key from ``_WB_CLASSICAL_KEYS`` and validation
    fails; stop forwarding it in ``make_aimip_classical_spectral_physics``
    and the config assertion fails.
    """
    import jax.numpy as jnp
    import numpy as np
    import yaml
    from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.training.aimip_params import (
        AIMIPTrainableBundle,
        make_aimip_classical_spectral_physics,
    )
    from legoesm.training.model_registry import build_variant
    from legoesm.training.scale_build import validate_wb_campaign_yaml

    deck = yaml.safe_load(
        open("config/wb/campaign/spectral_t63_cam6_tier3.yaml"))
    validate_wb_campaign_yaml(deck)
    cl = deck["classical"]
    assert cl["rrtmgp_overhead_layer"] is True
    assert cl["rrtmgp_compute_fp32"] is True

    def _rrtmgp(on):
        _, rad = make_aimip_classical_spectral_physics(
            AIMIPTrainableBundle(
                classical=build_variant(
                    "classical", nlev=8,
                    overrides={"spatial_surface": False,
                               "spatial_init_std": 0.0, "spatial_seed": 0}),
                schemes=_build({}, cl["param_fixed"])),
            create_gaussian_grid(8), 1800.0, radiation="rrtmgp",
            split_rad=True, rad_update_interval_steps=2,
            rrtmgp_gpoint_batch_size=8,
            rrtmgp_overhead_layer=on, rrtmgp_compute_fp32=on,
            param_overrides={k: dict(v) for k, v in cl["param_fixed"].items()},
            clubb_liquid_partition=True, cld_macmic_num_steps=3,
            convection_scheme=cl["convection"],
            turbulence_scheme=cl["turbulence"], gwd_scheme=cl["gwd"],
            microphysics_scheme=cl["microphysics"],
            cloud_scheme=cl["cloud"], surface_bulk_scheme=cl["surface_bulk"])
        return rad.radiation_config.rrtmgp

    off_cfg, on_cfg = _rrtmgp(False), _rrtmgp(True)
    assert (off_cfg.overhead_layer, off_cfg.compute_fp32) == (False, False)
    assert (on_cfg.overhead_layer, on_cfg.compute_fp32) == (True, True)

    ncol, nlev = 2, 8
    p_half = jnp.broadcast_to(jnp.array(
        [225.5, 503.0, 1.5e3, 5e3, 1.5e4, 3.5e4, 6e4, 8.5e4, 1e5]),
        (ncol, nlev + 1))
    inp = dict(
        T=jnp.broadcast_to(jnp.array(
            [255., 240., 225., 215., 220., 250., 275., 290.]), (ncol, nlev)),
        p_full=0.5 * (p_half[:, 1:] + p_half[:, :-1]), p_half=p_half,
        sfc_temperature=jnp.full((ncol,), 290.0),
        q_v=jnp.full((ncol, nlev), 3e-6), cos_zenith=jnp.full((ncol,), 0.6))
    clear = lambda c: RRTMGP.from_legoesm_config(  # noqa: E731
        c._replace(include_clouds=False)).solve_columns(**inp)
    r_off, r_on = clear(off_cfg), clear(on_cfg)
    h_off = np.asarray(r_off.sw_heating_rate)
    h_on = np.asarray(r_on.sw_heating_rate)
    assert h_off.dtype == np.float64
    assert h_on.dtype == np.float32
    # The lid's ozone shields the top model layer: its SW heating drops.
    assert np.all(h_on[:, 0] < h_off[:, 0])


def test_the_trainer_forwards_the_lid_and_fp32_deck_keys(monkeypatch):
    """The trainer's builder hands the two deck keys to the physics factory.

    Non-vacuity: drop either ``_cl.get`` forward in scale_build and the
    captured kwarg falls back to its False default.
    """
    import copy
    import pathlib
    from types import SimpleNamespace

    import yaml

    import legoesm.training.aimip_params as ap
    from legoesm.training.scale_build import build_mode_components

    seen = {}
    real = ap.make_aimip_classical_spectral_physics

    def spy(*a, **k):
        seen.update(k)
        return real(*a, **k)

    monkeypatch.setattr(ap, "make_aimip_classical_spectral_physics", spy)
    repo = pathlib.Path(__file__).resolve().parents[2]
    yml = copy.deepcopy(yaml.safe_load((
        repo / "config" / "wb" / "campaign"
        / "spectral_t63_cam6_tier3.yaml").read_text()))
    yml["spectral"] = dict(yml["spectral"], n_max=10)
    cfg = SimpleNamespace(mode="physics", training_core="spectral",
                          smoke=False, multi_step_hours=(6,), n_days=1)
    *_, params, mk, _loss, _dt = build_mode_components(cfg, yml)
    mk(params)
    assert seen["rrtmgp_overhead_layer"] is True
    assert seen["rrtmgp_compute_fp32"] is True

    import pytest as _pt
    bad = copy.deepcopy(yml)
    bad["classical"]["rrtmgp_compute_fp32"] = "false"
    with _pt.raises(ValueError, match="must be a YAML boolean"):
        build_mode_components(cfg, bad)


def _tier3():
    import pathlib

    import yaml
    repo = pathlib.Path(__file__).resolve().parents[2]
    return yaml.safe_load((repo / "config" / "wb" / "campaign"
                           / "spectral_t63_cam6_tier3.yaml").read_text())


def test_aerosol_climatology_averages_the_years_on_the_noleap_clock(
        monkeypatch):
    """Owner 2026-10-16: the AMIP aerosol as a 2005-2014 calendar-month
    climatology.  Every year is read at the SAME noleap day (29 Feb -> 28 Feb,
    1 Mar of a leap year -> day 59) and the three fields are averaged."""
    from types import SimpleNamespace

    import jax.numpy as jnp
    import numpy as np

    import legoesm.forcing.external as ext
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.training.scale_build import (
        build_spectral_forcing, wb_aerosol_spec)

    calls = []

    def _aod(c, day, lat_grid=None, band="gray"):
        calls.append((c.start_year, float(day), band))
        return np.full(np.shape(lat_grid), 1e-3 * (c.start_year - 2000)
                       + (0.5 if band == "vis" else 0.0))
    monkeypatch.setattr(ext, "get_aerosol_at_time", _aod)
    monkeypatch.setattr(ext, "get_aerosol_sw_volcanic_at_time",
                        lambda c, day, lat_grid=None: None)
    monkeypatch.setattr(ext, "get_aerosol_lw_at_time",
                        lambda c, day, lat_grid=None: None)

    spec = wb_aerosol_spec(_tier3())
    assert spec[1] == tuple(range(2005, 2015)) and spec[2] is True
    assert spec[0].ccn_path and spec[0].volcanic_path
    assert wb_aerosol_spec({"classical": {}}) is None

    grid, sigma = create_gaussian_grid(8), create_sigma_coordinate(4)
    carry = SimpleNamespace(
        p_s=jnp.full((grid.n_lat, grid.n_lon), 1e5),
        T=jnp.full((grid.n_lat, grid.n_lon, 4), 250.0))
    sl = SimpleNamespace(sst=np.full((4, 8), 290.0), lat=np.linspace(-80, 80, 4),
                         lon=np.linspace(0, 315, 8), sfc_shf=None, land_frac=None)
    for stamp, want in (("2016-03-01T00", 59.0), ("2016-02-29T06", 58.25),
                        ("2015-12-31T18", 364.75)):
        calls.clear()
        f = build_spectral_forcing(sl, grid, np.datetime64(stamp, "ns"), 2016,
                                   aerosol=spec, sigma=sigma, ic_carry=carry)
        assert sorted({y for y, _, _ in calls}) == list(range(2005, 2015))
        assert {d for _, d, _ in calls} == {want}
        col = np.asarray(f["aerosol_od"]).sum(axis=1)
        np.testing.assert_allclose(col, 1e-3 * 9.5, rtol=1e-12)
        np.testing.assert_allclose(np.asarray(f["aerosol_ccn_aod"]),
                                   0.5 + 1e-3 * 9.5, rtol=1e-12)
    plain = build_spectral_forcing(sl, grid, np.datetime64("2016-03-01", "ns"),
                                   2016)
    assert "aerosol_od" not in plain


def test_the_cam6_tier3_deck_runs_aerosol_ccn_like_amip(monkeypatch):
    """The droplet-number-from-aerosol switch reaches the BUILT physics (both
    radiation and microphysics read it), and the deck's aerosol keys are
    accepted.  Non-vacuity: drop ``nc_from_aerosol`` from the deck's
    param_fixed and the gate reads False."""
    import copy
    from types import SimpleNamespace

    import legoesm.atmosphere.physics.combined as combined
    from legoesm.training.scale_build import (
        build_mode_components, validate_wb_campaign_yaml)

    yml = copy.deepcopy(_tier3())
    validate_wb_campaign_yaml(yml)
    seen = []
    real = combined.make_physics

    def spy(cfg, *a, **k):
        seen.append(combined._aerosol_ccn_active(cfg))
        return real(cfg, *a, **k)
    monkeypatch.setattr(combined, "make_physics", spy)
    yml["spectral"] = dict(yml["spectral"], n_max=10)
    cfg = SimpleNamespace(mode="physics", training_core="spectral",
                          smoke=False, multi_step_hours=(6,), n_days=1)
    *_, params, mk, _loss, _dt = build_mode_components(cfg, yml)
    mk(params)
    assert seen and all(seen)
