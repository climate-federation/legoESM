"""The AIMIP lanes freeze the option-unread knobs by name (#1945 follow-up).

``aimip_inactive_fields`` = the cloud-scheme set plus ``resolved_inactive_fields``
of the config the AIMIP factory builds.  Every lane that builds the
spec-driven bundle (run_aimip trainer, WB scale_build, run_aimip_wb2_eval
skeleton) must route through it, or a Morrison arm trains k_au/k_ac/x_star
that kk2000 never reads (run_aimip's no-inert gate then aborts the run).
Each lane test fails if its call is reverted to the cloud-only set.
"""
from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.training import aimip_params as ap

ROOT = Path(__file__).resolve().parents[2]
_SB = {"atm.micro.MorrisonConfig." + f for f in ("k_au", "k_ac", "x_star")}
_KK = {"atm.micro.MorrisonConfig." + f
       for f in ("autocon_fact", "accre_enhan_fact")}
_HTOPO = "atm.gwd.McFarlaneConfig.h_topo"
_PIN_SB = {"atm.micro.MorrisonConfig": {"warm_rain_scheme": "seifert_beheng"}}


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Option-unread knobs of the default Morrison (mg flavor) and of any cloud
# scheme other than xu_randall / resolved without convective cloud (gate
# 2026-11-06; one rule each, forced-option tests below).
_MOR_DEFAULT = {"atm.micro.MorrisonConfig." + f for f in (
    "lami_max", "rho_snow", "snow_aggregation_eii", "rain_breakup_d0",
    "rain_breakup_steepness", "rain_selfcoll_k", "D_eq", "k_sc", "evap_coeff",
    "a_v_i", "b_v_i", "dep_coeff", "bergeron_rate", "T_center", "T_width",
    "agg_coeff", "hard_sat_adjust_threshold", "hard_sat_max_heating_K")}
_CL = "atm.clouds.CloudConfig."
_CONV = {_CL + f for f in ("conv_cloud_coeff", "conv_cloud_max", "conv_precip_scale")}
_XR = {_CL + f for f in ("alpha_xr", "gamma_xr", "p_xr")}
_RES = {_CL + "q_cloud_resolved_ref"}


def test_helper_follows_scheme_pins_and_orography():
    kw = dict(cloud_scheme="cam6_clubb", microphysics="morrison",
              gwd="mcfarlane")
    cloud = ap._INACTIVE_FIELDS_BY_CLOUD_SCHEME["cam6_clubb"] | _CONV | _XR | _RES
    assert ap.aimip_inactive_fields(**kw, subgrid_orography=False) == (
        cloud | _SB | _MOR_DEFAULT)
    assert ap.aimip_inactive_fields(**kw, subgrid_orography=True) == (
        cloud | _SB | _MOR_DEFAULT | {_HTOPO})
    assert ap.aimip_inactive_fields(**kw, subgrid_orography=False,
                                    param_fixed=_PIN_SB) == cloud | _KK | _MOR_DEFAULT
    assert ap.aimip_inactive_fields(
        cloud_scheme="xu_randall", microphysics="sundqvist", gwd="none",
        subgrid_orography=True) == _CONV | _RES


def _inactive(micro=None, cloud=None, clubb=None, cloud_scheme="cam6_clubb"):
    fixed = {k: v for k, v in (("atm.micro.MorrisonConfig", micro),
                               ("atm.clouds.CloudConfig", cloud),
                               ("atm.turb.CLUBBConfig", clubb)) if v}
    return ap.aimip_inactive_fields(cloud_scheme=cloud_scheme,
                                    microphysics="morrison", gwd="none",
                                    subgrid_orography=False, param_fixed=fixed)


# (pins that select the reading option, fields that option reads); each must
# be inactive at the defaults and active once its option is selected.
_READERS = [
    ({"morrison_flavor": "sam"}, ("lami_max", "rho_snow", "snow_aggregation_eii")),
    ({"morrison_flavor": "sam"}, ("rain_breakup_d0", "rain_breakup_steepness",
                                  "rain_selfcoll_k")),
    ({"rain_selfcoll_scheme": "legacy"}, ("D_eq", "k_sc")),
    ({"rain_evap_scheme": "bulk"}, ("evap_coeff",)),
    ({"fall_speed_scheme": "bulk_qpower"}, ("a_v_i", "b_v_i")),
    ({"ice_deposition_scheme": "heuristic"}, ("dep_coeff",)),
    ({"wbf_scheme": "bergeron_heuristic"}, ("bergeron_rate", "T_center", "T_width")),
    ({"morrison_flavor": "sam", "ice_to_snow_scheme": "heuristic"}, ("agg_coeff",)),
    ({"hard_saturation_adjustment": True}, ("hard_sat_adjust_threshold",
                                            "hard_sat_max_heating_K")),
]


@pytest.mark.parametrize("pins,fields", _READERS)
def test_a_morrison_knob_trains_once_its_option_is_selected(pins, fields):
    names = {"atm.micro.MorrisonConfig." + f for f in fields}
    assert names <= _inactive()
    assert not names & _inactive(micro=pins)


def test_the_mg_resolver_decides_ice_to_snow_and_self_collection():
    # mg overwrites ice_to_snow_scheme, so "heuristic" alone does not revive
    # agg_coeff; an explicit "legacy" self-collection survives the resolver.
    assert "atm.micro.MorrisonConfig.agg_coeff" in _inactive(
        micro={"ice_to_snow_scheme": "heuristic"})
    assert not {"atm.micro.MorrisonConfig.D_eq"} & _inactive(
        micro={"rain_selfcoll_scheme": "legacy"})


def test_cloud_knobs_follow_the_cloud_options():
    assert not _XR & _inactive(cloud_scheme="xu_randall")
    assert not _RES & _inactive(cloud_scheme="resolved")
    assert _CONV <= _inactive(cloud_scheme="xu_randall")
    assert not _CONV & _inactive(cloud_scheme="xu_randall",
                                 cloud={"convective_cloud": True})
    assert _CONV <= _inactive(cloud_scheme="resolved",
                              cloud={"convective_cloud": True})


def test_pdf_variance_scale_is_dead_only_with_prognostic_clubb():
    n = "atm.turb.CLUBBConfig.pdf_variance_scale"
    assert n not in _inactive()
    assert n in _inactive(clubb={"prognostic": True})


def test_the_wb_cam6_deck_excludes_the_option_unread_knobs():
    import yaml

    deck = yaml.safe_load(open(ROOT / "config/wb/campaign/spectral_t63_cam6_tier3_v9.yaml"))
    fixed = deck["classical"]["param_fixed"]
    got = ap.aimip_inactive_fields(cloud_scheme="cam6_clubb", microphysics="morrison",
                                   gwd="mcfarlane", subgrid_orography=True,
                                   param_fixed=fixed)
    assert _MOR_DEFAULT | _CONV | _XR | _RES | {
        "atm.turb.CLUBBConfig.pdf_variance_scale"} <= got


def _run_aimip_bundle(monkeypatch):
    import legoesm.training.neural_gcm_spectral as ngs
    mod = _load("run_aimip_lane_inactive", "scripts/run/run_aimip.py")
    cfg = mod._merge(mod._load_yaml(ROOT / "config" / "aimip" / "aimip_era5.yaml"), {
        "n_max": 8, "aimip_variant": "classical", "aimip_spatial_surface": False,
        "aimip_radiation": "gray", "aimip_rad_update_interval": 1,
        "aimip_microphysics": "morrison", "aimip_trainable_schemes": "extended"})
    spec_cfg = mod._build_spectral_config(cfg)

    def fake_data(spec_cfg_, grid, sigma, cache_dir, **kw):
        carry = types.SimpleNamespace(phis=jnp.zeros((grid.n_lat, grid.n_lon)))
        return [carry], [carry], [0]
    seen = {}
    monkeypatch.setattr(ngs, "load_training_data", fake_data)
    monkeypatch.setattr(ngs, "_train_spectral_loop",
                        lambda params, *a, **k: seen.update(params=params, kw=k))
    monkeypatch.setattr(ap, "make_aimip_classical_spectral_physics",
                        lambda *a, **k: (lambda *a2, **k2: None))
    mod._train_aimip_classical(spec_cfg, "unused", cfg=cfg)
    return seen, cfg, spec_cfg


def test_run_aimip_and_eval_skeleton_freeze_unread_morrison_knobs(monkeypatch):
    seen, cfg, spec_cfg = _run_aimip_bundle(monkeypatch)
    assert seen["kw"].get("assert_no_inert_params") is True
    bundle = seen["params"]
    assert isinstance(bundle, ap.AIMIPTrainableBundle)
    names = set(bundle.schemes.raw_values)
    assert not (names & _SB), sorted(names & _SB)      # kk2000 never reads them
    assert _KK <= names                                  # kk2000 does
    # the eval skeleton must hold the SAME tree, or the checkpoint will not load
    from legoesm.grids.gaussian import create_gaussian_grid
    ev = _load("run_aimip_wb2_eval_lane_inactive",
               "scripts/validate/run_aimip_wb2_eval.py")
    skel = ev._build_skeleton("classical", cfg, spec_cfg, create_gaussian_grid(
        spec_cfg.n_max, dealiasing="quadratic"))
    assert set(skel.schemes.raw_values) == names
    assert (jax.tree.structure(skel.schemes)
            == jax.tree.structure(bundle.schemes))


def _wb_params(monkeypatch, *, param_fixed=None, param_init=None,
               era5_surface_fluxes=False, spatial_surface=False):
    from legoesm.training import scale_build
    monkeypatch.setattr(ap, "make_aimip_classical_spectral_physics",
                        lambda *a, **k: None)
    cfg = types.SimpleNamespace(mode="physics", training_core="spectral",
                                smoke=True, multi_step_hours=(6,), n_days=1)
    yml = {"era5_zarr": "synthetic", "era5_surface_fluxes": era5_surface_fluxes,
           "train_years": [2019], "radiation": "gray",
           "n_lat": 8, "n_lon": 16, "nlev": 4,
           "spectral": {"n_max": 5, "dt": 60.0}, "loss": {},
           "classical": {"microphysics": "morrison",
                         "trainable_schemes": "extended",
                         "param_fixed": param_fixed or {},
                         "param_init": param_init or {},
                         "spatial_surface": spatial_surface}}
    return scale_build.build_mode_components(cfg, yml)[3]


def test_scale_build_freezes_by_resolved_warm_rain_law(monkeypatch):
    def names(param_fixed):
        return set(_wb_params(monkeypatch,
                              param_fixed=param_fixed).schemes.raw_values)

    kk = names({})
    assert not (kk & _SB) and _KK <= kk
    sb = names(_PIN_SB)                    # the deck pins the SB law
    assert not (sb & _KK) and _SB <= sb


def _array_knobs(classical):
    return {k for k, v in classical.raw_values.items()
            if isinstance(v, jax.Array)}


def test_wb_classical_tree_holds_only_the_selected_schemes_knobs(monkeypatch):
    # Owner 2026-10-15: a fixed parameter has NO gradient, not a zero one.
    # The WB legacy tree used to hold every family's knob as an array and
    # freeze the dead ones only after differentiating them in the probe.
    cl = _wb_params(monkeypatch).classical
    want = ap.aimip_classical_trainable_names(
        convection="tiedtke", turbulence="louis", gwd="mcfarlane",
        microphysics="morrison", cloud="xu_randall", radiation="gray",
        surface_bulk_scheme="constant")
    assert _array_knobs(cl) == want
    assert "sbm_tau_c" not in want and "tiedtke_epsilon_deep" in want
    assert type(cl.raw_values["sbm_tau_c"]) is float   # same leaf, no grad
    arr, _ = eqx.partition(cl, eqx.is_inexact_array)
    assert arr.raw_values["sbm_tau_c"] is None


def test_wb_era5_fluxes_take_surface_knobs_out_of_the_tree(monkeypatch):
    live = _array_knobs(_wb_params(monkeypatch).classical)
    era5 = _array_knobs(
        _wb_params(monkeypatch, era5_surface_fluxes=True).classical)
    surf = {"surface_Cd_neutral", "surface_Ch_neutral"}
    assert surf <= live and not (surf & era5)
    assert era5 == live - surf


def test_explicit_request_for_an_excluded_parameter_raises():
    from legoesm.training.param_collector import build_trainable_params
    k_au = "atm.micro.MorrisonConfig.k_au"
    with pytest.raises(ValueError, match="both requested"):
        build_trainable_params(active_scheme_keys={"atm.micro.MorrisonConfig"},
                               include=(k_au,), exclude=(k_au,))


def test_wb_param_init_of_an_inactive_parameter_raises(monkeypatch):
    # The WB deck's only per-name request channel: seeding k_au, which the
    # kk2000 law never reads, must fail at setup, not train a dead leaf.
    with pytest.raises(ValueError, match="k_au"):
        _wb_params(monkeypatch,
                   param_init={"atm.micro.MorrisonConfig.k_au": 1.0})


def test_wb_checkpoint_of_frozen_knobs_loads_into_an_all_array_template(
        monkeypatch, tmp_path):
    # The WB trainer serialises checkpoint_view(params): float knobs go to
    # disk as arrays of the live dtype, or an all-array template (registry
    # default, eval skeleton) fails equinox's dtype check.
    from legoesm.ml.training import checkpoint_view
    from legoesm.training.model_registry import build_variant
    cl = _wb_params(monkeypatch).classical
    path = tmp_path / "p.eqx"
    eqx.tree_serialise_leaves(path, checkpoint_view(cl))
    full = eqx.tree_deserialise_leaves(path, build_variant("classical", nlev=4))
    back = eqx.tree_deserialise_leaves(path, cl)
    assert float(full.raw_values["sbm_tau_c"]) == cl.raw_values["sbm_tau_c"]
    assert type(back.raw_values["sbm_tau_c"]) is float


# --- Decision 5 (2026-10-15): unread SPATIAL surface fields -----------------
# A spatial field the resolved surface path never reads is not built at all
# (it used to be an array leaf whose gradient stop_gradient cut to exactly 0).

_ARM = dict(convection="tiedtke", gwd="mcfarlane", microphysics="sundqvist",
            cloud="xu_randall", radiation="rrtmgp")


def test_run_aimip_main_deck_drops_unread_spatial_fields_and_eval_loads(
        monkeypatch, tmp_path):
    import legoesm.training.aimip_spatial as sp
    import legoesm.training.neural_gcm_spectral as ngs
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.ml.training import load_checkpoint, save_checkpoint
    mod = _load("run_aimip_spatial_unread", "scripts/run/run_aimip.py")
    cfg = mod._merge(mod._load_yaml(
        ROOT / "config" / "aimip" / "aimip_era5.yaml"), {
        "n_max": 8, "aimip_variant": "classical", "aimip_radiation": "gray",
        "aimip_rad_update_interval": 1})
    assert cfg["aimip_spatial_surface"] is True     # the deck's own setting
    spec_cfg = mod._build_spectral_config(cfg)

    def fake_data(spec_cfg_, grid, sigma, cache_dir, **kw):
        carry = types.SimpleNamespace(phis=jnp.zeros((grid.n_lat, grid.n_lon)))
        return [carry], [carry], [0]
    seen = {}
    monkeypatch.setattr(sp, "era5_land_fraction",
                        lambda grid: jnp.full((grid.n_lat, grid.n_lon), 0.5))
    monkeypatch.setattr(ngs, "load_training_data", fake_data)
    monkeypatch.setattr(ngs, "_train_spectral_loop",
                        lambda params, *a, **k: seen.update(params=params))
    monkeypatch.setattr(ap, "make_aimip_classical_spectral_physics",
                        lambda *a, **k: (lambda *a2, **k2: None))
    mod._train_aimip_classical(spec_cfg, "unused", cfg=cfg)
    params = seen["params"]
    # louis + constant bulk law: z0 is unread, the other four are read
    assert set(params.spatial_surface.fields) == (
        set(sp.SPATIAL_FIELD_NAMES) - {"z0"})
    diffd = [jax.tree_util.keystr(k) for k, _ in jax.tree_util.
             tree_leaves_with_path(eqx.filter(params, eqx.is_inexact_array))]
    assert not [n for n in diffd if "spatial_surface" in n and "z0" in n]
    assert any("fields['Cd_neutral']" in n for n in diffd)
    # the eval skeleton builds the same tree, so the new checkpoint loads
    ev = _load("run_aimip_wb2_eval_spatial_unread",
               "scripts/validate/run_aimip_wb2_eval.py")
    skel = ev._build_skeleton("classical", cfg, spec_cfg, create_gaussian_grid(
        spec_cfg.n_max, dealiasing="quadratic"))
    save_checkpoint(params, tmp_path / "m.eqx")
    back = load_checkpoint(skel, tmp_path / "m.eqx")
    assert set(back.spatial_surface.fields) == set(params.spatial_surface.fields)


def test_wb_spatial_surface_request_raises_at_setup(monkeypatch):
    # The WB classical physics gets no land mask, so every spatial field would
    # be a fixed parameter: asking for them is a setup-time error.
    with pytest.raises(ValueError, match="spatial_surface: true"):
        _wb_params(monkeypatch, spatial_surface=True)


def test_resume_from_a_checkpoint_holding_an_unread_field_fails_loudly(
        tmp_path):
    from legoesm.ml.checkpoint_io import CheckpointStructureError
    from legoesm.ml.training import save_checkpoint
    from legoesm.training.neural_gcm_spectral import maybe_resume_model
    names = ap.aimip_classical_trainable_names(
        **_ARM, turbulence="louis", surface_bulk_scheme="constant")
    # the pre-2026-10-15 on-disk layout: all five fields as arrays
    save_checkpoint(ap.AIMIPClassicalParams.from_defaults(
        spatial_surface=True, spatial_init_std=0.005), tmp_path / "epoch_0000.eqx")
    new = ap.AIMIPClassicalParams.from_defaults(
        spatial_surface=True, spatial_init_std=0.005, trainable=names)
    with pytest.raises(CheckpointStructureError, match="2026-10-15"):
        maybe_resume_model(new, tmp_path)


def test_unread_helper_matches_the_trainable_rule_and_count_names_the_set():
    by_count = {}
    for turb in sorted(ap._SURFACE_CARRYING_TURBULENCE | {"none"}):
        for bulk in sorted(ap._SURFACE_KNOBS_BY_BULK):
            names = ap.aimip_classical_trainable_names(
                **_ARM, turbulence=turb, surface_bulk_scheme=bulk)
            got = ap.AIMIPClassicalParams.from_defaults(
                spatial_surface=True, trainable=names).spatial_surface.frozen
            assert got == ap.aimip_spatial_unread_fields(
                turbulence=turb, surface_bulk_scheme=bulk), (turb, bulk)
            by_count.setdefault(len(got), set()).add(got)
    # positional checkpoints: two different dropped sets of equal size would
    # load into each other silently; within this table the size names the set
    assert all(len(v) == 1 for v in by_count.values()), by_count
    names = ap.aimip_classical_trainable_names(
        **_ARM, turbulence="louis", surface_bulk_scheme="constant")
    with pytest.raises(ValueError, match="disagrees"):
        ap.AIMIPClassicalParams.from_defaults(
            spatial_surface=True, trainable=names, spatial_unread=())


@pytest.mark.parametrize("bulk", ["constant", "most", "coare3", "large_yeager",
                                  "large_yeager_cesm", "nemo_si3_constant"])
def test_unread_spatial_fields_do_not_change_the_surface_fluxes(bulk):
    """An unread field may fall back to its scalar: any value gives bitwise
    the same surface fluxes, the only reader of Cd/Ch/z0."""
    from legoesm.atmosphere.physics.turbulence.surface_layer import (
        compute_surface_fluxes,
    )
    unread = ap.aimip_spatial_unread_fields(turbulence="louis",
                                            surface_bulk_scheme=bulk)
    cfg = ap.AIMIPClassicalParams.from_defaults().to_surface_config(
        bulk_scheme=bulk)
    u, v = jnp.asarray([5.0, 8.0, 2.0, 10.0]), jnp.asarray([1.0, -2.0, 0.5, 0.0])
    T = jnp.asarray([290.0, 300.0, 285.0, 296.0])
    T_sfc = jnp.asarray([295.0, 295.0, 290.0, 296.2])
    q = jnp.asarray([0.010, 0.012, 0.006, 0.014])
    q_sfc = jnp.asarray([0.016, 0.017, 0.012, 0.018])
    rho = jnp.full((4,), 1.2)
    other = cfg._replace(**{f: getattr(cfg, f) * jnp.asarray([3.0, 0.2, 7.0, 0.5])
                            for f in unread})
    a = compute_surface_fluxes(u, v, T, q, T_sfc, q_sfc, rho, cfg)
    b = compute_surface_fluxes(u, v, T, q, T_sfc, q_sfc, rho, other)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
