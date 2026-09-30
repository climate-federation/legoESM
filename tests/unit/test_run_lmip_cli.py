"""CLI coverage for the real LMIP entrypoint."""

from __future__ import annotations

import pytest

from scripts.run.run_lmip import _parse_args, build_config_from_args, _get_pft_row


def test_land_surface_scheme_dispatch():
    """--land-surface-scheme selects the right surface_scheme config TYPE."""
    from legoesm.land.surface_scheme import SimpleSEBConfig, TwoLeafCanopyConfig
    from legoesm.land.canopy.config import CLMMLCanopyConfig

    # Default is the TWO-LEAF CANOPY since 2026-08-20 (the simplified scheme is
    # academic-only: its evaporation runs at potential with stomata off and its
    # humidity gradient self-extinguishes with them on).
    default = build_config_from_args(_parse_args(["--lat", "45.0"])).land
    assert isinstance(default.surface_scheme, TwoLeafCanopyConfig)
    # and the simplified scheme still arrives when explicitly selected
    seb = build_config_from_args(
        _parse_args(["--lat", "45.0", "--land-surface-scheme", "simple_seb"])).land
    assert isinstance(seb.surface_scheme, SimpleSEBConfig)

    two = build_config_from_args(
        _parse_args(["--lat", "45.0", "--land-surface-scheme", "two_leaf"])).land
    assert isinstance(two.surface_scheme, TwoLeafCanopyConfig)

    clm = build_config_from_args(
        _parse_args(["--lat", "45.0", "--land-surface-scheme", "clm_ml"])).land
    assert isinstance(clm.surface_scheme, CLMMLCanopyConfig)


def test_clm_ml_subflags_flow_to_config():
    """The CLM-ML sub-flags override the scheme config defaults."""
    from legoesm.land.canopy.config import CLMMLCanopyConfig
    base = CLMMLCanopyConfig()

    cfg = build_config_from_args(_parse_args([
        "--lat", "45.0", "--land-surface-scheme", "clm_ml",
        "--clm-ml-pft", "13",
        "--clm-ml-turbulence-scheme", "most",
        "--clm-ml-dtime-target", "150.0",
    ])).land.surface_scheme
    assert cfg.pft_clm == 13
    assert cfg.turbulence_scheme == "most"
    assert cfg.dtime_ml_target_s == 150.0

    # Sub-flags left off keep the scheme defaults.
    unset = build_config_from_args(
        _parse_args(["--lat", "45.0", "--land-surface-scheme", "clm_ml"])
    ).land.surface_scheme
    assert unset.pft_clm == base.pft_clm
    assert unset.turbulence_scheme == base.turbulence_scheme
    assert unset.dtime_ml_target_s == base.dtime_ml_target_s


def test_clm_ml_subflags_ignored_without_clm_ml():
    """CLM-ML sub-flags on a non-clm_ml scheme don't change the config."""
    from legoesm.land.surface_scheme import SimpleSEBConfig
    cfg = build_config_from_args(_parse_args([
        "--lat", "45.0", "--land-surface-scheme", "simple_seb",
        "--clm-ml-pft", "13"])).land
    assert isinstance(cfg.surface_scheme, SimpleSEBConfig)


def test_pft_params_calibrated_is_default_and_matches_bake():
    """LMIP defaults to the ERA5-calibrated MULTILAYER land params shared with AMIP/CMIP
    (per-PFT + snow albedo); --pft-params raw reproduces the untuned CLM5 table."""
    import legoesm.land.clm_surface_map as C
    args = _parse_args(["--lat", "60.0"])
    assert args.pft_params == "calibrated"
    cal = build_config_from_args(args).land
    # snow feedback baked (cover threshold + fresh-snow brightness)
    assert abs(cal.land_albedo.snow_depth_crit - C.TUNED_SNOW_DCRIT_MULTILAYER) < 1e-9
    assert abs(cal.land_albedo.alpha_snow_max - C.TUNED_SNOW_ALBEDO_MAX_MULTILAYER) < 1e-9
    # per-PFT albedo equals the baked tuple (bare_soil = index 0)
    assert abs(_get_pft_row("bare_soil")["albedo_veg"]
               - C._TUNED_PFT_ALBEDO_MULTILAYER[0]) < 1e-9
    # raw path reproduces the untuned table (!= calibrated) + default LandAlbedoConfig
    raw = build_config_from_args(_parse_args(["--lat", "60.0", "--pft-params", "raw"])).land
    assert raw.land_albedo.snow_depth_crit == 50.0            # LandAlbedoConfig default
    assert _get_pft_row("bare_soil", calibrated=False)["albedo_veg"] == 0.3


def test_freeze_thaw_flag_flows_to_config():
    """--freeze-thaw toggles SoilThermalConfig.enable_freeze_thaw (default off)."""
    cfg_off = build_config_from_args(_parse_args(["--lat", "45.0"]))
    assert cfg_off.land.thermal.enable_freeze_thaw is False
    cfg_on = build_config_from_args(_parse_args(["--lat", "45.0", "--freeze-thaw"]))
    assert cfg_on.land.thermal.enable_freeze_thaw is True


def test_snow_scheme_flag_flows_to_config():
    """--snow-scheme selects the land snowpack (default bulk)."""
    assert build_config_from_args(_parse_args(["--lat", "45.0"])).land.snow_scheme == "bulk"
    cfg = build_config_from_args(_parse_args(["--lat", "45.0", "--snow-scheme", "layered"]))
    assert cfg.land.snow_scheme == "layered"


def test_issue484_new_lmip_flags_flow_to_config():
    args = _parse_args([
        "--lat", "45.5",
        "--max-wallclock-seconds", "5400",
        "--restart-buffer-seconds", "450",
        "--seed", "91",
        "--cd-land", "0.004",
        "--ch-land", "0.005",
        "--z0-land", "0.08",
        "--beta-min", "0.2",
        "--carbon-scheme", "differland",
        "--no-snow-albedo-feedback",
    ])
    cfg = build_config_from_args(args)

    assert cfg.max_wallclock_seconds == 5400
    assert cfg.restart_buffer_seconds == 450
    assert cfg.seed == 91
    assert cfg.land.Cd_land == 0.004
    assert cfg.land.Ch_land == 0.005
    assert cfg.land.z0_land == 0.08
    assert cfg.land.beta_min == 0.2
    assert cfg.land.carbon.scheme == "differland"
    assert cfg.land.snow_albedo_feedback is False


def test_elev_bands_flag_flows_to_config():
    # Default: no elevation bands (legacy cell-mean snowpack)
    assert build_config_from_args(_parse_args(["--lat", "45"])).land.elev_bands is None
    # Enabled with a column sub-grid relief -> 5 equal-area bands
    cfg = build_config_from_args(
        _parse_args(["--lat", "45", "--elev-bands", "--elev-std-m", "800"]))
    assert cfg.land.elev_bands is not None
    assert cfg.land.elev_bands.band_dz.shape == (1, 5)
    # dz scales with the supplied std (top band ~ +1.4 sigma * 800 m)
    assert float(cfg.land.elev_bands.band_dz[0, -1]) > 1000.0


def test_carbon_woody_flag_flows_to_config():
    """--carbon-woody / --no-carbon-woody toggles CarbonConfig.woody (default True)."""
    cfg_default = build_config_from_args(_parse_args(["--lat", "45.0"]))
    assert cfg_default.land.carbon.woody is True
    cfg_herb = build_config_from_args(
        _parse_args(["--lat", "45.0", "--no-carbon-woody"]))
    assert cfg_herb.land.carbon.woody is False


def test_leaf_c_resorption_flag_flows_to_config():
    """--leaf-c-resorption-frac flows to CarbonConfig.leaf_c_resorption_frac;
    default 0.0 (off, byte-identical production)."""
    cfg_default = build_config_from_args(_parse_args(["--lat", "60.0"]))
    assert cfg_default.land.carbon.leaf_c_resorption_frac == 0.0
    cfg = build_config_from_args(
        _parse_args(["--lat", "60.0", "--leaf-c-resorption-frac", "0.3"]))
    assert cfg.land.carbon.leaf_c_resorption_frac == 0.3


def test_cwd_humification_flag_flows_to_config():
    """--cwd-humification-eff flows to CarbonConfig.cwd_humification_eff."""
    cfg = build_config_from_args(
        _parse_args(["--lat", "45.0", "--cwd-humification-eff", "0.15"]))
    assert cfg.land.carbon.cwd_humification_eff == 0.15


def test_arctic_productivity_flags_flow_to_config():
    """The opt-in NSC-gate + cold-deciduous-dormancy flags flow to CarbonConfig;
    default off (byte-identical production)."""
    cfg = build_config_from_args(_parse_args([
        "--lat", "60.0",
        "--nsc-gated-respiration", "--cold-deciduous-dormancy", "--cold-deciduous",
        "--nsc-reserve-days", "12.0", "--r-maint-floor-frac", "0.15",
        "--freeze-dormancy-threshold-k", "271.0"]))
    cc = cfg.land.carbon
    assert cc.nsc_gated_respiration is True
    assert cc.cold_deciduous_dormancy is True
    assert cc.cold_deciduous is True
    assert cc.nsc_reserve_days == 12.0
    assert cc.r_maint_floor_frac == 0.15
    assert cc.freeze_dormancy_threshold_K == 271.0
    d = build_config_from_args(_parse_args(["--lat", "60.0"])).land.carbon
    assert d.nsc_gated_respiration is False
    assert d.cold_deciduous_dormancy is False
    assert d.cold_deciduous is False


def test_carbon_q10_het_flag_flows_to_config():
    """--carbon-q10-het flows to CarbonConfig.Q10_het_exp."""
    cfg = build_config_from_args(
        _parse_args(["--lat", "45.0", "--carbon-q10-het", "0.08"]))
    assert cfg.land.carbon.Q10_het_exp == 0.08


def test_carbon_spinup_flag_parses():
    """--carbon-spinup selects the soil-C spin-up mode; default off; guarded."""
    assert _parse_args(["--lat", "45.0"]).carbon_spinup == "none"
    assert _parse_args(
        ["--lat", "45.0", "--carbon-spinup", "semi_analytic"]
    ).carbon_spinup == "semi_analytic"
    import pytest
    with pytest.raises(SystemExit):
        _parse_args(["--lat", "45.0", "--carbon-spinup", "bogus"])


# --- issue #691: --config / --require-config -------------------------------

def _lmip_example_config():
    from pathlib import Path
    return (Path(__file__).resolve().parents[2]
            / "config" / "lmip" / "lmip_example.yaml")


def test_config_yaml_round_trips_to_args():
    """The committed example config loads; keys reach args (incl. lat from the
    file, which is otherwise required)."""
    args = _parse_args(["--config", str(_lmip_example_config())])
    assert args.lat == 40.0
    assert args.lon == -105.0
    assert args.soil_texture == "loam"
    assert args.veg_type == "c3_grass"
    cfg = build_config_from_args(args)
    assert cfg.land is not None


def test_config_yaml_explicit_cli_flag_overrides_file():
    args = _parse_args([
        "--config", str(_lmip_example_config()), "--lat", "12.5"])
    assert args.lat == 12.5


def test_lat_required_from_cli_or_config():
    """--lat is mandatory but may come from either source; missing both errors."""
    with pytest.raises(SystemExit):
        _parse_args(["--lon", "0.0"])  # no --lat, no --config
    # supplied via CLI is fine
    assert _parse_args(["--lat", "0.0"]).lat == 0.0


def test_require_config_without_config_errors():
    with pytest.raises(SystemExit):
        _parse_args(["--require-config", "--lat", "0.0"])


def test_require_config_with_config_ok():
    args = _parse_args(["--require-config", "--config", str(_lmip_example_config())])
    assert args.config is not None


def test_params_flag_parses():
    assert _parse_args(["--lat", "0.0", "--params", "x.yaml"]).params == "x.yaml"


def test_params_routes_land_override_into_config():
    """A calibration --params entry routes into the built LMIPRunConfig's nested
    land *Config (the qualified-name loader, #691)."""
    from legoesm.driver.run_config_yaml import apply_params_to_config
    from legoesm.training.param_collector import build_registry
    m = next(m for m in build_registry() if m.config_class == "MultiLayerLandConfig")
    lo, hi = m.bounds
    val = (lo + hi) / 2.0
    cfg = build_config_from_args(_parse_args(["--lat", "0.0"]))
    out = apply_params_to_config(cfg, {m.qualified_name: val}, driver="run_lmip")
    assert getattr(out.land, m.field) == val


def test_example_params_file_loads_and_applies():
    """The committed config/lmip/params_example.yaml is a valid calibration
    file (every key in the registry, in bounds, routable)."""
    from legoesm.driver.run_config_yaml import (
        apply_params_to_config,
        load_params_config,
    )
    p = _lmip_example_config().parent / "params_example.yaml"
    cfg = build_config_from_args(_parse_args(["--lat", "0.0"]))
    out = apply_params_to_config(cfg, load_params_config(str(p)), driver="run_lmip")
    assert out.land.Cd_land == 3.0e-3


def test_carbon_ic_flag_round_trips():
    """--carbon-ic reaches args; it defaults to "" (off, byte-identical)."""
    assert _parse_args(["--lat", "0.0"]).carbon_ic == ""
    args = _parse_args(["--lat", "0.0", "--carbon-ic", "/tmp/global_carbon_ic.npz"])
    assert args.carbon_ic == "/tmp/global_carbon_ic.npz"


def test_carbon_ic_config_yaml_round_trips_to_args():
    """The committed config/lmip/lmip_carbon_ic.yaml is a valid LMIP config and
    carries the seed keys (a config file whose keys are not argparse dests would
    raise at load)."""
    p = _lmip_example_config().parent / "lmip_carbon_ic.yaml"
    args = _parse_args(["--config", str(p)])
    assert args.carbon_scheme == "differland"
    assert args.carbon_ic.endswith("global_carbon_ic.npz")
    # A seeded config must still build a usable land config.
    assert build_config_from_args(args).land is not None


def test_restart_snow_layers_round_trip_and_mismatch_refused(tmp_path):
    """A layered restart reloads its four snow-layer fields; a bulk restart into
    a layered run, a layered restart into a bulk run, and a partial field set
    are refused instead of silently reseeding or dropping the pack."""
    import numpy as np
    from scripts.run.run_lmip import _load_restart, _save_restart
    from legoesm.land.multilayer_land import (
        MultiLayerLandConfig, init_multilayer_land_state, seed_snow_layers)
    lay = MultiLayerLandConfig(snow_scheme="layered")
    bulk = MultiLayerLandConfig()
    st = seed_snow_layers(init_multilayer_land_state(1, lay, T_init=265.0)._replace(
        snow_depth=np.array([12.0])), lay)
    p_lay, p_bulk = tmp_path / "lay.npz", tmp_path / "bulk.npz"
    _save_restart(p_lay, 3, 1.0, st)
    _save_restart(p_bulk, 3, 1.0, init_multilayer_land_state(1, bulk, T_init=265.0))
    got = _load_restart(p_lay, lay)[0]
    np.testing.assert_allclose(got.snow_T_layers, st.snow_T_layers)
    with pytest.raises(ValueError, match="snow-layer"):
        _load_restart(p_bulk, lay)
    with pytest.raises(ValueError, match="snow-layer"):
        _load_restart(p_lay, bulk)
    d = dict(np.load(p_lay))
    d.pop("snow_rho_layers")
    np.savez(tmp_path / "part.npz", **d)
    with pytest.raises(ValueError, match="snow-layer"):
        _load_restart(tmp_path / "part.npz", lay)


def test_lmip_runs_the_drainage_limiter_at_the_calibrated_value():
    """The land tables were calibrated with the field-capacity drainage limiter
    at 0.5; LMIP must not inherit the RichardsConfig library default (0.0)."""
    land = build_config_from_args(_parse_args(["--lat", "45.0"])).land
    assert land.richards.fc_drain_saturation == 0.5
