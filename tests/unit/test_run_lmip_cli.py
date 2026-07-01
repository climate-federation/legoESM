"""CLI coverage for the real LMIP entrypoint."""

from __future__ import annotations

from scripts.run.run_lmip import _parse_args, build_config_from_args, _get_pft_row


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
