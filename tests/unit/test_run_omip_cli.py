"""CLI coverage for the real OMIP entrypoint."""

from __future__ import annotations

import pytest

from scripts.run.run_omip import build_config_from_args, parse_args


def test_issue484_new_omip_flags_flow_to_config():
    args = parse_args([
        "--grid", "latlon",
        "--max-wallclock-seconds", "3600",
        "--restart-buffer-seconds", "300",
        "--seed", "77",
        "--vertical-mixing-scheme", "tke",
        "--kpp-ri-crit", "0.35",
        "--kpp-k-max", "1.2",
        "--kpp-k-conv", "1.4",
        "--kpp-k-bg", "2e-5",
        "--kpp-a-bg", "2e-4",
    ])
    cfg = build_config_from_args(args)

    assert cfg.max_wallclock_seconds == 3600
    assert cfg.restart_buffer_seconds == 300
    assert cfg.seed == 77
    assert cfg.vertical_mixing.scheme == "tke"
    assert cfg.vertical_mixing.kpp.Ri_crit == 0.35
    assert cfg.vertical_mixing.kpp.K_max == 1.2
    assert cfg.vertical_mixing.kpp.K_conv == 1.4
    assert cfg.vertical_mixing.kpp.K_bg == 2e-5
    assert cfg.vertical_mixing.kpp.A_bg == 2e-4


def test_langmuir_flag_flows_to_config():
    """--langmuir toggles KPPConfig.enable_langmuir (default off)."""
    off = build_config_from_args(parse_args(["--grid", "latlon"]))
    assert off.vertical_mixing.kpp.enable_langmuir is False
    on = build_config_from_args(parse_args([
        "--grid", "latlon", "--langmuir",
        "--langmuir-coeff", "0.12", "--langmuir-number-default", "0.25",
    ]))
    assert on.vertical_mixing.kpp.enable_langmuir is True
    assert on.vertical_mixing.kpp.langmuir_coeff == 0.12
    assert on.vertical_mixing.kpp.langmuir_number_default == 0.25


def test_default_nlev_is_40_for_climate_fidelity():
    """The default ocean vertical resolution is L40 (climate-usable minimum;
    SOTA OMIP models use ~60-75).  Pass --nlev to override."""
    assert parse_args(["--grid", "latlon"]).nlev == 40
    assert parse_args(["--grid", "latlon", "--nlev", "20"]).nlev == 20


def test_enable_latlon_spmd_flags_round_trip():
    """--enable-latlon-spmd / --spmd-n-devices parse and reach OMIPRunConfig
    (the lat-band SPMD restoring lane, part 2a of the ocean-SPMD promotion)."""
    args = parse_args(["--grid", "latlon"])
    assert args.enable_latlon_spmd is False
    assert args.spmd_n_devices == 0
    cfg = build_config_from_args(args)
    assert cfg.enable_latlon_spmd is False
    assert cfg.spmd_n_devices == 0

    args = parse_args(["--grid", "latlon", "--enable-latlon-spmd",
                       "--spmd-n-devices", "4"])
    cfg = build_config_from_args(args)
    assert cfg.enable_latlon_spmd is True
    assert cfg.spmd_n_devices == 4


def test_multicontroller_flags_round_trip():
    """--multicontroller / --coordinator parse and reach OMIPRunConfig
    (the route-B cross-process lane, part 2c of the ocean-SPMD promotion)."""
    args = parse_args(["--grid", "latlon"])
    assert args.multicontroller is False
    assert args.coordinator is None
    cfg = build_config_from_args(args)
    assert cfg.multicontroller is False
    assert cfg.coordinator is None

    args = parse_args([
        "--grid", "latlon", "--enable-latlon-spmd", "--multicontroller",
        "--coordinator", "localhost:12345"])
    cfg = build_config_from_args(args)
    assert cfg.multicontroller is True
    assert cfg.coordinator == "localhost:12345"


def test_multicontroller_without_spmd_refused():
    """--multicontroller alone (no --enable-latlon-spmd) must hard-fail BEFORE
    any device work: otherwise every rank runs the full serial model and
    clobbers the same output paths (codex r2 #2).  The guard sits right after
    build_config_from_args, so this raises without building a model."""
    from scripts.run.run_omip import run_omip_single

    args = parse_args(["--grid", "latlon", "--multicontroller"])
    assert args.enable_latlon_spmd is False
    with pytest.raises(SystemExit, match="requires --enable-latlon-spmd"):
        run_omip_single("latlon", args)


def test_jra55_sea_ice_flag_parses():
    """--jra55-sea-ice opt-in (default off) drives the prognostic slab ice
    wired into the JRA55 scan block loop."""
    assert parse_args(["--grid", "latlon"]).jra55_sea_ice is False
    assert parse_args(["--grid", "latlon", "--jra55-sea-ice"]).jra55_sea_ice is True


def test_surface_stability_scheme_flag_round_trip():
    """--surface-stability-scheme parses, defaults byte-identically to
    dyer1974, and rejects unknown names (dispatch hardening at argparse)."""
    assert (parse_args(["--grid", "latlon"]).surface_stability_scheme
            == "dyer1974")
    assert (parse_args(["--grid", "latlon", "--surface-stability-scheme",
                        "grachev2007_sheba"]).surface_stability_scheme
            == "grachev2007_sheba")
    with pytest.raises(SystemExit):
        parse_args(["--grid", "latlon",
                    "--surface-stability-scheme", "dyer1975"])


def test_precision_default_is_fp64_backward_compatible():
    """OMIP ran unconditional fp64 before the flag; the default MUST stay
    fp64 so existing runs are numerically unchanged."""
    cfg = build_config_from_args(parse_args(["--grid", "latlon"]))
    assert cfg.precision == "fp64"


def test_precision_flag_flows_to_config():
    for mode in ("fp32", "fp64", "mixed"):
        cfg = build_config_from_args(
            parse_args(["--grid", "latlon", "--precision", mode]))
        assert cfg.precision == mode


def test_precision_unknown_mode_rejected():
    """Argparse choices reject an unknown precision (no silent default)."""
    with pytest.raises(SystemExit):
        parse_args(["--grid", "latlon", "--precision", "bf16"])


def test_apply_run_precision_applies_global_policy():
    """run_omip_single() applies precision even when called directly (not via
    main): the unconditional module-import set_policy(fp64) was removed, so a
    direct in-process caller must not silently inherit a stale/default policy
    (codex 2026-06-21). apply_run_precision is the shared idempotent entry."""
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from legoesm.core.precision import (
        PrecisionPolicy, set_policy, resolve_dtype, clear_module_overrides)
    from scripts.run.run_omip import apply_run_precision

    try:
        # Simulate a stale/wrong active policy (what a long-lived process or a
        # prior import could leave behind).
        clear_module_overrides()
        set_policy(PrecisionPolicy.fp32())
        assert resolve_dtype("tracer_advection", "compute") == jnp.dtype("float32")
        apply_run_precision(
            parse_args(["--grid", "latlon", "--precision", "fp64"]))
        assert resolve_dtype("tracer_advection", "compute") == jnp.dtype("float64")
    finally:
        clear_module_overrides()
        set_policy(PrecisionPolicy.fp64())


# --- issue #691: --config / --require-config -------------------------------

def _omip_example_config():
    from pathlib import Path
    return (Path(__file__).resolve().parents[2]
            / "config" / "omip" / "omip_example.yaml")


def test_config_yaml_round_trips_to_args():
    """The committed example config loads and every key reaches args (a key
    that were not a valid dest would raise in load_yaml_config)."""
    args = parse_args(["--config", str(_omip_example_config())])
    assert args.grid == "latlon"
    assert args.nlev == 40
    assert args.dt == 3600.0
    assert args.vertical_mixing_scheme == "kpp"
    assert args.kpp_ri_crit == 0.3
    cfg = build_config_from_args(args)
    assert cfg.vertical_mixing.scheme == "kpp"
    assert cfg.vertical_mixing.kpp.Ri_crit == 0.3


def test_config_yaml_explicit_cli_flag_overrides_file():
    """Precedence: an explicit CLI flag beats the config file value."""
    args = parse_args(["--config", str(_omip_example_config()), "--nlev", "20"])
    assert args.nlev == 20


def test_config_unknown_key_raises():
    """A config key that is not a valid argument dest is a hard error."""
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("not_a_real_dest: 5\n")
        bad = f.name
    with pytest.raises(SystemExit):
        parse_args(["--config", bad])


def test_config_invalid_choice_raises():
    """A scheme literal supplied via --config is choices-validated at load
    (set_defaults bypasses argparse's own choices check) — dispatch-hardening."""
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("vertical_mixing_scheme: garbage\n")
        bad = f.name
    with pytest.raises(SystemExit):
        parse_args(["--config", bad])


def test_require_config_without_config_errors():
    with pytest.raises(SystemExit):
        parse_args(["--require-config", "--grid", "latlon"])


def test_require_config_with_config_ok():
    args = parse_args(["--require-config", "--config", str(_omip_example_config())])
    assert args.config is not None


def test_params_flag_parses():
    assert parse_args(["--grid", "latlon", "--params", "x.yaml"]).params == "x.yaml"


def test_params_routes_kpp_override_into_config():
    """A calibration --params entry routes into the built OMIPRunConfig's nested
    KPPConfig (the whole point of the qualified-name loader, #691)."""
    from legoesm.driver.run_config_yaml import apply_params_to_config
    from legoesm.training.param_collector import build_registry
    m = next(m for m in build_registry() if m.config_class == "KPPConfig")
    lo, hi = m.bounds
    val = (lo + hi) / 2.0
    cfg = build_config_from_args(parse_args(["--grid", "latlon"]))
    out = apply_params_to_config(cfg, {m.qualified_name: val}, driver="run_omip")
    assert getattr(out.vertical_mixing.kpp, m.field) == val


def test_params_routes_treguier_aei0_into_config():
    """#724: the Treguier-1997 adaptive-GM cap (ocean.lat.treguier.aei0) routes
    into the built OMIPRunConfig's nested GMRediConfig.treguier — the config
    _create_setup threads into the realistic-bathymetry lat-lon path."""
    from legoesm.driver.run_config_yaml import apply_params_to_config
    cfg = build_config_from_args(parse_args(["--grid", "latlon"]))
    out = apply_params_to_config(
        cfg, {"ocean.lat.treguier.aei0": 1800.0}, driver="run_omip")
    assert out.gm_redi.treguier.aei0 == 1800.0
    # Sibling GM/Redi + Visbeck tunables ride the same nested config.
    out2 = apply_params_to_config(
        cfg,
        {"ocean.lat.gm_redi.kappa_GM": 900.0,
         "ocean.lat.visbeck.alpha": 0.02},
        driver="run_omip")
    assert out2.gm_redi.kappa_GM == 900.0
    assert out2.gm_redi.visbeck.alpha == 0.02


def test_gm_redi_default_matches_production_bathy_values():
    """Hoisting bathy_gm_redi out of _create_setup must not change the
    production ETOPO-bathymetry defaults (byte-identical run)."""
    cfg = build_config_from_args(parse_args(["--grid", "latlon"]))
    assert cfg.gm_redi.kappa_GM == 800.0
    assert cfg.gm_redi.kappa_Redi == 800.0
    assert cfg.gm_redi.S_max == 0.005
    assert cfg.gm_redi.visbeck.enabled is True
    assert cfg.gm_redi.visbeck.alpha == 0.015
    assert cfg.gm_redi.visbeck.kappa_min == 200.0
    assert cfg.gm_redi.visbeck.kappa_max == 2000.0
    # Treguier stays at its disabled default (mutually exclusive with visbeck).
    assert cfg.gm_redi.treguier.enabled is False


def test_example_params_file_loads_and_applies():
    """The committed config/omip/params_example.yaml is a valid calibration
    file (every key in the registry, in bounds, routable)."""
    from legoesm.driver.run_config_yaml import (
        apply_params_to_config,
        load_params_config,
    )
    p = _omip_example_config().parent / "params_example.yaml"
    cfg = build_config_from_args(parse_args(["--grid", "latlon"]))
    out = apply_params_to_config(cfg, load_params_config(str(p)), driver="run_omip")
    assert out.vertical_mixing.kpp.K_bg == 1.0e-5


def test_iwm_flags_flow_to_config():
    """--iwm* flags round-trip into VerticalMixingConfig.iwm (zdfiwm)."""
    off = build_config_from_args(parse_args(["--grid", "latlon"]))
    assert off.vertical_mixing.iwm.enabled is False
    assert off.vertical_mixing.iwm.mevar is False
    assert off.vertical_mixing.iwm.tsdiff is False

    on = build_config_from_args(parse_args([
        "--grid", "latlon",
        "--iwm", "--iwm-mevar",
        "--iwm-power-bot", "2e-4",
        "--iwm-power-cri", "3e-4",
        "--iwm-power-nsq", "4e-4",
        "--iwm-power-sho", "5e-4",
        "--iwm-scale-bot", "250.0",
        "--iwm-scale-cri", "125.0",
    ]))
    iwm = on.vertical_mixing.iwm
    assert iwm.enabled is True
    assert iwm.mevar is True
    assert iwm.tsdiff is False
    assert iwm.power_bot_wm2 == 2e-4
    assert iwm.power_cri_wm2 == 3e-4
    assert iwm.power_nsq_wm2 == 4e-4
    assert iwm.power_sho_wm2 == 5e-4
    assert iwm.scale_bot_m == 250.0
    assert iwm.scale_cri_m == 125.0


def test_bottom_drag_scheme_flags_parse():
    """--bottom-drag-scheme + NEMO zdfdrg parameter flags parse; the ORACLE
    (ORCA1 namdrg_bot) values are the flag defaults."""
    args = parse_args(["--grid", "latlon"])
    assert args.bottom_drag_scheme == "legacy"
    assert args.bottom_drag_cd0 == 1.0e-3
    assert args.bottom_drag_cdmax == 0.1
    assert args.bottom_drag_z0 == 3.0e-3
    assert args.bottom_drag_ke0 == 2.5e-3

    args = parse_args([
        "--grid", "latlon",
        "--bottom-drag-scheme", "nemo_loglayer",
        "--bottom-drag-cd0", "2e-3",
        "--bottom-drag-ke0", "1e-3",
    ])
    assert args.bottom_drag_scheme == "nemo_loglayer"
    assert args.bottom_drag_cd0 == 2e-3
    assert args.bottom_drag_ke0 == 1e-3

    with pytest.raises(SystemExit):
        parse_args(["--grid", "latlon",
                    "--bottom-drag-scheme", "nemo_typo"])


def test_iwm_override_installs_physics_on_flat_latlon():
    """codex r2 #1: --iwm on the flat-bottom lat-lon path (config.physics is
    None) must NOT silently no-op — the override helper installs a minimal
    physics pipeline carrying the IWM rider, forces the implicit vertical
    solve, and forces the NEMO zdfiwm_init molecular backgrounds."""
    import jax
    jax.config.update("jax_enable_x64", True)
    from legoesm import constants
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.vertical import create_ocean_z_star
    from scripts.run.run_omip import _apply_drag_iwm_overrides

    args = parse_args(["--grid", "latlon", "--iwm"])
    grid = create_latlon_grid(n_lat=6, n_lon=8)
    z = create_ocean_z_star(n_levels=4, H_max=2000.0)
    config = LatLonCGridOceanConfig.from_flat()   # physics=None (flat path)
    model = LatLonCGridOceanModel(grid, z, config)
    config2, model2 = _apply_drag_iwm_overrides(
        args, "latlon", grid, z, config, model)
    assert config2.physics is not None
    assert config2.physics.vertical_mixing.iwm.enabled is True
    assert config2.implicit_vertical_mixing is True
    assert config2.A_v == constants.nu_ocean_molecular
    assert config2.K_v == 1.0e-10
    assert model2._iwm_forcing is None            # uniform fallback mode


def test_barotropic_wide_halo_flags_round_trip():
    """--barotropic-wide-halo(-chunk) reach the nested BarotropicConfig via
    the flat-name mapping and the drag/iwm override hook; default is OFF
    (bit-identical config)."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.vertical import create_ocean_z_star
    from scripts.run.run_omip import _apply_drag_iwm_overrides

    off = parse_args(["--grid", "latlon"])
    assert off.barotropic_wide_halo is False
    assert off.barotropic_wide_halo_chunk == 0

    args = parse_args(["--grid", "latlon", "--barotropic-wide-halo",
                       "--barotropic-wide-halo-chunk", "4"])
    assert args.barotropic_wide_halo is True
    assert args.barotropic_wide_halo_chunk == 4

    grid = create_latlon_grid(n_lat=6, n_lon=8)
    z = create_ocean_z_star(n_levels=4, H_max=2000.0)
    config = LatLonCGridOceanConfig.from_flat()
    model = LatLonCGridOceanModel(grid, z, config)
    config2, _model2 = _apply_drag_iwm_overrides(
        args, "latlon", grid, z, config, model)
    assert config2.barotropic.barotropic_wide_halo is True
    assert config2.barotropic.barotropic_wide_halo_chunk == 4
    # The flag also sets the (validator-required) explicit local clamp.
    assert config2.barotropic.barotropic_local_subcycle_clamp is True
    # No-flag path leaves the config object bit-identical.
    config3, _ = _apply_drag_iwm_overrides(off, "latlon", grid, z,
                                           config, model)
    assert config3.barotropic.barotropic_wide_halo is False


def test_barotropic_wide_halo_refused_off_latlon():
    """Non-latlon grids must refuse the flag loudly, not silently ignore."""
    import pytest as _pytest

    from scripts.run.run_omip import _apply_drag_iwm_overrides

    args = parse_args(["--grid", "cubed_sphere", "--barotropic-wide-halo"])

    class _StubCfg:  # cube ocean config: no flat barotropic fields
        bottom_drag_scheme = "legacy"

    with _pytest.raises(SystemExit, match="wide-halo"):
        _apply_drag_iwm_overrides(
            args, "cubed_sphere", None, None, _StubCfg(), object())
