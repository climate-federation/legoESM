"""CLI coverage for the real OMIP entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

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


def test_enable_mpas_spmd_flags_round_trip():
    """--enable-mpas-spmd parses and reaches OMIPRunConfig (the Voronoi SPMD
    ocean lane); it shares --spmd-n-devices / --multicontroller with the
    lat-band lane."""
    args = parse_args(["--grid", "mpas"])
    assert args.enable_mpas_spmd is False
    cfg = build_config_from_args(args)
    assert cfg.enable_mpas_spmd is False

    args = parse_args(["--grid", "mpas", "--enable-mpas-spmd",
                       "--spmd-n-devices", "8"])
    cfg = build_config_from_args(args)
    assert cfg.enable_mpas_spmd is True
    assert cfg.enable_latlon_spmd is False
    assert cfg.spmd_n_devices == 8


def test_mpas_lloyd_flag_round_trip():
    """--mpas-lloyd defaults to the production SCVT (50) and parses."""
    assert parse_args(["--grid", "mpas"]).mpas_lloyd == 50
    assert parse_args(["--grid", "mpas", "--mpas-lloyd", "0"]).mpas_lloyd == 0


def test_no_final_snapshot_flag_round_trip():
    """--no-final-snapshot is off by default; probe arms select it."""
    assert parse_args(["--grid", "mpas"]).no_final_snapshot is False
    assert parse_args(["--grid", "tripole", "--no-final-snapshot"]).no_final_snapshot is True


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


def test_mpas_k_zeta_bih_pin_round_trips():
    """--mpas-k-zeta-bih pins the biharmonic vorticity damping; absent = derived
    from the mesh spacing (None reaches _create_setup, which leaves the model to
    scale it)."""
    assert parse_args(["--grid", "mpas"]).mpas_k_zeta_bih is None
    assert parse_args(["--grid", "mpas", "--mpas-k-zeta-bih", "1e14"]
                      ).mpas_k_zeta_bih == 1.0e14
    assert parse_args(["--grid", "mpas", "--mpas-k-zeta-bih", "0"]
                      ).mpas_k_zeta_bih == 0.0


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


# ===========================================================================
# Double-diffusive mixing (zdfddm) reachability
# ===========================================================================

def test_ddm_flag_flows_to_config():
    """--ddm round-trips into VerticalMixingConfig.ddm.

    Before this flag existed, DDM was UNREACHABLE: it is implemented and
    oracle-pinned (PR #1074), defaults OFF, and its only gate --
    ``DoubleDiffusionConfig.enabled`` -- is a BOOL. Only ``:float`` fields are
    ``__param_spec__``-eligible, so ``--params`` can set ddm's rn_avts/rn_hsbfr
    but can NEVER set the switch that makes them do anything.

    NOTE this asserts the HELPER only. That is not sufficient on its own --
    see test_ddm_reaches_the_model_on_the_flat_latlon_path, which is the load-
    bearing one.
    """
    off = build_config_from_args(parse_args(["--grid", "latlon"]))
    assert off.vertical_mixing.ddm.enabled is False, "default must stay OFF"

    on = build_config_from_args(parse_args(["--grid", "latlon", "--ddm"]))
    assert on.vertical_mixing.ddm.enabled is True


def test_ddm_reaches_the_model_on_the_flat_latlon_path():
    """THE load-bearing pin: --ddm must reach config.physics in PRODUCTION.

    The first version of the --ddm tests asserted only that
    build_vertical_mixing_config_from_args returned ddm.enabled=True -- helper
    wiring, not reachability. The flag was still INERT on the default flat
    lat-lon path, because run_omip ships config.physics=None there and
    _apply_drag_iwm_overrides returned early unless drag or --iwm was set. So
    the tests were green and the scheme was still unselectable: the exact bug
    this branch exists to remove, re-created by its own fix (codex).

    The identical trap is already recorded in that function for --iwm ("codex
    r2 #1"). It was fixed for --iwm only; --ddm was added later and fell in.
    """
    import jax
    jax.config.update("jax_enable_x64", True)
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.vertical import create_ocean_z_star
    from scripts.run.run_omip import _apply_drag_iwm_overrides

    args = parse_args(["--grid", "latlon", "--ddm"])
    grid = create_latlon_grid(n_lat=6, n_lon=8)
    z = create_ocean_z_star(n_levels=4, H_max=2000.0)
    config = LatLonCGridOceanConfig.from_flat()   # physics=None (flat path)
    model = LatLonCGridOceanModel(grid, z, config)
    config2, _ = _apply_drag_iwm_overrides(args, "latlon", grid, z, config, model)

    assert config2.physics is not None, "--ddm never reached config.physics"
    assert config2.physics.vertical_mixing.ddm.enabled is True
    # DDM contributes only through the implicit solve and RAISES on an explicit
    # path, so the flag must force it rather than die on an invisible guard.
    assert config2.implicit_vertical_mixing is True


def test_bare_ddm_does_not_steal_the_zdfiwm_molecular_backgrounds():
    """The molecular-background override is a zdfiwm_init convention (the wave
    field IS the interior background), not a property of additive mixing. A
    bare --ddm must NOT silently strip the user's A_v/K_v."""
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

    args = parse_args(["--grid", "latlon", "--ddm"])
    grid = create_latlon_grid(n_lat=6, n_lon=8)
    z = create_ocean_z_star(n_levels=4, H_max=2000.0)
    # Pin explicit, non-molecular backgrounds so "unchanged" is observable --
    # an earlier version of this test compared against whatever the default
    # happened to be and was tautological (codex).
    config = LatLonCGridOceanConfig.from_flat(A_v=3.0e-4, K_v=7.0e-5)
    model = LatLonCGridOceanModel(grid, z, config)
    config2, _ = _apply_drag_iwm_overrides(args, "latlon", grid, z, config, model)
    assert config2.physics.vertical_mixing.ddm.enabled is True   # ddm DID apply
    assert config2.A_v == 3.0e-4, "bare --ddm stole the user's A_v"
    assert config2.K_v == 7.0e-5, "bare --ddm stole the user's K_v"
    assert config2.A_v != constants.nu_ocean_molecular

    # ... and --iwm DOES take them (the zdfiwm_init convention), so the gating
    # is real rather than vacuous.
    cfg_i = LatLonCGridOceanConfig.from_flat(A_v=3.0e-4, K_v=7.0e-5)
    cfg_i2, _ = _apply_drag_iwm_overrides(
        parse_args(["--grid", "latlon", "--iwm"]), "latlon", grid, z, cfg_i,
        LatLonCGridOceanModel(grid, z, cfg_i))
    assert cfg_i2.A_v == constants.nu_ocean_molecular
    assert cfg_i2.K_v == 1.0e-10


def test_ddm_float_knobs_stay_on_params_not_flags():
    """The float knobs are reachable via --params, per the repo convention that
    spec'd float tunables need no dedicated flag -- so this PR adds ONLY the
    bool gate. Pins that they really are reachable that way."""
    from legoesm.ocean.physics.vertical_mixing.double_diffusion import (
        __param_spec__ as ddm_spec,
    )
    params = ddm_spec["DoubleDiffusionConfig"]["params"]
    assert {"rn_avts", "rn_hsbfr"} <= set(params)
    assert ddm_spec["DoubleDiffusionConfig"]["scheme_key"] == "ocean.vm.ddm"
    # and `enabled` must NOT be spec'd (it is a bool -- not spec-eligible)
    assert "enabled" not in params


@pytest.mark.parametrize("scheme", [None, "catke", "kpp"])
@pytest.mark.parametrize("flag,attr", [("--iwm", "iwm"), ("--ddm", "ddm")])
def test_additive_mixing_flags_reach_every_scheme(scheme, flag, attr):
    """iwm/ddm are both ADDITIVE riders applied AFTER the primary closure (NEMO
    zdfphy ordering), so they are independent of which closure ran and must
    survive every --vertical-mixing-scheme.

    They land in different places: IWM adds onto the tracer AND momentum
    profiles inside compute_vertical_K_profiles, while DDM contributes
    heat/salt-only diffusivities (avm untouched) applied later in the lat-lon
    model's implicit salinity solve (codex corrected an earlier claim here).

    REGRESSION: the catke branch of build_vertical_mixing_config_from_args
    returned early WITHOUT passing iwm=, so `--iwm --vertical-mixing-scheme
    catke` parsed fine and then ran with internal-wave mixing silently OFF.
    Only the KPP-tuning flags are legitimately inapplicable to catke.
    """
    argv = ["--grid", "latlon", flag]
    if scheme is not None:
        argv += ["--vertical-mixing-scheme", scheme]
    vm = build_config_from_args(parse_args(argv)).vertical_mixing
    assert getattr(vm, attr).enabled is True, (
        f"{flag} was silently dropped for --vertical-mixing-scheme {scheme}"
    )


def test_ddm_forces_implicit_mixing_at_parse_time():
    """The force must happen in parse_args, not only in the later override hook.

    The bathymetry path builds its config+model from `args` BEFORE
    _apply_drag_iwm_overrides runs, so a force that lives only in that hook came
    too late: the LatLonCGridOceanModel constructor guard raised, and the user
    got an error they could not connect to the flag they passed (codex).
    """
    assert parse_args(["--ddm"]).implicit_vertical_mixing is True
    assert parse_args(["--iwm"]).implicit_vertical_mixing is True
    # and a run that asked for neither is untouched
    assert parse_args([]).implicit_vertical_mixing is False


@pytest.mark.parametrize("grid", ["cubed_sphere"])
def test_ddm_rejected_loudly_on_unsupported_grids(grid):
    """--ddm was silently INERT on cubed_sphere (setup ships physics=None and
    the non-latlon branch never wired it). A flag that parses and does nothing
    is the failure this branch removes -- reject it instead, mirroring the
    existing --iwm grid guard."""
    from scripts.run.run_omip import _apply_drag_iwm_overrides

    args = parse_args(["--grid", grid, "--ddm"])
    with pytest.raises(SystemExit, match="--ddm is supported on the lat-lon"):
        _apply_drag_iwm_overrides(args, grid, None, None, object(), object())


# ---------------------------------------------------------------------------
# --dz-ref-file: exact external vertical grid (FESOM CORE2 47 levels, NEMO
# e3t_1d).  Without it the level COUNT can be matched but not the PLACEMENT --
# `dz_ref_override` existed in `_create_setup` but was unreachable from the CLI.
# ---------------------------------------------------------------------------

def test_dz_ref_file_defaults_to_none():
    from scripts.run.run_omip import load_dz_ref_file

    assert parse_args(["--grid", "latlon"]).dz_ref_file is None
    assert load_dz_ref_file(None) is None


def test_dz_ref_file_round_trips_text_and_npy(tmp_path):
    import numpy as np

    from scripts.run.run_omip import load_dz_ref_file

    dz = np.array([5.0, 5.0, 10.0, 20.0, 40.0])

    txt = tmp_path / "levels.txt"
    txt.write_text("# FESOM CORE2-style thicknesses [m]\n"
                   + "\n".join(f"{v}" for v in dz) + "\n")
    args = parse_args(["--grid", "latlon", "--dz-ref-file", str(txt)])
    assert args.dz_ref_file == str(txt)
    np.testing.assert_allclose(load_dz_ref_file(str(txt)), dz)

    npy = tmp_path / "levels.npy"
    np.save(npy, dz)
    np.testing.assert_allclose(load_dz_ref_file(str(npy)), dz)


def test_dz_ref_file_builds_the_exact_z_star_coordinate(tmp_path):
    """The loaded thicknesses must become the model's ACTUAL layers, not a
    stretched approximation to them."""
    import numpy as np

    from legoesm.ocean.vertical import create_z_star_from_thicknesses
    from scripts.run.run_omip import load_dz_ref_file

    dz = np.array([5.0, 5.0, 10.0, 10.0, 10.0, 15.0, 20.0, 25.0])
    p = tmp_path / "levels.txt"
    p.write_text("\n".join(f"{v}" for v in dz))
    z_coord = create_z_star_from_thicknesses(load_dz_ref_file(str(p)))
    np.testing.assert_allclose(np.asarray(z_coord.dz_ref), dz, rtol=0, atol=0)
    # Interfaces are the cumulative sum, so the bottom is the total depth.
    assert float(np.abs(np.asarray(z_coord.z_half_ref)[-1])) == pytest.approx(
        float(dz.sum()))


def test_thickness_only_setup_explicitly_selects_legacy_e3w(monkeypatch):
    """The real override path cannot claim a raw mesh operand it does not own."""
    import numpy as np
    import legoesm.ocean.vertical as vertical
    from scripts.run import run_omip

    seen = {}

    class SetupReached(Exception):
        pass

    def capture(dz, t_depth_ref_m=None, **kwargs):
        seen["t_depth_ref_m"] = t_depth_ref_m
        seen.update(kwargs)
        raise SetupReached

    monkeypatch.setattr(vertical, "create_z_star_from_thicknesses", capture)
    with pytest.raises(SetupReached):
        run_omip._create_setup(
            "mpas", "ico1", 3, 60.0, "full", "type1",
            dz_ref_override=np.array([10.0, 20.0, 30.0]))
    assert seen == {"t_depth_ref_m": None, "nemo_e3w_source": "depth_difference"}


def test_viscosity_overrides_reach_the_plain_latlon_lane():
    """--A-h / --K-h used to be read only by the realistic-bathymetry branch,
    so on every other lane (the tripole NEMO-mesh lane included) they parsed
    and did nothing.  A 1/12 degree run needs a viscosity ~30x below the ~5
    degree default to satisfy the explicit stability limit, so an inert flag
    there is a silently unstable run."""
    import scripts.run.run_omip as run_omip

    _, _, cfg, _, _ = run_omip._create_setup(
        "latlon", "8x16", 3, 1000.0, "none", "type1",
        A_h_override=3.0e3, K_h_override=2.0e3)
    assert float(cfg.lateral_viscosity.A_h) == 3.0e3
    assert float(cfg.K_h) == 2.0e3

    _, _, cfg_default, _, _ = run_omip._create_setup(
        "latlon", "8x16", 3, 1000.0, "none", "type1")
    assert float(cfg_default.lateral_viscosity.A_h) == 1.0e5    # default untouched


def test_overrides_reach_the_tripole_recipe(monkeypatch, tmp_path):
    """The tripole lane does NOT build its configuration in the branch the
    other two tests cover: it calls the NEMO-match recipe, which the driver
    used to invoke with its physics argument only.  A probe arm that lowered
    the viscosity by a factor of 33 therefore reproduced the baseline to three
    digits -- the knob reached nothing.  Pin that the overrides now arrive as
    recipe fields, and that an un-overridden run still asks for the recipe
    default."""
    import scripts.run.run_omip as run_omip
    from legoesm.ocean.fidelity import nemo_match_recipe

    seen = {}
    real = nemo_match_recipe.nemo_match_tripole_model_config

    def capture(cfg=None, **kw):
        seen["cfg"] = cfg
        return real(cfg, **kw)

    monkeypatch.setattr(nemo_match_recipe,
                        "nemo_match_tripole_model_config", capture)
    # reuse the synthetic NEMO-like mesh the tripole grid tests write, so this
    # needs no data files
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "grids"))
    from test_tripole_multifile_mesh import _write_tripole_like_mesh
    mesh = tmp_path / "mesh.nc"
    _write_tripole_like_mesh(mesh, 8, 12, dead_north_row=False)

    _, _, cfg, _, _ = run_omip._create_setup(
        "tripole", "eorca1", 3, 1000.0, "none", "type1",
        tripole_mesh=str(mesh), tripole_fold_convention="(n_lon-i)%n_lon",
        A_h_override=3.0e3, pgf_scheme="smc03")
    assert seen["cfg"] is not None
    assert seen["cfg"].A_h == 3.0e3
    assert seen["cfg"].pgf_scheme == "smc03"
    assert cfg.pgf_scheme == "smc03"
    assert float(cfg.lateral_viscosity.A_h) == 3.0e3

    seen.clear()
    run_omip._create_setup("tripole", "eorca1", 3, 1000.0, "none", "type1",
                           tripole_mesh=str(mesh),
                           tripole_fold_convention="(n_lon-i)%n_lon")
    assert seen["cfg"] is None          # recipe default, byte-identical


def test_pgf_scheme_override_reaches_the_plain_latlon_lane():
    """Same inert-flag defect as the viscosity overrides: the pressure-gradient
    scheme selector was read only by the realistic-bathymetry branch.  The
    tripole lane now carries partial cells, where the density-Jacobian scheme
    is the one built for the geometry, so the selector has to reach it."""
    import scripts.run.run_omip as run_omip

    _, _, cfg, _, _ = run_omip._create_setup(
        "latlon", "8x16", 3, 1000.0, "none", "type1", pgf_scheme="smc03")
    assert cfg.pgf_scheme == "smc03"

    _, _, cfg_default, _, _ = run_omip._create_setup(
        "latlon", "8x16", 3, 1000.0, "none", "type1")
    assert cfg_default.pgf_scheme == "adcroft"    # default untouched


@pytest.mark.parametrize("bad,match", [
    ("", "empty"),
    ("5.0\n-1.0\n", "non-positive"),
    ("5.0\nnan\n", "non-finite"),
])
def test_dz_ref_file_rejects_bad_profiles(tmp_path, bad, match):
    """A non-positive or non-finite thickness makes z* non-monotonic and every
    depth-indexed diagnostic downstream quietly wrong -- fail at parse time."""
    from scripts.run.run_omip import load_dz_ref_file

    p = tmp_path / "bad.txt"
    p.write_text(bad)
    with pytest.raises(SystemExit, match=match):
        load_dz_ref_file(str(p))


def test_dz_ref_file_missing_path_raises():
    from scripts.run.run_omip import load_dz_ref_file

    with pytest.raises(SystemExit, match="not found"):
        load_dz_ref_file("/nonexistent/levels.txt")


# ---------------------------------------------------------------------------
# Sea-ice rheology.  The --jra55-sea-ice lane used to hard-code
# SeaIceConfig(), i.e. dynamics="none" / n_categories=1 -- a thermodynamic slab
# with NO rheology, unreachable from the CLI, while FESOM2 runs EVP with 120
# subcycles.  These pin the new flags AND the unchanged default.
# ---------------------------------------------------------------------------

def test_ice_rheology_defaults_reproduce_the_historical_slab():
    args = parse_args(["--grid", "latlon"])
    assert args.ice_dynamics == "none"
    assert args.ice_categories == 1
    for f in ("ice_n_evp", "ice_p_star", "ice_e_yield", "ice_c_strength",
              "ice_delta_min", "ice_alpha_mevp", "ice_beta_mevp"):
        assert getattr(args, f) is None, f


def test_ice_rheology_flags_parse():
    args = parse_args([
        "--grid", "latlon",
        "--ice-dynamics", "evp",
        "--ice-categories", "7",
        "--ice-n-evp", "120",
        "--ice-p-star", "30000",
        "--ice-e-yield", "2.0",
        "--ice-c-strength", "20.0",
        "--ice-delta-min", "1e-11",
        "--ice-alpha-mevp", "250",
        "--ice-beta-mevp", "250",
    ])
    assert args.ice_dynamics == "evp"
    assert args.ice_categories == 7
    assert args.ice_n_evp == 120
    assert args.ice_p_star == 30000.0
    assert args.ice_delta_min == 1e-11
    assert args.ice_alpha_mevp == 250.0


def test_ice_dynamics_rejects_unknown_solver():
    with pytest.raises(SystemExit):
        parse_args(["--grid", "latlon", "--ice-dynamics", "vp"])


# ---------------------------------------------------------------------------
# River-runoff routing (--runoff-routing / --runoff-radius-km).
# Default "none" keeps the historical behaviour, in which runoff landing on a
# dry cell is DISCARDED by the model's masking (~71 % of the JRA55-do total).
# ---------------------------------------------------------------------------

def test_runoff_routing_defaults_to_the_historical_behaviour():
    args = parse_args(["--grid", "latlon"])
    assert args.runoff_routing == "none"
    assert args.runoff_radius_km == 500.0   # FESOM2 runoff_radius


def test_runoff_routing_flags_parse():
    args = parse_args(["--grid", "latlon", "--runoff-routing", "spread",
                       "--runoff-radius-km", "300"])
    assert args.runoff_routing == "spread"
    assert args.runoff_radius_km == 300.0


def test_runoff_routing_rejects_unknown_scheme():
    with pytest.raises(SystemExit):
        parse_args(["--grid", "latlon", "--runoff-routing", "nearset"])


def test_route_runoff_stack_is_identity_without_a_map():
    """`--runoff-routing none` must not touch a single value."""
    import numpy as np

    from scripts.run.run_omip import _route_runoff_stack

    stack = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    out = _route_runoff_stack(stack, None)
    assert out is stack


def test_route_runoff_stack_conserves_over_the_whole_stack():
    """Routing is applied per record; the discharge of EVERY record must
    survive, not just the first."""
    import numpy as np

    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.forcing.runoff_mapper import build_runoff_map
    from scripts.run.run_omip import _route_runoff_stack

    n_lat, n_lon = 6, 12
    g = create_latlon_grid(n_lat, n_lon=n_lon)
    area = np.asarray(g.area)
    lat_c = np.asarray(g.lat2d)
    lon_c = np.asarray(g.lon2d)
    mask = np.ones((n_lat, n_lon), dtype=bool)
    mask[:, 4:7] = False

    stack = np.zeros((3, n_lat, n_lon))
    stack[0, 2, 5] = 1.0e-3
    stack[1, 3, 5] = 2.0e-3
    stack[2, 4, 6] = 5.0e-4

    rmap = build_runoff_map(mask, area, lat_c, lon_c, radius_m=1.2e7)
    out = np.asarray(_route_runoff_stack(stack, rmap))
    assert out.shape == stack.shape
    for k in range(stack.shape[0]):
        assert float(np.sum(out[k] * area)) == pytest.approx(
            float(np.sum(stack[k] * area)), rel=1e-12)
        assert np.all(out[k][~mask] == 0.0)


# --- codex round-4 regressions ---------------------------------------------

def test_dz_ref_file_must_agree_with_H_max(tmp_path):
    """An external vertical grid whose total depth differs from --H-max
    silently rescales every full-depth column: the coordinate comes from the
    file, the bathymetry from --H-max."""
    import numpy as np

    from scripts.run.run_omip import (
        _validate_dz_ref_against_setup, load_dz_ref_file,
    )

    dz = np.full(8, 100.0)                      # 800 m total
    p = tmp_path / "levels.txt"
    p.write_text("\n".join(f"{v}" for v in dz))
    loaded = load_dz_ref_file(str(p))

    _validate_dz_ref_against_setup(loaded, 8, 800.0)          # agrees: fine
    with pytest.raises(SystemExit, match="sums to"):
        _validate_dz_ref_against_setup(loaded, 8, 5500.0)
    with pytest.raises(SystemExit, match="but --nlev"):
        _validate_dz_ref_against_setup(loaded, 47, 800.0)
    _validate_dz_ref_against_setup(None, 40, 5500.0)          # no file: no-op


def test_dz_ref_file_rejects_2d_input(tmp_path):
    """Ravelling a 2-D file would invent a vertical grid from its storage
    order rather than failing."""
    import numpy as np

    from scripts.run.run_omip import load_dz_ref_file

    p = tmp_path / "levels.npy"
    np.save(p, np.full((4, 2), 10.0))
    with pytest.raises(SystemExit, match="1-D"):
        load_dz_ref_file(str(p))


def test_runoff_routing_refused_on_mpas():
    """friver reaches MPAS through a non-conservative IDW interpolation, so
    routing there would conserve an already-wrong discharge."""
    import numpy as np

    from scripts.run.run_omip import _build_runoff_map_for_run

    args = parse_args(["--grid", "mpas", "--runoff-routing", "nearest"])
    with pytest.raises(SystemExit, match="not available on --grid mpas"):
        _build_runoff_map_for_run(args, object(), "mpas",
                                  np.ones((4, 4), dtype=bool))


def test_runoff_map_not_built_when_routing_is_off():
    import numpy as np

    from scripts.run.run_omip import _build_runoff_map_for_run

    args = parse_args(["--grid", "mpas"])
    assert _build_runoff_map_for_run(
        args, object(), "mpas", np.ones((4, 4), dtype=bool)) is None


def test_tripole_mesh_flags_parse():
    """--tripole-mesh (pathsep-joined split mesh) and
    --tripole-strip-north-rows (dead T-pivot halo row) reach args; defaults
    keep the registry mesh + the mesh as stored."""
    import os
    a = parse_args(["--grid", "tripole"])
    assert a.tripole_mesh is None and a.tripole_strip_north_rows == 0
    spec = os.pathsep.join(["h.nc", "z.nc", "m.nc"])
    a = parse_args(["--grid", "tripole", "--tripole-mesh", spec,
                    "--tripole-strip-north-rows", "1"])
    assert a.tripole_mesh == spec and a.tripole_strip_north_rows == 1
    assert a.tripole_fold_convention == "auto"
    assert a.tripole_closed_seas is None
    a = parse_args(["--grid", "tripole", "--tripole-closed-seas", "marmara,black_sea"])
    assert a.tripole_closed_seas == "marmara,black_sea"
    a = parse_args(["--grid", "tripole", "--tripole-mesh", spec,
                    "--tripole-fold-convention", "(n_lon-i)%n_lon"])
    assert a.tripole_fold_convention == "(n_lon-i)%n_lon"
    with pytest.raises(SystemExit):
        parse_args(["--grid", "tripole", "--tripole-fold-convention", "bogus"])


def test_spmd_sea_ice_guard_slab_only():
    """--enable-latlon-spmd + --jra55-sea-ice is allowed ONLY for the slab
    thermodynamic tile (dynamics none, one category); dynamics/ITD run their
    halo pads outside the sharded ocean body and must be refused."""
    from scripts.run.run_omip import _spmd_sea_ice_guard
    _spmd_sea_ice_guard("none", 1)                    # slab: allowed
    for dyn, ncat in (("evp", 1), ("mevp", 1), ("free_drift", 1), ("none", 3)):
        with pytest.raises(SystemExit, match="slab thermodynamic tile"):
            _spmd_sea_ice_guard(dyn, ncat)


def test_spmd_device_count_off_and_single_controller():
    from scripts.run.run_omip import _spmd_device_count
    off = build_config_from_args(parse_args(["--grid", "tripole"]))
    assert _spmd_device_count(off) == 1
    on = build_config_from_args(parse_args([
        "--grid", "tripole", "--enable-latlon-spmd", "--spmd-n-devices", "3"]))
    assert _spmd_device_count(on) == 3


def test_frazil_flag_round_trips_and_defaults_off():
    assert parse_args(["--grid", "mpas"]).frazil is False
    assert parse_args(["--grid", "mpas", "--frazil"]).frazil is True
    assert parse_args(["--grid", "mpas", "--frazil", "--no-frazil"]).frazil is False


def test_tke_card_round_trip_and_rejection():
    """--tke-card fesom2 selects the prognostic FESOM2 constant set; unknown
    cards are refused by argparse AND by the builder (dispatch hardening)."""
    import argparse

    import scripts.run.run_omip as run_omip

    args = run_omip.parse_args(["--vertical-mixing-scheme", "tke", "--tke-card", "fesom2"])
    vm = run_omip.build_vertical_mixing_config_from_args(args)
    assert vm.scheme == "tke"
    assert vm.tke.prognostic is True
    assert vm.tke.surface_flux_coeff == 3.75
    assert vm.tke.prandtl_mode == "richardson"
    assert vm.tke.enable_kappaH_profile is False

    dflt = run_omip.build_vertical_mixing_config_from_args(
        run_omip.parse_args(["--vertical-mixing-scheme", "tke"]))
    assert dflt.tke.prognostic is False
    assert dflt.tke.surface_flux_coeff == 1.0
    assert dflt.tke.prandtl_mode == "unit"

    with pytest.raises(SystemExit):
        run_omip.parse_args(["--tke-card", "bogus"])
    ns = run_omip.parse_args(["--vertical-mixing-scheme", "tke"])
    ns = argparse.Namespace(**{**vars(ns), "tke_card": "bogus"})
    with pytest.raises(ValueError, match="unknown --tke-card"):
        run_omip.build_vertical_mixing_config_from_args(ns)


def test_ic_from_fesom_mesh_round_trip():
    a = parse_args(["--grid", "mpas"])
    assert a.ic_from_fesom_mesh is None
    a = parse_args(["--grid", "tripole", "--woa-init", "--ic-from-fesom-mesh", "/meshes/forca20"])
    assert a.ic_from_fesom_mesh == "/meshes/forca20"
    assert a.woa_init is True
    assert a.ic_cache_dir is None
    a = parse_args(["--grid", "tripole", "--woa-init", "--ic-from-fesom-mesh", "/meshes/forca20",
                    "--ic-cache-dir", "/work/ic_cache"])
    assert a.ic_cache_dir == "/work/ic_cache"


def test_driver_forwards_ic_cache_dir_to_the_fesom_initializer():
    import inspect
    from scripts.run import run_omip
    src = inspect.getsource(run_omip.run_omip_single)
    i = src.index("init_ocean_from_fesom_mesh(")
    assert 'cache_dir=getattr(args, "ic_cache_dir", None)' in src[i:i + 600]


def test_kpp_cfl_cap_follows_the_run_dt():
    """KPPConfig.cfl_cap_dt_s must equal the dynamics dt (default 300 s
    matched only the MPAS default), so the driver sets it from --dt."""
    import inspect
    from scripts.run import run_omip
    cfg = build_config_from_args(parse_args(["--grid", "mpas", "--dt", "600"]))
    assert cfg.vertical_mixing.kpp.cfl_cap_dt_s == 300.0          # config default
    assert run_omip.with_kpp_cfl_dt(cfg, 600.0).vertical_mixing.kpp.cfl_cap_dt_s == 600.0
    src = inspect.getsource(run_omip.run_omip_single)
    i = src.index('dt = args.dt or GRID_DEFAULTS[grid_type]["dt"]')
    assert "run_config = with_kpp_cfl_dt(run_config, dt)" in src[i:i + 200]
    assert src.index("vertical_mixing=run_config.vertical_mixing") > i
