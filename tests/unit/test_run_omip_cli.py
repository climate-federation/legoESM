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


def test_default_nlev_is_40_for_climate_fidelity():
    """The default ocean vertical resolution is L40 (climate-usable minimum;
    SOTA OMIP models use ~60-75).  Pass --nlev to override."""
    assert parse_args(["--grid", "latlon"]).nlev == 40
    assert parse_args(["--grid", "latlon", "--nlev", "20"]).nlev == 20


def test_jra55_sea_ice_flag_parses():
    """--jra55-sea-ice opt-in (default off) drives the prognostic slab ice
    wired into the JRA55 scan block loop."""
    assert parse_args(["--grid", "latlon"]).jra55_sea_ice is False
    assert parse_args(["--grid", "latlon", "--jra55-sea-ice"]).jra55_sea_ice is True


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
