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
