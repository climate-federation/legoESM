"""Polar-cap radiative cloud floor: applies only inside its gates, is off by default,
and threads from the CLI to the standalone cloud config."""
from __future__ import annotations

import numpy as np
import pytest
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.clouds.cloud_fraction import CloudProperties, apply_cap_cloud_floor
from legoesm.atmosphere.physics.clouds.config import CloudConfig, build_cloud_config


def _props(ncol=3, nlev=4):
    return CloudProperties(cloud_fraction=jnp.full((ncol, nlev), 0.2), lwp=jnp.full((ncol, nlev), 1e-3),
                           iwp=jnp.full((ncol, nlev), 2e-3), r_eff_liq=jnp.full((ncol, nlev), 1e-5),
                           r_eff_ice=jnp.full((ncol, nlev), 3e-5), lwp_lw=jnp.full((ncol, nlev), 1e-3))


def test_off_is_identity():
    props = _props()
    out = apply_cap_cloud_floor(props, jnp.deg2rad(jnp.array([80.0, 80.0, 30.0])),
                                jnp.full((3, 4), 90000.0), jnp.full((3, 4), 2000.0), CloudConfig(scheme="xu_randall"))
    assert out is props


def test_floor_applies_only_poleward_and_below_the_pressure_gate():
    cfg = CloudConfig(scheme="xu_randall", cap_floor_on=True, cap_floor_lat_deg=70.0,
                      cap_floor_p_max_pa=70000.0, cap_floor_cf=0.8, cap_floor_q_c=5e-5)
    lat = jnp.deg2rad(jnp.array([80.0, 60.0, 89.0]))
    p_full = jnp.array([[30000.0, 60000.0, 80000.0, 95000.0]] * 3)
    dp = jnp.full((3, 4), 2000.0)
    out = apply_cap_cloud_floor(_props(), lat, p_full, dp, cfg)
    cf = np.asarray(out.cloud_fraction); lwp = np.asarray(out.lwp)
    assert cf[1] == pytest.approx([0.2] * 4)                # 60N untouched
    assert cf[0] == pytest.approx([0.2, 0.2, 0.8, 0.8])     # 80N: only below 700 hPa
    expected = 0.8 * 5e-5 * 2000.0 / constants.g
    assert lwp[0, 2] == pytest.approx(expected) and lwp[0, 0] == pytest.approx(1e-3)
    assert np.asarray(out.lwp_lw)[2, 3] == pytest.approx(expected)
    assert np.array_equal(np.asarray(out.iwp), np.asarray(_props().iwp))   # ice untouched


def test_floor_never_lowers_an_existing_cloud():
    cfg = CloudConfig(scheme="xu_randall", cap_floor_on=True, cap_floor_cf=0.5, cap_floor_q_c=1e-6)
    props = _props()._replace(cloud_fraction=jnp.full((3, 4), 0.9), lwp=jnp.full((3, 4), 0.5))
    out = apply_cap_cloud_floor(props, jnp.deg2rad(jnp.full(3, 85.0)), jnp.full((3, 4), 90000.0),
                                jnp.full((3, 4), 2000.0), cfg)
    assert np.asarray(out.cloud_fraction) == pytest.approx(0.9) and np.asarray(out.lwp) == pytest.approx(0.5)


def test_build_cloud_config_threads_the_knobs_and_keeps_defaults_when_none():
    base = build_cloud_config("xu_randall")
    assert base.cap_floor_on is False and base.cap_floor_lat_deg == 70.0
    cfg = build_cloud_config("xu_randall", cap_floor_on=True, cap_floor_q_c=1e-4)
    assert cfg.cap_floor_on is True and cfg.cap_floor_q_c == 1e-4 and cfg.cap_floor_cf == 0.8


def test_cli_round_trip_bounds_and_fv_refusal():
    import importlib.util, pathlib, sys
    root = pathlib.Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("run_amip", root / "scripts" / "run" / "run_amip.py")
    ra = importlib.util.module_from_spec(spec); sys.modules["run_amip"] = ra; spec.loader.exec_module(ra)
    parser = ra.build_arg_parser()
    base = ra.build_config_from_args(ra._postprocess_args(parser.parse_args(["--dataset", "analytical"]), parser))
    assert base.cloud_cap_floor_on is False and base.cloud_cap_floor_q_c is None
    cfg = ra.build_config_from_args(ra._postprocess_args(parser.parse_args(
        ["--dataset", "analytical", "--cloud-cap-floor", "--cloud-cap-floor-q-c", "1e-4",
         "--cloud-cap-floor-lat-deg", "70"]), parser))
    assert cfg.cloud_cap_floor_on is True and cfg.cloud_cap_floor_q_c == 1e-4
    bad = ra.build_config_from_args(ra._postprocess_args(parser.parse_args(
        ["--dataset", "analytical", "--cloud-cap-floor-cf", "1.5"]), parser))
    with pytest.raises(Exception, match="cloud_cap_floor_cf"):
        bad.validate_strict()
    from legoesm.driver.model_driver import _standalone_cloud_config
    cc = _standalone_cloud_config(cfg, "xu_randall", allow_convective_cloud=True)
    assert cc.cap_floor_on is True and cc.cap_floor_q_c == 1e-4
