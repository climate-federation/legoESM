"""E3SM/CAM 'background' gravity-wave source (the frontal spectrum launched in
every column), its composability as the non-orographic member of a
'mcfarlane+e3sm_cam' composite, and the deck/CLI scalars that reach the
kernel through gwd_config_for."""
from __future__ import annotations

import importlib.util
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.gravity_wave_drag.config import (
    E3SMCAMConfig, E3SMFrontalConfig, GravityWaveDragConfig)
from legoesm.atmosphere.physics.gravity_wave_drag.e3sm_cam import e3sm_cam_gwd
from legoesm.atmosphere.physics.gravity_wave_drag.integration import (
    get_gwd_fn, make_gwd_physics)
from legoesm.driver.config import ExperimentConfig
from legoesm.driver.physics_pipeline import gwd_config_for

_HERE = os.path.dirname(os.path.abspath(__file__))
_HYDRO = os.path.normpath(os.path.join(_HERE, os.pardir, "atmosphere", "hydrostatic", "unit"))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, os.pardir, os.pardir, "scripts", "run")))
from run_amip import build_arg_parser, build_config_from_args  # noqa: E402


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HYDRO, filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_e3sm = _load("_gwd_e3sm_cam_helpers", "test_gwd_e3sm_cam.py")
_comp = _load("_gwd_composite_helpers", "test_gwd_composite.py")

FRONTAL = E3SMFrontalConfig(taubgnd=1.5e-3, latitude_taper=False)
BG = E3SMCAMConfig(source="background", pgwv=8, dc=5.0, frontal=FRONTAL)
DT = 1800.0


def test_background_launches_everywhere_frontal_does_not():
    u, v, T, pf, ph, zf, zh, rho, lat = _e3sm._driver_column(ncol=3)
    cfg_fr = BG._replace(source="frontal", frontal=FRONTAL._replace(frontgfc=1e-10))
    zeros = jnp.zeros_like(u)
    out_fr = e3sm_cam_gwd(u, v, T, pf, ph, zf, zh, rho, lat, DT, cfg_fr, frontgf_col=zeros)
    out_bg = e3sm_cam_gwd(u, v, T, pf, ph, zf, zh, rho, lat, DT, BG, frontgf_col=zeros)
    assert float(jnp.max(jnp.abs(out_fr.du_dt))) < 1e-15
    du = np.asarray(out_bg.du_dt)
    assert np.all(np.isfinite(du))
    assert np.all(np.abs(du).max(axis=1) > 0.0)                # every column launches
    assert float(np.sum(np.asarray(u) * du)) < 0.0               # net drag opposes the wind
    out_inf = e3sm_cam_gwd(u, v, T, pf, ph, zf, zh, rho, lat, DT, cfg_fr,
                           frontgf_col=jnp.full_like(u, jnp.inf))
    assert np.array_equal(du, np.asarray(out_inf.du_dt))         # only the threshold differs


def test_background_is_jit_safe():
    u, v, T, pf, ph, zf, zh, rho, lat = _e3sm._driver_column(ncol=3)
    eager = e3sm_cam_gwd(u, v, T, pf, ph, zf, zh, rho, lat, DT, BG)
    jitted = jax.jit(lambda *a: e3sm_cam_gwd(*a, DT, BG))(u, v, T, pf, ph, zf, zh, rho, lat)
    np.testing.assert_allclose(np.asarray(eager.du_dt), np.asarray(jitted.du_dt), rtol=1e-10, atol=1e-18)


def _tend(cfg):
    state, grid, sigma = _comp._make_state()
    fn = make_gwd_physics(cfg, model_type="hydrostatic", dt=300.0)
    tend, _ = fn(state, grid, sigma)
    return tend


def test_composite_with_background_is_member_sum():
    name, fn, _ = get_gwd_fn(GravityWaveDragConfig(scheme="mcfarlane+e3sm_cam", e3sm_cam=BG))
    assert name == "mcfarlane+e3sm_cam" and callable(fn)
    for bad in ("orographic", "frontal", "convective"):
        with pytest.raises(ValueError, match="background"):
            get_gwd_fn(GravityWaveDragConfig(scheme="mcfarlane+e3sm_cam",
                                             e3sm_cam=BG._replace(source=bad)))
    t_c = _tend(GravityWaveDragConfig(scheme="mcfarlane+e3sm_cam", e3sm_cam=BG))
    t_m = _tend(GravityWaveDragConfig(scheme="mcfarlane", e3sm_cam=BG))
    t_e = _tend(GravityWaveDragConfig(scheme="e3sm_cam", e3sm_cam=BG))
    assert float(jnp.max(jnp.abs(t_e.du_dt.data))) > 0.0
    for attr in ("du_dt", "dv_dt", "dT_dt"):
        np.testing.assert_allclose(
            np.asarray(getattr(t_c, attr).data),
            np.asarray(getattr(t_m, attr).data) + np.asarray(getattr(t_e, attr).data),
            rtol=1e-12, atol=1e-18)


def test_validate_strict_gates():
    ExperimentConfig(gravity_wave_drag="mcfarlane+e3sm_cam", e3sm_cam_source="background",
                     e3sm_cam_pgwv=32).validate_strict()
    ExperimentConfig().validate_strict()
    with pytest.raises(ValueError, match="e3sm_cam_source"):
        ExperimentConfig(gravity_wave_drag="mcfarlane+e3sm_cam",
                         e3sm_cam_source="orographic").validate_strict()
    with pytest.raises(ValueError, match="pgwv"):
        ExperimentConfig(e3sm_cam_source="background", e3sm_cam_pgwv=0).validate_strict()
    for bad in (dict(e3sm_cam_source="bogus"), dict(e3sm_cam_effgw=1.5),
                dict(e3sm_cam_taubgnd=0.05), dict(e3sm_cam_latitude_taper="no"),
                dict(e3sm_cam_pgwv=True)):
        with pytest.raises(ValueError, match="e3sm_cam"):
            ExperimentConfig(**bad).validate_strict()


def test_overlay_and_cli_round_trip():
    args = build_arg_parser().parse_args([
        "--gravity-wave-drag", "mcfarlane+e3sm_cam", "--e3sm-cam-source", "background",
        "--e3sm-cam-pgwv", "32", "--e3sm-cam-effgw", "1.0", "--e3sm-cam-taubgnd", "0.0007",
        "--e3sm-cam-c0", "30", "--e3sm-cam-launch-p", "50000", "--no-e3sm-cam-latitude-taper"])
    cfg = build_config_from_args(args)
    assert (cfg.e3sm_cam_source, cfg.e3sm_cam_pgwv, cfg.e3sm_cam_effgw, cfg.e3sm_cam_taubgnd,
            cfg.e3sm_cam_c0, cfg.e3sm_cam_launch_p, cfg.e3sm_cam_latitude_taper) == (
        "background", 32, 1.0, 0.0007, 30.0, 50000.0, False)
    gc = gwd_config_for(cfg)
    assert gc.e3sm_cam.source == "background" and gc.e3sm_cam.pgwv == 32
    assert gc.e3sm_cam.effgw == 1.0 and gc.e3sm_cam.frontal.taubgnd == 0.0007
    assert gc.e3sm_cam.frontal.c0 == 30.0 and gc.e3sm_cam.frontal.launch_p == 50000.0
    assert gc.e3sm_cam.frontal.latitude_taper is False
    assert get_gwd_fn(gc)[0] == "mcfarlane+e3sm_cam"
    empty = build_arg_parser().parse_args([])
    assert all(getattr(empty, f) is None for f in (
        "e3sm_cam_source", "e3sm_cam_pgwv", "e3sm_cam_effgw", "e3sm_cam_taubgnd",
        "e3sm_cam_c0", "e3sm_cam_launch_p", "e3sm_cam_latitude_taper"))
    d = ExperimentConfig()
    built = build_config_from_args(empty)
    assert (built.e3sm_cam_source, built.e3sm_cam_pgwv, built.e3sm_cam_latitude_taper) == (
        d.e3sm_cam_source, d.e3sm_cam_pgwv, d.e3sm_cam_latitude_taper)
    assert gwd_config_for(ExperimentConfig(gravity_wave_drag="mcfarlane+hines")).e3sm_cam == (
        GravityWaveDragConfig().e3sm_cam)                        # production path byte-identical
