"""The FV3-duo and simple tiled-cube Kessler lanes run the RUN'S resolved
Kessler config (hard saturation-adjustment flags), not ``KesslerConfig()``
(issue #1969)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from legoesm.atmosphere.forcing.idealized import kessler_forcing
from legoesm.atmosphere.physics.microphysics.config import KesslerConfig
from legoesm.driver.config import ExperimentConfig
from legoesm.driver.model_driver import ModelDriver

_HARD = dict(hard_saturation_adjustment=True, hard_sat_adjust_threshold=1.3,
             hard_sat_max_heating_K=7.0)


def _cfg(**over):
    return ExperimentConfig()._replace(
        microphysics="kessler", radiation="none", convection="none",
        turbulence="none", gravity_wave_drag="none", cloud_scheme="none",
        **over)


def _expected(cfg):
    if not cfg.hard_saturation_adjustment:
        return KesslerConfig()
    return KesslerConfig()._replace(**_HARD)


@pytest.mark.parametrize("over", [{}, _HARD])
def test_tiled_cube_lane_passes_run_kessler_config(monkeypatch, over):
    seen = {}

    def _spy(sigma, dt, config=None):
        seen["config"] = config
        return "fn"

    monkeypatch.setattr(kessler_forcing, "make_kessler_column_physics_fn", _spy)
    cfg = _cfg(**over)
    stub = SimpleNamespace(config=cfg, sigma=None)
    assert ModelDriver._tiled_cube_column_physics_fn(stub) == "fn"
    assert seen["config"] == _expected(cfg)


@pytest.mark.parametrize("over", [{}, _HARD])
def test_fv3_duo_lane_passes_run_kessler_config(monkeypatch, over):
    import jax.numpy as jnp
    seen = {}

    def _spy(state, press, q, *, config=None, **kw):
        seen["config"] = config
        return state, press, q

    monkeypatch.setattr(kessler_forcing, "apply_kessler_step_sixface_jax", _spy)
    cfg = _cfg(**over)
    model = SimpleNamespace(grid=SimpleNamespace(n=2, ng=1),
                            config=SimpleNamespace(km=2), _ptop=1.0,
                            step_out_shardings=None, window_layout=None)
    stub = SimpleNamespace(config=cfg, model=model)
    z = jnp.zeros((6, 4, 4, 2))
    bundle = {"state": {"pt": z, "delp": z}, "press": {"pe": z}, "q": [z]}
    ModelDriver._fv3_duo_apply_kessler(stub, bundle, 60.0)
    assert seen["config"] == _expected(cfg)


def _call_lane(lane, cfg):
    if lane == "tiled":
        return ModelDriver._tiled_cube_column_physics_fn(
            SimpleNamespace(config=cfg, sigma=None))
    import jax.numpy as jnp
    model = SimpleNamespace(grid=SimpleNamespace(n=2, ng=1),
                            config=SimpleNamespace(km=2), _ptop=1.0,
                            step_out_shardings=None, window_layout=None)
    z = jnp.zeros((6, 4, 4, 2))
    bundle = {"state": {"pt": z, "delp": z}, "press": {"pe": z}, "q": [z]}
    return ModelDriver._fv3_duo_apply_kessler(
        SimpleNamespace(config=cfg, model=model), bundle, 60.0)


@pytest.mark.parametrize("lane", ["tiled", "fv3_duo"])
@pytest.mark.parametrize("flag,match", [
    ("subgrid_autoconversion", "subgrid_autoconversion"),
    ("nc_from_aerosol", "no aerosol droplet-number mode")])
def test_unsupported_flag_on_kessler_lane_raises(lane, flag, match):
    with pytest.raises(ValueError, match=match):
        _call_lane(lane, _cfg(**{flag: True}))
