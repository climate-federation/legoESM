"""DycoreConfig / run_amip switch for the lat-lon vertical temperature transport.

``latlon_vert_advection_scheme`` -> lat-lon C-grid PE ``vert_advection_scheme``.
Default "auto" = centred Simmons-Burridge on hybrid levels, upwind on sigma
(user decision 2026-11-26); "upwind" is the opt-out reproducing earlier runs.
"""
from __future__ import annotations

import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

_NLEV = 8
_F = "latlon_vert_advection_scheme"


@pytest.fixture(autouse=True)
def _fp64_policy():
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(prev)


def _cfg(grid_type="latlon", resolution=24, vertical_coord="hybrid", **dycore):
    from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig
    disc = "spectral" if grid_type == "gaussian" else "latlon_cgrid"
    return ExperimentConfig(
        grid=GridConfig(grid_type=grid_type, resolution=resolution, nlev=_NLEV,
                        vertical_coord=vertical_coord),
        dycore=DycoreConfig(model_type="hydrostatic", discretization=disc,
                            energy_fixer=False, dt=300.0, **dycore))


def _build(cfg):
    from legoesm.driver.component_factory import create_atmosphere_dycore
    from legoesm.grids.factory import create_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    grid = create_grid(cfg.grid.grid_type, cfg.grid.resolution)
    return create_atmosphere_dycore(cfg, grid, create_sigma_coordinate(_NLEV))


def _errors(cfg):
    try:
        cfg.validate_strict()
    except ValueError as exc:
        return str(exc)
    return ""


def test_default_is_auto_not_upwind():
    """Fails if the production default reverts to the upwind T transport."""
    from legoesm.driver.config import DycoreConfig
    assert getattr(DycoreConfig(), _F) == "auto"


@pytest.mark.parametrize("scheme", ["upwind", "auto"])
def test_threaded_into_the_latlon_dycore(scheme):
    on, default = _build(_cfg(**{_F: scheme})), _build(_cfg())
    assert on.config.vert_advection_scheme == scheme
    assert default.config.vert_advection_scheme == "auto"
    assert on.config._replace(vert_advection_scheme="auto") == default.config


@pytest.mark.parametrize("kw, match", [
    (dict(grid_type="gaussian", resolution=21, **{_F: "upwind"}), "lat-lon C-grid dycore only"),
    (dict(vertical_coord="sigma", **{_F: "sb"}), "acts only"),
    (dict(vertical_form="as83", **{_F: "sb"}), "acts only"),
    (dict(**{_F: "van_leer"}), "must be one of"),
])
def test_validate_strict_refuses(kw, match):
    assert match in _errors(_cfg(**kw))


@pytest.mark.parametrize("kw", [
    dict(**{_F: "sb"}), dict(**{_F: "upwind"}), dict(vertical_coord="sigma", **{_F: "upwind"}),
    dict(vertical_form="sb81_theta", **{_F: "sb"}),
])
def test_validate_strict_accepts(kw):
    """The setting adds no validation error over the same config at "auto"."""
    base = {k: v for k, v in kw.items() if k != _F}
    assert _errors(_cfg(**kw)) == _errors(_cfg(**base))


def test_driver_built_model_runs_centred_sb_by_default():
    """Through the driver factory and the model's own entry points (the
    ``tendencies`` method and one ``step``, which share the tendency dispatch
    of every lat-lon lane) on a CAM L32 hybrid coordinate: the default equals
    explicit "sb" bit for bit and differs from the "upwind" opt-out."""
    import jax.numpy as jnp
    import numpy as np
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonHydrostaticState)
    from legoesm.driver.component_factory import create_atmosphere_dycore
    from legoesm.grids.factory import create_grid
    from legoesm.grids.vertical import make_cam6_l32_levels
    grid, coord = create_grid("latlon", 24), make_cam6_l32_levels()
    nj, ni, nk = grid.n_lat, grid.n_lon, 32
    rng = np.random.default_rng(3)
    k = jnp.arange(float(nk))
    st = CGridLatLonHydrostaticState(
        u=10.0 + 0.5 * k + jnp.asarray(rng.normal(size=(nj, ni + 1, nk))),
        v=jnp.asarray(rng.normal(size=(nj + 1, ni, nk))).at[0].set(0.0).at[-1].set(0.0),
        T=200.0 + 2.5 * k + jnp.asarray(rng.normal(size=(nj, ni, nk))),
        p_s=1e5 + 300.0 * jnp.asarray(rng.normal(size=(nj, ni))),
        phis=jnp.zeros((nj, ni)))
    out = {}
    for s in ("auto", "sb", "upwind"):
        cfg = _cfg(**({} if s == "auto" else {_F: s}))
        cfg = cfg._replace(grid=cfg.grid._replace(nlev=nk))
        model = create_atmosphere_dycore(cfg, grid, coord)
        assert model.config.vert_advection_scheme == s
        dT = np.asarray(model.tendencies(st)[2])
        T1 = np.asarray(model.step(st, 60.0).T)
        assert np.all(np.isfinite(dT)) and np.all(np.isfinite(T1))
        out[s] = (dT, T1)
    for a, b in zip(out["auto"], out["sb"]):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(out["sb"], out["upwind"]):
        assert not np.allclose(a, b, rtol=0.0, atol=1e-12)


def test_cli_and_yaml_round_trip(tmp_path):
    from legoesm.driver.restart import resolved_config_record
    from legoesm.driver.run_config_yaml import load_yaml_config
    from scripts.run import run_amip
    p = run_amip.build_arg_parser()
    up = run_amip.build_config_from_args(
        p.parse_args(["--latlon-vert-advection-scheme", "upwind"]))
    deck = tmp_path / "deck.yaml"
    deck.write_text(f"{_F}: upwind\n")
    p2 = run_amip.build_arg_parser()
    p2.set_defaults(**load_yaml_config(deck, p2))
    up_yaml = run_amip.build_config_from_args(p2.parse_args([]))
    off = run_amip.build_config_from_args(run_amip.build_arg_parser().parse_args([]))
    for cfg in (up, up_yaml):
        assert getattr(cfg.dycore, _F) == "upwind"
        assert resolved_config_record(cfg)["dycore"][_F] == "upwind"
    assert getattr(off.dycore, _F) == "auto"
