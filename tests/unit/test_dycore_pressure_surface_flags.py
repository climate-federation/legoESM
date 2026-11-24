"""DycoreConfig / run_amip switches for the pressure-surface T-damping options.

``latlon_ah_pressure_surfaces`` -> lat-lon C-grid PE ``A_h_pressure_surfaces``;
``spectral_t_damping_pressure_surfaces`` -> spectral PE
``T_damping_pressure_surfaces``.  Both default OFF, each routed only to its own
core and refused elsewhere (user decision 2026-11-24).
"""
from __future__ import annotations

import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

_FLAGS = ("latlon_ah_pressure_surfaces", "spectral_t_damping_pressure_surfaces")
_NLEV = 8


@pytest.fixture(autouse=True)
def _fp64_policy():
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(prev)


def _cfg(grid_type, resolution, **dycore):
    from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig
    disc = "spectral" if grid_type == "gaussian" else "latlon_cgrid"
    return ExperimentConfig(
        grid=GridConfig(grid_type=grid_type, resolution=resolution, nlev=_NLEV),
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


def test_defaults_off():
    from legoesm.driver.config import DycoreConfig
    assert all(getattr(DycoreConfig(), f) is False for f in _FLAGS)


@pytest.mark.parametrize("grid_type, res, flag, field", [
    ("latlon", 24, "latlon_ah_pressure_surfaces", "A_h_pressure_surfaces"),
    ("gaussian", 21, "spectral_t_damping_pressure_surfaces", "T_damping_pressure_surfaces"),
])
def test_routed_to_its_core_and_off_by_default(grid_type, res, flag, field):
    on, off = _build(_cfg(grid_type, res, **{flag: True})), _build(_cfg(grid_type, res))
    assert getattr(on.config, field) is True and getattr(off.config, field) is False
    # only that one field differs between the built dycore configs
    assert on.config._replace(**{field: False}) == off.config
    # the flag adds no validation error on its own core
    assert _errors(_cfg(grid_type, res, **{flag: True})) == _errors(_cfg(grid_type, res))


@pytest.mark.parametrize("grid_type, res, flag", [
    ("gaussian", 21, "latlon_ah_pressure_surfaces"),
    ("latlon", 24, "spectral_t_damping_pressure_surfaces"),
])
def test_refused_on_the_other_core(grid_type, res, flag):
    cfg = _cfg(grid_type, res, **{flag: True})
    assert flag in _errors(cfg)
    with pytest.raises(ValueError, match=flag):
        _build(cfg)


def test_refused_when_inert():
    ll = _cfg("latlon", 24, latlon_ah_pressure_surfaces=True, a_h_scale=0.0)
    sp = _cfg("gaussian", 21, spectral_t_damping_pressure_surfaces=True, hyperdiff_scale=0.0)
    assert "a_h_scale" in _errors(ll) and "hyperdiff_scale" in _errors(sp)
    # the factory refuses too (paths that skip validate_strict)
    with pytest.raises(ValueError, match="A_h = 0"):
        _build(ll)
    with pytest.raises(ValueError, match="no T hyperdiffusion"):
        _build(sp)


def test_factory_refuses_non_bool():
    with pytest.raises(ValueError, match="must be a bool"):
        _build(_cfg("latlon", 24, latlon_ah_pressure_surfaces="false"))


@pytest.mark.parametrize("cli, flag", [
    ("--latlon-ah-pressure-surfaces", "latlon_ah_pressure_surfaces"),
    ("--spectral-t-damping-pressure-surfaces", "spectral_t_damping_pressure_surfaces"),
])
def test_cli_and_yaml_round_trip_and_manifest_record(cli, flag, tmp_path):
    """Each flag on its own: CLI and YAML deck set exactly that field, and the
    run manifest's resolved config records both values."""
    from legoesm.driver.restart import resolved_config_record
    from legoesm.driver.run_config_yaml import load_yaml_config
    from scripts.run import run_amip
    other = next(f for f in _FLAGS if f != flag)
    p = run_amip.build_arg_parser()
    on = run_amip.build_config_from_args(p.parse_args([cli]))
    deck = tmp_path / "deck.yaml"
    deck.write_text(f"{flag}: true\n")
    p2 = run_amip.build_arg_parser()
    p2.set_defaults(**load_yaml_config(deck, p2))
    on_yaml = run_amip.build_config_from_args(p2.parse_args([]))
    off = run_amip.build_config_from_args(run_amip.build_arg_parser().parse_args([]))
    for cfg in (on, on_yaml):
        assert getattr(cfg.dycore, flag) is True and getattr(cfg.dycore, other) is False
        rec = resolved_config_record(cfg)["dycore"]
        assert rec[flag] is True and rec[other] is False
    assert getattr(off.dycore, flag) is False


def test_spectral_flag_refused_on_a_non_gaussian_grid_and_with_two_levels():
    from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig
    wrong_grid = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=24, nlev=_NLEV),
        dycore=DycoreConfig(model_type="hydrostatic", discretization="spectral",
                            energy_fixer=False, dt=300.0,
                            spectral_t_damping_pressure_surfaces=True))
    assert "spectral_t_damping_pressure_surfaces" in _errors(wrong_grid)
    two = _cfg("gaussian", 21, spectral_t_damping_pressure_surfaces=True)
    two = two._replace(grid=two.grid._replace(nlev=2))
    assert "nlev >= 3" in _errors(two)
    from legoesm.driver.component_factory import create_atmosphere_dycore
    from legoesm.grids.factory import create_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    with pytest.raises(ValueError, match="nlev >= 3"):
        create_atmosphere_dycore(two, create_grid("gaussian", 21), create_sigma_coordinate(2))
