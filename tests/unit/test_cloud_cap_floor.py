"""Polar-cap radiative cloud floor: applies only inside its gates, is off by default,
reaches the RRTMGP solver through the real backend, and every silent-no-op path refuses."""
from __future__ import annotations

import importlib.util
import pathlib
import sys

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


_LAT = jnp.deg2rad(jnp.array([80.0, 60.0, 89.0]))
_P = jnp.array([[30000.0, 60000.0, 80000.0, 95000.0]] * 3)
_DP = jnp.full((3, 4), 2000.0)


def test_off_leaves_values_unchanged_even_inside_the_gates():
    cfg = CloudConfig(scheme="xu_randall", cap_floor_on=False, cap_floor_cf=0.9, cap_floor_q_c=1e-3)
    out = apply_cap_cloud_floor(_props(), _LAT, _P, _DP, cfg)
    assert np.asarray(out.cloud_fraction) == pytest.approx(0.2)
    assert np.asarray(out.lwp) == pytest.approx(1e-3)


def test_floor_applies_only_poleward_and_below_the_pressure_gate():
    cfg = CloudConfig(scheme="xu_randall", cap_floor_on=True, cap_floor_lat_deg=70.0,
                      cap_floor_p_max_pa=70000.0, cap_floor_cf=0.8, cap_floor_q_c=5e-5)
    out = apply_cap_cloud_floor(_props(), _LAT, _P, _DP, cfg)
    cf = np.asarray(out.cloud_fraction); lwp = np.asarray(out.lwp)
    assert cf[1] == pytest.approx([0.2] * 4)                # 60N untouched
    assert cf[0] == pytest.approx([0.2, 0.2, 0.8, 0.8])     # 80N: only below 700 hPa
    expected = 0.8 * 5e-5 * 2000.0 / constants.g
    assert lwp[0, 2] == pytest.approx(expected) and lwp[0, 0] == pytest.approx(1e-3)
    assert np.asarray(out.lwp_lw)[2, 3] == pytest.approx(expected)
    assert np.array_equal(np.asarray(out.iwp), np.asarray(_props().iwp))   # ice untouched
    # a non-default latitude gate moves the boundary: 60N now floored, 30N not
    cfg2 = cfg._replace(cap_floor_lat_deg=55.0)
    out2 = apply_cap_cloud_floor(_props(), jnp.deg2rad(jnp.array([80.0, 60.0, 30.0])), _P, _DP, cfg2)
    assert np.asarray(out2.cloud_fraction)[1, 3] == pytest.approx(0.8)
    assert np.asarray(out2.cloud_fraction)[2, 3] == pytest.approx(0.2)


def test_floor_never_lowers_an_existing_cloud():
    cfg = CloudConfig(scheme="xu_randall", cap_floor_on=True, cap_floor_cf=0.5, cap_floor_q_c=1e-6)
    props = _props()._replace(cloud_fraction=jnp.full((3, 4), 0.9), lwp=jnp.full((3, 4), 0.5))
    out = apply_cap_cloud_floor(props, jnp.deg2rad(jnp.full(3, 85.0)), jnp.full((3, 4), 90000.0), _DP, cfg)
    assert np.asarray(out.cloud_fraction) == pytest.approx(0.9) and np.asarray(out.lwp) == pytest.approx(0.5)


def test_build_cloud_config_threads_the_knobs_and_keeps_defaults_when_none():
    base = build_cloud_config("xu_randall", cap_floor_on=None, cap_floor_q_c=None)
    assert base == build_cloud_config("xu_randall")
    assert base.cap_floor_on is False and base.cap_floor_lat_deg == 70.0
    cfg = build_cloud_config("xu_randall", cap_floor_on=True, cap_floor_q_c=1e-4)
    assert cfg.cap_floor_on is True and cfg.cap_floor_q_c == 1e-4 and cfg.cap_floor_cf == 0.8


class _Stop(Exception):
    pass


def _backend_kwargs_reaching_the_solver(cloud_config):
    """Run the REAL standalone radiation backend on a small Arctic column set with
    a solver stub that records what it is handed; returns those kwargs."""
    from legoesm.atmosphere.physics.radiation import integration as rad_int
    from legoesm.atmosphere.physics.radiation.config import RadiationConfig
    seen = {}

    class _Solver:
        def solve_columns(self, **kw):
            seen.update(kw); raise _Stop()
    ncol, nlev = 3, 4
    lat = jnp.deg2rad(jnp.array([85.0, 80.0, 40.0]))         # radians, as the driver hands them
    p_full = jnp.array([[30000.0, 60000.0, 80000.0, 95000.0]] * ncol)
    p_half = jnp.array([[10000.0, 45000.0, 70000.0, 90000.0, 100000.0]] * ncol)
    cfg = RadiationConfig(scheme="rrtmgp", cloud_scheme=cloud_config.scheme, cloud_config=cloud_config)
    with pytest.raises(_Stop):
        rad_int._call_radiation_backend(
            cfg, T=jnp.full((ncol, nlev), 250.0), p_full=p_full, p_half=p_half,
            sfc_temperature=jnp.full(ncol, 250.0), lat=lat, q_v=jnp.full((ncol, nlev), 1e-4),
            insolation=jnp.zeros(ncol), q_cloud=jnp.zeros((ncol, nlev)), q_ice=jnp.zeros((ncol, nlev)),
            rrtmgp_solver=_Solver())
    return seen


def test_floor_reaches_the_solver_through_the_real_backend():
    """The solver is handed GRID-MEAN liquid paths (cloud_fraction is deliberately
    not passed, see ``to_rrtmg_kwargs``), so the floor must show up there."""
    on = CloudConfig(scheme="xu_randall", cloud_vertical_overlap_optics="none", cap_floor_on=True,
                     cap_floor_lat_deg=70.0, cap_floor_p_max_pa=70000.0, cap_floor_cf=0.8, cap_floor_q_c=5e-5)
    k_on = _backend_kwargs_reaching_the_solver(on)
    k_off = _backend_kwargs_reaching_the_solver(on._replace(cap_floor_on=False))
    assert "cloud_fraction" not in k_on and "cloud_fraction" not in k_off
    lwp_on, lwp_off = np.asarray(k_on["cloud_path_liq"]), np.asarray(k_off["cloud_path_liq"])
    assert lwp_off == pytest.approx(0.0)                      # dry columns: no cloud without the floor
    dp = np.array([35000.0, 25000.0, 20000.0, 10000.0])
    expected = 0.8 * 5e-5 * dp / constants.g
    assert lwp_on[0] == pytest.approx([0.0, 0.0, expected[2], expected[3]])   # 85N below 700 hPa
    assert lwp_on[1] == pytest.approx(lwp_on[0])                               # 80N
    assert lwp_on[2] == pytest.approx(0.0)                                     # 40N untouched


def test_floor_reaches_the_solver_with_production_subcolumn_overlap():
    """Production runs ``max_random`` subcolumns: the floored fraction sets how
    many subcolumns are cloudy and the liquid path is the in-cloud value."""
    on = CloudConfig(scheme="xu_randall", cloud_vertical_overlap_optics="max_random", cloud_n_subcolumns=8,
                     cap_floor_on=True, cap_floor_lat_deg=70.0, cap_floor_p_max_pa=70000.0,
                     cap_floor_cf=0.75, cap_floor_q_c=5e-5)
    k = _backend_kwargs_reaching_the_solver(on)
    lwp = np.asarray(k["cloud_path_liq"]).reshape(8, 3, 4).transpose(1, 0, 2)   # solver gets (n_sub, ncol, nlev)
    cloudy = (lwp > 0)
    assert (cloudy[:2, :, 2:].sum(axis=1) == 6).all()               # 6 of 8 subcolumns, every column and layer
    assert not cloudy[2].any() and not cloudy[:, :, :2].any()
    dp = np.array([35000.0, 25000.0, 20000.0, 10000.0])
    in_cloud = 5e-5 * dp / constants.g                              # grid-mean 0.75*q*dp/g over 0.75 cover
    for col in range(2):
        for lev in (2, 3):
            assert lwp[col, :, lev][cloudy[col, :, lev]] == pytest.approx(in_cloud[lev])
            assert lwp[col, :, lev].mean() == pytest.approx(0.75 * in_cloud[lev])


def test_silent_no_op_paths_refuse():
    from legoesm.driver.physics_pipeline import (build_physics_pipeline, cap_floor_lane_applies,
                                                  refuse_cap_floor_on_fv)
    from legoesm.driver.model_driver import ModelDriver, _standalone_cloud_config
    from types import SimpleNamespace

    def cfg(grid, disc, on=True):
        return SimpleNamespace(cloud_cap_floor_on=on, grid=SimpleNamespace(grid_type=grid),
                               dycore=SimpleNamespace(discretization=disc))
    assert cap_floor_lane_applies(cfg("mpas", "finite_volume"))
    assert cap_floor_lane_applies(cfg("latlon", "spectral"))
    assert not cap_floor_lane_applies(cfg("latlon", "finite_volume"))
    assert not cap_floor_lane_applies(cfg("cubed_sphere", "fv3_duo"))
    with pytest.raises(ValueError, match="finite-volume"):
        refuse_cap_floor_on_fv(cfg("cubed_sphere", "finite_volume"))
    refuse_cap_floor_on_fv(cfg("cubed_sphere", "finite_volume", on=False))
    refuse_cap_floor_on_fv(cfg("mpas", "finite_volume"))

    # ModelDriver.run itself: the gate is its first statement, so a stub whose
    # next call raises a sentinel shows whether the gate let the lane through.
    class _Entered(Exception):
        pass
    def run_with(config):
        stub = SimpleNamespace(config=config)
        stub._reject_shallow_water_unrunnable = lambda: (_ for _ in ()).throw(_Entered())
        return ModelDriver.run(stub)
    with pytest.raises(_Entered):                     # MPAS + floor on: proceeds
        run_with(cfg("mpas", "finite_volume"))
    with pytest.raises(_Entered):                     # any lane, floor off: proceeds
        run_with(cfg("cubed_sphere", "finite_volume", on=False))
    with pytest.raises(ValueError, match="finite-volume"):   # FV lane + floor on: refused
        run_with(cfg("cubed_sphere", "finite_volume"))
    # The pipeline is built on EVERY lane (MPAS included), so the refusal must
    # not live in the builder: the MPAS cap arm died at setup when it did.
    import inspect
    assert "refuse_cap_floor_on_fv" not in inspect.getsource(build_physics_pipeline)
    with pytest.raises(ValueError, match="never be applied"):
        _standalone_cloud_config(SimpleNamespace(cloud_cap_floor_on=True), "none")
    assert _standalone_cloud_config(SimpleNamespace(cloud_cap_floor_on=False), "none") is None


def _run_amip():
    root = pathlib.Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("run_amip", root / "scripts" / "run" / "run_amip.py")
    ra = importlib.util.module_from_spec(spec); sys.modules["run_amip"] = ra; spec.loader.exec_module(ra)
    return ra


def test_cli_round_trip_bounds_and_cross_checks():
    ra = _run_amip()
    parser = ra.build_arg_parser()
    def cfg_from(argv):
        return ra.build_config_from_args(ra._postprocess_args(parser.parse_args(argv), parser))
    base = cfg_from(["--dataset", "analytical"])
    assert base.cloud_cap_floor_on is False and base.cloud_cap_floor_q_c is None
    cfg = cfg_from(["--dataset", "analytical", "--clouds", "xu_randall", "--radiation", "rrtmgp",
                    "--cloud-cap-floor", "--cloud-cap-floor-q-c", "1e-4", "--cloud-cap-floor-lat-deg", "70"])
    assert cfg.cloud_cap_floor_on is True and cfg.cloud_cap_floor_q_c == 1e-4
    for rad in ("rrtmgp", "rrtmg"):                                  # "rrtmg" = the production deck's alias
        cfg_from(["--dataset", "analytical", "--clouds", "xu_randall", "--radiation", rad,
                  "--cloud-cap-floor"]).validate_strict()
    from legoesm.driver.model_driver import _standalone_cloud_config
    cc = _standalone_cloud_config(cfg, "xu_randall", allow_convective_cloud=True)
    assert cc.cap_floor_on is True and cc.cap_floor_q_c == 1e-4
    bad = cfg_from(["--dataset", "analytical", "--cloud-cap-floor-cf", "1.5"])
    with pytest.raises(Exception, match="cloud_cap_floor_cf"):
        bad.validate_strict()
    for argv in (["--clouds", "none"], ["--clouds", "xu_randall", "--radiation", "gray"]):
        noop = cfg_from(["--dataset", "analytical", "--cloud-cap-floor", *argv])
        with pytest.raises(ValueError, match="silent no-op"):
            noop.validate_strict()
    aimip = ra._postprocess_args(parser.parse_args(
        ["--dataset", "analytical", "--clouds", "xu_randall", "--cloud-cap-floor"]), parser)
    aimip.aimip_classical_checkpoint = "/nonexistent.eqx"
    with pytest.raises(SystemExit, match="aimip-classical-checkpoint"):
        ra._apply_aimip_classical_overrides(aimip)
