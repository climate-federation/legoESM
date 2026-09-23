"""CAM6 gravity-wave suite wiring: multi-source ``e3sm_cam`` selection from
the experiment config / AMIP CLI, the convective-heating lag carry, the Beres
table plumbing, the Voronoi frontogenesis producer and the composite rules.

Run with ``JAX_ENABLE_X64=1``.
"""

from __future__ import annotations

import types

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.physics.gravity_wave_drag.config import (
    E3SMBeresConfig,
    E3SMCAMConfig,
    GravityWaveDragConfig,
)
from legoesm.atmosphere.physics.gravity_wave_drag.integration import (
    _e3sm_source_kwargs,
    _validate_gwd_composite,
    e3sm_mfcc_table,
    e3sm_sources,
    get_gwd_fn,
    gwd_reads_conv_heating,
)
from legoesm.driver.config import ExperimentConfig
from legoesm.driver.physics_pipeline import gwd_config_for

from legoesm import constants

CAM6_KW = dict(
    gravity_wave_drag="e3sm_cam",
    e3sm_cam_source="orographic+frontal+convective",
    e3sm_cam_pgwv=32,
    e3sm_cam_effgw=0.125,
    e3sm_cam_effgw_cm=1.0,
    e3sm_cam_effgw_beres=0.4,
    e3sm_cam_frontgfc=3.0e-15,
    e3sm_cam_beres_variant="cam6",
)


def test_cam6_suite_resolves_from_experiment_config():
    cfg = ExperimentConfig(**CAM6_KW)
    cfg.validate_strict()
    g = gwd_config_for(cfg)
    assert e3sm_sources(g) == ("orographic", "frontal", "convective")
    assert gwd_reads_conv_heating(g)
    assert g.e3sm_cam.effgw == 0.125
    assert g.e3sm_cam.frontal.effgw == 1.0
    assert g.e3sm_cam.beres.effgw == 0.4
    assert g.e3sm_cam.beres.spectrum_shift == "end_off"
    assert g.e3sm_cam.beres.storm_speed_truncate is False
    assert g.e3sm_cam.beres.source_level_rule == "interface_below_p"
    assert g.e3sm_cam.beres.hd_index_rule == "nearest_grid"
    assert g.e3sm_cam.beres.hdepth_min_km == 1.0          # gw_drag.F90:882
    assert g.e3sm_cam.beres.mfcc_table_path == ""
    assert g.e3sm_cam.dttke_use_intrinsic is False        # separate explicit knob
    g2 = gwd_config_for(ExperimentConfig(**dict(CAM6_KW, e3sm_cam_dttke_intrinsic=True)))
    assert g2.e3sm_cam.dttke_use_intrinsic is True
    # get_gwd_fn accepts the multi-source config
    name, fn, sub = get_gwd_fn(g)
    assert name == "e3sm_cam" and sub.source == "orographic+frontal+convective"


def test_defaults_are_byte_identical_to_before():
    g = gwd_config_for(ExperimentConfig(gravity_wave_drag="e3sm_cam"))
    assert g.e3sm_cam.frontal.effgw is None and g.e3sm_cam.beres.effgw is None
    assert g.e3sm_cam.beres.spectrum_shift == "circular"
    assert g.e3sm_cam.beres.storm_speed_truncate is True
    assert g.e3sm_cam.beres.source_level_rule == "nearest_midpoint"
    assert g.e3sm_cam.beres.hd_index_rule == "nint"
    assert g.e3sm_cam.beres.hdepth_min_km == 2.5
    assert not gwd_reads_conv_heating(g)
    assert e3sm_sources(GravityWaveDragConfig(scheme="mcfarlane")) == ()


@pytest.mark.parametrize("bad", [
    dict(e3sm_cam_source="orographic+bogus"),
    dict(e3sm_cam_source="frontal+frontal"),
    dict(e3sm_cam_effgw_cm=1.5),
    dict(e3sm_cam_effgw_beres=0.0),
    dict(e3sm_cam_beres_variant="cam5"),
    dict(e3sm_cam_mfcc_table_path="/nonexistent/mfcc.nc"),
    dict(gravity_wave_drag="mcfarlane+e3sm_cam", e3sm_cam_source="orographic+frontal"),
    dict(convection="none"),
])
def test_validate_strict_rejects(bad):
    kw = dict(CAM6_KW)
    kw.update(bad)
    with pytest.raises(ValueError):
        ExperimentConfig(**kw).validate_strict()


def test_composite_with_mcfarlane_allows_frontal_and_convective():
    kw = dict(CAM6_KW, gravity_wave_drag="mcfarlane+e3sm_cam",
              e3sm_cam_source="frontal+convective")
    cfg = ExperimentConfig(**kw)
    cfg.validate_strict()
    g = gwd_config_for(cfg)
    _validate_gwd_composite(g.scheme, g.e3sm_cam.source)
    assert e3sm_sources(g) == ("frontal", "convective")
    with pytest.raises(ValueError, match="double-count"):
        _validate_gwd_composite("mcfarlane+e3sm_cam", "orographic+frontal")
    with pytest.raises(ValueError, match="Unknown e3sm_cam source"):
        _validate_gwd_composite("hines+e3sm_cam", "bogus")


def test_amip_cli_round_trips_the_cam6_suite():
    from scripts.run.run_amip import (
        _postprocess_args,
        build_arg_parser,
        build_config_from_args,
    )
    parser = build_arg_parser()
    argv = ["--dataset", "analytical", "--gravity-wave-drag", "e3sm_cam",
            "--e3sm-cam-source", "orographic+frontal+convective",
            "--e3sm-cam-pgwv", "32", "--e3sm-cam-effgw", "0.125",
            "--e3sm-cam-effgw-cm", "1.0", "--e3sm-cam-effgw-beres", "0.4",
            "--e3sm-cam-frontgfc", "3.0e-15",
            "--e3sm-cam-beres-variant", "cam6", "--e3sm-cam-dttke-intrinsic"]
    cfg = build_config_from_args(_postprocess_args(parser.parse_args(argv), parser))
    assert cfg.e3sm_cam_source == "orographic+frontal+convective"
    assert cfg.e3sm_cam_frontgfc == 3.0e-15
    from legoesm.driver.physics_pipeline import gwd_config_for
    assert gwd_config_for(cfg).e3sm_cam.frontal.frontgfc == 3.0e-15
    with pytest.raises(ValueError, match="frontgfc"):
        cfg._replace(e3sm_cam_frontgfc=-1.0).validate_strict()
    assert cfg.e3sm_cam_dttke_intrinsic is True
    assert cfg.e3sm_cam_effgw_cm == 1.0 and cfg.e3sm_cam_effgw_beres == 0.4
    assert cfg.e3sm_cam_beres_variant == "cam6"
    cfg0 = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg0.e3sm_cam_effgw_cm is None and cfg0.e3sm_cam_beres_variant == "e3sm"
    assert cfg0.e3sm_cam_mfcc_table_path == "" and cfg0.e3sm_cam_dttke_intrinsic is False


# --- convective-heating lag carry ---------------------------------------------

def test_physics_state_carries_conv_heating():
    from legoesm.atmosphere.physics.combined import PhysicsConfig
    from legoesm.atmosphere.physics.physics_state import (
        init_physics_state,
        update_physics_state,
    )
    ps = init_physics_state(5, 4, PhysicsConfig())
    assert ps.conv_heating.shape == (5, 4) and not np.any(np.asarray(ps.conv_heating))
    heat = jnp.full((5, 4), 1e-4)
    ps1 = update_physics_state(ps, {"conv_heating": heat})
    np.testing.assert_array_equal(np.asarray(ps1.conv_heating), np.asarray(heat))
    ps2 = update_physics_state(ps1, {})                  # carried forward
    np.testing.assert_array_equal(np.asarray(ps2.conv_heating), np.asarray(heat))


def test_source_kwargs_read_lagged_carry_and_fall_back_to_zeros():
    g = GravityWaveDragConfig(
        scheme="e3sm_cam",
        e3sm_cam=E3SMCAMConfig(source="convective", pgwv=4))
    grid = types.SimpleNamespace(land_frac=jnp.ones(3))
    T = jnp.full((3, 4), 250.0)
    heat = jnp.full((3, 4), 2e-4)
    kw = _e3sm_source_kwargs(
        g, grid, 3, 4, types.SimpleNamespace(conv_heating=heat), "TBL",
        None, None, T, None)
    np.testing.assert_array_equal(np.asarray(kw["netdt_col"]), np.asarray(heat))
    assert kw["mfcc_table"] == "TBL" and "frontgf_col" not in kw
    assert np.all(np.asarray(kw["land_frac_col"]) == 1.0)
    kw0 = _e3sm_source_kwargs(g, grid, 3, 4, None, None, None, None, T, None)
    assert not np.any(np.asarray(kw0["netdt_col"])) and kw0["mfcc_table"] is None
    # a mis-sized carry (column set changed) is a LOUD error, never silent zeros
    with pytest.raises(ValueError, match="conv_heating"):
        _e3sm_source_kwargs(
            g, grid, 3, 4, types.SimpleNamespace(conv_heating=jnp.ones((7, 4))),
            None, None, None, T, None)


def test_convection_factory_publishes_conv_heating():
    """The shared hydrostatic/MPAS convection bridge publishes this step's
    heating into the carry the GWD factory reads next step."""
    from legoesm.atmosphere.forcing.idealized.held_suarez import (
        held_suarez_init_latlon,
    )
    from legoesm.atmosphere.physics.convection.config import ConvectionConfig
    from legoesm.atmosphere.physics.convection.integration import (
        make_convection_physics,
    )
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    grid = create_latlon_grid(n_lat=6, n_lon=8)
    sc = create_sigma_coordinate(8, sigma_top=0.05)
    state = held_suarez_init_latlon(grid, sc, T_init=290.0, perturbation_amplitude=0.0)
    fn = make_convection_physics(ConvectionConfig(scheme="sbm"),
                                 model_type="hydrostatic", dt=300.0)
    tend, carry = fn(state, grid, sc)
    assert isinstance(carry, dict) and "conv_heating" in carry
    assert carry["conv_heating"].shape == (6 * 8, 8)
    np.testing.assert_allclose(
        np.asarray(carry["conv_heating"]),
        np.asarray(tend.dT_dt.data).reshape(6 * 8, 8), rtol=0, atol=0)


def test_stateful_predicate_counts_the_beres_carry():
    from legoesm.atmosphere.physics.combined import (
        PhysicsConfig,
        physics_config_requires_phys_state,
    )
    g = GravityWaveDragConfig(scheme="e3sm_cam",
                              e3sm_cam=E3SMCAMConfig(source="convective", pgwv=4))
    assert physics_config_requires_phys_state(PhysicsConfig(gravity_wave_drag=g))
    g0 = GravityWaveDragConfig(scheme="e3sm_cam", e3sm_cam=E3SMCAMConfig(source="orographic"))
    assert not physics_config_requires_phys_state(PhysicsConfig(gravity_wave_drag=g0))


# --- Beres table plumbing -----------------------------------------------------

def test_mfcc_table_loaded_once_from_config_path(tmp_path):
    import xarray as xr
    n_ps, n_mw, n_hd = 9, 5, 3
    data = np.random.default_rng(0).uniform(0, 1, (n_ps, n_mw, n_hd))
    path = tmp_path / "mfcc.nc"
    xr.Dataset({"mfcc": (("PS", "MW", "HD"), data)},
               coords={"HD": np.arange(1, n_hd + 1, dtype=np.float64)}).to_netcdf(path)
    g = GravityWaveDragConfig(
        scheme="e3sm_cam",
        e3sm_cam=E3SMCAMConfig(source="frontal+convective", pgwv=2,
                               beres=E3SMBeresConfig(mfcc_table_path=str(path))))
    with pytest.raises(ValueError, match="maxuh"):
        e3sm_mfcc_table(g)                                 # default maxh/maxuh != file
    g = g._replace(e3sm_cam=g.e3sm_cam._replace(
        beres=g.e3sm_cam.beres._replace(maxh=n_hd, maxuh=(n_mw - 1) // 2)))
    tbl = e3sm_mfcc_table(g)
    assert tbl.shape == (n_hd, n_mw, 5) and not tbl.flags.writeable
    np.testing.assert_array_equal(tbl, np.transpose(data, (2, 1, 0))[:, :, 2:7])
    assert e3sm_mfcc_table(g) is tbl                     # lru-cached: loaded once
    assert e3sm_mfcc_table(g._replace(e3sm_cam=g.e3sm_cam._replace(source="frontal"))) is None
    # the experiment-level path reaches the resolved config
    cfg = ExperimentConfig(**dict(CAM6_KW, e3sm_cam_mfcc_table_path=str(path),
                                  e3sm_cam_pgwv=2))
    cfg.validate_strict()
    assert gwd_config_for(cfg).e3sm_cam.beres.mfcc_table_path == str(path)


# --- Voronoi frontogenesis producer -------------------------------------------

def test_frontogenesis_on_voronoi_mesh():
    from legoesm.atmosphere.physics.gravity_wave_drag.frontogenesis import (
        compute_frontogenesis,
        frontogenesis_supported,
    )
    from legoesm.grids.voronoi import create_voronoi_mesh
    mesh = create_voronoi_mesh(subdivision_level=3)
    assert frontogenesis_supported(mesh)
    n = int(np.asarray(mesh.latCell).shape[0])
    lat = np.asarray(mesh.latCell)
    lon = np.asarray(mesh.lonCell)
    nlev = 3
    p = np.full((n, nlev), 5.0e4)
    T = (250.0 + 20.0 * np.sin(lat) ** 2 + 5.0 * np.cos(3 * lon) * np.cos(lat))[:, None] \
        * np.ones((1, nlev))
    # solid-body rotation: (a.grad)U = Omega x a -> F == 0 for ANY theta
    u_sb = (constants.Omega * constants.R_earth * np.cos(lat))[:, None] * np.ones((1, nlev))
    v_sb = np.zeros((n, nlev))
    f_sb, ang = compute_frontogenesis(jnp.asarray(u_sb), jnp.asarray(v_sb),
                                      jnp.asarray(T), jnp.asarray(p), mesh)
    assert f_sb.shape == (n, nlev) and ang.shape == (n, nlev)
    assert np.all(np.isfinite(np.asarray(f_sb)))
    # confluent deformation flow: u = -alpha x (east), a genuine front-maker
    alpha = 1e-4
    x = constants.R_earth * np.cos(lat) * np.sin(lon)
    u_cf = (-alpha * x)[:, None] * np.ones((1, nlev))
    f_cf, _ = compute_frontogenesis(jnp.asarray(u_cf), jnp.asarray(v_sb),
                                    jnp.asarray(T), jnp.asarray(p), mesh)
    away = np.abs(lat) < 1.2                                 # polar cells excluded
    ref = np.max(np.abs(np.asarray(f_cf)[away]))
    assert ref > 0.0
    # covariant canary: solid-body F is small next to the deformation flow's
    # (the Perot reconstruction is second-order, not exact: ~1e-2 ratio)
    assert np.max(np.abs(np.asarray(f_sb)[away])) < 0.1 * ref
    assert np.any(np.asarray(f_cf)[away] > 0.0)


# --- factory lanes end to end -------------------------------------------------

def test_mpas_factory_runs_all_three_sources_with_the_lag_carry():
    """The MPAS GWD factory runs the CAM6 three-source e3sm_cam selection:
    frontogenesis from the Voronoi producer, Beres from the lagged
    ``conv_heating`` carry, landfrac on the orographic source — and the carry
    is READ (a heated carry changes the tendency); jit parity.  (The cube
    ``hydrostatic`` factory is exercised through the compiled pipeline; the
    lat-lon lane runs the pipeline too.)"""
    lane = "mpas"
    from legoesm.atmosphere.forcing.idealized.held_suarez import (
        held_suarez_init_mpas,
    )
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    _prev = get_policy()
    set_policy(PrecisionPolicy.fp64())                    # state arrays f64 (codex)
    try:
        _mpas_lane_body(lane, held_suarez_init_mpas)
    finally:
        set_policy(_prev)


def _mpas_lane_body(lane, held_suarez_init_mpas):
    from legoesm.atmosphere.physics.combined import PhysicsConfig
    from legoesm.atmosphere.physics.gravity_wave_drag.integration import (
        make_gwd_physics,
    )
    from legoesm.atmosphere.physics.physics_state import init_physics_state, update_physics_state
    from legoesm.grids.vertical import create_sigma_coordinate
    sc = create_sigma_coordinate(10, sigma_top=0.02)
    if lane == "mpas":
        from legoesm.grids.voronoi import create_voronoi_mesh
        grid = create_voronoi_mesh(subdivision_level=1, lloyd_iterations=2)
        state = held_suarez_init_mpas(grid, sc, T_init=290.0, perturbation_amplitude=0.0)
        ncol = int(np.asarray(grid.latCell).shape[0])
        # Held-Suarez starts at rest (no waves anywhere): give it a sheared
        # zonal jet, projected onto the edge normals (angleEdge from east).
        lat_e = np.asarray(grid.latEdge)
        shear = np.linspace(0.3, 1.0, 10)[None, :]
        u_zonal = (30.0 * np.cos(lat_e) ** 2)[:, None] * shear
        u_edge = u_zonal * np.cos(np.asarray(grid.angleEdge))[:, None]
        state = state._replace(u=state.u.replace(data=jnp.asarray(u_edge)))
        grid = grid._replace(land_frac=jnp.full(ncol, 0.5),
                             subgrid_topo_stddev=jnp.full(ncol, 300.0))
    g = GravityWaveDragConfig(
        scheme="e3sm_cam",
        e3sm_cam=E3SMCAMConfig(
            source="orographic+frontal+convective", pgwv=8, effgw=0.125,
            beres=E3SMBeresConfig(effgw=0.4, hdepth_min_km=1.0,
                                  spectrum_shift="end_off",
                                  storm_speed_truncate=False,
                                  source_level_rule="interface_below_p")))
    fn = make_gwd_physics(g, lane, 900.0)
    pcfg = PhysicsConfig(gravity_wave_drag=g)
    ps0 = init_physics_state(ncol, 10, pcfg)
    tend0, _ = fn(state, grid, sc, phys_state=ps0)
    du0 = np.asarray(tend0.du_dt.data)
    assert np.all(np.isfinite(du0)) and np.all(np.isfinite(np.asarray(tend0.dT_dt.data)))
    # a convectively heated carry (deep heating 1-9 km) must change the drag
    z = 7000.0 * np.log(1.0 / np.asarray(sc.sigma_full))
    heat = np.where((z > 1000.0) & (z < 9000.0), 5e-4, 0.0)[None, :] * np.ones((ncol, 1))
    ps1 = update_physics_state(ps0, {"conv_heating": jnp.asarray(heat)})
    tend1, _ = fn(state, grid, sc, phys_state=ps1)
    du1 = np.asarray(tend1.du_dt.data)
    assert np.all(np.isfinite(du1))
    assert not np.allclose(du0, du1, rtol=0, atol=0), "conv_heating carry not read"
    # jit parity
    # jit parity at f64 (the earlier 1e-7 relative gap was the default f32
    # precision policy building the state, codex round 2)
    # (atol: the bottom-level entries are O(1e-14) cancellation residuals of
    # the momentum fixer against O(1e-5) tendencies; fusion order moves them
    # at the 1e-20 level)
    f_jit = jax.jit(lambda s, p: fn(s, grid, sc, phys_state=p)[0].du_dt.data)
    np.testing.assert_allclose(np.asarray(f_jit(state, ps1)), du1,
                               rtol=1e-10, atol=1e-12 * np.max(np.abs(du1)))
