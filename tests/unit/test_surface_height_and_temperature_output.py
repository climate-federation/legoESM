"""Direct regressions for land forcing height, hourly extrema and skin output."""
from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace as NS

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.driver.model_driver import ModelDriver
from legoesm.driver.diagnostics import DiagnosticCollector
from legoesm.grids.vertical import create_sigma_coordinate, make_cam6_l32_levels
from legoesm.diagnostics.monthly_means import SpatialDailyAccumulator
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.surface_scheme import TwoLeafCanopyConfig


def _forcing():
    f = lambda value: jnp.full(2, value)
    return AtmToSurface(
        sw_down=f(500.), lw_down=f(330.), precip_total=f(0.), precip_snow=f(0.),
        T_lowest=f(290.), q_lowest=f(.005), u_lowest=f(5.), v_lowest=f(0.),
        p_lowest=f(98000.), p_surface=f(100000.), rho_lowest=f(1.2),
        cos_zenith=f(.7), co2_ppmv=f(412.), has_radiation=f(1.),
        has_precipitation=f(1.))


@pytest.mark.parametrize('coordinate', [make_cam6_l32_levels, lambda: create_sigma_coordinate(36)])
def test_driver_land_closure_uses_grid_height(coordinate):
    """Execute the REAL nested closure; it cannot be imported as a top-level API."""
    tree = ast.parse(inspect.getsource(inspect.getmodule(ModelDriver)))
    node, = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
             and n.name == '_marshal_land_forcing']
    T = jnp.array([[270., 290.], [265., 280.]])
    ps = jnp.array([100000., 72000.])  # pressure/topography must not become altitude
    sigma = coordinate()
    driver = NS(state=NS(T=NS(data=T), p_s=NS(data=ps), tracers={},
                         u=NS(data=jnp.ones_like(T))), sigma=sigma, grid=None,
                model=NS(_sfc_diag=tuple(NS(data=jnp.ones(2)) for _ in range(10))))
    env = dict(self=driver, cfg=NS(co2_ppmv=412.), jnp=jnp, constants=constants,
               AtmToSurface=AtmToSurface, _doy=32., _sod=3600.,
               reconstruct_cell_velocity=lambda u, grid: (u, u * 0),
               _cos_zen_fn=lambda day, hour: jnp.ones(2))
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<driver closure>', 'exec'), env)
    forcing = env['_marshal_land_forcing']()
    expected = (constants.R_d * T[:, -1] / constants.g
                * jnp.log(sigma.pressure_at_half(ps)[:, -1]
                          / sigma.pressure_at_full(ps)[:, -1]))
    np.testing.assert_allclose(forcing.z_lowest, expected, rtol=1e-12)
    assert np.all(np.asarray(forcing.z_lowest) > 10.)
    np.testing.assert_array_equal(forcing.T_lowest, T[:, -1])


def test_two_leaf_height_and_temperature_are_paired_jit_and_grad():
    from legoesm.land.surface_scheme.two_leaf_canopy import compute_two_leaf_canopy_fluxes
    forcing = _forcing()
    height = jnp.array([62., 137.])
    config = MultiLayerLandConfig()
    def run(f, cfg):
        return compute_two_leaf_canopy_fluxes(
            T_soil_top=jnp.full(2, 294.), forcing=f,
            canopy_config=TwoLeafCanopyConfig(max_iters=30), land_config=cfg,
            canopy_params=None, w_frac_rz=jnp.full(2, .5),
            wind_speed=jnp.full(2, 5.), wind_dir_x=jnp.ones(2),
            wind_dir_y=jnp.zeros(2), soil_thermal_fn=lambda G, dt: jnp.full(2, 294.),
            TgC_override=forcing.T_lowest - constants.T_freeze,
            dt=600.).shflx
    # Independent equivalent observed forcing: both reference transformations.
    expected = run(forcing._replace(
        T_lowest=forcing.T_lowest + constants.g / constants.c_pd * height),
        config._replace(z_ref=height))
    f = lambda z: run(forcing._replace(z_lowest=z), config)
    actual = f(height)
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)
    assert np.max(np.abs(np.asarray(actual - run(forcing, config)))) > .1
    np.testing.assert_allclose(jax.jit(f)(height), actual, rtol=1e-8, atol=1e-8)
    gradient = jax.grad(lambda z: jnp.sum(f(z)))(height)
    assert np.all(np.isfinite(gradient))
    assert np.max(np.abs(gradient)) > 1e-6


def test_multilayer_simple_seb_consumes_height_pair():
    from legoesm.land.multilayer_land import init_multilayer_land_state, step_multilayer_land
    from legoesm.land.surface_scheme import SimpleSEBConfig
    cfg = MultiLayerLandConfig(bulk_scheme='most', surface_scheme=SimpleSEBConfig())
    forcing = _forcing()
    height = jnp.array([62., 137.])
    state = init_multilayer_land_state(2, cfg, T_init=294.)
    def run(f, c):
        return step_multilayer_land(state, f, c, U_min=1., dt=60.)[1].shflx
    expected = run(forcing._replace(
        T_lowest=forcing.T_lowest + constants.g / constants.c_pd * height),
        cfg._replace(z_ref=height))
    actual = run(forcing._replace(z_lowest=height), cfg)
    np.testing.assert_allclose(actual, expected, rtol=1e-10)
    assert np.all(np.asarray(actual) < np.asarray(run(forcing, cfg)))


def test_extremes_only_preserves_means_counts_sparse_days_and_restart():
    acc = SpatialDailyAccumulator(1, 2)
    acc.add_2d(3.5, 0, {'tas': np.full((1, 2), 290.)})
    counts = acc._call_counts.copy()
    for day in (1, 2, 3):
        for hour in range(24):
            acc.add_extremes_2d(day + hour / 24, 0,
                               {'tas': np.array([[270. + hour, 300. - hour]])})
    assert acc._call_counts == counts
    assert acc._max_count_ever == 1
    assert acc._data[(0, 3)]['tas'][1] == 1
    assert acc._data[(0, 1)]['tas'][1] == 0
    restored = SpatialDailyAccumulator(1, 2)
    restored.set_state(acc.get_state())
    result = restored.finalize()
    assert result['days'] == [(0, 1), (0, 2), (0, 3)]
    np.testing.assert_array_equal(result['field_2d_tas'][2], [[290., 290.]])
    assert np.isnan(result['field_2d_tas'][:2]).all()
    np.testing.assert_array_equal(result['field_2d_tas_min'][0], [[270., 277.]])
    np.testing.assert_array_equal(result['field_2d_tas_max'][0], [[293., 300.]])
    dc = DiagnosticCollector.__new__(DiagnosticCollector)
    dc._spatial_daily = restored
    attrs = dc._daily_extreme_attrs()
    assert 'tas' not in attrs
    assert attrs['tasmin']['cell_methods'] == 'time: minimum'
    assert attrs['tasmax']['cell_methods'] == 'time: maximum'
    assert 'hourly' in attrs['tasmax']['comment']


def test_actual_mpas_loop_samples_hourly_across_restart():
    """Execute the actual loop's scheduling statements, including its feed call."""
    tree = ast.parse(inspect.getsource(inspect.getmodule(ModelDriver)))
    loop = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_run_mpas')
    block = next(n for n in ast.walk(loop) if isinstance(n, ast.For)
                 and any(isinstance(s, ast.Assign) and any(
                     isinstance(t, ast.Name) and t.id == '_sample_day' for t in s.targets)
                         for s in n.body))
    i = next(i for i, n in enumerate(block.body) if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == '_sample_day' for t in n.targets))
    code = compile(ast.Module(body=block.body[i:i+3], type_ignores=[]), '<hourly loop>', 'exec')
    samples = []
    driver = NS(_mpas_cmip_feed_on=True,
                _feed_mpas_cmip_accumulators=lambda day, **kw: samples.append((day, kw)))
    env = dict(self=driver, np=np, START_DAY=.4, DT=112.)
    for step in range(800):
        env['step'] = step
        exec(code, env)
    assert len(samples) == 25
    assert all(kw == {'state_only': True} for day, kw in samples)
    hours = np.arange(10, 35)
    times = np.array([day * 24 for day, kw in samples])
    assert np.all(times >= hours)
    assert np.all(times - hours < 112. / 3600. + 1e-10)


def test_skin_temperature_blend_reaches_native_accumulator():
    from legoesm.io.cmor_output import lookup_cmor_entry
    from tests.unit.test_mpas_cmip_accumulator_feed import (
        _make_collector, _synthetic_cell_fields, _as_driver)
    from legoesm.grids.factory import create_grid
    mesh = create_grid('mpas', 1, lloyd_iterations=1)
    dc, sigma_full, _ = _make_collector(mesh)
    f = _synthetic_cell_fields(mesh, sigma_full)
    n = int(mesh.nCells)
    land = jnp.linspace(0., 1., n)
    driver = _as_driver(NS(
        state=NS(T=NS(data=f['T']), p_s=NS(data=f['p_s']),
                 u=NS(data=f['u_edge']), phis=NS(data=f['phis']), tracers={}),
        grid=mesh, model=NS(_sfc_diag=None),
        config=NS(T_ice=260.), get_sst_sic=lambda day: (jnp.full(n, 280.), jnp.full(n, .25)),
        _land_T_skin_last=jnp.full(n, 310.), _f_land=land))
    kw = ModelDriver._mpas_cmip_native_kwargs(driver, .5, dc)
    expected = land * 310. + (1. - land) * 275.
    np.testing.assert_allclose(kw['ts'], expected)
    dc.feed_cmip_accumulators_native(.5, **kw)
    actual = dc._spatial_monthly.finalize(min_sample_fraction=0)['field_2d_ts'][0]
    np.testing.assert_allclose(actual, dc._regrid_to_latlon_2d(expected))
    _, entry = lookup_cmor_entry('ts', 'Amon')
    assert entry['standard_name'] == 'surface_temperature'
    assert entry['units'] == 'K'
    assert entry['long_name'] == 'Surface Temperature'
    assert 'time: mean' in entry['cell_methods']
    # Hourly feed reuses the SAME surface diagnostics and preserves flux sums.
    calls = dc._spatial_daily._call_counts.copy()
    driver.diagnostics = dc
    resets = []
    driver._mpas_sfc_accum = NS(reset=lambda **kw: resets.append(kw))
    ModelDriver._feed_mpas_cmip_accumulators(driver, .6, state_only=True)
    assert dc._spatial_daily._extreme_counts  # swallowed feed errors must fail
    assert resets == []
    assert sum(dc._spatial_daily._call_counts.values()) == sum(calls.values()) + 1
    np.testing.assert_array_equal(
        dc._spatial_monthly.finalize(min_sample_fraction=0)['field_2d_ts'][0], actual)


def test_clm_warm_reference_geometry_is_traceable_and_correction_once():
    from legoesm.land.canopy.clm_ml_interface import (
        refresh_reference_height, _build_stubs, _ensure_clm_initialized)
    from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLCanopyFluxesType import create_mlcanopy
    from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varcon import lapse_rate
    ml = create_mlcanopy(1, 2)
    ml = ml._replace(
        ntop_canopy=jnp.array([0, 2, 2]), ncan_canopy=jnp.array([0, 4, 4]),
        ztop_canopy=jnp.array([0., 4., 6.]), zref_forcing=jnp.array([0., 10., 10.]))
    ml = ml._replace(zw_profile=ml.zw_profile.at[:, :5].set(jnp.array([
        [0., 0., 0., 0., 0.], [0., 2., 4., 7., 10.], [0., 3., 6., 8., 10.]])))
    heights = jnp.array([0., 62., 137.])
    out = jax.jit(refresh_reference_height)(ml, heights)
    np.testing.assert_array_equal(out.zref_forcing, heights)
    np.testing.assert_allclose(out.zw_profile[1:, 4], heights[1:])
    np.testing.assert_allclose(out.dz_profile[1:, 4], (heights[1:] - jnp.array([4., 6.])) / 2)
    np.testing.assert_array_equal(out.zw_profile[:, :3], ml.zw_profile[:, :3])
    assert out.zw_profile.shape == ml.zw_profile.shape
    np.testing.assert_allclose(lapse_rate, constants.g / constants.c_pd, rtol=1e-14)

    from legoesm.land.canopy.config import CLMMLCanopyConfig
    _ensure_clm_initialized()
    forcing = _forcing()._replace(z_lowest=heights[1:])
    stubs = _build_stubs(
        2, forcing, CLMMLCanopyConfig(), MultiLayerLandConfig(), None,
        jnp.full(2, 290.), None, None, np.array([.1, .2]), None)
    np.testing.assert_array_equal(stubs['atm2lnd'].forc_t_downscaled_col[1:],
                                  forcing.T_lowest)
    np.testing.assert_array_equal(stubs['frictionvel'].forc_hgt_u_patch[1:], heights[1:])


def test_clm_warm_step_uses_new_height_and_one_temperature_correction():
    from tests.land.integration.test_clm_ml_jit_forward import _warm_start, _forcing as clm_forcing
    from legoesm.land.canopy.clm_ml_interface import compute_clm_ml_canopy_fluxes
    old, grid_info, kw = _warm_start()
    forcing = clm_forcing(jnp.array([295.]))._replace(z_lowest=jnp.array([62.]))
    _, new = compute_clm_ml_canopy_fluxes(
        T_soil_top=kw['Ts'][:, 0], forcing=forcing,
        canopy_config=kw['cfg'], land_config=kw['lc'], land_params=None,
        canopy_state=old, dt=1800., T_soil=kw['Ts'], psi_soil=kw['psi'],
        theta_soil=kw['th'], lat=kw['lat'], doy=180., grid_info=grid_info)
    ml = new.mlcanopy
    np.testing.assert_allclose(ml.zref_forcing[1], 62.)
    np.testing.assert_allclose(ml.zw_profile[1, ml.ncan_canopy[1]], 62.)
    np.testing.assert_allclose(ml.thref_forcing[1], 295. + constants.g / constants.c_pd * 62.)
    assert ml.zw_profile.shape == old.mlcanopy.zw_profile.shape
