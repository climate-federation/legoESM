"""Published means must resolve a diurnal cycle, independent of flux cadence."""
import ast
import functools
import inspect
import os
import time
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from legoesm.driver.diagnostics import DiagnosticCollector
from legoesm.driver.model_driver import ModelDriver, _MPASSfcFluxAccum
from legoesm.grids.factory import create_grid
from legoesm.io.cmor_output import CFWriter
from tests.unit.test_mpas_cmip_accumulator_feed import _as_driver
from tests.unit.test_mpas_cmor_flux_feed import _make_collector


@pytest.fixture
def hourly_case(tmp_path):
    from legoesm.atmosphere.physics.clouds.config import build_cloud_config
    resolution = int(os.environ.get('HOURLY_BENCH_RES', '1'))
    mesh = create_grid('mpas', resolution, lloyd_iterations=1)
    cloud = build_cloud_config('resolved', convective_cloud=False)
    if resolution == 1:
        dc, sigma = _make_collector(mesh, cloud)
    else:
        from legoesm.grids.vertical import create_sigma_coordinate
        coord = create_sigma_coordinate(36)
        sigma = np.asarray(coord.sigma_full)
        dc = DiagnosticCollector(
            nlev=36, sigma_full=sigma, dsigma=np.asarray(coord.dsigma),
            experiment_id='amip', monthly_means=True, cmip_output=True,
            cmip_resolution_deg=5., cloud_config=cloud)
        dc.set_cmip_grid_info(grid_type='mpas', grid=mesh, start_year=1979)
    dc.cf_writer = CFWriter(output_dir=tmp_path, experiment_id='amip',
                            model_id='legoESM', ref_date='1979-01-01')
    n, nz = int(mesh.nCells), len(sigma)
    f = lambda value: NS(data=np.full((n, nz), value))
    driver = _as_driver(NS(
        diagnostics=dc, grid=mesh, _mpas_cmip_feed_on=True,
        config=NS(output=NS(clear_sky_diag=True), dycore=NS(dt=3600.)),
        state=NS(T=f(280.), p_s=NS(data=np.full(n, 100000.)),
                 u=NS(data=np.zeros((int(mesh.nEdges), nz))),
                 phis=NS(data=np.zeros(n)),
                 tracers={'q_v': f(.005), 'q_c': f(.0001), 'q_i': f(.00002)}),
        model=NS(_sfc_diag=None),
        _mpas_sfc_accum=_MPASSfcFluxAccum(expected_steps=24,
                                         window_start_day=29., dt_s=3600.),
        _feed_mpas_moisture_budget=lambda *args: None))
    from legoesm.grids.vertical import create_sigma_coordinate
    driver.sigma = create_sigma_coordinate(nz)
    driver._feed_mpas_cmip_accumulators = functools.partial(
        ModelDriver._feed_mpas_cmip_accumulators, driver)
    # Execute the production hook and interval call, not a copied scheduler.
    tree = ast.parse(inspect.getsource(inspect.getmodule(ModelDriver)))
    run = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
               and n.name == '_run_column')
    loop = next(n for n in ast.walk(run) if isinstance(n, ast.For)
                and any(isinstance(s, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == '_sample_day'
                    for t in s.targets) for s in n.body))
    i = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == '_sample_day' for t in n.targets))
    hourly = compile(ast.Module(body=loop.body[i:i+3], type_ignores=[]), '<hourly>', 'exec')
    daily_call = next(n for n in ast.walk(run) if isinstance(n, ast.Call)
                      and isinstance(n.func, ast.Attribute)
                      and n.func.attr == '_feed_mpas_cmip_accumulators'
                      and isinstance(n.args[0], ast.BinOp))
    daily = compile(ast.Expression(body=daily_call), '<daily>', 'eval')
    return dc, driver, hourly, daily


def test_published_hourly_diurnal_mean_not_midnight(hourly_case, tmp_path):
    """Three complete days across January/February, with a mid-day restart."""
    xr = pytest.importorskip('xarray')
    dc, driver, hourly, daily = hourly_case
    env = dict(self=driver, np=np, START_DAY=29., DT=3600.)
    for step in range(72):
        day = 29. + (step + 1) / 24.
        # Known mean 280 K; midnight value 292 K at every longitude.
        driver.state.T.data[:] = 280. + 12. * np.cos(2. * np.pi * day)
        slots = [None] * 12
        slots[6] = NS(data=np.full(int(driver.grid.nCells), 20. + step % 24))
        driver._mpas_sfc_accum.add(slots)
        local_step = step if step < 36 else step - 36
        env.update(step=local_step, elapsed_day=(local_step + 1) / 24.)
        before = driver._mpas_sfc_accum._steps
        exec(hourly, env)
        assert driver._mpas_sfc_accum._steps == before
        if step == 35:
            # New collector + real checkpoint sidecar, then rebase the loop.
            path = tmp_path / 'cmor_state.npz'
            dc.save_cmor_accumulators(path)
            resumed, _ = _make_collector(driver.grid, dc._cloud_config)
            resumed.cf_writer = dc.cf_writer
            assert resumed.load_cmor_accumulators(path)
            dc = driver.diagnostics = resumed
            env['START_DAY'] = day
        if (step + 1) % 24 == 0:
            eval(daily, env)
            assert driver._mpas_sfc_accum._steps == 0
    data = dc._spatial_monthly.finalize(min_sample_fraction=0)
    dc._write_cmip_data(data)
    for path in tmp_path.rglob('tas_*.nc'):
        with xr.open_dataset(path) as ds:
            np.testing.assert_allclose(ds.tas.values, 280., atol=1e-10, rtol=0)
            assert np.min(np.abs(ds.tas.values - 292.)) > 10.
    assert list(tmp_path.rglob('tas_*.nc')), 'must test published output'
    np.testing.assert_allclose(data['field_3d_ta'], 280., atol=1e-10, rtol=0)
    np.testing.assert_allclose(data['field_2d_hfss'], 31.5, atol=1e-10, rtol=0)
    assert data['months'] == [(0, 1), (0, 2)]
    assert dc._spatial_monthly._data_2d[(0, 1)]['tas'][1] == 48
    assert dc._spatial_monthly._data_2d[(0, 2)]['tas'][1] == 24
    days = dc._spatial_daily.finalize()
    np.testing.assert_allclose(days['field_2d_tas'], 280., atol=1e-10, rtol=0)
    assert len(days['days']) == 3


def test_published_state_sampling_attributes_never_snapshot(hourly_case, tmp_path):
    """All eleven scorecard inputs and other state sources retain mean semantics."""
    xr = pytest.importorskip('xarray')
    dc, driver, hourly, _ = hourly_case
    for step in range(24):
        exec(hourly, dict(self=driver, np=np, START_DAY=29., DT=3600., step=step))
    # A resurrected obsolete label must not change output, even on a restored object.
    dc.cmip_snapshot_vars = {'clt', 'clwvi', 'clivi', 'tas', 'ta', 'ua', 'hus',
                             'va', 'ps', 'psl', 'prw', 'wap'}
    dc._write_cmip_data(dc._spatial_monthly.finalize(min_sample_fraction=0))
    dc._write_cmip_daily_files()
    # lwp is clwvi-clivi; level scores slice ta/ua/hus, not separate CMOR vars.
    required = {'clt', 'clwvi', 'clivi', 'prw', 'tas', 'ta', 'ua', 'hus',
                'va', 'ps', 'psl', 'wap'}
    seen = set()
    for path in tmp_path.rglob('*.nc'):
        with xr.open_dataset(path) as ds:
            for name in required.intersection(ds.data_vars):
                attrs = ds[name].attrs
                assert 'time: mean' in attrs['cell_methods'], (name, attrs)
                assert 'time: point' not in attrs['cell_methods'], (name, attrs)
                comment = attrs.get('comment', '').lower()
                for old in ('diurnally aliased', 'once-daily instantaneous',
                            'single instantaneous sample', 'sparsely sampled'):
                    assert old not in comment, (name, comment)
                seen.add(name)
    assert seen == required
    assert not hasattr(DiagnosticCollector, '_daily_snapshot_attrs')


def test_hourly_sampling_cost(hourly_case):
    """Repeatable full-hook timing; HOURLY_BENCH_RES=6 selects the production size.

    Cost is host CPU latency, including wind reconstruction, cloud diagnosis,
    pressure interpolation, regridding, and accumulation; no physics integration.
    Compare this same test under a baseline import overlay for the old hook.
    """
    from legoesm.atmosphere.physics.clouds.config import build_cloud_config
    dc, driver, hourly, _ = hourly_case
    dc._cloud_config = build_cloud_config('sundqvist', convective_cloud=False)
    from legoesm import constants
    import jax.numpy as jnp
    driver.config.T_ice = constants.T_freeze_ocean
    driver.get_sst_sic = lambda day: (
        jnp.full(int(driver.grid.nCells), 280.), jnp.zeros(int(driver.grid.nCells)))
    env = dict(self=driver, np=np, START_DAY=29., DT=3600., step=0)
    exec(hourly, env)  # warm caches / eager JAX primitives before timing
    regrid_s = [0.]
    originals = {}
    for name in ('_regrid_to_latlon_2d', '_regrid_to_latlon_3d'):
        original = getattr(dc, name)
        originals[name] = original
        def timed(*args, _fn=original, **kwargs):
            start = time.perf_counter()
            result = _fn(*args, **kwargs)
            regrid_s[0] += time.perf_counter() - start
            return result
        setattr(dc, name, timed)
    samples = []
    for step in range(1, 11):
        env['step'] = step
        start = time.perf_counter()
        exec(hourly, env)
        samples.append(time.perf_counter() - start)
    for name, original in originals.items():
        setattr(dc, name, original)
    assert dc._spatial_daily._extreme_counts
    assert driver._mpas_sfc_accum._steps == 0
    print('HOURLY_COST ' + json.dumps(dict(
        cells=int(driver.grid.nCells), levels=dc.nlev,
        cmor_grid=[dc._cmip_nlat, dc._cmip_nlon],
        seconds_per_hook=float(np.mean(samples)),
        seconds_regridding_per_hook=regrid_s[0] / len(samples),
        seconds_per_physics_step=float(np.mean(samples)) * 112.5 / 3600.,
        seconds_per_60_days=float(np.mean(samples)) * 24 * 60,
        backend='CPU', x64=True, repeats=len(samples))))
