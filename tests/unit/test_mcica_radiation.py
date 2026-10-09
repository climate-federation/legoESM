"""McICA radiation path (cloud_vertical_overlap_optics="mcica").

Each g-point solves its own maximum-random cloud subcolumn inside ONE solve.
The limits pin the sampling: an overcast sky must equal the plain homogeneous
solve and a clear sky the cloud-free one, exactly.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from legoesm.atmosphere.physics.clouds.subcolumns import in_cloud_paths
from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
from legoesm.atmosphere.physics.radiation.rrtmgp_radiation import _get_instance

NCOL, NLEV = 8, 20


def _columns(seed=0):
    r = np.random.default_rng(seed)
    ps = r.uniform(95000.0, 103000.0, NCOL)
    p_half = jnp.asarray(np.linspace(225.0 / 101325.0, 1.0, NLEV + 1)[None] * ps[:, None])
    p_full = 0.5 * (p_half[:, 1:] + p_half[:, :-1])
    ts = r.uniform(275.0, 300.0, NCOL)
    T = jnp.asarray(np.maximum(ts[:, None] - 45.5 * np.log(101325.0 / np.asarray(p_full)), 200.0))
    kw = dict(T=T, p_full=p_full, p_half=p_half, sfc_temperature=jnp.asarray(ts),
              q_v=jnp.asarray(0.012 * (np.asarray(p_full) / 101325.0) ** 3),
              cos_zenith=jnp.full(NCOL, 0.6), sfc_albedo=jnp.full(NCOL, 0.1),
              cloud_r_eff_liq=jnp.full((NCOL, NLEV), 10e-6),
              cloud_r_eff_ice=jnp.full((NCOL, NLEV), 30e-6))
    cf = jnp.asarray(r.uniform(0, 1, (NCOL, NLEV)) * (r.random((NCOL, NLEV)) < 0.4))
    return kw, cf, cf * 0.04, cf * 0.01


@pytest.fixture(scope="module")
def solver():
    return _get_instance(RRTMGPConfig(include_clouds=True, gpoint_batch_size=16))


def _mcica(solver, kw, cf, lwp, iwp):
    liq, ice = in_cloud_paths(cf, lwp, iwp)
    return solver.solve_columns(**kw, cloud_path_liq=liq, cloud_path_ice=ice,
                                mcica_cloud_fraction=cf)


def _same(a, b, atol=0.0):
    for x, y in zip(a, b):
        if x is not None:
            np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=0, atol=atol)


def test_overcast_equals_the_homogeneous_solve(solver):
    kw, _cf, lwp, iwp = _columns()
    one = jnp.ones((NCOL, NLEV))
    _same(_mcica(solver, kw, one, lwp, iwp),
          solver.solve_columns(**kw, cloud_path_liq=lwp, cloud_path_ice=iwp))


def test_clear_equals_the_cloud_free_solve(solver):
    kw, _cf, lwp, iwp = _columns()
    _same(_mcica(solver, kw, jnp.zeros((NCOL, NLEV)), lwp, iwp),
          solver.solve_columns(**kw), atol=1e-9)


def test_partial_cloud_lies_between_clear_and_overcast(solver):
    kw, cf, lwp, iwp = _columns()
    sw_up = lambda o: np.asarray(o.sw_flux_up)[:, 0]
    mc = sw_up(_mcica(solver, kw, cf, lwp, iwp))
    clear = sw_up(solver.solve_columns(**kw))
    liq, ice = in_cloud_paths(cf, lwp, iwp)       # every cell at in-cloud water
    overcast = sw_up(solver.solve_columns(**kw, cloud_path_liq=liq, cloud_path_ice=ice))
    assert np.all(mc >= clear - 1e-9) and np.any(mc > clear + 1.0)
    assert np.all(mc <= overcast + 1e-9)


@pytest.mark.parametrize("aerosol", [False, True])
def test_fused_clear_sky_equals_the_separate_cloud_free_solve(solver, aerosol):
    """``clear_sky=True`` returns the clouds-off TOA fluxes from the all-sky
    solve (shared gas optics): equal to a separate cloud-free solve, and the
    all-sky outputs are unchanged.  Night columns included."""
    kw, cf, lwp, iwp = _columns()
    kw["cos_zenith"] = jnp.asarray(np.linspace(-0.3, 0.9, NCOL))
    if aerosol:
        kw["aerosol_optical_depth"] = jnp.full((NCOL, NLEV), 0.01)
        kw["aerosol_absorption_optical_depth_lw"] = jnp.full((NCOL, NLEV), 0.002)
    liq, ice = in_cloud_paths(cf, lwp, iwp)
    cloudy = dict(cloud_path_liq=liq, cloud_path_ice=ice, mcica_cloud_fraction=cf)
    fused = solver.solve_columns(**kw, **cloudy, clear_sky=True)
    allsky = solver.solve_columns(**kw, **cloudy)
    clear = solver.solve_columns(**kw)
    assert allsky.sw_flux_up_toa_clr is None and allsky.lw_flux_up_toa_clr is None
    _same(fused[:7], allsky[:7], atol=1e-9)
    for got, want in ((fused.sw_flux_up_toa_clr, clear.sw_flux_up[:, 0]),
                      (fused.lw_flux_up_toa_clr, clear.lw_flux_up[:, 0])):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want),
                                   rtol=1e-12, atol=1e-9)
    # the clouds matter in this state, so clear != all-sky is a real check
    assert not np.allclose(np.asarray(clear.sw_flux_up[:, 0]),
                           np.asarray(allsky.sw_flux_up[:, 0]), atol=1.0)
    assert not np.allclose(np.asarray(clear.lw_flux_up[:, 0]),
                           np.asarray(allsky.lw_flux_up[:, 0]), atol=1.0)


def test_fused_clear_sky_all_night(solver):
    """Whole domain dark: the SW solve is skipped, so the clear-sky keys must
    come back from the night branch too (zero SW, LW still the clear solve)."""
    kw, cf, lwp, iwp = _columns()
    kw["cos_zenith"] = jnp.full(NCOL, -0.2)
    liq, ice = in_cloud_paths(cf, lwp, iwp)
    fused = solver.solve_columns(**kw, cloud_path_liq=liq, cloud_path_ice=ice,
                                 mcica_cloud_fraction=cf, clear_sky=True)
    clear = solver.solve_columns(**kw)
    np.testing.assert_array_equal(np.asarray(fused.sw_flux_up_toa_clr), 0.0)
    np.testing.assert_allclose(np.asarray(fused.lw_flux_up_toa_clr),
                               np.asarray(clear.lw_flux_up[:, 0]),
                               rtol=1e-12, atol=1e-9)


def test_column_chunking_is_exact(solver):
    kw, cf, lwp, iwp = _columns()
    liq, ice = in_cloud_paths(cf, lwp, iwp)
    whole = solver.solve_columns(**kw, cloud_path_liq=liq, cloud_path_ice=ice,
                                 mcica_cloud_fraction=cf)
    chunked = solver.solve_columns_chunked(
        column_chunk_size=NCOL // 2, **kw, cloud_path_liq=liq,
        cloud_path_ice=ice, mcica_cloud_fraction=cf)
    _same(whole, chunked, atol=1e-9)


def test_refuses_optical_depth_scaling_or_separate_longwave_paths(solver):
    kw, cf, lwp, iwp = _columns()
    for extra in ({"cloud_fraction": cf}, {"cloud_path_liq_lw": lwp},
                  {"sw_optical_field_only": True}, {"lw_optical_field_only": True}):
        with pytest.raises(ValueError, match="mcica_cloud_fraction"):
            solver.solve_columns(**kw, cloud_path_liq=lwp, cloud_path_ice=iwp,
                                 mcica_cloud_fraction=cf, **extra)


def test_run_amip_cli_offers_mcica():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "scripts/run/run_amip.py"
    spec = importlib.util.spec_from_file_location("run_amip_for_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args = mod.build_arg_parser().parse_args(
        ["--cloud-vertical-overlap-optics", "mcica"])
    assert args.cloud_vertical_overlap_optics == "mcica"


def test_validate_strict_refuses_mcica_with_two_column():
    from legoesm.driver.config import ExperimentConfig

    with pytest.raises((ValueError, SystemExit), match="mutually exclusive"):
        ExperimentConfig(cloud_partial_coverage_optics="two_column",
                         cloud_vertical_overlap_optics="mcica").validate_strict()
