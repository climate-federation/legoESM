"""MPAS land: per-step two-leaf parameter rebuild (seasonal LAI / canopy
height / wet-soil albedo), as the offline calibration runs it.

Pins: the rebuild at the start day reproduces the setup-time params leaf by
leaf; LAI follows the monthly climatology between two dates; fields the
rebuild leaves None are carried from the setup params (not dropped); the
packed gather of a full-grid rebuild equals the rebuild of the packed columns.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.driver.model_driver import mpas_land_step_params
from legoesm.land.boundary_data import (
    make_step_land_params_updater, surface_data_to_land_params)
from legoesm.land.canopy import CanopyConfig
from legoesm.land.global_surface_data import (
    GlobalSurfaceData, GlobalSurfaceDataConfig)
from legoesm.land.surface_params import CLM5_PFT_NAMES, N_PFT_CLM5

jax.config.update("jax_enable_x64", True)

_BD = CLM5_PFT_NAMES.index("broadleaf_deciduous_temperate")
_C3 = CLM5_PFT_NAMES.index("c3_grass")
# Deciduous seasonal cycle: ~0 in winter, 5 in July.
_LAI_BD = np.array([0.1, 0.1, 0.3, 1.0, 3.0, 4.5, 5.0, 4.5, 3.0, 1.0, 0.3, 0.1])


def _gsd(cols=(0, 1, 2)):
    """3-column fixture; ``cols`` keeps a subset (a rank's local columns)."""
    c = np.asarray(cols)
    ncol, npft = 3, N_PFT_CLM5
    pft = np.zeros((1, ncol, npft))
    pft[0, 0, _BD] = 1.0          # deciduous forest
    pft[0, 1, _C3] = 1.0          # grass
    # col 2: zero cover -> bare fallback
    lai = np.zeros((12, ncol, npft))
    lai[:, 0, _BD] = _LAI_BD
    lai[:, 1, _C3] = np.linspace(0.5, 2.0, 12)
    htop = np.zeros((12, ncol, npft))
    htop[:, 0, _BD] = np.linspace(18.0, 22.0, 12)
    htop[:, 1, _C3] = 0.5
    n = c.size
    z = np.zeros((n, 4))
    return GlobalSurfaceData(
        sand_frac=jnp.asarray(np.full((n, 4), 0.4)),
        clay_frac=jnp.asarray(np.full((n, 4), 0.2)),
        organic=jnp.asarray(z), bulk_density=jnp.asarray(z),
        soil_color=jnp.asarray(np.array([5, 3, 8])[c]),
        cell_area=jnp.ones(n),
        years=jnp.asarray(np.array([2000.0])),
        f_land=jnp.ones((1, n)), f_lake=jnp.zeros((1, n)),
        f_glacier=jnp.zeros((1, n)),
        pft_frac=jnp.asarray(pft[:, c]),
        months=jnp.arange(12.0),
        lai_monthly=jnp.asarray(lai[:, c]), sai_monthly=jnp.zeros((12, n, npft)),
        htop_monthly=jnp.asarray(htop[:, c]), hbot_monthly=jnp.zeros((12, n, npft)),
        config=GlobalSurfaceDataConfig(),
    )


_THETA = jnp.full(3, 0.2)   # init_land_surface_data's nominal top-layer wetness


def _setup_and_update():
    gsd = _gsd()
    base = surface_data_to_land_params(gsd, CanopyConfig(), 0.0, _THETA,
                                       year=2000.0)
    upd = make_step_land_params_updater(gsd, CanopyConfig())
    return base, upd


def test_start_day_rebuild_reproduces_setup_params():
    base, upd = _setup_and_update()
    got = mpas_land_step_params(upd, base, _THETA, jnp.asarray(0.0),
                                jnp.asarray(2000.0))
    for f in base._fields:
        a, b = getattr(base, f), getattr(got, f)
        assert (a is None) == (b is None), f
        if a is not None:
            np.testing.assert_allclose(np.asarray(b), np.asarray(a),
                                       rtol=1e-12, atol=1e-12, err_msg=f)


def test_lai_follows_the_monthly_climatology():
    base, upd = _setup_and_update()
    fn = jax.jit(lambda th, d, y: mpas_land_step_params(upd, base, th, d, y))
    jan = fn(_THETA, jnp.asarray(15.5), jnp.asarray(2000.0))
    jul = fn(_THETA, jnp.asarray(196.5), jnp.asarray(2000.0))   # mid-July centre
    assert fn._cache_size() == 1                                  # no retrace
    np.testing.assert_allclose(float(jan.LAI[0]), _LAI_BD[0], rtol=1e-12)
    np.testing.assert_allclose(float(jul.LAI[0]), _LAI_BD[6], rtol=1e-12)
    # The setup params are January's for the whole run without the refresh.
    np.testing.assert_allclose(float(base.LAI[0]), float(jan.LAI[0]), rtol=0.05)
    assert float(jul.LAI[0]) > 10.0 * float(base.LAI[0])
    assert float(jul.hc[0]) > float(jan.hc[0])


def test_wetter_top_soil_darkens_the_soil_albedo():
    base, upd = _setup_and_update()
    dry = mpas_land_step_params(upd, base, jnp.full(3, 0.05), jnp.asarray(0.0),
                                jnp.asarray(2000.0))
    wet = mpas_land_step_params(upd, base, jnp.full(3, 0.40), jnp.asarray(0.0),
                                jnp.asarray(2000.0))
    assert np.all(np.asarray(wet.ALB_VIS) < np.asarray(dry.ALB_VIS))


def test_fields_the_rebuild_leaves_unset_are_carried():
    """Without calibrated root tables the driver keeps the CLM map's per-column
    root depth / wilting point / field capacity on the setup params; a rebuild
    that returned None for them must not drop them."""
    base, upd = _setup_and_update()
    assert base.root_depth is None
    roots = dict(root_depth=jnp.asarray([0.3, 0.6, 0.9]),
                 theta_wp=jnp.asarray([0.10, 0.11, 0.12]),
                 theta_fc=jnp.asarray([0.30, 0.31, 0.32]))
    base = base._replace(**roots)
    got = mpas_land_step_params(upd, base, _THETA, jnp.asarray(180.0),
                                jnp.asarray(2000.0))
    for f, v in roots.items():
        np.testing.assert_array_equal(np.asarray(getattr(got, f)), np.asarray(v))


def test_rank_local_rebuild_matches_the_global_columns():
    """Under MPI each rank builds its updater from its OWN columns' surfdata,
    and the driver packs the land columns of a full-grid rebuild.  Both are
    right only if the rebuild is column-local: a rebuild on a column subset
    must equal the global rebuild restricted to those columns."""
    from legoesm.land.multilayer_land import gather_land_columns
    theta = jnp.asarray([0.1, 0.2, 0.3])
    args = (jnp.asarray(150.0), jnp.asarray(2000.0))
    g = _gsd()
    full = mpas_land_step_params(
        make_step_land_params_updater(g, CanopyConfig()),
        surface_data_to_land_params(g, CanopyConfig(), 0.0, _THETA, year=2000.0),
        theta, *args)
    sub = (0, 2)
    gs = _gsd(sub)
    local = mpas_land_step_params(
        make_step_land_params_updater(gs, CanopyConfig()),
        surface_data_to_land_params(gs, CanopyConfig(), 0.0, _THETA[:2],
                                    year=2000.0),
        theta[np.asarray(sub)], *args)
    packed = gather_land_columns(full, jnp.asarray(sub), 3)
    for f in full._fields:
        a, b = getattr(packed, f), getattr(local, f)
        assert (a is None) == (b is None), f
        if a is not None:
            np.testing.assert_allclose(np.asarray(b), np.asarray(a),
                                       rtol=1e-12, atol=1e-12, err_msg=f)
