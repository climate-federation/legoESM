"""The per-step land-params updater must be usable THROUGH ``jax.jit``.

The updater used to close over its host arrays, so a calibrator that batches
grid cells inside a jitted function could not hand it in.  It is now a
host-side ``precompute_canopy_updater`` returning a pytree plus a pure
``apply_canopy_updater``; ``make_step_land_params_updater`` stays as the
wrapper every driver already calls.

What is pinned here: the split reproduces the wrapper BIT-FOR-BIT (the
refactor is behaviour-preserving), the pytree really does survive a jit
boundary as an argument (the point of the change), and the optional per-PFT
root-zone table is carried on both paths.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.land.boundary_data.step_updater import (
    apply_canopy_updater,
    make_step_land_params_updater,
    precompute_canopy_updater,
)
from legoesm.land.canopy import CanopyConfig

from legoesm.land.global_surface_data import (
    GlobalSurfaceData, GlobalSurfaceDataConfig)
from legoesm.land.surface_params import CLM5_PFT_NAMES, N_PFT_CLM5


_THETA = jnp.asarray([0.28])
_BE = CLM5_PFT_NAMES.index("broadleaf_evergreen_tropical")
_C4 = CLM5_PFT_NAMES.index("c4_grass")


def _forest_then_grass():
    """1-column transient cover: pure forest in 1850, pure C4 grass in 2000
    (same shape as the transient test's fixture; kept local so this module has
    no cross-test import)."""
    ncol, npft = 1, N_PFT_CLM5
    pft = np.zeros((2, ncol, npft))
    pft[0, 0, _BE] = 1.0
    pft[1, 0, _C4] = 1.0
    lai = np.zeros((12, ncol, npft))
    lai[:, 0, _BE], lai[:, 0, _C4] = 5.0, 1.5
    htop = np.zeros((12, ncol, npft))
    htop[:, 0, _BE] = 25.0
    z = np.zeros((ncol, 4))
    return GlobalSurfaceData(
        sand_frac=jnp.asarray(np.full((ncol, 4), 0.4)),
        clay_frac=jnp.asarray(np.full((ncol, 4), 0.2)),
        organic=jnp.asarray(z), bulk_density=jnp.asarray(z),
        soil_color=jnp.asarray(np.array([5])),
        cell_area=jnp.ones(ncol),
        years=jnp.asarray(np.array([1850.0, 2000.0])),
        f_land=jnp.ones((2, ncol)), f_lake=jnp.zeros((2, ncol)),
        f_glacier=jnp.zeros((2, ncol)),
        pft_frac=jnp.asarray(pft),
        months=jnp.arange(12.0),
        lai_monthly=jnp.asarray(lai), sai_monthly=jnp.zeros((12, ncol, npft)),
        htop_monthly=jnp.asarray(htop), hbot_monthly=jnp.zeros((12, ncol, npft)),
        config=GlobalSurfaceDataConfig(),
    )


def _mixed_columns():
    """Three columns exercising every branch of the updater at once: a
    transient vegetated column, a GLACIER column (albedo override, no leaves)
    and an UNCOVERED column (gap-filled from the bare template)."""
    ncol, npft, years = 3, N_PFT_CLM5, [1850.0, 2000.0]
    pft = np.zeros((2, ncol, npft))
    pft[0, 0, _BE] = 1.0          # col 0: forest -> grass
    pft[1, 0, _C4] = 1.0
    pft[:, 1, _C4] = 1.0          # col 1: vegetated but glaciated
    # col 2 keeps zero cover -> not surfdata-covered -> bare fallback
    lai = np.zeros((12, ncol, npft))
    lai[:, 0, _BE], lai[:, 0, _C4], lai[:, 1, _C4] = 5.0, 1.5, 2.0
    htop = np.zeros((12, ncol, npft))
    htop[:, 0, _BE] = 25.0
    # A column counts as ice only where the glacier fraction EXCEEDS both the
    # land and lake fractions (builders.glacier_mask), so the land fraction has
    # to be lowered on that column as well.
    f_land = np.ones((2, ncol))
    f_land[:, 1] = 0.1
    f_glac = np.zeros((2, ncol))
    f_glac[:, 1] = 0.9
    z = np.zeros((ncol, 4))
    return GlobalSurfaceData(
        sand_frac=jnp.asarray(np.full((ncol, 4), 0.4)),
        clay_frac=jnp.asarray(np.full((ncol, 4), 0.2)),
        organic=jnp.asarray(z), bulk_density=jnp.asarray(z),
        soil_color=jnp.asarray(np.array([5, 3, 8])),
        cell_area=jnp.ones(ncol),
        years=jnp.asarray(np.asarray(years, dtype=float)),
        f_land=jnp.asarray(f_land), f_lake=jnp.zeros((2, ncol)),
        f_glacier=jnp.asarray(f_glac),
        pft_frac=jnp.asarray(pft),
        months=jnp.arange(12.0),
        lai_monthly=jnp.asarray(lai), sai_monthly=jnp.zeros((12, ncol, npft)),
        htop_monthly=jnp.asarray(htop), hbot_monthly=jnp.zeros((12, ncol, npft)),
        config=GlobalSurfaceDataConfig(),
    )
_ROOTS = {"root_depth": np.array([0.5] * 17), "theta_wp": np.array([0.1] * 17),
          "theta_fc": np.array([0.3] * 17)}


def _leaves_equal(a, b):
    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    assert len(la) == len(lb), (len(la), len(lb))
    for x, y in zip(la, lb):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_split_reproduces_the_wrapper_bitwise():
    """precompute+apply == the closure the drivers call, on every leaf."""
    gsd = _forest_then_grass()
    cfg = CanopyConfig()
    wrapper = make_step_land_params_updater(gsd, cfg)
    pre = precompute_canopy_updater(gsd)
    for doy, year in ((15.0, 1850.0), (200.0, 1925.0), (355.0, 2000.0)):
        lp_w, lai_w = wrapper(_THETA, jnp.asarray(doy), jnp.asarray(year))
        lp_s, lai_s = apply_canopy_updater(pre, _THETA, jnp.asarray(doy),
                                           jnp.asarray(year))
        _leaves_equal(lp_w, lp_s)
        np.testing.assert_array_equal(np.asarray(lai_w), np.asarray(lai_s))


def test_the_pytree_crosses_a_jit_boundary_as_an_argument():
    """The reason for the split: an outer jit may take the inputs as an ARG.

    A closure over host arrays cannot be passed in; this can, and must give
    the same answer as calling it eagerly.
    """
    gsd = _forest_then_grass()
    pre = precompute_canopy_updater(gsd)

    @jax.jit
    def step(pre_in, theta, doy, year):
        lp, lai = apply_canopy_updater(pre_in, theta, doy, year)
        return lp.LAI, lp.hc, lai

    got = step(pre, _THETA, jnp.asarray(200.0), jnp.asarray(2000.0))
    want = apply_canopy_updater(pre, _THETA, jnp.asarray(200.0),
                                jnp.asarray(2000.0))
    np.testing.assert_allclose(np.asarray(got[0]), np.asarray(want[0].LAI),
                               rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(got[1]), np.asarray(want[0].hc),
                               rtol=0, atol=0)


def test_optional_root_table_is_carried_on_both_paths():
    """With per-PFT root parameters the leaf is present and identical; without
    them it is None on BOTH paths (the model then falls back to its scalars)."""
    gsd = _forest_then_grass()
    cfg = CanopyConfig()
    args = (_THETA, jnp.asarray(200.0), jnp.asarray(2000.0))

    with_w = make_step_land_params_updater(gsd, cfg, pft_root_params=_ROOTS)(*args)[0]
    with_s = apply_canopy_updater(
        precompute_canopy_updater(gsd, pft_root_params=_ROOTS), *args)[0]
    assert with_w.root_depth is not None and with_s.root_depth is not None
    np.testing.assert_array_equal(np.asarray(with_w.root_depth),
                                  np.asarray(with_s.root_depth))

    without_w = make_step_land_params_updater(gsd, cfg)(*args)[0]
    without_s = apply_canopy_updater(precompute_canopy_updater(gsd), *args)[0]
    assert without_w.root_depth is None and without_s.root_depth is None


def test_every_leaf_agrees_on_glacier_bare_and_vegetated_columns():
    """The split must reproduce the wrapper on EVERY leaf with all branches
    live at once -- a transient vegetated column, a glacier column with a
    custom ice albedo, and an uncovered column filled from the bare template.
    Selected-field tests would miss a quietly dropped leaf such as ``rd``
    (codex review)."""
    gsd = _mixed_columns()
    cfg = CanopyConfig()
    theta = jnp.asarray([0.28, 0.05, 0.40])
    glac = (0.7, 0.55)
    wrapper = make_step_land_params_updater(
        gsd, cfg, glacier_alb=glac, pft_root_params=_ROOTS)
    pre = precompute_canopy_updater(gsd, glacier_alb=glac,
                                    pft_root_params=_ROOTS)
    for doy, year in ((15.0, 1850.0), (200.0, 1925.0), (355.0, 2000.0)):
        w = wrapper(theta, jnp.asarray(doy), jnp.asarray(year))
        s_ = apply_canopy_updater(pre, theta, jnp.asarray(doy),
                                  jnp.asarray(year))
        _leaves_equal(w, s_)
    # The branches really are live: ice albedo on the glacier column, no
    # leaves there, and the bare template on the uncovered one.
    lp, _lai = wrapper(theta, jnp.asarray(200.0), jnp.asarray(2000.0))
    np.testing.assert_allclose(float(lp.ALB_VIS[1]), glac[0])
    assert float(lp.LAI[1]) == 0.0
    assert float(lp.LAI[2]) == 0.0
