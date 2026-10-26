"""The packed-land-column contract (gather/scatter helpers, 2026-08-25).

The coupled driver solves the multilayer land tile only on the f_land > 0
columns; these tests pin the convention the driver relies on:
- only leaves whose LEADING axis is the full column count are packed,
- scatter restores land columns exactly and leaves ocean columns untouched,
- per-cell outputs scatter with finite (zero) fills, never NaN.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.land.multilayer_land import (
    gather_land_columns, scatter_land_columns, scatter_cells)

NCOL = 7
IDX = jnp.asarray([1, 3, 4])   # the "land" columns


class _State(NamedTuple):
    T_soil: jnp.ndarray      # (ncol, nz) — packed
    snow: jnp.ndarray        # (ncol,)    — packed
    pft_table: jnp.ndarray   # (3, 5)     — NOT packed (no leading ncol)
    scalar: float            # static leaf


def _mk_state():
    return _State(
        T_soil=jnp.arange(NCOL * 2, dtype=jnp.float64).reshape(NCOL, 2),
        snow=jnp.arange(NCOL, dtype=jnp.float64),
        pft_table=jnp.ones((3, 5)),
        scalar=2.5,
    )


def test_gather_packs_only_leading_ncol_leaves():
    packed = gather_land_columns(_mk_state(), IDX, NCOL)
    assert packed.T_soil.shape == (3, 2)
    assert packed.snow.shape == (3,)
    assert packed.pft_table.shape == (3, 5)      # untouched
    np.testing.assert_array_equal(packed.snow, np.asarray(IDX, float))


def test_scatter_roundtrip_land_exact_ocean_frozen():
    full = _mk_state()
    packed = gather_land_columns(full, IDX, NCOL)
    advanced = packed._replace(T_soil=packed.T_soil + 100.0,
                               snow=packed.snow + 100.0)
    out = scatter_land_columns(full, advanced, IDX, NCOL)
    idx = np.asarray(IDX)
    ocean = np.setdiff1d(np.arange(NCOL), idx)
    np.testing.assert_array_equal(
        np.asarray(out.T_soil)[idx], np.asarray(full.T_soil)[idx] + 100.0)
    np.testing.assert_array_equal(       # ocean columns byte-frozen
        np.asarray(out.T_soil)[ocean], np.asarray(full.T_soil)[ocean])
    np.testing.assert_array_equal(
        np.asarray(out.snow)[ocean], np.asarray(full.snow)[ocean])


def test_scatter_cells_zero_fill_never_nan():
    v = jnp.asarray([300.0, 301.0, 302.0])
    out = np.asarray(scatter_cells(v, IDX, NCOL))
    assert out.shape == (NCOL,)
    assert np.isfinite(out).all()
    np.testing.assert_array_equal(out[np.asarray(IDX)], np.asarray(v))
    assert (out[np.setdiff1d(np.arange(NCOL), np.asarray(IDX))] == 0.0).all()


def _forcing(ncol, seed=0):
    from legoesm.core.coupling_fields import AtmToSurface
    rng = np.random.default_rng(seed)

    def f(lo, hi):
        return jnp.asarray(rng.uniform(lo, hi, ncol))
    return AtmToSurface(
        sw_down=f(0, 600), lw_down=f(250, 400),
        precip_total=f(0, 1e-4), precip_snow=jnp.zeros(ncol),
        T_lowest=f(270, 300), q_lowest=f(1e-3, 1.5e-2),
        u_lowest=f(-5, 5), v_lowest=f(-5, 5),
        p_lowest=jnp.full(ncol, 95000.0), p_surface=jnp.full(ncol, 1e5),
        rho_lowest=jnp.full(ncol, 1.2), cos_zenith=f(0, 1),
        co2_ppmv=jnp.full(ncol, 400.0),
        has_radiation=jnp.ones(ncol), has_precipitation=jnp.ones(ncol),
    )


def test_packed_step_bitwise_equals_unpacked_on_land_columns():
    """The whole acceleration claim: per-column physics is column-local, so
    solving only the packed columns is BITWISE identical on those columns to
    the full-grid solve — and poisoning the discarded columns with NaN must
    not leak into them (the two reviewer-required checks, 2026-08-26)."""
    from legoesm.land.config import MultiLayerLandConfig
    from legoesm.land.surface_scheme import TwoLeafCanopyConfig
    from legoesm.land.multilayer_land import (
        init_multilayer_land_state, step_multilayer_land,
        gather_land_columns, scatter_land_columns)

    ncol = 6
    idx = jnp.asarray([0, 2, 5])
    config = MultiLayerLandConfig(surface_scheme=TwoLeafCanopyConfig())
    state = init_multilayer_land_state(ncol, config, T_init=288.0)
    forcing = _forcing(ncol)
    lat = jnp.asarray(np.linspace(-30.0, 30.0, ncol))
    kw = dict(U_min=1.0, dt=300.0, doy=40.0)

    full_state, full_resp, _ = step_multilayer_land(
        state, forcing, config, lat=lat, **kw)

    # Poison the columns the packed solve never touches.
    def _poison(x):
        if hasattr(x, "shape") and getattr(x, "ndim", 0) >= 1 \
                and x.shape[0] == ncol:
            mask = jnp.ones((ncol,) + (1,) * (x.ndim - 1), x.dtype)
            keep = mask.at[jnp.asarray([1, 3, 4])].set(jnp.nan)
            return x * keep
        return x
    state_poisoned = jax.tree_util.tree_map(_poison, state)

    packed_state, packed_resp, _ = step_multilayer_land(
        gather_land_columns(state_poisoned, idx, ncol),
        gather_land_columns(forcing, idx, ncol),
        gather_land_columns(config, idx, ncol),
        lat=lat[idx], **kw)

    out_state = scatter_land_columns(
        state_poisoned, packed_state, idx, ncol)

    idx_np = np.asarray(idx)
    for name in ("T_soil", "theta_soil"):
        full = np.asarray(getattr(full_state, name))
        out = np.asarray(getattr(out_state, name))
        np.testing.assert_array_equal(
            out[idx_np], full[idx_np],
            err_msg=f"{name} not bitwise on packed columns")
        assert np.isfinite(out[idx_np]).all()
    for name in ("T_sfc", "shflx", "lhflx", "q_surface"):
        np.testing.assert_array_equal(
            np.asarray(getattr(packed_resp, name)).reshape(-1),
            np.asarray(getattr(full_resp, name)).reshape(-1)[idx_np],
            err_msg=f"resp.{name} not bitwise on packed columns")
