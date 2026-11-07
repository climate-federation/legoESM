"""Unit test for ``ModelDriver._mpas_global_diag`` (batched-allreduce form).

The method owns the MPAS cell-partition MPI diagnostics: owned-cell masked
local reductions, then THREE batched ``Allreduce`` rounds (SUM / MAX / MIN,
with the finite flag riding the MIN batch) plus a cached one-time owned-cell
count.  Run single-rank here (COMM_WORLD size 1: every allreduce is an
identity), which pins the masking + batching + cache algebra; the multi-rank
semantics are the same three MPI ops the mass fixer already exercises under
``tests/distributed/``.
"""

from __future__ import annotations

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("mpi4py")

from legoesm.driver.model_driver import ModelDriver
from legoesm.grids.vertical import create_sigma_coordinate


class _Partition:
    def __init__(self, n_owned_cells):
        self.n_owned_cells = n_owned_cells


class _VLayout:
    def __init__(self, owned_mask_cells, owned_mask_edges, n_owned_cells):
        self.owned_mask_cells = owned_mask_cells
        self.owned_mask_edges = owned_mask_edges
        self.partition = _Partition(n_owned_cells)


def _make_stub(n_cells=6, n_owned=4, n_edges=5, n_owned_edges=3, nlev=3):
    """A bare ModelDriver instance (no __init__) with just the attrs
    ``_mpas_global_diag`` reads (incl. ``sigma`` for the pressure-weighted
    mean T, 2026-07-23)."""
    drv = ModelDriver.__new__(ModelDriver)
    om_c = jnp.asarray(np.arange(n_cells) < n_owned)
    om_e = jnp.asarray(np.arange(n_edges) < n_owned_edges)
    drv._voronoi_layout = _VLayout(om_c, om_e, n_owned)
    drv._mpas_g_n_cells = None
    drv.sigma = create_sigma_coordinate(nlev)
    return drv, np.asarray(om_c), np.asarray(om_e)


def _expected_mean_T(drv, T, ps, om_c):
    """Pressure-weighted owned-cell mean, mirroring the fixed convention."""
    p_half = np.asarray(drv.sigma.pressure_at_half(jnp.asarray(ps)))
    dp = p_half[..., 1:] - p_half[..., :-1]
    own = om_c == 1
    return (T[own] * dp[own]).sum() / dp[own].sum()


def test_owned_masked_means_and_extrema_single_rank():
    drv, om_c, om_e = _make_stub()
    n_cells, nlev, n_edges = 6, 3, 5
    rng = np.random.default_rng(0)
    T = rng.uniform(250.0, 300.0, size=(n_cells, nlev))
    ps = rng.uniform(9.0e4, 1.05e5, size=(n_cells,))
    u = rng.uniform(-30.0, 30.0, size=(n_edges, nlev))
    # Halo entries carry extreme sentinels — they must NOT leak into any
    # returned statistic (the owned mask is the point of this method).
    T[om_c == 0] = 9999.0
    ps[om_c == 0] = 9.0e9
    u[om_e == 0] = -1.0e6
    cwv = rng.uniform(10.0, 60.0, size=(n_cells,))
    cwv[om_c == 0] = 1.0e9

    mean_T, mean_ps, max_u, T_min, T_max, finite, cwv_mean, ni_max = (
        drv._mpas_global_diag(jnp.asarray(T), jnp.asarray(ps),
                              jnp.asarray(u), jnp.asarray(cwv)))

    T_own = T[om_c == 1]
    assert mean_T == pytest.approx(_expected_mean_T(drv, T, ps, om_c))
    assert mean_ps == pytest.approx(ps[om_c == 1].mean())
    assert max_u == pytest.approx(np.abs(u[om_e == 1]).max())
    assert T_min == pytest.approx(T_own.min())
    assert T_max == pytest.approx(T_own.max())
    assert finite is True
    assert cwv_mean == pytest.approx(cwv[om_c == 1].mean())
    assert np.isnan(ni_max)          # no N_i field passed


def test_ni_max_owned_only_and_nan_is_visible():
    drv, om_c, _ = _make_stub(nlev=2)
    args = (jnp.full((6, 2), 280.0), jnp.full((6,), 1.0e5),
            jnp.zeros((5, 2)), None)
    ni = np.full((6, 2), 1.0e4)
    ni[0, 1] = 3.0e6                       # owned maximum
    ni[om_c == 0] = 9.0e12                 # halo sentinel must not leak
    assert drv._mpas_global_diag(*args, ni_field=jnp.asarray(ni))[7] == (
        pytest.approx(3.0e6))
    ni[1, 0] = np.nan                      # owned NaN -> +inf, not hidden
    assert drv._mpas_global_diag(*args, ni_field=jnp.asarray(ni))[7] == np.inf


def test_finite_flag_owned_only():
    drv, om_c, _ = _make_stub(nlev=2)
    T = np.full((6, 2), 280.0)
    ps = np.full((6,), 1.0e5)
    u = np.zeros((5, 2))
    # NaN in a HALO cell only: owned field is finite -> flag stays True,
    # and the pressure-weighted mean must not be poisoned through the
    # T*dp product (codex F-B6: the where() wraps the product).
    T_halo_nan = T.copy()
    T_halo_nan[int(np.argmax(om_c == 0)), 0] = np.nan
    mean_T_h, *_, finite, _, _ = drv._mpas_global_diag(
        jnp.asarray(T_halo_nan), jnp.asarray(ps), jnp.asarray(u), None)
    assert finite is True
    assert np.isfinite(mean_T_h) and mean_T_h == pytest.approx(280.0)
    # Non-finite HALO p_s must not leak either (0*NaN trap).
    ps_halo_nan = ps.copy()
    ps_halo_nan[int(np.argmax(om_c == 0))] = np.nan
    mean_T_p, *_, finite_p, _, _ = drv._mpas_global_diag(
        jnp.asarray(T), jnp.asarray(ps_halo_nan), jnp.asarray(u), None)
    assert finite_p is True
    assert np.isfinite(mean_T_p) and mean_T_p == pytest.approx(280.0)
    # NaN in an OWNED cell -> False.
    T_owned_nan = T.copy()
    T_owned_nan[0, 0] = np.nan
    *_, finite, _, _ = drv._mpas_global_diag(
        jnp.asarray(T_owned_nan), jnp.asarray(ps), jnp.asarray(u), None)
    assert finite is False


def test_cwv_none_returns_nan():
    drv, _, _ = _make_stub(nlev=2)
    out = drv._mpas_global_diag(
        jnp.full((6, 2), 280.0), jnp.full((6,), 1.0e5),
        jnp.zeros((5, 2)), None)
    assert np.isnan(out[6]) and np.isnan(out[7])


def test_n_cells_cache_populated_once():
    drv, _, _ = _make_stub(n_owned=4, nlev=2)
    args = (jnp.full((6, 2), 280.0), jnp.full((6,), 1.0e5),
            jnp.zeros((5, 2)), None)
    assert drv._mpas_g_n_cells is None
    mean_T_1, *_ = drv._mpas_global_diag(*args)
    assert drv._mpas_g_n_cells == 4
    # Mutate the partition count: the CACHED global count must keep being
    # used (the count is partition-static by contract).
    drv._voronoi_layout.partition.n_owned_cells = 999
    mean_T_2, *_ = drv._mpas_global_diag(*args)
    assert drv._mpas_g_n_cells == 4
    assert mean_T_2 == pytest.approx(mean_T_1)
