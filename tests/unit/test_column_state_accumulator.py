"""Unit tests for :mod:`legoesm.training.column_state_accumulator`.

Time-mean accumulation of a ColumnState for climatological AMIP/CMIP-vs-ERA5
comparison: identity mean, arithmetic mean (incl. optional fields), the
optional-field structure guard, lax.scan-carry usage, the empty-case guard, and
differentiability.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.training.column_state_accumulator import (  # noqa: E402
    accumulate_column_state,
    init_column_accumulator,
    mean_column_state,
    time_mean_column_states,
)
from legoesm.training.compare_reanalysis import ColumnState  # noqa: E402


def _state(scale, *, nlat=3, nlon=4, nlev=5, with_optional=True):
    shp = (nlat, nlon, nlev)
    sfc = (nlat, nlon)
    return ColumnState(
        T=jnp.full(shp, 280.0 + scale),
        q_v=jnp.full(shp, 5e-3 + 1e-4 * scale),
        u=jnp.full(shp, scale),
        v=jnp.full(shp, -scale),
        p_s=jnp.full(sfc, 1.0e5 + scale),
        precip_mm_day=jnp.full(sfc, 2.0 + scale) if with_optional else None,
        sst_K=jnp.full(sfc, 290.0 + scale) if with_optional else None,
    )


def test_mean_of_identical_states_is_that_state():
    s = _state(3.0)
    mean = time_mean_column_states([s, s, s])
    for name in ("T", "q_v", "u", "v", "p_s", "precip_mm_day", "sst_K"):
        np.testing.assert_allclose(
            np.asarray(getattr(mean, name)), np.asarray(getattr(s, name)), rtol=1e-12)


def test_arithmetic_mean_including_optional_fields():
    a, b = _state(0.0), _state(4.0)
    mean = time_mean_column_states([a, b])
    for name in ("T", "q_v", "u", "v", "p_s", "precip_mm_day", "sst_K"):
        expected = 0.5 * (np.asarray(getattr(a, name)) + np.asarray(getattr(b, name)))
        np.testing.assert_allclose(np.asarray(getattr(mean, name)), expected, rtol=1e-12)


def test_accumulator_sums_in_float64_so_a_float32_state_does_not_drift():
    """Under x64 the running sums are FLOAT64 (iter 259), so a float32 model state
    (an FV dycore — x64 loop + float32 dynamics) time-meaned over a long climatology
    window does NOT lose its low bits.  Demonstrated with a base whose 0.5 fractional
    part a float32 running sum DROPS once the partial sum exceeds float32 resolution —
    the spurious time-mean bias vs the float64 ERA5 reference that the promotion avoids."""
    val = 1.0e6 + 0.5            # float32 holds it (ulp ~0.12) but its N-fold sum cannot
    s32 = ColumnState(
        T=jnp.full((1, 1, 1), val, dtype=jnp.float32),
        q_v=jnp.full((1, 1, 1), 5e-3, dtype=jnp.float32),
        u=jnp.zeros((1, 1, 1), dtype=jnp.float32),
        v=jnp.zeros((1, 1, 1), dtype=jnp.float32),
        p_s=jnp.full((1, 1), 1.0e5, dtype=jnp.float32))
    acc = init_column_accumulator(s32)
    assert acc.sum_state.T.dtype == jnp.float64          # the running sum is promoted
    n = 256
    for _ in range(n):
        acc = accumulate_column_state(acc, s32)
    mean = float(np.asarray(mean_column_state(acc).T).reshape(-1)[0])
    assert mean == pytest.approx(float(np.float32(val)), abs=1e-3)  # float64 keeps the 0.5
    # CONTRAST: a naive float32 running sum drops the 0.5 (partial sum >> float32 ulp),
    # biasing the mean by ~0.5 — exactly the drift the float64 accumulation avoids.
    p32 = np.float32(0.0)
    for _ in range(n):
        p32 = np.float32(p32 + np.float32(val))
    assert abs(float(p32) / n - float(np.float32(val))) > 0.1


def test_optional_field_preserved_when_absent():
    a, b = _state(1.0, with_optional=False), _state(3.0, with_optional=False)
    mean = time_mean_column_states([a, b])
    assert mean.precip_mm_day is None and mean.sst_K is None
    np.testing.assert_allclose(np.asarray(mean.T), 282.0, rtol=1e-12)


def _mpas_state(scale, *, n_cells=6, n_edges=15, nlev=5):
    """An MPAS-like ColumnState carrying the optional native u_edge (nEdges,nlev)
    — a DIFFERENT cardinality from the (nCells,nlev) cell fields."""
    return ColumnState(
        T=jnp.full((n_cells, nlev), 280.0 + scale),
        q_v=jnp.full((n_cells, nlev), 5e-3),
        u=jnp.full((n_cells, nlev), scale),
        v=jnp.full((n_cells, nlev), -scale),
        p_s=jnp.full((n_cells,), 1.0e5),
        u_edge=jnp.full((n_edges, nlev), 3.0 + scale),
    )


def test_accumulator_means_optional_u_edge_array():
    """The native MPAS u_edge (nEdges cardinality) is meaned like any present leaf
    — the accumulator's generic tree.map handles the off-cardinality field."""
    a, b = _mpas_state(0.0), _mpas_state(4.0)
    mean = time_mean_column_states([a, b])
    assert mean.u_edge is not None and mean.u_edge.shape == (15, 5)
    np.testing.assert_allclose(np.asarray(mean.u_edge), 5.0, rtol=1e-12)   # (3+7)/2
    np.testing.assert_allclose(np.asarray(mean.T), 282.0, rtol=1e-12)


def test_accumulator_preserves_u_edge_none_for_cell_grids():
    """A cell-wind (lat-lon/cubed) state has u_edge=None ⇒ preserved (not summed)."""
    a, b = _state(1.0), _state(3.0)
    assert a.u_edge is None
    mean = time_mean_column_states([a, b])
    assert mean.u_edge is None


def test_u_edge_presence_mismatch_raises():
    """Mixing an MPAS (u_edge present) and a cell-wind (u_edge=None) state in one
    accumulation is a structure mismatch — raised loudly (same as precip/sst)."""
    acc = init_column_accumulator(_mpas_state(0.0))
    cell = ColumnState(
        T=jnp.zeros((6, 5)), q_v=jnp.zeros((6, 5)), u=jnp.zeros((6, 5)),
        v=jnp.zeros((6, 5)), p_s=jnp.zeros((6,)))          # u_edge=None
    with pytest.raises(ValueError, match="structure mismatch"):
        accumulate_column_state(acc, cell)


def test_optional_field_structure_mismatch_raises_both_directions():
    """Accumulating a state whose optional fields differ in presence is a bug —
    the explicit treedef check raises the SAME deterministic ValueError in BOTH
    directions (present-acc/absent-sample and absent-acc/present-sample), not the
    obscure direction-dependent jax.tree.map error."""
    acc_with = init_column_accumulator(_state(0.0, with_optional=True))
    with pytest.raises(ValueError, match="structure mismatch"):
        accumulate_column_state(acc_with, _state(1.0, with_optional=False))

    acc_without = init_column_accumulator(_state(0.0, with_optional=False))
    with pytest.raises(ValueError, match="structure mismatch"):
        accumulate_column_state(acc_without, _state(1.0, with_optional=True))


def test_count_increments():
    acc = init_column_accumulator(_state(0.0))
    assert int(acc.count) == 0
    acc = accumulate_column_state(acc, _state(1.0))
    acc = accumulate_column_state(acc, _state(2.0))
    assert int(acc.count) == 2


def test_empty_sequence_raises():
    with pytest.raises(ValueError, match="at least one ColumnState"):
        time_mean_column_states([])


def test_scan_carry_matches_manual_mean():
    """The accumulator is a valid lax.scan carry: accumulating stacked states in
    a scan gives the same mean as the host-side fold."""
    states = [_state(float(i)) for i in range(6)]
    stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *states)  # leading time axis

    def step(acc, one):
        return accumulate_column_state(acc, one), None

    acc0 = init_column_accumulator(states[0])
    acc_final, _ = jax.lax.scan(step, acc0, stacked)
    scan_mean = mean_column_state(acc_final)
    fold_mean = time_mean_column_states(states)
    for name in ("T", "q_v", "u", "v", "p_s", "precip_mm_day", "sst_K"):
        np.testing.assert_allclose(
            np.asarray(getattr(scan_mean, name)),
            np.asarray(getattr(fold_mean, name)), rtol=1e-12)
    assert int(acc_final.count) == 6


def test_mean_empty_accumulator_guarded_not_nan():
    """An un-advanced scan carry (count=0) must not produce 0/0=NaN — the
    maximum(count,1) guard returns the zero template."""
    acc = init_column_accumulator(_state(0.0))
    mean = mean_column_state(acc)
    assert bool(jnp.all(jnp.isfinite(mean.T)))
    np.testing.assert_allclose(np.asarray(mean.T), 0.0)


def test_differentiable_through_mean():
    """grad of a scalar loss on the time mean flows back to the input states."""
    a, b = _state(1.0), _state(2.0)

    def loss(ta, tb):
        sa = a._replace(T=ta)
        sb = b._replace(T=tb)
        mean = time_mean_column_states([sa, sb])
        return jnp.sum(mean.T ** 2)

    ga, gb = jax.grad(loss, argnums=(0, 1))(a.T, b.T)
    assert bool(jnp.all(jnp.isfinite(ga)))
    assert bool(jnp.all(jnp.isfinite(gb)))
    # d/da_T sum(((a_T+b_T)/2)^2) = sum(2*mean*0.5) = sum(mean); finite + nonzero.
    assert float(jnp.sum(jnp.abs(ga))) > 0.0
