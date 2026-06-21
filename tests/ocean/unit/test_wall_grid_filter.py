"""GH #480: unit tests for the N/S free-slip-wall grid-mode filter
(_bc_wall_grid_filter in ocean_pe_latlon_cgrid). Verifies it is a no-op by
default / for non-WENO schemes, acts ONLY on the wall rows, leaves the interior
bit-identical, and applies the correct 2dx-in-lon high-pass."""
import numpy as np
import jax.numpy as jnp
import pytest
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
    _bc_wall_grid_filter, _WALL_FILTER_ROWS as NW,
)

N_LAT, N_LON, NLEV = 40, 16, 3


def _fields():
    rng = np.random.default_rng(0)
    u = jnp.asarray(rng.standard_normal((N_LAT, N_LON + 1, NLEV)))
    v = jnp.asarray(rng.standard_normal((N_LAT + 1, N_LON, NLEV)))
    du = jnp.zeros_like(u); dv = jnp.zeros_like(v)
    um = jnp.ones_like(u); vm = jnp.ones_like(v)
    return du, dv, u, v, um, vm


def test_noop_when_rate_zero():
    du, dv, u, v, um, vm = _fields()
    du2, dv2 = _bc_wall_grid_filter(du, dv, u, v, um, vm, 0.0, "weno9")
    assert jnp.array_equal(du2, du) and jnp.array_equal(dv2, dv)


def test_noop_for_nonweno_scheme():
    du, dv, u, v, um, vm = _fields()
    du2, dv2 = _bc_wall_grid_filter(du, dv, u, v, um, vm, 1e-3, "vector_invariant")
    assert jnp.array_equal(du2, du) and jnp.array_equal(dv2, dv)


def test_interior_bit_unchanged():
    du, dv, u, v, um, vm = _fields()
    _, dv2 = _bc_wall_grid_filter(du, dv, u, v, um, vm, 1e-3, "weno9")
    # interior v-rows (away from both walls) must be exactly zero (du/dv start at 0)
    nrows = dv2.shape[0]
    assert jnp.all(dv2[NW:nrows - NW] == 0.0)


def test_acts_on_wall_rows_with_correct_highpass():
    du, dv, u, v, um, vm = _fields()
    rate = 1e-3
    _, dv2 = _bc_wall_grid_filter(du, dv, u, v, um, vm, rate, "weno9")
    hp = (2.0 * v - jnp.roll(v, 1, axis=1) - jnp.roll(v, -1, axis=1)) * 0.25
    # south wall (row 0): dv = -rate * hp(v) since dv started at 0 and mask=1
    np.testing.assert_allclose(np.asarray(dv2[0]), np.asarray(-rate * hp[0]), rtol=1e-12, atol=1e-14)
    # north wall (last row) likewise
    np.testing.assert_allclose(np.asarray(dv2[-1]), np.asarray(-rate * hp[-1]), rtol=1e-12, atol=1e-14)
    # and a wall row is actually nonzero
    assert jnp.max(jnp.abs(dv2[0])) > 0.0


def test_masking_zeros_filter_on_land():
    du, dv, u, v, um, vm = _fields()
    vm = vm.at[0].set(0.0)   # south wall row is land
    _, dv2 = _bc_wall_grid_filter(du, dv, u, v, um, vm, 1e-3, "weno9")
    assert jnp.all(dv2[0] == 0.0)


def test_mpi_static_gating_south_end_only(monkeypatch):
    """MPI: a rank whose slice holds only the SOUTH domain end must filter the
    south band ONLY — the north end of its array is a real interior cut."""
    import legoesm.grids.operators_latlon_cgrid as ops
    monkeypatch.setattr(ops, "lat_ends_are_poles", lambda: (True, False))
    du, dv, u, v, um, vm = _fields()
    _, dv2 = _bc_wall_grid_filter(du, dv, u, v, um, vm, 1e-3, "weno9")
    nrows = dv2.shape[0]
    assert jnp.max(jnp.abs(dv2[:NW])) > 0.0          # south band filtered
    assert jnp.all(dv2[nrows - NW:] == 0.0)          # north (interior cut) NOT filtered


def test_mpi_static_gating_interior_rank_no_filter(monkeypatch):
    """MPI: an interior rank (neither domain end) must apply NO wall filter —
    both its array ends are interior cuts."""
    import legoesm.grids.operators_latlon_cgrid as ops
    monkeypatch.setattr(ops, "lat_ends_are_poles", lambda: (False, False))
    du, dv, u, v, um, vm = _fields()
    du2, dv2 = _bc_wall_grid_filter(du, dv, u, v, um, vm, 1e-3, "weno9")
    assert jnp.all(dv2 == 0.0) and jnp.all(du2 == 0.0)
