"""Wrapping several fields in longitude together must give what wrapping them
separately gives.

On a latitude band every device owns the whole circle of longitude, so the
wrap is a local copy and doing it per field costs nothing. On a tile it is a
ring exchange, and doing it per field is why the tiled step sends 28
point-to-point messages where a band sends 13 -- and why its communication
costs 2.67 times the band's while moving several times fewer halo rows.

These gates say the batched wrap moves the same columns, that the operators
which take a pre-wrapped field give the same answer as the ones that wrap for
themselves, and that a wrong pre-wrap is refused rather than used.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from legoesm.grids.operators_latlon_cgrid import (  # noqa: E402
    gradient_x_cgrid, interp_cell_to_uface, pad_lon_cgrid,
    pad_lon_cgrid_many,
)

N_LAT, N_LON = 8, 12


def _fields(seed=0):
    rng = np.random.default_rng(seed)
    return (
        jnp.asarray(rng.standard_normal((N_LAT, N_LON)), dtype=jnp.float32),
        jnp.asarray(rng.standard_normal((N_LAT, N_LON, 4)), dtype=jnp.float32),
        jnp.asarray(rng.standard_normal((N_LAT, N_LON, 1)), dtype=jnp.float32),
    )


def test_batched_wrap_moves_the_same_columns():
    for got, want in zip(pad_lon_cgrid_many(_fields()), _fields()):
        np.testing.assert_array_equal(
            np.asarray(got), np.asarray(pad_lon_cgrid(want, halo=1)))


def test_batched_wrap_keeps_each_field_its_own_shape():
    """Non-vacuity of the gate above for the two-dimensional case: a field
    without a trailing axis must come back without one, not with a length-one
    axis the packing introduced."""
    padded = pad_lon_cgrid_many(_fields())
    assert padded[0].shape == (N_LAT, N_LON + 2)
    assert padded[1].shape == (N_LAT, N_LON + 2, 4)
    assert padded[2].shape == (N_LAT, N_LON + 2, 1)


def test_an_empty_group_is_not_an_error():
    assert pad_lon_cgrid_many(()) == ()


def test_fields_that_cannot_ride_together_are_refused():
    a, b, _ = _fields()
    staggered = jnp.zeros((N_LAT, N_LON + 1), jnp.float32)
    with pytest.raises(ValueError, match="share its .lat, lon. extent"):
        pad_lon_cgrid_many((a, staggered))
    with pytest.raises(ValueError, match="share one dtype"):
        pad_lon_cgrid_many((a, a.astype(jnp.bfloat16)))


def _grid():
    from legoesm.grids.latlon import create_latlon_grid

    return create_latlon_grid(n_lat=N_LAT, n_lon=N_LON)


@pytest.mark.parametrize("nlev", [None, 4])
def test_the_operators_agree_with_and_without_a_pre_wrap(nlev):
    """The whole point: an operator handed a pre-wrapped field must give
    exactly what it gives when it wraps for itself."""
    rng = np.random.default_rng(3)
    shape = (N_LAT, N_LON) if nlev is None else (N_LAT, N_LON, nlev)
    f = jnp.asarray(rng.standard_normal(shape), dtype=jnp.float32)
    (pad,) = pad_lon_cgrid_many((f,))
    grid = _grid()
    np.testing.assert_array_equal(
        np.asarray(interp_cell_to_uface(f, f_pad_lon=pad)),
        np.asarray(interp_cell_to_uface(f)))
    np.testing.assert_array_equal(
        np.asarray(gradient_x_cgrid(f, grid, f_pad_lon=pad)),
        np.asarray(gradient_x_cgrid(f, grid)))


def test_a_wrong_pre_wrap_is_refused_not_used():
    """A pre-wrap from the wrong field is silent: the numbers stay plausible
    and only the seam is wrong. The shape and the dtype are checked."""
    rng = np.random.default_rng(4)
    f = jnp.asarray(rng.standard_normal((N_LAT, N_LON, 4)), dtype=jnp.float32)
    grid = _grid()
    with pytest.raises(ValueError, match="halo-1 longitude wrap"):
        interp_cell_to_uface(f, f_pad_lon=f)
    with pytest.raises(ValueError, match="halo-1 longitude wrap"):
        gradient_x_cgrid(f, grid, f_pad_lon=f)
    (pad,) = pad_lon_cgrid_many((f,))
    with pytest.raises(ValueError, match="dtype"):
        interp_cell_to_uface(f, f_pad_lon=pad.astype(jnp.bfloat16))
