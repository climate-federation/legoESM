"""Full 2-D halo pad, wall-pole case (regular lat-lon / closed ocean) —
SINGLE-PROCESS pytest lane (no MPI): forward pad, AD grad finiteness,
and the two guard raises (fold/tripole + halo-exceeds-block).

The MPI parity + reverse-mode-AD-through-the-sendrecv-VJP checks live in
``tests/distributed/test_latlon_2d_pad_wall_mpi.py`` (pytest-collected,
size-adaptive, run under ``mpirun -np {2,3,6}`` and wired into the MPI CI
workflow) — codex re-review 2026-06-13 flagged that an ``__main__``-only
MPI check is never collected, so an AD regression could slip through.

pole_bc='wall': pole rows fill a constant wall, interior cuts sendrecv,
E/W ring — deadlock-free (all ranks post their interior sendrecv) and
AD-safe (sendrecv-VJP both axes, no transpose).  fold/tripole raise.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.parallel.latlon_mpi import (
    make_latlon_2d_layout,
    pad_halo_latlon_2d,
)

SV, NV = -1.0, -2.0   # distinct non-zero walls to catch confusion


def _serial_ref(g, halo):
    """lat-wall (axis0) THEN lon-periodic (axis1) — matches the 2-D pad's
    N/S-then-E/W order (wall rows are constant so their lon-wrap is the
    same constant => corners = wall)."""
    n_lon = g.shape[1]
    sw = np.full((halo, n_lon) + g.shape[2:], SV, g.dtype)
    nw = np.full((halo, n_lon) + g.shape[2:], NV, g.dtype)
    latw = np.concatenate([sw, g, nw], axis=0)
    pads = [(0, 0)] * latw.ndim
    pads[1] = (halo, halo)
    return np.pad(latw, pads, mode="wrap")


def test_single_proc_wall_pad():
    g = np.arange(4 * 6).reshape(4, 6).astype(float)
    L = make_latlon_2d_layout(0, 1, 1, 4, 6)   # 1x1: both poles local
    out = np.asarray(pad_halo_latlon_2d(
        jnp.asarray(g), L, halo=1, south_value=SV, north_value=NV))
    np.testing.assert_array_equal(out, _serial_ref(g, 1))


def test_fold_tripole_raise():
    L = make_latlon_2d_layout(0, 1, 1, 4, 6)
    for bc in ("fold", "tripole"):
        with pytest.raises(NotImplementedError, match="transpose"):
            pad_halo_latlon_2d(jnp.ones((4, 6)), L, halo=1, pole_bc=bc)


def test_halo_exceeds_block_raises():
    """codex MAJOR 2026-06-13: halo > smallest local block would mismatch
    a neighbour's send/recv (MPI abort/hang).  Guard raises BEFORE any
    MPI, so a middle-rank layout (proc_lat=3, both neighbours interior)
    raises serially.  rank 1 of a 3x1 grid over n_lat=6 owns 2 rows;
    halo=3 must not be accepted."""
    L = make_latlon_2d_layout(1, 3, 1, 6, 4)   # middle rank, 2 rows owned
    with pytest.raises(ValueError, match="exceeds the smallest local"):
        pad_halo_latlon_2d(jnp.ones((2, 4)), L, halo=3,
                           south_value=SV, north_value=NV)


def test_single_proc_grad_finite():
    g = jnp.asarray(np.linspace(0, 1, 4 * 6).reshape(4, 6))
    L = make_latlon_2d_layout(0, 1, 1, 4, 6)

    def loss(x):
        return jnp.sum(pad_halo_latlon_2d(
            x, L, halo=1, south_value=SV, north_value=NV) ** 2)

    grd = jax.grad(loss)(g)
    assert np.all(np.isfinite(np.asarray(grd)))
