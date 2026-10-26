"""Pack/split/reshape correctness of ``halo_y_packed``.

The MPI transport itself is covered by the distributed parity tests
(``tests/distributed/test_pseudo_incompressible_*_mpi.py`` under
``mpirun``). Here the ring exchange is replaced by a periodic
``jnp.pad(mode='wrap')`` stand-in (what a single rank's ring reduces
to), which isolates the NEW logic: flatten → concat → one exchange →
split → reshape, for fields with heterogeneous trailing shapes.
"""

from __future__ import annotations

import pytest

import jax.numpy as jnp
import numpy as np
from legoesm.atmosphere._future import (
    pseudo_incompressible_poisson_mpi as pmpi,
)


def _fake_halo_y(field, halo, comm):
    return jnp.pad(
        field, [(halo, halo)] + [(0, 0)] * (field.ndim - 1), mode="wrap",
    )


def test_packed_matches_per_field(monkeypatch):
    monkeypatch.setattr(pmpi, "halo_y", _fake_halo_y)
    rng = np.random.default_rng(0)
    ny, nx, nz, nt = 6, 5, 4, 3
    u = jnp.asarray(rng.normal(size=(ny, nx, nz)))
    w = jnp.asarray(rng.normal(size=(ny, nx, nz + 1)))     # different nz
    tr = jnp.asarray(rng.normal(size=(ny, nx, nz, nt)))    # 4-D block
    for halo in (1, 3):
        packed = pmpi.halo_y_packed((u, w, tr), halo, comm=None)
        for got, f in zip(packed, (u, w, tr)):
            np.testing.assert_array_equal(
                np.asarray(got), np.asarray(_fake_halo_y(f, halo, None)),
            )
            assert got.shape == (f.shape[0] + 2 * halo, *f.shape[1:])


def test_packed_single_field_roundtrip(monkeypatch):
    monkeypatch.setattr(pmpi, "halo_y", _fake_halo_y)
    f = jnp.arange(2 * 3 * 4, dtype=jnp.float64).reshape(2, 3, 4)
    out, = pmpi.halo_y_packed((f,), 1, comm=None)
    np.testing.assert_array_equal(
        np.asarray(out), np.asarray(_fake_halo_y(f, 1, None)),
    )


# Parked module: see its docstring.
pytestmark = pytest.mark.skip(
    reason="parked in _future/: not wired into production (ponytail item poisson_mpi)")
