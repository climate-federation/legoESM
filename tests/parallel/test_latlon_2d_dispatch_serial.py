"""Serial (single-process) lane for the lat-lon 2-D pencil HALO DISPATCH.

The 2-D pad PRIMITIVES are tested in ``test_latlon_2d_pad_wall.py`` (serial)
and ``tests/distributed/test_latlon_2d_pad_wall_mpi.py`` (MPI parity + AD).
This file tests the BACKEND DISPATCH layer that wires those primitives into
the operators: when the global halo backend is ``"mpi"`` with a
:class:`LatLon2DLayout` active, the public ``halo_latlon`` entry points
(``pad_halo_latlon`` & friends, ``pad_with_pole_bc_lat[_multi]``,
``zero_polar_lat_ends``) MUST route to the 2-D wall-pole path — not silently
fall through to the serial pole-fold (the riskiest 2-D bug).

A ``proc=1×1`` layout makes the 2-D path fully LOCAL (both poles local +
single-member lon ring => no mpi4jax), so the routing is checkable without
``mpirun``.  The genuine 2×N distributed routing + AD is covered in
``tests/distributed/test_latlon_2d_dispatch_mpi.py``.

Also pins the loud guards: ``make_latlon_2d_mpi_step`` WIRES a longitude
split (proc_lon>1) for regular / wall-pole grids (operator lon ops route
through the dispatched lon halo; the polar filter through the lat-pencil
transpose) but refuses a TRIPOLAR grid and, with ``use_polar_filter=True``,
an UNEVEN lon split (the allgather needs equal blocks) — before any
mutation.  ``pad_with_pole_bc_lat`` refuses the tripolar fold seam on a
2-D layout (wall-pole benchmark only).
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.halo import set_halo_backend
from legoesm.grids.halo_latlon import (
    pad_halo_latlon,
    pad_halo_latlon_3d,
    pad_halo_latlon_3d_local,
    pad_halo_latlon_local,
    pad_halo_latlon_vector,
    pad_halo_latlon_vector_3d,
    pad_halo_latlon_vector_3d_local,
    pad_halo_latlon_vector_local,
    pad_with_pole_bc_lat,
    pad_with_pole_bc_lat_multi,
    zero_polar_lat_ends,
)
from legoesm.parallel.latlon_mpi import (
    make_latlon_2d_layout,
    make_latlon_2d_mpi_step,
    pad_with_pole_bc_lat_2d,
)

SV, NV = -1.0, -2.0   # distinct non-zero walls to catch S/N confusion
N_LAT, N_LON = 4, 6


@pytest.fixture()
def layout_1x1():
    """1×1 pencil: both poles local, single-member lon ring => the 2-D
    path runs serially (no mpi4jax)."""
    return make_latlon_2d_layout(0, 1, 1, N_LAT, N_LON)


@pytest.fixture(autouse=True)
def _arm_2d_mpi_backend(layout_1x1):
    """Arm the MPI halo backend with the 2-D layout, restore after.

    ``set_halo_backend`` only stores the topology (no mpi4py), and the
    1×1 2-D pad is fully local, so the operators' dispatch fires the 2-D
    branch without any collective."""
    set_halo_backend("mpi", layout_1x1)
    yield
    set_halo_backend("local")


def test_scalar_pad_folds_at_proc_lon1(layout_1x1):
    """``pad_halo_latlon`` (scalar fold family) at proc_lon==1 must reuse the
    band POLE-FOLD (lon full per rank => local 180-deg fold), NOT a wall-zero
    ghost — this is what keeps the global 2-D dispatch safe for every scalar
    caller.  At 1×1 it equals the serial local fold bit-for-bit."""
    g = jnp.asarray(np.arange(N_LAT * N_LON).reshape(N_LAT, N_LON).astype(float))
    out = pad_halo_latlon(g, halo=1)
    np.testing.assert_array_equal(
        np.asarray(out), np.asarray(pad_halo_latlon_local(g, 1)))


def test_vector_pad_folds_at_proc_lon1(layout_1x1):
    """Meridional-vector fold-family pad reuses the band fold WITH the
    sign-flip at proc_lon==1 (== serial vector local fold)."""
    g = jnp.asarray(np.linspace(-1, 1, N_LAT * N_LON).reshape(N_LAT, N_LON))
    out = pad_halo_latlon_vector(g, halo=1)
    np.testing.assert_array_equal(
        np.asarray(out), np.asarray(pad_halo_latlon_vector_local(g, 1)))


def test_scalar_and_vector_3d_pad_fold_at_proc_lon1(layout_1x1):
    """3-D (level axis) scalar + vector fold-family pads reuse the band fold
    at proc_lon==1 (== the serial 3-D local fold)."""
    g = jnp.asarray(
        np.arange(N_LAT * N_LON * 3).reshape(N_LAT, N_LON, 3).astype(float))
    np.testing.assert_array_equal(
        np.asarray(pad_halo_latlon_3d(g, halo=1)),
        np.asarray(pad_halo_latlon_3d_local(g, 1)))
    np.testing.assert_array_equal(
        np.asarray(pad_halo_latlon_vector_3d(g, halo=1)),
        np.asarray(pad_halo_latlon_vector_3d_local(g, 1)))


def test_pad_with_pole_bc_lat_routes_to_2d_latonly(layout_1x1):
    """``pad_with_pole_bc_lat`` routes to the lat-ONLY 2-D wall pad
    (longitude untouched), equal to a plain lat-axis constant pad at 1×1."""
    g = jnp.asarray(np.arange(N_LAT * N_LON).reshape(N_LAT, N_LON).astype(float))
    out = pad_with_pole_bc_lat(g, halo=1, south_value=SV, north_value=NV)
    want = pad_with_pole_bc_lat_2d(
        g, layout_1x1, halo=1, south_value=SV, north_value=NV)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(want))
    # lat-only: lon width is unchanged, lat grew by 2*halo.
    assert out.shape == (N_LAT + 2, N_LON)
    # equal to a plain lat-axis constant pad (the local-backend semantics).
    plain = jnp.pad(g, ((1, 1), (0, 0)),
                    constant_values=((SV, NV), (0, 0)))
    np.testing.assert_array_equal(np.asarray(out), np.asarray(plain))


def test_pad_with_pole_bc_lat_1d_metric_routes_to_2d(layout_1x1):
    """A 1-D lat metric (sin_lat etc.) also routes to the lat-only 2-D
    wall pad (trailing-shape-agnostic reshape)."""
    g = jnp.asarray(np.linspace(-1.0, 1.0, N_LAT))
    out = pad_with_pole_bc_lat(g, halo=1, south_value=SV, north_value=NV)
    np.testing.assert_array_equal(
        np.asarray(out),
        np.asarray(jnp.concatenate([jnp.array([SV]), g, jnp.array([NV])])))


def test_pad_with_pole_bc_lat_multi_routes_per_field(layout_1x1):
    """``pad_with_pole_bc_lat_multi`` on a 2-D layout routes through the
    per-field path (the fused path is band-only) => each field takes the
    lat-only 2-D wall pad."""
    f1 = jnp.asarray(np.arange(N_LAT * N_LON).reshape(N_LAT, N_LON).astype(float))
    f2 = f1 + 100.0
    outs = pad_with_pole_bc_lat_multi(
        (f1, f2), halo=1, south_values=(SV, 0.0), north_values=(NV, 0.0))
    assert len(outs) == 2
    np.testing.assert_array_equal(
        np.asarray(outs[0]),
        np.asarray(pad_with_pole_bc_lat_2d(
            f1, layout_1x1, halo=1, south_value=SV, north_value=NV)))
    np.testing.assert_array_equal(
        np.asarray(outs[1]),
        np.asarray(pad_with_pole_bc_lat_2d(
            f2, layout_1x1, halo=1, south_value=0.0, north_value=0.0)))


def test_zero_polar_lat_ends_2d_zeros_both_poles(layout_1x1):
    """At 1×1 both lat ends are physical poles, so ``zero_polar_lat_ends``
    must zero index 0 and -1 (the 2-D pole-touch test == band test)."""
    field = jnp.asarray(
        1.0 + np.arange(N_LAT * N_LON).reshape(N_LAT, N_LON).astype(float))
    out = np.asarray(zero_polar_lat_ends(field))
    assert np.all(out[0] == 0.0) and np.all(out[-1] == 0.0)
    np.testing.assert_array_equal(out[1:-1], np.asarray(field)[1:-1])


def test_step_refuses_tripolar_longitude_split():
    """``make_latlon_2d_mpi_step`` must refuse proc_lon>1 on a TRIPOLAR grid
    LOUDLY — the curl tripolar-cap fold keeps a local lon roll (the 180° fold
    under a lon split needs the lat-pencil transpose).  Keyed on
    ``fold.is_active`` ALONE (grid-global, so all ranks raise together — NOT
    ``fold_j``, which is rank-local and would deadlock).  Regular / wall-pole
    proc_lon>1 is now WIRED (validated by the 2-D dycore conservation gate
    ``tests/distributed/test_latlon_2d_mpi_step.py``)."""
    from types import SimpleNamespace
    layout_1x2 = make_latlon_2d_layout(0, 1, 2, N_LAT, N_LON)
    # fold active, NO fold_j attr -> the is_active-alone guard must still fire.
    tri = SimpleNamespace(
        grid=SimpleNamespace(fold=SimpleNamespace(is_active=True)),
        config=SimpleNamespace(use_polar_filter=False))
    with pytest.raises(NotImplementedError, match="TRIPOLAR"):
        make_latlon_2d_mpi_step(tri, layout_1x2)


def test_step_refuses_polar_filter_uneven_longitude_split():
    """``make_latlon_2d_mpi_step`` with ``use_polar_filter=True`` now WIRES the
    lon-split filter (via the lat-pencil transpose), but the AD-safe allgather
    needs an EQUAL split — so an UNEVEN proc_lon (n_lon % proc_lon != 0) must
    refuse at the FACTORY, BEFORE any mutation (the halo backend must stay
    'local').  N_LON=6, proc_lon=4 -> 6%4=2 != 0.  (Even splits are exercised
    end-to-end in tests/distributed/test_latlon_2d_polar_filter_mpi.py.)"""
    from types import SimpleNamespace

    from legoesm.grids.halo import get_halo_backend, set_halo_backend
    set_halo_backend("local")
    layout_1x4 = make_latlon_2d_layout(0, 1, 4, N_LAT, N_LON)
    m = SimpleNamespace(
        grid=SimpleNamespace(fold=None),
        config=SimpleNamespace(use_polar_filter=True))
    with pytest.raises(NotImplementedError, match="EQUAL split"):
        make_latlon_2d_mpi_step(m, layout_1x4)
    # No partial mutation: the guard fired before set_halo_backend.
    assert get_halo_backend() == "local"


def test_pad_with_pole_bc_lat_refuses_fold_seam_on_2d(layout_1x1):
    """The tripolar north fold / vector-u seam needs the lat-pencil
    transpose (wall-pole 2-D benchmark excludes it) => fail loud."""
    g = jnp.ones((N_LAT, N_LON))
    with pytest.raises(NotImplementedError, match="transpose"):
        pad_with_pole_bc_lat(g, halo=1, north_fold=True)
    with pytest.raises(NotImplementedError, match="transpose"):
        pad_with_pole_bc_lat(g, halo=1, is_vector_u=True)
