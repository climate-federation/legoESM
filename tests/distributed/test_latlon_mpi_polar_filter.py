"""MPI bit-equivalence for the lat-lon Fourier polar filter (Stage 3-E).

The polar filter (``legoesm.grids.polar_filter``) is a lon-only FFT
applied independently at each latitude.  Under lat-band MPI:

* the band-decomposition splits the lat axis but every rank owns the
  full lon axis →  ``jnp.fft.rfft(..., axis=-1)`` runs locally;
* the filter mask depends on ``grid.lat`` and ``grid.cos_lat`` which
  are already rank-local slices of the global metric (Stage 3-B).

Therefore the MPI invariant the filter must satisfy is purely a
slice-equivalence property — *not* a halo-exchange property:

    global_mask[s:e]               == rank-local mask on band [s:e)
    fourier_filter(global_f)[s:e]  == rank-local fourier_filter(f[s:e])

These tests pin both equalities.  A regression of the mask-construction
slicing (e.g. computing ``cos_lat`` from a global axis instead of the
band's local lat array) would break both assertions.

Stage 3-E gates the production 100-y AMIP at 1° lat-lon FV: without
the filter the pole-cell CFL clamps ``dt`` to ~5 s and 100 yr ≈ 600 B
steps which is infeasible.  With the filter the equatorial CFL gives
``dt`` ~ 600 s and 100 yr ≈ 5 M steps — tractable on a chained
72-h SLURM budget.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

mpi4py = pytest.importorskip("mpi4py")
from mpi4py import MPI  # noqa: E402

from legoesm import constants  # noqa: E402
from legoesm.grids.latlon import create_latlon_grid  # noqa: E402
from legoesm.grids.polar_filter import (  # noqa: E402
    compute_polar_filter_mask,
    fourier_filter,
    fourier_filter_3d,
)
from legoesm.parallel.latlon_mpi import (  # noqa: E402
    make_latlon_band_layout,
)


# Use a resolution where the polar-CFL effect is non-trivial: at
# n_lat=24 the pole-cell dx is ~30x smaller than the equatorial dx,
# so the filter mask zeros out a substantial fraction of the wave-
# numbers near the poles — any mask-construction bug shows up clearly.
N_LAT = 24
N_LON = 48
NLEV = 4
DT = 600.0  # s — large enough that the filter is non-trivial near poles
MAX_WAVE_SPEED = 300.0
CUTOFF_LAT_DEG = 60.0


@pytest.fixture(scope="module")
def comm():
    return MPI.COMM_WORLD


@pytest.fixture(scope="module")
def layout(comm):
    return make_latlon_band_layout(
        rank=comm.Get_rank(),
        n_ranks=comm.Get_size(),
        n_lat=N_LAT,
        n_lon=N_LON,
    )


@pytest.fixture(scope="module")
def global_grid():
    return create_latlon_grid(
        n_lat=N_LAT, radius=constants.R_earth, omega=constants.Omega,
    )


@pytest.fixture(scope="module")
def band_grid(global_grid, layout):
    """Rank-local grid produced by the same slicing pattern
    ``ModelDriver._create_grid`` uses (Stage 3-B)."""
    s, e = layout.lat_start, layout.lat_end
    band_area = global_grid.area[s:e, :]
    return global_grid._replace(
        n_lat=layout.n_lat_local,
        lat=global_grid.lat[s:e],
        lat2d=global_grid.lat2d[s:e, :],
        lon2d=global_grid.lon2d[s:e, :],
        cos_lat=global_grid.cos_lat[s:e],
        sin_lat=global_grid.sin_lat[s:e],
        dy=global_grid.dy[s:e],
        f=global_grid.f[s:e, :],
        dx=global_grid.dx[s:e, :],
        area=band_area,
        total_area=jnp.sum(band_area),
    )


def test_polar_filter_mask_matches_global_slice(global_grid, band_grid, layout):
    """Rank-local mask equals the global mask sliced to this band.

    If ``compute_polar_filter_mask`` accidentally reads off the parent
    grid's lat axis (or recomputes ``cos_lat`` from the wrong index),
    the rank-local mask would diverge from ``global_mask[s:e]`` and
    every subsequent ``fourier_filter`` call would silently disagree
    with the serial reference.
    """
    global_mask = compute_polar_filter_mask(
        global_grid, dt=DT,
        max_wave_speed=MAX_WAVE_SPEED,
        cutoff_lat_deg=CUTOFF_LAT_DEG,
    )
    band_mask = compute_polar_filter_mask(
        band_grid, dt=DT,
        max_wave_speed=MAX_WAVE_SPEED,
        cutoff_lat_deg=CUTOFF_LAT_DEG,
    )
    s, e = layout.lat_start, layout.lat_end
    np.testing.assert_array_equal(
        np.asarray(band_mask), np.asarray(global_mask[s:e]),
        err_msg=(
            f"Rank {layout.rank} polar-filter mask diverges from "
            f"global_mask[{s}:{e}].  Mask construction is not "
            "slice-equivariant under band decomposition."
        ),
    )


def test_polar_filter_2d_matches_global_slice(global_grid, band_grid, layout):
    """``fourier_filter(f)[s:e] == fourier_filter(f[s:e])`` on rank's band.

    The 2-D filter is a per-row independent FFT, so any
    slice-equivariance violation here would also break the model's
    per-step filter call.
    """
    rng = np.random.default_rng(seed=2024)
    f_global = jnp.asarray(
        rng.standard_normal((N_LAT, N_LON))
    )
    s, e = layout.lat_start, layout.lat_end

    global_mask = compute_polar_filter_mask(
        global_grid, dt=DT,
        max_wave_speed=MAX_WAVE_SPEED,
        cutoff_lat_deg=CUTOFF_LAT_DEG,
    )
    band_mask = compute_polar_filter_mask(
        band_grid, dt=DT,
        max_wave_speed=MAX_WAVE_SPEED,
        cutoff_lat_deg=CUTOFF_LAT_DEG,
    )

    global_filtered = fourier_filter(f_global, global_grid, global_mask)
    band_filtered = fourier_filter(
        f_global[s:e], band_grid, band_mask,
    )

    np.testing.assert_allclose(
        np.asarray(band_filtered), np.asarray(global_filtered[s:e]),
        rtol=1.0e-10, atol=1.0e-12,
        err_msg=(
            f"Rank {layout.rank} 2-D filter output diverges from "
            f"global_filtered[{s}:{e}].  Either the mask or the FFT "
            "is not slice-equivariant."
        ),
    )


def test_polar_filter_3d_matches_global_slice(global_grid, band_grid, layout):
    """3-D variant of the above — important because the dynamics calls
    ``fourier_filter_3d`` on the temperature tendency every step."""
    rng = np.random.default_rng(seed=2025)
    f_global = jnp.asarray(
        rng.standard_normal((N_LAT, N_LON, NLEV))
    )
    s, e = layout.lat_start, layout.lat_end

    global_mask = compute_polar_filter_mask(
        global_grid, dt=DT,
        max_wave_speed=MAX_WAVE_SPEED,
        cutoff_lat_deg=CUTOFF_LAT_DEG,
    )
    band_mask = compute_polar_filter_mask(
        band_grid, dt=DT,
        max_wave_speed=MAX_WAVE_SPEED,
        cutoff_lat_deg=CUTOFF_LAT_DEG,
    )

    global_filtered = fourier_filter_3d(f_global, global_grid, global_mask)
    band_filtered = fourier_filter_3d(
        f_global[s:e], band_grid, band_mask,
    )

    np.testing.assert_allclose(
        np.asarray(band_filtered), np.asarray(global_filtered[s:e]),
        rtol=1.0e-10, atol=1.0e-12,
        err_msg=(
            f"Rank {layout.rank} 3-D filter output diverges from "
            f"global_filtered[{s}:{e}]."
        ),
    )


def test_polar_filter_mask_v_face_matches_global_slice(
    global_grid, band_grid, layout,
):
    """v-face mask (shape (n_lat+1, n_freq)) equals the global v-face
    mask sliced to this rank's band ``[lat_start, lat_end+1)``.

    Codex review Stage 3-E round 2 BLOCK #1 caught that using the
    cell-centered mask on v-face indices admits k modes the v-face
    CFL forbids near the poles (half-cell lat offset → ~0.5° error,
    which at n_lat=180 allows ``k=3`` where the actual v-face CFL
    allows only ``k=2``).  The v-face mask uses ``cos_lat_v`` and the
    half-cell-offset lat-interface coordinates instead.
    """
    global_mask_v = compute_polar_filter_mask(
        global_grid, dt=DT,
        max_wave_speed=MAX_WAVE_SPEED,
        cutoff_lat_deg=CUTOFF_LAT_DEG,
        is_v_face=True,
    )
    band_mask_v = compute_polar_filter_mask(
        band_grid, dt=DT,
        max_wave_speed=MAX_WAVE_SPEED,
        cutoff_lat_deg=CUTOFF_LAT_DEG,
        is_v_face=True,
    )
    s, e = layout.lat_start, layout.lat_end
    # v-face mask spans [s, e+1] rows for this rank's band.
    np.testing.assert_array_equal(
        np.asarray(band_mask_v), np.asarray(global_mask_v[s:e + 1]),
        err_msg=(
            f"Rank {layout.rank} v-face polar-filter mask diverges "
            f"from global_mask_v[{s}:{e + 1}].  v-face mask "
            "construction is not slice-equivariant under band "
            "decomposition."
        ),
    )


def test_polar_filter_mask_is_global_invariant_across_ranks(
    global_grid, band_grid, layout, comm,
):
    """Different ranks must compute mask values that agree with one
    rank-0-built global reference.

    Distinct from the per-rank slice-equivariance check: this gathers
    every band's mask back to rank 0 and asserts the concatenation
    equals the global-computed mask.  A bug that affects a single
    rank's slice would still slip past the per-rank test if every
    rank computes the SAME wrong mask.  This test catches that.
    """
    if comm.Get_size() == 1:
        pytest.skip("Single-rank: the concat is trivial")

    band_mask = compute_polar_filter_mask(
        band_grid, dt=DT,
        max_wave_speed=MAX_WAVE_SPEED,
        cutoff_lat_deg=CUTOFF_LAT_DEG,
    )

    gathered = comm.gather(np.asarray(band_mask), root=0)
    if comm.Get_rank() == 0:
        concat = np.concatenate(gathered, axis=0)
        global_mask = compute_polar_filter_mask(
            global_grid, dt=DT,
            max_wave_speed=MAX_WAVE_SPEED,
            cutoff_lat_deg=CUTOFF_LAT_DEG,
        )
        np.testing.assert_array_equal(
            concat, np.asarray(global_mask),
            err_msg=(
                "Concatenated band masks do not equal the global mask. "
                "Some rank is computing its band mask against an "
                "unexpected grid metric."
            ),
        )
