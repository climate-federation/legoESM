"""Wet-cell-aware latitude-band boundaries (load balance for lat-lon MPI).

Covers ``legoesm.parallel.latlon_mpi.wet_band_boundaries`` (the pure
quantile-on-cumulative-wet-count split), the ``boundaries=`` opt-in on
``make_latlon_band_layout`` (explicit boundaries flow into
``lat_start``/``lat_end`` — the single source of truth every band slicer
reads), and a multi-band slicer reassembly with non-uniform wet-balanced
bands. All host-side — no MPI required.
"""
from __future__ import annotations

import numpy as np
import pytest
from legoesm.parallel.latlon_mpi import (
    LatLonBandLayout,
    make_latlon_band_layout,
    wet_band_boundaries,
)


# ------------------------------------------------------------ pure function --
def _band_wet(w, b):
    return [float(np.sum(w[b[r]:b[r + 1]])) for r in range(len(b) - 1)]


def test_uniform_mask_reproduces_even_split():
    """All-wet rows: wet balancing must give a maximally even row split
    (exact quarters when divisible; row counts differing by <= 1 otherwise —
    the quantile cut may interleave the remainder differently from the
    front-loaded divmod rule, but the balance is identical)."""
    w = np.full(16, 32.0)
    assert wet_band_boundaries(w, 4) == (0, 4, 8, 12, 16)
    b = wet_band_boundaries(np.full(10, 7.0), 4)
    sizes = np.diff(b)
    assert b[0] == 0 and b[-1] == 10
    assert sizes.max() - sizes.min() <= 1, b


def test_land_heavy_south_shifts_boundaries():
    """Rows 0..7 all land (0 wet), rows 8..15 open ocean: the wet split must
    concentrate ranks in the ocean half instead of giving half the ranks
    all-land bands."""
    w = np.concatenate([np.zeros(8), np.full(8, 64.0)])
    b = wet_band_boundaries(w, 4)
    assert b[0] == 0 and b[-1] == 16
    per_band = _band_wet(w, b)
    # Every band carries wet work within one row's worth of the ideal quarter
    # share (the contiguity granularity).
    ideal = float(w.sum()) / 4
    assert max(per_band) - min(per_band) <= 64.0 + 1e-9
    assert all(x >= ideal - 64.0 - 1e-9 for x in per_band)
    # The even ROW split would give bands 0-1 zero wet cells — prove the wet
    # split does better (non-vacuity of the whole feature).
    even = (0, 4, 8, 12, 16)
    assert min(_band_wet(w, even)) == 0.0
    assert min(per_band) > 0.0


def test_min_rows_floor_enforced():
    """Extreme concentration: all wet cells in one row — every band must still
    keep >= min_rows rows (the halo floor), boundaries strictly increasing."""
    w = np.zeros(12)
    w[6] = 100.0
    for min_rows in (1, 2):
        b = wet_band_boundaries(w, 4, min_rows=min_rows)
        sizes = np.diff(b)
        assert (sizes >= min_rows).all(), (b, sizes)
        assert b[0] == 0 and b[-1] == 12
        assert (sizes > 0).all()


def test_all_land_falls_back_to_even_split():
    # Even-split remainder rule (front-loaded sizes [3, 3, 2, 2]).
    b = wet_band_boundaries(np.zeros(10), 4)
    assert b == (0, 3, 6, 8, 10)


def test_validation_errors():
    with pytest.raises(ValueError, match="1-D"):
        wet_band_boundaries(np.ones((4, 4)), 2)
    with pytest.raises(ValueError, match="n_ranks"):
        wet_band_boundaries(np.ones(8), 0)
    with pytest.raises(ValueError, match="min_rows"):
        wet_band_boundaries(np.ones(8), 2, min_rows=0)
    with pytest.raises(ValueError, match="non-negative"):
        wet_band_boundaries(np.array([1.0, -1.0, 2.0]), 2)
    with pytest.raises(ValueError, match="Cannot decompose"):
        wet_band_boundaries(np.ones(3), 2, min_rows=2)


def test_deterministic():
    w = np.abs(np.sin(np.arange(48))) * 10
    assert wet_band_boundaries(w, 6) == wet_band_boundaries(w.copy(), 6)


# ----------------------------------------------------- layout with boundaries --
def test_layout_uses_explicit_boundaries():
    b = (0, 2, 9, 16)
    layouts = [make_latlon_band_layout(r, 3, 16, 32, boundaries=b)
               for r in range(3)]
    assert [(lo.lat_start, lo.lat_end) for lo in layouts] == \
        [(0, 2), (2, 9), (9, 16)]
    assert [lo.n_lat_local for lo in layouts] == [2, 7, 7]
    # Neighbour wiring unchanged by non-uniform bands.
    assert layouts[0].south_rank is None and layouts[0].north_rank == 1
    assert layouts[1].south_rank == 0 and layouts[1].north_rank == 2
    assert layouts[2].south_rank == 1 and layouts[2].north_rank is None
    assert all(isinstance(lo, LatLonBandLayout) for lo in layouts)


def test_layout_default_boundaries_byte_identical():
    """boundaries=None must reproduce the historical even split exactly."""
    for n_lat, n_ranks in ((22, 4), (16, 4), (17, 3)):
        for r in range(n_ranks):
            a = make_latlon_band_layout(r, n_ranks, n_lat, 32)
            c = make_latlon_band_layout(r, n_ranks, n_lat, 32, boundaries=None)
            assert a == c


def test_layout_rejects_bad_boundaries():
    with pytest.raises(ValueError, match="n_ranks\\+1"):
        make_latlon_band_layout(0, 3, 16, 32, boundaries=(0, 8, 16))
    with pytest.raises(ValueError, match="span"):
        make_latlon_band_layout(0, 2, 16, 32, boundaries=(0, 8, 15))
    with pytest.raises(ValueError, match="span"):
        make_latlon_band_layout(0, 2, 16, 32, boundaries=(1, 8, 16))
    with pytest.raises(ValueError, match="strictly increasing"):
        make_latlon_band_layout(0, 3, 16, 32, boundaries=(0, 8, 8, 16))
    # Non-integral values must be rejected, not silently truncated
    # (boundaries=(0, 1.9, 16) would otherwise become (0, 1, 16); codex).
    with pytest.raises(ValueError, match="integers"):
        make_latlon_band_layout(0, 2, 16, 32, boundaries=(0, 1.9, 16))
    # Integral floats are fine (e.g. np.diff/np.cumsum products).
    lo = make_latlon_band_layout(0, 2, 16, 32, boundaries=(0.0, 8.0, 16.0))
    assert (lo.lat_start, lo.lat_end) == (0, 8)


# ------------------------------------------------- re-init boundary change --
def test_reinit_with_different_boundaries_rearms(monkeypatch):
    """initialize_distributed_latlon re-called with DIFFERENT band boundaries
    must NOT return the stale active layout (codex): every slicer/scatter
    reads lat_start/lat_end from the layout, so silently keeping the old
    decomposition mis-slices every band. Hermetic: require_mpi_stack is
    sentinel-patched so the re-arm path stops before touching real MPI."""
    import legoesm.parallel.distributed as dist

    stale = make_latlon_band_layout(1, 4, 16, 32)          # rows [4, 8)
    monkeypatch.setattr(dist, "_active_topology", stale)

    class _SentinelError(RuntimeError):
        pass

    def _boom():
        raise _SentinelError("re-arm reached require_mpi_stack")
    monkeypatch.setattr(dist, "require_mpi_stack", _boom)

    # Different boundaries for this rank ([2, 9) != [4, 8)) -> warn + re-arm
    # (the sentinel proves the stale early-return was NOT taken).
    with pytest.warns(RuntimeWarning, match="different band boundaries"):
        with pytest.raises(_SentinelError):
            dist.initialize_distributed_latlon(
                global_n_lat=16, global_n_lon=32,
                band_boundaries=(0, 2, 9, 14, 16))

    # Same span as the active layout -> the historical reuse path (warn +
    # return the existing topology, no re-arm).
    monkeypatch.setattr(dist, "_active_topology", stale)
    with pytest.warns(RuntimeWarning, match="more than once"):
        out = dist.initialize_distributed_latlon(
            global_n_lat=16, global_n_lon=32,
            band_boundaries=(0, 4, 8, 12, 16))
    assert out is stale

    # INVALID boundaries on the re-init path must raise, never be silently
    # served by a coincidentally-matching stale layout (codex round 2):
    # (0, 4.9, 8, 12, 16) truncates to the stale span for rank 1 ([4, 8))
    # but is non-integral -> ValueError from the shared validator.
    monkeypatch.setattr(dist, "_active_topology", stale)
    with pytest.raises(ValueError, match="integers"):
        dist.initialize_distributed_latlon(
            global_n_lat=16, global_n_lon=32,
            band_boundaries=(0, 4.9, 8, 12, 16))


# ------------------------------------------- slicer reassembly (non-uniform) --
def test_wet_balanced_bands_reassemble_grid():
    """slice_latlon_grid_to_band on wet-balanced (non-uniform) bands must tile
    the global grid exactly — boundaries are the single source of truth every
    slicer reads, so coverage/contiguity here proves the layout threading."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.parallel.latlon_mpi import slice_latlon_grid_to_band

    from legoesm import constants

    n_lat, n_lon, n_ranks = 24, 48, 4
    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon,
                              radius=constants.R_earth)
    # Land-heavy north third.
    w = np.concatenate([np.full(16, 48.0), np.zeros(8)])
    b = wet_band_boundaries(w, n_ranks, min_rows=2)
    layouts = [make_latlon_band_layout(r, n_ranks, n_lat, n_lon, boundaries=b)
               for r in range(n_ranks)]
    bands = [slice_latlon_grid_to_band(grid, lo, skip_total_area_reduce=True)
             for lo in layouts]

    # Bands tile the global latitude rows exactly (no gap, no overlap).
    covered = np.concatenate([np.arange(lo.lat_start, lo.lat_end)
                              for lo in layouts])
    np.testing.assert_array_equal(covered, np.arange(n_lat))
    # Band metric rows equal the corresponding global rows.
    lat_c = np.asarray(grid.lat)
    for lo, bg in zip(layouts, bands):
        np.testing.assert_array_equal(
            np.asarray(bg.lat), lat_c[lo.lat_start:lo.lat_end])
        assert int(bg.n_lat) == lo.n_lat_local


def test_wet_balance_on_etopo_mask_beats_row_split():
    """Realistic-continents gate (scaling-audit item 4): on the shipped
    ETOPO mask the wet-balanced boundaries must (a) be deterministic,
    (b) respect the halo floor, and (c) cut the wet-cell imbalance
    (max/mean) vs the even row split by a real margin — the whole point
    of the flag.  Skips cleanly if the shipped file is absent."""
    import numpy as np
    import pytest

    from pathlib import Path

    bathy = Path(__file__).resolve().parents[2] / "data" / "bathymetry" / "etopo_1deg.nc"
    if not bathy.exists():
        pytest.skip("shipped etopo_1deg.nc not present")

    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.bathymetry import (
        BathymetryConfig,
        load_bathymetry_latlon_cgrid,
    )
    from legoesm.parallel.latlon_mpi import (
        validate_band_boundaries,
        wet_band_boundaries,
    )

    grid = create_latlon_grid(n_lat=48, n_lon=96)
    _, mask = load_bathymetry_latlon_cgrid(
        grid, BathymetryConfig(source="file", path=str(bathy)))
    wet_rows = np.asarray(mask).sum(axis=1)
    n_ranks = 8

    b1 = wet_band_boundaries(wet_rows, n_ranks, min_rows=2)
    b2 = wet_band_boundaries(wet_rows, n_ranks, min_rows=2)
    assert b1 == b2, "wet boundaries must be deterministic"
    validate_band_boundaries(b1, n_ranks, 48)
    rows = np.diff(b1)
    assert rows.min() >= 2, "halo floor violated"

    def imbalance(bounds):
        wet = np.array([wet_rows[bounds[r]:bounds[r + 1]].sum()
                        for r in range(n_ranks)])
        return float(wet.max() / max(wet.mean(), 1.0))

    base, rem = divmod(48, n_ranks)
    even = tuple(np.concatenate(
        [[0], np.cumsum([base + 1 if r < rem else base
                         for r in range(n_ranks)])]).astype(int))
    imb_row, imb_wet = imbalance(even), imbalance(b1)
    # The ETOPO land distribution is strongly asymmetric by latitude; the
    # wet split must recover most of the imbalance (empirically ~1.0x vs
    # ~1.4x at this size — assert a conservative margin, not the exact
    # numbers, so coarse-grid regridding changes don't flake the gate).
    assert imb_wet < imb_row - 0.05, (imb_row, imb_wet)
    assert imb_wet < 1.15, imb_wet
