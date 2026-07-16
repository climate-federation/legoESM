"""Unit tests for the Dai-Trenberth river-runoff loader + projector."""

from __future__ import annotations

import numpy as np
import pytest

from legoesm.ocean.forcing.dai_trenberth import (
    RiverRunoffData,
    load_dai_trenberth,
    synthetic_dai_trenberth,
    project_runoff_to_grid,
    project_runoff_to_mpas_cells,
    _nearest_cell_indices,
)


# ==============================================================================
# Synthetic data
# ==============================================================================

class TestSyntheticDaiTrenberth:

    def test_shape_and_dtype(self):
        d = synthetic_dai_trenberth()
        assert d.monthly_flux_kg_s.shape == (12, d.latitudes.shape[0])
        assert d.latitudes.shape == d.longitudes.shape
        assert d.latitudes.shape[0] == len(d.names)

    def test_lons_in_zero_360(self):
        d = synthetic_dai_trenberth()
        assert np.all(d.longitudes >= 0.0)
        assert np.all(d.longitudes < 360.0)

    def test_amazon_is_largest(self):
        d = synthetic_dai_trenberth()
        idx_amazon = d.names.index("Amazon")
        annual = d.monthly_flux_kg_s.mean(axis=0)
        assert int(np.argmax(annual)) == idx_amazon

    def test_fluxes_are_positive(self):
        d = synthetic_dai_trenberth()
        assert np.all(d.monthly_flux_kg_s > 0.0)

    def test_northern_river_peaks_in_summer(self):
        d = synthetic_dai_trenberth()
        idx_lena = d.names.index("Lena")
        monthly = d.monthly_flux_kg_s[:, idx_lena]
        # Lena phase = June → peak at month 6 (0-indexed 5).
        peak_month = int(np.argmax(monthly)) + 1
        assert peak_month in (5, 6, 7)

    def test_seasonal_cycle_amplitude(self):
        d = synthetic_dai_trenberth(seasonal_amplitude=0.5)
        # max / min ratio: (1 + 0.5)/(1 - 0.5) = 3.
        idx = d.names.index("Yangtze")
        m = d.monthly_flux_kg_s[:, idx]
        ratio = float(m.max() / m.min())
        assert ratio == pytest.approx(3.0, rel=1e-6)


# ==============================================================================
# Loader fallback
# ==============================================================================

class TestLoadFallback:

    def test_missing_cache_returns_synthetic(self, tmp_path):
        d = load_dai_trenberth(cache_dir=tmp_path)
        ref = synthetic_dai_trenberth()
        assert np.allclose(d.monthly_flux_kg_s, ref.monthly_flux_kg_s)

    def test_missing_cache_strict_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_dai_trenberth(cache_dir=tmp_path, allow_synthetic=False)


# ==============================================================================
# Nearest-cell mapping
# ==============================================================================

class TestNearestCellIndices:

    def test_exact_match(self):
        lat_grid = np.array([0.0, 10.0, 20.0])
        lon_grid = np.array([0.0, 90.0, 180.0, 270.0])
        rlat = np.array([10.0])
        rlon = np.array([90.0])
        j, i = _nearest_cell_indices(rlat, rlon, lat_grid, lon_grid)
        assert int(j[0]) == 1
        assert int(i[0]) == 1

    def test_lon_wrap_around(self):
        """River at 359°E should map to the cell at 0°E (1° away),
        not the cell at 270°E (89° away)."""
        lat_grid = np.array([0.0])
        lon_grid = np.array([0.0, 90.0, 180.0, 270.0])
        rlat = np.array([0.0])
        rlon = np.array([359.0])
        j, i = _nearest_cell_indices(rlat, rlon, lat_grid, lon_grid)
        assert int(i[0]) == 0   # 0°E

    def test_negative_lon_handled(self):
        """River at -1°E (= 359°E) maps to 0°E."""
        lat_grid = np.array([0.0])
        lon_grid = np.array([0.0, 90.0, 180.0, 270.0])
        rlat = np.array([0.0])
        rlon = np.array([-1.0])
        j, i = _nearest_cell_indices(rlat, rlon, lat_grid, lon_grid)
        assert int(i[0]) == 0


# ==============================================================================
# Projection to grid
# ==============================================================================

class TestProjectRunoffToGrid:

    def _grid(self, n_lat=18, n_lon=36):
        lat = np.linspace(-85.0, 85.0, n_lat)
        lon = np.linspace(0.0, 360.0 - 360.0 / n_lon, n_lon)
        cell_area = np.full((n_lat, n_lon), 1.0e10)   # 10 000 km² per cell
        return lat, lon, cell_area

    def test_single_river_projects_to_one_cell(self):
        rivers = RiverRunoffData(
            latitudes=np.array([10.0]),
            longitudes=np.array([90.0]),
            monthly_flux_kg_s=np.full((12, 1), 1.0e7),
            names=("test",),
        )
        lat, lon, area = self._grid()
        out = project_runoff_to_grid(
            rivers,
            grid_lat_deg=lat, grid_lon_deg=lon, cell_area_m2=area,
            month=None,
        )
        # Total flux preserved.
        total = float((out * area).sum())
        assert total == pytest.approx(1.0e7, rel=1e-6)

    def test_total_flux_preserved(self):
        rivers = synthetic_dai_trenberth()
        lat, lon, area = self._grid(n_lat=36, n_lon=72)
        out = project_runoff_to_grid(
            rivers, grid_lat_deg=lat, grid_lon_deg=lon, cell_area_m2=area,
        )
        expected = float(rivers.monthly_flux_kg_s.mean(axis=0).sum())
        observed = float((out * area).sum())
        assert observed == pytest.approx(expected, rel=1e-6)

    def test_monthly_snapshot_changes_total(self):
        """Per-month projection sums to the per-month total flux."""
        rivers = synthetic_dai_trenberth(seasonal_amplitude=0.5)
        lat, lon, area = self._grid(n_lat=36, n_lon=72)
        for m in (1, 6, 12):
            out = project_runoff_to_grid(
                rivers,
                grid_lat_deg=lat, grid_lon_deg=lon,
                cell_area_m2=area, month=m,
            )
            expected = float(rivers.monthly_flux_kg_s[m - 1].sum())
            observed = float((out * area).sum())
            assert observed == pytest.approx(expected, rel=1e-6)

    def test_ocean_mask_relocates_land_landing(self):
        """A river landing in a land cell is moved to the nearest
        ocean cell within the 5° search box.  Uses a fine 2.5° grid
        so the neighbouring ocean cell is < 5° away (the documented
        relocation radius).  Also asserts mass conservation across
        the relocation."""
        rivers = RiverRunoffData(
            latitudes=np.array([0.0]),
            longitudes=np.array([0.0]),
            monthly_flux_kg_s=np.full((12, 1), 1.0e7),
            names=("test",),
        )
        # 2.5° grid (72 lat × 144 lon).
        lat, lon, area = self._grid(n_lat=72, n_lon=144)
        ocean_mask = np.ones((lat.shape[0], lon.shape[0]), dtype=np.int32)
        j_target = int(np.argmin(np.abs(lat - 0.0)))
        i_target = int(np.argmin(np.abs(lon - 0.0)))
        ocean_mask[j_target, i_target] = 0
        out = project_runoff_to_grid(
            rivers,
            grid_lat_deg=lat, grid_lon_deg=lon, cell_area_m2=area,
            ocean_mask=ocean_mask,
        )
        # Masked cell stays zero.
        assert out[j_target, i_target] == 0.0
        # Total flux preserved after relocation (kg/s).
        total = float((out * area).sum())
        assert total == pytest.approx(1.0e7, rel=1e-6)

    def test_ocean_mask_drops_when_no_ocean_within_box(self):
        """River landing far from any ocean cell is silently dropped."""
        rivers = RiverRunoffData(
            latitudes=np.array([0.0]),
            longitudes=np.array([0.0]),
            monthly_flux_kg_s=np.full((12, 1), 1.0e7),
            names=("test",),
        )
        lat, lon, area = self._grid()
        ocean_mask = np.zeros((lat.shape[0], lon.shape[0]), dtype=np.int32)
        out = project_runoff_to_grid(
            rivers,
            grid_lat_deg=lat, grid_lon_deg=lon, cell_area_m2=area,
            ocean_mask=ocean_mask,
        )
        assert float(out.sum()) == 0.0

    def test_ocean_mask_5deg_radius_enforced_on_coarse_grid(self):
        """On an explicit 10° grid where the only ocean cell sits 10°
        diagonally away from the river's landing land cell, the
        river must be DROPPED — the documented relocation radius is
        5° and a 10° diagonal hop exceeds it.  Guards against the
        ``ceil(max_deg/dx)`` cell-count window over-permissively
        sweeping past the 5° cutoff."""
        rivers = RiverRunoffData(
            latitudes=np.array([0.0]),
            longitudes=np.array([0.0]),
            monthly_flux_kg_s=np.full((12, 1), 1.0e7),
            names=("test",),
        )
        # Explicit 2x2 grid: lat ∈ {0°, 10°}, lon ∈ {0°, 10°}.
        lat = np.array([0.0, 10.0])
        lon = np.array([0.0, 10.0])
        area = np.full((2, 2), 1.0e10)
        ocean_mask = np.zeros((2, 2), dtype=np.int32)
        # Only the diagonal cell at (10°N, 10°E) is ocean — 10°
        # diagonally from the river landing at (0°, 0°).
        ocean_mask[1, 1] = 1
        out = project_runoff_to_grid(
            rivers,
            grid_lat_deg=lat, grid_lon_deg=lon, cell_area_m2=area,
            ocean_mask=ocean_mask,
        )
        # Distance 10° > 5° radius → river dropped.
        assert float((out * area).sum()) == 0.0

    def test_units_kg_m2_s(self):
        """Output should have units of kg/m²/s after dividing by cell area."""
        rivers = RiverRunoffData(
            latitudes=np.array([0.0]),
            longitudes=np.array([0.0]),
            monthly_flux_kg_s=np.full((12, 1), 1.0e7),  # 10 Mt/s
            names=("test",),
        )
        lat = np.array([0.0])
        lon = np.array([0.0])
        area = np.array([[1.0e10]])  # 10 000 km²
        out = project_runoff_to_grid(
            rivers,
            grid_lat_deg=lat, grid_lon_deg=lon, cell_area_m2=area,
        )
        # 1e7 kg/s / 1e10 m² = 1e-3 kg/m²/s.
        assert out[0, 0] == pytest.approx(1.0e-3, rel=1e-6)


class TestProjectRunoffToMPASCells:
    """The unstructured (MPAS) counterpart of project_runoff_to_grid.

    apply_runoff_step_mpas was implemented and imported by run_centennial_spinup
    but the runoff SETUP warned "not yet supported on grid=mpas" because there
    was no river->cell projector for an unstructured mesh. These pin the new
    conservative nearest-ocean-cell projector.
    """

    def _mesh(self, n=64):
        # A crude quasi-uniform scatter of cells on the sphere.
        rng = np.random.RandomState(0)  # noqa: NPY002 - fixed seed, test only
        lat = rng.uniform(-80.0, 80.0, n)
        lon = rng.uniform(0.0, 360.0, n)
        area = np.full(n, 1.0e10)
        return lat, lon, area

    def test_single_river_conserves_flux(self):
        lat, lon, area = self._mesh()
        rivers = RiverRunoffData(
            latitudes=np.array([lat[3]]),           # sits exactly on a cell
            longitudes=np.array([lon[3]]),
            monthly_flux_kg_s=np.full((12, 1), 1.0e7),
            names=("test",),
        )
        out = project_runoff_to_mpas_cells(
            rivers, lat_cell_deg=lat, lon_cell_deg=lon, area_cell_m2=area,
        )
        assert out.shape == (lat.size,)
        assert float((out * area).sum()) == pytest.approx(1.0e7, rel=1e-6)
        assert int(np.argmax(out)) == 3           # landed on the nearest cell

    def test_total_flux_preserved_all_ocean(self):
        lat, lon, area = self._mesh(n=128)
        rivers = synthetic_dai_trenberth()
        # No-drop radius: isolate CONSERVATION from the drop behaviour (which
        # has its own test). On a sparse toy mesh a 5deg cutoff would correctly
        # drop rivers falling in the ~30deg gaps -- orthogonal to conservation.
        out = project_runoff_to_mpas_cells(
            rivers, lat_cell_deg=lat, lon_cell_deg=lon, area_cell_m2=area,
            max_search_deg=180.0,
        )
        expected = float(rivers.monthly_flux_kg_s.mean(axis=0).sum())
        assert float((out * area).sum()) == pytest.approx(expected, rel=1e-6)

    def test_land_river_relocates_to_nearest_ocean_cell(self):
        # Two cells close together; the nearest one to the river is LAND, so the
        # flux must go to the other (ocean) cell -- not be dropped.
        lat = np.array([10.0, 10.5, -40.0])
        lon = np.array([200.0, 200.0, 100.0])
        area = np.full(3, 1.0e10)
        ocean = np.array([0, 1, 1])          # cell 0 is land
        rivers = RiverRunoffData(
            latitudes=np.array([10.01]),     # nearest is cell 0 (land)
            longitudes=np.array([200.0]),
            monthly_flux_kg_s=np.full((12, 1), 5.0e6),
            names=("r",),
        )
        out = project_runoff_to_mpas_cells(
            rivers, lat_cell_deg=lat, lon_cell_deg=lon, area_cell_m2=area,
            ocean_mask=ocean,
        )
        assert out[0] == 0.0                 # land cell receives nothing
        assert out[1] > 0.0                  # relocated to the ocean neighbour
        assert float((out * area).sum()) == pytest.approx(5.0e6, rel=1e-6)

    def test_river_with_no_ocean_cell_in_radius_is_dropped(self):
        # Only ocean cell is on the far side of the planet -> outside 5°.
        lat = np.array([0.0, 80.0])
        lon = np.array([0.0, 180.0])
        area = np.full(2, 1.0e10)
        ocean = np.array([0, 1])             # only the antipodal cell is ocean
        rivers = RiverRunoffData(
            latitudes=np.array([0.0]),
            longitudes=np.array([0.0]),
            monthly_flux_kg_s=np.full((12, 1), 3.0e6),
            names=("r",),
        )
        out = project_runoff_to_mpas_cells(
            rivers, lat_cell_deg=lat, lon_cell_deg=lon, area_cell_m2=area,
            ocean_mask=ocean, max_search_deg=5.0,
        )
        assert float(out.sum()) == 0.0       # dropped, not misassigned

    def test_no_ocean_cells_returns_zero_field(self):
        lat, lon, area = self._mesh(n=8)
        rivers = synthetic_dai_trenberth()
        out = project_runoff_to_mpas_cells(
            rivers, lat_cell_deg=lat, lon_cell_deg=lon, area_cell_m2=area,
            ocean_mask=np.zeros(8),
        )
        assert out.shape == (8,) and float(out.sum()) == 0.0

    def test_monthly_snapshot_conserves_per_month_total(self):
        lat, lon, area = self._mesh(n=128)
        rivers = synthetic_dai_trenberth(seasonal_amplitude=0.5)
        for m in (1, 6, 12):
            out = project_runoff_to_mpas_cells(
                rivers, lat_cell_deg=lat, lon_cell_deg=lon, area_cell_m2=area,
                month=m, max_search_deg=180.0,
            )
            expected = float(rivers.monthly_flux_kg_s[(m - 1) % 12].sum())
            assert float((out * area).sum()) == pytest.approx(expected, rel=1e-6)
