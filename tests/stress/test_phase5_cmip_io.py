"""Phase 5: CMIP I/O completeness stress tests.

Validates that all CMIP-required inputs load correctly, all outputs write
correctly, and files have proper CF/CMOR metadata.
"""

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# 5.1-5.4  CFWriter writes all variable tables
# ---------------------------------------------------------------------------

class TestCFWriterAmon:
    """Amon table: 2D and 3D variables write correctly."""

    @pytest.fixture(scope="class")
    def writer_dir(self, tmp_path_factory):
        return tmp_path_factory.mktemp("cmor_amon")

    @pytest.fixture(scope="class")
    def writer(self, writer_dir):
        from legoesm.io.cmor_output import CFWriter
        return CFWriter(
            output_dir=str(writer_dir),
            experiment_id="piControl",
            model_id="legoESM-1-0",
            freq="mon",
        )

    def _make_latlondata(self, ndim=2, nlat=8, nlon=16, nplev=None):
        lat = np.linspace(-87.5, 87.5, nlat)
        lon = np.linspace(0, 348.75, nlon)
        if ndim == 2:
            data = np.random.randn(nlat, nlon).astype(np.float32) * 10 + 280
        else:
            data = np.random.randn(nplev, nlat, nlon).astype(np.float32) * 10 + 280
        return data, lat, lon

    def test_amon_2d_variables(self, writer):
        """All Amon 2D variables write without error."""
        from legoesm.io.cmor_output import CMOR_TABLES

        amon = CMOR_TABLES["Amon"]
        vars_2d = [k for k, v in amon.items()
                   if "plev" not in v["dimensions"] and "depth" not in v["dimensions"]]

        data, lat, lon = self._make_latlondata(ndim=2)
        for var_name in vars_2d:
            path = writer.write_field(
                var_name=var_name,
                data=data,
                time=15.0,
                time_bounds=(0.0, 30.0),
                lat=lat,
                lon=lon,
                table="Amon",
            )
            assert path.exists(), f"File not created for {var_name}"

    def test_amon_3d_variables(self, writer):
        """All Amon 3D (plev) variables write without error."""
        from legoesm.io.cmor_output import CMOR_TABLES, CMIP6_PLEV19

        amon = CMOR_TABLES["Amon"]
        vars_3d = [k for k, v in amon.items() if "plev" in v["dimensions"]]

        nplev = len(CMIP6_PLEV19)
        data, lat, lon = self._make_latlondata(ndim=3, nplev=nplev)
        for var_name in vars_3d:
            path = writer.write_field(
                var_name=var_name,
                data=data,
                time=15.0,
                time_bounds=(0.0, 30.0),
                lat=lat,
                lon=lon,
                plev=CMIP6_PLEV19,
                table="Amon",
            )
            assert path.exists(), f"File not created for {var_name}"

    def test_written_files_readable(self, writer, writer_dir):
        """Written files are readable with xarray and have CF attributes."""
        import xarray as xr

        nc_files = list(Path(writer_dir).rglob("*.nc"))
        assert len(nc_files) > 0, "No NC files were written"

        # Check a sample file
        ds = xr.open_dataset(nc_files[0])
        assert "Conventions" in ds.attrs
        # "CF-1.8" is REJECTED by the CMIP6 CV, whose Conventions regex is
        # ^CF-1.7 CMIP-6.[0-2]( UGRID-1.0){0,}$ -- the value now comes from
        # the vendored table Header instead of a hand-typed string.
        assert ds.attrs["Conventions"] == "CF-1.7 CMIP-6.2"
        assert "experiment_id" in ds.attrs
        ds.close()


class TestCFWriterLmon:
    """Lmon table variables."""

    def test_lmon_variables(self, tmp_path):
        from legoesm.io.cmor_output import CFWriter, CMOR_TABLES

        writer = CFWriter(
            output_dir=str(tmp_path),
            experiment_id="piControl",
            model_id="legoESM-1-0",
        )
        lmon = CMOR_TABLES["Lmon"]
        lat = np.linspace(-87.5, 87.5, 8)
        lon = np.linspace(0, 348.75, 16)

        for var_name, entry in lmon.items():
            if "depth" in entry["dimensions"]:
                depth = np.array([0.05, 0.15, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0])
                data = np.random.randn(len(depth), 8, 16).astype(np.float32) + 280
                path = writer.write_field(
                    var_name=var_name, data=data,
                    time=15.0, time_bounds=(0.0, 30.0),
                    lat=lat, lon=lon, depth=depth, table="Lmon",
                )
            else:
                data = np.random.randn(8, 16).astype(np.float32)
                path = writer.write_field(
                    var_name=var_name, data=data,
                    time=15.0, time_bounds=(0.0, 30.0),
                    lat=lat, lon=lon, table="Lmon",
                )
            assert path.exists(), f"Lmon {var_name} not written"


class TestCFWriterOmonAday:
    """Omon and Aday table variables."""

    def test_omon_variables(self, tmp_path):
        from legoesm.io.cmor_output import CFWriter, CMOR_TABLES

        writer = CFWriter(
            output_dir=str(tmp_path),
            experiment_id="amip",
            model_id="legoESM-1-0",
        )
        lat = np.linspace(-87.5, 87.5, 8)
        lon = np.linspace(0, 348.75, 16)
        data = np.random.randn(8, 16).astype(np.float32) + 290
        # Ocean depth levels (metres below sea surface) for the 3-D
        # Omon variables (thetao/so/uo/vo/wo/rhopoto and the basin
        # stream functions).  The writer fail-fasts if a depth-
        # dimensioned variable is written without a depth coordinate,
        # so dispatch on the declared dimensions exactly as the Lmon
        # test does for soil-depth fields.
        ocean_depth = np.array(
            [5.0, 15.0, 30.0, 60.0, 100.0, 150.0, 200.0, 300.0, 500.0, 1000.0]
        )

        for var_name, entry in CMOR_TABLES["Omon"].items():
            if "depth" in entry["dimensions"]:
                depth_data = (
                    np.random.randn(len(ocean_depth), 8, 16).astype(np.float32)
                    + 5.0
                )
                path = writer.write_field(
                    var_name=var_name, data=depth_data,
                    time=15.0, time_bounds=(0.0, 30.0),
                    lat=lat, lon=lon, depth=ocean_depth, table="Omon",
                )
            else:
                path = writer.write_field(
                    var_name=var_name, data=data,
                    time=15.0, time_bounds=(0.0, 30.0),
                    lat=lat, lon=lon, table="Omon",
                )
            assert path.exists(), f"Omon {var_name} not written"

    def test_aday_variables(self, tmp_path):
        from legoesm.io.cmor_output import CFWriter, CMOR_TABLES

        writer = CFWriter(
            output_dir=str(tmp_path),
            experiment_id="amip",
            model_id="legoESM-1-0",
            freq="day",
        )
        lat = np.linspace(-87.5, 87.5, 8)
        lon = np.linspace(0, 348.75, 16)
        data = np.random.randn(8, 16).astype(np.float32) + 280

        # ``Aday`` is the legacy alias for the CMIP6 ``day`` table.  Its
        # ``ua``/``va`` are plev8 variables, so they need a pressure axis;
        # ``rsut`` is no longer here at all (it is a ``CFday`` variable --
        # it does not exist in the CMIP6 ``day`` table).
        assert "rsut" not in CMOR_TABLES["Aday"]
        for var_name, entry in CMOR_TABLES["Aday"].items():
            if "plev" in entry["dimensions"]:
                path = writer.write_field(
                    var_name=var_name, data=data[np.newaxis, ...],
                    time=0.5, time_bounds=(0.0, 1.0),
                    lat=lat, lon=lon, plev=np.array([85000.0]), table="Aday",
                )
            else:
                path = writer.write_field(
                    var_name=var_name, data=data,
                    time=0.5, time_bounds=(0.0, 1.0),
                    lat=lat, lon=lon, table="Aday",
                )
            assert path.exists(), f"Aday {var_name} not written"


# ---------------------------------------------------------------------------
# 5.5  DRS filename convention
# ---------------------------------------------------------------------------

class TestDRSFilename:
    """CMIP6 Data Reference Syntax filename compliance."""

    def test_filename_pattern(self, tmp_path):
        from legoesm.io.cmor_output import CFWriter

        writer = CFWriter(
            output_dir=str(tmp_path),
            experiment_id="piControl",
            model_id="legoESM-1-0",
            variant_label="r1i1p1f1",
            grid_label="gn",
        )
        lat = np.linspace(-87.5, 87.5, 8)
        lon = np.linspace(0, 348.75, 16)
        data = np.random.randn(8, 16).astype(np.float32) + 280

        path = writer.write_field(
            var_name="tas", data=data,
            time=15.0, time_bounds=(0.0, 30.0),
            lat=lat, lon=lon,
        )

        fname = path.name
        assert fname.startswith("tas_Amon_legoESM-1-0_piControl_r1i1p1f1_gn"), (
            f"Unexpected filename: {fname}"
        )
        assert fname.endswith(".nc")


# ---------------------------------------------------------------------------
# 5.6  Time axis and bounds
# ---------------------------------------------------------------------------

class TestTimeAxis:
    """Time coordinate attributes and bounds correctness."""

    def test_time_bounds_single_month(self, tmp_path):
        """A single monthly write produces correct time and time_bnds."""
        import xarray as xr
        from legoesm.io.cmor_output import CFWriter

        writer = CFWriter(
            output_dir=str(tmp_path),
            experiment_id="test",
            model_id="legoESM-1-0",
        )
        lat = np.linspace(-87.5, 87.5, 8)
        lon = np.linspace(0, 348.75, 16)
        data = np.random.randn(8, 16).astype(np.float32) + 280

        # January: days 0-31, midpoint 15.5
        path = writer.write_field(
            var_name="tas", data=data,
            time=15.5, time_bounds=(0.0, 31.0),
            lat=lat, lon=lon,
        )
        assert path.exists()

        ds = xr.open_dataset(path)
        assert "time" in ds.dims
        assert "time_bnds" in ds or "time_bounds" in ds
        assert ds.attrs.get("calendar", "") in ("noleap", "365_day", "")
        ds.close()


# ---------------------------------------------------------------------------
# 5.7  Global attributes
# ---------------------------------------------------------------------------

class TestGlobalAttributes:
    """Required CMIP6 global attributes are present."""

    def test_required_global_attrs(self, tmp_path):
        import xarray as xr
        from legoesm.io.cmor_output import CFWriter

        writer = CFWriter(
            output_dir=str(tmp_path),
            experiment_id="historical",
            model_id="legoESM-1-0",
            institution="Columbia University",
        )
        lat = np.linspace(-87.5, 87.5, 8)
        lon = np.linspace(0, 348.75, 16)
        data = np.random.randn(8, 16).astype(np.float32) + 280

        path = writer.write_field(
            var_name="tas", data=data,
            time=15.0, time_bounds=(0.0, 30.0),
            lat=lat, lon=lon,
        )

        ds = xr.open_dataset(path)
        required = [
            "Conventions", "experiment_id", "source_id",
            "variant_label", "grid_label", "institution",
            "frequency", "table_id", "variable_id", "creation_date",
        ]
        for attr in required:
            assert attr in ds.attrs, f"Missing global attribute: {attr}"
            assert ds.attrs[attr], f"Empty global attribute: {attr}"

        assert ds.attrs["experiment_id"] == "historical"
        assert ds.attrs["source_id"] == "legoESM-1-0"
        ds.close()


# ---------------------------------------------------------------------------
# 5.9  Prescribed concentration input
# ---------------------------------------------------------------------------

class TestForcingInput:
    """GHG forcing configuration and loading."""

    def test_ghg_constant_source(self):
        """Constant GHG source returns config values."""
        from legoesm.forcing.external import GHGConfig, get_ghg_at_time

        config = GHGConfig(
            co2_ppmv=400.0,
            ch4_ppbv=1800.0,
            n2o_ppbv=320.0,
            source="constant",
        )
        result = get_ghg_at_time(config, day=100.0)
        assert result["co2_ppmv"] == pytest.approx(400.0)
        assert result["ch4_ppbv"] == pytest.approx(1800.0)
        assert result["n2o_ppbv"] == pytest.approx(320.0)

    def test_ghg_vmr_conversion(self):
        """GHG concentrations convert to VMR correctly."""
        from legoesm.forcing.external import ghg_concentrations_to_vmr

        ghg = {"co2_ppmv": 400.0, "ch4_ppbv": 1800.0, "n2o_ppbv": 320.0}
        vmr = ghg_concentrations_to_vmr(ghg)
        # CO2: 400 ppmv = 400e-6
        assert vmr["co2"] == pytest.approx(400e-6, rel=1e-3)
        assert vmr["ch4"] == pytest.approx(1800e-9, rel=1e-3)
        assert vmr["n2o"] == pytest.approx(320e-9, rel=1e-3)


# ---------------------------------------------------------------------------
# 5.10  CMIP experiment config GHG chain
# ---------------------------------------------------------------------------

class TestExperimentGHGChain:
    """CMIP experiment templates produce correct GHG values."""

    def test_picontrol_ghg(self):
        from legoesm.forcing.experiments import ghg_at_year
        co2, ch4, n2o = ghg_at_year("piControl", 1850)
        assert co2 == pytest.approx(284.3, abs=0.1)

    def test_historical_1850(self):
        from legoesm.forcing.experiments import ghg_at_year
        co2, ch4, n2o = ghg_at_year("historical", 1850)
        assert co2 == pytest.approx(284.3, abs=0.1)

    def test_historical_2000(self):
        from legoesm.forcing.experiments import ghg_at_year
        co2, ch4, n2o = ghg_at_year("historical", 2000)
        assert co2 == pytest.approx(369.5, abs=0.1)

    def test_ssp585_2100(self):
        from legoesm.forcing.experiments import ghg_at_year
        co2, ch4, n2o = ghg_at_year("ssp585", 2100)
        assert co2 == pytest.approx(1135.0, abs=0.1)
