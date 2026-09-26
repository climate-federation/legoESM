"""--runoff-source core2_climatology (FESOM2 parity batch 2, item 3)."""
import numpy as np
import pytest

import scripts.run.run_omip as run_omip


def test_route_runoff_stack_static_replaces_records():
    rng = np.random.default_rng(20240101)
    runoff_stack = rng.normal(size=(3, 4, 5))
    static = rng.normal(size=(4, 5))
    routed = np.asarray(run_omip._route_runoff_stack(runoff_stack, None, static=static))
    assert routed.shape == (3, 4, 5)
    np.testing.assert_allclose(routed, np.broadcast_to(static, (3, 4, 5)), rtol=1e-12)
    assert np.abs(routed - runoff_stack).max() > 0.1          # really replaced
    with pytest.raises(ValueError, match=r"\(4, 4\).*\(4, 5\)"):
        run_omip._route_runoff_stack(runoff_stack, None, static=np.zeros((4, 4)))
    assert run_omip._route_runoff_stack(runoff_stack, None) is runoff_stack


def test_cli_and_every_preload_path_passes_the_static_field():
    import inspect
    a = run_omip.parse_args(["--grid", "mpas", "--runoff-source", "core2_climatology",
                             "--runoff-file", "/x/CORE2_runoff.nc"])
    assert (a.runoff_source, a.runoff_file) == ("core2_climatology", "/x/CORE2_runoff.nc")
    assert run_omip.parse_args(["--grid", "mpas"]).runoff_source == "jra55_friver"
    src = inspect.getsource(run_omip)
    assert src.count('static=jra55_state.get("runoff_static")') == 3


def test_core2_loader_zeroes_land_and_regrids_on_host(tmp_path, monkeypatch):
    netCDF4 = pytest.importorskip("netCDF4")
    lat = np.linspace(-89.5, 89.5, 4); lon = np.linspace(0.5, 359.5, 6)
    f = tmp_path / "runoff.nc"
    with netCDF4.Dataset(f, "w") as ds:
        ds.createDimension("Time", 1); ds.createDimension("lat", 4); ds.createDimension("lon", 6)
        ds.createVariable("lat", "f4", ("lat",))[:] = lat
        ds.createVariable("lon", "f4", ("lon",))[:] = lon
        v = ds.createVariable("Foxx_o_roff", "f4", ("Time", "lat", "lon"), fill_value=None)
        v.missing_value = np.float32(1e30)
        data = np.full((1, 4, 6), 1e30, dtype=np.float32)
        data[0, 1, 2] = 2e-4; data[0, 2, 3] = 5e-5
        v[:] = data
    recorded = []

    class RW:  # noqa: D401 - stand-in for the regrid weights
        target_shape = (4, 6)

    def stub(records, rw):
        recorded.append(np.asarray(records))
        return np.asarray(records)
    monkeypatch.setattr(run_omip, "_regrid_records_host", stub)
    out = run_omip._load_core2_runoff_static(str(f), RW(), lat, lon)
    assert out.shape == (4, 6) and out.dtype == np.float64
    assert recorded[0].shape == (1, 4, 6) and not np.isnan(recorded[0]).any()
    assert out.sum() == pytest.approx(2.5e-4, rel=1e-6)      # land -> 0, rivers kept
    with pytest.raises(ValueError, match="lat"):
        run_omip._load_core2_runoff_static(str(f), RW(), lat + 1.0, lon)
