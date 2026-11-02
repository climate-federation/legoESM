"""cam6_cirrus_aerosol: mapping a CAM prescribed MAM3 file onto cells, cyclic time interpolation and the
median exp(logm) at model levels, on a small synthetic file with known answers."""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from legoesm.atmosphere.physics.microphysics import cam6_cirrus_aerosol as A  # noqa: E402

netCDF4 = pytest.importorskip("netCDF4")


def _file(path, value_of_month):
    lat = np.linspace(-90, 90, 5)
    lon = np.arange(0, 360, 60.0)
    with netCDF4.Dataset(path, "w") as d:
        for n, s in (("time", 12), ("lev", 3), ("lat", 5), ("lon", 6)):
            d.createDimension(n, s)
        d.createVariable("lat", "f8", ("lat",))[:] = lat
        d.createVariable("lon", "f8", ("lon",))[:] = lon
        d.createVariable("hyam", "f8", ("lev",))[:] = [0.1, 0.05, 0.0]
        d.createVariable("hybm", "f8", ("lev",))[:] = [0.0, 0.4, 0.9]
        d.createVariable("P0", "f8", ())[...] = 1.0e5
        d.createVariable("date", "i4", ("time",))[:] = [20000116, 20000214, 20000316, 20000415, 20000516,
                                                       20000615, 20000716, 20000816, 20000915, 20001016,
                                                       20001115, 20001216]
        d.createVariable("PS", "f4", ("time", "lat", "lon"))[:] = 1.0e5
        for v in A.CIRRUS_FIELDS:
            x = np.zeros((12, 3, 5, 6), np.float32) + np.asarray(
                [value_of_month(m) for m in range(12)], np.float32)[:, None, None, None]
            d.createVariable(f"{v}_logm", "f4", ("time", "lev", "lat", "lon"))[:] = x


def test_map_time_interp_and_median(tmp_path):
    nc = tmp_path / "a.nc"
    _file(nc, lambda m: float(m))            # logm = month index, uniform in space and height
    lat = np.radians([0.0, 30.0, -60.0])
    lon = np.radians([10.0, 200.0, 350.0])
    m = A.map_to_cells(nc, lat, lon)
    np.savez(tmp_path / "a.npz", **m)
    assert m["num_a2_logm"].shape == (12, 3, 3)
    np.testing.assert_allclose(m["num_a2_logm"][3], 3.0, rtol=0, atol=1e-12)
    setup = A.load_setup(tmp_path / "a.npz", top_lev=0)
    # 1 April (day 90 since Jan 1, noleap) lies between the 16 March (day 74) and 15 April (day 104) records
    t = 90.0
    _, f = A.cirrus_aerosol_forcing(tmp_path / "a.npz", setup, t)
    pmid = np.tile([[2.0e4, 5.0e4, 8.5e4]], (3, 1))
    out = A.cirrus_aerosol_at_model_levels(f, setup, pmid)
    w = (104.0 - t) / (104.0 - 74.0)
    np.testing.assert_allclose(np.asarray(out["num_a2"]), np.exp(w * 2.0 + (1 - w) * 3.0), rtol=1e-12)
    # cyclic wrap between 16 December and 16 January
    _, f2 = A.cirrus_aerosol_forcing(tmp_path / "a.npz", setup, 360.0)
    v = np.asarray(A.cirrus_aerosol_at_model_levels(f2, setup, pmid)["so4_a3"])
    assert np.all((v > np.exp(0.0)) & (v < np.exp(11.0)))
