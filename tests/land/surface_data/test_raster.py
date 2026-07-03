"""Unit tests for the ENVI/BIL raster reader (pure NumPy, no GDAL)."""

import numpy as np
import pytest

from legoesm.land.surface_data.raster import (
    EnviBilHeader,
    read_envi_hdr,
    open_bil_memmap,
    cell_centers,
    write_envi_bil,
    _resolve_dtype,
)


def _toy_header(nrows=4, ncols=6, dtype="<u2", nodata=65535.0):
    return EnviBilHeader(
        nrows=nrows, ncols=ncols, nbands=1, dtype=np.dtype(dtype), nodata=nodata,
        ulxmap=-179.75, ulymap=89.75, xdim=0.5, ydim=0.5, layout="BIL",
    )


def test_dtype_resolution():
    assert _resolve_dtype(16, "UNSIGNEDINT", "I") == np.dtype("<u2")
    assert _resolve_dtype(16, "UNSIGNEDINT", "M") == np.dtype(">u2")
    assert _resolve_dtype(32, "FLOAT", "I") == np.dtype("<f4")
    assert _resolve_dtype(8, "UNSIGNEDINT", "I") == np.dtype("u1")
    with pytest.raises(ValueError):
        _resolve_dtype(24, "UNSIGNEDINT", "I")


def test_write_read_roundtrip(tmp_path):
    hdr = _toy_header()
    data = np.arange(hdr.nrows * hdr.ncols, dtype="<u2").reshape(hdr.nrows, hdr.ncols)
    bil = str(tmp_path / "toy.bil")
    write_envi_bil(bil, data, hdr)

    hdr2 = read_envi_hdr(bil[:-4] + ".hdr")
    assert (hdr2.nrows, hdr2.ncols) == (hdr.nrows, hdr.ncols)
    assert hdr2.dtype == np.dtype("<u2")
    assert hdr2.nodata == 65535.0
    assert hdr2.xdim == 0.5 and hdr2.ydim == 0.5

    mm, _ = open_bil_memmap(bil)
    assert mm.shape == (hdr.nrows, hdr.ncols)
    np.testing.assert_array_equal(np.asarray(mm), data)


def test_cell_centers_north_to_south():
    hdr = _toy_header()
    lat, lon = cell_centers(hdr)
    assert lat.shape == (4,) and lon.shape == (6,)
    # UL pixel centre is (ulymap, ulxmap); lat descends, lon ascends.
    assert np.isclose(lat[0], 89.75) and np.isclose(lon[0], -179.75)
    assert lat[1] < lat[0]                       # north -> south
    assert np.isclose(lon[1] - lon[0], 0.5)


def test_big_endian_roundtrip(tmp_path):
    hdr = _toy_header(dtype=">u2")
    data = (np.arange(24, dtype=">u2").reshape(4, 6))
    bil = str(tmp_path / "be.bil")
    write_envi_bil(bil, data, hdr)
    hdr2 = read_envi_hdr(bil[:-4] + ".hdr")
    assert hdr2.dtype.byteorder == ">"
    mm, _ = open_bil_memmap(bil)
    np.testing.assert_array_equal(np.asarray(mm), data)
