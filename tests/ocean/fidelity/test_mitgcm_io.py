"""Unit tests for the dependency-free MITgcm mdsio reader/writer."""

from __future__ import annotations

import numpy as np
import pytest
from legoesm.ocean.fidelity import mitgcm_io


def test_roundtrip_2d_preserves_values_and_order(tmp_path):
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((5, 7)).astype(np.float32)  # (ny, nx)
    mitgcm_io.write_mds(tmp_path / "Eta", arr)
    back, meta = mitgcm_io.read_mds(tmp_path / "Eta")
    np.testing.assert_array_equal(back, arr)
    # dimList is fastest-first (x, y) = reverse of the (ny, nx) array shape.
    assert meta.dims == (7, 5)
    assert meta.nrecords == 1
    assert meta.dataprec == "float32"


def test_roundtrip_3d_preserves_axis_order(tmp_path):
    rng = np.random.default_rng(1)
    arr = rng.standard_normal((3, 5, 7)).astype(np.float32)  # (nz, ny, nx)
    mitgcm_io.write_mds(tmp_path / "T", arr)
    back, meta = mitgcm_io.read_mds(tmp_path / "T")
    np.testing.assert_array_equal(back, arr)
    assert meta.dims == (7, 5, 3)  # x, y, z fastest-first


def test_roundtrip_float64(tmp_path):
    arr = np.linspace(0, 1, 12, dtype=np.float64).reshape(3, 4)
    mitgcm_io.write_mds(tmp_path / "S", arr, dataprec="float64")
    back, meta = mitgcm_io.read_mds(tmp_path / "S")
    assert meta.dataprec == "float64"
    assert back.dtype == np.float64
    np.testing.assert_array_equal(back, arr)


def test_data_is_big_endian_on_disk(tmp_path):
    """The .data file must be big-endian regardless of host byte order."""
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    _, data_path = mitgcm_io.write_mds(tmp_path / "U", arr)
    raw = np.fromfile(data_path, dtype=">f4")  # force big-endian read
    np.testing.assert_array_equal(raw.reshape(2, 3), arr)
    # A little-endian interpretation would NOT match (sanity: bytes differ).
    le = np.fromfile(data_path, dtype="<f4").reshape(2, 3)
    assert not np.array_equal(le, arr)


def test_iteration_suffix_naming(tmp_path):
    arr = np.ones((2, 2), dtype=np.float32)
    meta_path, data_path = mitgcm_io.write_mds(tmp_path / "U", arr, iteration=10)
    assert meta_path.name == "U.0000000010.meta"
    assert data_path.name == "U.0000000010.data"
    back, meta = mitgcm_io.read_mds(tmp_path / "U", iteration=10)
    np.testing.assert_array_equal(back, arr)
    assert meta.time_step is None or isinstance(meta.time_step, int)


def test_write_records_time_step_and_fldlist(tmp_path):
    arr = np.zeros((2, 3), dtype=np.float32)
    mitgcm_io.write_mds(
        tmp_path / "THETA", arr, iteration=20, fld_list=("THETA",), time_step=20
    )
    _, meta = mitgcm_io.read_mds(tmp_path / "THETA", iteration=20)
    assert meta.time_step == 20
    assert meta.fld_list == ("THETA",)


def test_parse_real_meta_format():
    """Parse the exact MITgcm .meta layout (multi-line dimList, etc.)."""
    text = (
        " nDims = [   2 ];\n"
        " dimList = [\n"
        "    62,    1,   62,\n"
        "    62,    1,   62\n"
        " ];\n"
        " dataprec = [ 'float32' ];\n"
        " nrecords = [     1 ];\n"
        " timeStepNumber = [         10 ];\n"
    )
    meta = mitgcm_io.parse_meta(text)
    assert meta.dims == (62, 62)
    assert meta.dataprec == "float32"
    assert meta.nrecords == 1
    assert meta.time_step == 10
    assert meta.fld_list is None


def test_parse_meta_missing_required_raises():
    with pytest.raises(ValueError, match="missing required"):
        mitgcm_io.parse_meta(" dataprec = [ 'float32' ];\n")


def test_read_missing_pair_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        mitgcm_io.read_mds(tmp_path / "absent")


def test_size_mismatch_raises(tmp_path):
    """A .meta whose dims disagree with the .data byte count must fail loudly."""
    (tmp_path / "X.meta").write_text(
        " nDims = [ 2 ];\n dimList = [ 4, 1, 4, 4, 1, 4 ];\n"
        " dataprec = [ 'float32' ];\n nrecords = [ 1 ];\n"
    )
    np.arange(3, dtype=">f4").tofile(tmp_path / "X.data")  # 3 != 16 elements
    with pytest.raises(ValueError, match="elements"):
        mitgcm_io.read_mds(tmp_path / "X")


def test_squeeze_preserves_singleton_vertical_axis(tmp_path):
    """A genuine nz=1 field must stay 3-D, not collapse to 2-D (vs np.squeeze)."""
    arr = np.arange(35, dtype=np.float32).reshape(1, 5, 7)  # (nz=1, ny, nx)
    mitgcm_io.write_mds(tmp_path / "U", arr)
    back, _ = mitgcm_io.read_mds(tmp_path / "U")
    assert back.shape == (1, 5, 7)  # not (5, 7)
    np.testing.assert_array_equal(back, arr)


def test_squeeze_preserves_singleton_horizontal_axis(tmp_path):
    """A zonal slice (nx=1) keeps its rank so axes stay identifiable."""
    arr = np.arange(3, dtype=np.float32).reshape(3, 1)  # (ny=3, nx=1)
    mitgcm_io.write_mds(tmp_path / "V", arr)
    back, _ = mitgcm_io.read_mds(tmp_path / "V")
    assert back.shape == (3, 1)


def test_write_rejects_4d_array(tmp_path):
    with pytest.raises(ValueError, match="ndim=4"):
        mitgcm_io.write_mds(tmp_path / "bad", np.zeros((2, 2, 2, 2)))


def test_write_rejects_multifield_fldlist(tmp_path):
    with pytest.raises(ValueError, match="exactly\n?\\s*one field|one field"):
        mitgcm_io.write_mds(
            tmp_path / "bad", np.zeros((2, 3)), fld_list=("THETA", "SALT")
        )


def test_read_rejects_fldlist_nrecords_mismatch(tmp_path):
    """fldList length must divide nrecords, else the dump is corrupt."""
    (tmp_path / "D.meta").write_text(
        " nDims = [ 2 ];\n dimList = [ 2, 1, 2, 2, 1, 2 ];\n"
        " dataprec = [ 'float32' ];\n nrecords = [ 3 ];\n"
        " fldList = { 'A' 'B' };\n"  # 3 % 2 != 0
    )
    np.zeros(3 * 4, dtype=">f4").tofile(tmp_path / "D.data")
    with pytest.raises(ValueError, match="not a multiple of"):
        mitgcm_io.read_mds(tmp_path / "D")
