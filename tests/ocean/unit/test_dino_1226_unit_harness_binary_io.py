"""Direct unit test for the #1226 NEMO unit-call harness's Python-side binary
reader/writer (``scripts/validate/ocean_fidelity/dino_1226/unit_harness/binary_io.py``).

This is the Python mirror of the Fortran driver's self-describing raw fp64
format (``cfgs/DINO/UNIT_HARNESS/unit_harness_io.F90``). Tested here in
isolation (no NEMO build, no legoESM ocean state) -- the Fortran<->Python
cross-language round-trip is exercised separately by
``unit_harness/run_eos_rab_probe.py``'s own round-trip self-check, which
requires the compiled NEMO harness binaries and so is not a pytest target.
"""
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
from validate.ocean_fidelity.dino_1226.unit_harness import binary_io  # noqa: E402

sys.path.remove(str(SCRIPTS_DIR))


def test_roundtrip_3d(tmp_path):
    arr = np.arange(4 * 5 * 3, dtype=np.float64).reshape(4, 5, 3) + 0.5
    path = tmp_path / "a.bin"
    binary_io.write_array(str(path), arr)
    got = binary_io.read_array(str(path), (4, 5, 3))
    np.testing.assert_array_equal(got, arr)
    assert got.dtype == np.float64


def test_roundtrip_4d_fortran_order_not_confused_with_c_order(tmp_path):
    # asymmetric shape + asymmetric values: a C/Fortran order bug would
    # corrupt this (unlike a symmetric or constant array, which cannot
    # distinguish the two layouts)
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((2, 3, 4, 5))
    path = tmp_path / "b.bin"
    binary_io.write_array(str(path), arr)
    got = binary_io.read_array(str(path), (2, 3, 4, 5))
    np.testing.assert_array_equal(got, arr)


def test_shape_mismatch_raises(tmp_path):
    arr = np.zeros((4, 5, 3))
    path = tmp_path / "c.bin"
    binary_io.write_array(str(path), arr)
    with pytest.raises(ValueError, match="shape mismatch"):
        binary_io.read_array(str(path), (4, 5, 4))


def test_ndim_mismatch_raises(tmp_path):
    arr = np.zeros((4, 5, 3))
    path = tmp_path / "d.bin"
    binary_io.write_array(str(path), arr)
    with pytest.raises(ValueError, match="ndim mismatch"):
        binary_io.read_array(str(path), (4, 5))


def test_scalar_int_roundtrip(tmp_path):
    path = tmp_path / "e.bin"
    binary_io.write_scalar_int(str(path), 2)
    assert binary_io.read_scalar_int(str(path)) == 2


def test_scalar_int_rejects_non_scalar_file(tmp_path):
    arr = np.zeros((2, 2))
    path = tmp_path / "f.bin"
    binary_io.write_array(str(path), arr)
    with pytest.raises(ValueError, match="ndim=0"):
        binary_io.read_scalar_int(str(path))


def test_header_matches_fortran_little_endian_int64_convention(tmp_path):
    # Pin the byte layout explicitly against Python's struct module (not
    # just numpy's own round-trip) so a numpy default-dtype/byte-order
    # regression cannot silently break cross-language compatibility.
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    path = tmp_path / "g.bin"
    binary_io.write_array(str(path), arr)
    raw = path.read_bytes()
    ndim = struct.unpack_from("<q", raw, 0)[0]
    assert ndim == 2
    shape = struct.unpack_from("<qq", raw, 8)
    assert shape == (2, 2)
    payload = struct.unpack_from("<4d", raw, 24)
    # Fortran (column-major) order: [[1,2],[3,4]] -> 1,3,2,4
    assert payload == (1.0, 3.0, 2.0, 4.0)
