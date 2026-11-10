"""Direct test of the Cariolle coefficient builder: the big-endian record
reader round-trips a synthetic updo3ch-layout file, and the deck pressures it
writes are the ones the packaged file carries."""
from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

_PY = Path(__file__).resolve().parents[2] / "scripts/data/build_ifs_cariolle_coefficients.py"


def _mod():
    spec = importlib.util.spec_from_file_location("build_ifs_cariolle_coefficients", _PY)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def _rec(body):
    return struct.pack(">i", len(body)) + body + struct.pack(">i", len(body))


def _o3chem_bytes(month, coef):
    ncoef, nlev, nlat = coef.shape
    out = _rec(struct.pack(">i", month)) + _rec(struct.pack(">i", ncoef))
    for j in range(ncoef):
        out += _rec(struct.pack(">3i", j + 1, nlev, nlat)) + _rec(b"name    ")
        for k in range(nlev):
            out += _rec(coef[j, k].astype(">f8").tobytes())
    return out


def test_read_o3chem_round_trips_and_refuses_bad_files(tmp_path):
    m = _mod()
    coef = np.random.default_rng(0).standard_normal((m.N_COEF, 3, 4))
    f = tmp_path / "O3CHEM05_H1.0"
    f.write_bytes(_o3chem_bytes(5, coef))
    month, got = m.read_o3chem(f)
    assert month == 5
    np.testing.assert_array_equal(got, coef)
    f.write_bytes(_o3chem_bytes(5, coef) + _rec(b"x"))
    with pytest.raises(ValueError, match="trailing"):
        m.read_o3chem(f)
    raw = bytearray(_o3chem_bytes(5, coef))
    raw[-1] ^= 1
    f.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="marker"):
        m.read_o3chem(f)


def test_gaussian_latitudes_and_deck_pressures_match_packaged_file():
    m = _mod()
    lat = m.gaussian_latitudes_deg(64)
    assert lat[0] > 0 > lat[-1] and np.allclose(lat, -lat[::-1])
    from legoesm.atmosphere.physics.radiation import ozone_cariolle as oc
    with np.load(oc.CARIOLLE_FILE) as d:
        np.testing.assert_array_equal(m.deck_standard_pressures("cam_l32"),
                                      d["p_std_full"])
        np.testing.assert_allclose(lat, d["lat_deg"], rtol=0, atol=1e-12)
    with pytest.raises(SystemExit):
        m.deck_standard_pressures("hybrid")
