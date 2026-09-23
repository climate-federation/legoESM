"""Tests for the runoff stratification of the coastal salinity gap.

Every test below is built so it FAILS if the piece it covers is removed: the
weighting test uses areas that make the weighted and unweighted answers
differ, the additivity test plants error in disjoint strata, and the
end-to-end test plants ALL of the excess in river cells so a stratifier that
ignored runoff would report the area share instead of 100%.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_PROBE = (Path(__file__).resolve().parents[2]
          / "scripts" / "validate" / "ocean_fidelity" / "coastal_sss_by_runoff.py")


def _mod():
    spec = importlib.util.spec_from_file_location("_coastal_runoff", _PROBE)
    m = importlib.util.module_from_spec(spec)
    sys.modules["_coastal_runoff"] = m
    spec.loader.exec_module(m)
    return m


def test_wrms_actually_weights_by_area():
    m = _mod()
    err = np.array([[1.0, 3.0]])
    sel = np.ones_like(err, dtype=bool)
    # Equal areas: plain rms of 1 and 3 is sqrt(5).
    assert m.wrms(err, np.array([[1.0, 1.0]]), sel) == pytest.approx(np.sqrt(5.0))
    # Nine times the area on the small error must pull it DOWN, well away from
    # the unweighted answer -- this is what fails if the weights are dropped.
    got = m.wrms(err, np.array([[9.0, 1.0]]), sel)
    assert got == pytest.approx(np.sqrt((9 * 1 + 1 * 9) / 10.0))
    assert got < np.sqrt(5.0) - 0.4


def test_wsse_adds_across_disjoint_strata():
    m = _mod()
    err = np.array([[1.0, 2.0, 3.0, 4.0]])
    area = np.array([[1.0, 2.0, 3.0, 4.0]])
    whole = np.ones_like(err, dtype=bool)
    left = np.array([[True, True, False, False]])
    right = np.array([[False, False, True, True]])
    assert (m.wsse(err, area, left) + m.wsse(err, area, right)
            == pytest.approx(m.wsse(err, area, whole)))


def test_empty_selection_raises_rather_than_returning_nan():
    m = _mod()
    err = np.array([[1.0, 2.0]])
    area = np.array([[1.0, 1.0]])
    with pytest.raises(SystemExit):
        m.wrms(err, area, np.zeros_like(err, dtype=bool))


def _write_case(tmp_path, excess_only_on_rivers: bool):
    """A 6x12 box: one edge ring, a river in two cells, planted error."""
    nc = pytest.importorskip("netCDF4")
    ny, nx = 6, 12
    lat = np.linspace(-20.0, 20.0, ny)
    lon = np.linspace(0.0, 330.0, nx)
    lon2d, lat2d = np.meshgrid(lon, lat)
    area = np.cos(np.deg2rad(lat2d))
    scored = np.ones((ny, nx), dtype=bool)
    ring = np.full((ny, nx), 9, dtype=np.int16)
    ring[0, :] = 1
    ring[-1, :] = 1

    river = np.zeros((ny, nx), dtype=bool)
    river[0, 2] = river[0, 3] = True

    nemo = np.zeros((ny, nx))
    a = np.zeros((ny, nx))
    b = np.zeros((ny, nx))
    if excess_only_on_rivers:
        b[river] = 2.0          # all of lane b's excess sits on the rivers
    else:
        b[ring == 1] = 2.0      # uniform across the ring

    np.savez(tmp_path / "dump.npz", lat2d=lat2d, lon2d=lon2d, area=area,
             edge_ring=ring, scored=scored, SSS_a=a, SSS_b=b, SSS_nemo=nemo,
             label_a="trip", label_b="fes")

    rf = tmp_path / "runoff.nc"
    with nc.Dataset(rf, "w") as f:
        f.createDimension("y", ny)
        f.createDimension("x", nx)
        f.createDimension("time_counter", 1)
        v = f.createVariable("sorunoff", "f8", ("time_counter", "y", "x"))
        r = np.zeros((ny, nx))
        r[river] = 1.0
        v[:] = r[None]
        f.createVariable("nav_lat", "f8", ("y", "x"))[:] = lat2d
        f.createVariable("nav_lon", "f8", ("y", "x"))[:] = lon2d
    return tmp_path / "dump.npz", rf


def _run(dump, rf):
    out = subprocess.run(
        [sys.executable, str(_PROBE), "--dump", str(dump),
         "--runoff-file", str(rf), "--ring", "1",
         "--quantiles", "50", "--max-deg", "5.0"],
        capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    return out.stdout


def _zero_row(txt):
    rows = [l for l in txt.splitlines() if l.startswith("zero runoff")]
    assert rows, txt
    f = rows[0].split()
    # "zero runoff" is two words, so: cells area a b excess share% lift
    return {"cells": int(f[2]), "area": float(f[3]),
            "share": float(f[-2].rstrip("%")), "lift": float(f[-1])}


def test_nearest_source_does_not_smear_rivers_across_the_ring(tmp_path):
    """The fix that made this probe able to discriminate at all.

    Only two of the ring's 24 cells carry a river. Under nearest-source the
    other 22 must land in the zero stratum; an inverse-distance average over a
    multi-degree radius puts a nonzero value nearly everywhere and collapses
    that stratum, which is exactly how the first version of this test came out
    vacuous on real data.
    """
    dump, rf = _write_case(tmp_path, excess_only_on_rivers=True)
    z = _zero_row(_run(dump, rf))
    assert z["cells"] == 22, z
    assert z["area"] > 0.8, z


def test_river_only_excess_is_attributed_to_the_river_stratum(tmp_path):
    dump, rf = _write_case(tmp_path, excess_only_on_rivers=True)
    txt = _run(dump, rf)
    assert "recombine OK" in txt
    z = _zero_row(txt)
    # Runoff-free cells are most of the ring by area, so a stratifier blind to
    # runoff would park most of the gap here. It must get none, and the lift
    # says so independently of how big the stratum is.
    assert z["share"] == pytest.approx(0.0, abs=1e-6)
    assert z["lift"] == pytest.approx(0.0, abs=1e-6)


def test_uniform_excess_is_not_attributed_to_rivers(tmp_path):
    dump, rf = _write_case(tmp_path, excess_only_on_rivers=False)
    txt = _run(dump, rf)
    assert "recombine OK" in txt
    z = _zero_row(txt)
    # Same planted magnitude, spread evenly over the ring: the runoff-free
    # cells must now carry it in proportion to their area, i.e. lift 1. The
    # two cases differ ONLY in where the error sits, so a probe that ignored
    # runoff cannot pass both.
    assert z["share"] > 0.5
    assert z["lift"] == pytest.approx(1.0, abs=0.15)
