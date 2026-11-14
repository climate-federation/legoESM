"""Contract tests for the resumable ORAS5 downloader (no network)."""
import json
import sys
import threading
import zipfile
from pathlib import Path

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "data"))

import download_oras5 as d  # noqa: E402


def _nc(path):
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("x", 3)
        ds.createVariable("sosstsst", "f4", ("x",))[:] = np.arange(3)


def test_requests_split_single_by_year_and_all_levels_by_month():
    reqs = d.build_requests(range(2000, 2002), ["sea_surface_height"], ["salinity"])
    single = [r for r in reqs if r["vertical_resolution"] == "single_level"]
    full = [r for r in reqs if r["vertical_resolution"] == "all_levels"]
    assert len(single) == 2 and all(r["month"] == d.MONTHS for r in single)
    assert len(full) == 24 and {tuple(r["month"]) for r in full} == {(m,) for m in d.MONTHS}
    assert all(r["product_type"] == ["consolidated"] for r in reqs)


def test_years_past_the_consolidated_product_are_refused():
    with pytest.raises(ValueError, match="2014"):
        d.build_requests(range(2014, 2016), ["sea_surface_height"], [])


class FakeClient:
    calls = []

    def retrieve(self, name, req, target):
        FakeClient.calls.append(req)
        nc = Path(target).with_name("member.nc")
        _nc(nc)
        with zipfile.ZipFile(target, "w") as z:
            z.write(nc, arcname="sosstsst_2000.nc")
        nc.unlink()


def test_fetch_unpacks_records_and_a_rerun_skips(tmp_path):
    req = d.build_requests([2000], ["sea_surface_temperature"], [])[0]
    manifest = tmp_path / "manifest.jsonl"
    d.fetch_one(FakeClient, req, tmp_path, manifest, threading.Lock())
    assert (tmp_path / "sea_surface_temperature" / "sosstsst_2000.nc").exists()
    assert not list((tmp_path / "_incoming").iterdir())
    row = json.loads(manifest.read_text())
    assert row["request"] == req and set(row["files"]) == {"sosstsst_2000.nc"}
    assert d.request_key(req) in d.done_keys(manifest)


def test_archive_member_with_a_path_is_refused(tmp_path):
    nc = tmp_path / "a.nc"
    _nc(nc)
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.write(nc, arcname="../escape.nc")
    with pytest.raises(ValueError, match="unexpected archive member"):
        d.unpack(bad, tmp_path / "out")
    assert not (tmp_path / "escape.nc").exists()


def test_unreadable_member_does_not_count(tmp_path):
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("broken.nc", b"not a netcdf file")
    with pytest.raises(OSError):
        d.unpack(bad, tmp_path / "out")
    assert not (tmp_path / "out" / "broken.nc").exists()
