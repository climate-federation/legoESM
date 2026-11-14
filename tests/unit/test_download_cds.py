"""Contract tests for the resumable Copernicus downloader (no network)."""
import json
import sys
import threading
import zipfile
from pathlib import Path

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "data"))

import download_cds as d  # noqa: E402


def _nc(path):
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("x", 3)
        ds.createVariable("sosstsst", "f4", ("x",))[:] = np.arange(3)


def test_oras5_requests_split_single_by_year_and_all_levels_by_month():
    jobs = d.build_requests(range(2000, 2002), ["sea_surface_height"], ["salinity"])
    single = [r for _, r, _, _ in jobs if r["vertical_resolution"] == "single_level"]
    full = [r for _, r, _, _ in jobs if r["vertical_resolution"] == "all_levels"]
    assert len(single) == 2 and all(r["month"] == d.MONTHS for r in single)
    assert len(full) == 24 and {tuple(r["month"]) for r in full} == {(m,) for m in d.MONTHS}
    assert all(r["product_type"] == ["consolidated"] for _, r, _, _ in jobs)
    assert {sub for _, _, sub, _ in jobs} == {"sea_surface_height", "salinity"}


def test_oras5_request_keys_are_unchanged_for_the_running_job():
    # the first ORAS5 job's manifest keys were written by the pre-rename script
    req = d.build_requests([1993], ["sea_surface_height"], [])[0][1]
    assert d.request_key(req) == json.dumps(
        {"product_type": ["consolidated"], "vertical_resolution": "single_level",
         "variable": ["sea_surface_height"], "year": ["1993"], "month": d.MONTHS},
        sort_keys=True)


def test_years_past_the_consolidated_product_are_refused():
    with pytest.raises(ValueError, match="2014"):
        d.build_requests(range(2014, 2016), ["sea_surface_height"], [])


def test_era5_requests_every_hour_for_accumulations_and_valid_days():
    jobs = d.build_era5_requests(range(1996, 1998))  # 1996 is a leap year
    assert len(jobs) == 1 + 2 * 12 * 2
    by_tag = {tag: (req, sub) for _, req, sub, tag in jobs}
    inst, sub_i = by_tag["era5_instant_199602"]
    acc, sub_a = by_tag["era5_accum_199602"]
    assert (sub_i, sub_a) == ("instant", "accum")
    assert inst["time"] == [f"{h:02d}:00" for h in range(0, 24, 3)]
    assert acc["time"] == [f"{h:02d}:00" for h in range(24)]
    assert len(inst["day"]) == 29 and len(by_tag["era5_accum_199702"][0]["day"]) == 28
    assert set(inst["variable"]) == set(d.ERA5_INSTANT) and set(acc["variable"]) == set(d.ERA5_ACCUM)
    assert all(r["grid"] == [1.0, 1.0] for _, r, _, _ in jobs)
    assert by_tag["era5_land_sea_mask"][1] == "static"


class FakeZipClient:
    def retrieve(self, name, req, target):
        nc = Path(target).with_name("member.nc")
        _nc(nc)
        with zipfile.ZipFile(target, "w") as z:
            z.write(nc, arcname="sosstsst_2000.nc")
        nc.unlink()


class FakeNetcdfClient:
    def retrieve(self, name, req, target):
        _nc(target)


def test_fetch_unpacks_records_and_a_rerun_skips(tmp_path):
    job = d.build_requests([2000], ["sea_surface_temperature"], [])[0]
    manifest = tmp_path / "manifest.jsonl"
    d.fetch_one(FakeZipClient, job, tmp_path, manifest, threading.Lock())
    assert (tmp_path / "sea_surface_temperature" / "sosstsst_2000.nc").exists()
    assert not list((tmp_path / "_incoming").iterdir())
    row = json.loads(manifest.read_text())
    assert row["request"] == job[1] and set(row["files"]) == {"sosstsst_2000.nc"}
    assert d.request_key(job[1]) in d.done_keys(manifest)


def test_bare_netcdf_result_is_named_by_its_tag(tmp_path):
    job = [j for j in d.build_era5_requests([2000]) if j[3] == "era5_accum_200003"][0]
    d.fetch_one(FakeNetcdfClient, job, tmp_path, tmp_path / "m.jsonl", threading.Lock())
    assert (tmp_path / "accum" / "era5_accum_200003.nc").exists()
    assert not list((tmp_path / "_incoming").iterdir())


def test_archive_member_with_a_path_is_refused(tmp_path):
    nc = tmp_path / "a.nc"
    _nc(nc)
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.write(nc, arcname="../escape.nc")
    with pytest.raises(ValueError, match="unexpected archive member"):
        d.unpack(bad, tmp_path / "out")
    assert not (tmp_path / "escape.nc").exists()


def test_unreadable_result_does_not_count(tmp_path):
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("broken.nc", b"not a netcdf file")
    with pytest.raises(OSError):
        d.unpack(bad, tmp_path / "out")
    assert not (tmp_path / "out" / "broken.nc").exists()
    raw = tmp_path / "raw.download"
    raw.write_bytes(b"CDF? no")
    with pytest.raises(OSError):
        d.place(raw, tmp_path / "out2", "tag")
    assert not (tmp_path / "out2" / "tag.nc").exists()
