import functools
import http.server
import json
import threading
import traceback
import zipfile

import netCDF4
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from arm_sgp_downloader import (config, dataset_builder, discovery, download, extract, metadata, netcdf_reader,
                                quality_control, report, time_alignment)


def write_nc(path, values, seconds, bounds=None, name="corrected_sensible_heat_flux", extra=None):
    with netCDF4.Dataset(path, "w") as data:
        data.createDimension("time", len(values))
        data.createDimension("bound", 2)
        time = data.createVariable("time", "f8", ("time",))
        time.units = "seconds since 2014-06-16 00:00:00 0:00"
        time[:] = seconds
        if bounds is not None:
            data.createVariable("time_bounds", "f8", ("time", "bound"))[:] = bounds
        data.createVariable(name, "f8", ("time",))[:] = values
        for key, value in (extra or {}).items():
            data.createVariable(key, "f8", ("time",))[:] = value


def test_facility_filter_keeps_only_central_facility():
    tree = [{"path": f"data/{n}", "type": "blob", "size": 1, "sha": "a"} for n in
            ("sgp30ecorE14.b1.20190601.000000.cdf", "sgp30ecorE6.b1.20040705.000000.cdf",
             "sgpswatsE8.b1.20071229.000700.cdf")]
    assert [e["filename"] for e in discovery.github_entries(tree)] == ["sgp30ecorE14.b1.20190601.000000.cdf"]


def test_bounds_give_interval_end(tmp_path):
    path = tmp_path / "a.nc"
    write_nc(path, [1.0, 2.0], [900.0, 2700.0], [[0, 1800], [1800, 3600]], name="sensible_heat_flux")
    table = netcdf_reader.read_file(path, "sgpco2flx4mC1.b1")
    assert table.index[0] == pd.Timestamp("2014-06-16 00:30", tz="UTC")


def test_qc_ecor_time_is_start(tmp_path):
    path = tmp_path / "a.nc"
    write_nc(path, [1.0, 2.0], [0.0, 1800.0], [[-1800, 0], [0, 1800]])
    assert netcdf_reader.read_file(path, "sgp30qcecorE14.c1").index[0] == pd.Timestamp("2014-06-16 00:30", tz="UTC")


def test_start_stamp_moves_to_end(tmp_path):
    path = tmp_path / "a.nc"
    with netCDF4.Dataset(path, "w") as data:
        data.createDimension("time", 1)
        stamp = data.createVariable("yyyydddhhmmss", "f8", ("time",))
        stamp.long_name = "start of integration interval"
        stamp[:] = [2011112000000.0]
        data.createVariable("h", "f8", ("time",))[:] = [5.0]
    assert netcdf_reader.read_file(path, "sgp30co2flx4mC1.b1").index[0] == pd.Timestamp("2011-04-22 00:30", tz="UTC")


def test_ebbr_flux_sign_becomes_upward(tmp_path):
    path = tmp_path / "a.nc"
    write_nc(path, [-150.0], [1800.0], name="sensible_heat_flux")
    assert netcdf_reader.read_file(path, "sgp30baebbrE13.c1")["H"].iloc[0] == 150.0


def test_sentinel_is_missing(tmp_path):
    path = tmp_path / "a.nc"
    write_nc(path, [-9999.0, 3.0], [0.0, 1800.0], [[0, 1800], [1800, 3600]], name="sensible_heat_flux")
    assert netcdf_reader.read_file(path, "sgpco2flx4mC1.b1")["H"].isna().tolist() == [True, False]


def one_minute(n, start="2014-06-16 00:01"):
    return pd.DataFrame({"T_2": np.arange(n, dtype=float)}, index=pd.date_range(start, periods=n, freq="1min", tz="UTC"))


def test_full_interval_is_averaged_into_end_label():
    grid = time_alignment.to_grid(one_minute(30), 60)
    assert grid.index[0] == pd.Timestamp("2014-06-16 00:30", tz="UTC") and grid["T_2"].iloc[0] == 14.5


def test_sparse_interval_stays_missing_and_gap_is_not_filled():
    grid = time_alignment.to_grid(pd.concat([one_minute(30), one_minute(5, "2014-06-16 03:01")]), 60)
    assert grid["T_2"].isna().tolist() == [False, True]
    assert pd.Timestamp("2014-06-16 01:00", tz="UTC") not in grid.index


def test_filter_blanks_flagged_values_only_in_copy():
    raw = pd.DataFrame({"H": [10.0, 20.0], "qc_fail_H": [0.0, 1.0]})
    clean = quality_control.apply_flags(raw)
    assert clean["H"].isna().tolist() == [False, True] and raw["H"].notna().all()


def test_limits_blank_impossible_values():
    assert quality_control.apply_flags(pd.DataFrame({"T_2": [20.0, 90.0]}))["T_2"].isna().tolist() == [False, True]


def test_wind_from_north_blows_south():
    u, v = dataset_builder.wind_components(np.array([5.0]), np.array([0.0]))
    assert abs(u[0]) < 1e-12 and v[0] == -5.0


def test_soil_averages_pits_per_depth():
    index = pd.date_range("2020-01-01", periods=2, freq="30min", tz="UTC")
    table = pd.DataFrame({"Tsoil_W_5": [1.0, 2.0], "Tsoil_E_5": [3.0, np.nan]}, index=index)
    assert dataset_builder.soil({"x": table}, "Tsoil")[5.0].tolist() == [2.0, 2.0]


def test_soil_first_stream_wins_at_a_shared_depth():
    index = pd.date_range("2020-01-01", periods=2, freq="30min", tz="UTC")
    first = pd.DataFrame({"SM_W_5": [0.1, np.nan]}, index=index)
    second = pd.DataFrame({"SM_W_5": [0.9, 0.8], "SM_W_15": [0.3, 0.3]}, index=index)
    result = dataset_builder.soil({"a": first, "b": second}, "SM")
    assert result[5.0].tolist() == [0.1, 0.8]
    assert result[15.0].tolist() == [0.3, 0.3]


def test_stamp_is_preferred_over_swats():
    assert dataset_builder.SOIL_STREAMS == ("sgpstampE13.b1", "sgpswatsE13.b1")


def test_merged_never_fills_from_absent_stream():
    assert dataset_builder.merged({}, [("a", "H")]) is None


def test_build_keeps_unmeasured_heights_empty(tmp_path):
    folder = tmp_path / "raw" / "armlive" / "sgp30twrmrC1.c1"
    folder.mkdir(parents=True)
    extra = {"temp_25m": [21.0, 22.0], "temp_60m": [20.0, 21.0]}
    write_nc(folder / "a.nc", [22.0, 23.0], [1800.0, 3600.0], name="temp_02m", extra=extra)
    ds, _ = dataset_builder.build(tmp_path)
    assert ds.T.sel(height=4).isnull().all() and ds.T.sel(height=25).notnull().all()
    assert (tmp_path / "unified" / "sgp_cf_30min.nc").exists()


@pytest.fixture
def server(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    (site / "f.bin").write_bytes(b"CDF" + b"x" * 5000)

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=site))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}/", site
    httpd.shutdown()


def entry(url, checksum=None):
    return {"source": "s", "url": url + "f.bin", "filename": "f.bin", "datastream": "d", "size": None, "checksum": checksum}


def test_existing_original_is_never_overwritten(server, tmp_path):
    url, _ = server
    root = tmp_path / "root"
    assert download.fetch(entry(url), root)["status"] == "downloaded"
    dest = root / "raw" / "s" / "f.bin"
    dest.chmod(0o644)
    dest.write_bytes(b"kept")
    assert download.fetch(entry(url), root)["status"] == "present" and dest.read_bytes() == b"kept"


def test_bad_checksum_is_discarded(server, tmp_path):
    url, _ = server
    record = download.fetch(entry(url, "md5:" + "0" * 32), tmp_path / "root")
    assert record["status"] == "failed" and not (tmp_path / "root" / "raw" / "s" / "f.bin").exists()


def test_gitblob_checksum_passes(tmp_path):
    path = tmp_path / "x"
    path.write_bytes(b"hello\n")
    assert download.digest(path, "gitblob:ce013625030ba8dba906f756967f9e9ca394464a")


def test_html_error_page_is_rejected_for_authenticated_route(server, tmp_path, monkeypatch):
    url, site = server
    (site / "f.bin").write_bytes(b"<html>error</html>")
    monkeypatch.setattr(config, "ARMLIVE_URL", url)
    item = dict(entry(url), auth="armlive")
    monkeypatch.setattr(download, "request_url", lambda e, c: url + "f.bin")
    assert download.fetch(item, tmp_path / "root", "user:secret")["status"] == "failed"


def test_credential_never_reaches_catalogue(server, tmp_path, monkeypatch):
    url, _ = server
    monkeypatch.setattr(download, "request_url", lambda e, c: url + "f.bin")
    monkeypatch.setattr(discovery, "armlive_credential", lambda: "user:secret")
    root = tmp_path / "root"
    root.mkdir()
    download.run(root, [dict(entry(url), auth="armlive")])
    assert "secret" not in (root / "catalogue.jsonl").read_text()


def test_request_url_carries_user_first():
    url = download.request_url({"auth": "armlive", "filename": "f.nc"}, "a:b")
    assert url.endswith("saveData?user=a:b&file=f.nc")


def test_unsafe_zip_refused(tmp_path):
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../evil.txt", "x")
    with pytest.raises(ValueError):
        extract.unpack(archive)


def test_zip_keeps_original(tmp_path):
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("b.txt", "x")
    assert (extract.unpack(archive) / "b.txt").exists() and archive.exists()


def test_catalogue_round_trip(tmp_path):
    metadata.append(tmp_path, {"status": "x"})
    assert metadata.read(tmp_path)[0]["status"] == "x"


def test_closure_of_perfect_budget():
    ds = xr.Dataset({"Rn": ("t", [100.0, 300.0, 500.0, 700.0]), "G": ("t", [0.0, 0.0, 0.0, 0.0]),
                     "H_surface": ("t", [40.0, 120.0, 200.0, 280.0]), "LE_surface": ("t", [60.0, 180.0, 300.0, 420.0])})
    assert abs(quality_control.closure(ds)["slope"] - 1.0) < 1e-9


def test_lag_detects_shift():
    series = np.sin(np.arange(300) / 7.0)
    assert quality_control.lag_of_best_match(series[:-2], series[2:]) == -2 or quality_control.lag_of_best_match(series[:-2], series[2:]) == 2


def test_mixed_stream_wind_is_not_built():
    index = pd.date_range("2020-01-01", periods=2, freq="30min", tz="UTC")
    old = pd.DataFrame({"wspd": [5.0, 6.0], "wdir": [np.nan, 90.0]}, index=index)
    new = pd.DataFrame({"wspd": [1.0, 1.0], "wdir": [180.0, 180.0]}, index=index)
    frame = dataset_builder.merged_records({"old": old, "new": new}, ("old", "new"), ("wspd", "wdir"))
    assert frame.wspd.tolist() == [1.0, 6.0] and frame.wdir.tolist() == [180.0, 90.0]


def test_source_codes_name_the_winning_stream():
    index = pd.date_range("2020-01-01", periods=3, freq="30min", tz="UTC")
    tables = {"a": pd.DataFrame({"H": [1.0, np.nan, np.nan]}, index=index),
              "b": pd.DataFrame({"H": [2.0, 2.0, np.nan]}, index=index)}
    assert dataset_builder.source_codes(tables, [("a", "H"), ("b", "H")], index).tolist() == [0, 1, -1]


def test_radiation_limits_apply():
    table = pd.DataFrame({"SW_down": [500.0, 5000.0], "LW_down": [300.0, 5000.0]})
    clean = quality_control.apply_flags(table)
    assert clean.SW_down.isna().tolist() == [False, True] and clean.LW_down.isna().tolist() == [False, True]


def test_query_failure_hides_credential(monkeypatch):
    def refuse(url, **_):
        raise discovery.requests.ConnectionError(f"could not reach {url}")
    monkeypatch.setattr(discovery.requests, "get", refuse)
    credential = "user:" + "secret"  # a literal here would appear in the traceback's source line
    with pytest.raises(ConnectionError) as caught:
        discovery.armlive_files("s", credential, 1993)
    assert "secret" not in "".join(traceback.format_exception(caught.value))


class Reply:
    def __init__(self, status, headers, body=b""):
        self.status_code, self.headers, self.body = status, headers, body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def iter_content(self, _):
        yield self.body


def partial(root):
    part = root / "raw" / "s" / "f.bin.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(b"CDF")
    return part


@pytest.mark.parametrize("reply", [Reply(416, {}), Reply(206, {"content-range": "bytes 0-5/6"}, b"CDFabc")])
def test_bad_resume_discards_partial(tmp_path, monkeypatch, reply):
    part = partial(tmp_path)
    monkeypatch.setattr(download.requests, "get", lambda *a, **k: reply)
    assert download.fetch(entry("http://x/"), tmp_path)["status"] == "failed" and not part.exists()


def test_good_resume_appends(tmp_path, monkeypatch):
    partial(tmp_path)
    reply = Reply(206, {"content-range": "bytes 3-5/6"}, b"abc")
    monkeypatch.setattr(download.requests, "get", lambda *a, **k: reply)
    assert download.fetch(entry("http://x/"), tmp_path)["status"] == "downloaded"
    assert (tmp_path / "raw" / "s" / "f.bin").read_bytes() == b"CDFabc"


def test_original_appearing_during_download_survives(tmp_path, monkeypatch):
    dest = tmp_path / "raw" / "s" / "f.bin"

    def racing_get(*_, **__):
        dest.write_bytes(b"kept")
        return Reply(200, {}, b"new")
    dest.parent.mkdir(parents=True)
    monkeypatch.setattr(download.requests, "get", racing_get)
    assert download.fetch(entry("http://x/"), tmp_path)["status"] == "conflict" and dest.read_bytes() == b"kept"


def test_second_download_run_is_refused(tmp_path):
    with download.only_run(tmp_path), pytest.raises(RuntimeError):
        download.run(tmp_path, [])
    assert not (tmp_path / ".download.lock").exists()


def flagged(path, flags, attrs=None, global_attrs=None):
    data = netCDF4.Dataset(path, "w")
    data.createDimension("time", len(flags))
    variable = data.createVariable("qc_x", "i4", ("time",))
    variable[:] = flags
    variable.setncatts(attrs or {})
    data.setncatts(global_attrs or {})
    return data, variable


def test_grades_reject_bad_and_orientation_only(tmp_path):
    attrs = {"flag_1_assessment": "Acceptable", "flag_1_description": "Fundamental research quality",
             "flag_5_assessment": "Indeterminate", "flag_5_description": "General use only",
             "flag_7_assessment": "Indeterminate", "flag_7_description": "high; Use for orientation only",
             "flag_9_assessment": "Bad", "flag_9_description": "Do not use"}
    data, variable = flagged(tmp_path / "a.nc", [1, 5, 7, 9, -9999], attrs)
    with data:
        result = netcdf_reader.rejected(data, variable, np.array([1, 5, 7, 9, -9999]))
    assert result.tolist() == [False, False, True, True, False]


def test_mentor_bits_keep_suspect_values(tmp_path):
    data, variable = flagged(tmp_path / "a.nc", [0, 4, 6, 8, 32, -9999],
                             global_attrs={"MENTOR_QC_FIELD_INFORMATION": "text"})
    with data:
        result = netcdf_reader.rejected(data, variable, np.array([0, 4, 6, 8, 32, -9999]))
    assert result.tolist() == [False, True, True, False, False, False]


def test_text_bits_keep_only_the_jump_test(tmp_path):
    text = ("0x0 = fine\n0x1 = value is equal to 'missing_value'\n0x8 = value failed the 'valid_delta' check\n"
            "0x10 = value unusable as another field it depends on failed\n")
    flags = [0, 1, 8, 16, 24, 32]
    data, variable = flagged(tmp_path / "a.nc", flags, global_attrs={"Mentor_QC_Field_Information": text})
    with data, pytest.warns(UserWarning, match="1 flags"):
        result = netcdf_reader.rejected(data, variable, np.array(flags))
    assert result.tolist() == [False, True, False, True, True, True]


@pytest.mark.parametrize("prefix,where", [("bit", "attrs"), ("qc_bit", "global_attrs")])
def test_bits_reject_only_bad(tmp_path, prefix, where):
    meaning = {f"{prefix}_1_assessment": "Bad", f"{prefix}_2_assessment": "Indeterminate"}
    data, variable = flagged(tmp_path / "a.nc", [0, 1, 2, 3], **{where: meaning})
    with data:
        result = netcdf_reader.rejected(data, variable, np.array([0, 1, 2, 3]))
    assert result.tolist() == [False, True, False, True]


def test_unexplained_flag_is_rejected_with_a_warning(tmp_path):
    data, variable = flagged(tmp_path / "a.nc", [0, 1])
    with data, pytest.warns(UserWarning, match="no stated meaning"):
        assert netcdf_reader.rejected(data, variable, np.array([0, 1])).tolist() == [False, True]


def test_precipitation_is_summed_over_the_interval():
    index = pd.date_range("2020-01-01 00:01", periods=30, freq="1min", tz="UTC")
    table = pd.DataFrame({"precip": np.full(30, 0.1), "p_sfc": np.full(30, 9.0e4)}, index=index)
    grid = time_alignment.to_grid(table, 60)
    assert grid.precip.iloc[0] == pytest.approx(3.0) and grid.p_sfc.iloc[0] == pytest.approx(9.0e4)


@pytest.mark.parametrize("attrs,flags", [({"flag_1_assessment": "Acceptable"}, [1, 99]), ({"bit_1_assessment": "Bad"}, [1, 4])])
def test_partly_documented_flags_are_rejected_with_a_warning(tmp_path, attrs, flags):
    data, variable = flagged(tmp_path / "a.nc", flags, attrs)
    with data, pytest.warns(UserWarning, match="no stated meaning"):
        result = netcdf_reader.rejected(data, variable, np.array(flags))
    assert result[1]


def test_averaged_stream_keeps_a_trace_of_its_flags():
    index = pd.date_range("2020-01-01 00:01", periods=30, freq="1min", tz="UTC")
    table = pd.DataFrame({"T_2": np.ones(30), "qc_T_2": [2.0] * 15 + [0.0] * 15, "qc_fail_T_2": np.zeros(30)}, index=index)
    grid = time_alignment.to_grid(table, 60)
    assert grid.qc_T_2_fraction_nonzero.iloc[0] == pytest.approx(0.5) and grid.qc_fail_T_2_fraction.iloc[0] == 0


def test_flags_follow_the_supplying_stream():
    index = pd.date_range("2020-01-01", periods=2, freq="30min", tz="UTC")
    tables = {"a": pd.DataFrame({"H": [1.0, np.nan], "qc_H": [7.0, 1.0]}, index=index),
              "b": pd.DataFrame({"H": [2.0, 2.0], "qc_H": [3.0, 3.0]}, index=index)}
    names = [("a", "H"), ("b", "H")]
    codes = dataset_builder.source_codes(tables, names, index)
    assert dataset_builder.source_flags(tables, names, index, codes).tolist() == [7.0, 3.0]


def test_build_names_sources_per_height_and_records_wind_source(tmp_path):
    for stream, h in (("sgp30co2flx4mC1.b1", 100.0), ("sgp30co2flx60mC1.b1", 200.0)):
        folder = tmp_path / "raw" / "armlive" / stream
        folder.mkdir(parents=True)
        with netCDF4.Dataset(folder / "a.nc", "w") as data:
            data.createDimension("time", 1)
            data.createVariable("yyyydddhhmmss", "f8", ("time",))[:] = 2010001000000
            data["yyyydddhhmmss"].long_name = "start of integration interval"
            for name, value in (("h", h), ("mean_rot_u", 3.0), ("wdir", 90.0), ("qc_wdir", 4.0)):
                data.createVariable(name, "f8", ("time",))[:] = value
            data.MENTOR_QC_FIELD_INFORMATION = "text"
    ds, _ = dataset_builder.build(tmp_path)
    assert ds.H_source.attrs["sources_4m"].split()[0] == "sgp30co2flx4mC1.b1"
    assert ds.H_source.attrs["sources_60m"].split()[0] == "sgp30co2flx60mC1.b1"
    assert ds.wind_source.sel(height=4).values.tolist() == [-1], "direction rejected, so no wind record"
    assert ds.H.sel(height=4).values.tolist() == [100.0], "single record survives"


def test_build_without_data_says_so(tmp_path):
    with pytest.raises(ValueError, match="no readable ARM files"):
        dataset_builder.build(tmp_path)


def test_wind_flag_follows_wind_source(tmp_path):
    folder = tmp_path / "raw" / "armlive" / "sgp30co2flx4mC1.b1"
    folder.mkdir(parents=True)
    with netCDF4.Dataset(folder / "a.nc", "w") as data:
        data.createDimension("time", 1)
        data.createVariable("yyyydddhhmmss", "f8", ("time",))[:] = 2010001000000
        data["yyyydddhhmmss"].long_name = "start of integration interval"
        for name, value in (("mean_rot_u", 3.0), ("wdir", 90.0), ("qc_wdir", 32.0)):
            data.createVariable(name, "f8", ("time",))[:] = value
        data.MENTOR_QC_FIELD_INFORMATION = "text"
    ds, _ = dataset_builder.build(tmp_path)
    assert ds.wind_source.sel(height=4).values.tolist() == [0]
    assert ds.wind_direction_qc.sel(height=4).values.tolist() == [32.0]


def test_test_samples_only_stand_in_for_missing_archive(tmp_path):
    stream = "sgp30qcecorE14.c1"
    sample = tmp_path / "raw" / "arm-test-data" / f"{stream}.20140616.nc"
    sample.parent.mkdir(parents=True)
    write_nc(sample, [1.0], [0.0], bounds=[[0.0, 1800.0]])
    assert netcdf_reader.read_stream(tmp_path, stream).H.tolist() == [1.0]
    archive = tmp_path / "raw" / "armlive" / stream
    archive.mkdir(parents=True)
    write_nc(archive / "a.nc", [2.0], [3600.0], bounds=[[0.0, 1800.0]])
    assert netcdf_reader.read_stream(tmp_path, stream).H.tolist() == [2.0]


def test_vector_base_time_in_trimmed_samples(tmp_path):
    path = tmp_path / "a.nc"
    with netCDF4.Dataset(path, "w") as data:
        data.createDimension("time", 2)
        data.createVariable("base_time", "f8", ("time",))[:] = [100.0, 100.0]
        data.createVariable("time_offset", "f8", ("time",))[:] = [0.0, 60.0]
        with_vector = netcdf_reader._epoch_seconds(data)
    assert with_vector[0].tolist() == [100.0, 160.0]


def test_vector_base_time_with_bounds(tmp_path):
    path = tmp_path / "a.nc"
    write_nc(path, [1.0, 2.0], [0.0, 1800.0], bounds=[[0.0, 1800.0], [1800.0, 3600.0]], name="h",
             extra={"base_time": [100.0, 100.0], "time_offset": [0.0, 1800.0]})
    table = netcdf_reader.read_file(path, "sgp30ecorE14.b1")
    assert [t.timestamp() for t in table.index] == [1900.0, 3700.0]


def test_soil_reads_lower_case_names(tmp_path):
    path = tmp_path / "a.nc"
    with netCDF4.Dataset(path, "w") as data:
        data.createDimension("time", 1)
        data.createDimension("depth", 2)
        data.createVariable("time", "f8", ("time",))[:] = 0.0
        data["time"].units = "seconds since 2003-01-01 00:00:00"
        data.createVariable("depth", "f8", ("depth",))[:] = [5.0, 15.0]
        data.createVariable("tsoil_w", "f8", ("time", "depth"))[:] = [[10.0, 12.0]]
    table = netcdf_reader.read_file(path, "sgpswatsE13.b1")
    assert table["Tsoil_W_5"].tolist() == [10.0] and table["Tsoil_W_15"].tolist() == [12.0]


def test_first_valid_finds_the_first_value_in_two_dimensions():
    time = pd.date_range("2020-01-01", periods=3, freq="30min")
    values = np.array([[np.nan, np.nan], [np.nan, 1.0], [2.0, 3.0]])
    variable = xr.DataArray(values, dims=("time", "height"), coords={"time": time})
    assert report.first_valid(variable).startswith("2020-01-01T00:30")
    assert report.first_valid(variable * np.nan) == ""
