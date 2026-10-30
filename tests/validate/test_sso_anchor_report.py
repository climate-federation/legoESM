"""Direct tests for scripts/validate/sso_anchor_report.py (#1712)."""
from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

_SCRIPT = (pathlib.Path(__file__).resolve().parents[2]
           / "scripts" / "validate" / "sso_anchor_report.py")
_spec = importlib.util.spec_from_file_location("sso_anchor_report", _SCRIPT)
rep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rep)


def _write(path, value, res=2.0, **attrs):
    lat = np.arange(-90 + res / 2, 90, res)
    lon = np.arange(res / 2, 360, res)
    sgh = np.full((lat.size, lon.size), float(value))
    ds = xr.Dataset({"SSO_STDH": (("lat", "lon"), sgh)},
                    coords={"lat": lat, "lon": lon}, attrs=attrs)
    ds.to_netcdf(path)
    return path


@pytest.fixture(scope="module")
def mesh():
    from legoesm.grids.voronoi import create_voronoi_mesh
    return create_voronoi_mesh(4)                       # 2562 cells


def _planted(path, res=1.0):
    """sgh = 10 + 5 sin(lat) + 3 cos(lon): known area-weighted moments."""
    lat = np.arange(-90 + res / 2, 90, res)
    lon = np.arange(res / 2, 360, res)
    LAT, LON = np.meshgrid(np.deg2rad(lat), np.deg2rad(lon), indexing="ij")
    sgh = 10 + 5 * np.sin(LAT) + 3 * np.cos(LON)
    xr.Dataset({"SSO_STDH": (("lat", "lon"), sgh)},
               coords={"lat": lat, "lon": lon}).to_netcdf(path)
    s1, s2 = np.sin(np.deg2rad(-60.0)), np.sin(np.deg2rad(-40.0))
    so_mean = 10 + 5 * (s2 ** 2 - s1 ** 2) / 2 / (s2 - s1)
    # <sin^2> over the band: (sin^3 / 3) / sin, cos-weighted
    so_sin2 = (s2 ** 3 - s1 ** 3) / 3 / (s2 - s1)
    so_ms = 100 + 25 * so_sin2 + 9 / 2 + 100 * (so_mean - 10) / 5
    return {"mean": 10.0, "ms": 100 + 25 / 3 + 9 / 2,
            "so_mean": so_mean, "so_ms": so_ms}


def test_constant_fields_give_exact_ratios_in_both_spaces(tmp_path, mesh):
    ref = _write(tmp_path / "ref.nc", 10.0, block_deg=2.0, fine_res_deg=1.0)
    new = _write(tmp_path / "new.nc", 20.0, res=1.0, block_deg=1.0,
                 fine_res_deg=1 / 60, resolved_cutoff_deg=3.5)
    rows = rep.report(ref, [new], mesh=mesh)
    assert rows[1]["latlon"]["ratio_ms"] == pytest.approx(4.0)
    assert rows[1]["latlon"]["ratio_so_ms"] == pytest.approx(4.0)
    assert rows[1]["mesh"]["ratio_ms"] == pytest.approx(4.0, rel=1e-6)
    assert rows[1]["latlon"]["mean"] == pytest.approx(20.0)


def test_area_weighting_is_applied(tmp_path):
    lat = np.arange(-89, 90, 2.0)
    lon = np.arange(1, 360, 2.0)
    sgh = np.where(np.abs(lat)[:, None] > 60, 100.0, 0.0) * np.ones(lon.size)
    xr.Dataset({"SSO_STDH": (("lat", "lon"), sgh)},
               coords={"lat": lat, "lon": lon}).to_netcdf(tmp_path / "p.nc")
    m = rep.latlon_moments(tmp_path / "p.nc")
    # polar caps beyond 60 deg cover 1 - sin(60) = 13.4% of the sphere
    assert m["mean"] == pytest.approx(100 * (1 - np.sin(np.deg2rad(60))),
                                      rel=0.02)


def test_planted_field_moments_in_both_spaces(tmp_path, mesh):
    """Catches a wrong Southern-Ocean mask, lat/lon swap or area weighting."""
    want = _planted(tmp_path / "p.nc")
    ll = rep.latlon_moments(tmp_path / "p.nc")
    for key in want:
        assert ll[key] == pytest.approx(want[key], rel=2e-3), key
    ms = rep.mesh_moments(tmp_path / "p.nc", mesh)
    for key in want:
        assert ms[key] == pytest.approx(want[key], rel=2e-2), key


def test_guard_verdicts_follow_the_loader(tmp_path):
    ncells = 40962                                     # level 6: cell 1.0035 deg
    ok = _write(tmp_path / "ok.nc", 5.0, res=1.0, block_deg=1.0,
                fine_res_deg=1 / 60, resolved_cutoff_deg=3.5)
    bad = _write(tmp_path / "bad.nc", 5.0, res=1.0, block_deg=1.0,
                 fine_res_deg=0.25, resolved_cutoff_deg=4.0)
    assert rep.guard_verdict(ok, ncells) == "PASS"
    assert rep.guard_verdict(bad, ncells) == "MISMATCH"


def test_non_finite_values_are_fatal(tmp_path):
    p = _write(tmp_path / "nan.nc", np.nan)
    with pytest.raises(ValueError, match="non-finite"):
        rep.latlon_moments(p)


def test_cli_writes_json(tmp_path):
    ref = _write(tmp_path / "ref.nc", 10.0)
    new = _write(tmp_path / "new.nc", 30.0)
    out = tmp_path / "r.json"
    assert rep.main(["--reference", str(ref), "--files", str(new),
                     "--json", str(out)]) == 0
    rows = json.loads(out.read_text())
    assert rows[1]["latlon"]["ratio_ms"] == pytest.approx(9.0)
