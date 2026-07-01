"""Tests for ``scripts/data/prepare_omip_forcing.py``.

The CLI is a thin wrapper around ``build_jra55_cache``; these tests
focus on:

- argparse handling (required args, validation, default values)
- target-grid edge construction (including invalid resolutions)
- end-to-end main(argv) invocation against a synthetic source store
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
zarr = pytest.importorskip("zarr")


# Load the script as a module via its file path (it lives outside
# the importable package layout).
_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "data" / "prepare_omip_forcing.py"
_spec = importlib.util.spec_from_file_location("prepare_omip_forcing", _SCRIPT)
prepare_omip_forcing = importlib.util.module_from_spec(_spec)
sys.modules["prepare_omip_forcing"] = prepare_omip_forcing
_spec.loader.exec_module(prepare_omip_forcing)


# ============================================================================
# Argparse: required and default values
# ============================================================================

def test_parse_args_minimal_required(tmp_path):
    args = prepare_omip_forcing.parse_args([
        "--source", str(tmp_path / "src.zarr"),
        "--years", "1958", "2018",
        "--cache-dir", str(tmp_path / "cache"),
    ])
    assert args.source == str(tmp_path / "src.zarr")
    assert args.years == [1958, 2018]
    assert args.cache_dir == str(tmp_path / "cache")
    assert args.target_resolution_deg == 1.0
    assert args.ref_year == 1958
    assert args.overwrite is False
    assert args.quiet is False
    assert args.cache_filename == "jra55_do_v14_omip2_1deg_noleap.zarr"


def test_parse_args_missing_source_fails():
    with pytest.raises(SystemExit):
        prepare_omip_forcing.parse_args([
            "--years", "1958", "2018",
            "--cache-dir", "/tmp/cache",
        ])


def test_parse_args_missing_years_fails():
    with pytest.raises(SystemExit):
        prepare_omip_forcing.parse_args([
            "--source", "/tmp/src",
            "--cache-dir", "/tmp/cache",
        ])


def test_parse_args_overwrite_flag():
    args = prepare_omip_forcing.parse_args([
        "--source", "/tmp/src",
        "--years", "2000", "2000",
        "--cache-dir", "/tmp/cache",
        "--overwrite",
    ])
    assert args.overwrite is True


# ============================================================================
# Target-grid edge construction
# ============================================================================

def test_build_target_edges_1deg():
    lat_e, lon_e = prepare_omip_forcing._build_target_edges(1.0)
    assert lat_e.size == 181  # 180 cells + 1
    assert lon_e.size == 361  # 360 cells + 1
    np.testing.assert_allclose(np.rad2deg(lat_e[0]), -90.0, atol=1e-12)
    np.testing.assert_allclose(np.rad2deg(lat_e[-1]), 90.0, atol=1e-12)
    np.testing.assert_allclose(np.rad2deg(lon_e[0]), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.rad2deg(lon_e[-1]), 360.0, atol=1e-12)


def test_build_target_edges_2deg():
    lat_e, lon_e = prepare_omip_forcing._build_target_edges(2.0)
    assert lat_e.size == 91
    assert lon_e.size == 181


def test_build_target_edges_rejects_non_dividing_resolution():
    # 7° does not divide 180 evenly.
    with pytest.raises(ValueError, match="must divide 180"):
        prepare_omip_forcing._build_target_edges(7.0)


def test_build_target_edges_rejects_non_positive():
    with pytest.raises(ValueError, match="must be > 0"):
        prepare_omip_forcing._build_target_edges(0.0)
    with pytest.raises(ValueError, match="must be > 0"):
        prepare_omip_forcing._build_target_edges(-1.0)


# ============================================================================
# Validation
# ============================================================================

def test_validate_rejects_inverted_year_window(tmp_path):
    src = tmp_path / "src.zarr"
    src.mkdir()  # placeholder so existence check passes
    args = prepare_omip_forcing.parse_args([
        "--source", str(src),
        "--years", "2018", "1958",
        "--cache-dir", str(tmp_path / "cache"),
    ])
    with pytest.raises(ValueError, match="end year must be"):
        prepare_omip_forcing._validate_args(args)


def test_validate_rejects_out_of_range_year(tmp_path):
    src = tmp_path / "src.zarr"
    src.mkdir()
    args = prepare_omip_forcing.parse_args([
        "--source", str(src),
        "--years", "1800", "1900",
        "--cache-dir", str(tmp_path / "cache"),
    ])
    with pytest.raises(ValueError, match="plausible range"):
        prepare_omip_forcing._validate_args(args)


def test_validate_rejects_missing_local_source(tmp_path):
    args = prepare_omip_forcing.parse_args([
        "--source", str(tmp_path / "nonexistent.zarr"),
        "--years", "1958", "1958",
        "--cache-dir", str(tmp_path / "cache"),
    ])
    with pytest.raises(FileNotFoundError, match="does not exist"):
        prepare_omip_forcing._validate_args(args)


def test_validate_passes_remote_url_through(tmp_path):
    """Remote URLs (gs://, s3://, http) are passed to xarray for
    handling — the script should not pre-flight them."""
    args = prepare_omip_forcing.parse_args([
        "--source", "gs://noresm-jra55do/v1.4/corrected",
        "--years", "1958", "1958",
        "--cache-dir", str(tmp_path / "cache"),
    ])
    # Should not raise, even though gs:// path is not on disk.
    prepare_omip_forcing._validate_args(args)


# ============================================================================
# End-to-end: main(argv) on a synthetic source store
# ============================================================================

def _make_synthetic_jra55_zarr(out_path: Path, year_start: int, year_end: int):
    """Minimal synthetic source store covering one year at daily cadence."""
    n_lat, n_lon = 4, 8
    half_lat = 90.0 / n_lat
    half_lon = 180.0 / n_lon
    lat = np.linspace(90.0 - half_lat, -90.0 + half_lat, n_lat)
    lon = np.linspace(half_lon, 360.0 - half_lon, n_lon)
    # Daily cadence for speed.
    n_days = 365 * (year_end - year_start + 1)
    time = np.arange(n_days, dtype=np.float64)
    rng = np.random.default_rng(0)
    constants = {
        "uas": 5.0, "vas": -3.0, "tas": 290.0, "huss": 0.01,
        "psl": 1.013e5, "rsds": 250.0, "rlds": 350.0,
        "prra": 1e-5, "prsn": 0.0, "friver": 0.0,
    }
    data_vars = {
        var: (("time", "lat", "lon"),
              np.full((n_days, n_lat, n_lon), val) + rng.uniform(-1e-6, 1e-6))
        for var, val in constants.items()
    }
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={"time": time, "lat": lat, "lon": lon},
    )
    ds["time"].attrs["units"] = f"days since {year_start}-01-01 00:00:00"
    ds["time"].attrs["calendar"] = "gregorian"
    ds.to_zarr(str(out_path), mode="w", consolidated=True)


def test_main_end_to_end_creates_cache(tmp_path, capsys):
    # The cache builder writes via ``.chunk(...).to_zarr``, requiring the
    # optional dask chunk manager. Skip (don't error) when dask is absent.
    pytest.importorskip("dask")
    src = tmp_path / "synthetic.zarr"
    _make_synthetic_jra55_zarr(src, year_start=1958, year_end=1958)

    cache_dir = tmp_path / "cache"
    rc = prepare_omip_forcing.main([
        "--source", str(src),
        "--years", "1958", "1958",
        "--target-resolution-deg", "30.0",  # coarse for speed
        "--cache-dir", str(cache_dir),
        "--quiet",
    ])
    assert rc == 0

    # Cache file exists with default name
    cache = cache_dir / "jra55_do_v14_omip2_1deg_noleap.zarr"
    assert cache.exists()

    # Header text was printed
    captured = capsys.readouterr()
    assert "JRA55-do cache builder" in captured.out
    assert "status         : OK" in captured.out


def test_main_skips_when_cache_exists_without_overwrite(tmp_path, capsys):
    pytest.importorskip("dask")  # cache write requires the dask chunk manager
    src = tmp_path / "synthetic.zarr"
    _make_synthetic_jra55_zarr(src, year_start=1958, year_end=1958)

    cache_dir = tmp_path / "cache"
    # First build (quiet to keep output focused):
    rc1 = prepare_omip_forcing.main([
        "--source", str(src),
        "--years", "1958", "1958",
        "--target-resolution-deg", "30.0",
        "--cache-dir", str(cache_dir),
        "--quiet",
    ])
    assert rc1 == 0
    capsys.readouterr()  # Discard first-run output.

    # Second build without --overwrite: should skip and announce it.
    # (Don't use --quiet — the "skipping build" line is gated on progress=True.)
    rc2 = prepare_omip_forcing.main([
        "--source", str(src),
        "--years", "1958", "1958",
        "--target-resolution-deg", "30.0",
        "--cache-dir", str(cache_dir),
    ])
    assert rc2 == 0
    captured = capsys.readouterr()
    assert "skipping build" in captured.out
