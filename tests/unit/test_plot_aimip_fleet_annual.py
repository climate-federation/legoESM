"""Direct test for the AIMIP fleet-annual plotter's anomaly helper.

Pins the paper Fig-3 convention: each series is rendered as an anomaly from its
own mean over the 1979-2014 baseline window, with the inter-member envelope
preserved.  The network/plotting paths (public AIMIP S3 bucket) are NOT
exercised here -- only the pure ``_subtract_baseline`` transform.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")
pytest.importorskip("fsspec")  # module imports fsspec at top

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts" / "plot" / "plot_aimip_fleet_annual.py"
)


def _load_mod():
    spec = importlib.util.spec_from_file_location("_fleet_annual_mod", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_subtract_baseline_removes_window_mean():
    mod = _load_mod()
    years = np.arange(1979, 2025)
    vals = (years - 1979).astype(float)  # linear ramp -> analytic baseline mean
    da = xr.DataArray(vals, dims="year", coords={"year": years})
    anom = mod._subtract_baseline(da, 1979, 2014)
    base = float(da.sel(year=slice(1979, 2014)).mean())
    assert base == pytest.approx(17.5)  # mean(0..35)
    assert float(anom.sel(year=1979)) == pytest.approx(-17.5)
    assert float(anom.sel(year=2024)) == pytest.approx(45.0 - 17.5)
    # the baseline window averages to ~0 by construction
    assert float(anom.sel(year=slice(1979, 2014)).mean()) == pytest.approx(
        0.0, abs=1e-9
    )


def test_subtract_baseline_preserves_member_spread():
    mod = _load_mod()
    years = np.arange(1979, 1981)
    members = np.array([[10.0, 12.0], [20.0, 24.0]])  # (member, year)
    da = xr.DataArray(
        members, dims=("member", "year"),
        coords={"member": [1, 2], "year": years},
    )
    anom = mod._subtract_baseline(da, 1979, 2014)
    assert float(da.mean()) == pytest.approx(16.5)  # full window -> grand mean
    spread_before = float((da.max("member") - da.min("member")).sel(year=1979))
    spread_after = float((anom.max("member") - anom.min("member")).sel(year=1979))
    assert spread_after == pytest.approx(spread_before)  # scalar shift keeps spread


def test_subtract_baseline_empty_window_falls_back_to_full_mean():
    mod = _load_mod()
    years = np.arange(2015, 2025)  # no overlap with a 1979-2014 baseline
    da = xr.DataArray(
        (years - 2015).astype(float), dims="year", coords={"year": years},
    )
    anom = mod._subtract_baseline(da, 1979, 2014)
    assert float(anom.mean()) == pytest.approx(0.0, abs=1e-9)  # full-series mean removed


def test_legoesm_variant_from_stem():
    """Overlay CSVs are labelled by the variant parsed from the filename;
    a non-conforming stem falls back to itself (still plotted)."""
    mod = _load_mod()
    assert mod._legoesm_variant_from_stem(
        "legoesm_classical_amip_annual") == "classical"
    assert mod._legoesm_variant_from_stem(
        "legoesm_column_nn_amip_r0_annual") == "column_nn"
    assert mod._legoesm_variant_from_stem(
        "legoesm_sfno_physics_amip_annual") == "sfno_physics"
    assert mod._legoesm_variant_from_stem("mystery_run") == "mystery_run"
    # every known variant has a color
    for v in ("classical", "column_nn", "sfno_physics"):
        assert v in mod.LEGOESM_VARIANT_COLORS
