"""Smoke test for the committed EC-site evaluation runner
(scripts/run/run_ec_site_evaluation.py).

Reuses the synthetic DifferBESS-style v2 driver from ``test_ec_site_run`` (so the
forcing construction is single-sourced), runs ``run_evaluation`` end to end
(read -> lax.scan multilayer land -> obs comparison -> NetCDF), and asserts the
runner writes exactly the variables the publication plotter reads, with finite,
physical fluxes on the valid steps.

The site "SYN-Test" is not in ``EC_SITE_PHYSICS``; ``ec_site_physics`` falls back
to its defaults, which is the path we assert here (no real-site coupling).
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import xarray as xr
import pytest

_REPO = pathlib.Path(__file__).resolve().parents[3]
_RUNNER_PY = _REPO / "scripts" / "run" / "run_ec_site_evaluation.py"
_DRIVER_TEST_PY = pathlib.Path(__file__).with_name("test_ec_site_run.py")


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Single-source the synthetic driver: reuse the sibling integration test's builder
# rather than re-deriving the forcing (CLAUDE.md: no duplicate numerics in tests).
_make_driver = _load("test_ec_site_run", _DRIVER_TEST_PY)._make_driver

# Every variable the publication plotter (plot_ec_site_combined.py) reads.
_PLOTTER_VARS = {
    "gpp_mod", "le_mod", "le_canopy", "le_soil", "h_mod", "theta_prof",
    "gpp_obs", "le_obs", "le_obs_corr", "h_obs", "h_obs_corr", "swc_obs",
    "lai", "valid",
}


def test_run_ec_site_evaluation_smoke(tmp_path):
    driver_dir = tmp_path
    _make_driver(str(driver_dir / "SYN-Test_driver_v2.nc"))
    run = _load("run_ec_site_evaluation", _RUNNER_PY)

    out_nc = str(tmp_path / "SYN-Test.nc")
    p = run.run_evaluation("SYN-Test", out_nc, str(driver_dir))
    assert pathlib.Path(p).exists()

    ds = xr.open_dataset(p)
    # the runner writes exactly what the plotter needs, plus site attrs
    assert _PLOTTER_VARS <= set(ds.data_vars)
    assert ds.attrs["site"] == "SYN-Test"
    for key in ("dt_s", "theta_sat", "theta_r", "theta_fc", "theta_wp"):
        assert np.isfinite(ds.attrs[key])

    v = ds.valid.values.astype(bool)
    assert v.any()
    gpp, le, h = ds.gpp_mod.values, ds.le_mod.values, ds.h_mod.values
    # modelled fluxes finite + physical on the valid steps
    for a in (gpp, le, h):
        assert np.all(np.isfinite(a[v]))

    # the NaN guard records reverted steps; clean synthetic forcing reverts none,
    # so the soil profile is fully finite (a reverted step masks its profile).
    assert "reverted" in ds.data_vars
    reverted = ds.reverted.values.astype(bool)
    assert not reverted.any()
    assert np.all(np.isfinite(ds.theta_prof.values))
    assert np.nanmax(gpp[v]) > 0.0                 # daytime photosynthesis > 0
    assert np.nanmax(le[v]) < 1000.0               # no latent-heat runaway
    # the canopy/soil latent-heat partition sums to the total (both finite)
    assert np.all(np.isfinite(ds.le_canopy.values[v]))
    assert np.all(np.isfinite(ds.le_soil.values[v]))
    part = ds.le_canopy.values[v] + ds.le_soil.values[v]
    assert np.nanmax(np.abs(part - le[v])) < 1.0   # W/m2 rounding slack


def test_run_ec_site_evaluation_rejects_empty_window(tmp_path):
    """A year window matching no step is a hard error, never a silent empty run."""
    _make_driver(str(tmp_path / "SYN-Test_driver_v2.nc"))
    run = _load("run_ec_site_evaluation", _RUNNER_PY)
    with pytest.raises(ValueError, match="no steps"):
        run.run_evaluation("SYN-Test", str(tmp_path / "o.nc"), str(tmp_path),
                           year_lo=1990, year_hi=1991)


def test_find_driver_missing_is_error(tmp_path):
    run = _load("run_ec_site_evaluation", _RUNNER_PY)
    with pytest.raises(FileNotFoundError, match="driver_v2"):
        run._find_driver(str(tmp_path), "ZZ-Nowhere")


def test_corrected_flux_range_mask_rejects_sentinels(tmp_path):
    """Fill sentinels / out-of-range values in the FLUXNET closure-corrected
    fields must never leak into the le_obs_corr / h_obs_corr band -- they are
    masked to NaN before the ratio gate (locks the range-mask fix, non-vacuous)."""
    p = str(tmp_path / "SYN-Test_driver_v2.nc")
    _make_driver(p, n=96)
    with xr.open_dataset(p) as _d:            # detach so we can rewrite the path
        ds = _d.load().copy(deep=True)
    n = ds.sizes["time"]
    et = np.full(n, 2.0)                       # finite ET so the ratio gate is clean
    etc = np.full(n, 2.2)
    hc = np.full(n, 50.0)
    etc[10] = -9999.0                          # fill sentinel  -> must mask
    etc[20] = 500.0                            # 500 mm/day > 100 -> must mask
    hc[10] = -9999.0                           # sentinel -> must mask
    ds["ET"] = ("time", et); ds["ET_CORR"] = ("time", etc); ds["H_CORR"] = ("time", hc)
    ds.to_netcdf(p)

    run = _load("run_ec_site_evaluation", _RUNNER_PY)
    out = run.run_evaluation("SYN-Test", str(tmp_path / "o.nc"), str(tmp_path))
    r = xr.open_dataset(out)
    lec, hcorr = r.le_obs_corr.values, r.h_obs_corr.values
    assert np.isnan(lec[10]) and np.isnan(lec[20]) and np.isnan(hcorr[10])
    # no sentinel leaked: a -9999 mm/day would be ~ -2.9e8 W/m2; bands stay physical
    assert np.nanmax(np.abs(lec)) < 2000.0
    assert np.nanmax(np.abs(hcorr)) < 2000.0


def test_non_monotonic_time_axis_raises(tmp_path):
    """A non-strictly-increasing driver time axis is a hard error, not a silent
    slice across a jump (the prognostic scan needs a monotonic axis)."""
    p = str(tmp_path / "SYN-Test_driver_v2.nc")
    _make_driver(p, n=48)
    with xr.open_dataset(p) as _d:            # detach so we can rewrite the path
        ds = _d.load().copy(deep=True)
    tt = ds.time.values.copy()
    tt[10], tt[11] = tt[11], tt[10]            # swap two stamps -> non-monotonic
    ds = ds.assign_coords(time=tt)
    ds.to_netcdf(p)

    run = _load("run_ec_site_evaluation", _RUNNER_PY)
    with pytest.raises(ValueError, match="strictly increasing"):
        run.run_evaluation("SYN-Test", str(tmp_path / "o.nc"), str(tmp_path))
