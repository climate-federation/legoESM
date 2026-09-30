"""Direct tests for ``scripts/data/build_phc3_ic_for_omip.py``.

The script does two things that change numbers, and both are silent if wrong:
it converts in-situ to potential temperature, and it re-levels the column onto
``WOA_DEPTHS`` (historically required because ``init_ocean_from_woa`` took its
source depth axis from that constant rather than the file; the loader now
reads the file's own axis, and the re-levelling is kept so the conversion
happens at the target depths). A test that only checked shapes would pass
with either conversion skipped, so these check the physics instead:
the conversion must COOL, must grow with depth, must leave the surface
untouched, and the re-levelling must land the file's own values at the right
depths.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.ocean.init_woa import WOA_DEPTHS  # noqa: E402

_SCRIPT = (Path(__file__).resolve().parents[2]
           / "scripts" / "data" / "build_phc3_ic_for_omip.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("build_phc3", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


build_phc3 = _load_module()

_DEPTHS = np.array([0.0, 100.0, 1000.0, 3000.0, 5000.0])


def _write_phc3_like(tmp_path, T_insitu_C=4.0, S=34.7, land_col=True):
    """A PHC3-shaped file: (depth, lat, lon), vars ``temp``/``salt``."""
    xr = pytest.importorskip("xarray")
    lat = np.array([-45.5, 0.5, 45.5])
    lon = np.array([0.5, 120.5, 240.5])
    shape = (_DEPTHS.size, lat.size, lon.size)
    T = np.full(shape, float(T_insitu_C))
    Sf = np.full(shape, float(S))
    if land_col:
        T[:, 0, 0] = np.nan          # a land column
        Sf[:, 0, 0] = np.nan
    ds = xr.Dataset(
        {"temp": (("depth", "lat", "lon"), T),
         "salt": (("depth", "lat", "lon"), Sf)},
        coords={"depth": _DEPTHS, "lat": lat, "lon": lon},
    )
    path = tmp_path / "phc3_like.nc"
    ds.to_netcdf(path)
    return path


def test_output_shape_and_depth_axis(tmp_path):
    t_an, s_an, lat, lon, depths = build_phc3.build_phc3_ic(
        _write_phc3_like(tmp_path))
    assert t_an.shape == (1, WOA_DEPTHS.size, lat.size, lon.size)
    assert s_an.shape == t_an.shape
    np.testing.assert_allclose(depths, WOA_DEPTHS)


def test_conversion_cools_and_grows_with_depth(tmp_path):
    """theta < T_in-situ below the surface, monotonically more so with depth,
    and EXACTLY equal at the surface (p_ref = 0 there)."""
    t_an, _, _, _, depths = build_phc3.build_phc3_ic(
        _write_phc3_like(tmp_path, T_insitu_C=4.0))
    # Only down to the deepest OBSERVED level: below it the column is NaN by
    # contract now, because holding the deepest observed value there is what
    # wrote warm shelf water into the abyss.
    k5000 = int(np.argmin(np.abs(depths - 5000.0)))
    col = t_an[0, :k5000 + 1, 1, 1]            # an ocean column
    assert col[0] == pytest.approx(4.0, abs=1e-12)
    drop = 4.0 - col
    assert np.all(np.diff(drop) >= -1e-12), "cooling must not reverse"
    # At 5000 m the UNESCO correction is a few tenths of a degree.
    assert 0.1 < drop[-1] < 1.0
    assert np.all(np.isnan(t_an[0, k5000 + 1:, 1, 1]))


def test_salinity_passes_through_unchanged(tmp_path):
    """Only temperature is converted; salinity must survive the re-levelling
    of a constant column exactly."""
    _, s_an, _, _, depths = build_phc3.build_phc3_ic(
        _write_phc3_like(tmp_path, S=34.7))
    k5000 = int(np.argmin(np.abs(depths - 5000.0)))
    np.testing.assert_allclose(
        s_an[0, :k5000 + 1, 1, 1], 34.7, rtol=0, atol=1e-12)
    # Below the deepest observed level there is nothing to pass through.
    assert np.all(np.isnan(s_an[0, k5000 + 1:, 1, 1]))


def test_land_column_stays_nan(tmp_path):
    """A land column must come out NaN, not filled -- the downstream WOA path
    treats NaN as no-data and would otherwise get fabricated water."""
    t_an, s_an, _, _, depths = build_phc3.build_phc3_ic(
        _write_phc3_like(tmp_path, land_col=True))
    assert np.all(np.isnan(t_an[0, :, 0, 0]))
    assert np.all(np.isnan(s_an[0, :, 0, 0]))
    k5000 = int(np.argmin(np.abs(depths - 5000.0)))
    assert np.all(np.isfinite(t_an[0, :k5000 + 1, 1, 1]))


def _write_ragged(tmp_path):
    """One column per bottom-depth case, so the per-column pressure cap is
    exercised on all of them at once.

    Columns (lat index, lon index):
      (0,0) all NaN               -> land
      (0,1) valid only at k=0     -> deepest valid source level 0 m
      (1,0) interior NaN gap      -> deepest valid is still the last level
      (1,1) fully valid           -> deepest valid = 5000 m
    """
    xr = pytest.importorskip("xarray")
    lat = np.array([-10.5, 10.5])
    lon = np.array([0.5, 180.5])
    T = np.full((_DEPTHS.size, 2, 2), 4.0)
    S = np.full((_DEPTHS.size, 2, 2), 34.7)
    T[:, 0, 0] = np.nan
    S[:, 0, 0] = np.nan
    T[1:, 0, 1] = np.nan
    S[1:, 0, 1] = np.nan
    T[2, 1, 0] = np.nan            # interior gap at 1000 m
    S[2, 1, 0] = np.nan
    ds = xr.Dataset(
        {"temp": (("depth", "lat", "lon"), T),
         "salt": (("depth", "lat", "lon"), S)},
        coords={"depth": _DEPTHS, "lat": lat, "lon": lon},
    )
    p = tmp_path / "ragged.nc"
    ds.to_netcdf(p)
    return p


def test_pressure_cap_handles_every_bottom_depth_case(tmp_path):
    t_an, _, _, _, depths = build_phc3.build_phc3_ic(_write_ragged(tmp_path))
    k5000 = int(np.argmin(np.abs(depths - 5000.0)))

    # land column: NaN all the way down, never a fabricated value.
    assert np.all(np.isnan(t_an[0, :, 0, 0]))

    # surface-only column: fewer than two valid source entries, so
    # interp_column_to_depths returns all-NaN by contract.
    assert np.all(np.isnan(t_an[0, :, 0, 1]))

    # interior-gap column: the gap is bridged, the column is finite, and the
    # cap uses the DEEPEST valid level (5000 m), not the level above the gap.
    gap_col = t_an[0, :, 1, 0]
    full_col = t_an[0, :, 1, 1]
    assert np.all(np.isfinite(gap_col[:k5000 + 1]))
    assert np.all(np.isnan(gap_col[k5000 + 1:]))
    # Compare only where both are observed: below 5000 m both are NaN, and
    # assert_allclose treats NaN == NaN as a match, so comparing the whole
    # column would pass with the columns agreeing about nothing at all.
    np.testing.assert_allclose(gap_col[:k5000 + 1], full_col[:k5000 + 1],
                               rtol=0, atol=1e-12)

    # The cap must not touch levels shallower than the deepest valid source
    # depth -- there p_eff == the target depth.
    assert full_col[0] == pytest.approx(4.0, abs=1e-12)
    assert 0.1 < 4.0 - full_col[k5000] < 1.0


def test_nothing_is_written_below_the_deepest_valid_source_level(tmp_path):
    """Below the source's data there must be NO value at all.  This used to
    hold the deepest observed value instead, which wrote a shelf column's warm
    bottom water to 5500 m; the pressure cap existed only to stop that held
    value from also being adiabatically corrected as if it sat in the abyss."""
    t_an, s_an, _, _, depths = build_phc3.build_phc3_ic(
        _write_phc3_like(tmp_path, T_insitu_C=4.0, land_col=False))
    below = depths > _DEPTHS[-1]          # deeper than the source's 5000 m
    assert below.any()
    assert np.all(np.isnan(t_an[0, below, 1, 1]))
    assert np.all(np.isnan(s_an[0, below, 1, 1]))
    # ... and the deepest OBSERVED level is still there, so this is a cut and
    # not a wholesale blanking of the column.
    assert np.isfinite(t_an[0, ~below, 1, 1]).all()


def test_missing_variable_raises_with_the_available_names(tmp_path):
    with pytest.raises(ValueError, match="no variable 'nope'"):
        build_phc3.build_phc3_ic(_write_phc3_like(tmp_path), t_var="nope")


def test_non_monotonic_depth_axis_raises(tmp_path):
    xr = pytest.importorskip("xarray")
    bad = np.array([0.0, 1000.0, 500.0])
    ds = xr.Dataset(
        {"temp": (("depth", "lat", "lon"), np.zeros((3, 2, 2))),
         "salt": (("depth", "lat", "lon"), np.full((3, 2, 2), 35.0))},
        coords={"depth": bad, "lat": [0.5, 1.5], "lon": [0.5, 1.5]},
    )
    p = tmp_path / "bad.nc"
    ds.to_netcdf(p)
    with pytest.raises(ValueError, match="not strictly ascending"):
        build_phc3.build_phc3_ic(p)
