"""Tests for the residual-variance SSO construction (#1712).

The point of the new mode is a property the classic construction lacks: a
mountain the MODEL RESOLVES must contribute ~nothing to the launch-stress
field, while sub-cutoff roughness contributes in full.  Synthetic waves with
known wavelengths make that a pass/fail rather than a plausibility argument.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_GEN = _ROOT / "scripts" / "data" / "prep_subgrid_orography.py"


def _load():
    sys.path.insert(0, str(_GEN.parent))
    spec = importlib.util.spec_from_file_location("_sso_gen", _GEN)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_sso_gen"] = mod
    spec.loader.exec_module(mod)
    return mod


def _terrain_ds(z):
    n_lat, n_lon = z.shape
    lat = np.linspace(-90 + 90.0 / n_lat, 90 - 90.0 / n_lat, n_lat)
    lon = np.linspace(0.25 / 2, 360 - 0.25 / 2, n_lon)
    return xr.Dataset({"elevation": (("lat", "lon"), z)},
                      coords={"lat": lat, "lon": lon})


def _grid(fine=0.25):
    n_lat, n_lon = int(180 / fine), int(360 / fine)
    lat = np.linspace(-90 + 90.0 / n_lat, 90 - 90.0 / n_lat, n_lat)
    lon = np.linspace(fine / 2, 360 - fine / 2, n_lon)
    return np.meshgrid(lat, lon, indexing="ij"), (n_lat, n_lon)


def test_constant_terrain_gives_zero():
    mod = _load()
    (_, _), (n_lat, n_lon) = _grid()
    out = mod.subgrid_orography_residual_stddev(
        _terrain_ds(np.full((n_lat, n_lon), 500.0)),
        fine_res_deg=0.25, block_deg=2.0, resolved_cutoff_deg=4.0)
    assert float(np.abs(out["SSO_STDH"].values).max()) < 1e-6


def test_resolved_long_wave_is_removed_but_classic_reports_it():
    """THE discriminating property (#1712's double-count defect).

    A 30-degree zonal wave is resolved by any model this field feeds, so the
    residual construction must return ~nothing for it — while the classic
    per-block stddev happily reports it as launch-stress orography.  If both
    constructions agree on this terrain, the new mode adds nothing and the
    double-count is still being fed.
    """
    mod = _load()
    (LAT, LON), (n_lat, n_lon) = _grid()
    z = 800.0 + 500.0 * np.sin(2 * np.pi * LON / 30.0)     # >= 0 everywhere
    ds = _terrain_ds(z)
    resid = mod.subgrid_orography_residual_stddev(
        ds, fine_res_deg=0.25, block_deg=2.0, resolved_cutoff_deg=4.0)
    classic = mod.subgrid_orography_stddev(
        ds, fine_res_deg=0.25, block_deg=2.0)
    band = slice(30, 690)   # away from the truncated polar smooth
    r = float(np.mean(resid["SSO_STDH"].values[band]))
    c = float(np.mean(classic["SSO_STDH"].values[band]))
    assert c > 30.0, f"classic construction does not even see the wave ({c})"
    assert r < 0.15 * c, (
        f"residual mode keeps {r:.1f} m of a resolved 30-deg wave the classic "
        f"mode reports as {c:.1f} m — the scale decomposition is not working")


def test_subgrid_short_wave_survives_in_full():
    """Roughness below the cutoff must come through ~undiminished."""
    mod = _load()
    (LAT, LON), (n_lat, n_lon) = _grid()
    z = 800.0 + 500.0 * np.sin(2 * np.pi * LON / 1.0)      # 1-deg wave
    ds = _terrain_ds(z)
    resid = mod.subgrid_orography_residual_stddev(
        ds, fine_res_deg=0.25, block_deg=2.0, resolved_cutoff_deg=4.0)
    classic = mod.subgrid_orography_stddev(
        ds, fine_res_deg=0.25, block_deg=2.0)
    band = slice(30, 690)
    r = float(np.mean(resid["SSO_STDH"].values[band]))
    c = float(np.mean(classic["SSO_STDH"].values[band]))
    assert c > 200.0
    assert r > 0.9 * c, (r, c)


def test_lon_seam_is_genuinely_periodic():
    """A wave crossing lon=0 must smooth identically to one at mid-domain.

    Codex P2: the global-mean tests dilute a seam error.  Compare the
    residual field of a short wave against the same wave phase-shifted by
    half the domain -- lon-periodicity means the residual sgh field must be
    identical up to the same shift.
    """
    mod = _load()
    (LAT, LON), (n_lat, n_lon) = _grid()
    z1 = 800.0 + 500.0 * np.sin(2 * np.pi * LON / 1.0)
    z2 = 800.0 + 500.0 * np.sin(2 * np.pi * (LON + 180.0) / 1.0)
    kw = dict(fine_res_deg=0.25, block_deg=2.0, resolved_cutoff_deg=4.0)
    r1 = mod.subgrid_orography_residual_stddev(_terrain_ds(z1), **kw)
    r2 = mod.subgrid_orography_residual_stddev(_terrain_ds(z2), **kw)
    a = np.asarray(r1["SSO_STDH"].values)
    b = np.roll(np.asarray(r2["SSO_STDH"].values), 90, axis=1)  # 180deg/2deg
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9)


def test_polar_truncated_window_stays_a_weighted_mean():
    """Poles: uniform terrain must smooth to itself even where the lat
    window truncates -- residual exactly 0 INCLUDING the polar rows (the
    renormalisation, not padding-with-zeros, is what this pins)."""
    mod = _load()
    (_, _), (n_lat, n_lon) = _grid()
    out = mod.subgrid_orography_residual_stddev(
        _terrain_ds(np.full((n_lat, n_lon), 750.0)),
        fine_res_deg=0.25, block_deg=2.0, resolved_cutoff_deg=4.0)
    v = np.asarray(out["SSO_STDH"].values)
    assert float(np.abs(v[0, :]).max()) < 1e-9    # southernmost block row
    assert float(np.abs(v[-1, :]).max()) < 1e-9   # northernmost


def test_cutoff_window_is_exactly_the_requested_width():
    """Codex P1 regression: the smooth must use EXACTLY n = cutoff/fine cells.

    Geometry: a 4-deg plateau CENTRED mid-domain (away from the seam — the
    first version of this test parked it on the seam and asserted symmetry of
    the wrong thing).  The pinned property is mirror symmetry of the residual
    sgh about the plateau centre: the corrected window covers [i-8, i+8)
    (16 cells, half-cell off-centre — a pure registration shift), while the
    old n+1-cell window at even n was symmetric but 4.25 deg wide; symmetry
    alone does not separate them, so ALSO pin the total residual energy
    against the value the 16-cell window gives, which a 17-cell window moves
    by more than the tolerance.
    """
    mod = _load()
    (LAT, LON), (n_lat, n_lon) = _grid()
    z = np.zeros((n_lat, n_lon))
    c = n_lon // 2
    z[:, c - 8: c + 8] = 1000.0            # 16 cells = 4 deg, centred
    out = mod.subgrid_orography_residual_stddev(
        _terrain_ds(z), fine_res_deg=0.25, block_deg=2.0,
        resolved_cutoff_deg=4.0)
    v = np.asarray(out["SSO_STDH"].values)
    mid = v[v.shape[0] // 2, :]
    cb = mid.shape[0] // 2                  # block index of the centre
    lo = mid[:cb]
    hi = mid[cb:]
    # Mirror symmetry about the plateau centre (registration shift folded
    # by comparing block sums, which are shift-invariant at this width).
    np.testing.assert_allclose(lo.sum(), hi.sum(), rtol=5e-2)
    # Width pin, measured 2026-09-05 with the exact-16-cell window (my first
    # pin was an unmeasured guess at half this value and the test rightly
    # failed -- the constant below is the measured one).
    total = float(mid.sum())
    assert abs(total - 572.822) / 572.822 < 0.01, total
    # Discrimination is MEASURED, not assumed: an explicit 4.25-deg cutoff
    # (17 cells -- what the pre-fix even-n window silently used for a
    # "4 deg" request) must move the total by more than the pin tolerance,
    # or this regression could not catch the off-by-one returning.
    out425 = mod.subgrid_orography_residual_stddev(
        _terrain_ds(z), fine_res_deg=0.25, block_deg=2.0,
        resolved_cutoff_deg=4.25)
    t425 = float(np.asarray(out425["SSO_STDH"].values)[v.shape[0] // 2, :].sum())
    assert abs(t425 - total) / total > 0.01, (
        f"a 17-cell window ({t425:.2f}) is indistinguishable from the "
        f"16-cell one ({total:.2f}) at the pin tolerance -- this width "
        "regression is vacuous and needs a sharper observable")


def test_cutoff_below_block_is_refused():
    mod = _load()
    (_, _), (n_lat, n_lon) = _grid(1.0)
    with pytest.raises(ValueError, match="cutoff"):
        mod.subgrid_orography_residual_stddev(
            _terrain_ds(np.zeros((n_lat, n_lon))),
            fine_res_deg=1.0, block_deg=4.0, resolved_cutoff_deg=2.0)
