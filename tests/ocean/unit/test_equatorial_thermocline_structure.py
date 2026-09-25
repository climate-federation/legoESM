"""The cancellation discriminator must be able to see cancellation.

A probe that cannot distinguish a sharp thermocline from a smeared one would
pass silently and certify the wrong conclusion, so these build columns with
KNOWN answers: an analytic two-layer profile whose isotherm depth and peak
gradient are set by construction, and a twin put through an actual vertical
average -- the pair the probe exists to tell apart.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[3]
_P = _ROOT / "scripts/validate/ocean_fidelity/equatorial_thermocline_structure.py"
_SPEC = importlib.util.spec_from_file_location("eq_struct", _P)
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules["eq_struct"] = _MOD
_SPEC.loader.exec_module(_MOD)


def _tanh_column(z, t_top=28.0, t_bot=10.0, z_c=120.0, width=20.0):
    """Two-layer profile: the transition is centred at z_c with the given width."""
    return t_bot + 0.5 * (t_top - t_bot) * (1.0 - np.tanh((z - z_c) / width))


def test_sharpness_falls_when_the_thermocline_is_smeared():
    """The whole discriminator: a wider transition gives a smaller peak
    gradient.  If this ever passes with equal values the probe is blind and
    every conclusion drawn from it is void.

    An earlier version of this test also demanded the two columns share a
    surface temperature.  They do not -- widening a tanh centred at 120 m
    changes the value at 2 m as well -- so the assertion failed and the
    EXPECTATION was the thing that was wrong, not the probe.  Sharpness is
    tested here; the temperature-and-sharpness pair that is the actual
    cancellation signature is the next test.
    """
    z = np.linspace(2.0, 400.0, 60)
    sharp = _tanh_column(z, width=15.0)[None, :]
    smeared = _tanh_column(z, width=60.0)[None, :]
    g_sharp = _MOD._max_dtdz(sharp, z, 400.0)[0]
    g_smeared = _MOD._max_dtdz(smeared, z, 400.0)[0]
    assert g_sharp > 2.0 * g_smeared


def test_vertical_mixing_cools_the_surface_and_smears_the_thermocline():
    """The cancellation signature itself, built by actually mixing.

    Over-mixing is a vertical average: it conserves the column's heat while
    cooling the surface and flattening the gradient.  Both halves must show up
    or the probe cannot tell a well-mixed ocean from a well-simulated one.

    The averaging window has to REACH the thermocline from the surface, or
    nothing happens -- a first version of this test mixed the top 50 m of a
    column whose thermocline sat at 120 m, found the surface unchanged, and
    was measuring its own kernel rather than the probe.  Here the thermocline
    is at 60 m and the window spans about 135 m.
    """
    z = np.linspace(2.0, 400.0, 120)
    col = _tanh_column(z, z_c=60.0, width=15.0)
    k = 41
    kernel = np.ones(k) / k
    mixed = np.convolve(np.pad(col, (k, k), mode="edge"), kernel,
                        mode="same")[k:-k]
    assert mixed[0] < col[0] - 1.0, "mixing must cool the surface"
    assert np.trapezoid(mixed, z) == pytest.approx(
        np.trapezoid(col, z), rel=0.01), "mixing must roughly conserve heat"
    assert (_MOD._max_dtdz(mixed[None, :], z, 400.0)[0]
            < 0.75 * _MOD._max_dtdz(col[None, :], z, 400.0)[0])


def test_sharpness_matches_the_analytic_peak_gradient():
    """A tanh's peak gradient is (t_top - t_bot) / (2 * width); the discrete
    estimate must land on it, or the number is not the quantity it is named."""
    z = np.linspace(1.0, 400.0, 400)
    width, t_top, t_bot = 25.0, 28.0, 10.0
    col = _tanh_column(z, t_top=t_top, t_bot=t_bot, width=width)[None, :]
    expected = (t_top - t_bot) / (2.0 * width)
    assert _MOD._max_dtdz(col, z, 400.0)[0] == pytest.approx(expected, rel=0.02)


def test_sharpness_respects_its_depth_window():
    """A gradient below the window must not be reported."""
    z = np.linspace(5.0, 800.0, 160)
    deep = _tanh_column(z, z_c=600.0, width=10.0)[None, :]
    shallow_window = _MOD._max_dtdz(deep, z, 200.0)[0]
    full_window = _MOD._max_dtdz(deep, z, 800.0)[0]
    assert shallow_window < 0.1 * full_window


def test_flatten_handles_both_grid_layouts(tmp_path):
    """Structured (ny, nx, nlev) and node-cloud (n, nlev) must reduce to the
    same point cloud -- the probe exists to compare one against the other."""
    nlev = 12
    zc = np.linspace(2.0, 400.0, nlev)
    ny, nx = 4, 5
    T3 = np.tile(_tanh_column(zc), (ny, nx, 1))
    lat3 = np.tile(np.linspace(-3.0, 3.0, ny)[:, None], (1, nx))
    lon3 = np.tile(np.linspace(150.0, 250.0, nx)[None, :], (ny, 1))
    mask3 = np.ones((ny, nx))

    s = tmp_path / "structured.npz"
    np.savez(s, T=T3, lat_T=lat3, lon_T=lon3, land_mask=mask3, z_center_ref=zc)
    n = tmp_path / "cloud.npz"
    np.savez(n, T=T3.reshape(-1, nlev), lat_T=lat3.ravel(),
             lon_T=lon3.ravel(), land_mask=mask3.ravel(), z_center_ref=zc)

    Ts, lats, lons, wets, zs, areas = _MOD._flatten(s)
    Tn, latn, lonn, wetn, zn, arean = _MOD._flatten(n)
    # Neither fixture carries cell_area, so both must fall back to cos(lat)
    # rather than silently weighting by something else.
    assert areas is None and arean is None
    assert Ts.shape == Tn.shape == (ny * nx, nlev)
    assert np.allclose(Ts, Tn) and np.allclose(lats, latn)
    assert np.allclose(lons, lonn) and np.array_equal(wets, wetn)


def test_land_columns_are_excluded_from_the_band_mean():
    """A land column carried into the mean is the silent-wrong-number failure
    mode; the mask must actually bite."""
    nlev = 10
    zc = np.linspace(2.0, 400.0, nlev)
    warm = _tanh_column(zc, t_top=28.0)
    cold = np.full(nlev, -99.0)                 # a land sentinel
    T = np.stack([warm, cold])
    lat = np.array([0.5, 0.5])
    lon = np.array([200.0, 200.0])
    wet = np.array([True, False])
    rows = _MOD._band_table(T, lat, lon, wet, zc, lambda c, z: 100.0,
                            2.0, [(190.0, 210.0)], 400.0)
    assert rows[0][5] == 1                       # one column counted
    assert rows[0][2] == pytest.approx(warm[0], rel=1e-9)


def _write_snapshot(path, sst, nlev=6):
    """A minimal structured snapshot whose surface temperature is prescribed."""
    ny, nx = 2, 4
    zc = np.linspace(5.0, 300.0, nlev)
    T = np.zeros((ny, nx, nlev))
    T[..., 0] = sst
    T[..., 1:] = 10.0
    np.savez(path,
             T=T,
             lat_T=np.zeros((ny, nx)),
             lon_T=np.tile(np.array([150.0, 170.0, 210.0, 230.0]), (ny, 1)),
             land_mask=np.ones((ny, nx)),
             z_center_ref=zc)
    return path


def test_the_tendency_table_reports_the_change_not_the_state(tmp_path, capsys):
    """A still-growing bias has to be judged on its RATE.

    The tripole's cold tongue warms while NEMO's settles, so a day-30 value is
    a rate caught mid-flight. This pins that --snapshot-early turns the tables
    into differences, and that the difference is the one the arm is judged on:
    a run whose day-30 SST matches NEMO exactly but arrived there by warming
    when NEMO cooled must NOT read as agreement.
    """
    import subprocess
    import sys as _sys

    early = _write_snapshot(tmp_path / "early.npz", 26.0)
    late = _write_snapshot(tmp_path / "late.npz", 27.5)
    # A NEMO stand-in is not available here, so drive the two model tables
    # directly through the same helper the CLI uses.
    z20 = _MOD._load_z20_helper()
    bins = [(200.0, 220.0), (220.0, 240.0)]
    Te, late_e = _MOD._flatten(early), _MOD._flatten(late)
    rows_e = _MOD._band_table(*Te[:5], z20, 2.0, bins, 400.0, Te[5])
    rows_l = _MOD._band_table(*late_e[:5], z20, 2.0, bins, 400.0, late_e[5])
    d = [l[2] - e[2] for l, e in zip(rows_l, rows_e)]
    assert np.allclose(d, [1.5, 1.5])

    # And the CLI must refuse a half-specified tendency rather than silently
    # comparing a late model state against an early oracle record.
    out = subprocess.run(
        [_sys.executable, str(_P), "--snapshot", str(late), "--label", "x",
         "--nemo-gridt", str(tmp_path / "missing.nc"),
         "--snapshot-early", str(early)],
        capture_output=True, text=True)
    assert out.returncode != 0
    assert "go together" in (out.stderr + out.stdout)


def test_an_equator_only_snapshot_is_not_mistaken_for_radians(tmp_path):
    """Latitude alone cannot tell degrees from radians near the equator.

    A band-limited snapshot has |lat| of a few degrees, which looks exactly
    like radians, and converting it moved a 230 E column to 218 E -- into the
    wrong longitude bin, silently. Longitude is what settles it: a degree
    field spans far more than 2*pi.
    """
    _write_snapshot(tmp_path / "band.npz", 26.0)
    _, lat, lon, _, _, _ = _MOD._flatten(tmp_path / "band.npz")
    assert np.allclose(sorted(set(np.round(lon, 6))), [150.0, 170.0, 210.0,
                                                       230.0])
    assert np.allclose(lat, 0.0)


def test_layer_means_separate_redistribution_from_net_loss():
    """The slab discriminator: vertical mixing moves heat between slabs and
    keeps their sum; uniform cooling drops every slab.  Built with a known
    answer (uniform slabs) so a sign or weighting error in the layer mean
    would be caught, not averaged away."""
    z = np.linspace(1.0, 400.0, 400)
    col = np.where(z < 50.0, 28.0, np.where(z < 150.0, 20.0, 12.0))
    T = col[None, :]
    got = [_MOD._layer_mean_T(T, z, z0, z1)[0] for z0, z1 in _MOD._LAYERS]
    assert got == pytest.approx([28.0, 20.0, 12.0], abs=0.15)
    # mixing the top two slabs: 0-50 cools, 50-150 warms, 150-300 untouched,
    # thickness-weighted sum conserved
    mixed = np.where(z < 150.0, (28.0 * 50 + 20.0 * 100) / 150.0, col)[None, :]
    d = [_MOD._layer_mean_T(mixed, z, z0, z1)[0] - g
         for (z0, z1), g in zip(_MOD._LAYERS, got)]
    assert d[0] < -4.0 and d[1] > 2.0 and abs(d[2]) < 0.15
    assert d[0] * 50 + d[1] * 100 == pytest.approx(0.0, abs=30 * 0.15)
    # uniform cooling drops every slab by the same amount
    cooled = (col - 1.0)[None, :]
    dc = [_MOD._layer_mean_T(cooled, z, z0, z1)[0] - g
         for (z0, z1), g in zip(_MOD._LAYERS, got)]
    assert dc == pytest.approx([-1.0, -1.0, -1.0], abs=1e-9)
    # the band table carries the three slab means in columns 6..8
    lat = np.zeros(1); lon = np.array([230.0]); wet = np.array([True])
    rows = _MOD._band_table(T, lat, lon, wet, z, lambda c, zz: 100.0, 2.0,
                            [(220.0, 240.0)], 400.0)
    assert rows[0][6:9] == pytest.approx(got, abs=1e-12)


def test_isopycnal_T_separates_heave_from_mixing():
    """Heave (shifting the whole T,S stack down) leaves T on density surfaces
    unchanged; vertically mixing the top of the column changes it.  Built
    with a known answer so a sign or interpolation error would be caught."""
    z = np.linspace(2.0, 400.0, 200)
    T = _tanh_column(z, z_c=80.0, width=25.0)
    S = 35.0 + 0.002 * z                      # weakly stable haline part
    base = _MOD._T_on_sigma(T[None, :], _MOD._sigma0(T[None, :], S[None, :]))[0]
    assert np.isfinite(base).sum() >= 4
    # heave: same (T,S) pairs, 30 m deeper
    Th = np.interp(z, z + 30.0, T); Sh = np.interp(z, z + 30.0, S)
    heaved = _MOD._T_on_sigma(Th[None, :], _MOD._sigma0(Th[None, :], Sh[None, :]))[0]
    ok = np.isfinite(base) & np.isfinite(heaved)
    assert ok.sum() >= 3
    assert np.max(np.abs(heaved[ok] - base[ok])) < 0.05
    # mixing the top 120 m changes T on the surfaces the mixed water crosses
    k = z <= 120.0
    Tm = T.copy(); Tm[k] = np.trapezoid(T[k], z[k]) / (z[k][-1] - z[k][0])
    Sm = S.copy(); Sm[k] = np.trapezoid(S[k], z[k]) / (z[k][-1] - z[k][0])
    mixed = _MOD._T_on_sigma(Tm[None, :], _MOD._sigma0(Tm[None, :], Sm[None, :]))[0]
    ok2 = np.isfinite(base) & np.isfinite(mixed)
    assert ok2.sum() >= 2
    assert np.max(np.abs(mixed[ok2] - base[ok2])) > 0.3


def test_isopycnal_depth_reports_heave_as_displacement():
    """Shifting the stack 30 m down must read as +30 m on every surface."""
    z = np.linspace(2.0, 400.0, 200)
    T = _tanh_column(z, z_c=80.0, width=25.0)
    S = 35.0 + 0.002 * z
    sig = _MOD._sigma0(T[None, :], S[None, :])
    z0 = _MOD._T_on_sigma(T[None, :], sig, zc=z)[0]
    Th = np.interp(z, z + 30.0, T); Sh = np.interp(z, z + 30.0, S)
    z1 = _MOD._T_on_sigma(Th[None, :], _MOD._sigma0(Th[None, :], Sh[None, :]), zc=z)[0]
    ok = np.isfinite(z0) & np.isfinite(z1) & (z0 > 40.0) & (z0 < 300.0)
    assert ok.sum() >= 3
    assert np.allclose(z1[ok] - z0[ok], 30.0, atol=2.0)


def test_surface_profile_sees_a_lens_the_layer_mean_hides():
    # One well-mixed column and one with a warm 3 K lens in the top 10 m over
    # colder water: the surface profile must separate them by T(0.5)-T(20).
    zc = np.array([0.5, 1.6, 2.7, 3.9, 5.1, 6.5, 8.1, 9.8, 11.8, 14.0, 16.5,
                   19.4, 22.8, 26.7, 31.1, 36.1, 41.8, 48.2, 55.4])
    mixed = np.full(zc.size, 23.0)
    lens = np.where(zc < 10.0, 26.0, 22.0)
    T = np.stack([mixed, lens])
    lat = np.array([0.0, 0.0]); lon = np.array([225.0, 235.0])
    wet = np.array([True, True])
    p_mixed = _MOD._surface_profile(T[:1], lat[:1], lon[:1], wet[:1], zc, None,
                                   2.0, 220.0, 240.0)
    p_lens = _MOD._surface_profile(T[1:], lat[1:], lon[1:], wet[1:], zc, None,
                                  2.0, 220.0, 240.0)
    assert p_mixed[0] - p_mixed[5] == pytest.approx(0.0)
    assert p_lens[0] - p_lens[5] > 2.5
    # a box with no column returns NaN rather than a number
    assert np.isnan(_MOD._surface_profile(T, lat, lon, wet, zc, None,
                                         2.0, 100.0, 120.0)).all()
