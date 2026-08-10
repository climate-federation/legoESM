"""Unit tests for the cross-grid baroclinic growth discriminator
(``scripts/validate/bci_growth_rate.py``, #1028/#1081).

The probe decides whether the cd-grid cube's dead jet is a GROWTH deficit or an
EQUILIBRATION deficit, so its reductions are load-bearing.  Every test here is
written to FAIL if the guard it covers is deleted — the codex review of the
first draft correctly called two of them vacuous (they passed on numpy's own
exceptions), so the file-contract tests now assert the message that only the
preflight produces.  The script is loaded from its file (scripts/ is not an
importable package).
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np
import pytest

_SCRIPT = (pathlib.Path(__file__).resolve().parents[2]
           / "scripts" / "validate" / "bci_growth_rate.py")


def _load():
    spec = importlib.util.spec_from_file_location("_bci", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bci = _load()

LAT = np.linspace(-90.0, 90.0, 181)
LON = np.linspace(-180.0, 180.0, 360, endpoint=False)
TIMES = np.arange(0.0, 11.0, 1.0)


def _growing_wave(sigma_ke: float, zonal_mean: float = 0.0):
    """Wavenumber-6 eddy with KE ~ exp(sigma_ke t), on a zonal background.

    The background is non-zero on purpose: with a zero zonal mean, deleting the
    zonal-mean subtraction would leave every number unchanged and the tests
    below would prove nothing.
    """
    amp = np.exp(0.5 * sigma_ke * TIMES)[:, None, None]
    wave = np.sin(np.deg2rad(6.0 * LON))[None, None, :] * np.ones(
        (1, LAT.size, 1))
    return amp * wave + zonal_mean, 0.5 * amp * wave + zonal_mean


def _write_arm(d: pathlib.Path, u, v, times=TIMES, lat=LAT, lon=LON,
               t_3d=None, psp_rate=0.25):
    d.mkdir(parents=True, exist_ok=True)
    payload = {"u": u, "v": v, "lat": lat, "lon": lon, "times_days": times}
    if t_3d is not None:
        payload["T_3d"] = t_3d
    np.savez(d / "snapshots_latlon.npz", **payload)
    # main() also reads the matrix's native-grid timeseries CSV.
    t = np.arange(0.0, 10.01, 0.25)
    lines = ["step,time_days,mass,max_wind,ps_perturbation"]
    lines += [f"{int(x * 100)},{x:.8f},5.1e19,28.0,{np.exp(psp_rate * x):.6e}"
              for x in t]
    (d / "mean_timeseries.csv").write_text("\n".join(lines) + "\n")
    return d


# --------------------------------------------------------------- the metric

def test_recovers_a_known_growth_rate():
    u, v = _growing_wave(0.63, zonal_mean=30.0)
    eke = bci.eddy_ke(u, v, LAT, (20.0, 80.0))
    rate, r2, n = bci.growth_rate(TIMES, eke, 4.0, 10.0)
    assert rate == pytest.approx(0.63, abs=1e-9)
    assert r2 > 1 - 1e-12
    assert n == 7


def test_zonal_mean_is_actually_removed():
    """Adding a pure zonal jet must not change the eddy statistic at all."""
    u, v = _growing_wave(0.5)
    plain = bci.eddy_ke(u, v, LAT, (20.0, 80.0))
    jet = np.broadcast_to(np.cos(np.deg2rad(LAT))[None, :, None],
                          u.shape) * 40.0
    with_jet = bci.eddy_ke(u + jet, v, LAT, (20.0, 80.0))
    assert np.allclose(plain, with_jet, rtol=1e-12)


def test_rate_is_independent_of_the_eddy_amplitude():
    """A growth RATE must not move when the perturbation is rescaled —
    otherwise a cross-grid comparison reads the IC, not the growth."""
    u, v = _growing_wave(0.4, zonal_mean=10.0)
    r_small, _, _ = bci.growth_rate(
        TIMES, bci.eddy_ke(u, v, LAT, (20.0, 80.0)), 4.0, 10.0)
    r_big, _, _ = bci.growth_rate(
        TIMES, bci.eddy_ke(1e3 * u, 1e3 * v, LAT, (20.0, 80.0)), 4.0, 10.0)
    assert r_small == pytest.approx(r_big, abs=1e-12)


def test_area_weighting_is_applied():
    """An eddy confined to high latitudes must score BELOW the same-amplitude
    eddy at low latitudes."""
    base = np.zeros((TIMES.size, LAT.size, LON.size))
    wave = np.sin(np.deg2rad(6.0 * LON))
    high, low = base.copy(), base.copy()
    # SAME number of rows in both bands (11 each): with equal row counts the
    # only thing that can separate them is the cos(lat) weight, so deleting the
    # weighting makes this test fail.
    hi_rows = (LAT >= 70.0) & (LAT <= 80.0)
    lo_rows = (LAT >= 25.0) & (LAT <= 35.0)
    assert hi_rows.sum() == lo_rows.sum() == 11
    high[:, hi_rows, :] = wave
    low[:, lo_rows, :] = wave
    e_high = bci.eddy_ke(high, 0 * high, LAT, (20.0, 80.0))[0]
    e_low = bci.eddy_ke(low, 0 * low, LAT, (20.0, 80.0))[0]
    assert e_high < e_low
    # ... and by the ratio the weights predict, not merely "less".
    assert e_high / e_low == pytest.approx(
        np.cos(np.deg2rad(LAT[hi_rows])).sum()
        / np.cos(np.deg2rad(LAT[lo_rows])).sum(), rel=1e-12)


def test_curvature_shows_up_as_disagreeing_sub_window_rates():
    """The reason the probe fits two windows: a saturating series must NOT
    look like one clean exponential."""
    eke = np.exp(0.6 * TIMES) / (1.0 + np.exp(0.6 * (TIMES - 7.0)))
    early, r2e, _ = bci.growth_rate(TIMES, eke, 2.0, 6.0)
    late, _, _ = bci.growth_rate(TIMES, eke, 6.0, 10.0)
    assert r2e > 0.99            # the early window alone still looks perfect
    assert late < 0.5 * early    # ... while the late window is far slower


# --------------------------------------------------------------- the guards

def test_nonfinite_input_is_fatal():
    u, v = _growing_wave(0.5)
    u[-1, 90, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        bci.eddy_ke(u, v, LAT, (20.0, 80.0))


def test_narrow_band_is_fatal():
    u, v = _growing_wave(0.5)
    with pytest.raises(ValueError, match="selects"):
        bci.eddy_ke(u, v, LAT, (89.0, 90.0))


def test_zonal_flow_has_no_eddy_energy_and_no_rate():
    zonal = 30.0 * np.ones((TIMES.size, LAT.size, LON.size))
    eke = bci.eddy_ke(zonal, 0.0 * zonal, LAT, (20.0, 80.0))
    assert np.allclose(eke, 0.0)
    with pytest.raises(ValueError, match="non-positive"):
        bci.growth_rate(TIMES, eke, 4.0, 10.0)


def test_truncated_run_cannot_be_reported_as_the_full_window():
    """A run that stopped at day 7 must not have days 4-7 printed as 4-10."""
    u, v = _growing_wave(0.5)
    eke = bci.eddy_ke(u, v, LAT, (20.0, 80.0))
    with pytest.raises(ValueError, match="truncated"):
        bci.growth_rate(TIMES[:8], eke[:8], 4.0, 10.0)


def test_flat_positive_series_is_fatal():
    flat = np.full(TIMES.size, 3.0)
    with pytest.raises(ValueError, match="not an\n?\\s*exponential|constant"):
        bci.growth_rate(TIMES, flat, 4.0, 10.0)


def test_floor_dominated_series_is_fatal():
    series = np.array([1e-12] * 4 + [1.0] * 7)
    with pytest.raises(ValueError, match="dynamic range"):
        bci.growth_rate(TIMES, series, 0.0, 10.0)


def test_too_few_samples_in_window_is_fatal():
    t = np.array([4.0, 6.0, 10.0])
    with pytest.raises(ValueError, match="not a growth rate"):
        bci.growth_rate(t, np.exp(0.5 * t), 4.0, 10.0)


def test_non_monotone_times_are_fatal():
    u, v = _growing_wave(0.5)
    eke = bci.eddy_ke(u, v, LAT, (20.0, 80.0))
    t = TIMES.copy()
    t[5], t[6] = t[6], t[5]
    with pytest.raises(ValueError, match="strictly increasing"):
        bci.growth_rate(t, eke, 4.0, 10.0)


def test_shape_guards():
    u, v = _growing_wave(0.5)
    with pytest.raises(ValueError, match="disagree"):
        bci.eddy_ke(u, v[:, :, :10], LAT, (20.0, 80.0))
    with pytest.raises(ValueError, match="lat axis"):
        bci.eddy_ke(u, v, LAT[:-1], (20.0, 80.0))


# ------------------------------------------------------- the file contract

def test_load_arm_names_the_missing_file(tmp_path):
    """Non-vacuous: numpy's own FileNotFoundError carries no such message."""
    with pytest.raises(FileNotFoundError, match="run the matrix case first"):
        bci.load_arm(tmp_path)


def test_load_arm_lists_every_missing_key_at_once(tmp_path):
    """Non-vacuous: indexing the npz would raise on the FIRST key only."""
    np.savez(tmp_path / "snapshots_latlon.npz", u=np.zeros((3, 4, 5)))
    with pytest.raises(KeyError) as exc:
        bci.load_arm(tmp_path)
    for key in ("v", "lat", "lon", "times_days"):
        assert key in str(exc.value)


def test_load_arm_rejects_a_canvas_that_disagrees_with_its_axes(tmp_path):
    u, v = _growing_wave(0.5)
    _write_arm(tmp_path, u, v, lon=LON[:-1])
    with pytest.raises(ValueError, match="does not match its own"):
        bci.load_arm(tmp_path)


def test_load_arm_rejects_a_permuted_lat_axis(tmp_path):
    u, v = _growing_wave(0.5)
    _write_arm(tmp_path, u, v, lat=LAT[::-1])
    with pytest.raises(ValueError, match="strictly increasing"):
        bci.load_arm(tmp_path)


def test_arms_with_different_times_cannot_be_compared(tmp_path):
    u, v = _growing_wave(0.5)
    a = bci.load_arm(_write_arm(tmp_path / "a", u, v))
    b = bci.load_arm(_write_arm(tmp_path / "b", u, v, times=TIMES + 0.5))
    with pytest.raises(ValueError, match="matched-time"):
        bci.require_matched_arms({"a": a, "b": b})


def test_arms_with_different_canvases_cannot_be_compared(tmp_path):
    u, v = _growing_wave(0.5)
    lat2 = np.linspace(-90.0, 90.0, 91)
    u2 = u[:, ::2, :]
    a = bci.load_arm(_write_arm(tmp_path / "a", u, v))
    b = bci.load_arm(_write_arm(tmp_path / "b", u2, u2, lat=lat2))
    with pytest.raises(ValueError, match="canvas"):
        bci.require_matched_arms({"a": a, "b": b})


def test_t_level_series_rejects_an_out_of_range_level(tmp_path):
    u, v = _growing_wave(0.5)
    t3 = np.ones((TIMES.size, LAT.size, LON.size, 8))
    arm = bci.load_arm(_write_arm(tmp_path, u, v, t_3d=t3))
    with pytest.raises(IndexError, match="outside 0..7"):
        bci.t_level_series(arm, 99, arm["lat"], (20.0, 80.0))


def test_t_level_series_measures_the_named_level(tmp_path):
    """Only the requested level's eddy structure may enter the statistic."""
    u, v = _growing_wave(0.5)
    t3 = np.zeros((TIMES.size, LAT.size, LON.size, 4))
    t3[..., 2] = np.sin(np.deg2rad(6.0 * LON))[None, None, :]
    arm = bci.load_arm(_write_arm(tmp_path, u, v, t_3d=t3))
    assert bci.t_level_series(arm, 2, arm["lat"], (20.0, 80.0))[0] > 0
    assert bci.t_level_series(arm, 1, arm["lat"], (20.0, 80.0))[0] == 0.0


# ------------------------------------------------------------------- CLI

def test_cli_reports_the_time_resolved_ratio_and_json(tmp_path, capsys):
    u, v = _growing_wave(0.5)
    _write_arm(tmp_path / "ref", u, v)
    # The second arm grows at HALF the rate, so its ratio to the reference must
    # fall monotonically — the signature the probe exists to expose.
    slow_u, slow_v = _growing_wave(0.25)
    _write_arm(tmp_path / "slow", slow_u, slow_v)
    rc = bci.main([f"ref={tmp_path / 'ref'}", f"slow={tmp_path / 'slow'}",
                   "--reference", "ref",
                   "--json-out", str(tmp_path / "out.json")])
    assert rc == 0
    assert "ratio to 'ref'" in capsys.readouterr().out
    data = json.loads((tmp_path / "out.json").read_text())
    ratio = data["arms"]["slow"]["eke_ratio_to_ref"]
    assert np.all(np.diff(ratio) < 0)
    assert data["arms"]["slow"]["eke_early"]["rate_per_day"] == pytest.approx(
        0.25, abs=1e-9)
    assert data["arms"]["ref"]["eke_early"]["rate_per_day"] == pytest.approx(
        0.5, abs=1e-9)


def test_native_ps_perturbation_reads_the_csv_by_column_name(tmp_path):
    """Non-vacuous: a reordered CSV must still be read correctly, and a CSV
    without the column must raise rather than silently use another one."""
    d = tmp_path / "a"
    d.mkdir()
    (d / "mean_timeseries.csv").write_text(
        "step,ps_perturbation,time_days,mass\n"
        + "\n".join(f"{i},{np.exp(0.3 * i):.6e},{float(i):.8f},5e19"
                    for i in range(11)) + "\n")
    t, v = bci.native_ps_perturbation(d)
    assert np.allclose(t, np.arange(11.0))
    rate, _, _ = bci.growth_rate(t, v, 2.0, 6.0)
    # 1e-5, not 1e-9: the fixture writes the series with 6 significant digits,
    # so the recoverable rate is limited by the file, not by the fit.
    assert rate == pytest.approx(0.3, abs=1e-5)

    (d / "mean_timeseries.csv").write_text(
        "step,time_days,mass\n" + "\n".join(f"{i},{float(i)},5e19"
                                            for i in range(11)) + "\n")
    with pytest.raises(KeyError, match="ps_perturbation"):
        bci.native_ps_perturbation(d)


def test_native_ps_perturbation_requires_the_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="matrix writes this"):
        bci.native_ps_perturbation(tmp_path)


def test_level_pressure_uses_the_models_own_sigma_ladder(tmp_path):
    """The reported pressure must track the model coordinate, not a guess:
    level 0 must be near the model top and the last level near the surface."""
    u, v = _growing_wave(0.5)
    t3 = np.ones((TIMES.size, LAT.size, LON.size, 40))
    d = _write_arm(tmp_path, u, v, t_3d=t3)
    np.savez(d / "snapshots_latlon.npz", u=u, v=v, lat=LAT, lon=LON,
             times_days=TIMES, T_3d=t3,
             p_s=np.full((TIMES.size, LAT.size, LON.size), 1.0e5))
    arm = bci.load_arm(d)
    p_top = bci.level_pressure_hpa(arm, 0, (20.0, 80.0))
    p_bot = bci.level_pressure_hpa(arm, 39, (20.0, 80.0))
    assert p_top is not None and p_top < 50.0
    assert 950.0 < p_bot < 1000.0


def test_cli_rejects_duplicate_labels(tmp_path):
    u, v = _growing_wave(0.5)
    _write_arm(tmp_path / "a", u, v)
    with pytest.raises(SystemExit):
        bci.main([f"cube={tmp_path / 'a'}", f"cube={tmp_path / 'a'}"])


def test_self_test_passes():
    bci.self_test()


def test_additive_floor_is_not_quotable(tmp_path, capsys):
    """codex round-2 #5a: EKE = floor + exp(sigma t) passes every range guard
    and fits a slope far below the true rate with a plausible R^2 ~ 0.9.  The
    quotability gate, not prose, has to catch it."""
    t = np.arange(0.0, 11.0, 1.0)
    series = 100.0 + np.exp(0.6 * (t - 2.0))
    rate, r2, _ = bci.growth_rate(t, series, 2.0, 6.0)
    assert rate < 0.1 and r2 < 0.95          # the trap, reproduced
    assert not (r2 >= bci._QUOTABLE_R2
                and rate > bci._QUOTABLE_RATE_PER_DAY)


def test_cli_marks_a_floor_dominated_arm_unquotable(tmp_path, capsys):
    amp = (100.0 + np.exp(0.6 * (TIMES - 2.0)))[:, None, None]
    wave = np.sin(np.deg2rad(6.0 * LON))[None, None, :] * np.ones(
        (1, LAT.size, 1))
    _write_arm(tmp_path / "floor", np.sqrt(amp) * wave, np.sqrt(amp) * wave)
    clean_u, clean_v = _growing_wave(0.6)
    _write_arm(tmp_path / "clean", clean_u, clean_v)
    assert bci.main([f"floor={tmp_path / 'floor'}", f"clean={tmp_path / 'clean'}",
                     "--json-out", str(tmp_path / "o.json")]) == 0
    data = json.loads((tmp_path / "o.json").read_text())
    assert data["arms"]["clean"]["eke_early"]["quotable"] is True
    assert data["arms"]["floor"]["eke_early"]["quotable"] is False
    assert "quotable" in capsys.readouterr().out


def test_cli_emits_the_late_window_and_the_second_field(tmp_path):
    u, v = _growing_wave(0.5)
    t3 = np.zeros((TIMES.size, LAT.size, LON.size, 40))
    t3[..., 27] = (np.exp(0.25 * TIMES)[:, None, None]
                   * np.sin(np.deg2rad(6.0 * LON))[None, None, :])
    _write_arm(tmp_path / "a", u, v, t_3d=t3)
    assert bci.main([f"a={tmp_path / 'a'}",
                     "--json-out", str(tmp_path / "o.json")]) == 0
    rec = json.loads((tmp_path / "o.json").read_text())["arms"]["a"]
    for key in ("eke_early", "eke_late", "tvar_early", "tvar_late",
                "psp_early", "psp_late", "eke_t_first", "t_first_days"):
        assert key in rec, key
    assert rec["tvar_early"]["rate_per_day"] == pytest.approx(0.5, abs=1e-6)


def test_hybrid_runs_do_not_get_a_pressure_label(tmp_path):
    """A fixed INDEX is a fixed pressure only on the sigma ladder."""
    u, v = _growing_wave(0.5)
    t3 = np.ones((TIMES.size, LAT.size, LON.size, 40))
    d = _write_arm(tmp_path, u, v, t_3d=t3)
    np.savez(d / "snapshots_latlon.npz", u=u, v=v, lat=LAT, lon=LON,
             times_days=TIMES, T_3d=t3,
             p_s=np.full((TIMES.size, LAT.size, LON.size), 1.0e5))
    arm = bci.load_arm(d)
    assert bci.level_pressure_hpa(arm, 27, (20.0, 80.0), "sigma") is not None
    assert bci.level_pressure_hpa(arm, 27, (20.0, 80.0), "hybrid") is None


def test_temperature_on_a_different_canvas_is_fatal(tmp_path):
    u, v = _growing_wave(0.5)
    t3 = np.ones((TIMES.size, LAT.size, LON.size // 2, 40))
    _write_arm(tmp_path, u, v, t_3d=t3)
    with pytest.raises(ValueError, match="same"):
        bci.load_arm(tmp_path)
