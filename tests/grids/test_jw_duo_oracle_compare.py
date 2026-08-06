"""Direct tests for ``scripts/validate/fv3_native/jw_duo_oracle_compare.py``.

The script is an INSTRUMENT: it decides whether we believe the 3-D cube
lane is close to the FV3 duo-grid oracle.  Per the ``oracle-fidelity``
rules a diagnostic is untrusted code until it passes its own controls, so
every assertion here is paired with a synthetic violation proving the
check can actually fail.

No oracle NetCDF is required -- the loaders are exercised against
synthetic npz files, and the oracle-reading path is covered only by a
skipped-if-absent smoke test.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

_SCRIPT = (pathlib.Path(__file__).resolve().parents[2]
           / "scripts" / "validate" / "fv3_native" / "jw_duo_oracle_compare.py")


def _load():
    spec = importlib.util.spec_from_file_location("jw_duo_oracle_compare",
                                                  _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


jw = _load()

LAT = np.linspace(-90.0, 90.0, 181)
LON = np.arange(0.5, 360.0, 1.0)


def _uniform_ps(value_pa=1.0e5):
    return np.full((LAT.size, LON.size), value_pa)


# ---------------------------------------------------------------- weights

def test_area_weights_sum_to_one_and_are_cos_lat():
    w = jw.area_weights(LAT, LON.size)
    assert w.shape == (LAT.size, LON.size)
    assert abs(w.sum() - 1.0) < 1e-12
    # equator must outweigh 60 deg by 1/cos(60) = 2
    eq = w[LAT.size // 2, 0]
    p60 = w[int(np.argmin(np.abs(LAT - 60.0))), 0]
    assert abs(eq / p60 - 2.0) < 1e-6


def test_unweighted_mean_would_differ_so_the_weighting_is_not_vacuous():
    """A pole-heavy anomaly must score differently weighted vs unweighted;
    otherwise the cos(lat) weighting is decoration."""
    ps = _uniform_ps()
    ps[:10, :] += 500.0                       # a polar cap only
    weighted = jw.ps_stats(ps, LAT)["rms_prime_hPa"]
    unweighted = float(np.std(ps - ps.mean())) / 100.0
    assert weighted < 0.5 * unweighted


# ------------------------------------------------------------- ps_stats

def test_uniform_field_has_zero_rms_prime():
    s = jw.ps_stats(_uniform_ps(), LAT)
    assert s["rms_prime_hPa"] == pytest.approx(0.0, abs=1e-12)
    assert s["mean_hPa"] == pytest.approx(1000.0, abs=1e-9)


def test_rms_prime_is_invariant_under_a_longitude_roll():
    """The whole point of the metric: a coordinate-convention mismatch
    between the oracle (0..360) and the matrix (-180..180) must not be
    able to change the number."""
    rng = np.random.default_rng(0)
    ps = _uniform_ps() + rng.normal(0.0, 300.0, (LAT.size, LON.size))
    a = jw.ps_stats(ps, LAT)["rms_prime_hPa"]
    b = jw.ps_stats(np.roll(ps, 180, axis=1), LAT)["rms_prime_hPa"]
    assert a == pytest.approx(b, rel=1e-12)


def test_rms_prime_responds_to_amplitude():
    """Non-vacuity: doubling the anomaly must double the score."""
    rng = np.random.default_rng(1)
    d = rng.normal(0.0, 300.0, (LAT.size, LON.size))
    a = jw.ps_stats(_uniform_ps() + d, LAT)["rms_prime_hPa"]
    b = jw.ps_stats(_uniform_ps() + 2.0 * d, LAT)["rms_prime_hPa"]
    assert b == pytest.approx(2.0 * a, rel=1e-10)


def test_rms_prime_ignores_a_uniform_offset():
    """It measures structure, not the mean -- so a model with a different
    reference surface pressure is not penalised for that alone."""
    rng = np.random.default_rng(2)
    d = rng.normal(0.0, 300.0, (LAT.size, LON.size))
    a = jw.ps_stats(_uniform_ps() + d, LAT)["rms_prime_hPa"]
    b = jw.ps_stats(_uniform_ps() + d + 5000.0, LAT)["rms_prime_hPa"]
    assert a == pytest.approx(b, rel=1e-12)


def test_nonfinite_ps_is_fatal():
    """nanmax/nanmean hiding a blown-up run is exactly the failure mode
    the fidelity rules call out."""
    ps = _uniform_ps()
    ps[5, 5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        jw.ps_stats(ps, LAT)


def test_wrong_rank_is_fatal():
    with pytest.raises(ValueError, match="2-D"):
        jw.ps_stats(_uniform_ps()[None], LAT)


# -------------------------------------------------------------- at_day

def test_at_day_picks_the_exact_frame():
    t = np.arange(0.0, 10.0, 1.0)
    assert jw.at_day(t, 7.0) == 7


def test_at_day_refuses_a_far_frame():
    """A silent nearest-frame snap is how a day-0/day-1 lens error becomes
    a physics claim.  It must raise, not snap."""
    t = np.array([0.0, 5.0, 10.0])
    with pytest.raises(ValueError, match="no frame within"):
        jw.at_day(t, 7.0)


def test_at_day_tolerance_is_honoured():
    t = np.array([0.0, 6.99, 10.0])
    assert jw.at_day(t, 7.0, tol=0.02) == 1
    with pytest.raises(ValueError):
        jw.at_day(t, 7.0, tol=0.001)


# -------------------------------------------------------------- load_ours

def _write_npz(tmp_path, **kw):
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / "snapshots_latlon.npz"
    np.savez_compressed(p, **kw)
    return str(p)


def _good_npz_kwargs(nt=3):
    return dict(times_days=np.arange(float(nt)),
                p_s=np.full((nt, LAT.size, LON.size), 1.0e5),
                lat=LAT, lon=LON - 180.0)


def test_load_ours_roundtrip_and_lon_is_mapped_into_0_360(tmp_path):
    t, ps, lat, lon = jw.load_ours(_write_npz(tmp_path, **_good_npz_kwargs()))
    assert ps.shape == (3, LAT.size, LON.size)
    assert lon.min() >= 0.0 and lon.max() < 360.0
    assert np.allclose(np.sort(lon), LON)
    assert np.allclose(lat, LAT)
    assert np.allclose(t, [0.0, 1.0, 2.0])


def test_load_ours_rejects_hpa_valued_ps(tmp_path):
    """Reporting a 100x-wrong rms because someone stored hPa is a silently
    wrong number, which is worse than a crash."""
    kw = _good_npz_kwargs()
    kw["p_s"] = kw["p_s"] / 100.0
    with pytest.raises(ValueError, match="not Pa"):
        jw.load_ours(_write_npz(tmp_path, **kw))


def test_load_ours_reports_missing_keys_by_name(tmp_path):
    kw = _good_npz_kwargs()
    del kw["p_s"]
    with pytest.raises(KeyError, match="p_s"):
        jw.load_ours(_write_npz(tmp_path, **kw))


def test_load_ours_rejects_a_2d_ps(tmp_path):
    kw = _good_npz_kwargs()
    kw["p_s"] = kw["p_s"][0]
    with pytest.raises(ValueError, match=r"nt, nlat, nlon"):
        jw.load_ours(_write_npz(tmp_path, **kw))


# ------------------------------------------------------- common_days

def test_common_days_intersects_and_reports(capsys):
    """A short arm must shrink the window for EVERYONE — scoring arms over
    different spans is the differing-window confound."""
    spans = {"a": (0.0, 10.0), "b": (0.0, 2.0)}
    keep = jw.common_days(spans, [1.0, 2.0, 3.0, 9.0])
    assert keep == [1.0, 2.0]
    out = capsys.readouterr().out
    assert "dropped day(s) 3, 9" in out
    assert "'b'" in out          # names the limiting arm


def test_common_days_keeps_everything_when_spans_agree(capsys):
    keep = jw.common_days({"a": (0.0, 10.0), "b": (0.0, 10.0)}, [1.0, 9.0])
    assert keep == [1.0, 9.0]
    assert "dropped" not in capsys.readouterr().out


def test_common_days_raises_when_the_window_is_empty():
    with pytest.raises(ValueError, match="no requested day"):
        jw.common_days({"a": (0.0, 1.0), "b": (5.0, 9.0)}, [3.0])


def test_build_curves_scores_every_arm_on_the_same_days(tmp_path, monkeypatch):
    """Non-vacuity for the intersection: the long arm must be truncated to
    the short arm's days, not scored on its own longer set."""
    def fake_oracle(res, arm):
        t = np.arange(0.0, 10.0)
        return t, np.full((t.size, LAT.size, LON.size), 1.0e5), LAT, LON

    monkeypatch.setattr(jw, "load_oracle", fake_oracle)
    short = _write_npz(tmp_path / "s", times_days=np.arange(0.0, 3.0),
                       p_s=np.full((3, LAT.size, LON.size), 1.0e5),
                       lat=LAT, lon=LON)
    curves, used, spans = jw.build_curves("C48", [("short", short)],
                                          [1.0, 2.0, 9.0])
    assert used == [1.0, 2.0]
    for per_day in curves.values():
        assert sorted(per_day) == [1.0, 2.0]
    assert spans[("ours", "short")] == (0.0, 2.0)
    assert spans[("oracle", "duo")] == (0.0, 9.0)


def test_truncated_arm_is_flagged_in_the_table(tmp_path, monkeypatch, capsys):
    """A crashed run yields a short arm.  Its row must be marked, else the
    number reads as a converged imprint (this exact confound appeared in
    the baroclinic latlon/hybrid arm, which had blown up at day 2)."""
    def fake_oracle(res, arm):
        t = np.arange(0.0, 10.0)
        return t, np.full((t.size, LAT.size, LON.size), 1.0e5), LAT, LON

    monkeypatch.setattr(jw, "load_oracle", fake_oracle)
    short = _write_npz(tmp_path / "s", times_days=np.arange(0.0, 3.0),
                       p_s=np.full((3, LAT.size, LON.size), 1.0e5),
                       lat=LAT, lon=LON)
    full = _write_npz(tmp_path / "f", times_days=np.arange(0.0, 11.0),
                      p_s=np.full((11, LAT.size, LON.size), 1.0e5),
                      lat=LAT, lon=LON)
    jw.main(["--ours", short, "--label", "crashed",
             "--ours", full, "--label", "healthy", "--days", "1,2"])
    out = capsys.readouterr().out
    # main() prints the rms' table first and a min/max block afterwards, and
    # both start with the arm label -- keep the FIRST occurrence so the
    # assertions read the table row, not the later block.
    rows: dict[str, str] = {}
    for ln in out.splitlines():
        if not ln[:1].isalpha() or ln.startswith(("AMPLITUDE", "DISTANCE",
                                                  "arm", "NOTE", "wrote")):
            continue
        parts = ln.split()
        key = " ".join(parts[:2]) if ln.startswith("oracle") else parts[0]
        rows.setdefault(key, ln)
    assert "TRUNCATED" in rows["crashed"]
    # An arm that RUNS PAST the reference window is fine, not truncated...
    assert "TRUNCATED" not in rows["healthy"]
    # ...and the oracle rows must never be flagged for being 9 d not 10 d,
    # which is a property of the released dataset, not of any run.
    assert "TRUNCATED" not in rows["oracle duo"]
    assert "TRUNCATED" not in rows["oracle plain"]


# ----------------------------------------------------------------- gate

def test_gate_is_not_vacuous(tmp_path, monkeypatch, capsys):
    """A quiet arm must PASS and a noisy arm must FAIL under the same
    threshold -- a gate that only ever passes proves nothing."""
    rng = np.random.default_rng(3)
    nt = 10
    t = np.arange(float(nt))

    def fake_oracle(res, arm):
        ps = np.full((nt, LAT.size, LON.size), 1.0e5)
        ps += rng.normal(0.0, 3.0, ps.shape)          # tiny: rms' ~0.03 hPa
        return t, ps, LAT, LON

    monkeypatch.setattr(jw, "load_oracle", fake_oracle)

    quiet = _write_npz(tmp_path / "q", **{
        **_good_npz_kwargs(nt),
        "p_s": np.full((nt, LAT.size, LON.size), 1.0e5)
        + rng.normal(0.0, 4.0, (nt, LAT.size, LON.size))})
    noisy = _write_npz(tmp_path / "n", **{
        **_good_npz_kwargs(nt),
        "p_s": np.full((nt, LAT.size, LON.size), 1.0e5)
        + rng.normal(0.0, 1500.0, (nt, LAT.size, LON.size))})

    assert jw.main(["--ours", quiet, "--label", "quiet",
                    "--days", "1,2", "--max-rmse-hpa", "1.0"]) == 0
    assert "JW_DUO_GATE: PASS" in capsys.readouterr().out

    assert jw.main(["--ours", noisy, "--label", "noisy",
                    "--days", "1,2", "--max-rmse-hpa", "1.0"]) == 1
    assert "JW_DUO_GATE: FAIL" in capsys.readouterr().out


def test_label_count_must_match_ours_count(tmp_path):
    p = _write_npz(tmp_path, **_good_npz_kwargs())
    with pytest.raises(SystemExit):
        jw.main(["--ours", p, "--label", "a", "--label", "b"])


# ------------------------------------------------------- oracle smoke

def test_oracle_reference_loads_if_present():
    """Skipped off-box.  On the box this is the only check that the real
    Zenodo files still parse and still carry Pa."""
    try:
        t, ps, lat, lon = jw.load_oracle("C48", "duo")
    except (FileNotFoundError, ImportError) as exc:
        pytest.skip(f"oracle reference unavailable: {exc}")
    assert ps.shape[1:] == (181, 360)
    assert t[-1] == pytest.approx(9.0, abs=1e-6)
    # The J&W wave must actually grow, else we are reading the wrong case.
    early = jw.ps_stats(ps[jw.at_day(t, 1.0)], lat)["rms_prime_hPa"]
    late = jw.ps_stats(ps[jw.at_day(t, 9.0)], lat)["rms_prime_hPa"]
    assert late > 20.0 * early


# ============================================================== round 2
# Every test below pins a defect the adversarial review found in round 1.

def _oracle_stub(ps_field=None, nt=10):
    t = np.arange(float(nt))

    def fake(res, arm):
        ps = (np.full((nt, LAT.size, LON.size), 1.0e5)
              if ps_field is None else ps_field)
        return t, ps, LAT, LON
    return fake


def test_amplitude_ratio_alone_cannot_prove_agreement(monkeypatch, tmp_path):
    """THE round-1 RED. rms' compares each field to ITSELF, so a field with
    the right amplitude and a totally wrong pattern scores 1.0x. The new
    rmse column must expose it."""
    rng = np.random.default_rng(7)
    truth = 1.0e5 + rng.normal(0.0, 200.0, (10, LAT.size, LON.size))
    monkeypatch.setattr(jw, "load_oracle", _oracle_stub(truth))
    # Same amplitude, independent pattern.
    impostor = 1.0e5 + rng.normal(0.0, 200.0, (10, LAT.size, LON.size))
    p = _write_npz(tmp_path / "imp", times_days=np.arange(10.0),
                   p_s=impostor, lat=LAT, lon=LON)
    curves, days, _ = jw.build_curves("C48", [("impostor", p)], [1.0])
    st = curves[("ours", "impostor")][1.0]
    ratio = st["rms_prime_hPa"] / curves[("oracle", "duo")][1.0]["rms_prime_hPa"]
    assert 0.9 < ratio < 1.1, "amplitude ratio should look innocent"
    # ...while the true distance is ~sqrt(2)x the field amplitude.
    assert st["rmse_hPa"] > 1.2 * st["rms_prime_hPa"]


def test_rmse_is_zero_for_an_identical_field(monkeypatch, tmp_path):
    rng = np.random.default_rng(8)
    truth = 1.0e5 + rng.normal(0.0, 200.0, (10, LAT.size, LON.size))
    monkeypatch.setattr(jw, "load_oracle", _oracle_stub(truth))
    p = _write_npz(tmp_path / "same", times_days=np.arange(10.0),
                   p_s=truth.copy(), lat=LAT, lon=LON)
    curves, _, _ = jw.build_curves("C48", [("same", p)], [1.0])
    assert curves[("ours", "same")][1.0]["rmse_hPa"] == pytest.approx(0.0,
                                                                     abs=1e-12)


def test_rmse_sees_a_uniform_offset_that_rms_prime_hides(monkeypatch,
                                                         tmp_path):
    """`ours = oracle + 5000 Pa` is a 50 hPa mass bias, not a convention."""
    truth = np.full((10, LAT.size, LON.size), 1.0e5)
    monkeypatch.setattr(jw, "load_oracle", _oracle_stub(truth))
    p = _write_npz(tmp_path / "off", times_days=np.arange(10.0),
                   p_s=truth + 5000.0, lat=LAT, lon=LON)
    curves, _, _ = jw.build_curves("C48", [("off", p)], [1.0])
    st = curves[("ours", "off")][1.0]
    assert st["rms_prime_hPa"] == pytest.approx(0.0, abs=1e-12)   # hidden
    assert st["rmse_hPa"] == pytest.approx(50.0, rel=1e-9)        # caught
    assert st["bias_hPa"] == pytest.approx(50.0, rel=1e-9)


def test_a_different_canvas_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(jw, "load_oracle", _oracle_stub())
    small_lat = np.linspace(-90.0, 90.0, 73)
    small_lon = np.arange(0.5, 360.0, 2.5)
    p = _write_npz(tmp_path / "wrong", times_days=np.arange(10.0),
                   p_s=np.full((10, 73, small_lon.size), 1.0e5),
                   lat=small_lat, lon=small_lon)
    with pytest.raises(ValueError, match="grid size mismatch"):
        jw.build_curves("C48", [("wrong", p)], [1.0])


def test_axis_lengths_must_match_the_data(tmp_path):
    p = _write_npz(tmp_path / "bad", times_days=np.arange(3.0),
                   p_s=np.full((5, LAT.size, LON.size), 1.0e5),
                   lat=LAT, lon=LON)
    with pytest.raises(ValueError, match="do not match"):
        jw.load_ours(p)


def test_a_label_cannot_impersonate_the_oracle(monkeypatch, tmp_path):
    """--label 'oracle duo' used to overwrite the reference, making the
    denominator user data. Origin is now structural."""
    monkeypatch.setattr(jw, "load_oracle", _oracle_stub())
    p = _write_npz(tmp_path / "spoof", times_days=np.arange(10.0),
                   p_s=np.full((10, LAT.size, LON.size), 1.0e5),
                   lat=LAT, lon=LON)
    curves, _, _ = jw.build_curves("C48", [("oracle duo", p)], [1.0])
    assert ("oracle", "duo") in curves and ("ours", "oracle duo") in curves
    assert curves[("oracle", "duo")][1.0]["rmse_hPa"] == pytest.approx(0.0,
                                                                       abs=1e-12)


def test_duplicate_labels_are_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(jw, "load_oracle", _oracle_stub())
    p = _write_npz(tmp_path / "dup", times_days=np.arange(10.0),
                   p_s=np.full((10, LAT.size, LON.size), 1.0e5),
                   lat=LAT, lon=LON)
    with pytest.raises(ValueError, match="duplicate --label"):
        jw.build_curves("C48", [("a", p), ("a", p)], [1.0])


def test_gate_fails_with_no_arms(monkeypatch, capsys):
    """An empty `bad` set used to exit PASS having compared nothing."""
    monkeypatch.setattr(jw, "load_oracle", _oracle_stub())
    assert jw.main(["--days", "1,2", "--max-rmse-hpa", "1.0"]) == 1
    out = capsys.readouterr().out
    assert "JW_DUO_GATE: FAIL" in out and "no --ours arm" in out


def test_gate_fails_on_a_truncated_arm(monkeypatch, tmp_path, capsys):
    """TRUNCATED used to be a printed comment the gate ignored."""
    monkeypatch.setattr(jw, "load_oracle", _oracle_stub())
    short = _write_npz(tmp_path / "sh", times_days=np.arange(3.0),
                       p_s=np.full((3, LAT.size, LON.size), 1.0e5),
                       lat=LAT, lon=LON)
    assert jw.main(["--ours", short, "--label", "crashed",
                    "--days", "1,2", "--max-rmse-hpa", "500"]) == 1
    assert "TRUNCATED" in capsys.readouterr().out


def test_gate_fails_when_day_1_is_not_the_gate_day(monkeypatch, tmp_path,
                                                   capsys):
    """--days 3,5 used to silently gate day 3 while advertising day 1."""
    monkeypatch.setattr(jw, "load_oracle", _oracle_stub())
    p = _write_npz(tmp_path / "ok", times_days=np.arange(10.0),
                   p_s=np.full((10, LAT.size, LON.size), 1.0e5),
                   lat=LAT, lon=LON)
    assert jw.main(["--ours", p, "--label", "a", "--days", "3,5",
                    "--max-rmse-hpa", "1.0"]) == 1
    assert "not day 1" in capsys.readouterr().out


def test_gate_still_passes_a_genuinely_good_arm(monkeypatch, tmp_path, capsys):
    """Non-vacuity in the other direction: the hardening must not make the
    gate unpassable."""
    rng = np.random.default_rng(9)
    truth = 1.0e5 + rng.normal(0.0, 100.0, (10, LAT.size, LON.size))
    monkeypatch.setattr(jw, "load_oracle", _oracle_stub(truth))
    p = _write_npz(tmp_path / "good", times_days=np.arange(10.0),
                   p_s=truth.copy(), lat=LAT, lon=LON)
    assert jw.main(["--ours", p, "--label", "good", "--days", "1,2",
                    "--max-rmse-hpa", "1.0", "--max-rms-ratio", "5"]) == 0
    assert "JW_DUO_GATE: PASS" in capsys.readouterr().out


def test_rmse_aligns_the_two_longitude_conventions_exactly():
    """The correctness property of ps_rmse: the SAME field stored on the
    oracle's 0.5..359.5 axis and on the matrix's -179.5..179.5 axis must
    difference to exactly zero. The first implementation indexed axis 1
    (latitude) instead of the last axis and raised IndexError; a version
    that indexed nothing would silently score a 180-degree phase error as
    a huge RMSE, which is the same bug pointing the other way."""
    rng = np.random.default_rng(11)
    lon_o = LON                        # 0.5 .. 359.5
    lon_m = (LON - 180.0) % 360.0      # matrix convention, mapped back
    truth = 1.0e5 + rng.normal(0.0, 200.0, (1, LAT.size, LON.size))
    ours = np.empty_like(truth)
    ours[..., np.argsort(lon_m)] = truth[..., np.argsort(lon_o)]
    st = jw.ps_rmse(ours, lon_m, truth, lon_o, LAT)
    assert st["rmse_hPa"] == pytest.approx(0.0, abs=1e-12)
    assert st["max_abs_diff_hPa"] == pytest.approx(0.0, abs=1e-12)
    # Non-vacuity: a genuine half-globe phase error must NOT score zero.
    shifted = np.roll(truth, LON.size // 2, axis=-1)
    bad = jw.ps_rmse(shifted, lon_o, truth, lon_o, LAT)
    assert bad["rmse_hPa"] > 1.0


def test_rmse_rejects_a_latitude_axis_of_the_wrong_length():
    rng = np.random.default_rng(12)
    a = 1.0e5 + rng.normal(0.0, 100.0, (1, LAT.size, LON.size))
    with pytest.raises(ValueError, match="latitude axis"):
        jw.ps_rmse(a, LON, a, LON, LAT[:-1])


# ============================================================== round 3
# Round 2 rejected the gate: it computed RMSE and then gated on amplitude.

def test_amplitude_only_gate_is_refused(monkeypatch, tmp_path, capsys):
    """--max-rms-ratio alone must not be accepted as an oracle gate."""
    monkeypatch.setattr(jw, "load_oracle", _oracle_stub())
    p = _write_npz(tmp_path / "a", times_days=np.arange(10.0),
                   p_s=np.full((10, LAT.size, LON.size), 1.0e5),
                   lat=LAT, lon=LON)
    assert jw.main(["--ours", p, "--label", "a", "--days", "1,2",
                    "--max-rms-ratio", "5"]) == 1
    assert "cannot stand alone" in capsys.readouterr().out


def test_uniform_offset_impostor_now_FAILS_the_gate(monkeypatch, tmp_path,
                                                    capsys):
    """THE round-2 RED, made permanent. ours = duo + 5000 Pa has amplitude
    ratio 1.0 and would have PASSED the old amplitude gate; it is a 50 hPa
    mass bias and must fail on distance."""
    truth = np.full((10, LAT.size, LON.size), 1.0e5)
    monkeypatch.setattr(jw, "load_oracle", _oracle_stub(truth))
    p = _write_npz(tmp_path / "off", times_days=np.arange(10.0),
                   p_s=truth + 5000.0, lat=LAT, lon=LON)
    assert jw.main(["--ours", p, "--label", "offset", "--days", "1,2",
                    "--max-rmse-hpa", "1.0", "--max-rms-ratio", "5"]) == 1
    out = capsys.readouterr().out
    assert "RMSE vs duo = 50.0000" in out and "bias +50.0000" in out


def test_nan_threshold_is_refused(monkeypatch, tmp_path, capsys):
    """`x > nan` is always False, so a nan threshold disabled the gate."""
    monkeypatch.setattr(jw, "load_oracle", _oracle_stub())
    p = _write_npz(tmp_path / "n", times_days=np.arange(10.0),
                   p_s=np.full((10, LAT.size, LON.size), 5.0e5) * 0 + 1.0e5,
                   lat=LAT, lon=LON)
    assert jw.main(["--ours", p, "--label", "a", "--days", "1,2",
                    "--max-rmse-hpa", "nan"]) == 1
    assert "not finite" in capsys.readouterr().out


def test_rmse_gate_passes_an_actually_matching_arm(monkeypatch, tmp_path,
                                                   capsys):
    rng = np.random.default_rng(21)
    truth = 1.0e5 + rng.normal(0.0, 100.0, (10, LAT.size, LON.size))
    monkeypatch.setattr(jw, "load_oracle", _oracle_stub(truth))
    p = _write_npz(tmp_path / "same", times_days=np.arange(10.0),
                   p_s=truth.copy(), lat=LAT, lon=LON)
    assert jw.main(["--ours", p, "--label", "same", "--days", "1,2",
                    "--max-rmse-hpa", "0.001"]) == 0
    assert "JW_DUO_GATE: PASS" in capsys.readouterr().out


def test_ps_rmse_is_a_mean_over_frames_not_a_sum():
    """With w summing to 1 per frame, a stack of nf frames must not make
    the 'rms' grow as sqrt(nf)."""
    rng = np.random.default_rng(22)
    one = rng.normal(0.0, 100.0, (1, LAT.size, LON.size))
    many = np.repeat(one, 7, axis=0)
    a = jw.ps_rmse(one, LON, np.zeros_like(one), LON, LAT)["rmse_hPa"]
    b = jw.ps_rmse(many, LON, np.zeros_like(many), LON, LAT)["rmse_hPa"]
    assert b == pytest.approx(a, rel=1e-12)


def test_ps_rmse_rejects_incommensurable_longitude_axes():
    rng = np.random.default_rng(23)
    a = 1.0e5 + rng.normal(0.0, 100.0, (1, LAT.size, LON.size))
    shifted = LON + 0.25          # same count, different physical centres
    with pytest.raises(ValueError, match="same physical centres"):
        jw.ps_rmse(a, shifted, a, LON, LAT)


def test_load_ours_rejects_a_non_monotonic_time_axis(tmp_path):
    kw = _good_npz_kwargs()
    kw["times_days"] = np.array([0.0, 2.0, 1.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        jw.load_ours(_write_npz(tmp_path, **kw))


def test_load_ours_rejects_nonfinite_times(tmp_path):
    kw = _good_npz_kwargs()
    kw["times_days"] = np.array([0.0, 1.0, np.nan])
    with pytest.raises(ValueError, match="non-finite"):
        jw.load_ours(_write_npz(tmp_path, **kw))


def test_at_day_rejects_a_nan_time():
    """abs(nan - day) > tol is False, so a NaN would satisfy the check."""
    with pytest.raises(ValueError, match="non-finite"):
        jw.at_day(np.array([0.0, np.nan, 2.0]), 1.0)
