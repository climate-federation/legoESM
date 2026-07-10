"""Tests for scripts/validate/inspect_checkpoint_realism.py — the fast
physical-realism gate on a raw model checkpoint.  Verifies the pure bounds logic
(healthy pass, NaN/out-of-bounds/negative/runaway fails, context reporting,
checkpoint resolution) on synthetic ``.npz`` checkpoints so a real regression
(a blow-up or an unphysical field) can't slip past."""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

_MOD = (pathlib.Path(__file__).resolve().parents[2]
        / "scripts" / "validate" / "inspect_checkpoint_realism.py")
_spec = importlib.util.spec_from_file_location("inspect_checkpoint_realism", _MOD)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _healthy(shp=(2, 4, 4, 5)):
    s3 = shp[:3]
    return dict(
        T=np.full(shp, 250.0), u=np.zeros(shp), v=np.zeros(shp),
        p_s=np.full(s3, 1.0e5), q_v=np.full(shp, 0.005),
        q_c=np.zeros(shp), q_r=np.zeros(shp),
        carry_T_land=np.full(s3, 288.0), carry_w_land=np.full(s3, 100.0),
        carry_snow=np.zeros(s3), carry_seg_precip=np.full(s3, 3.0),
        carry_held_lw_up_toa=np.full(s3, 240.0),
        day=np.asarray(19.0), config_json=np.asarray("{}"))


def _write(tmp_path, name="checkpoint_day_0019.npz", **ov):
    d = _healthy()
    d.update(ov)
    f = tmp_path / name
    np.savez(f, **d)
    return f


def test_healthy_checkpoint_passes(tmp_path):
    r = mod.inspect_checkpoint_realism(_write(tmp_path))
    assert r["passed"] is True and r["n_violations"] == 0
    assert r["day"] == 19.0
    assert r["context"]["air_T"]["in_hint"] is True
    assert r["context"]["land_T"]["value"] == pytest.approx(288.0)
    assert mod.format_realism_report(r).startswith("PASS")


def test_nan_in_T_fails(tmp_path):
    T = np.full((2, 4, 4, 5), 250.0)
    T[0, 0, 0, 0] = np.nan
    r = mod.inspect_checkpoint_realism(_write(tmp_path, T=T))
    assert r["passed"] is False
    assert r["fields"]["T"]["within"] is False
    assert "NaN" in r["fields"]["T"]["reason"]
    assert "VIOLATION T" in mod.format_realism_report(r)


def test_out_of_bounds_T_fails(tmp_path):
    r = mod.inspect_checkpoint_realism(_write(tmp_path, T=np.full((2, 4, 4, 5), 500.0)))
    assert r["passed"] is False
    assert "max" in r["fields"]["T"]["reason"]


def test_nan_sentinel_field_passes(tmp_path):
    """surface_T_sfc_override is all-NaN BY DESIGN (the documented "no
    override" sentinel, physics_state.py) — the NaN gate must not fire on it
    (it flagged every MPAS checkpoint as failing)."""
    r = mod.inspect_checkpoint_realism(_write(
        tmp_path,
        physstate_surface_T_sfc_override=np.full((32,), np.nan)))
    assert r["passed"] is True
    assert r["fields"]["physstate_surface_T_sfc_override"]["within"] is True


def test_nan_sentinel_field_still_gates_inf(tmp_path):
    """Inf in a NaN-sentinel field is a real defect: only the NaN half of the
    finiteness gate is exempt (synthetic-violation self-test)."""
    bad = np.full((32,), np.nan)
    bad[3] = np.inf
    r = mod.inspect_checkpoint_realism(_write(
        tmp_path, physstate_surface_T_sfc_override=bad))
    assert r["passed"] is False
    assert "Inf" in r["fields"]["physstate_surface_T_sfc_override"]["reason"]


def test_negative_humidity_fails(tmp_path):
    q = np.full((2, 4, 4, 5), 0.005)
    q[0, 0, 0, 0] = -1e-4
    r = mod.inspect_checkpoint_realism(_write(tmp_path, q_v=q))
    assert r["passed"] is False
    assert r["fields"]["q_v"]["within"] is False    # negative -> below lo=0.0
    assert "< 0.0" in r["fields"]["q_v"]["reason"]


def test_runaway_land_T_fails(tmp_path):
    """The #733 multilayer cold-start failure mode: land skin-T runs to ~900 K."""
    r = mod.inspect_checkpoint_realism(
        _write(tmp_path, carry_T_land=np.full((2, 4, 4), 900.0)))
    assert r["passed"] is False
    assert r["fields"]["carry_T_land"]["within"] is False


def test_string_keys_skipped_not_crash(tmp_path):
    r = mod.inspect_checkpoint_realism(_write(tmp_path))
    assert "config_json" not in r["fields"]     # non-numeric skipped


def test_find_latest_checkpoint_in_dir(tmp_path):
    _write(tmp_path, name="checkpoint_day_0005.npz")
    _write(tmp_path, name="checkpoint_day_0019.npz")
    r = mod.inspect_checkpoint_realism(tmp_path)  # dir -> latest by sort
    assert r["checkpoint"] == "checkpoint_day_0019.npz"


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        mod.inspect_checkpoint_realism(tmp_path)               # empty dir
    with pytest.raises(FileNotFoundError):
        mod.inspect_checkpoint_realism(tmp_path / "nope.npz")  # missing file


def test_context_outside_hint_flagged_but_not_gated(tmp_path):
    """Low OLR is a coarse-res/spin-up bias -> flagged as out-of-hint context,
    but NOT a violation (context is reported, not gated)."""
    r = mod.inspect_checkpoint_realism(
        _write(tmp_path, carry_held_lw_up_toa=np.full((2, 4, 4), 200.0)))
    assert r["passed"] is True                  # context never gates
    assert r["context"]["OLR"]["in_hint"] is False
    assert "(!)" in mod.format_realism_report(r)


def test_complex_spectral_nan_fails(tmp_path):
    """A NaN in a complex spectral coefficient must be caught, not skipped."""
    vor = np.ones((3, 4), dtype=np.complex128)
    vor[0, 0] = complex(np.nan, 0.0)
    r = mod.inspect_checkpoint_realism(_write(tmp_path, vor_hat=vor))
    assert r["passed"] is False
    assert r["fields"]["vor_hat"]["within"] is False
    assert r["fields"]["vor_hat"]["n_nan"] == 1


def test_trc_prefixed_negative_humidity_fails(tmp_path):
    """MPAS/spectral tracer alias trc_q_v must get the q_v water bounds."""
    trc = np.full((2, 4, 4, 5), 0.005)
    trc[0, 0, 0, 0] = -1e-4
    r = mod.inspect_checkpoint_realism(_write(tmp_path, trc_q_v=trc))
    assert r["passed"] is False
    assert r["fields"]["trc_q_v"]["within"] is False
    assert "< 0.0" in r["fields"]["trc_q_v"]["reason"]


def test_latest_checkpoint_is_numeric_max_not_lexicographic(tmp_path):
    """day_10000 must beat day_9999 (numeric), not lose lexicographically."""
    _write(tmp_path, name="checkpoint_day_9999.npz")
    _write(tmp_path, name="checkpoint_day_10000.npz")
    r = mod.inspect_checkpoint_realism(tmp_path)
    assert r["checkpoint"] == "checkpoint_day_10000.npz"


def _write_series(tmp_path, days, land_T):
    for d, lt in zip(days, land_T):
        _write(tmp_path, name=f"checkpoint_day_{d:04d}.npz",
               day=np.asarray(float(d)),
               carry_T_land=np.full((2, 4, 4), float(lt)))


def test_timeseries_builds_sorted_trajectory(tmp_path):
    _write_series(tmp_path, [5, 10, 15], [288.0, 288.0, 288.0])
    ts = mod.inspect_checkpoint_timeseries(tmp_path)
    assert ts["n"] == 3
    assert ts["days"] == [5.0, 10.0, 15.0]           # numeric-sorted
    assert ts["all_passed"] is True
    assert len(ts["series"]["land_T"]) == 3
    assert "TRAJECTORY" in mod.format_timeseries(ts)


def test_timeseries_detects_land_T_drift(tmp_path):
    # +0.6 K/day sustained (12 K over 20 days) > 0.5 K/day threshold
    _write_series(tmp_path, [5, 10, 15, 20, 25],
                  [288.0, 291.0, 294.0, 297.0, 300.0])
    ts = mod.inspect_checkpoint_timeseries(tmp_path)
    assert ts["drift"]["land_T"]["drifting"] is True
    assert ts["drift"]["land_T"]["slope_per_day"] == pytest.approx(0.6, abs=1e-6)
    assert ts["any_drifting"] is True
    assert ts["all_passed"] is True                  # still within bounds
    assert "(!)" in mod.format_timeseries(ts)


def test_timeseries_stationary_not_drifting(tmp_path):
    _write_series(tmp_path, [5, 10, 15, 20, 25], [288.0] * 5)
    ts = mod.inspect_checkpoint_timeseries(tmp_path)
    assert ts["any_drifting"] is False
    assert ts["drift"]["land_T"]["slope_per_day"] == pytest.approx(0.0, abs=1e-9)


def test_timeseries_single_checkpoint_no_drift(tmp_path):
    _write_series(tmp_path, [5], [288.0])
    ts = mod.inspect_checkpoint_timeseries(tmp_path)
    assert ts["n"] == 1
    assert ts["any_drifting"] is False               # <2 points -> no slope
    assert np.isnan(ts["drift"]["land_T"]["slope_per_day"])


def test_timeseries_slope_uses_day_spacing_not_index(tmp_path):
    """Unequal spacing: slope must be per-DAY (0->1->99), not per-checkpoint-index
    (else a 100 K jump over 99 days reads as ~50 K/index)."""
    _write_series(tmp_path, [1, 2, 100], [288.0, 288.0, 388.0])
    ts = mod.inspect_checkpoint_timeseries(tmp_path)
    slope = ts["drift"]["land_T"]["slope_per_day"]
    assert slope == pytest.approx(1.015, abs=0.05)   # day-based ~1.0, not ~50
    assert ts["drift"]["land_T"]["drifting"] is True


def test_timeseries_recent_nan_shrinks_window_and_is_caught(tmp_path):
    """A recent all-NaN field must NOT let drift reach back to stale finite data,
    AND must trip the per-checkpoint gate + missing-context surfacing."""
    _write(tmp_path, name="checkpoint_day_0018.npz", day=np.asarray(18.0),
           carry_T_land=np.full((2, 4, 4), 288.0))
    _write(tmp_path, name="checkpoint_day_0019.npz", day=np.asarray(19.0),
           carry_T_land=np.full((2, 4, 4), 294.0))
    _write(tmp_path, name="checkpoint_day_0020.npz", day=np.asarray(20.0),
           carry_T_land=np.full((2, 4, 4), np.nan))     # lost/blown carry
    ts = mod.inspect_checkpoint_timeseries(tmp_path, last_n=2)  # window = days 19,20
    assert np.isnan(ts["drift"]["land_T"]["slope_per_day"])     # window shrank to 1
    assert ts["drift"]["land_T"]["drifting"] is False
    assert ts["all_passed"] is False                            # NaN caught by gate
    assert (20.0, ["land_T"]) in ts["missing_context"]


def test_timeseries_missing_context_field_surfaced(tmp_path):
    """A checkpoint that dropped its OLR carry is surfaced, not hidden as ok=Y."""
    d = _healthy()
    del d["carry_held_lw_up_toa"]                     # no OLR carry this checkpoint
    d["day"] = np.asarray(12.0)
    np.savez(tmp_path / "checkpoint_day_0012.npz", **d)
    ts = mod.inspect_checkpoint_timeseries(tmp_path)
    assert ts["all_context_present"] is False
    assert ts["missing_context"] == [(12.0, ["OLR"])]
    assert ts["all_passed"] is True                  # bounds gate itself still passes
    assert "MISSING CONTEXT" in mod.format_timeseries(ts)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
