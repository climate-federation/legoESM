"""CMOR accumulator sidecar persistence across a restart-chained AMIP run.

The restart-chain launcher resumes each ~10-day SLURM link from a checkpoint,
but the CMOR **monthly** accumulator was recreated empty every link, so a
calendar month (~30 days) never completed within one link and the ``Amon``
means were never written.  The fix persists the three CMOR accumulators
(``SpatialMonthlyAccumulator`` / ``SpatialDailyAccumulator`` / zonal
``MonthlyAccumulator``) to an additive ``cmor_accum_day_*.npz`` sidecar written
next to the checkpoint, and restores it on resume.

These tests pin:

* ``get_state`` / ``set_state`` exact round-trip identity for each accumulator
  (state and ``finalize()`` equal),
* the KEY cross-restart claim — a month split at day 15 across a
  ``save_cmor_accumulators`` → fresh diagnostics → ``load_cmor_accumulators``
  boundary completes, with the monthly mean equal to the exact average of all
  31 daily fields,
* daily + zonal restore through the same sidecar,
* backward-compat (missing sidecar → ``False``, no crash),
* a dims-mismatch / unknown-version ``set_state`` raising ``ValueError``.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from legoesm.diagnostics.monthly_means import (
    MonthlyAccumulator,
    SpatialDailyAccumulator,
    SpatialMonthlyAccumulator,
)

# Tiny CMIP target grid so the payload arrays stay small (res=30° → 6×12).
_RES_DEG = 30.0
_NLAT = 6
_NLON = 12
_NLEV = 4
_MON2D = ("tas", "rsut", "rlut")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _make_diag(output_dir):
    """A minimal CMIP-enabled ``DiagnosticCollector`` (all three accumulators
    + a ``cf_writer``, no NetCDF write until a flush)."""
    from legoesm.driver.diagnostics import DiagnosticCollector

    sigma = np.linspace(0.1, 0.9, _NLEV)
    dsigma = np.full(_NLEV, 1.0 / _NLEV)
    return DiagnosticCollector(
        nlev=_NLEV,
        sigma_full=sigma,
        dsigma=dsigma,
        experiment_id="test",
        cmip_output=True,
        output_dir=str(output_dir),
        cmip_resolution_deg=_RES_DEG,
        n_days=400,
    )


def _day_fields(rng):
    """One day's synthetic fields: 3 named 2-D fields + a 3-D field, float64."""
    base = {
        "tas": 280.0,
        "rsut": 100.0,
        "rlut": 240.0,
    }
    fields_2d = {
        name: (rng.standard_normal((_NLAT, _NLON)) + base[name])
        for name in _MON2D
    }
    field_3d = rng.standard_normal((_NLAT, _NLON, _NLEV)) + 250.0
    return fields_2d, field_3d


def _seq_mean(frames):
    """Sequential float64 mean matching the accumulator's day-by-day sum."""
    acc = np.zeros_like(frames[0], dtype=np.float64)
    for f in frames:
        acc += f.astype(np.float64)
    return acc / len(frames)


# --------------------------------------------------------------------------- #
# get_state / set_state round-trip identity
# --------------------------------------------------------------------------- #
def _assert_spatial_monthly_equal(a, b):
    assert a._call_counts == b._call_counts
    assert a._max_count_ever == b._max_count_ever
    assert set(a._data_2d) == set(b._data_2d)
    assert set(a._data_3d) == set(b._data_3d)
    for store_a, store_b in ((a._data_2d, b._data_2d), (a._data_3d, b._data_3d)):
        for key in store_a:
            assert set(store_a[key]) == set(store_b[key])
            for name in store_a[key]:
                sa, ca = store_a[key][name]
                sb, cb = store_b[key][name]
                assert ca == cb
                np.testing.assert_array_equal(sa, sb)
                assert sa.dtype == sb.dtype == np.float64


def _assert_daily_equal(a, b):
    assert a._call_counts == b._call_counts
    assert a._max_count_ever == b._max_count_ever
    assert a.track_extremes == b.track_extremes
    assert set(a._data) == set(b._data)
    for key in a._data:
        assert set(a._data[key]) == set(b._data[key])
        for name in a._data[key]:
            sa, ca, mna, mxa = a._data[key][name]
            sb, cb, mnb, mxb = b._data[key][name]
            assert ca == cb
            np.testing.assert_array_equal(sa, sb)
            if mna is None:
                assert mnb is None and mxb is None
            else:
                np.testing.assert_array_equal(mna, mnb)
                np.testing.assert_array_equal(mxa, mxb)


def _assert_monthly_zonal_equal(a, b):
    assert a._call_counts == b._call_counts
    assert a._max_count_ever == b._max_count_ever
    assert set(a._data) == set(b._data)
    assert set(a._scalars) == set(b._scalars)
    for key in a._data:
        assert set(a._data[key]) == set(b._data[key])
        for fkey in a._data[key]:
            sa, ca = a._data[key][fkey]
            sb, cb = b._data[key][fkey]
            np.testing.assert_array_equal(sa, sb)
            np.testing.assert_array_equal(ca, cb)
    for key in a._scalars:
        assert set(a._scalars[key]) == set(b._scalars[key])
        for name in a._scalars[key]:
            sa, ca = a._scalars[key][name]
            sb, cb = b._scalars[key][name]
            assert sa == sb  # exact float
            assert ca == cb


def test_spatial_monthly_roundtrip():
    a = SpatialMonthlyAccumulator(nlat=_NLAT, nlon=_NLON, nlev=_NLEV)
    rng = np.random.default_rng(1)
    for doy in (5, 40, 70):  # Jan, Feb, Mar
        f2d, f3d = _day_fields(rng)
        a.add_2d(doy, 0, f2d)
        a.add_3d(doy, 0, {"ta": f3d})
    state = a.get_state()
    b = SpatialMonthlyAccumulator(nlat=_NLAT, nlon=_NLON, nlev=_NLEV)
    b.set_state(state)
    _assert_spatial_monthly_equal(a, b)
    fa = a.finalize(min_sample_fraction=0)
    fb = b.finalize(min_sample_fraction=0)
    assert fa["months"] == fb["months"]
    for k in fa:
        if k == "months":
            continue
        np.testing.assert_array_equal(fa[k], fb[k])


def test_spatial_daily_roundtrip():
    a = SpatialDailyAccumulator(nlat=_NLAT, nlon=_NLON, track_extremes={"tas"})
    rng = np.random.default_rng(2)
    # Two samples on day 5 (non-trivial min/max), one on day 6.
    for doy, n in ((5, 2), (6, 1)):
        for _ in range(n):
            f2d, _ = _day_fields(rng)
            a.add_2d(doy, 0, f2d)
    state = a.get_state()
    b = SpatialDailyAccumulator(nlat=_NLAT, nlon=_NLON, track_extremes={"tas"})
    b.set_state(state)
    _assert_daily_equal(a, b)
    fa = a.finalize(min_sample_fraction=0)
    fb = b.finalize(min_sample_fraction=0)
    assert fa["days"] == fb["days"]
    for k in fa:
        if k == "days":
            continue
        np.testing.assert_array_equal(fa[k], fb[k])


def test_monthly_zonal_roundtrip():
    a = MonthlyAccumulator(nlev=_NLEV, n_lat_bins=8)
    rng = np.random.default_rng(3)
    lat = np.linspace(-89.0, 89.0, 20)
    for doy in (5, 40):
        a.add_2d(doy, 0, {"tas": rng.standard_normal(20) + 280.0}, lat)
        a.add_3d(doy, 0, {"T": rng.standard_normal((20, _NLEV)) + 250.0}, lat)
        a.add_scalar(doy, 0, {"toa": float(rng.standard_normal())})
    state = a.get_state()
    b = MonthlyAccumulator(nlev=_NLEV, n_lat_bins=8)
    b.set_state(state)
    _assert_monthly_zonal_equal(a, b)
    fa = a.finalize(min_sample_fraction=0)
    fb = b.finalize(min_sample_fraction=0)
    assert fa["months"] == fb["months"]
    for k in fa:
        if k in ("months", "lat"):
            continue
        np.testing.assert_array_equal(fa[k], fb[k])


# --------------------------------------------------------------------------- #
# KEY cross-restart test: a month split at day 15 completes on resume.
# --------------------------------------------------------------------------- #
def test_cross_restart_month_completes(tmp_path):
    year = 0
    rng = np.random.default_rng(10)
    # Pre-generate 31 January days so the reference average is exact.
    days = {}  # doy -> (fields_2d, field_3d)
    for doy in range(1, 32):
        days[doy] = _day_fields(rng)

    # --- link 1: accumulate days 1..15 into diagnostics A -----------------
    diag_a = _make_diag(tmp_path / "runA")
    for doy in range(1, 16):
        f2d, f3d = days[doy]
        diag_a._spatial_monthly.add_2d(doy, year, f2d)
        diag_a._spatial_monthly.add_3d(doy, year, {"ta": f3d})
    sidecar = tmp_path / "cmor_accum_day_0015.npz"
    diag_a.save_cmor_accumulators(sidecar)
    assert sidecar.exists()

    # --- link 2: FRESH diagnostics B, restore, accumulate days 16..31 -----
    diag_b = _make_diag(tmp_path / "runB")
    # Fresh accumulator is empty before restore.
    assert diag_b._spatial_monthly.finalize().get("months") == []
    assert diag_b.load_cmor_accumulators(sidecar) is True
    for doy in range(16, 32):
        f2d, f3d = days[doy]
        diag_b._spatial_monthly.add_2d(doy, year, f2d)
        diag_b._spatial_monthly.add_3d(doy, year, {"ta": f3d})

    # Step into February (month 2) and pop the now-complete January.
    data = diag_b._spatial_monthly.pop_completed_months(year, 2)
    assert (year, 1) in data["months"]
    i = data["months"].index((year, 1))

    # January mean == exact average of ALL 31 daily fields (spanning restart).
    for var in _MON2D:
        ref = _seq_mean([days[doy][0][var] for doy in range(1, 32)])
        got = data[f"field_2d_{var}"][i]
        np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-9)
    ref3d = _seq_mean([days[doy][1] for doy in range(1, 32)])
    got3d = data["field_3d_ta"][i]
    np.testing.assert_allclose(got3d, ref3d, rtol=1e-9, atol=1e-9)


def test_cross_restart_daily_and_zonal(tmp_path):
    """Daily + zonal accumulators also survive the sidecar round-trip."""
    year = 0
    rng = np.random.default_rng(11)
    lat = np.linspace(-89.0, 89.0, 20)
    daily = {}   # doy -> 2-D tas field
    scalars = {}  # doy -> scalar value

    diag_a = _make_diag(tmp_path / "A")
    for doy in range(1, 16):
        tas = rng.standard_normal((_NLAT, _NLON)) + 280.0
        daily[doy] = tas
        diag_a._spatial_daily.add_2d(doy, year, {"tas": tas})
        val = float(rng.standard_normal())
        scalars[doy] = val
        diag_a.monthly_accum.add_scalar(doy, year, {"toa": val})
    sidecar = tmp_path / "cmor_accum_day_0015.npz"
    diag_a.save_cmor_accumulators(sidecar)

    diag_b = _make_diag(tmp_path / "B")
    assert diag_b.load_cmor_accumulators(sidecar) is True
    for doy in range(16, 32):
        tas = rng.standard_normal((_NLAT, _NLON)) + 280.0
        daily[doy] = tas
        diag_b._spatial_daily.add_2d(doy, year, {"tas": tas})
        val = float(rng.standard_normal())
        scalars[doy] = val
        diag_b.monthly_accum.add_scalar(doy, year, {"toa": val})

    # Daily: all 31 days present, each mean == its single sample.
    ddata = diag_b._spatial_daily.finalize(min_sample_fraction=0)
    assert ddata["days"] == [(year, doy) for doy in range(1, 32)]
    for j, doy in enumerate(range(1, 32)):
        np.testing.assert_allclose(
            ddata["field_2d_tas"][j], daily[doy], rtol=1e-9, atol=1e-9
        )

    # Zonal scalar: January global-mean toa == average of all 31 scalars.
    zdata = diag_b.monthly_accum.finalize(min_sample_fraction=0)
    assert (year, 1) in zdata["months"]
    k = zdata["months"].index((year, 1))
    ref = np.mean([scalars[doy] for doy in range(1, 32)])
    np.testing.assert_allclose(zdata["scalar_toa"][k], ref, rtol=1e-9, atol=1e-9)


# --------------------------------------------------------------------------- #
# backward-compat + validation
# --------------------------------------------------------------------------- #
def test_load_missing_sidecar_returns_false(tmp_path):
    diag = _make_diag(tmp_path / "run")
    missing = tmp_path / "no_such_cmor_accum.npz"
    assert diag.load_cmor_accumulators(missing) is False
    # Accumulators left empty (fresh-start behaviour) — no crash.
    assert diag._spatial_monthly.finalize().get("months") == []
    assert diag._spatial_daily.finalize().get("days") == []


def test_set_state_dims_mismatch_raises():
    a = SpatialMonthlyAccumulator(nlat=_NLAT, nlon=_NLON, nlev=_NLEV)
    a.add_2d(5, 0, {"tas": np.zeros((_NLAT, _NLON))})
    with pytest.raises(ValueError):
        SpatialMonthlyAccumulator(
            nlat=_NLAT, nlon=_NLON, nlev=_NLEV + 4
        ).set_state(a.get_state())

    d = SpatialDailyAccumulator(nlat=_NLAT, nlon=_NLON)
    d.add_2d(5, 0, {"tas": np.zeros((_NLAT, _NLON))})
    with pytest.raises(ValueError):
        SpatialDailyAccumulator(nlat=_NLAT + 2, nlon=_NLON).set_state(
            d.get_state()
        )

    m = MonthlyAccumulator(nlev=_NLEV, n_lat_bins=8)
    m.add_scalar(5, 0, {"x": 1.0})
    with pytest.raises(ValueError):
        MonthlyAccumulator(nlev=_NLEV, n_lat_bins=16).set_state(m.get_state())


def test_set_state_unknown_version_raises():
    a = SpatialMonthlyAccumulator(nlat=_NLAT, nlon=_NLON, nlev=_NLEV)
    a.add_2d(5, 0, {"tas": np.zeros((_NLAT, _NLON))})
    state = dict(a.get_state())
    manifest = json.loads(str(state["__manifest__"].item()))
    manifest["version"] = 999
    state["__manifest__"] = np.asarray(json.dumps(manifest))
    with pytest.raises(ValueError):
        SpatialMonthlyAccumulator(
            nlat=_NLAT, nlon=_NLON, nlev=_NLEV
        ).set_state(state)


# --------------------------------------------------------------------------- #
# codex round-2 regressions (double-write / data-loss)
# --------------------------------------------------------------------------- #
def test_flush_then_save_yields_drained_sidecar(tmp_path):
    """HIGH #1: a sidecar re-persisted AFTER ``flush_cmip_monthly`` drains the
    completed months holds ONLY the in-progress month — so restarting from a
    checkpoint whose flush already wrote those months does not re-append them.
    """
    year = 0
    rng = np.random.default_rng(20)
    diag_a = _make_diag(tmp_path / "A")
    # Month 1 (Jan): days 1..15 — will complete + be flushed.
    for doy in range(1, 16):
        f2d, f3d = _day_fields(rng)
        diag_a._spatial_monthly.add_2d(doy, year, f2d)
        diag_a._spatial_monthly.add_3d(doy, year, {"ta": f3d})
    # Month 2 (Feb): one in-progress sample at doy 41.
    f2d, f3d = _day_fields(rng)
    diag_a._spatial_monthly.add_2d(41, year, f2d)
    diag_a._spatial_monthly.add_3d(41, year, {"ta": f3d})

    # Drain completed months WITHOUT NetCDF I/O (write=False pops < current
    # month); day 40 → day-of-year 41 → February, so January is popped.
    diag_a.flush_cmip_monthly(40.0, write=False)
    assert (year, 1) not in diag_a._spatial_monthly._data_2d  # drained

    sidecar = tmp_path / "cmor_accum_day_0040.npz"
    diag_a.save_cmor_accumulators(sidecar)

    diag_b = _make_diag(tmp_path / "B")
    assert diag_b.load_cmor_accumulators(sidecar) is True
    out = diag_b._spatial_monthly.finalize(min_sample_fraction=0)
    # The drained January must NOT reappear; only the in-progress February.
    assert (year, 1) not in out["months"]
    assert (year, 2) in out["months"]


def test_save_cmor_accumulators_atomic(tmp_path):
    """MED #3: the sidecar write is atomic — the target is always a complete
    file and no ``.tmp`` scratch is left behind, including on overwrite."""
    diag = _make_diag(tmp_path / "run")
    rng = np.random.default_rng(21)
    for doy in range(1, 6):
        f2d, _ = _day_fields(rng)
        diag._spatial_monthly.add_2d(doy, 0, f2d)
    sidecar = tmp_path / "cmor_accum_day_0005.npz"
    tmp_scratch = tmp_path / "cmor_accum_day_0005.npz.tmp"

    diag.save_cmor_accumulators(sidecar)
    assert sidecar.exists()
    assert not tmp_scratch.exists()  # no lingering temp

    # Overwrite (a later, drained re-persist) must still land atomically.
    for doy in range(6, 11):
        f2d, _ = _day_fields(rng)
        diag._spatial_monthly.add_2d(doy, 0, f2d)
    diag.save_cmor_accumulators(sidecar)
    assert sidecar.exists()
    assert not tmp_scratch.exists()

    # The overwritten target is a valid, loadable sidecar reflecting 10 samples.
    diag2 = _make_diag(tmp_path / "run2")
    assert diag2.load_cmor_accumulators(sidecar) is True
    s, c = diag2._spatial_monthly._data_2d[(0, 1)]["tas"]
    assert c == 10


def test_suppress_flag_skips_terminal_sidecar(tmp_path):
    """MED #4: ``_suppress_cmor_sidecar`` makes the driver skip the sidecar for
    the terminal final checkpoint (a completed run has no month to resume and
    its buckets are already in NetCDF)."""
    import types

    from legoesm.driver.model_driver import ModelDriver

    diag = _make_diag(tmp_path / "run")
    diag._spatial_monthly.add_2d(1, 0, {name: np.zeros((_NLAT, _NLON))
                                        for name in _MON2D})
    fake = types.SimpleNamespace(
        diagnostics=diag, _output_dir=tmp_path, _suppress_cmor_sidecar=True,
    )
    target = tmp_path / "cmor_accum_day_0015.npz"

    ModelDriver._save_cmor_accumulator_sidecar(fake, 15.0)
    assert not target.exists()  # suppressed

    fake._suppress_cmor_sidecar = False
    ModelDriver._save_cmor_accumulator_sidecar(fake, 15.0)
    assert target.exists()  # written when not suppressed
