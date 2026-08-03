"""CMOR append must not write the same time twice, and must stay monotonic.

Found in a real century arm (2026-08-01): `prodA`'s Amon time axis was
`105, 105, 135.5, ...` — two rows for the same month with different rsut,
because a chained link re-flushed a window it had already written and the
append path used `t_idx = len(time)` with no duplicate check. That made a
matched-window comparison ambiguous (-6.8% vs -5.7% albedo depending on which
row was picked), which is exactly the class of silent corruption that turns a
measured result into a coin flip.

A run that completes in ONE job was clean, so this only bites chained runs —
i.e. every century simulation.
"""
import logging

import numpy as np
import pytest

pytest.importorskip("netCDF4")
pytest.importorskip("xarray")

from legoesm.io.cmor_output import CFWriter  # noqa: E402


def _writer(tmp_path):
    return CFWriter(output_dir=str(tmp_path), experiment_id="amip",
                    model_id="legoESM-1-0")


def _write(w, t, value, nlat=4, nlon=8):
    """A 2-D surface field is (nlat, nlon) -- not (1, nlat, nlon)."""
    lat = np.linspace(-80.0, 80.0, nlat)
    lon = np.linspace(0.0, 315.0, nlon)
    data = np.full((nlat, nlon), value, dtype=np.float64)
    return w.write_field(
        var_name="rsdt", data=data, lat=lat, lon=lon,
        time=float(t), time_bounds=(float(t) - 15.0, float(t) + 15.0))


def _times(path):
    from netCDF4 import Dataset
    with Dataset(str(path)) as d:
        return np.asarray(d["time"][:], dtype=np.float64)


def _values(path):
    from netCDF4 import Dataset
    with Dataset(str(path)) as d:
        return np.asarray(d["rsdt"][:], dtype=np.float64)


def test_appends_distinct_times_normally(tmp_path):
    w = _writer(tmp_path)
    for t in (15.5, 45.0, 74.5):
        p = _write(w, t, 300.0 + t)
    np.testing.assert_allclose(_times(p), [15.5, 45.0, 74.5])


def test_repeated_time_is_skipped_not_duplicated(tmp_path, caplog):
    """The exact prodA failure: a chained restart re-flushes a written month."""
    w = _writer(tmp_path)
    _write(w, 105.0, 334.5)
    with caplog.at_level(logging.WARNING, logger="legoesm.io.cmor_output"):
        p = _write(w, 105.0, 332.5)          # the re-flush
    np.testing.assert_allclose(_times(p), [105.0])
    # the FIRST (complete) value is kept, not the partial re-accumulation
    assert float(_values(p)[0].mean()) == pytest.approx(334.5)
    assert any("ALREADY on disk" in r.getMessage() for r in caplog.records), \
        "the skip must be loud, not silent"


def test_out_of_order_time_raises(tmp_path):
    """A non-monotonic axis breaks every downstream time-aligned comparison."""
    w = _writer(tmp_path)
    _write(w, 105.0, 334.5)
    _write(w, 135.5, 330.0)
    with pytest.raises(ValueError, match="monotonically increasing"):
        _write(w, 74.5, 340.0)


def test_guard_survives_a_realistic_chain_sequence(tmp_path):
    """Simulate links that each re-flush their first window on restart."""
    w = _writer(tmp_path)
    written = [15.5, 45.0, 74.5]
    for t in written:
        _write(w, t, 300.0 + t)
    # link 2 restarts inside the 74.5 window and re-flushes it, then continues
    for t in (74.5, 105.0, 135.5):
        p = _write(w, t, 300.0 + t)
    got = _times(p)
    assert len(got) == len(set(np.round(got, 6))), f"duplicates in {got}"
    np.testing.assert_allclose(got, [15.5, 45.0, 74.5, 105.0, 135.5])


def test_guard_covers_the_xarray_fallback_path(tmp_path, monkeypatch):
    """The fallback (no netCDF4) must guard too — codex found it unguarded.

    Forces `_import_netcdf4` to return None so the xarray-concat branch runs,
    then repeats a time. Without the hoisted guard this branch appends a
    duplicate silently.
    """
    from legoesm.io import cmor_output

    w = _writer(tmp_path)
    p = _write(w, 105.0, 334.5)
    monkeypatch.setattr(cmor_output, "_import_netcdf4", lambda: None)
    p = _write(w, 105.0, 332.5)              # re-flush via the fallback
    np.testing.assert_allclose(_times(p), [105.0])
    assert float(_values(p)[0].mean()) == pytest.approx(334.5)
    # and the fallback still APPENDS a genuinely new time
    p = _write(w, 135.5, 330.2)
    np.testing.assert_allclose(_times(p), [105.0, 135.5])


def test_reproduces_the_observed_prodA_axis_without_the_duplicate(tmp_path):
    """End-to-end against the real failure.

    prodA's Amon axis was 105, 105, 135.5, ... 349.5 -- rows 0 and 1 sharing a
    time AND bounds but carrying different data. The later months are all
    present at clean 30.5-day spacing, so dropping the repeat loses nothing.
    """
    w = _writer(tmp_path)
    seq = [(105.0, 334.48), (105.0, 332.45), (135.5, 330.23), (166.0, 327.96),
           (196.5, 327.24), (227.5, 328.39), (258.0, 331.79)]
    for t, v in seq:
        p = _write(w, t, v)
    got = _times(p)
    np.testing.assert_allclose(got, [105.0, 135.5, 166.0, 196.5, 227.5, 258.0])
    assert float(_values(p)[0].mean()) == pytest.approx(334.48)
    # gapless: every gap is one month
    assert np.all(np.abs(np.diff(got) - 30.5) < 1.0)
