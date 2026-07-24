"""Century-AMIP monitor: day-line stitching across chain links.

The load-bearing logic is the link stitch — each 12 h link restarts its
in-link day counter at 1, so a naive read reports the run going backwards
(and the drift check would compare unrelated windows).
"""

import numpy as np
import pytest

from scripts.validate.monitor_century_amip import (
    cmor_inventory,
    era_switches,
    main,
    parse_day_lines,
)

_LINE = ("  Day {d:6.1f}: T=[{tmin:.1f},{tmax:.1f}]K mean={tm:.1f}K  "
         "p_s=985.0hPa  |u|_max={u:.1f}m/s  CWV={c:.1f}kg/m2  (0.0 sim-days/s)")


def _log(days, tm=250.0, tmin=188.0, tmax=299.0, u=40.0, c=24.0):
    return "\n".join(_LINE.format(d=d, tmin=tmin, tmax=tmax, tm=tm, u=u, c=c)
                     for d in days)


def test_parses_fields(tmp_path):
    (tmp_path / "slurm-1.out").write_text(_log([1.0, 2.0]))
    arr = parse_day_lines(tmp_path)
    assert arr.shape == (2, 6)
    np.testing.assert_allclose(arr[0], [1.0, 188.0, 299.0, 250.0, 40.0, 24.0])


def test_stitches_link_restarts(tmp_path):
    """Link 2 restarts at day 1 — absolute days must stay monotone."""
    (tmp_path / "slurm-1.out").write_text(_log([1.0, 2.0, 3.0]))
    (tmp_path / "slurm-2.out").write_text(_log([1.0, 2.0]))
    day = parse_day_lines(tmp_path)[:, 0]
    np.testing.assert_allclose(day, [1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.all(np.diff(day) > 0)


def test_no_logs_is_empty_not_error(tmp_path):
    assert parse_day_lines(tmp_path).size == 0
    assert main([str(tmp_path)]) == 0


def test_era_switches_collected(tmp_path):
    (tmp_path / "slurm-1.out").write_text(
        "CENTURY_DECK: sim year 1923 -> volcanic 1923, ozone x.nc\n"
        + _log([1.0]))
    assert era_switches(tmp_path)[0].startswith("CENTURY_DECK: sim year 1923")


def test_cmor_inventory_reads_tables(tmp_path):
    d = tmp_path / "cmor" / "Amon"
    d.mkdir(parents=True)
    (d / "tas_Amon_legoESM_gn.nc").touch()
    (d / "pr_Amon_legoESM_gn.nc").touch()
    inv = cmor_inventory(tmp_path)
    assert sorted(inv["Amon"]) == ["pr", "tas"]


def test_missing_run_dir_raises(tmp_path):
    with pytest.raises(SystemExit):
        main([str(tmp_path / "nope")])


def test_reports_temperature_excursion(tmp_path, capsys):
    (tmp_path / "slurm-1.out").write_text(_log([1.0], tmax=430.0))
    main([str(tmp_path)])
    assert "T excursion" in capsys.readouterr().out


def test_drift_window_needs_enough_years(tmp_path, capsys):
    (tmp_path / "slurm-1.out").write_text(_log(np.arange(1.0, 30.0)))
    main([str(tmp_path), "--trend-years", "5"])
    assert "drift: needs" in capsys.readouterr().out


def test_drift_reported_when_long_enough(tmp_path, capsys):
    days = np.arange(1.0, 4000.0, 1.0)
    # Step placed at the BOUNDARY between the two comparison windows
    # (last = 3999, window = 2*365): recent > 3269 warm, prior 2539-3269 cold.
    lines = [_LINE.format(d=d, tmin=188.0, tmax=299.0,
                          tm=250.0 + (1.0 if d > 3269 else 0.0),
                          u=40.0, c=24.0) for d in days]
    (tmp_path / "slurm-1.out").write_text("\n".join(lines))
    main([str(tmp_path), "--trend-years", "2"])
    out = capsys.readouterr().out
    assert "drift (last 2 yr" in out
    assert "+1.0" in out or "+0.9" in out


def test_volcano_watch_switches_at_the_year(tmp_path, capsys):
    (tmp_path / "slurm-1.out").write_text(_log([365.0 * 45]))  # 1923+45=1968
    main([str(tmp_path), "--start-year", "1923"])
    out = capsys.readouterr().out
    assert "Agung" in out and "volcanoes simulated" in out
    assert "next volcano: El Chichon" in out
