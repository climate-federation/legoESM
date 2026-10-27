"""Direct tests for the #2105 / #2172 / #2109 re-score probes in scripts/validate/amip_bias/.

Pure pieces only (the probes' data live outside the repo): the day reduction
of hourly ERA5 samples, the noleap date labels of a CMOR day table, the
relaunch helper's log parsing, and the clear-sky replay's profile placement.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest
import xarray as xr

_DIR = pathlib.Path(__file__).resolve().parents[2] / "scripts/validate/amip_bias"
sys.path.insert(0, str(_DIR))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_daily_stats_max_min_mean_and_refusals():
    dn = _load("desert_daynight")
    h = np.arange(24.0)[:, None] * np.ones((24, 3))
    mx, mn, me = dn.daily_stats(h)
    assert np.allclose(mx, 23.0) and np.allclose(mn, 0.0) and np.allclose(me, 11.5)
    with pytest.raises(SystemExit):
        dn.daily_stats(h[:23])
    bad = h.copy(); bad[5, 1] = np.nan
    with pytest.raises(SystemExit):
        dn.daily_stats(bad)


def test_dates_of_reads_noleap_units():
    dn = _load("desert_daynight")
    d = xr.Dataset(coords={"time": ("time", [0.5, 29.5, 181.5, 210.5],
                                    {"units": "days since 1979-01-01"})})
    assert dn.dates_of(d) == ["1979-01-01", "1979-01-30", "1979-07-01", "1979-07-30"]
    assert dn.next_date("1979-01-31") == "1979-02-01"


def test_launch_argv_accepts_direct_run_line(tmp_path, monkeypatch):
    H = _load("mpas_onestep_param_grad")
    run = tmp_path / "r"; run.mkdir()
    (run / "checkpoint_day_0010.npz").write_bytes(b"")
    (run / "slurm-1.out").write_text(
        "[run] srun --ntasks=4 python -u scripts/run/run_amip.py --config c.yaml "
        "--days 30 --output /x --distributed\n")
    monkeypatch.setattr(H, "ROOT", tmp_path)
    argv = H.launch_argv("r", 10, tmp_path / "o")
    assert argv[:2] == ["--config", "c.yaml"]
    assert "--distributed" not in argv and argv[-2:] == ["--days", "1"]
    assert "/x" not in argv


def test_rlutcs_profile_identity_outside_measured_range():
    rk = _load("rlutcs_kernel")
    plev = np.array([100000.0, 50000.0, 10000.0])
    prof = np.array([1.0, 2.0, 3.0])
    p = np.array([[105000.0, 50000.0, 5000.0]])
    out = rk._on_columns(plev, prof, p, 0.0)
    assert np.allclose(out, [[0.0, 2.0, 0.0]])
