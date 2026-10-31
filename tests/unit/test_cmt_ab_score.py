"""Direct test for scripts/validate/amip_bias/cmt_ab_score.py (#2266 A/B score)."""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

_DIR = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "validate" / "amip_bias"


def _load():
    sys.path.insert(0, str(_DIR))
    spec = importlib.util.spec_from_file_location("cmt_ab_score", _DIR / "cmt_ab_score.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


LAT = np.arange(-87.5, 90.0, 5.0)


def _r(t850, t200, g850=3.0, g200=3.0):
    return {("tropics", 85000.0): t850, ("tropics", 20000.0): t200,
            ("global", 85000.0): g850, ("global", 20000.0): g200}


@pytest.mark.parametrize("on,verdict", [
    (_r(1.4, 4.0), "CONFIRM"),             # 850 falls 0.6 >= max(0.5, 0.2); 200 flat
    (_r(1.6, 4.0), "INCONCLUSIVE"),        # 850 falls 0.4: >= 0.25 (not REFUTE), < 0.5 (not CONFIRM)
    (_r(2.0, 3.0), "CONFIRM"),             # 200 falls 1.0 >= max(0.5, 0.4)
    (_r(2.0, 4.6), "REFUTE"),              # 200 rises 0.6 >= max(0.5, 0.4)
    (_r(2.0, 4.0), "REFUTE"),              # neither falls >= max(0.25, 5%)
    (_r(1.4, 4.3), "INCONCLUSIVE"),        # 850 falls 0.6, 200 rises 0.3 >= 0.25 (< BIG)
    (_r(1.4, 4.0, g200=3.3), "INCONCLUSIVE"),  # global 200 rises 0.3 >= max(0.25, 0.15)
    (_r(1.4, 4.0, g200=3.2), "CONFIRM"),   # global 200 rises 0.2 < 0.25
])
def test_prereg_verdict(on, verdict):
    m = _load()
    off = _r(2.0, 4.0)
    assert m.prereg_verdict(off, on) == verdict


def test_percent_threshold_binds_for_large_rmse():
    m = _load()
    off = _r(2.0, 10.0)
    assert m.prereg_verdict(off, _r(2.0, 9.2)) == "INCONCLUSIVE"   # 0.8 < 10% of 10
    assert m.prereg_verdict(off, _r(2.0, 9.0)) == "CONFIRM"        # 1.0 >= 10% of 10
    assert m.prereg_verdict(off, _r(2.0, 10.6)) == "REFUTE"        # rises 0.6 < 10%, but nothing falls
    assert m.prereg_verdict(off, _r(2.0, 11.0)) == "REFUTE"        # rises 1.0 >= 10%


def test_profile_rmse_and_band_mean_are_cos_lat_weighted():
    m = _load()
    ref = np.zeros_like(LAT)
    model = np.where(np.abs(LAT) < 20, 2.0, 0.0)
    assert m.profile_rmse(model, ref, LAT, m.BOXES["tropics"]) == pytest.approx(2.0)
    w = np.cos(np.deg2rad(LAT))
    expect = np.sqrt((w * model ** 2).sum() / w.sum())
    assert m.profile_rmse(model, ref, LAT, m.BOXES["global"]) == pytest.approx(expect, rel=1e-12)
    assert m.band_mean(model, LAT, m.BOXES["global"]) == pytest.approx((w * model).sum() / w.sum(), rel=1e-12)
    assert m.band_mean(LAT, LAT, m.BOXES["tropics"]) == pytest.approx(0.0, abs=1e-12)


def test_check_dates_refuses_a_shifted_or_short_window():
    m = _load()
    days = [f"1979-01-{d:02d}" for d in range(1, 31)]
    m.check_dates(days, "1979-01-01", 30)
    with pytest.raises(SystemExit):
        m.check_dates(days[1:] + ["1979-01-31"], "1979-01-01", 30)
    with pytest.raises(SystemExit):
        m.check_dates(days[:29], "1979-01-01", 30)


def test_check_arm_window_needs_the_day_n_checkpoint(tmp_path):
    m = _load()
    (tmp_path / "checkpoint_day_0020.npz").write_bytes(b"")
    with pytest.raises(SystemExit, match="day 20"):
        m.check_arm_window(str(tmp_path), "1979-01-01", 30)


def test_check_arm_window_refuses_a_record_older_than_the_last_checkpoint(tmp_path):
    import os
    m = _load()
    (tmp_path / "cmor" / "Amon").mkdir(parents=True)
    ua = tmp_path / "cmor" / "Amon" / "ua_Amon_x.nc"
    ua.write_bytes(b"")
    ck = tmp_path / "checkpoint_day_0030.npz"
    ck.write_bytes(b"")
    os.utime(ua, (1000, 1000)); os.utime(ck, (2000, 2000))
    with pytest.raises(SystemExit, match="older"):
        m.check_arm_window(str(tmp_path), "1979-01-01", 30)
