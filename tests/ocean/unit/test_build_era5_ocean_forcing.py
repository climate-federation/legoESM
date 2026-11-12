"""Direct test of scripts/data/build_era5_ocean_forcing.py helpers (no cdo, no pool reads)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.data import build_era5_ocean_forcing as B  # noqa: E402


def test_month_span_and_source_paths():
    assert B.month_span(0) == (1993, 1, 31) and B.month_span(37) == (1996, 2, 29)
    assert B.source_file("an", "165", 1993, 1).name == "E5sf00_1D_1993-01_165.grb"
    assert B.source_file("fc", "169", 2022, 12).name == "E5sf12_1D_2022-12_169.grb"
    assert len({c[0] for c in B.CHANNELS}) == len(B.CHANNELS) == 17
    assert set(B.UNITS) == {c[0] for c in B.CHANNELS}


def test_stack_day_order_and_validator(tmp_path):
    f = {name: np.full((3, 180, 360), float(i)) for i, (name, *_) in enumerate(B.CHANNELS)}
    a = B.stack_day(f, 2)
    assert a.shape == (17, 180, 360) and a.dtype == np.float32
    assert [float(a[i, 0, 0]) for i in range(17)] == list(range(17))
    good, nan = tmp_path / "g.npy", tmp_path / "n.npy"
    np.save(good, a)
    b = a.copy(); b[3, 5, 5] = np.nan
    np.save(nan, b)
    assert B.done_ok(good) and not B.done_ok(nan)


def test_climatology_helpers(tmp_path):
    from scripts.data import build_ocean_wb_climatologies as C
    assert C.noleap_doy("1995-01-01") == 0 and C.noleap_doy("1995-12-31") == 364
    assert C.noleap_doy("1996-02-29") == C.noleap_doy("1996-02-28") == 58
    assert C.noleap_doy("1996-03-01") == C.noleap_doy("1995-03-01") == 59
    assert C.noleap_doy("1996-12-31") == 364
