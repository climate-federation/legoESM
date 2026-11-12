"""Direct test of ``scripts/data/build_ocean_wb_climatologies.py``'s day-of-year binning."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.data.build_ocean_wb_climatologies import noleap_doy  # noqa: E402


def test_noleap_doy_pools_feb29_and_keeps_365_bins():
    assert noleap_doy(np.datetime64("1993-01-01")) == 0
    assert noleap_doy(np.datetime64("1993-12-31")) == 364
    assert noleap_doy(np.datetime64("1996-02-28")) == 58
    assert noleap_doy(np.datetime64("1996-02-29")) == 58     # pooled with Feb 28
    assert noleap_doy(np.datetime64("1996-03-01")) == noleap_doy(np.datetime64("1993-03-01")) == 59
    assert noleap_doy(np.datetime64("1996-12-31")) == 364
    assert noleap_doy(np.datetime64("2000-02-29")) == 58     # 400-year leap
    assert noleap_doy(np.datetime64("1900-03-01")) == 59     # century, not leap
