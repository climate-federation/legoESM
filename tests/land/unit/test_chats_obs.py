"""Smoke test for :mod:`legoesm.land.boundary_data.chats_obs`.

Loads the May 2007 CHATS 30-min CSV and checks shape/time-index/height
invariants documented in the Zenodo 17426258 README.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from legoesm.land.boundary_data.chats_obs import (
    TA_HEIGHTS,
    load_chats_obs,
)

# Data is shipped outside the package under docs/. Skip cleanly when the
# repo checkout does not include the vendored Zenodo files (e.g. minimal
# CI images) so the test stays useful locally without breaking CI.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CHATS_DIR = (
    _REPO_ROOT
    / "docs"
    / "MLC_experiment_plan"
    / "Data"
    / "Zenodo_17426258_CHATS_30-min"
)
_MAY_CSV = _CHATS_DIR / "chats_30min_data_2007_05_ver_250801.csv"


@pytest.mark.skipif(
    not _MAY_CSV.exists(),
    reason=f"CHATS May 2007 CSV not present at {_MAY_CSV}",
)
def test_load_chats_obs_may_2007() -> None:
    obs = load_chats_obs(_MAY_CSV)

    # 31 days * 48 half-hours = 1488 samples.
    assert len(obs.time) == 1488
    assert obs.Ta.shape == (1488, 12)
    assert obs.Qh.shape == (1488, 13)

    # README: timestamps mark the center of each 30-min interval, so the
    # first UTC sample of the month is 00:15 and the last is 23:45.
    assert obs.time[0] == pd.Timestamp("2007-05-01 00:15:00", tz="UTC")
    assert obs.time[-1] == pd.Timestamp("2007-05-31 23:45:00", tz="UTC")

    expected_heights = np.array(
        [1.55, 3.11, 4.54, 6.07, 7.65, 9.04, 10.04, 11.12,
         13.57, 17.52, 22.55, 28.51]
    )
    assert np.all(obs.Ta_heights == expected_heights)
    # Module-level constant and container attribute stay in sync.
    assert np.all(TA_HEIGHTS == expected_heights)
