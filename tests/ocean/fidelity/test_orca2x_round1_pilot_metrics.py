"""Known-answer checks for the ORCA2X round-1 pilot metric card."""
from pathlib import Path

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2x_round1_pilot_metrics as pm,
)

RECORD = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2x_rounds/round1/"
              "acquisition/orca1ice_longrun_30day_np2")


def _column_geo(mbkt: int = 5):
    gdepw = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    return {"gdepw_1d": gdepw, "e3w_1d": np.full(7, 10.0), "nlb10": 2,
            "mbkt": np.array([[mbkt]]), "ssmask": np.ones((1, 1))}


def test_hmlp_neutral_column_reaches_the_w_level_below_mbkt():
    geo = _column_geo(mbkt=5)
    h = pm.hmlp_nemo(np.zeros((1, 1, 5)), geo, np.zeros((1, 1)), 9.8, 1000.0)
    assert h[0, 0] == geo["gdepw_1d"][5]          # nmln = mbkt + 1


def test_hmlp_strong_stratification_stays_at_nlb10():
    geo = _column_geo()
    h = pm.hmlp_nemo(np.ones((1, 1, 5)), geo, np.zeros((1, 1)), 9.8, 1000.0)
    assert h[0, 0] == geo["gdepw_1d"][1]          # nmln = nlb10 = 2 -> 10 m


def test_hmlp_crosses_threshold_at_the_planted_level_and_stretches():
    geo = _column_geo()
    zn2_c = 9.8 * pm.ZDFMXL_RHO_C / 1000.0
    rn2b = np.zeros((1, 1, 5))
    rn2b[..., 2] = 2.0 * zn2_c / 10.0              # W level jk = 4 crosses
    h = pm.hmlp_nemo(rn2b, geo, np.zeros((1, 1)), 9.8, 1000.0)
    assert h[0, 0] == geo["gdepw_1d"][3]          # nmln = jk(3) + 1 = 4 -> 30 m
    h = pm.hmlp_nemo(rn2b, geo, np.full((1, 1), 0.01), 9.8, 1000.0)
    assert h[0, 0] == pytest.approx(30.0 * 1.01)


def test_band_fractions_split_a_partial_cell_exactly():
    geo = {"e3t_0": np.array([[[60.0, 80.0, 900.0, 500.0]]]),
           "tmask": np.ones((1, 1, 4))}
    fracs = pm.band_fractions(geo, np.zeros((1, 1)))
    np.testing.assert_allclose(fracs[0][0, 0], [1.0, 0.5, 0.0, 0.0])
    np.testing.assert_allclose(fracs[1][0, 0], [0.0, 0.5, 860.0 / 900.0, 0.0])
    np.testing.assert_allclose(sum(fracs), geo["tmask"])


def test_bias_rms_and_pattern_correlation():
    w = np.array([1.0, 3.0])
    assert pm.bias_rms(np.array([2.0, -2.0]), w) == {"bias": -1.0, "rms": 2.0}
    a = np.array([0.0, 1.0, 2.0])
    assert pm.pattern_correlation(a, 3 * a + 1, np.ones(3)) == pytest.approx(1.0)
    assert pm.pattern_correlation(a, -a, np.ones(3)) == pytest.approx(-1.0)


def test_surface_ke_of_uniform_flow():
    shape = (3, 4, 1)
    geo = {k: np.ones(shape[:2]) for k in ("e1u", "e2u", "e1v", "e2v")}
    geo["umask"] = np.ones(shape)
    geo["vmask"] = np.ones(shape)
    ke = pm.surface_ke({"u": np.full(shape, 2.0), "v": np.full(shape, 1.0)}, geo)
    np.testing.assert_allclose(ke[1:], 0.5 * (4.0 + 1.0))


@pytest.mark.skipif(not RECORD.is_dir(), reason="admitted ORCA2X record absent")
def test_restart_reader_matches_step_entry_frames_and_nemo_area():
    rows = pm.restart_reader_control(RECORD, range(1, 3))
    assert rows and all(all(r[f] for f in pm.FIELDS) for r in rows)
    geo = pm.static_geometry(RECORD)
    assert geo["area_t"].sum() / 1e6 == pytest.approx(357005237.10546613, rel=1e-14)
    assert geo["nlb10"] == 2
