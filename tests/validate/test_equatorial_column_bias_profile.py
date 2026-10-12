"""Direct test of the equatorial column-bias probe's integration kernels (known answers)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "validate" / "ocean_fidelity"))
import equatorial_column_bias_profile as P  # noqa: E402


def test_overlap_clips_layers_at_depth():
    h = np.array([[10.0, 10.0, 20.0, 50.0]])
    np.testing.assert_allclose(P._overlap(h, 25.0), [[10.0, 10.0, 5.0, 0.0]])
    np.testing.assert_allclose(P._overlap(h, 90.0), [[10.0, 10.0, 20.0, 50.0]])


def test_depth_mean_known_answer_and_weighting():
    # two columns, layers 10/10/30 m; column 0 is 20 C over 0-20 m and 10 C below, column 1 uniform 10 C
    h = np.tile([10.0, 10.0, 30.0], (2, 1))
    T = np.array([[20.0, 20.0, 10.0], [10.0, 10.0, 10.0]])
    wet = np.ones_like(T, dtype=bool)
    s = P._summaries(T, h, wet, np.array([1.0, 1.0]), [50.0])
    # col 0: (20*20 + 10*30)/50 = 14 ; col 1: 10 ; unweighted mean 12
    assert abs(s["mean0_50"] - 12.0) < 1e-12
    s3 = P._summaries(T, h, wet, np.array([3.0, 1.0]), [50.0])
    assert abs(s3["mean0_50"] - (3 * 14.0 + 10.0) / 4.0) < 1e-12
    np.testing.assert_allclose(s["profile"], [15.0, 15.0, 10.0])


def test_shallow_column_excluded_from_depth_mean():
    # planted violation: a 20-m-deep column must NOT enter the 0-50 m mean (it would bias it warm)
    h = np.tile([10.0, 10.0, 30.0], (2, 1))
    T = np.array([[30.0, 30.0, 0.0], [10.0, 10.0, 10.0]])
    wet = np.array([[True, True, False], [True, True, True]])
    s = P._summaries(T, h, wet, np.ones(2), [50.0])
    assert s["ncol0_50"] == 1
    assert abs(s["mean0_50"] - 10.0) < 1e-12
