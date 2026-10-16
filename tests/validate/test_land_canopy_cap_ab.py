"""Direct test of the canopy-cap A/B scorer's pure pieces."""
import numpy as np

from scripts.validate.land_canopy_cap_ab import cellwise_stats, parse_log


def test_cellwise_stats_reports_p99_max_and_signed_bias_over_the_mask():
    a = np.zeros(200)
    b = np.zeros(200)
    b[:100] = 1.0          # land cells: +1 everywhere ...
    b[0] = 5.0             # ... one outlier
    b[100:] = -50.0        # ocean cells must be ignored
    mask = np.arange(200) < 100
    p99, mx, rms, bias = cellwise_stats(a, b, mask)
    assert mx == 5.0
    assert abs(rms - np.sqrt((99 + 25) / 100)) < 1e-12
    assert 1.0 <= p99 <= 5.0
    assert abs(bias - 1.04) < 1e-12
    # NaN cells drop out instead of poisoning the statistic
    b[1] = np.nan
    assert np.isfinite(cellwise_stats(a, b, mask)[0])
    assert all(np.isnan(v) for v in cellwise_stats(a, b, np.zeros(200, bool)))


def test_parse_log_reads_wall_and_held_windows():
    text = ("land: 23 column-steps held in the last 432 steps (23 of them on LAND "
            "columns — the physically meaningful share; the rest are ocean columns)\n"
            "MPAS run COMPLETED in 14795.8s\nMPAS run COMPLETED in 14806.8s\n"
            "land: 1347 column-steps held in the last 432 steps (1158 of them on "
            "LAND columns — x)\n")
    wall, held = parse_log(text)
    assert wall == 14806.8
    assert held == [(23, 432, 23), (1347, 432, 1158)]
    assert parse_log("nothing here") == (None, [])
