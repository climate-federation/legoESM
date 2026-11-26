"""Spectral (T21) row of the Williamson case 1 cosine bell in the matrix.

The lane used to apply the order-8 exponential filter EVERY step on a pure
transport case, which cut the 1000 m peak to 818 m (alpha = pi/4) / 805 m
(alpha = 0) after 1 day.  With the filter off
(``spectral_filter_every_step=False``) the 1-day peaks are 888.8 / 872.2 m
(the IC projected on the 2/3 band peaks at 889.6 m at t = 0), and the L2
error against the analytically advected bell drops 0.177 / 0.1765 -> 0.142
for both angles.  Measured fp64.
"""

from __future__ import annotations

import re

import jax
import numpy as np
import pytest

from scripts.matrix.run_atmosphere_test_matrix import (
    TestCase as MatrixCase,
    run_cosine_bell,
)


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


@pytest.mark.parametrize("case,kw", [("cosine_bell", {}),
                                     ("cosine_bell_a0", {"alpha": 0.0})])
def test_spectral_cosine_bell_one_day_peak(tmp_path, case, kw):
    tc = MatrixCase("shallow_water", case, "spectral", "T21", "none", 12, 1,
                    dict(kw))
    status, _, notes = run_cosine_bell(tc, tmp_path, 1.0)
    assert status == "PASS"
    with open(tmp_path / "mean_timeseries.csv") as f:
        header = f.readline().strip().split(",")
    rows = np.loadtxt(tmp_path / "mean_timeseries.csv", delimiter=",",
                      skiprows=1)
    assert rows[-1, header.index("time_days")] == 1.0
    h_max = rows[-1, header.index("max_height")]
    # 1 day: filtered 818.5 / 804.8 m, unfiltered 888.8 / 872.2 m
    # (alpha = pi/4 / 0).
    assert h_max > 860.0, h_max
    # Error vs the analytically ADVECTED bell (catches stalled transport):
    # filtered 0.177 / 0.1765, unfiltered 0.142 / 0.142.
    l2 = float(re.search(r"L2=([-+0-9.eE]+)", notes).group(1))
    assert l2 < 0.16, l2
