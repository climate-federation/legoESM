"""Spectral (Gaussian) row of the DCMIP-2012 transport matrix cases.

``run_dcmip_transport`` used to SKIP the spectral grid.  These tests pin the
wiring: tests 1-1 / 1-3 run a few steps on T21, PASS and move the tracer;
test 1-2 runs its full flow-reversal period and reports the same l1/l2/linf
error norms as the grid-point rows.  For 1-2 the analytic IC is a
pure function of height, so its spectral projection must reproduce it to the
fp32 precision of the shared IC helpers (shows the spectral row starts from
the same analytic IC as the lat-lon row).
"""

from __future__ import annotations

import re

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from scripts.matrix.run_atmosphere_test_matrix import (  # noqa: E402
    TestCase as MatrixCase,
    run_dcmip_transport,
)

_DT = {11: 1800.0, 12: 600.0, 13: 1800.0}
_NUM = r"([-+0-9.eE]+)"


def _case(test_num):
    return MatrixCase("hydrostatic", f"dcmip_transport_{test_num}",
                      "spectral", "T21", "sigma", 12, 1,
                      {"test_num": test_num})


@pytest.mark.parametrize("test_num", [11, 13])
def test_spectral_dcmip_transport_short_run(tmp_path, test_num):
    """Two steps PASS and actually move the tracer; no reversal norms
    (the run does not reach the flow-reversal time)."""
    days = 2 * _DT[test_num] / 86400.0
    status, _, notes = run_dcmip_transport(_case(test_num), tmp_path, days)
    assert status == "PASS"
    text = (tmp_path / "results.txt").read_text()
    assert "grid: spectral" in text and "status: PASS" in text
    assert "L1=" not in notes
    rows = np.loadtxt(tmp_path / "mean_timeseries.csv", delimiter=",",
                      skiprows=1)
    assert np.all(np.isfinite(rows))
    q1_max = rows[:, 3]
    assert abs(q1_max[-1] - q1_max[0]) > 1e-6


def test_spectral_dcmip_transport_12_full_reversal(tmp_path):
    """Full 1-day flow reversal: finite, nonzero l1/l2/linf vs the analytic
    IC, and the spectral IC reproduces the analytic one (the 1-2 tracer is
    a pure function of height) to the fp32 precision of the IC helpers."""
    status, _, notes = run_dcmip_transport(_case(12), tmp_path, 1.0)
    assert status == "PASS"
    m = re.search(rf"q1 L1={_NUM} L2={_NUM} Linf={_NUM}", notes)
    assert m, notes
    norms = [float(x) for x in m.groups()]
    assert all(np.isfinite(norms)) and all(n > 0 for n in norms), notes
    ic_l2 = float(re.search(rf"IC-projection q1 L2={_NUM}", notes).group(1))
    assert ic_l2 < 1e-6, notes
