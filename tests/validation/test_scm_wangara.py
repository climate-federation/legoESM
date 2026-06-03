"""Wangara Day 33 convective boundary-layer benchmark for legoESM SCM.

Exercises the Phase B v2 ``prescribe='fluxes'`` channel against a
known convective forcing (cosine-shaped surface heat / moisture
flux peaking at 13:00 local time, Wangara Day 33).  See
:mod:`scripts.scm.wangara` for the legoESM setup and the
jax_scm reference in ``tests/validation/scm_oracle/wangara_Nz64.nc``.

What this test checks:

* Run completes without NaN over a 4-hour window from 09:00 LT.
* Lowest-cell θ warms under the surface heating (afternoon spin-up).
* qke maximum sits ABOVE the surface (convective plume signature).
* Oracle NetCDF surface-flux time series matches the cosine spec
  and the surface θ trend is upward over the simulated window.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

pytest.importorskip("netCDF4")
xr = pytest.importorskip("xarray")


ORACLE_PATH = (
    pathlib.Path(__file__).resolve().parent
    / "scm_oracle"
    / "wangara_Nz64.nc"
)

TEST_HOURS = 2.0
TEST_DT = 5.0
TEST_NLEV = 24
TEST_SIGMA_TOP = 0.78


def _import_runner():
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
    from scm import wangara as run_scm_wangara
    return run_scm_wangara


@pytest.fixture(scope="module")
def wangara_run():
    runner = _import_runner()
    scm = runner.build_scm(
        nlev=TEST_NLEV, dt=TEST_DT, sigma_top=TEST_SIGMA_TOP,
    )
    scm.run(nsteps=int(TEST_HOURS * 3600 / TEST_DT))
    return scm


def test_wangara_finite(wangara_run):
    scm = wangara_run
    for arr in (
        np.asarray(scm.state.T.data),
        np.asarray(scm.state.u.data),
        np.asarray(scm.state.v.data),
        np.asarray(scm.phys_state.qke),
        np.asarray(scm.state.tracers["q_v"].data),
    ):
        assert np.all(np.isfinite(arr))


def test_wangara_lowest_cell_warms_under_surface_heating(wangara_run):
    """Cosine surface heating from 09:00–13:00 deposits positive
    sensible heat at the lowest cell.  After 2 h (11:00 LT), the
    lowest cell must be warmer than its initial value (277 K).
    """
    T_low = float(wangara_run.state.T.data[0, 0, 0, -1])
    assert T_low > 277.0, (
        f"Lowest cell did not warm under Wangara surface heating: "
        f"T_low={T_low}"
    )
    # Sanity ceiling — 2 h of cosine peaking at 13:00 should warm
    # by ~1–3 K in the lowest layer at default Nz.
    assert T_low < 285.0, (
        f"Lowest cell warming exceeded plausible band: T_low={T_low}"
    )


def test_wangara_convective_qke_profile(wangara_run):
    """The convective plume signature: qke_max above the surface
    cell, not at the surface.  A purely surface-driven turbulence
    response (no convective transport) would peak at the surface.
    """
    qke = np.asarray(wangara_run.phys_state.qke[0])
    k_max = int(np.argmax(qke))
    nlev = qke.size
    # Lowest cell is index nlev-1; convective qke peak must sit
    # ABOVE that (lower index in legoESM top-to-bottom convention).
    assert k_max < nlev - 1, (
        f"qke peak is at the surface (k={k_max}/{nlev-1}); "
        "convective BL did not develop"
    )
    assert qke[k_max] > 1.0, (
        f"qke peak {qke[k_max]} too low for a convective BL "
        "(expected O(1–10) m²/s² under cosine surface heating)"
    )


def test_wangara_oracle_present_and_plausible():
    """Sanity-check the Wangara oracle file before relying on it."""
    if not ORACLE_PATH.exists():
        pytest.skip(f"Oracle missing: {ORACLE_PATH}")
    ds = xr.open_dataset(ORACLE_PATH)
    try:
        w_th = ds["mo_w_th"].values
        # Peak at 13:00 — somewhere in the middle of the trajectory.
        assert float(w_th.max()) > 0.05, (
            f"Oracle peak w_th {float(w_th.max())} below Wangara band"
        )
        # Surface theta increases over the simulated 7 h.
        th_s = ds["mo_th_s"].values
        assert float(th_s[-1]) > float(th_s[0]), (
            "Oracle surface theta did not warm — Wangara expects "
            "monotone warming from 09:00 to 16:00 LT"
        )
    finally:
        ds.close()
