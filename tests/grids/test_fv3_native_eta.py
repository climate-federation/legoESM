"""set_eta, km in {5, 10} analytic branch (fv_eta.F90:334-344).

The npz=5 case is the vertical coordinate of the 3-D duo reference run, so
these values are not arbitrary: they set delp at initialisation, and delp is
a prognostic of every acoustic unit.
"""
from __future__ import annotations

import numpy as np
import pytest

from legoesm.core.fv3_native_eta import (
    interface_pressure,
    layer_thickness,
    set_eta_analytic,
)


def test_km5_matches_the_oracle_branch_exactly():
    """fv_eta.F90:339-344 for km=5, in the ORACLE'S EVALUATION ORDER.

    ak(5) IS NOT 10000.0. bk = 0.8 is not representable, so 1-0.8 gives
    0.19999999999999996 and ptop*(1-bk) gives 9999.999999999998. The Fortran
    evaluates the identical expression in the identical f64
    (-fdefault-real-8), so the ORACLE CARRIES THAT VALUE TOO -- reproducing it
    bit-for-bit is what parity means here.

    An earlier version of this test asserted the hand-written integers
    [50000, 40000, 30000, 20000, 10000, 0] and failed. Had the CODE been
    "fixed" to produce exact integers instead, the port would have silently
    diverged from the oracle by 2e-12 Pa in ak(5) at the very foundation of
    the vertical coordinate. The test was wrong, not the code.
    """
    ak, bk, ptop, ks = set_eta_analytic(5)

    assert ptop == 50000.0
    assert ks == 0
    assert ak.shape == bk.shape == (6,)

    # the formula, evaluated exactly as fv_eta.F90:342-343 writes it
    want_bk = np.array([np.float64(k) / np.float64(5) for k in range(6)])
    want_ak = np.array([50000.0 * (1.0 - b) for b in want_bk])
    np.testing.assert_array_equal(bk, want_bk)
    np.testing.assert_array_equal(ak, want_ak)

    # and the value that is NOT the idealised integer, pinned explicitly so a
    # future "cleanup" to exact arithmetic fails loudly here
    assert ak[4] == 9999.999999999998
    assert ak[4] != 10000.0

    # everything else IS exact in binary
    np.testing.assert_array_equal(
        ak[[0, 1, 2, 3, 5]], [50000.0, 40000.0, 30000.0, 20000.0, 0.0])


def test_km5_gives_the_reference_run_pressures():
    """ps = 1e5 Pa (DCMIP16_BC sets ps = p0 everywhere, flat topography).

    pe must be 50000..100000 in 10000 Pa steps, i.e. delp = 10000 Pa in every
    layer. That is the number to compare against the oracle restart, whose
    delp is 9999.6-10000.4 -- a wave perturbation about exactly this.
    """
    ak, bk, _, _ = set_eta_analytic(5)
    pe = interface_pressure(ak, bk, 1.0e5)
    np.testing.assert_array_equal(
        pe, np.array([50000.0, 60000.0, 70000.0, 80000.0, 90000.0, 100000.0]))

    delp = layer_thickness(ak, bk, 1.0e5)
    np.testing.assert_array_equal(delp, np.full(5, 10000.0))
    # every layer equal is a real property of this coordinate at uniform ps,
    # not a coincidence: ak is linear in bk and bk is linear in k.
    assert len(set(delp.tolist())) == 1


def test_ptop_is_the_top_interface_when_ps_is_uniform():
    ak, bk, ptop, _ = set_eta_analytic(5)
    pe = interface_pressure(ak, bk, 1.0e5)
    assert pe[0] == ptop
    assert pe[-1] == 1.0e5           # bk(km+1) = 1 => pe = ps exactly


def test_km10_is_the_same_branch():
    ak, bk, ptop, ks = set_eta_analytic(10)
    assert (ptop, ks) == (50000.0, 0)
    np.testing.assert_array_equal(bk, np.arange(11) / 10.0)
    np.testing.assert_allclose(ak, 50000.0 * (1.0 - np.arange(11) / 10.0),
                               rtol=0, atol=0)


def test_ps_may_be_an_array_and_the_level_axis_goes_last():
    ak, bk, _, _ = set_eta_analytic(5)
    ps = np.full((3, 4), 1.0e5)
    pe = interface_pressure(ak, bk, ps)
    assert pe.shape == (3, 4, 6)
    np.testing.assert_array_equal(pe[0, 0], interface_pressure(ak, bk, 1.0e5))
    assert layer_thickness(ak, bk, ps).shape == (3, 4, 5)


def test_a_non_uniform_ps_still_closes_on_ps():
    """bk(km+1) = 1 exactly, so the bottom interface must equal ps for ANY ps.

    Catches an ak/bk transposition, which a uniform-ps test cannot see.
    """
    ak, bk, _, _ = set_eta_analytic(5)
    ps = np.array([[9.0e4, 1.0e5], [1.1e5, 1.05e5]])
    pe = interface_pressure(ak, bk, ps)
    np.testing.assert_array_equal(pe[..., -1], ps)
    assert np.all(layer_thickness(ak, bk, ps) > 0.0)


@pytest.mark.parametrize("km", [1, 3, 4, 24, 32, 64])
def test_other_km_are_refused_rather_than_extrapolated(km):
    """The other fv_eta cases are HAND-TABULATED. Silently extrapolating this
    analytic branch to km=24 would produce a plausible, wrong coordinate."""
    with pytest.raises(ValueError, match="analytic"):
        set_eta_analytic(km)


def test_ak_and_bk_shape_mismatch_is_refused():
    ak, bk, _, _ = set_eta_analytic(5)
    with pytest.raises(ValueError, match="must match"):
        interface_pressure(ak, bk[:-1], 1.0e5)
