"""omip_strip_ml_base_lc: known-answer checks for the step depth, the turbocline,
the Langmuir depth read from an increment profile built with NEMO's own formula
(zdftke.F90:358-370), and the zus inversion; each with a perturbation that must
change the answer (non-vacuity)."""
import numpy as np

from scripts.validate.ocean_fidelity import omip_strip_ml_base_lc as P


def test_step_depth_finds_the_planted_step_and_moves_with_it():
    zt = np.array([0.5, 1.5, 2.6, 3.8, 5.0, 6.4, 8.0, 10.0, 12.0, 13.6, 16.0, 20.0, 90.0])
    T = np.full(13, 28.0); S = np.full(13, 35.0)
    T[9:] -= 1.0                                     # step between T-levels 8 and 9
    z, i = P.step_depth(T, S, zt, 40.0)
    assert i == 8 and np.isclose(z, 0.5 * (12.0 + 13.6))
    T2 = np.full(13, 28.0); T2[6:] -= 1.0
    assert P.step_depth(T2, S, zt, 40.0)[1] == 5
    T3 = np.full(13, 28.0); T3[12:] -= 5.0           # strongest step below ZTOP is excluded
    T3[9:] -= 0.1
    assert P.step_depth(T3, S, zt, 40.0)[1] == 8


def test_turbocline_is_first_interface_below_threshold():
    zw = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    K = np.array([1e-2, 3e-3, 4e-4, 1e-2, 1e-5])
    assert P.turbocline(K, zw, 5e-4) == 3.0
    assert P.turbocline(np.full(5, 1e-2), zw, 5e-4) == 5.0


def test_lc_depth_and_zus_inversion_recover_nemo_formula_inputs():
    zw = np.array([1.0, 2.1, 3.3, 4.6, 6.0, 7.6, 9.4, 11.5, 13.9, 16.6, 19.6, 23.0])
    for imlc, zus in ((6, 0.05), (9, 0.11), (11, 0.02)):
        h = zw[imlc]
        w = P.RN_LC * np.sin(np.pi * zw / h)
        inc = np.where(zw < h, P.DT * zus ** 3 * w ** 3 / h, 0.0)
        hl, first, st = P.lc_depth(inc[None], zw)
        assert st[0] == 0
        assert np.isclose(hl[0], h) and first[0] == 0
        assert np.isclose(P.implied_us(inc[None], hl, first, zw)[0], zus, rtol=1e-12)
        # perturbation: a doubled source must double zus^3, i.e. move zus
        assert not np.isclose(P.implied_us(2 * inc[None], hl, first, zw)[0], zus, rtol=1e-3)
        top, spread = P.profile_us(inc[None], hl, zw)
        assert np.isclose(top[0], zus, rtol=1e-12) and np.isclose(spread[0], 1.0, rtol=1e-12)
        bad = inc * np.where(zw < h / 2, 1.0, 1.5)       # a shape defect must show as spread > 1
        assert P.profile_us(bad[None], hl, zw)[1][0] > 1.1
        assert P.gap_share(inc[None]) == 0.0
        holed = inc.copy(); holed[1] = 0.0
        assert P.gap_share(holed[None]) == 1.0
    h0, _, s0 = P.lc_depth(np.zeros((1, 12)), zw)       # cutoff above the first interface
    assert s0[0] == -1 and h0[0] == zw[0]
    hd, _, sd = P.lc_depth(np.ones((1, 12)), zw)        # positive below the last saved interface
    assert sd[0] == 1 and np.isinf(hd[0])


def test_nemo_check_constant_matches_zdftke_literals():
    assert np.isclose(P.L_CSD(), 0.5 * 0.016 ** 2 / (1.22 * 1.5e-3))
