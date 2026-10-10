"""SMT-RUNGS round 7: the MLE branch replay is signed right and the plant fires."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]
                       / "scripts/validate/ocean_fidelity/testcases"))
import nemo_testcase_l1_vortex_smtrungs_round7_smt7_survey as S  # noqa: E402

G500 = np.arange(0.0, 5500.0, 500.0)          # gdepw_1d of the seamount deck


def test_500m_top_layer_gives_nla10_zero():
    assert S.nla10_nlb10(G500, 500.0)[:2] == (0, 1)


def test_thin_top_layer_gives_nla10_one():
    assert S.nla10_nlb10(G500, 10.0)[:2] == (1, 2)


def _column(nk=5, nla10=0, drho=0.47):
    rho = np.array([1025.0 + drho * k for k in range(nk)])[:, None, None] * np.ones((1, 1, 3))
    tm = np.ones_like(rho)
    e3 = 500.0 * np.ones_like(rho)
    r3 = np.zeros((1, 3))
    mb = np.full((1, 3), nk)
    return rho, tm, e3, r3, mb


def test_nla10_zero_makes_whole_column_the_mixed_layer():
    rho, tm, e3, r3, mb = _column()
    _, ikmax, zmld, _ = S.mle_mixed_layer(rho, tm, e3, r3, mb, 0, 1)
    assert ikmax == 4 and np.allclose(zmld, 2000.0)      # jpkm1 = 4 levels


def test_nla10_one_makes_first_level_the_mixed_layer_and_psi_exactly_zero():
    rho, tm, e3, r3, mb = _column()
    inml, ikmax, zmld, zbm = S.mle_mixed_layer(rho, tm, e3, r3, mb, 1, 2)
    assert np.allclose(zmld, 500.0)
    rho[:, :, 2] += 1.0                                    # a horizontal buoyancy step
    _, _, zmld, zbm = S.mle_mixed_layer(rho, tm, e3, r3, mb, 1, 2)
    um = np.ones_like(tm)
    _, psi = S.mle_psi_u(zmld, zbm, G500[:6], r3, 3e4 * np.ones((1, 3)),
                         3e4 * np.ones((1, 3)), um, ikmax)
    assert np.abs(psi).max() == 0.0


def test_front_gives_restratifying_signed_increments_that_sum_to_zero():
    # east column (i=1) lighter than west (i=0): NEMO adds psi[jk]-psi[jk+1];
    # psi[0] = 0 and psi >= 0 for psim > 0, so the TOP increment is negative
    # (light water over dense: westward at the top), and the column sums to 0.
    rho, tm, e3, r3, mb = _column()
    rho[:, :, 1] -= 1.0                          # column i=1 lighter
    _, ikmax, zmld, zbm = S.mle_mixed_layer(rho, tm, e3, r3, mb, 0, 1)
    assert zbm[0, 1] > zbm[0, 0]
    _, psi = S.mle_psi_u(zmld, zbm, G500[:6], r3, 3e4 * np.ones((1, 3)),
                         3e4 * np.ones((1, 3)), np.ones_like(tm), ikmax)
    inc = psi[:-1, 0, 0] - psi[1:, 0, 0]
    assert psi[0, 0, 0] == 0.0 and np.abs(psi).max() > 1.0
    assert inc[0] < 0.0 and inc[-1] > 0.0 and abs(inc.sum() + psi[-1, 0, 0]) < 1e-6


def test_legoesm_mixed_layer_is_the_first_level_when_nla10_is_zero():
    pytest.importorskip("jax")
    rho, tm, e3, _, _ = _column(nk=6)
    z, k = S.legoesm_zmld(rho, tm, e3, G500[:7], G500[:6] + 250.0)
    assert np.allclose(z, 500.0) and np.allclose(k, 1.0)


@pytest.mark.skipif(not S.RECORD.exists(), reason="NEMO SMT-6 record absent")
def test_main_exit_codes_unplanted_zero_planted_three(tmp_path):
    assert S.main(["--out", str(tmp_path / "a.json")]) == 0
    assert S.main(["--out", str(tmp_path / "b.json"), "--plant-min-e3w", "10"]) == 3
