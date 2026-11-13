"""KNOWN DIFFERENCE of the default ``variant="smooth"`` Kain-Fritsch vs the WRF
KF-Eta oracle fixture (``oracle_fixtures/kain_fritsch.npz``).

The smooth variant is a surrogate, not a smoothed WRF: its closure, downdraft
and shallow branch are different algorithms, so no width recovers WRF.  The
reference is reachable only as ``variant="wrf_kfeta"`` (oracle-gated at 1e-10
in ``test_kain_fritsch_oracle.py``).  This file pins, per fixture column, how
far the default sits from WRF, so a change to the surrogate re-measures them.

Measured (fp64): on the default-trigger arms the surrogate's heating and
moistening miss WRF's by 70-128 % of each firing column's peak, and it rains
nothing (vapour + cloud water is conserved column by column) where WRF rains
4e-5 to 8e-3 kg/m^2/s.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from jax.test_util import check_grads  # noqa: E402

from legoesm import constants  # noqa: E402
from legoesm.atmosphere.physics.convection.config import KainFritschConfig  # noqa: E402
from legoesm.atmosphere.physics.convection.kain_fritsch import (  # noqa: E402
    kain_fritsch_convection,
)
from tests.atmosphere.test_kain_fritsch_oracle import _FIX, _wrapper_inputs  # noqa: E402

# max|smooth - WRF| / max|WRF| per column (absolute where WRF is 0)
_GAP = {
    "L_nat_w05": {
        "dT": [9.9889434580141456e-01, 7.9518113591832273e-01, 9.9882686521581987e-01,
               4.3601105689974260e-05, 8.0843700253897726e-07, 8.9417547518893170e-01,
               3.3645103688818416e-07, 8.7403834464456854e-11, 6.9952659424538354e-05,
               9.8550920266821684e-01, 1.0035320033423523e+00, 1.0036070045593724e+00,
               1.0000000000000000e+00, 9.8795812783287607e-01, 9.9052722731008891e-01],
        "dq": [1.0416619547501211e+00, 1.2778410764238652e+00, 9.9575765726038112e-01,
               1.2613245395065995e-07, 2.1824407848137580e-10, 8.6751868230309415e-01,
               1.1233574545146725e-11, 7.0655496120378106e-15, 2.1193677358060895e-07,
               8.7530737753713717e-01, 1.0168496018904900e+00, 9.5763697242719781e-01,
               1.0000000000000000e+00, 9.0890704103131559e-01, 9.5682802789167853e-01],
    },
    "L_nat_w0": {
        "dT": [9.9893907602120957e-01, 7.0483572805692429e-01, 9.9676428263220551e-01,
               5.1528171410329736e-06, 1.0939487737003497e-04, 2.8724067415412436e-05,
               3.7587383646578626e-10, 5.8406840056457429e-08, 9.2883949630075694e-06,
               1.9203043052874166e-04, 3.5621315593336970e-04, 4.5893154293135015e-04,
               1.0000000000000000e+00, 1.0230968872892938e+00, 9.9158394060835531e-01],
        "dq": [9.9571075375897844e-01, 1.1852086475611039e+00, 9.3221122181515959e-01,
               1.5262780841171521e-08, 2.9274711578000460e-08, 1.2339270207526465e-07,
               1.2549913377029557e-14, 3.9030303536579296e-12, 2.7755339941268713e-08,
               4.0949260654799487e-07, 6.9600123727825956e-07, 5.8096519273693695e-07,
               1.0000000000000000e+00, 9.7026655411169904e-01, 9.6198495652569305e-01],
    },
}


@pytest.fixture(scope="module")
def fix():
    return dict(np.load(_FIX))


def _col_gap(ours, ref):
    ours, ref = np.asarray(ours, float), np.asarray(ref, float)
    d = np.max(np.abs(ours - ref), axis=1)
    s = np.max(np.abs(ref), axis=1)
    return np.where(s > 0, d / np.where(s > 0, s, 1.0), d)


@pytest.mark.parametrize("arm", sorted(_GAP))
def test_smooth_default_gap_per_column(fix, arm):
    assert KainFritschConfig().variant == "smooth"
    _, x = _wrapper_inputs(fix, arm)
    out, _ = kain_fritsch_convection(**x, config=KainFritschConfig())
    td = lambda a: np.asarray(a)[:, ::-1]  # noqa: E731
    np.testing.assert_allclose(_col_gap(out.dT_dt, td(fix[f"{arm}_dtdt"])),
                               _GAP[arm]["dT"], rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(_col_gap(out.dq_v_dt, td(fix[f"{arm}_dqdt"])),
                               _GAP[arm]["dq"], rtol=1e-6, atol=1e-12)
    # no precipitation: vapour + cloud water is conserved in every column,
    # while WRF rains on its firing columns
    dp = fix["p_half"][:, 1:] - fix["p_half"][:, :-1]
    rain = -np.sum((np.asarray(out.dq_v_dt) + np.asarray(out.dq_c_conv_dt)) * dp,
                   axis=1) / constants.g
    assert out.dq_r_conv_dt is None
    assert np.max(np.abs(rain)) <= 1e-15
    assert np.max(fix[f"{arm}_pratec"]) > 1e-4


def test_smooth_default_gradient(fix):
    _, x = _wrapper_inputs(fix, "L_nat_w05")
    x = {k: v for k, v in x.items() if k not in ("u", "v")}
    cols = np.array([0, 1, 2])

    kw = {k: (v[cols] if k != "dt" else v) for k, v in x.items()}

    def f(T):
        out, _ = kain_fritsch_convection(**{**kw, "T": T}, config=KainFritschConfig())
        return 1e4 * (out.dT_dt ** 2).sum()

    check_grads(f, (kw["T"],), order=1, modes=("rev",), atol=1e-6, rtol=1e-4)
    g = jax.grad(f)(kw["T"])
    assert np.all(np.isfinite(np.asarray(g))) and float(np.max(np.abs(g))) > 0.0
