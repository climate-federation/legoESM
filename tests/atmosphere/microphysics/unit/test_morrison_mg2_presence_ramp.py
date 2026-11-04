"""RULE 5 tests of the smoothed MG2 condensate-presence test q >= qsmall
(1e-18 kg/kg) in the ported droplet-number and cloud-ice code (user decision
2026-10-22, 1a: C2 step from 1e-18 to 1e-17 kg/kg,
``MorrisonConfig.mg2_presence_ramp_top``).

Hard limit: ``mg2_presence_ramp_top = 1e-18`` (in the oracle tests' HARD_LIMIT)
is MG2's test exactly; every prognostic-number oracle test runs there.  Gap:
per fixture column, the default's departure from that CAM-matching hard
limit, max over the droplet/ice/vapour/heat tendencies, relative to the
column's largest MG2 value of each.  x64.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

jax.config.update("jax_enable_x64", True)

from legoesm import constants as C  # noqa: E402
from legoesm.atmosphere.physics.microphysics import _warm_rain as W  # noqa: E402

from tests.atmosphere.microphysics.unit.test_morrison_prognostic_nc import _cfg  # noqa: E402
from tests.atmosphere.microphysics.unit.test_mg2_warm_rain_oracle import HARD_LIMIT  # noqa: E402
from tests.atmosphere.microphysics.unit import test_morrison_prognostic_nc_act as A  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
FXD = ROOT / "tests/atmosphere/microphysics/oracle_fixtures"
QS = W.MG2_QSMALL
TOP = 1.0e-17
# measured 2026-10-22 at the default top 1e-17; re-measure on any change
GAP = {
    "mg2_prognostic_nc": [0.0] * 6,
    "mg2_prognostic_nc_act": [0.0] * 9,
    # column 10 ('clearnc': cloud water up to 5e-19 with droplets) reaches the ramp
    "mg2_prognostic_nc_sed": [0.0] * 10 + [0.9989958622472904],
    "mg2_prognostic_nc_melt": [0.0] * 2,
    "mg2_prognostic_nc_icelim": [9.272066593919881e-13] + [0.0] * 11,
    "mg2_prognostic_nc_frz": [0.0] * 6,
}


def test_defaults_and_hard_limit():
    assert _cfg()._field_defaults["mg2_presence_ramp_top"] == TOP
    assert HARD_LIMIT["mg2_presence_ramp_top"] == QS
    q = jnp.asarray([0.0, 0.5 * QS, QS, 1.5 * QS, 1e-17, 1e-3, -1e-3])
    assert np.array_equal(np.asarray(W.mg2_presence(q, QS)), np.asarray(q >= QS, float))
    s = np.asarray(W.mg2_presence(q, TOP))
    assert s[0] == s[1] == s[2] == s[6] == 0.0 and s[4] == s[5] == 1.0 and 0.0 < s[3] < 1.0


def _run(fx, top):
    fx = dict(fx)
    fx.setdefault("col_npccn", np.zeros_like(fx["col_T"]))
    hyd = {k: jnp.asarray(fx[f"col_{v}"]) for k, v in
           (("q_i", "q_i"), ("N_i", "ni_kg"), ("q_s", "q_s"), ("N_s", "ns_kg")) if f"col_{v}" in fx}
    cfg = _cfg(do_snow_to_graupel=False)._replace(mg2_presence_ramp_top=top)
    return A._run(fx, cfg=cfg, hyd_kw=hyd)


@pytest.mark.parametrize("name", sorted(GAP))
def test_default_gap_per_column_is_the_recorded_one(name):
    fx = dict(np.load(FXD / f"{name}.npz"))
    hard, soft = _run(fx, QS), _run(fx, TOP)
    rho = np.asarray(fx["col_rho"])
    pairs = [("dq_c_dt", "ora_qctend", 1.0), ("dq_v_dt", "ora_qvlat", 1.0),
             ("dT_dt", "ora_tlat", 1.0 / C.c_pd), ("dN_c_dt", "ora_nctend", None)]
    if "ora_qitend" in fx:
        pairs.append(("dq_i_dt", "ora_qitend", 1.0))
    gap = np.zeros(fx["col_T"].shape[0])
    for field, ora, k in pairs:
        d = np.abs(np.asarray(getattr(soft, field)) - np.asarray(getattr(hard, field)))
        scale = np.abs(fx[ora]) * (rho if k is None else k)
        gap = np.maximum(gap, d.max(axis=1) / np.maximum(scale.max(axis=1), 1e-300))
    print(name, "presence gap per column", [float(x) for x in gap])
    np.testing.assert_allclose(gap, GAP[name], rtol=1e-6, atol=1e-12)
    # a gap below the 1e-12 floor must not vanish (or appear) unnoticed
    assert np.array_equal(gap > 0.0, np.asarray(GAP[name]) > 0.0)


def test_presence_ramp_gradients():
    """check_grads through the ramp: the step itself and the five kernels that
    use it, on condensate between qsmall and the top (scaled to O(1))."""
    u = jnp.asarray([0.1, 0.35, 0.6, 0.85])

    def q(x):
        return QS + x * (TOP - QS)

    check_grads(lambda x: W.mg2_presence(q(x), TOP), (u,), order=2, modes=["rev"],
                eps=1e-3, atol=1e-6, rtol=1e-4)

    one = jnp.ones_like(u)

    def readjust(x):
        return W.mg2_droplet_number_readjust(q(x), 1e8 * one, 2e8 * one, one, one, 600.0,
                                             -1e4 * one, presence_top=TOP)

    def dep_limiter(x):
        return W.mg2_deposition_limiter(1e-4 * one, 240.0 * one, 2e-5 * one, q(x), 0.5 * one,
                                        1e-9 * one, 0.0 * one, 0.0 * one, 600.0, 0.01,
                                        presence_top=TOP)[0] * 1e9

    def melt(x):
        return W.mg2_instant_ice_melt_homog_freeze(
            1e-4 * one, 1e8 * one, q(x), 1e5 * one, 275.0 * one, 233.15, 2.0, 0.2,
            presence_top=TOP)[3] * 1e-5

    def fall(x):
        return W.mg2_droplet_fall_coefficients(q(x)[None], 1e8 * one[None], one[None],
                                               280.0 * one[None], one[None],
                                               presence_top=TOP)[1][0]

    def ice_dep(x):
        T, p = 250.0 * one, 5.0e4 * one
        vd, su, bg = W.mg2_ice_deposition_sublimation(
            2e-4 * one, q(x), 1e3 * one, one, T, p, p / (C.R_d * T), 1.2e-4 * one, 1e-4 * one,
            500.0, 5e-4, 0.01, 2.0, presence_top=TOP)
        return (vd + bg) * 1e20

    for f in (readjust, dep_limiter, melt, fall, ice_dep):
        check_grads(f, (u,), order=2, modes=["rev"], eps=1e-3, atol=1e-6, rtol=1e-3)
        g = jax.grad(lambda x: jnp.sum(f(x)))(u)
        assert np.all(np.isfinite(np.asarray(g))) and np.any(np.asarray(g) != 0.0), f.__name__


def test_kernels_scale_with_the_step():
    """Value checks in the ramp: readjust and ice deposition take exactly the
    presence share of their full (q >= top) effect."""
    u = jnp.asarray([0.0, 0.3, 0.7])
    q = QS + u * (TOP - QS)
    f = np.asarray(W.mg2_presence(q, TOP))
    one = jnp.ones_like(u)
    base = -1e4
    full = np.asarray(W.mg2_droplet_number_readjust(q, 1e8 * one, 2e8 * one, one, one, 600.0,
                                                    base * one, presence_top=QS))
    soft = np.asarray(W.mg2_droplet_number_readjust(q, 1e8 * one, 2e8 * one, one, one, 600.0,
                                                    base * one, presence_top=TOP))
    np.testing.assert_allclose(soft, f * full + (1 - f) * base, rtol=1e-12)
    assert soft[0] == base and not np.isclose(soft[1], full[1])
    T, p = 250.0 * one, 5.0e4 * one
    args = (2e-4 * one, q, 1e3 * one, one, T, p, p / (C.R_d * T), 1.2e-4 * one, 1e-4 * one,
            500.0, 5e-4, 0.01, 2.0)
    hard = W.mg2_ice_deposition_sublimation(*args, presence_top=QS)
    smooth = W.mg2_ice_deposition_sublimation(*args, presence_top=TOP)
    for h, s in zip(hard, smooth):
        np.testing.assert_allclose(np.asarray(s), f * np.asarray(h), rtol=1e-12, atol=0)
    assert float(np.asarray(hard[0])[1]) > 0


def test_activation_reads_the_step(monkeypatch):
    """Mid-ramp cloud water activates only the step's share of npccn*dt."""
    from legoesm.atmosphere.physics.microphysics import morrison as M
    fx = dict(np.load(FXD / "mg2_prognostic_nc_act.npz"))
    seen = []
    real = M.mg2_incloud_warm_rain_inputs

    def spy(q_c, q_r, N_c, *a, **k):
        seen.append(np.asarray(N_c))
        return real(q_c, q_r, N_c, *a, **k)

    monkeypatch.setattr(M, "mg2_incloud_warm_rain_inputs", spy)
    q_c = jnp.full_like(jnp.asarray(fx["col_q_c"]), 5.0e-18)
    npccn_kg = jnp.full_like(q_c, 1.0e6)
    A._run(fx, cfg=_cfg()._replace(mg2_presence_ramp_top=TOP), q_c=q_c, npccn_kg=npccn_kg)
    n0 = np.asarray(fx["col_nc_kg"] * fx["col_rho"])
    on = float(W.mg2_presence(jnp.asarray(5.0e-18), 1.0e-17))
    want = n0 + on * np.asarray(npccn_kg) * fx["col_rho"] * float(fx["dt"])
    np.testing.assert_allclose(seen[0], want, rtol=1e-12)
