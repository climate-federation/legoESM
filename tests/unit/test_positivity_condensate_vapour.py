"""Positivity stage: a column's net-negative condensate paid from the column's
vapour with its latent heat.

Ported from I. Di Carlo's ENSO fork (iadicarlo/legoESM-enso), isotope
passenger cases dropped.  With q_c net negative over the whole globe and q_v positive, the legacy
borrow floors q_c and creates the deficit; with ``condensate_from_vapour``
total water ``sum(area dp (q_v + q_c + ...))`` is conserved to 1e-15 relative
and the column frozen moist static energy ``sum dp (c_pd T + L_v q_v - L_f
q_frozen)`` to 1e-15 relative per column.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants  # noqa: E402
from legoesm.core.conservation import (  # noqa: E402
    apply_water_positivity, pay_condensate_deficit_from_vapour)

NCOL, NLEV = 12, 16
WATER = ("q_v", "q_c", "q_r", "q_i", "q_s", "q_g")
FROZEN = ("q_i", "q_s", "q_g")


def _state(seed=0, net_negative_qc=True):
    rng = np.random.default_rng(seed)
    dp = jnp.asarray(rng.uniform(1.0e3, 4.0e3, (NCOL, NLEV)))
    area = jnp.asarray(rng.uniform(0.5, 1.5, NCOL))
    T = jnp.asarray(rng.uniform(220.0, 300.0, (NCOL, NLEV)))
    tr = {"q_v": jnp.asarray(rng.uniform(1e-5, 2e-2, (NCOL, NLEV)))}
    for n in WATER[1:]:
        x = rng.uniform(0.0, 1e-5, (NCOL, NLEV)) * (rng.uniform(size=(NCOL, NLEV)) < 0.3)
        x = x - rng.uniform(0.0, 2e-6, (NCOL, NLEV)) * (rng.uniform(size=(NCOL, NLEV)) < 0.4)
        tr[n] = jnp.asarray(x)
    if net_negative_qc:
        # the stage T start: q_c net negative in almost every column
        tr["q_c"] = -jnp.abs(tr["q_c"]) + 1e-9 * (jnp.arange(NLEV) == 3)
    return tr, T, dp, area


def _water(tr, dp, area, suffix=""):
    return float(sum(jnp.sum(area[:, None] * dp * tr[n + suffix]) for n in WATER))


def _mse_col(tr, T, dp):
    frozen = sum(tr[n] for n in FROZEN)
    return np.asarray(jnp.sum(dp * (constants.c_pd * T + constants.L_v * tr["q_v"]
                                    - constants.L_f * frozen), axis=-1))


def test_legacy_floor_creates_water_on_a_net_negative_globe():
    tr, T, dp, area = _state()
    out, _ = apply_water_positivity(tr, T, dp, conservative=True, energy_consistent=False,
                                    area=area, condensate_from_vapour=False)
    w0, w1 = _water(tr, dp, area), _water(out, dp, area)
    print("legacy created", (w1 - w0) / w0)
    assert (w1 - w0) / w0 > 1e-6


def test_pay_from_vapour_conserves_water_and_column_energy():
    tr, T, dp, area = _state()
    out, T1 = apply_water_positivity(tr, T, dp, conservative=True, energy_consistent=False,
                                     area=area, condensate_from_vapour=True)
    w0, w1 = _water(tr, dp, area), _water(out, dp, area)
    print("water", abs(w1 - w0) / w0)
    assert abs(w1 - w0) / w0 < 1e-15
    e0, e1 = _mse_col(tr, T, dp), _mse_col(out, T1, dp)
    print("column MSE", np.max(np.abs(e1 - e0) / np.abs(e0)))
    assert np.max(np.abs(e1 - e0) / np.abs(e0)) < 1e-15
    for n in out:
        assert float(jnp.min(out[n])) >= 0.0


def test_default_pays_from_vapour_and_no_deficit_is_bitwise_legacy():
    tr, T, dp, area = _state()
    a, _ = apply_water_positivity(tr, T, dp, conservative=True, energy_consistent=False,
                                  area=area)
    w0, w1 = _water(tr, dp, area), _water(a, dp, area)
    assert abs(w1 - w0) / w0 < 1e-15
    # with no net-negative condensate column the payment does nothing
    tr, T, dp, area = _state(1, net_negative_qc=False)
    tr = {n: (jnp.maximum(x, 0.0) if n != "q_v" else x) for n, x in tr.items()}
    a, Ta = apply_water_positivity(tr, T, dp, conservative=True, energy_consistent=False,
                                   area=area)
    b, Tb = apply_water_positivity(tr, T, dp, conservative=True, energy_consistent=False,
                                   area=area, condensate_from_vapour=False)
    for n in a:
        np.testing.assert_array_equal(np.asarray(a[n]), np.asarray(b[n]))
    np.testing.assert_array_equal(np.asarray(Ta), np.asarray(Tb))


def test_pay_function_is_column_local():
    tr, T, dp, _ = _state(3)
    a, Ta = pay_condensate_deficit_from_vapour(tr, T, dp)
    b, Tb = pay_condensate_deficit_from_vapour({n: x[:5] for n, x in tr.items()},
                                               T[:5], dp[:5])
    for n in b:
        np.testing.assert_array_equal(np.asarray(a[n][:5]), np.asarray(b[n]))
    np.testing.assert_array_equal(np.asarray(Ta[:5]), np.asarray(Tb))


# --- the payment is continuous in the state (RULE 5) -----------------------

def _fork_hard_pay(tr, T, dp):
    """The ENSO fork's form: zero each net-negative condensate column and pay
    it from vapour, only where the vapour can pay all."""
    V = jnp.sum(jnp.maximum(tr["q_v"], 0.0) * dp, axis=-1, keepdims=True)
    D = {x: jnp.maximum(-jnp.sum(tr[x] * dp, axis=-1, keepdims=True), 0.0) for x in WATER[1:]}
    D_tot = sum(D.values())
    paid = (D_tot > 0.0) & (V > D_tot)
    out = dict(tr)
    for x in WATER[1:]:
        out[x] = jnp.where(paid & (D[x] > 0.0), 0.0, tr[x])
    vpos = jnp.maximum(tr["q_v"], 0.0)
    out["q_v"] = tr["q_v"] - vpos * jnp.where(paid, D_tot / V, 0.0)
    # latent heat per unit paid: L_v for liquid, L_v + L_f for frozen
    heat = sum((constants.L_v + (0.0 if x in ("q_c", "q_r") else constants.L_f)) * D[x]
               for x in WATER[1:])
    return out, T + jnp.where(paid, vpos / V * heat / constants.c_pd, 0.0)


def test_full_stage_equals_the_fork_form_where_vapour_pays():
    """After the column borrow, a paid column ends as the fork left it."""
    tr, T, dp, area = _state()
    kw = dict(conservative=True, energy_consistent=False, area=area)
    a, Ta = apply_water_positivity(tr, T, dp, **kw)
    hard, Th = _fork_hard_pay(tr, T, dp)
    b, Tb = apply_water_positivity(hard, Th, dp, condensate_from_vapour=False, **kw)
    for n in WATER:
        np.testing.assert_allclose(np.asarray(a[n]), np.asarray(b[n]), rtol=1e-12, atol=1e-20,
                                   err_msg=n)
    np.testing.assert_allclose(np.asarray(Ta), np.asarray(Tb), rtol=1e-14, atol=0)


def test_payment_is_continuous_at_a_zero_column_total():
    """The fork zeroed q_c on the sign of its column total (an O(q) jump);
    here a column total of -1e-14 vs +1e-14 moves q_c by O(1e-17) (the
    fork's form moves it by 1e-5)."""
    dp = jnp.full((1, 2), 2000.0)
    T = jnp.full((1, 2), 280.0)
    out = []
    for e in (-1.0e-14, 1.0e-14):
        tr = {"q_v": jnp.full((1, 2), 1.0e-2), "q_c": jnp.array([[1.0e-5, -1.0e-5 - e / 2000.0]])}
        out.append(np.asarray(pay_condensate_deficit_from_vapour(tr, T, dp)[0]["q_c"]))
    assert np.max(np.abs(out[0] - out[1])) < 1.0e-16


def test_full_stage_conserves_water_where_vapour_barely_pays():
    """Columns near both former switches (column total near 0, vapour near
    the deficit), every column zero or net-negative in q_c: the full stage
    creates no water."""
    r = np.array([0.0, 0.004, 0.01, 0.02, 0.05])
    qc = np.concatenate([np.stack([np.full(5, 1.0e-5), -1.0e-5 * (1.0 + r)], axis=-1),
                         [[-2.0e-6, 1.0e-7]]])
    qv = np.concatenate([np.full((5, 2), 1.0e-2), [[1.9e-6, 9.5e-9]]])
    tr = {"q_v": jnp.asarray(qv), "q_c": jnp.asarray(qc)}
    dp, T = jnp.full((6, 2), 2000.0), jnp.full((6, 2), 280.0)
    out, _ = apply_water_positivity(tr, T, dp, conservative=True, energy_consistent=False,
                                    area=jnp.ones(6))
    w0 = float(sum(jnp.sum(dp * tr[n]) for n in tr))
    w1 = float(sum(jnp.sum(dp * out[n]) for n in out))
    assert abs(w1 - w0) / w0 < 1e-15
    assert float(jnp.min(out["q_c"])) >= 0.0


def test_gradient_through_the_payment():
    """Away from the max/min kinks: finite, nonzero, correct derivatives,
    one column fully paid and one where the vapour pays only part."""
    from jax.test_util import check_grads
    dp = jnp.full((2, 2), 2000.0)
    T = jnp.full((2, 2), 280.0)

    def f(qc, qv):
        qc, qv = jnp.asarray(qc), jnp.asarray(qv)
        out, T1 = pay_condensate_deficit_from_vapour({"q_v": qv, "q_c": qc}, T, dp)
        return jnp.sum(out["q_c"] ** 2) + jnp.sum(out["q_v"] * T1)

    qc = jnp.array([[1.0e-5, -1.2e-5], [-2.0e-6, 1.0e-7]])
    qv = jnp.array([[1.0e-2, 1.0e-2], [1.0e-6, 5.0e-7]])     # row 1: V < D
    g = jax.grad(f, argnums=(0, 1))(qc, qv)
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in g)
    assert float(jnp.abs(g[0][0, 1])) > 0.0 and float(jnp.abs(g[1][1, 0])) > 0.0
    check_grads(f, (qc, qv), order=1, modes=("fwd", "rev"), eps=1e-11, atol=1e-6, rtol=1e-4)


def test_vapour_short_column_pays_what_it_holds():
    """V+ < D_tot: the column pays all its vapour (the fork paid nothing and
    left the whole deficit to the borrow); water is conserved."""
    dp = jnp.full((1, 2), 2000.0)
    T = jnp.full((1, 2), 280.0)
    tr = {"q_v": jnp.array([[1.0e-6, 5.0e-7]]), "q_c": jnp.array([[-2.0e-6, 1.0e-7]])}
    out, _ = pay_condensate_deficit_from_vapour(tr, T, dp)
    V = float(jnp.sum(tr["q_v"] * dp))
    np.testing.assert_allclose(np.asarray(out["q_v"]), 0.0, atol=1e-22)
    gained = float(jnp.sum((out["q_c"] - tr["q_c"]) * dp))
    np.testing.assert_allclose(gained, V, rtol=1e-14)
