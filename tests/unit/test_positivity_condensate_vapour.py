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
