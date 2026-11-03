"""Water positivity right after ZM (core.conservation.after_convection_positivity,
used by the combined macmic loop where CAM6 runs qneg3).  fp64."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants  # noqa: E402
from legoesm.core.conservation import after_convection_positivity  # noqa: E402

NCOL, NLEV = 4, 32


def _cols(seed=0):
    rng = np.random.default_rng(seed)
    dp = jnp.asarray(np.tile(np.linspace(500.0, 3500.0, NLEV), (NCOL, 1)))
    qv = jnp.asarray(np.tile(np.geomspace(1e-6, 1.6e-2, NLEV), (NCOL, 1)) * rng.uniform(0.9, 1.1, (NCOL, NLEV)))
    qc = jnp.asarray(rng.uniform(0.0, 1e-4, (NCOL, NLEV)))
    T = jnp.asarray(np.tile(np.linspace(210.0, 300.0, NLEV), (NCOL, 1)))
    return dp, qv, qc, T


def _col(x, dp):
    return jnp.sum(x * dp, axis=-1) / constants.g


def test_vapour_hole_is_filled_and_water_closes():
    dp, qv, qc, T = _cols()
    qv = qv.at[1, 6].set(-4.05e-6)
    tr = {"q_v": qv, "q_c": qc}
    out, T2, d = after_convection_positivity(tr, T, dp)
    assert bool(jnp.all(out["q_v"] >= 0.0))
    for k in tr:
        rel = jnp.abs(_col(out[k], dp) - _col(tr[k], dp)) / _col(jnp.abs(tr[k]), dp)
        assert float(jnp.max(rel)) < 1e-14, k
    for k in tr:
        np.testing.assert_array_equal(np.asarray(out[k])[[0, 2, 3]], np.asarray(tr[k])[[0, 2, 3]])
    np.testing.assert_array_equal(np.asarray(T2), np.asarray(T))
    assert [int(x) for x in d["water_cells"]] == [0, 1, 0, 0]
    np.testing.assert_allclose(float(d["water_deficit_kg_m2"][1]), 4.05e-6 * float(dp[1, 6]) / constants.g, rtol=1e-12)


def test_net_negative_condensate_paid_from_vapour_conserves_water_and_mse():
    dp, qv, qc, T = _cols(3)
    qc = qc.at[2].set(0.0).at[2, 20].set(-2.0e-5)
    tr = {"q_v": qv, "q_c": qc}
    out, T2, _ = after_convection_positivity(tr, T, dp)
    assert bool(jnp.all(out["q_c"] >= 0.0))
    tot_in = _col(tr["q_v"] + tr["q_c"], dp)
    tot_out = _col(out["q_v"] + out["q_c"], dp)
    np.testing.assert_allclose(np.asarray(tot_out), np.asarray(tot_in), rtol=1e-14)
    mse_in = _col(constants.c_pd * T + constants.L_v * tr["q_v"], dp)
    mse_out = _col(constants.c_pd * T2 + constants.L_v * out["q_v"], dp)
    np.testing.assert_allclose(np.asarray(mse_out), np.asarray(mse_in), rtol=1e-14)


def test_nothing_negative_is_bitwise():
    dp, qv, qc, T = _cols(1)
    tr = {"q_v": qv, "q_c": qc}
    out, T2, d = after_convection_positivity(tr, T, dp)
    for k in tr:
        np.testing.assert_array_equal(np.asarray(out[k]), np.asarray(tr[k]))
    np.testing.assert_array_equal(np.asarray(T2), np.asarray(T))
    assert int(jnp.sum(d["water_cells"])) == 0 and float(jnp.sum(d["water_deficit_kg_m2"])) == 0.0


def test_number_deficit_is_not_water_mass():
    dp, qv, qc, T = _cols(5)
    nc = jnp.asarray(np.full((NCOL, NLEV), 5.0e7))
    qv = qv.at[1, 6].set(-4.05e-6)
    nc = nc.at[1, 9].set(-3.0e6).at[3, 4].set(-1.0e6)
    out, _, d = after_convection_positivity({"q_v": qv, "q_c": qc, "N_c": nc}, T, dp)
    assert [int(x) for x in d["water_cells"]] == [0, 1, 0, 0]
    assert [int(x) for x in d["number_cells"]] == [0, 1, 0, 1]
    np.testing.assert_allclose(float(d["number_deficit_per_m2"][3]), 1.0e6 * float(dp[3, 4]) / constants.g, rtol=1e-12)
    assert bool(jnp.all(out["N_c"] >= 0.0))
    np.testing.assert_allclose(np.asarray(_col(out["N_c"], dp)), np.asarray(_col(nc, dp)), rtol=1e-13)


def test_gradient_finite_through_a_hole():
    dp, qv, qc, T = _cols(7)
    qv = qv.at[0, 5].set(-1.0e-6)

    def f(x):
        out, T2, _ = after_convection_positivity({"q_v": qv.at[0, 5].set(x), "q_c": qc}, T, dp)
        return jnp.sum(out["q_v"] ** 2) + jnp.sum(T2)

    g = jax.grad(f)(-1.0e-6)
    assert bool(jnp.isfinite(g)) and float(g) != 0.0
