"""Per-column process-ledger helpers (the MPAS attribution port, #1311).

The FV ledger reduces every row to a GLOBAL MEAN.  That is unusable for the
question these helpers exist to answer -- "which process sustains THIS
column" -- because the 2026-07-31 detonations were one cell in 10242, i.e.
~1e-4 of the global mean.  These tests pin the per-column forms, the
area-weighted reduction, and the closure identity the ledger rests on.

Run with JAX_ENABLE_X64=1: these are conservation/closure assertions.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from legoesm import constants  # noqa: E402
from legoesm.diagnostics.process_ledger import (  # noqa: E402
    N_LEDGER,
    column_store_snapshot,
    column_store_snapshot_column,
    ledger_entry,
    ledger_entry_column,
    reduce_ledger_global,
    zero_ledger_column,
)

NCOL, NLEV = 12, 8


def _grid(seed=0):
    rng = np.random.default_rng(seed)
    p_s = jnp.asarray(1.0e5 + 3.0e3 * rng.standard_normal(NCOL))
    dsigma = jnp.asarray(np.full(NLEV, 1.0 / NLEV))
    return p_s, dsigma, rng


def test_column_form_reduces_to_the_global_form():
    """The per-column entry must be the same quantity the FV row means over."""
    p_s, dsigma, rng = _grid(1)
    dq = jnp.asarray(rng.standard_normal((NCOL, NLEV)) * 1e-8)
    dT = jnp.asarray(rng.standard_normal((NCOL, NLEV)) * 1e-5)

    per = ledger_entry_column(dq, dT, p_s, dsigma)
    assert per.shape == (NCOL, 2)

    glob = ledger_entry(dq, dT, p_s, dsigma)
    # Unweighted mean of the per-column form == the global form.
    np.testing.assert_allclose(np.mean(np.asarray(per), axis=0),
                               np.asarray(glob), rtol=1e-12, atol=0)


def test_none_tendency_zeroes_only_its_own_component():
    """radiation passes dq=None (water row 0) and must not zero energy."""
    p_s, dsigma, rng = _grid(2)
    dT = jnp.asarray(rng.standard_normal((NCOL, NLEV)) * 1e-5)
    per = ledger_entry_column(None, dT, p_s, dsigma)
    assert per.shape == (NCOL, 2)
    np.testing.assert_array_equal(np.asarray(per)[:, 0], np.zeros(NCOL))
    assert np.abs(np.asarray(per)[:, 1]).max() > 0.0

    dq = jnp.asarray(rng.standard_normal((NCOL, NLEV)) * 1e-8)
    per2 = ledger_entry_column(dq, None, p_s, dsigma)
    np.testing.assert_array_equal(np.asarray(per2)[:, 1], np.zeros(NCOL))
    assert np.abs(np.asarray(per2)[:, 0]).max() > 0.0


def test_energy_row_is_c_pd_times_the_column_temperature_integral():
    """Units/definition pin: energy = c_pd * int dT dp/g [W/m2]."""
    p_s, dsigma, _ = _grid(3)
    dT = jnp.full((NCOL, NLEV), 1.0e-5)          # uniform 1e-5 K/s
    per = ledger_entry_column(None, dT, p_s, dsigma)
    expect = constants.c_pd * 1.0e-5 * np.asarray(p_s) / constants.g
    np.testing.assert_allclose(np.asarray(per)[:, 1], expect, rtol=1e-10)


def test_snapshot_difference_equals_a_uniform_tendency_row():
    """CLOSURE, the identity the dynamics/clips rows depend on:
    (snapshot_after - snapshot_before)/dt of a state advanced by a known
    tendency must equal that tendency's ledger row, per column."""
    p_s, dsigma, rng = _grid(4)
    T0 = jnp.asarray(250.0 + 20.0 * rng.standard_normal((NCOL, NLEV)))
    q0 = jnp.asarray(np.abs(rng.standard_normal((NCOL, NLEV))) * 1e-3)
    dT = jnp.asarray(rng.standard_normal((NCOL, NLEV)) * 1e-4)
    dq = jnp.asarray(rng.standard_normal((NCOL, NLEV)) * 1e-7)
    dt = 75.0

    before = column_store_snapshot_column(p_s, dsigma, T0, q0)
    after = column_store_snapshot_column(p_s, dsigma, T0 + dt * dT,
                                         q0 + dt * dq)
    rate = (after - before) / dt
    row = ledger_entry_column(dq, dT, p_s, dsigma)
    np.testing.assert_allclose(np.asarray(rate), np.asarray(row),
                               rtol=1e-9, atol=1e-18)


def test_area_weighted_reduction_differs_from_unweighted_and_is_correct():
    """A weighted reduction must actually weight -- and the test must fail if
    the weights were dropped (non-vacuous by construction: the areas here
    span the 1.471 ratio measured on the production SCVT mesh)."""
    rng = np.random.default_rng(5)
    led = jnp.asarray(rng.standard_normal((NCOL, N_LEDGER, 2)))
    area = jnp.asarray(np.linspace(1.0, 1.471, NCOL))

    got = reduce_ledger_global(led, area)
    assert got.shape == (N_LEDGER, 2)
    want = (np.sum(np.asarray(led) * np.asarray(area)[:, None, None], axis=0)
            / np.asarray(area).sum())
    np.testing.assert_allclose(np.asarray(got), want, rtol=1e-12)

    unweighted = reduce_ledger_global(led, None)
    # The two MUST differ, else this test would pass with the weights ignored.
    assert not np.allclose(np.asarray(got), np.asarray(unweighted),
                           rtol=1e-6), (
        "weighted and unweighted reductions agree -- the weighting is not "
        "being exercised, so this assertion proves nothing")


def test_equal_area_weights_match_the_unweighted_mean():
    rng = np.random.default_rng(6)
    led = jnp.asarray(rng.standard_normal((NCOL, N_LEDGER, 2)))
    area = jnp.ones(NCOL)
    np.testing.assert_allclose(
        np.asarray(reduce_ledger_global(led, area)),
        np.asarray(reduce_ledger_global(led, None)), rtol=1e-12)


def test_global_snapshot_helper_still_matches_its_column_form():
    """The pre-existing FV helper must remain consistent with the new one."""
    p_s, dsigma, rng = _grid(7)
    T0 = jnp.asarray(250.0 + 20.0 * rng.standard_normal((NCOL, NLEV)))
    q0 = jnp.asarray(np.abs(rng.standard_normal((NCOL, NLEV))) * 1e-3)
    per = column_store_snapshot_column(p_s, dsigma, T0, q0)
    glob = column_store_snapshot(p_s, dsigma, T0, q0)
    np.testing.assert_allclose(np.mean(np.asarray(per), axis=0),
                               np.asarray(glob), rtol=1e-12)


def test_zero_ledger_column_shape_and_dtype():
    z = zero_ledger_column(NCOL, dtype=jnp.float64)
    assert z.shape == (NCOL, N_LEDGER, 2)
    assert np.asarray(z).sum() == 0.0


def test_hybrid_dp_changes_the_answer_versus_pure_sigma():
    """#1400: on a hybrid column the p_s*dsigma form is wrong.  Passing dp
    must therefore change the row -- if it did not, dp would be ignored."""
    p_s, dsigma, rng = _grid(8)
    dT = jnp.asarray(rng.standard_normal((NCOL, NLEV)) * 1e-5)
    sigma_row = ledger_entry_column(None, dT, p_s, dsigma)
    dp = jnp.asarray(np.asarray(p_s)[:, None] * np.asarray(dsigma)[None, :]
                     * 1.3)          # a deliberately different layer mass
    hybrid_row = ledger_entry_column(None, dT, p_s, dsigma, dp=dp)
    assert not np.allclose(np.asarray(sigma_row), np.asarray(hybrid_row)), (
        "dp= was ignored -- the hybrid path is not wired")
    np.testing.assert_allclose(np.asarray(hybrid_row)[:, 1],
                               np.asarray(sigma_row)[:, 1] * 1.3, rtol=1e-10)


def test_grads_flow_through_the_column_entry():
    """The ledger rides inside a differentiable step; it must not break AD."""
    p_s, dsigma, rng = _grid(9)
    dT = jnp.asarray(rng.standard_normal((NCOL, NLEV)) * 1e-5)

    def loss(x):
        return jnp.sum(ledger_entry_column(None, x, p_s, dsigma)[:, 1])

    g = jax.grad(loss)(dT)
    assert np.isfinite(np.asarray(g)).all()
    assert np.abs(np.asarray(g)).max() > 0.0
