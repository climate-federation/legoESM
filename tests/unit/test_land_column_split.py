"""Multi-process land split (FV3 duo lane): each process solves a share of the
land columns and one gather rebuilds the full land output on every process.

Simulates P processes in one: pass 1 records each process's gather payload,
pass 2 hands every process the stacked payloads, as the collective would.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.driver.model_driver import gather_land_step, split_land_columns

NCOL = 23


class _State(NamedTuple):
    T_soil: jax.Array      # (ncol, nlev) per column
    wet: jax.Array         # (ncol,) bool per column


def _step(state, cols):
    """Column-independent stand-in for the land step on columns ``cols``:
    the same tuple layout (state, 8 per-cell fields, 5 counts, iter max)."""
    T = state.T_soil.at[cols].set(state.T_soil[cols] * 1.01 + 0.5)
    wet = state.wet.at[cols].set(T[cols, 0] > 290.0)
    f = jnp.zeros(NCOL).at[cols].set(T[cols, 0] - 273.0)
    held = jnp.zeros(NCOL).at[cols].set((T[cols, 1] > 280.0).astype(float))
    n = jnp.sum(T[cols, 0] > 290.0).astype(jnp.int32)
    return ((_State(T, wet),) + (f,) * 7 + (held,)
            + (n, n, n, 2 * n, n) + (jnp.max(cols).astype(jnp.int32),))


def _run_procs(state, land, n_proc):
    outs, payloads = [], {}
    split = [split_land_columns(land, n_proc, p) for p in range(n_proc)]
    for p in range(n_proc):           # pass 1: record payloads
        def rec(tree, p=p):
            payloads[p] = tree
            return jax.tree_util.tree_map(
                lambda x: np.stack([np.asarray(x)] * n_proc), tree)
        def rec_flag_then_payload(tree, p=p):
            if not isinstance(tree, tuple):
                return np.zeros(n_proc)
            return rec(tree)
        gather_land_step(lambda p=p: _step(state, split[p][0]), NCOL,
                         split[p][1], p, rec_flag_then_payload)
    for p in range(n_proc):           # pass 2: the real collective result
        def ag(tree):
            if not isinstance(tree, tuple):
                return np.zeros(n_proc)
            return jax.tree_util.tree_map(
                lambda *xs: np.stack([np.asarray(x) for x in xs]),
                *[payloads[q] for q in range(n_proc)])
        outs.append(gather_land_step(
            lambda p=p: _step(state, split[p][0]), NCOL, split[p][1], p, ag))
    return outs


@pytest.mark.parametrize("n_proc", [1, 3, 6])
def test_every_process_gets_the_single_process_result(n_proc):
    rng = np.random.default_rng(0)
    state = _State(jnp.asarray(rng.uniform(270, 300, (NCOL, 3))),
                   jnp.zeros(NCOL, bool))
    # 11 columns: uneven shares at 3 and 6 processes, so padding is exercised.
    land = np.array([0, 2, 3, 5, 8, 9, 10, 13, 17, 19, 22])
    ref = _step(state, jnp.asarray(land))
    for got in _run_procs(state, land, n_proc):
        for a, b in zip(jax.tree_util.tree_leaves(ref),
                        jax.tree_util.tree_leaves(got)):
            assert np.asarray(a).dtype == np.asarray(b).dtype
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_split_covers_every_column_once_and_refuses_empty_shares():
    land = np.arange(10, 21)
    shares = [split_land_columns(land, 4, p)[0] for p in range(4)]
    np.testing.assert_array_equal(np.sort(np.concatenate(shares)), land)
    _, padded = split_land_columns(land, 4, 0)
    assert padded.shape == (4, 3) and set(padded.ravel()) == set(land)
    with pytest.raises(ValueError, match="empty"):
        split_land_columns(np.arange(3), 4, 0)


def test_a_leaf_that_is_not_per_column_is_refused():
    state = _State(jnp.ones((NCOL, 3)) * 280.0, jnp.zeros(NCOL, bool))
    out = list(_step(state, jnp.arange(4)))
    out[1] = jnp.zeros(NCOL + 1)
    _, padded = split_land_columns(np.arange(4), 2, 0)
    with pytest.raises(ValueError, match="not per-column"):
        gather_land_step(lambda: tuple(out), NCOL, padded, 0,
                         lambda tree: np.zeros(2))


def test_a_failure_on_one_process_stops_every_process_before_the_payload():
    state = _State(jnp.ones((NCOL, 3)) * 280.0, jnp.zeros(NCOL, bool))
    _, padded = split_land_columns(np.arange(4), 2, 0)
    calls = []

    def ag(tree):                     # process 1 reports failure in the flag
        calls.append(isinstance(tree, tuple))
        return np.array([0.0, 1.0])

    with pytest.raises(RuntimeError, match="another process"):
        gather_land_step(lambda: _step(state, jnp.arange(2)), NCOL, padded,
                         0, ag)
    assert calls == [False], "the payload gather ran after a peer failed"

    def boom():
        raise FloatingPointError("local land failure")
    with pytest.raises(FloatingPointError):
        gather_land_step(boom, NCOL, padded, 1, lambda tree: np.array([0.0, 1.0]))
