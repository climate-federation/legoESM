"""The default barotropic path must be able to take a step (#1455).

The shared weight helper returns FOUR values, the fourth being the number of
substep iterations the averaging window covers. A work-in-progress commit
reduced the lat-lon caller to a three-value unpack and substituted the substep
count for the fourth, which cannot execute at all: the default barotropic path
on this grid raised on its first step and every test that steps it went red.

Two things are asserted, because either alone passes for a broken tree: the
helper's arity, and that the caller unpacks all four. The loop count matters
beyond the crash -- the cosine window deliberately runs PAST the end of the
step so it is centred there, so replacing it with the substep count truncates
the window and changes the filter even where the unpack happens to work.
"""
from __future__ import annotations

import ast
import inspect

import jax.numpy as jnp
import pytest

from legoesm.ocean.dynamics import barotropic_latlon_cgrid as _latlon
from legoesm.ocean.dynamics.barotropic_common import compute_filter_weights


@pytest.mark.parametrize("use_cosine", [False, True])
def test_the_helper_returns_four_values(use_cosine):
    out = compute_filter_weights(6, jnp.float64, use_cosine=use_cosine)
    assert len(out) == 4, f"helper returned {len(out)} values, not four"


@pytest.mark.parametrize("use_cosine", [False, True])
def test_the_window_extends_past_the_step(use_cosine):
    """The reason the fourth value is not the substep count.

    The window is centred on the END of the step, not its middle, so it runs
    on past it: 2n-1 substeps for n. Substituting the substep count would
    truncate it and change the filter even in a tree where the unpack happened
    to work -- and if the two were equal this whole gate would be measuring
    nothing.
    """
    n = 6
    _, w_total, _, n_loop = compute_filter_weights(
        n, jnp.float64, use_cosine=use_cosine)
    assert n_loop == 2 * n - 1, (
        f"the window covers {n_loop} substeps, not the {2*n-1} a window "
        f"centred on the end of the step needs")
    assert n_loop > n


def test_every_caller_unpacks_all_four():
    src = inspect.getsource(_latlon)
    bad = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        fn = node.value.func
        if (getattr(fn, "id", None) or getattr(fn, "attr", None)) != "compute_filter_weights":
            continue
        for target in node.targets:
            if isinstance(target, ast.Tuple) and len(target.elts) != 4:
                bad.append(len(target.elts))
    assert not bad, (
        f"a call to compute_filter_weights unpacks {bad} values instead of four; "
        f"the default barotropic path raises on its first step")
