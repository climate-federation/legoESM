"""Bit-identity gate for the vectorised fill_corners_h2 "avg" branch.

The default "avg" cube-vertex corner fill was a ``for f in range(6)`` loop (96
serialised ``.at[f,i,j]`` scatters); it is now vectorised over the 6-face axis
(16 batched ``.at[:,i,j]`` scatters).  Faces are independent (each reads only
its own cells) and the inner->outer dependency within each corner is preserved
by statement order, so the vectorised result MUST equal the legacy loop
BIT-FOR-BIT (each cell is an independent 0.5*(a+b) — no reduction, no FMA-order
change).  Deep-dive 2026-06-14 lever #1 (per-device kernel-count reduction).
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids import halo


def _legacy_avg_fill_h2(padded):
    """The pre-vectorisation reference: per-face inside-out 2-point average."""
    for f in range(6):
        padded = padded.at[f, 1, 1].set(0.5 * (padded[f, 1, 2] + padded[f, 2, 1]))
        padded = padded.at[f, 0, 1].set(0.5 * (padded[f, 0, 2] + padded[f, 1, 1]))
        padded = padded.at[f, 1, 0].set(0.5 * (padded[f, 2, 0] + padded[f, 1, 1]))
        padded = padded.at[f, 0, 0].set(0.5 * (padded[f, 0, 1] + padded[f, 1, 0]))
        padded = padded.at[f, -2, 1].set(0.5 * (padded[f, -2, 2] + padded[f, -3, 1]))
        padded = padded.at[f, -1, 1].set(0.5 * (padded[f, -1, 2] + padded[f, -2, 1]))
        padded = padded.at[f, -2, 0].set(0.5 * (padded[f, -3, 0] + padded[f, -2, 1]))
        padded = padded.at[f, -1, 0].set(0.5 * (padded[f, -1, 1] + padded[f, -2, 0]))
        padded = padded.at[f, 1, -2].set(0.5 * (padded[f, 1, -3] + padded[f, 2, -2]))
        padded = padded.at[f, 0, -2].set(0.5 * (padded[f, 0, -3] + padded[f, 1, -2]))
        padded = padded.at[f, 1, -1].set(0.5 * (padded[f, 2, -1] + padded[f, 1, -2]))
        padded = padded.at[f, 0, -1].set(0.5 * (padded[f, 0, -2] + padded[f, 1, -1]))
        padded = padded.at[f, -2, -2].set(0.5 * (padded[f, -2, -3] + padded[f, -3, -2]))
        padded = padded.at[f, -1, -2].set(0.5 * (padded[f, -1, -3] + padded[f, -2, -2]))
        padded = padded.at[f, -2, -1].set(0.5 * (padded[f, -3, -1] + padded[f, -2, -2]))
        padded = padded.at[f, -1, -1].set(0.5 * (padded[f, -1, -2] + padded[f, -2, -1]))
    return padded


@pytest.mark.parametrize("n", [6, 12, 18])
def test_vectorised_avg_fill_h2_bit_identical(n):
    assert halo._corner_fill_mode == "avg", "test assumes default avg mode"
    rng = np.random.default_rng(n)
    padded = jnp.asarray(rng.standard_normal((6, n + 4, n + 4)))
    got = np.asarray(halo.fill_corners_h2(padded))
    ref = np.asarray(_legacy_avg_fill_h2(padded))
    np.testing.assert_array_equal(
        got, ref,
        err_msg=f"vectorised fill_corners_h2(avg) != legacy loop, n={n}")


def test_vectorised_avg_fill_h2_only_touches_corners():
    """The fill writes ONLY the four 2x2 cube-vertex L-blocks; interior + the
    edge halos (set upstream) are untouched."""
    n = 12
    rng = np.random.default_rng(99)
    padded = jnp.asarray(rng.standard_normal((6, n + 4, n + 4)))
    out = np.asarray(halo.fill_corners_h2(padded))
    base = np.asarray(padded)
    diff = np.abs(out - base) > 0
    # interior [2:-2, 2:-2] must be untouched
    assert not diff[:, 2:-2, 2:-2].any(), "interior modified"
