"""Shared NEMO ``fld_read`` weighted-map regressions."""

from __future__ import annotations

import jax
import numpy as np
import pytest

from legoesm.ocean.forcing.nemo_fld_read import (
    decode_source_indices,
    nemo_fld_interp,
    rotate_en_to_ij,
)


def test_decode_source_indices_preserves_nemo_fortran_flat_values():
    tables = np.array([[[1.0]], [[2.0]], [[5.0]], [[6.0]]])
    np.testing.assert_array_equal(
        decode_source_indices(tables), np.array([[[0]], [[1]], [[4]], [[5]]])
    )
    malformed = tables.copy()
    malformed[0, 0, 0] = 1.5
    with pytest.raises(ValueError, match="positive integer"):
        decode_source_indices(malformed)


def test_bilinear_map_runs_through_production_jit_in_source_order():
    source = np.arange(12.0, dtype=np.float64).reshape(3, 4)
    indices = np.array(
        [
            [[0, 1], [4, 5]],
            [[1, 2], [5, 6]],
            [[4, 5], [8, 9]],
            [[5, 6], [9, 10]],
        ],
        dtype=np.int64,
    )
    weights = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)[:, None, None]
    weights = np.broadcast_to(weights, indices.shape)
    expected = np.zeros(indices.shape[1:], dtype=np.float64)
    flat = source.reshape(-1)
    for corner in range(4):
        expected = expected + weights[corner] * flat[indices[corner]]
    actual = np.asarray(
        jax.jit(lambda x, i, w: nemo_fld_interp(x, i, w, bicubic=False))(
            source, indices, weights
        )
    )
    np.testing.assert_array_equal(actual, expected)


def test_bicubic_map_keeps_three_complete_gradient_loops():
    source = np.arange(30.0, dtype=np.float64).reshape(5, 6)
    indices = np.array([[[7]], [[8]], [[13]], [[14]]], dtype=np.int64)
    weights = np.arange(1.0, 17.0, dtype=np.float64).reshape(16, 1, 1) / 100.0
    flat = source.reshape(-1)
    expected = np.array([[0.0]], dtype=np.float64)
    for corner in range(4):
        expected = expected + weights[corner] * flat[indices[corner]]
    differences = []
    for corner in range(4):
        point = int(indices[corner, 0, 0])
        jj, ii = divmod(point, source.shape[1])
        di = source[jj, ii + 1] - source[jj, ii - 1]
        dj = source[jj + 1, ii] - source[jj - 1, ii]
        dij = (
            source[jj + 1, ii + 1] - source[jj + 1, ii - 1]
            - (source[jj - 1, ii + 1] - source[jj - 1, ii - 1])
        )
        differences.append((di, dj, dij))
    for offset, scale, component in ((4, 0.5, 0), (8, 0.5, 1), (12, 0.25, 2)):
        for corner in range(4):
            expected = (
                expected
                + weights[corner + offset] * scale * differences[corner][component]
            )
    actual = np.asarray(
        jax.jit(lambda x, i, w: nemo_fld_interp(x, i, w, bicubic=True))(
            source, indices, weights
        )
    )
    np.testing.assert_array_equal(actual, expected)


def test_rotation_geographic_arm_is_identity_under_jit():
    u = np.array([[1.0, -2.0]], dtype=np.float64)
    v = np.array([[3.0, 4.0]], dtype=np.float64)
    ui, vj = jax.jit(rotate_en_to_ij)(u, v, np.ones_like(u), np.zeros_like(u))
    np.testing.assert_array_equal(np.asarray(ui), u)
    np.testing.assert_array_equal(np.asarray(vj), v)
