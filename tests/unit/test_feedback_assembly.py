"""Unit tests for :mod:`legoesm.training.feedback_assembly`.

Stage-7 glue: reduce per-column LES diagnoses to a feedback field.  Covers the
eddy-diffusivity valid-mean reduction, entrainment scalar, invalid-column
background, the scatter placement at flat indices, dispatch, and differentiability.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.training.feedback_assembly import (
    assemble_feedback_field,
    reduce_column_diagnosis,
)


class _Eddy(NamedTuple):
    K: jax.Array
    valid: jax.Array


class _Ent(NamedTuple):
    w_entrainment: jax.Array
    valid: jax.Array


class _Rec(NamedTuple):
    flat_index: int


def test_reduce_eddy_diffusivity_valid_mean():
    diag = _Eddy(K=jnp.array([10.0, 20.0, 30.0, 99.0]),
                 valid=jnp.array([True, True, True, False]))
    value, valid = reduce_column_diagnosis(diag, "eddy_diffusivity")
    assert bool(valid)
    assert float(value) == pytest.approx(20.0)  # mean(10,20,30), 99 excluded


def test_reduce_eddy_diffusivity_all_invalid():
    diag = _Eddy(K=jnp.array([5.0, 6.0]), valid=jnp.array([False, False]))
    value, valid = reduce_column_diagnosis(diag, "eddy_diffusivity")
    assert not bool(valid)
    assert float(value) == 0.0


def test_reduce_entrainment():
    diag = _Ent(w_entrainment=jnp.asarray(0.02), valid=jnp.asarray(True))
    value, valid = reduce_column_diagnosis(diag, "entrainment")
    assert float(value) == pytest.approx(0.02)
    assert bool(valid)


def test_reduce_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown diagnosis method"):
        reduce_column_diagnosis(_Ent(jnp.asarray(0.0), jnp.asarray(True)), "bogus")


def test_assemble_field_scatters_at_flat_indices():
    records = [_Rec(flat_index=1), _Rec(flat_index=5)]
    diagnoses = [
        _Eddy(K=jnp.array([10.0, 30.0]), valid=jnp.array([True, True])),  # ->20
        _Eddy(K=jnp.array([40.0, 60.0]), valid=jnp.array([True, True])),  # ->50
    ]
    field = assemble_feedback_field(
        records, diagnoses, (2, 3), method="eddy_diffusivity", background=-1.0
    )
    assert field.shape == (2, 3)
    f = np.asarray(field).reshape(-1)
    assert f[1] == pytest.approx(20.0)
    assert f[5] == pytest.approx(50.0)
    assert (f == -1.0).sum() == 4  # background elsewhere


def test_assemble_field_invalid_column_keeps_background():
    records = [_Rec(flat_index=0), _Rec(flat_index=3)]
    diagnoses = [
        _Eddy(K=jnp.array([10.0]), valid=jnp.array([True])),
        _Eddy(K=jnp.array([99.0]), valid=jnp.array([False])),  # invalid
    ]
    field = assemble_feedback_field(
        records, diagnoses, (2, 2), method="eddy_diffusivity", background=0.0
    )
    f = np.asarray(field).reshape(-1)
    assert f[0] == pytest.approx(10.0)
    assert f[3] == pytest.approx(0.0)  # invalid -> background


def test_assemble_entrainment_field():
    records = [_Rec(flat_index=2)]
    diagnoses = [_Ent(w_entrainment=jnp.asarray(0.05), valid=jnp.asarray(True))]
    field = assemble_feedback_field(
        records, diagnoses, (2, 2), method="entrainment", background=0.0
    )
    assert float(np.asarray(field).reshape(-1)[2]) == pytest.approx(0.05)


def test_reduce_entrainment_invalid():
    diag = _Ent(w_entrainment=jnp.asarray(0.0), valid=jnp.asarray(False))
    value, valid = reduce_column_diagnosis(diag, "entrainment")
    assert not bool(valid)


def test_valid_flag_not_value_drives_background():
    """A VALID column with value 0.0 keeps 0.0 (not the nonzero background);
    only the valid FLAG triggers the background fallback."""
    records = [_Rec(flat_index=0), _Rec(flat_index=3)]
    diagnoses = [
        _Eddy(K=jnp.array([0.0, 0.0]), valid=jnp.array([True, True])),  # valid, value 0
        _Eddy(K=jnp.array([5.0]), valid=jnp.array([False])),            # invalid
    ]
    field = assemble_feedback_field(
        records, diagnoses, (2, 2), method="eddy_diffusivity", background=9.0
    )
    f = np.asarray(field).reshape(-1)
    assert f[0] == pytest.approx(0.0)  # valid value 0 -> 0 (NOT background 9)
    assert f[3] == pytest.approx(9.0)  # invalid -> background


def test_assemble_empty_records_is_background():
    field = assemble_feedback_field([], [], (2, 2), background=7.0)
    np.testing.assert_array_equal(np.asarray(field), np.full((2, 2), 7.0))


def test_assemble_empty_records_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown diagnosis method"):
        assemble_feedback_field([], [], (2, 2), method="bogus")


def test_assemble_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        assemble_feedback_field(
            [_Rec(flat_index=0)], [], (2, 2), method="eddy_diffusivity")


def test_assemble_field_differentiable():
    records = [_Rec(flat_index=0), _Rec(flat_index=3)]

    def loss(k0, k1):
        diagnoses = [
            _Eddy(K=k0, valid=jnp.array([True, True])),
            _Eddy(K=k1, valid=jnp.array([True, True])),
        ]
        field = assemble_feedback_field(
            records, diagnoses, (2, 2), method="eddy_diffusivity")
        return jnp.sum(field ** 2)

    g0 = jax.grad(loss, argnums=0)(jnp.array([2.0, 4.0]), jnp.array([6.0, 8.0]))
    assert bool(jnp.all(jnp.isfinite(g0)))
    # field[0] = mean(2,4)=3; d(9)/dK0 via 0.5 each = 2*3*0.5 = 3.
    np.testing.assert_allclose(np.asarray(g0), [3.0, 3.0], rtol=1e-12)
