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


def test_reduce_eddy_nonfinite_marked_invalid():
    # A non-finite reduced coefficient (degenerate / blown-up LES) → invalid.
    nan_diag = _Eddy(K=jnp.array([jnp.nan, 1.0]), valid=jnp.array([True, True]))
    _, valid = reduce_column_diagnosis(nan_diag, "eddy_diffusivity")
    assert not bool(valid)
    inf_diag = _Eddy(K=jnp.array([jnp.inf]), valid=jnp.array([True]))
    _, valid_inf = reduce_column_diagnosis(inf_diag, "eddy_diffusivity")
    assert not bool(valid_inf)
    # a finite diagnosis stays valid (backward compatible).
    ok = _Eddy(K=jnp.array([2.0, 4.0]), valid=jnp.array([True, True]))
    val, valid_ok = reduce_column_diagnosis(ok, "eddy_diffusivity")
    assert bool(valid_ok)
    assert float(val) == pytest.approx(3.0)


class _Ck(NamedTuple):
    C_K: jax.Array
    valid: jax.Array


def test_reduce_clubb_coefficient_valid_mean():
    # dimensionless C_K profile → valid-level mean (here (0.2+0.4)/2 = 0.3).
    diag = _Ck(C_K=jnp.array([0.2, 0.4, 9.9]),
               valid=jnp.array([True, True, False]))
    value, valid = reduce_column_diagnosis(diag, "clubb_coefficient")
    assert bool(valid)
    assert float(value) == pytest.approx(0.3)


class _Prt(NamedTuple):
    Pr_t: jax.Array
    valid: jax.Array


def test_reduce_prandtl_number_valid_mean():
    diag = _Prt(Pr_t=jnp.array([0.7, 0.9, 9.9]),
               valid=jnp.array([True, True, False]))
    value, valid = reduce_column_diagnosis(diag, "prandtl_number")
    assert bool(valid)
    assert float(value) == pytest.approx(0.8)


def test_reduce_clubb_coefficient_all_invalid_keeps_background():
    diag = _Ck(C_K=jnp.array([0.5, 0.6]), valid=jnp.array([False, False]))
    _, valid = reduce_column_diagnosis(diag, "clubb_coefficient")
    assert not bool(valid)


def test_reduce_clubb_coefficient_nonfinite_marked_invalid():
    diag = _Ck(C_K=jnp.array([jnp.nan, 0.4]), valid=jnp.array([True, True]))
    _, valid = reduce_column_diagnosis(diag, "clubb_coefficient")
    assert not bool(valid)


def test_reduce_entrainment_nonfinite_marked_invalid():
    nan_ent = _Ent(w_entrainment=jnp.asarray(jnp.nan), valid=jnp.asarray(True))
    _, valid = reduce_column_diagnosis(nan_ent, "entrainment")
    assert not bool(valid)


def test_assemble_nonfinite_diagnosis_keeps_background():
    # A NaN diagnosis at the worst column keeps the background, never injects NaN.
    records = [_Rec(flat_index=0)]
    diagnoses = [_Eddy(K=jnp.array([jnp.nan]), valid=jnp.array([True]))]
    field = assemble_feedback_field(
        records, diagnoses, (2, 2), method="eddy_diffusivity", background=0.4)
    assert np.all(np.isfinite(np.asarray(field)))
    np.testing.assert_allclose(np.asarray(field), 0.4)   # dropped → all background


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


# --- environment-kernel generalization (iter 42) ----------------------------

from legoesm.training.column_manifest import (  # noqa: E402
    ColumnEnvironment,
    ColumnRecord,
)
from legoesm.training.feedback_assembly import column_environment_grid  # noqa: E402


def _erec(flat, sst, cape, shear):
    return ColumnRecord(
        flat_index=flat, grid_index=(flat,), lat_deg=0.0, lon_deg=0.0,
        time_index=0, combined_score=1.0, T_rmse_K=0.0, qv_rmse_kg_kg=0.0,
        wind_rmse_m_s=0.0, precip_err_mm_day=0.0,
        environment=ColumnEnvironment(sst_K=sst, cape_J_kg=cape, bulk_shear_m_s=shear))


def test_assemble_environment_generalizes_to_similar_columns():
    """strategy='environment' spreads each diagnosed value to columns whose env
    is similar — incl. columns NOT in the manifest."""
    # 4 columns: 0,1 share env A (cold/dry); 2,3 share env B (warm/moist).
    grid_env = jnp.asarray([
        [280.0, 100.0, 2.0], [281.0, 110.0, 2.2],     # env A (cols 0,1)
        [302.0, 3000.0, 25.0], [301.0, 2950.0, 24.0],  # env B (cols 2,3)
    ])
    length_scales = jnp.std(grid_env, axis=0)
    # worst columns: col 0 (env A) diagnosed K→10; col 3 (env B) diagnosed K→20.
    records = [_erec(0, 280.0, 100.0, 2.0), _erec(3, 301.0, 2950.0, 24.0)]
    diagnoses = [_Eddy(K=jnp.array([10.0, 10.0]), valid=jnp.array([True, True])),
                 _Eddy(K=jnp.array([20.0, 20.0]), valid=jnp.array([True, True]))]

    field = assemble_feedback_field(
        records, diagnoses, (2, 2), method="eddy_diffusivity", background=0.4,
        strategy="environment", grid_env=grid_env, length_scales=length_scales)
    f = np.asarray(field).reshape(-1)
    # The NON-manifest columns got generalized: col 1 (env A) ≈10, col 2 (env B) ≈20.
    assert f[1] == pytest.approx(10.0, abs=0.5)
    assert f[2] == pytest.approx(20.0, abs=0.5)
    assert f[0] == pytest.approx(10.0, abs=0.5) and f[3] == pytest.approx(20.0, abs=0.5)


def test_assemble_environment_requires_grid_env():
    records = [_erec(0, 280.0, 100.0, 2.0)]
    diagnoses = [_Eddy(K=jnp.array([10.0]), valid=jnp.array([True]))]
    with pytest.raises(ValueError, match="requires grid_env"):
        assemble_feedback_field(records, diagnoses, (1, 1), strategy="environment")


def test_assemble_unknown_strategy_raises():
    with pytest.raises(ValueError, match="Unknown strategy"):
        assemble_feedback_field([], [], (2, 2), strategy="bogus")


def test_column_environment_grid_shapes_and_sst():
    """column_environment_grid → (ncol, 3) env predictors + (3,) length scales;
    the SST predictor equals the state's sst_K."""
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.training.compare_reanalysis import ColumnState

    nlat, nlon, nlev = 3, 4, 5
    sigma = create_sigma_coordinate(nlev)
    sst = jnp.full((nlat, nlon), 295.0)
    model = ColumnState(
        T=jnp.full((nlat, nlon, nlev), 280.0), q_v=jnp.full((nlat, nlon, nlev), 5e-3),
        u=jnp.full((nlat, nlon, nlev), 5.0), v=jnp.zeros((nlat, nlon, nlev)),
        p_s=jnp.full((nlat, nlon), 1.0e5), sst_K=sst)
    grid_env, length_scales = column_environment_grid(model, sigma)
    assert grid_env.shape == (nlat * nlon, 3)
    assert length_scales.shape == (3,)
    assert bool(jnp.all(jnp.isfinite(grid_env)))
    np.testing.assert_allclose(np.asarray(grid_env[:, 0]), 295.0)  # SST predictor
    assert bool(jnp.all(length_scales > 0))
