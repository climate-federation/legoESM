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
    count_valid_diagnoses,
    count_valid_multi_diagnoses,
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


def test_reduce_grad_finite_through_all_invalid_and_mixed_columns():
    """``_valid_profile_mean``'s gradient must stay FINITE through a column with
    NO valid levels (n=0) and route only to VALID levels in a mixed column.

    The existing differentiability test only exercises ALL-valid diagnoses.  The
    n=0 reduction is the classic ``jnp.where`` NaN-gradient hazard: the mean is
    ``jnp.where(n>0, Σ.../jnp.maximum(n,1), 0)`` and its grad-safety rests ENTIRELY
    on the ``jnp.maximum(n, 1)`` floor — a plausible "simplify" back to ``/n`` would
    leave the unselected ``n>0`` branch computing ``0/0`` and leak a NaN adjoint
    into the diagnosed values (the forward value would still read 0 via the where,
    so the forward-only all-invalid tests would NOT catch it).  A laminar column
    where the LES resolved no turbulence anywhere is a real occurrence, and its
    feedback must flow zero — not NaN — gradient so a downstream ``jax.grad`` of a
    loss over the assembled field stays finite.
    """
    # All levels invalid (n=0): the reduced value is 0 and the gradient is finite 0.
    def loss_all_invalid(k):
        v, _ = reduce_column_diagnosis(
            _Eddy(K=k, valid=jnp.array([False, False])), "eddy_diffusivity")
        return v ** 2

    g = jax.grad(loss_all_invalid)(jnp.array([2.0, 4.0]))
    assert bool(jnp.all(jnp.isfinite(g)))               # NOT NaN (the 0/0 hazard)
    np.testing.assert_array_equal(np.asarray(g), [0.0, 0.0])  # no valid level ⇒ zero

    # Mixed column: the gradient routes ONLY to the valid level (the invalid level's
    # diagnosed value never influences the reduced mean, so its adjoint is exactly 0).
    def loss_mixed(k):
        v, _ = reduce_column_diagnosis(
            _Eddy(K=k, valid=jnp.array([True, False])), "eddy_diffusivity")
        return v ** 2

    gm = jax.grad(loss_mixed)(jnp.array([3.0, 99.0]))
    assert bool(jnp.all(jnp.isfinite(gm)))
    # value = mean over 1 valid level = 3; d(9)/dK_valid = 6; invalid level ⇒ 0.
    np.testing.assert_allclose(np.asarray(gm), [6.0, 0.0], rtol=1e-12)


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


def test_assemble_environment_nonfinite_valid_sample_raises():
    """A VALID worst-column sample with a non-finite env tag (e.g. a NaN SST over land)
    must FAIL LOUD — else the NW kernel lets that one NaN poison every grid column's
    total weight and silently collapses the regression to an all-background no-op.
    Mirrors the cluster_columns_by_environment guard (closes the asymmetry)."""
    records = [_erec(0, float("nan"), 100.0, 2.0)]              # NaN SST
    diagnoses = [_Eddy(K=jnp.array([10.0]), valid=jnp.array([True]))]
    grid_env = jnp.asarray([[280.0, 100.0, 2.0]])
    with pytest.raises(ValueError, match="non-finite environment tag"):
        assemble_feedback_field(
            records, diagnoses, (1, 1), method="eddy_diffusivity",
            strategy="environment", grid_env=grid_env,
            length_scales=jnp.array([5.0, 50.0, 2.0]))


def test_assemble_environment_nonfinite_invalid_sample_is_exempt():
    """An INVALID-diagnosis sample with a non-finite env tag does NOT trip the guard:
    build_parameter_field zeros invalid samples before the kernel, so they cannot
    poison it — only a VALID sample's NaN env is fatal. The valid sample still drives
    a finite field. (Non-vacuous companion to the raise test above.)"""
    # col 0: valid, good env, K→10; col 1: INVALID diagnosis, NaN SST (must be exempt).
    records = [_erec(0, 280.0, 100.0, 2.0), _erec(1, float("nan"), 100.0, 2.0)]
    diagnoses = [_Eddy(K=jnp.array([10.0]), valid=jnp.array([True])),
                 _Eddy(K=jnp.array([1.0]), valid=jnp.array([False]))]
    grid_env = jnp.asarray([[280.0, 100.0, 2.0], [281.0, 110.0, 2.2]])
    field = assemble_feedback_field(
        records, diagnoses, (1, 2), method="eddy_diffusivity", background=0.4,
        strategy="environment", grid_env=grid_env,
        length_scales=jnp.array([5.0, 50.0, 2.0]))
    f = np.asarray(field).reshape(-1)
    assert np.all(np.isfinite(f))                  # no NaN poison from the exempt sample
    assert f[0] == pytest.approx(10.0, abs=1.0)    # driven by the valid sample


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


def test_column_environment_grid_warns_on_sst_fallback():
    """The SECOND SST-fallback path (iter 281, symmetric with the compare path): when
    sst_K is None the env-kernel grid SST tag falls back to lowest-level AIR TEMPERATURE
    and WARNS, so the operator learns the cross-resolution deploy's similarity matching
    is approximate (the surface T_295 becomes the SST predictor, not a crash/NaN)."""
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.training.compare_reanalysis import ColumnState

    nlat, nlon, nlev = 2, 2, 5
    sigma = create_sigma_coordinate(nlev)
    # surface-last T: top 220 K, surface 295 K — the fallback must use the surface (295).
    t_profile = jnp.linspace(220.0, 295.0, nlev)
    model = ColumnState(
        T=jnp.broadcast_to(t_profile, (nlat, nlon, nlev)),
        q_v=jnp.full((nlat, nlon, nlev), 5e-3), u=jnp.full((nlat, nlon, nlev), 5.0),
        v=jnp.zeros((nlat, nlon, nlev)), p_s=jnp.full((nlat, nlon), 1.0e5), sst_K=None)
    with pytest.warns(UserWarning, match="model.sst_K is None"):
        grid_env, _ = column_environment_grid(model, sigma)
    np.testing.assert_allclose(np.asarray(grid_env[:, 0]), 295.0)  # surface air T, not top


def test_column_environment_grid_matches_manifest_sample_env():
    """CROSS-PATH CONSISTENCY (Codex iter-42, the env-kernel deploy invariant): the
    kernel's grid predictors (``column_environment_grid``) MUST equal the manifest's
    per-column sample predictors (the compare's ``record.environment``) for the SAME
    state, or the Nadaraya–Watson kernel compares mismatched quantities and the
    cross-resolution deploy is meaningless. Both paths share the SST fallback, the
    pure-sigma default pressure, and ``compute_column_environment`` — this LOCKS that
    so a future divergence (e.g. one path changing its SST fallback or default p) fails.
    """
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.training.compare_reanalysis import (
        ColumnState,
        compare_state_to_reference,
    )

    nlat, nlon, nlev = 3, 4, 6
    sigma = create_sigma_coordinate(nlev)
    # Spatially-VARYING fields so worst columns have DISTINCT environments (SST,
    # CAPE via T/q_v, shear via u) — a uniform state would make the test vacuous.
    ones = jnp.ones((nlat, nlon, nlev))
    iy = jnp.arange(nlat)[:, None, None]
    ix = jnp.arange(nlon)[None, :, None]
    lev = jnp.linspace(0.0, 1.0, nlev)[None, None, :]      # 0=top, 1=surface
    base = ColumnState(
        T=(230.0 + 60.0 * lev + 3.0 * iy + 1.5 * ix) * ones,  # warmer below + gradient
        q_v=(1e-3 + 8e-3 * lev) * (1.0 + 0.05 * ix) * ones,   # moister below + gradient
        u=2.0 * lev * (1.0 + 0.3 * iy) * ones,                # sheared, varying by row
        v=jnp.zeros((nlat, nlon, nlev)),
        p_s=jnp.full((nlat, nlon), 1.0e5),
        sst_K=290.0 + 2.0 * jnp.arange(nlat)[:, None] + 0.5 * jnp.arange(nlon)[None, :])
    lat_deg = jnp.zeros((nlat, nlon))
    lon_deg = jnp.zeros((nlat, nlon))

    def _assert_consistent(model):
        reference = model._replace(T=model.T - (0.5 + 0.4 * iy) * ones)  # column-varying bias
        comp = compare_state_to_reference(
            model=model, reference=reference,
            sigma_full=sigma.sigma_full, sigma_half=sigma.sigma_half,
            lat_deg=lat_deg, lon_deg=lon_deg, time_index=0, n_worst=nlat * nlon)
        grid_env = np.asarray(column_environment_grid(model, sigma)[0])  # (ncol, 3)
        # Finiteness guard: a NaN on BOTH paths would otherwise pass assert_allclose
        # silently (equal_nan=True default) — Codex. The env must be finite anyway.
        assert np.all(np.isfinite(grid_env))
        # Every worst record's stored environment MUST equal the grid predictor at its
        # flat index — exact match (same function, same inputs, same defaults).
        for rec in comp.manifest:
            assert all(np.isfinite([rec.environment.sst_K, rec.environment.cape_J_kg,
                                    rec.environment.bulk_shear_m_s]))
            np.testing.assert_allclose(
                grid_env[rec.flat_index],
                np.array([rec.environment.sst_K, rec.environment.cape_J_kg,
                          rec.environment.bulk_shear_m_s]),
                rtol=1e-6, atol=1e-6,
                err_msg=f"grid_env vs manifest env diverged at flat_index {rec.flat_index}")

    _assert_consistent(base)                          # explicit-SST path (model.sst_K)
    # SST-FALLBACK path: no sst_K ⇒ BOTH paths must derive SST from the surface-level
    # air temperature T[..., -1] identically (Codex: lock the fallback branch too, not
    # only the explicit-SST branch). T varies per column so the fallback SST still
    # discriminates a mis-index.
    _assert_consistent(base._replace(sst_K=None))


class _CEps(NamedTuple):
    C_eps: jax.Array
    valid: jax.Array


def test_reduce_c_eps_valid_mean():
    diag = _CEps(C_eps=jnp.array([0.2, 0.4, 9.9]),
                 valid=jnp.array([True, True, False]))
    value, valid = reduce_column_diagnosis(diag, "c_eps")
    assert bool(valid)
    assert float(value) == pytest.approx(0.3)


def test_count_valid_diagnoses_counts_usable_columns():
    """count_valid_diagnoses = number of columns with a VALID single-method diagnosis
    (≥1 valid level + finite) — the exact set assemble corrects (iter 100)."""
    diags = [
        _Eddy(K=jnp.array([10.0, 30.0]), valid=jnp.array([True, True])),   # valid
        _Eddy(K=jnp.array([5.0, 6.0]), valid=jnp.array([False, False])),   # no valid lvl
        _Eddy(K=jnp.array([jnp.nan, 1.0]), valid=jnp.array([True, True])),  # non-finite
        _Eddy(K=jnp.array([2.0]), valid=jnp.array([True])),                # valid
    ]
    assert count_valid_diagnoses(diags, "eddy_diffusivity") == 2
    assert count_valid_diagnoses([], "eddy_diffusivity") == 0


def test_count_valid_multi_diagnoses_any_method_valid():
    """A column counts if ANY requested coefficient's diagnosis is valid (the multi
    {method: diagnosis} dict path) — iter 100. clubb_coefficient reads .C_K and
    prandtl_number reads .Pr_t, so use the matching stand-in profiles."""
    col_a = {"clubb_coefficient": _Ck(jnp.array([0.4]), jnp.array([True])),
             "prandtl_number": _Prt(jnp.array([0.8]), jnp.array([False]))}  # C_K valid
    col_b = {"clubb_coefficient": _Ck(jnp.array([0.4]), jnp.array([False])),
             "prandtl_number": _Prt(jnp.array([0.8]), jnp.array([False]))}  # neither
    col_c = {"clubb_coefficient": _Ck(jnp.array([0.4]), jnp.array([True])),
             "prandtl_number": _Prt(jnp.array([0.8]), jnp.array([True]))}   # both valid
    methods = {"clubb_coefficient", "prandtl_number"}
    assert count_valid_multi_diagnoses([col_a, col_b, col_c], methods) == 2  # a + c
    assert count_valid_multi_diagnoses([col_b], methods) == 0
    assert count_valid_multi_diagnoses([], methods) == 0
    # an EXTRA non-spec method that is valid must NOT mark the column valid (only the
    # requested `methods` are checked) — iter 100 Codex fix.
    col_extra = {"clubb_coefficient": _Ck(jnp.array([0.4]), jnp.array([False])),
                 "prandtl_number": _Prt(jnp.array([0.8]), jnp.array([False])),
                 "entrainment": _Ent(jnp.asarray(0.02), jnp.asarray(True))}  # valid extra
    assert count_valid_multi_diagnoses([col_extra], methods) == 0
