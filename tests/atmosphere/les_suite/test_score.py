"""Unit tests for the LES-vs-SCM score assembly."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from legoesm.atmosphere.les_suite.bridge import LESTruth
from legoesm.atmosphere.les_suite.score import (
    DiagnosticScore,
    PrognosticScore,
    diagnostic_flux_score,
    prognostic_profile_score,
)

NZ = 12


def _z():
    return jnp.linspace(10.0, 1600.0, NZ)


# Fixtures use cbl_nieuwstadt, a catalog case with no sponge (asserted in
# test_case_weights_cut_at_the_catalog_sponge_base), so their weights are the
# plain thickness weights.
def _dry_snapshot(theta=None, wtheta=None) -> LESTruth:
    z = _z()
    return LESTruth(
        case_name="cbl_nieuwstadt",
        heights_m=z,
        times_s=jnp.array([3600.0]),
        theta=300.0 + 0.003 * z if theta is None else theta,
        u=jnp.full((NZ,), 1.0),
        v=jnp.zeros((NZ,)),
        wtheta=jnp.full((NZ,), 0.05) if wtheta is None else wtheta,
    )


def _dry_series(nt=4) -> LESTruth:
    z = _z()
    theta = 300.0 + 0.003 * z
    theta = jnp.broadcast_to(theta, (nt, NZ))
    return LESTruth(
        case_name="cbl_nieuwstadt",
        heights_m=z,
        times_s=jnp.linspace(0.0, 3600.0, nt),
        theta=theta,
        u=jnp.full((nt, NZ), 2.0),
        v=jnp.zeros((nt, NZ)),
        wtheta=jnp.full((nt, NZ), 0.05),
    )


def _moist_snapshot() -> LESTruth:
    z = _z()
    return LESTruth(
        case_name="cbl_nieuwstadt",
        heights_m=z,
        times_s=jnp.array([3600.0]),
        theta=300.0 + 0.003 * z,
        u=jnp.full((NZ,), -8.0),
        v=jnp.zeros((NZ,)),
        wtheta=jnp.full((NZ,), 0.02),
        qt=0.016 - 1.0e-6 * z,
        wqt=jnp.full((NZ,), 1.0e-4),
    )


# --- diagnostic flux score ----------------------------------------------------
def test_diagnostic_perfect_flux_zero_score():
    truth = _dry_snapshot()
    score = diagnostic_flux_score(truth, truth.wtheta)
    assert isinstance(score, DiagnosticScore)
    assert float(score.wtheta_rmse) == pytest.approx(0.0, abs=1e-9)
    assert float(score.combined) == pytest.approx(0.0, abs=1e-9)
    assert score.wqt_rmse is None


def test_diagnostic_worse_flux_higher_score():
    truth = _dry_snapshot()
    good = diagnostic_flux_score(truth, truth.wtheta * 1.05)
    bad = diagnostic_flux_score(truth, truth.wtheta * 1.5)
    assert float(bad.wtheta_rmse) > float(good.wtheta_rmse) > 0.0


def test_diagnostic_moist_combines_two_fluxes():
    truth = _moist_snapshot()
    score = diagnostic_flux_score(
        truth, truth.wtheta * 1.1, truth.wqt * 1.2
    )
    assert score.wqt_rmse is not None
    assert float(score.combined) > 0.0


def test_diagnostic_rejects_series_truth():
    with pytest.raises(ValueError):
        diagnostic_flux_score(_dry_series(), jnp.zeros((4, NZ)))


def test_diagnostic_flux_score_ad_finite_grad_at_perfect():
    truth = _dry_snapshot()

    def loss(flux):
        return diagnostic_flux_score(truth, flux).combined

    g = jax.grad(loss)(truth.wtheta)
    assert bool(jnp.all(jnp.isfinite(g)))


# --- prognostic profile score -------------------------------------------------
def test_prognostic_perfect_zero_score():
    truth = _dry_series()
    score = prognostic_profile_score(truth, truth.theta, truth.u, truth.v)
    assert isinstance(score, PrognosticScore)
    assert float(score.combined) == pytest.approx(0.0, abs=1e-9)
    assert score.qt_rmse is None


def test_prognostic_single_time():
    truth = _dry_snapshot()
    score = prognostic_profile_score(truth, truth.theta, truth.u, truth.v)
    assert float(score.combined) == pytest.approx(0.0, abs=1e-9)


def test_prognostic_worse_profiles_higher_score():
    truth = _dry_series()
    good = prognostic_profile_score(truth, truth.theta + 0.1, truth.u, truth.v)
    bad = prognostic_profile_score(truth, truth.theta + 1.0, truth.u, truth.v)
    assert float(bad.theta_rmse) > float(good.theta_rmse) > 0.0
    assert float(bad.combined) > float(good.combined)


def test_prognostic_moist_includes_qt():
    truth = _moist_snapshot()
    score = prognostic_profile_score(
        truth, truth.theta, truth.u, truth.v, truth.qt
    )
    assert score.qt_rmse is not None
    assert float(score.combined) == pytest.approx(0.0, abs=1e-9)


def test_prognostic_ad_finite_grad_at_perfect():
    truth = _dry_series()

    def loss(theta):
        return prognostic_profile_score(truth, theta, truth.u, truth.v).combined

    g = jax.grad(loss)(truth.theta)
    assert bool(jnp.all(jnp.isfinite(g)))


def test_prognostic_custom_weights_shape_checked():
    truth = _dry_snapshot()
    with pytest.raises(ValueError):
        prognostic_profile_score(
            truth, truth.theta, truth.u, truth.v, weights=jnp.ones((NZ + 1,))
        )


def test_negative_weights_rejected():
    truth = _dry_snapshot()
    w = jnp.full((NZ,), 1.0 / NZ).at[0].set(-0.1)
    with pytest.raises(ValueError, match="non-negative"):
        prognostic_profile_score(truth, truth.theta, truth.u, truth.v, weights=w)


def test_unnormalized_weights_rejected():
    truth = _dry_snapshot()
    with pytest.raises(ValueError, match="sum to 1"):
        prognostic_profile_score(
            truth, truth.theta, truth.u, truth.v, weights=jnp.ones((NZ,))
        )


def test_normalized_custom_weights_accepted():
    truth = _dry_snapshot()
    w = jnp.full((NZ,), 1.0 / NZ)
    score = prognostic_profile_score(truth, truth.theta, truth.u, truth.v, weights=w)
    assert float(score.combined) == pytest.approx(0.0, abs=1e-9)


def test_diagnostic_moist_truth_without_scm_wqt_raises():
    truth = _moist_snapshot()
    with pytest.raises(ValueError, match="requires scm_wqt"):
        diagnostic_flux_score(truth, truth.wtheta)  # scm_wqt omitted


def test_prognostic_moist_truth_without_scm_qt_raises():
    truth = _moist_snapshot()
    with pytest.raises(ValueError, match="requires scm_qt"):
        prognostic_profile_score(truth, truth.theta, truth.u, truth.v)  # scm_qt omitted


# --- sponge-base cutoff (user 2026-10-19) -------------------------------------

def _gabls3_z():
    return jnp.linspace(3.125, 796.875, 128)          # 6.25 m cells, lid at 800 m


def test_case_weights_cut_at_the_catalog_sponge_base():
    from legoesm.atmosphere.les_suite.score import case_score_weights, sponge_base_for_score
    from legoesm.core.profile_metrics import layer_weights_from_heights
    z = _gabls3_z()
    assert sponge_base_for_score("gabls3", z) == pytest.approx(600.0, rel=1e-12)
    w = case_score_weights("gabls3", z)
    base = layer_weights_from_heights(z)
    below = z <= 600.0
    assert float(jnp.max(jnp.where(below, 0.0, w))) == 0.0
    assert float(jnp.sum(w)) == pytest.approx(1.0, rel=1e-12)
    expect = jnp.where(below, base, 0.0) / jnp.sum(jnp.where(below, base, 0.0))
    assert jnp.allclose(w, expect, rtol=1e-12, atol=0.0)
    # no sponge in the catalog: no cutoff; not a catalog case: refused
    assert jnp.allclose(case_score_weights("cbl_nieuwstadt", z), base, rtol=0.0, atol=0.0)
    with pytest.raises(ValueError, match="not an LES catalog case"):
        case_score_weights("cbl_test", z)


def test_profile_score_ignores_the_layer_above_the_sponge_base():
    z = _gabls3_z()
    nz = z.shape[0]
    above = z > 600.0

    def truth(spoil):
        th = 290.0 + 0.003 * z + jnp.where(above, spoil, 0.0)
        return LESTruth(case_name="gabls3", heights_m=z, times_s=jnp.array([3600.0]),
                        theta=th, u=jnp.full((nz,), 5.0) + jnp.where(above, spoil, 0.0),
                        v=jnp.zeros((nz,)), wtheta=jnp.full((nz,), 0.01))
    scm_th, scm_u = 290.5 + 0.003 * z, jnp.full((nz,), 4.0)
    a = prognostic_profile_score(truth(0.0), scm_th, scm_u, jnp.zeros((nz,)))
    b = prognostic_profile_score(truth(25.0), scm_th, scm_u, jnp.zeros((nz,)))
    assert float(a.combined) == pytest.approx(float(b.combined), rel=1e-12)
    d_a = diagnostic_flux_score(truth(0.0), jnp.full((nz,), 0.02))
    spoiled = truth(0.0)
    import dataclasses
    spoiled = dataclasses.replace(spoiled, wtheta=jnp.where(above, 3.0, spoiled.wtheta))
    d_b = diagnostic_flux_score(spoiled, jnp.full((nz,), 0.02))
    assert float(d_a.combined) == pytest.approx(float(d_b.combined), rel=1e-12)


def test_cutoff_reads_driver_labels_traces_and_masks_excluded_levels():
    import jax
    from legoesm.atmosphere.les_suite.catalog import CATALOG_ALIASES, catalog_case
    from legoesm.atmosphere.les_suite.score import (
        case_score_weights,
        normalized_rmse_series,
        sponge_base_for_score,
    )
    z = _gabls3_z()
    for label, name in CATALOG_ALIASES.items():          # driver default labels
        assert catalog_case(name) is not None
        assert sponge_base_for_score(label, z) == sponge_base_for_score(name, z)
    assert sponge_base_for_score("rico", z) is not None
    for label, name in CATALOG_ALIASES.items():
        assert catalog_case(name).name == name
    # under jit the heights are closed over (run metadata); traced heights are refused
    eager = case_score_weights("gabls3", z)
    jitted = jax.jit(lambda: case_score_weights("gabls3", z))()
    assert jnp.allclose(eager, jitted, rtol=1e-12, atol=0.0)
    with pytest.raises(TypeError, match="concrete heights"):
        jax.jit(lambda h: case_score_weights("gabls3", h))(z)
    # a non-finite residual above the cutoff cannot erase the error below it
    w = case_score_weights("gabls3", z)
    ref = jnp.broadcast_to(290.0 + 0.003 * z, (2, z.shape[0]))
    scm = ref + 0.5
    clean = normalized_rmse_series(scm, ref, w, 0.1)
    bad = normalized_rmse_series(scm.at[:, -1].set(jnp.nan), ref, w, 0.1)
    assert float(clean) > 0.0 and float(bad) == pytest.approx(float(clean), rel=1e-12)
    g = jax.grad(lambda s: normalized_rmse_series(s, ref, w, 0.1))(
        scm.at[:, -1].set(jnp.nan))
    assert bool(jnp.all(jnp.isfinite(g)))


def test_cutoff_accepts_fp32_heights_and_refuses_a_stretched_grid():
    import numpy as np
    from legoesm.atmosphere.les_suite.score import sponge_base_for_score
    dz = 400.0 / 96                                   # sbl_gabls1: 96 levels, 400 m
    z32 = jnp.asarray((np.arange(96) + 0.5) * dz, dtype=jnp.float32)
    assert float(sponge_base_for_score("sbl_gabls1", z32)) == pytest.approx(300.0, rel=1e-5)
    stretched = jnp.asarray(np.cumsum(np.linspace(2.0, 8.0, 96)))
    with pytest.raises(ValueError, match="not uniform"):
        sponge_base_for_score("sbl_gabls1", stretched)
    # no sponge, or an absolute one: nothing to infer, any grid is accepted
    assert sponge_base_for_score("cbl_nieuwstadt", stretched) is None


def test_explicit_weights_score_a_non_catalog_column():
    """A non-catalog name has no sponge cutoff and is refused by default; explicit
    weights are the way to score it, and bypass the catalog lookup."""
    from legoesm.core.profile_metrics import layer_weights_from_heights
    z = _z()
    truth = LESTruth(case_name="not_a_catalog_case", heights_m=z,
                     times_s=jnp.array([3600.0]), theta=300.0 + 0.003 * z,
                     u=jnp.full((NZ,), 1.0), v=jnp.zeros((NZ,)),
                     wtheta=jnp.full((NZ,), 0.05))
    with pytest.raises(ValueError, match="not an LES catalog case"):
        prognostic_profile_score(truth, truth.theta + 0.1, truth.u, truth.v)
    s = prognostic_profile_score(truth, truth.theta + 0.1, truth.u, truth.v,
                                 weights=layer_weights_from_heights(z))
    assert float(s.combined) > 0.0
