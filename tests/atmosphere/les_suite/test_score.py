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


def _dry_snapshot(theta=None, wtheta=None) -> LESTruth:
    z = _z()
    return LESTruth(
        case_name="cbl_test",
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
        case_name="cbl_series",
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
        case_name="bomex_test",
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
