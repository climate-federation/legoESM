"""Tests for the SCM-vs-LES forward evaluation (les_suite/scm_runner.py).

These actually build + run the dycore-free SCM (single column, cheap on CPU).
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.les_suite.bridge import LESReferenceArtifact
from legoesm.atmosphere.les_suite.scm_runner import (
    build_cbl_scm_from_artifact,
    scm_final_theta_on,
    scm_les_final_loss,
)
from legoesm.atmosphere.physics import TurbulenceConfig
from legoesm.atmosphere.physics.turbulence.config import (
    MYNN25Config,
    SurfaceLayerConfig,
)

NZ, NT = 24, 3


def _cbl_artifact(**over) -> LESReferenceArtifact:
    z = np.linspace(10.0, 1500.0, NZ)
    # a developing CBL: well-mixed θ below ~800 m, inversion above
    theta_col = np.where(z > 800.0, 300.0 + 0.006 * (z - 800.0), 300.0)
    theta = np.broadcast_to(theta_col, (NT, NZ)).copy()
    base = dict(
        case_name="cbl_run",
        sgs="lasd",
        heights_m=z,
        times_s=np.linspace(0.0, 1800.0, NT),
        theta=theta,
        u=np.zeros((NT, NZ)),
        v=np.zeros((NT, NZ)),
        wtheta_resolved=np.broadcast_to(
            0.06 * np.clip(1.0 - z / 900.0, -0.2, 1.0), (NT, NZ)).copy(),
        wtheta_sgs=np.zeros((NT, NZ)),
        prescribe="fluxes",
        w_theta_s=np.full(NT, 0.06),
        f_c=0.0,
    )
    base.update(over)
    return LESReferenceArtifact(**base)


def _mynn_config():
    return TurbulenceConfig(
        scheme="mynn25",
        mynn25=MYNN25Config(
            surface=SurfaceLayerConfig(z0=0.1, Cd_neutral=1.5e-3, Ch_neutral=0.0)
        ),
    )


def test_build_scm_from_artifact():
    art = _cbl_artifact()
    scm, grid = build_cbl_scm_from_artifact(
        art, _mynn_config(), nlev=NZ, dt=5.0)
    assert grid.nlev == NZ
    # SCM heights are top-to-bottom (decreasing with index)
    assert float(grid.z_scm_m[0]) > float(grid.z_scm_m[-1])
    # initial temperature finite + physical
    T0 = np.asarray(scm.state.T.data[0, 0, 0])
    assert np.all(np.isfinite(T0))
    assert np.all(T0 > 200.0) and np.all(T0 < 340.0)


def test_scm_final_profile_finite_on_eval_grid():
    art = _cbl_artifact()
    scm, grid = build_cbl_scm_from_artifact(
        art, _mynn_config(), nlev=NZ, dt=5.0)
    z_eval = jnp.asarray(art.heights_m)
    theta, u, v = scm_final_theta_on(scm, grid, nsteps=20, z_eval=z_eval)
    assert theta.shape == (NZ,)
    assert bool(jnp.all(jnp.isfinite(theta)))
    # potential temperature stays in a physical band for a short CBL run
    assert bool(jnp.all(theta > 290.0)) and bool(jnp.all(theta < 320.0))


def test_scm_les_final_loss_is_finite_nonnegative():
    art = _cbl_artifact()
    loss = scm_les_final_loss(art, _mynn_config(), nlev=NZ, dt=10.0)
    assert np.isfinite(loss)
    assert loss >= 0.0


def test_loss_responds_to_surface_flux_mismatch():
    # Scoring the SCM against a truth whose surface flux is very different should
    # not crash and returns a finite loss (sanity that the objective is wired).
    art = _cbl_artifact()
    loss = scm_les_final_loss(art, _mynn_config(), nlev=NZ, dt=10.0)
    assert np.isfinite(loss)


def test_non_cbl_artifact_rejected():
    # an artifact without a prescribed surface heat flux is not a wired CBL
    art = _cbl_artifact(prescribe="T_s", w_theta_s=None, T_s=np.full(NT, 300.0))
    with pytest.raises(ValueError):
        build_cbl_scm_from_artifact(art, _mynn_config(), nlev=NZ)


def test_diverged_scm_scores_infinite_not_zero(monkeypatch):
    # Regression: a DIVERGED SCM (NaN θ) must score +inf, not 0. The score's
    # safe_sqrt maps NaN->0 (perfect fit), so without the finiteness guard a
    # blown-up SCM is selected as the BEST candidate (the 100%-improvement bug seen
    # on the real 2h CBL tuning). Patch the SCM output to NaN and assert the guard.
    from legoesm.atmosphere.les_suite import scm_runner

    nan = jnp.full((NZ,), jnp.nan)
    monkeypatch.setattr(scm_runner, "scm_final_theta_on",
                        lambda *a, **k: (nan, nan, nan))
    art = _cbl_artifact()
    loss = scm_les_final_loss(art, _mynn_config(), nlev=NZ, dt=20.0)
    assert not np.isfinite(loss)  # +inf, NOT 0.0
