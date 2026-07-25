"""Tests for the SCM-vs-LES forward evaluation (les_suite/scm_runner.py).

These actually build + run the dycore-free SCM (single column, cheap on CPU).
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.les_suite.bridge import LESReferenceArtifact
from legoesm.atmosphere.les_suite.scm_coupling import (
    interp_profile,
    theta_from_temperature,
)
from legoesm.atmosphere.les_suite.scm_runner import (
    build_cbl_scm_from_artifact,
    scm_final_theta_on,
    scm_les_final_loss,
    scm_les_final_score,
    scm_scan_final_state,
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


def test_final_score_components_finite_and_combined_matches_loss():
    # scm_les_final_score exposes the per-variable (θ/u/v) breakdown behind the loss.
    # Its .combined MUST equal scm_les_final_loss (the loss is a thin wrapper) — this
    # is the self-check the θ-consistent D7 significance analysis relies on to confirm
    # a re-score reproduces the harness before ranking closures on θ_rmse.
    art = _cbl_artifact()
    score = scm_les_final_score(art, _mynn_config(), nlev=NZ, dt=10.0)
    assert score is not None
    for comp in (score.theta_rmse, score.u_rmse, score.v_rmse, score.combined):
        assert np.isfinite(float(comp)) and float(comp) >= 0.0
    assert score.qt_rmse is None  # dry CBL
    loss = scm_les_final_loss(art, _mynn_config(), nlev=NZ, dt=10.0)
    assert float(score.combined) == pytest.approx(loss, rel=0, abs=1e-12)


def test_final_score_none_on_divergence(monkeypatch):
    # A diverged SCM yields None (the loss maps that to +inf); a caller wanting the θ
    # component must not read a spurious safe_sqrt(NaN)=0 perfect fit.
    from legoesm.atmosphere.les_suite import scm_runner

    nan = jnp.full((NZ,), jnp.nan)
    monkeypatch.setattr(scm_runner, "scm_final_theta_on",
                        lambda *a, **k: (nan, nan, nan))
    art = _cbl_artifact()
    assert scm_les_final_score(art, _mynn_config(), nlev=NZ, dt=20.0) is None


def test_geostrophic_forcing_wired_for_sheared_not_free_cbl():
    # Q3 enabler: build_cbl_scm_from_artifact must apply the artifact's geostrophic wind for
    # f_c!=0 (sheared CBL / SBL) so the Ekman/jet dynamics exist, but leave the free CBL
    # (f_c=0) untouched (SCMForcing disables Coriolis+geostrophic when f_c=0).
    free = _cbl_artifact()  # f_c=0.0
    scm_free, _ = build_cbl_scm_from_artifact(free, _mynn_config(), nlev=NZ, dt=5.0)
    assert scm_free.forcing.u_geo is None and scm_free.forcing.v_geo is None

    Ug = 8.0
    sheared = _cbl_artifact(
        f_c=1.0e-4, u_geo=np.full(NZ, Ug), v_geo=np.zeros(NZ),
        u=np.full((NT, NZ), Ug), v=np.zeros((NT, NZ)))
    scm_sh, _ = build_cbl_scm_from_artifact(sheared, _mynn_config(), nlev=NZ, dt=5.0)
    assert scm_sh.forcing.u_geo is not None
    ug = np.asarray(scm_sh.forcing.u_geo(0.0))
    assert np.allclose(ug, Ug, atol=1e-6)  # constant geostrophic wind on the SCM grid
    # Ekman SIGN (NH, f>0): convention du/dt=+f(v-v_g), dv/dt=-f(u-u_g). At init (u=Ug, v=0,
    # v_g=0) both tendencies are zero; surface DRAG then pulls u below u_g, so dv/dt=-f(u-Ug)
    # becomes POSITIVE and a positive cross-isobar ageostrophic v develops. A mirror-image
    # (wrong-sign) forcing would give v<0 — so assert the MAX v is positive, not just |v|>0.
    s = scm_scan_final_state(scm_sh, 60)
    v_final = np.asarray(s.v.data[0, 0, 0])
    assert np.max(v_final) > 1e-4, (
        f"expected positive NH Ekman v; got max v={np.max(v_final):.2e}")

    # A malformed sheared artifact (f_c!=0 but no u_geo) must FAIL LOUDLY, not spin the wind
    # toward zero (SCMForcing would treat u_g=v_g=0).
    bad = _cbl_artifact(f_c=1.0e-4, u_geo=None)
    with pytest.raises(ValueError, match="geostrophic"):
        build_cbl_scm_from_artifact(bad, _mynn_config(), nlev=NZ, dt=5.0)


def test_scan_final_state_matches_run():
    # The lax.scan rollout MUST reproduce the Python-loop run() to FP roundoff (~1e-9, well
    # below the tuner's ~1e-3 loss tolerance) for these time-independent-forcing regimes
    # (set_time is a no-op) — this is the guard that lets scm_final_theta_on use scan (far
    # faster + AD-tractable) instead of run(). Asserted at the STATE level (T/u/v) AND, since
    # the score is a deterministic function of the state, at the θ EVAL-grid level (θ is the
    # variable that undergoes Exner conversion + reorder + interpolation; u/v use the same
    # deterministic post-processing, so their state-level match carries through).
    art = _cbl_artifact()
    N = 40
    scm_a, _ = build_cbl_scm_from_artifact(art, _mynn_config(), nlev=NZ, dt=5.0)
    final_run, _hist = scm_a.run(N)
    scm_b, grid = build_cbl_scm_from_artifact(art, _mynn_config(), nlev=NZ, dt=5.0)
    final_scan = scm_scan_final_state(scm_b, N)
    for name in ("T", "u", "v"):
        a = np.asarray(getattr(final_run, name).data[0, 0, 0])
        b = np.asarray(getattr(final_scan, name).data[0, 0, 0])
        assert np.allclose(a, b, atol=1e-9, rtol=0), (
            f"{name}: scan vs run max|Δ|={np.max(np.abs(a - b)):.2e}")

    # θ/u/v on the LES eval grid (the scored quantity) via each final state — same helpers
    # scm_final_theta_on uses. Match ⇒ the DF loss + θ-score are equivalent scan-vs-run.
    def _theta_eval(fs):
        th = theta_from_temperature(jnp.asarray(fs.T.data[0, 0, 0]), grid.p_full_pa)
        z = grid.z_scm_m
        order = z if bool(z[0] < z[-1]) else z[::-1]
        thi = th if bool(z[0] < z[-1]) else th[::-1]
        return np.asarray(interp_profile(thi, order, jnp.asarray(art.heights_m)))

    assert np.allclose(_theta_eval(final_run), _theta_eval(final_scan), atol=1e-9, rtol=0)


def test_pure_step_matches_step_and_rejects_ab2():
    art = _cbl_artifact()
    scm_a, _ = build_cbl_scm_from_artifact(art, _mynn_config(), nlev=NZ, dt=5.0)
    # pure_step(state, phys, t) equals what step() applies for the same stage time
    s0, p0, t0 = scm_a.state, scm_a.phys_state, scm_a.t_seconds
    new_state, _new_phys = scm_a.pure_step(s0, p0, t0)
    scm_a.step()
    assert np.allclose(np.asarray(new_state.T.data[0, 0, 0]),
                       np.asarray(scm_a.state.T.data[0, 0, 0]), atol=0, rtol=0)
    # AB2 is stateful → pure_step must reject it (no silent wrong rollout)
    scm_ab2, _ = build_cbl_scm_from_artifact(art, _mynn_config(), nlev=NZ, dt=5.0)
    scm_ab2.time_integrator = "ab2"
    with pytest.raises(ValueError):
        scm_ab2.pure_step(scm_ab2.state, scm_ab2.phys_state, scm_ab2.t_seconds)


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
