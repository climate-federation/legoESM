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
    liquid_water_theta,
    theta_from_temperature,
)
from legoesm.atmosphere.les_suite.scm_runner import (
    build_cbl_scm_from_artifact,
    scm_final_moist_on,
    scm_final_theta_on,
    scm_les_final_loss,
    scm_les_final_score,
    scm_scan_final_state,
)
from legoesm.atmosphere.physics import TurbulenceConfig
from legoesm.atmosphere.physics._shared import exner_function
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


def _moist_artifact(**over) -> LESReferenceArtifact:
    """A BOMEX-like MOIST artifact (θ_l + q_t + moisture flux + large-scale forcing).

    Coarse + short so the unit test is cheap; the physically-faithful cumulus-forming
    validation is the real-BOMEX-artifact smoke, not this fixture. Carries the full moist
    field set the bridge requires (qt ⇒ wqt_resolved; w_qv_s + qv_adv) plus the Coriolis /
    geostrophic wind (BOMEX f_c=0.376e-4) so build_cbl_scm_from_artifact takes both the
    geostrophic and the moist branch.
    """
    z = np.linspace(10.0, 2500.0, NZ)
    thl_col = 298.5 + np.clip((z - 500.0) / 1000.0, 0.0, None) * 6.0   # θ_l [K]
    qt_col = np.clip(0.0165 - 6.0e-6 * z, 0.003, None)                 # q_t [kg/kg]
    base = dict(
        case_name="bomex_run",
        sgs="lasd",
        heights_m=z,
        times_s=np.linspace(0.0, 1800.0, NT),
        theta=np.broadcast_to(thl_col, (NT, NZ)).copy(),
        u=np.full((NT, NZ), -8.0),
        v=np.zeros((NT, NZ)),
        wtheta_resolved=np.broadcast_to(
            0.008 * np.clip(1.0 - z / 600.0, -0.3, 1.0), (NT, NZ)).copy(),
        wtheta_sgs=np.zeros((NT, NZ)),
        qt=np.broadcast_to(qt_col, (NT, NZ)).copy(),
        wqt_resolved=np.broadcast_to(
            5.0e-5 * np.clip(1.0 - z / 1500.0, -0.2, 1.0), (NT, NZ)).copy(),
        wqt_sgs=np.zeros((NT, NZ)),
        prescribe="fluxes",
        w_theta_s=np.full(NT, 0.008),
        w_qv_s=np.full(NT, 5.2e-5),
        f_c=0.376e-4,
        u_geo=np.full(NZ, -8.0),
        v_geo=np.zeros(NZ),
        subsidence_w=np.full(NZ, -0.005),   # large-scale subsidence [m/s], +up ⇒ downward
        theta_adv=np.full(NZ, -2.0e-5),     # radiative-cooling θ tendency [K/s]
        qv_adv=np.full(NZ, -1.0e-8),        # drying advective tendency [(kg/kg)/s]
    )
    base.update(over)
    return LESReferenceArtifact(**base)


def _saturated_moist_artifact(**over):
    """A stratocumulus-like moist artifact whose IC is SATURATED (cloud at t=0)."""
    z = np.linspace(10.0, 2500.0, NZ)
    thl_col = np.full(NZ, 286.0)                     # cold marine mixed layer [θ_l, K]
    qt_col = np.full(NZ, 0.012)                      # q_t above saturation → cloud
    return _moist_artifact(
        theta=np.broadcast_to(thl_col, (NT, NZ)).copy(),
        qt=np.broadcast_to(qt_col, (NT, NZ)).copy(),
        heights_m=z, **over)


def test_saturated_start_artifact_seeds_qc_and_matches_ic():
    # A cloudy start is saturation-adjusted at init (no longer refused): q_c is
    # seeded and the SCM's initial θ_l / q_t reproduce the artifact exactly
    # (saturation_adjust is the exact inverse of liquid_water_theta).
    art = _saturated_moist_artifact()
    scm, grid = build_cbl_scm_from_artifact(art, _mynn_config(), nlev=NZ, dt=5.0)

    q_c0 = np.asarray(scm.state.tracers["q_c"].data[0, 0, 0])
    q_v0 = np.asarray(scm.state.tracers["q_v"].data[0, 0, 0])
    assert q_c0.max() > 1.0e-4                         # cloud water actually seeded

    p_full = np.asarray(grid.p_full_pa)
    exner0 = np.asarray(exner_function(p_full))
    theta0 = np.asarray(theta_from_temperature(scm.state.T.data[0, 0, 0], p_full))
    thl_scm = np.asarray(liquid_water_theta(theta0, q_c0, exner0))
    qt_scm = q_v0 + q_c0

    z_les = np.asarray(art.heights_m)
    z_scm = np.asarray(grid.z_scm_m)
    thl_ref = np.asarray(interp_profile(np.asarray(art.theta)[0], z_les, z_scm))
    qt_ref = np.asarray(interp_profile(np.asarray(art.qt)[0], z_les, z_scm))
    assert np.allclose(thl_scm, thl_ref, atol=1e-3)   # θ_l reproduced at init
    assert np.allclose(qt_scm, qt_ref, atol=1e-7)     # total water conserved


def test_clear_moist_start_seeds_no_qc():
    # The clear BOMEX-like artifact (q_t < q_sat) forms no cloud at t=0, so q_c
    # is all-zero — the saturation adjustment leaves a cloud-free start unchanged.
    art = _moist_artifact()
    scm, _ = build_cbl_scm_from_artifact(art, _mynn_config(), nlev=NZ, dt=5.0)
    q_c0 = np.asarray(scm.state.tracers["q_c"].data[0, 0, 0])
    assert np.allclose(q_c0, 0.0, atol=1e-12)


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


def _windy_cbl_artifact(**over):
    """A well-mixed dry CBL with a light mean wind — so the bulk surface layer can carry
    the prescribed heat flux (the Q1b diagnostic-flux margin fixture)."""
    z = np.linspace(10.0, 1500.0, NZ)
    theta_col = np.where(z > 800.0, 300.0 + 0.006 * (z - 800.0), 300.0)
    theta = np.broadcast_to(theta_col, (NT, NZ)).copy()
    base = dict(
        case_name="cbl_windy",
        sgs="lasd",
        heights_m=z,
        times_s=np.linspace(0.0, 1800.0, NT),
        theta=theta,
        u=np.full((NT, NZ), 5.0),
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


def test_diagnostic_scheme_flux_nonlocal_beats_local():
    # Q1b measured diagnostic margin: at the well-mixed CBL mean state, driven by the SAME
    # LES surface heat flux, a LOCAL closure (F=-Kh·∂θ/∂z, ∂θ/∂z≈0) carries ~0 interior
    # flux while a NONLOCAL closure's counter-gradient carries the surface flux up. So the
    # nonlocal flux RMSE vs the LES flux is materially BELOW the local one.
    from legoesm.atmosphere.les_suite.bridge import diagnostic_truth
    from legoesm.atmosphere.les_suite.scm_runner import diagnostic_scheme_flux
    from legoesm.atmosphere.les_suite.score import diagnostic_flux_score

    art = _windy_cbl_artifact()
    truth = diagnostic_truth(art)
    rmse = {}
    for sch in ("smagorinsky", "louis", "holtslag_boville", "ysu"):
        flux = diagnostic_scheme_flux(art, TurbulenceConfig(scheme=sch), nlev=32)
        assert flux.shape == truth.heights_m.shape
        assert bool(jnp.all(jnp.isfinite(flux)))
        rmse[sch] = float(diagnostic_flux_score(truth, flux).wtheta_rmse)

    local_worst = max(rmse["smagorinsky"], rmse["louis"])
    nonlocal_best = min(rmse["holtslag_boville"], rmse["ysu"])
    # Local closures carry ≈0 flux through the mixed layer (structural ceiling) ⇒ their
    # RMSE ≈ the LES flux magnitude; nonlocal beats them by a clear margin.
    assert nonlocal_best < local_worst - 0.1


def test_diagnostic_scheme_flux_rejects_moist_and_tke():
    from legoesm.atmosphere.les_suite.scm_runner import diagnostic_scheme_flux

    # Moist artifact → rejected (counter-gradient ceiling is a dry-CBL notion).
    with pytest.raises(ValueError, match="dry-only"):
        diagnostic_scheme_flux(_moist_artifact(), TurbulenceConfig(scheme="smagorinsky"))
    # TKE-carrying scheme → rejected (no wtheta_flux exposure).
    with pytest.raises(ValueError, match="TKE-carrying"):
        diagnostic_scheme_flux(_windy_cbl_artifact(), _mynn_config())


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


def test_moist_scm_builds_scan_safe_with_condensate_tracers():
    # A moist artifact must build an SCM whose tracer registry already carries the full
    # condensate/precip set (incl. q_g), so the lax.scan free-run's carry pytree is FIXED —
    # otherwise microphysics auto-materialises a key mid-scan and the scan raises.
    art = _moist_artifact()
    scm, grid = build_cbl_scm_from_artifact(art, _mynn_config(), nlev=NZ, dt=10.0)
    keys = set(scm.state.tracers)
    assert {"q_v", "q_c", "q_r", "q_i", "q_s", "q_g"} <= keys
    # q_v IC = q_t (q_c≈0 at t0), interpolated onto the SCM grid → physical range
    q_v0 = np.asarray(scm.state.tracers["q_v"].data[0, 0, 0])
    assert np.all(np.isfinite(q_v0)) and q_v0.min() >= 0.0 and q_v0.max() < 0.05
    # the scan itself must not raise on the fixed-pytree requirement
    final = scm_scan_final_state(scm, nsteps=30)
    assert np.all(np.isfinite(np.asarray(final.tracers["q_c"].data[0, 0, 0])))


def test_moist_final_moist_on_returns_thetal_uv_qt():
    # scm_final_moist_on returns (θ_l, u, v, q_t) — four finite profiles on the LES grid.
    art = _moist_artifact()
    scm, grid = build_cbl_scm_from_artifact(art, _mynn_config(), nlev=NZ, dt=10.0)
    z_eval = jnp.asarray(art.heights_m)
    thl, u, v, qt = scm_final_moist_on(scm, grid, nsteps=30, z_eval=z_eval)
    for arr in (thl, u, v, qt):
        assert arr.shape == z_eval.shape and bool(jnp.all(jnp.isfinite(arr)))
    assert float(qt.min()) >= 0.0 and float(qt.max()) < 0.05     # physical total water
    assert 250.0 < float(thl.min()) and float(thl.max()) < 340.0  # physical θ_l


def test_moist_score_includes_qt_dry_does_not():
    # A moist artifact is scored in θ_l + q_t (qt_rmse populated + folded into combined); a
    # dry artifact leaves qt_rmse=None. This is the contrast that proves the moist channel
    # is actually wired into the objective, not silently dropped.
    moist = scm_les_final_score(_moist_artifact(), _mynn_config(), nlev=NZ, dt=10.0)
    assert moist is not None and moist.qt_rmse is not None
    assert bool(jnp.isfinite(moist.qt_rmse)) and float(moist.qt_rmse) >= 0.0
    assert bool(jnp.isfinite(moist.combined))

    dry = scm_les_final_score(_cbl_artifact(), _mynn_config(), nlev=NZ, dt=20.0)
    assert dry is not None and dry.qt_rmse is None


def test_liquid_water_theta_sign_and_exact_decrement():
    # θ_l = θ − (L_v/(c_pd·Π))·q_c: STRICTLY below θ where cloud exists (q_c>0), by exactly
    # the latent-heat decrement. A flipped sign (θ+…) would raise θ_l above θ — this is the
    # guard the broad-bounds test can't catch.
    from legoesm.atmosphere.les_suite.scm_runner import _liquid_water_theta

    from legoesm import constants

    theta = jnp.array([300.0, 305.0, 310.0])
    q_c = jnp.array([0.0, 1.0e-3, 2.0e-3])       # cloud water [kg/kg]
    p = jnp.array([1.0e5, 9.0e4, 8.0e4])
    thl = _liquid_water_theta(theta, q_c, p)
    exner = (p / constants.p_ref) ** constants.kappa
    expect = theta - (constants.L_v / constants.c_pd) * q_c / exner
    assert np.allclose(np.asarray(thl), np.asarray(expect), rtol=1e-12)
    assert float(thl[0]) == float(theta[0])                 # cloud-free ⇒ θ_l = θ
    assert float(thl[1]) < float(theta[1])                  # cloud ⇒ θ_l < θ (NOT above)
    assert float(thl[2]) < float(thl[1] - theta[1] + theta[2])  # bigger q_c ⇒ bigger drop


def test_moist_combined_folds_in_qt():
    # The moist combined score must be the RMS over (θ_l, u, v, q_t) — so a q_t-only error
    # moves `combined` even when θ_l/u/v match. Feed truth=SCM for θ_l/u/v and a controlled
    # q_t error; combined = sqrt((0+0+0+qt_rmse²)/4) = qt_rmse/2.
    from legoesm.atmosphere.les_suite.bridge import LESTruth
    from legoesm.atmosphere.les_suite.score import prognostic_profile_score

    z = jnp.linspace(10.0, 1500.0, NZ)
    theta = 300.0 + 0.003 * z
    qt_truth = 0.010 + jnp.zeros_like(z)
    truth = LESTruth(case_name="m", heights_m=z, times_s=jnp.array([1800.0]),
                     theta=theta, u=jnp.zeros_like(z), v=jnp.zeros_like(z),
                     wtheta=jnp.zeros_like(z), qt=qt_truth)
    # θ_l/u/v exact ⇒ their rmse 0; q_t offset ⇒ only qt_rmse nonzero
    sc = prognostic_profile_score(truth, theta, jnp.zeros_like(z), jnp.zeros_like(z),
                                  scm_qt=qt_truth + 5.0e-4)
    assert float(sc.theta_rmse) < 1e-9 and sc.qt_rmse is not None
    assert np.isclose(float(sc.combined), 0.5 * float(sc.qt_rmse), rtol=1e-6)


def test_moist_scm_les_loss_jax_differentiable():
    # The AD path (scm_les_loss_jax) must handle a MOIST artifact — final_prognostic_truth
    # now carries qt, so the score REQUIRES scm_qt; a dry reducer here raises. jax.grad must
    # return a finite gradient w.r.t. a traced closure param (the D4 moist-AD enabler).
    import jax
    from legoesm.atmosphere.les_suite import scm_runner as scmr

    art = _moist_artifact()

    def loss(cd):
        surf = SurfaceLayerConfig(z0=0.1, Cd_neutral=cd, Ch_neutral=0.0)
        turb = TurbulenceConfig(scheme="mynn25", mynn25=MYNN25Config(surface=surf))
        return scmr.scm_les_loss_jax(art, turb, nlev=NZ, dt=20.0)

    val = float(loss(jnp.asarray(1.5e-3)))
    assert np.isfinite(val) and val >= 0.0
    g = float(jax.grad(loss)(jnp.asarray(1.5e-3)))
    assert np.isfinite(g)


def test_partly_cloudy_reference_is_initialised_from_its_recorded_cloud_water():
    """Partial cloud cover hides from a saturation test — the recorded channel does not.

    A horizontally averaged column with partial cover carries cloud while sitting
    BELOW saturation in the MEAN, so a grid-mean saturation adjustment would miss it
    (it would return q_c=0 and start the column clear). When the reference records
    its cloud-water channel, the runner seeds the column from that recorded q_c
    DIRECTLY — the exact cloud water it carried — instead of re-deriving it from a
    (here misleading) mean-saturation test. Without the channel, the saturation
    fallback sees a subsaturated mean and starts clear (the known legacy limitation).
    """
    from legoesm.atmosphere.les_suite.bridge import LESReferenceArtifact
    from legoesm.atmosphere.les_suite.scm_runner import build_cbl_scm_from_artifact

    nz, nt = 12, 2
    z = jnp.linspace(0.0, 1500.0, nz)
    theta = jnp.broadcast_to(jnp.full((nz,), 290.0), (nt, nz))
    zero = jnp.zeros((nt, nz))
    # comfortably sub-saturated total water, so a mean-saturation test cannot fire
    qt = jnp.broadcast_to(jnp.full((nz,), 2.0e-3), (nt, nz))
    qc = zero.at[0, 5].set(2.0e-4)          # 0.2 g/kg of cloud at t=0

    def _artifact(with_qc):
        return LESReferenceArtifact(
            case_name="partly_cloudy", sgs="smagorinsky",
            heights_m=z, times_s=jnp.asarray([0.0, 3600.0]),
            theta=theta, u=zero, v=zero,
            wtheta_resolved=zero, wtheta_sgs=zero,
            qt=qt, wqt_resolved=zero, wqt_sgs=zero,
            qc=(qc if with_qc else None),
            prescribe="fluxes",
            w_theta_s=jnp.zeros(nt), w_qv_s=jnp.zeros(nt),
        )

    # Without the channel the mean-saturation fallback sees nothing → clear start.
    scm_no, _ = build_cbl_scm_from_artifact(_artifact(with_qc=False), _mynn_config())
    assert np.allclose(np.asarray(scm_no.state.tracers["q_c"].data[0, 0, 0]), 0.0)

    # With the channel the recorded cloud water is SEEDED (not missed, not refused).
    scm_yes, _ = build_cbl_scm_from_artifact(_artifact(with_qc=True), _mynn_config())
    q_c0 = np.asarray(scm_yes.state.tracers["q_c"].data[0, 0, 0])
    assert q_c0.max() > 1.0e-5              # the partly-cloud q_c is present at init


def test_a_saturation_adjustment_that_did_not_settle_is_refused():
    """The fallback initialisation contracts over some states and not others.

    The adjustment is a damped fixed point with a fixed trip count. Over this
    suite's boundary-layer columns it settles exactly; measured outside that
    range it does not -- a 300 K column at 25 g/kg comes back nearly a gram
    per kilogram from its own saturation -- and nothing in the iteration says
    so, which is the dangerous kind of wrong: a plausible number rather than
    an error. The runner checks the residual and refuses.
    """
    import jax.numpy as jnp
    import pytest

    from legoesm import constants
    from legoesm.atmosphere.les_suite.bridge import LESReferenceArtifact
    from legoesm.atmosphere.les_suite.scm_coupling import saturation_adjust
    from legoesm.atmosphere.les_suite.scm_runner import (
        build_cbl_scm_from_artifact,
    )
    from legoesm.thermo import saturation_mixing_ratio

    # First, the measurement the guard rests on, stated as an assertion so it
    # is re-checked rather than trusted: the iteration settles for a
    # stratocumulus column and does NOT for a hot, very moist one.
    p = jnp.full((4,), 1.0e5)
    exner = (p / constants.p_ref) ** constants.kappa

    def _residual(theta_l, q_t):
        th, qv, qc = saturation_adjust(
            jnp.full((4,), theta_l), jnp.full((4,), q_t), exner, p)
        q_sat = saturation_mixing_ratio(th * exner, p)
        return float(jnp.max(jnp.abs(jnp.where(qc > 0.0, qv - q_sat, 0.0))))

    # single precision: the settled residual is round-off (~1e-7 kg/kg), the
    # unsettled one is four orders larger.
    assert _residual(288.0, 12.0e-3) < 1e-6        # in range: settles
    assert _residual(300.0, 25.0e-3) > 1e-4        # out of range: does not

    # And now the guard, driven through the runner with no cloud-water channel
    # so the fallback is the path taken.
    nz, nt = 8, 2
    z = jnp.linspace(0.0, 800.0, nz)
    zero = jnp.zeros((nt, nz))
    art = LESReferenceArtifact(
        case_name="too_warm_and_wet", sgs="smagorinsky",
        heights_m=z, times_s=jnp.asarray([0.0, 3600.0]),
        theta=jnp.broadcast_to(jnp.full((nz,), 300.0), (nt, nz)),
        u=zero, v=zero, wtheta_resolved=zero, wtheta_sgs=zero,
        qt=jnp.broadcast_to(jnp.full((nz,), 25.0e-3), (nt, nz)),
        wqt_resolved=zero, wqt_sgs=zero,
        qc=None,                       # forces the iterative fallback
        prescribe="fluxes",
        w_theta_s=jnp.zeros(nt), w_qv_s=jnp.zeros(nt),
    )
    with pytest.raises(ValueError, match="did not settle"):
        build_cbl_scm_from_artifact(art, _mynn_config(), nlev=16, dt=10.0)
