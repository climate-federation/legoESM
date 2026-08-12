"""Post-step hard-saturation drain in ``run_rcemip_plane.py``.

The drain is what holds the warm phase ON the saturation curve for EVERY
microphysics scheme (the in-scheme adjustment differs per scheme and does not
exist at all for sundqvist). Pin the three properties that make it usable as a
scheme-independent correction: it removes warm super-saturation, it conserves
moist enthalpy, and it leaves cold (ice-supersaturated) cells alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in (str(REPO), str(REPO / "scripts" / "run")):
    if p not in sys.path:
        sys.path.insert(0, p)

jax.config.update("jax_enable_x64", True)

from legoesm import constants  # noqa: E402
from legoesm.thermo import relative_humidity  # noqa: E402


def _driver():
    import run_rcemip_plane
    return run_rcemip_plane


def _state_and_hc(q_scale, nlev=20):
    """A REAL ``PlaneNonHydrostaticState`` at the Wing reference, with q_v
    scaled to force super-saturation.

    Built through the driver's own IC builder rather than a duck type: Field
    and the state carry different replace APIs (``.replace`` vs ``._replace``),
    and a hand-rolled stub that accepts either hides the mismatch until the
    first jit call in a real run — which is exactly what happened once.
    """
    m = _driver()
    hc = m.create_height_coordinate(
        nlev, H=33_000.0, p_sfc=101_480.0,
        theta_ref_fn=lambda z: m._rcemip_theta_profile(z, T_sfc=300.0),
    )
    grid = m.create_plane_grid(nx=4, ny=4, nlev=nlev, dx=4000.0, dy=4000.0)
    state = m._build_rcemip_initial_state(
        grid, hc, theta_noise_amp=0.0, n_tracers=3,
    )
    tr = state.tracers.data.at[..., 0].multiply(q_scale)
    return state._replace(tracers=state.tracers.replace(data=tr)), hc


def _diagnose(state, hc):
    T = (hc.theta_ref + state.theta_prime.data) * hc.exner_ref
    p = constants.p_ref * hc.exner_ref ** (1.0 / constants.kappa)
    q_v = state.tracers.data[..., 0]
    return (np.asarray(T), np.asarray(p), np.asarray(q_v),
            np.asarray(relative_humidity(T, p, q_v)))


def test_drain_removes_warm_supersaturation():
    """Judged on the LIVE EOS — the state the drain now targets (#1560).

    (The probe IC is the rest state, so live and reference agree here; using
    the live diagnostic anyway keeps this test measuring the curve the drain
    actually aims at if the fixture is ever perturbed.)
    """
    m = _driver()
    state, hc = _state_and_hc(q_scale=1.6)
    T0, _, _, rh0 = _diagnose_live(state, hc)
    warm = T0 >= constants.T_freeze
    assert rh0[warm].max() > 1.2, "probe state is not super-saturated"

    for _ in range(20):  # the rate limiter drains a large pool over many steps
        state = m._apply_hard_saturation_drain(state, hc, 6.0, 1.0)
    T1, _, _, rh1 = _diagnose_live(state, hc)
    warm1 = T1 >= constants.T_freeze
    assert rh1[warm1].max() < 1.01, (
        f"warm cells still super-saturated after the drain: {rh1[warm1].max():.4f}"
    )


def test_drain_conserves_moist_enthalpy_and_total_water():
    """In the REFERENCE coordinate, which is where the increment is defined.

    Both sides use ``exner_ref``, the same Exner the increment divides by, so
    this pins the design intent (``c_pd dT_ref = L_v dq`` exactly) and NOT a
    live-state conservation — see
    ``test_live_enthalpy_residual_is_the_documented_factor_not_zero`` for what
    the live enthalpy actually does.
    """
    m = _driver()
    state, hc = _state_and_hc(q_scale=1.6)

    def _budget(s):
        T = (hc.theta_ref + s.theta_prime.data) * hc.exner_ref
        q_v = s.tracers.data[..., 0]
        q_c = s.tracers.data[..., 1]
        return (np.asarray(constants.c_pd * T + constants.L_v * q_v),
                np.asarray(q_v + q_c))

    h0, w0 = _budget(state)
    state = m._apply_hard_saturation_drain(state, hc, 6.0, 1.0)
    h1, w1 = _budget(state)
    np.testing.assert_allclose(h1, h0, rtol=1e-10)
    np.testing.assert_allclose(w1, w0, rtol=1e-12)


def test_drain_leaves_cold_ice_supersaturated_cells_alone():
    """Ice super-saturation is physical — draining it onto the LIQUID curve
    would manufacture cloud water at 230 K."""
    m = _driver()
    state, hc = _state_and_hc(q_scale=1.6)
    T0, _, q_v0, _ = _diagnose(state, hc)
    cold = T0 < constants.T_freeze
    assert cold.any(), "probe column has no sub-freezing cells"

    state = m._apply_hard_saturation_drain(state, hc, 6.0, 1.0)
    _, _, q_v1, _ = _diagnose(state, hc)
    np.testing.assert_array_equal(q_v1[cold], q_v0[cold])


def test_drain_is_a_noop_on_a_subsaturated_column():
    m = _driver()
    state, hc = _state_and_hc(q_scale=1.0)  # the Wing IC: ~83 % RH
    _, _, q_v0, rh0 = _diagnose(state, hc)
    assert rh0.max() < 1.0
    state = m._apply_hard_saturation_drain(state, hc, 6.0, 1.0)
    _, _, q_v1, _ = _diagnose(state, hc)
    np.testing.assert_array_equal(q_v1, q_v0)


# --- Live-EOS saturation judgement (#1560) ---------------------------------
# Every test above runs on a state with rho' = 0, where the live EOS and the
# reference profile agree exactly — so none of them can see which one the drain
# judges on. These two supply the perturbation that separates them.

def _diagnose_live(state, hc):
    """(T, p, q_v, RH) from the LIVE EOS — what the microphysics scheme and
    refresh_fn's diagnostic see, via the same two shared thermodynamics
    functions they call."""
    from legoesm.atmosphere.physics.thermodynamics import (
        pressure_from_eos, sanitize_theta_rho,
    )
    theta_total, rho_total = sanitize_theta_rho(
        hc.theta_ref + state.theta_prime.data,
        hc.rho_ref + state.rho_prime.data,
    )
    p = pressure_from_eos(rho_total, theta_total)
    T = theta_total * (p / constants.p_ref) ** constants.kappa
    q_v = state.tracers.data[..., 0]
    return (np.asarray(T), np.asarray(p), np.asarray(q_v),
            np.asarray(relative_humidity(T, p, q_v)))


def _compressed(state, hc, frac):
    """rho' = frac * rho_ref at every cell; theta' untouched."""
    rho_p = jnp.broadcast_to(
        frac * hc.rho_ref, state.rho_prime.data.shape).astype(
            state.rho_prime.data.dtype)
    return state._replace(rho_prime=state.rho_prime.replace(data=rho_p))


def test_drain_does_not_fire_on_a_cell_only_the_reference_calls_saturated():
    """The #1560 failure, in one cell.

    A warm COMPRESSED cell (rho' = 0.05 rho_ref, theta' = 0) reads super-
    saturated against the reference profile and sub-saturated on its live
    state: compression raises both T and p, and the Clausius-Clapeyron rise in
    q_sat beats the 1/p fall. Draining it would move vapour to cloud water and
    deposit latent heat in a cell that is not saturated at all — condensate out
    of nothing, with no error.
    """
    m = _driver()
    state, hc = _state_and_hc(q_scale=1.35)
    state = _compressed(state, hc, 0.05)

    T_ref, _, q_v0, rh_ref = _diagnose(state, hc)
    T_live, _, _, rh_live = _diagnose_live(state, hc)
    warm = (T_ref >= constants.T_freeze) & (T_live >= constants.T_freeze)
    # The cell the two disagree about must EXIST, or this test proves nothing.
    split = warm & (rh_ref > 1.05) & (rh_live < 1.0)
    assert split.any(), (
        f"probe has no reference-saturated / live-subsaturated cell: "
        f"max rh_ref={rh_ref[warm].max():.3f}, "
        f"max rh_live={rh_live[warm].max():.3f}")

    state = m._apply_hard_saturation_drain(state, hc, 6.0, 1.05)
    _, _, q_v1, _ = _diagnose_live(state, hc)
    np.testing.assert_array_equal(q_v1[split], q_v0[split])


def test_drain_still_fires_when_the_live_state_is_saturated():
    """The other half of the control: the test above must not be passing
    merely because the drain has stopped working. Same super-saturation, no
    compression, so live and reference agree — and the drain fires."""
    m = _driver()
    state, hc = _state_and_hc(q_scale=1.35)
    T_live, _, q_v0, rh_live = _diagnose_live(state, hc)
    hot = (T_live >= constants.T_freeze) & (rh_live > 1.05)
    assert hot.any(), "probe is not live-super-saturated anywhere warm"

    state = m._apply_hard_saturation_drain(state, hc, 6.0, 1.05)
    _, _, q_v1, _ = _diagnose_live(state, hc)
    assert (q_v1[hot] < q_v0[hot]).all()


def test_drain_conservation_probe_sees_the_exchange_and_no_water_creation():
    """``drain_conservation_probe`` is the instrument the run log carries, so
    it gets its own check: total water must not move across the drain."""
    m = _driver()
    state, hc = _state_and_hc(q_scale=1.35)
    qvc_before, h_before, qv_before = m.drain_conservation_probe(state, hc)
    drained = m._apply_hard_saturation_drain(state, hc, 6.0, 1.05)
    qvc_after, h_after, qv_after = m.drain_conservation_probe(drained, hc)

    assert drained is not state
    # Pure vapour -> cloud exchange: the exchanged pair is untouched.
    np.testing.assert_allclose(qvc_after, qvc_before, rtol=1e-12)
    # And the drain actually did something, or the check above is vacuous.
    assert h_after > h_before, "no latent heating — probe state did not drain"
    # The probe's third return is what makes the printed dh interpretable:
    # without dq there is no ratio to judge dh against.
    assert qv_after < qv_before

    # PIN the ratio the run log actually prints. It is R_d/c_v = 0.4, NOT the
    # 1.4 of the temperature ratio: h = c_pd*T + L_v*q_v, so condensing dq
    # raises c_pd*T by 1.4*L_v*dq while dropping L_v*q_v by L_v*dq. Getting
    # this backwards in the log would send a reader hunting a defect that is
    # not there.
    dq = qv_before - qv_after
    ratio = (h_after - h_before) / (constants.L_v * dq)
    np.testing.assert_allclose(ratio, constants.R_d / constants.c_vd,
                               rtol=5e-3)


def test_live_enthalpy_residual_is_the_documented_factor_not_zero():
    """PIN the residual the docstring claims, rather than asserting a
    conservation the increment does not have.

    ``theta' += (L_v/c_pd) dq / exner_ref`` closes ``c_pd dT = L_v dq`` in the
    REFERENCE coordinate, which is what the sibling conservation test above
    measures — with ``exner_ref`` on both sides, that test is a tautology and
    cannot see this. Against the LIVE state the increment over-heats by
    ``1 + R_d/c_v`` because the live Exner depends on the theta being
    incremented. Pinning the number keeps the docstring honest and turns any
    future change to the heating into a red test rather than a silent retune.

    This is the TEMPERATURE ratio ``c_pd dT_live / (L_v dq)`` = 1.4. The moist
    ENTHALPY ratio the run log prints is ``0.4`` — one less, because
    ``c_pd T + L_v q_v`` also loses ``L_v dq`` as the vapour condenses — and it
    is pinned separately in
    ``test_drain_conservation_probe_sees_the_exchange_and_no_water_creation``.

    Measured AT THE REFERENCE STATE (``rho' = theta' = 0``), where the general
    factor ``(1 + R_d/c_v) * exner_live/exner_ref`` collapses to its leading
    term. Away from it the ratio drifts by the Exner ratio (~2 % at
    ``rho' = 5 %``), which is why this fixture must not be perturbed.
    """
    m = _driver()
    state, hc = _state_and_hc(q_scale=1.35)
    # The probe IC is the rest state: assert that, so a future fixture change
    # that adds a density perturbation fails here instead of silently loosening
    # the number this test exists to pin.
    assert float(np.max(np.abs(state.rho_prime.data))) == 0.0
    T0, _, q_v0, _ = _diagnose_live(state, hc)
    drained = m._apply_hard_saturation_drain(state, hc, 6.0, 1.05)
    T1, _, q_v1, _ = _diagnose_live(drained, hc)

    dq = q_v0 - q_v1
    fired = dq > 1.0e-12
    assert fired.any(), "probe state did not drain — nothing to measure"
    ratio = (constants.c_pd * (T1 - T0)[fired]) / (constants.L_v * dq[fired])
    predicted = 1.0 + constants.R_d / constants.c_vd
    np.testing.assert_allclose(ratio, predicted, rtol=2e-3)
    # It is emphatically NOT 1: an increment that conserved the live moist
    # enthalpy would give exactly 1, and the gap is the ~40 % this driver
    # deliberately leaves to the RCE lane owner (#1560).
    assert float(np.min(ratio)) > 1.3
