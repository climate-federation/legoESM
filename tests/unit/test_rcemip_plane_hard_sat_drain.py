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
    m = _driver()
    state, hc = _state_and_hc(q_scale=1.6)
    T0, _, _, rh0 = _diagnose(state, hc)
    warm = T0 >= constants.T_freeze
    assert rh0[warm].max() > 1.2, "probe state is not super-saturated"

    for _ in range(20):  # the rate limiter drains a large pool over many steps
        state = m._apply_hard_saturation_drain(state, hc, 6.0, 1.0)
    T1, _, _, rh1 = _diagnose(state, hc)
    warm1 = T1 >= constants.T_freeze
    assert rh1[warm1].max() < 1.01, (
        f"warm cells still super-saturated after the drain: {rh1[warm1].max():.4f}"
    )


def test_drain_conserves_moist_enthalpy_and_total_water():
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
