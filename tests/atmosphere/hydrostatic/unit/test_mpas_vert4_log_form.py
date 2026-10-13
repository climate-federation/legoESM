"""Log-humidity form of the MPAS vertical del4 tracer filter.

The flux form's fourth difference of q is positive wherever q decays
exponentially with level, so it removes upper-tropospheric water (4.5-8.8 %/day
on ERA5 profiles, 2026-10-13).  The log form filters ln(q + q_eps) through
water-weighted, mass-conserving interface fluxes.  The tests that separate it
from the flux form are: the shifted-exponential invariance, the dry-layer
barrier and the Q/(Q+q_eps) zigzag factor.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
    VERT4_LOG_Q_EPS, vertical_del4_tracer_tendency_flux,
    vertical_del4_tracer_tendency_log)
from legoesm.grids.vertical import make_cam6_l32_levels

NU = 2.0e-6          # production nu_vert4_T [1/s]
DT = 112.5           # production MPAS dt [s]


@pytest.fixture(autouse=True)
def fp64_policy():
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    saved = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(saved)


def _l32_dp(ps=(1.0e5, 9.5e4, 7.0e4)):
    return jnp.asarray(make_cam6_l32_levels().layer_thickness_dp(jnp.asarray(ps)))


def _shifted_exponential(n=32):
    k = np.arange(n, dtype=np.float64)
    return jnp.asarray(2e-2 * np.exp(-0.35 * (n - 1 - k)) - VERT4_LOG_Q_EPS)   # dry at the top


def test_shifted_exponential_profile_has_no_drift():
    """q + q_eps exponential in level index -> zero tendency on real L32 masses;
    the flux form drifts the same profile (control: the forms differ here)."""
    dp = _l32_dp()
    q = jnp.broadcast_to(_shifted_exponential(), dp.shape)
    t_log = vertical_del4_tracer_tendency_log(q, NU, dp)
    t_flux = vertical_del4_tracer_tendency_flux(q, NU, dp)
    rate_flux = float(jnp.max(jnp.abs(t_flux / q)))
    assert rate_flux > 1e-7                                     # > 0.86 %/day somewhere
    assert float(jnp.max(jnp.abs(t_log / q))) < 1e-12 * rate_flux


@pytest.mark.parametrize("zeros", [(), (3,), (10, 11, 25)])
def test_column_water_conserved_on_l32_masses(zeros):
    dp = _l32_dp()
    rng = np.random.default_rng(1)
    q = np.array(_shifted_exponential()) * (1 + 0.3 * rng.standard_normal((3, 32)))
    q[:, list(zeros)] = 0.0
    t = vertical_del4_tracer_tendency_log(jnp.asarray(q), NU, dp)
    closure = jnp.abs((t * dp).sum(-1)) / (jnp.abs(t) * dp).sum(-1)
    assert float(jnp.max(closure)) < 1e-13


def test_dry_layer_is_a_barrier():
    """A layer at q <= 0 neither gains nor loses water and its neighbours do not
    drain into it (the flux form moves water into the same layer)."""
    dp = _l32_dp((1.0e5,))[0]
    q = np.array(_shifted_exponential())
    q[[8, 20]] = 0.0
    t = vertical_del4_tracer_tendency_log(jnp.asarray(q), NU, dp)
    assert float(jnp.max(jnp.abs(t[jnp.asarray([8, 20])]))) == 0.0
    t_flux = vertical_del4_tracer_tendency_flux(jnp.asarray(q), NU, dp)
    assert float(jnp.max(jnp.abs(t_flux[jnp.asarray([8, 20])]))) > 0.0      # control


@pytest.mark.parametrize("Q", [3e-6, 1e-3])
def test_zigzag_damped_at_production_rate(Q):
    """A small 2-level zigzag on L32 masses is damped at the flux form's rate
    times Q/(Q+q_eps), and one production step reduces its amplitude."""
    dp = _l32_dp((1.0e5,))[0]
    k = jnp.arange(32)
    eps = 1e-4
    q = Q * (1 + eps * (-1.0) ** k)
    t = vertical_del4_tracer_tendency_log(q, NU, dp)
    t_flux = vertical_del4_tracer_tendency_flux(q, NU, dp)
    ratio = float(jnp.mean(t[4:-4] / t_flux[4:-4]))
    assert abs(ratio - Q / (Q + VERT4_LOG_Q_EPS)) < 2e-3 * Q / (Q + VERT4_LOG_Q_EPS)
    amp0 = float(jnp.abs(q[4:-4] - Q).mean())
    amp1 = float(jnp.abs(q[4:-4] + DT * t[4:-4] - Q).mean())
    assert amp1 < amp0


def test_jit_parity_and_finite_gradients():
    dp = _l32_dp()
    rng = np.random.default_rng(2)
    q = jnp.asarray(np.asarray(_shifted_exponential()) * (1 + 0.2 * rng.random((3, 32))))
    eager = vertical_del4_tracer_tendency_log(q, NU, dp)
    jitted = jax.jit(vertical_del4_tracer_tendency_log)(q, NU, dp)
    np.testing.assert_allclose(np.asarray(jitted), np.asarray(eager), rtol=1e-12, atol=0)
    qz = q.at[0, 5].set(0.0).at[1, 7].set(q[1, 8])        # a dry cell and a mobility tie
    g = jax.grad(lambda x: jnp.sum(vertical_del4_tracer_tendency_log(x, NU, dp) ** 2))(qz)
    assert bool(jnp.all(jnp.isfinite(g)))


def test_check_grads_second_order():
    """Away from dry cells and mobility ties the operator is smooth."""
    from jax.test_util import check_grads
    dp = _l32_dp((1.0e5,))[0]
    k = np.arange(32, dtype=np.float64)
    q = jnp.asarray(2e-2 * np.exp(-0.3 * (31 - k)) * (1 + 0.05 * np.sin(k)))

    def f(x):
        return vertical_del4_tracer_tendency_log(x, NU, dp) / NU
    check_grads(f, (q,), order=2, modes=("fwd", "rev"), atol=1e-6, rtol=1e-6, eps=1e-9)


def test_flux_form_without_mobility_is_unchanged():
    """The mobility argument defaults to the flux form's own behaviour."""
    dp = _l32_dp()
    q = jnp.broadcast_to(_shifted_exponential(), dp.shape)
    a = vertical_del4_tracer_tendency_flux(q, NU, dp)
    b = vertical_del4_tracer_tendency_flux(q, NU, dp, mobility=jnp.ones(q.shape[:-1] + (31,)))
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_dycore_routes_log_tracers_and_refuses_overlap():
    """The dycore sends ONLY the named tracer through the log form and refuses a
    tracer named in both form lists."""
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import mpas_hydrostatic_tendencies
    from tests.atmosphere.hydrostatic.unit.test_mpas_vert4_flux_form import _tracer_state
    mesh, coord, cfg, state = _tracer_state()
    base = mpas_hydrostatic_tendencies(
        state, mesh, coord, cfg._replace(vert4_flux_form_tracers=("q_v",)), None, 0.0)
    sel = mpas_hydrostatic_tendencies(
        state, mesh, coord, cfg._replace(vert4_log_form_tracers=("q_v",)), None, 0.0)
    qv = state.tracers["q_v"].data
    expect = (vertical_del4_tracer_tendency_log(qv, NU, coord.dsigma)
              - vertical_del4_tracer_tendency_flux(qv, NU, coord.dsigma))
    assert float(jnp.max(jnp.abs(expect))) > 0.0
    np.testing.assert_allclose(
        np.asarray(sel.tracer_tendencies["q_v"].data - base.tracer_tendencies["q_v"].data),
        np.asarray(expect), rtol=1e-9, atol=1e-22)
    np.testing.assert_array_equal(np.asarray(sel.tracer_tendencies["q_c"].data),
                                  np.asarray(base.tracer_tendencies["q_c"].data))
    with pytest.raises(ValueError, match="each tracer takes one filter form"):
        mpas_hydrostatic_tendencies(state, mesh, coord, cfg._replace(
            vert4_flux_form_tracers=("q_v",), vert4_log_form_tracers=("q_v",)), None, 0.0)
    with pytest.raises(ValueError, match="unknown tracer names"):
        mpas_hydrostatic_tendencies(state, mesh, coord, cfg._replace(
            vert4_log_form_tracers=("qv",)), None, 0.0)
