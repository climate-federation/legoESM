"""Time integrator dispatch for LegoESM."""

import jax

from legoesm.timestepping.ssp_rk3 import ssp_rk3_step, ssp_rk3_step_scan
from legoesm.timestepping.ssp_rk34 import ssp_rk34_step
from legoesm.timestepping.ssp_rk54 import ssp_rk54_step


def _rk4_step(state, tendency_fn, dt):
    """Classical 4th-order Runge-Kutta step.

    Parameters
    ----------
    state : pytree
    tendency_fn : callable
    dt : float

    Returns
    -------
    pytree : state advanced by dt
    """
    k1 = tendency_fn(state)
    s1 = jax.tree.map(lambda s, k: s + 0.5 * dt * k, state, k1)

    k2 = tendency_fn(s1)
    s2 = jax.tree.map(lambda s, k: s + 0.5 * dt * k, state, k2)

    k3 = tendency_fn(s2)
    s3 = jax.tree.map(lambda s, k: s + dt * k, state, k3)

    k4 = tendency_fn(s3)
    state_new = jax.tree.map(
        lambda s, a, b, c, d: s + (dt / 6.0) * (a + 2.0 * b + 2.0 * c + d),
        state, k1, k2, k3, k4,
    )
    return state_new


_INTEGRATORS = {
    "ssp_rk3": ssp_rk3_step, "ssp3": ssp_rk3_step, "rk3": ssp_rk3_step,
    # Task #25: scan-folded SSP-RK3.  Same math as ssp_rk3, but the
    # 3 stages are folded into a single ``lax.scan`` body so XLA
    # optimizes the tendency pipeline ONCE instead of inlining it
    # three times.  At lat-lon C-grid scale this cuts JIT compile
    # time from ~2.5 h to (target) <15 min at 4 ranks.
    "ssp_rk3_scan": ssp_rk3_step_scan,
    "ssp3_scan": ssp_rk3_step_scan,
    "rk3_scan": ssp_rk3_step_scan,
    "ssp_rk34": ssp_rk34_step, "ssp34": ssp_rk34_step, "rk34": ssp_rk34_step,
    "ssp_rk54": ssp_rk54_step, "ssp54": ssp_rk54_step,
    "ssp45": ssp_rk54_step, "rk54": ssp_rk54_step,
    "rk4": _rk4_step, "runge_kutta_4": _rk4_step,
}


def dispatch_integrator(state, tendency_fn, dt, integrator_name):
    """Dispatch to the appropriate time integrator.

    Parameters
    ----------
    state : pytree (NamedTuple)
    tendency_fn : callable, state -> state-shaped tendencies
    dt : float
    integrator_name : str

    Returns
    -------
    new_state : same type as state
    """
    key = integrator_name.lower()
    step_fn = _INTEGRATORS.get(key)
    if step_fn is None:
        raise ValueError(
            f"Unsupported time_integrator={integrator_name!r}. "
            f"Options: {sorted(_INTEGRATORS.keys())}"
        )
    return step_fn(state, tendency_fn, dt)
