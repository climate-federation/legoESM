"""Time integrator dispatch for LegoESM."""

from legoesm.timestepping.ssp_rk3 import ssp_rk3_step
from legoesm.timestepping.ssp_rk34 import ssp_rk34_step
from legoesm.timestepping.ssp_rk54 import ssp_rk54_step

_INTEGRATORS = {
    "ssp_rk3": ssp_rk3_step, "ssp3": ssp_rk3_step, "rk3": ssp_rk3_step,
    "ssp_rk34": ssp_rk34_step, "ssp34": ssp_rk34_step, "rk34": ssp_rk34_step,
    "ssp_rk54": ssp_rk54_step, "ssp54": ssp_rk54_step,
    "ssp45": ssp_rk54_step, "rk54": ssp_rk54_step,
}


def dispatch_integrator(state, tendency_fn, dt, integrator_name):
    """Dispatch to the appropriate SSP-RK integrator.

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
