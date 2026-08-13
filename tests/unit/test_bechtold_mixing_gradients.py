"""A gradient reaches the deep plume-mixing rates, and it is the right gradient.

The rates were wired to the drivers so the campaign could sweep them by hand.
The next step is training them, and a parameter that the collector happily
hands to an optimiser while carrying a zero or wrong gradient is the failure
mode that produces a confident, useless tuning run: the loss barely moves and
the result gets blamed on physics.

So this asserts three things, in order of how quietly they fail:

1. the gradient is FINITE and NOT NEGLIGIBLE -- not silently killed by a
   stop-gradient, a frozen leaf or a dead branch;
2. it AGREES with a central finite difference, which is the only check that
   catches a gradient that flows but is wrong;
3. its SIGN matches the behaviour measured in the campaign: more entrainment
   dilutes the plumes and cuts convective rain in a single column.

Run under float64: a central difference of a step-size 1e-4 perturbation on a
1.75e-3 parameter is not resolvable in float32.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.convection.bechtold import (  # noqa: E402
    bechtold_convection,
)
from legoesm.atmosphere.physics.convection.config import (  # noqa: E402
    BechtoldConfig,
)

EPS_DEEP_DEFAULT = 1.75e-3
DELTA_DEEP_DEFAULT = 0.75e-4


def _column(ncol=1, nlev=24, p_s=1.0e5, p_top=1.0e4, T_sfc=301.0,
            lapse_rate=6.5, q_sfc=0.018):
    """A single warm, moist, conditionally unstable tropical sounding."""
    p_full = jnp.linspace(p_top, p_s, nlev)[None, :]
    p_full = jnp.broadcast_to(p_full, (ncol, nlev))
    p_half_inner = 0.5 * (p_full[:, 1:] + p_full[:, :-1])
    p_half = jnp.concatenate(
        [jnp.full((ncol, 1), p_top * 0.5), p_half_inner,
         jnp.full((ncol, 1), p_s)], axis=1)
    z_full = -8500.0 * jnp.log(p_full / p_s)
    T = jnp.full((ncol,), T_sfc)[:, None] - lapse_rate * 1e-3 * z_full
    q = q_sfc * jnp.exp(-z_full / 3000.0)
    u = jnp.broadcast_to(jnp.linspace(0.0, 25.0, nlev)[None, :], (ncol, nlev))
    v = jnp.zeros_like(u)
    return T, q, p_full, p_half, u, v


def _rain_of(eps_deep, delta_deep):
    """Column convective rain, as a function of the mixing rates.

    Taken as the column-integrated moisture SINK, which is the convective
    rainfall in a column budget.  ``ConvectionOutput`` carries no ``precip``
    field -- it publishes tendencies -- and the in-plume rain source
    ``dq_r_conv_dt`` exists only when the precipitation split is active, so the
    sink is the quantity that is always defined.
    """
    from legoesm import constants

    T, q, pf, ph, u, v = _column()
    cpp = jnp.zeros(T.shape)
    stoch = jnp.zeros((T.shape[0],))
    cfg = BechtoldConfig(epsilon_deep=eps_deep, delta_deep=delta_deep)
    out, _, _ = bechtold_convection(T, q, pf, ph, u, v, cpp, stoch, None,
                                    dt=600.0, config=cfg)
    dp = ph[:, 1:] - ph[:, :-1]
    return -jnp.sum(out.dq_v_dt * dp) / constants.g


def _fd(fn, x, h):
    return (fn(x + h) - fn(x - h)) / (2.0 * h)


@pytest.mark.parametrize("name,x0,h", [
    ("epsilon_deep", EPS_DEEP_DEFAULT, 1.0e-5),
    ("delta_deep", DELTA_DEEP_DEFAULT, 5.0e-7),
])
def test_gradient_flows_and_matches_finite_difference(name, x0, h):
    if name == "epsilon_deep":
        def f(x):
            return _rain_of(x, DELTA_DEEP_DEFAULT)
    else:
        def f(x):
            return _rain_of(EPS_DEEP_DEFAULT, x)

    g = float(jax.grad(f)(x0))
    assert np.isfinite(g), f"{name}: gradient is not finite"

    # "not negligible" measured against the loss's own scale over the sweep
    # this parameter is actually tuned across, not against an absolute number
    scale = float(abs(f(x0)))
    assert abs(g) * x0 > 1e-4 * max(scale, 1e-12), (
        f"{name}: gradient is effectively zero (g={g:.3e}, loss={scale:.3e}) "
        "-- a collector would hand this to an optimiser that could not move it")

    fd = float(_fd(f, x0, h))
    assert np.isfinite(fd)
    ratio = g / fd
    assert 0.8 < ratio < 1.25, (
        f"{name}: analytic gradient {g:.6e} disagrees with the central "
        f"difference {fd:.6e} (ratio {ratio:.3f}) -- it flows but is wrong")


def test_more_entrainment_dries_the_column_as_the_campaign_measured():
    """Sign check against the sweep: the 90-day arms found that raising deep
    entrainment from 1.75e-3 to 3.0e-3 cut column convective rain while
    reorganising the band. A gradient of the opposite sign would mean the
    trained direction fights the measured one."""
    g = float(jax.grad(lambda x: _rain_of(x, DELTA_DEEP_DEFAULT))(EPS_DEEP_DEFAULT))
    lo = float(_rain_of(1.0e-3, DELTA_DEEP_DEFAULT))
    hi = float(_rain_of(3.0e-3, DELTA_DEEP_DEFAULT))
    assert hi < lo, "raising entrainment did not reduce single-column rain"
    assert g < 0.0, f"gradient sign {g:.3e} disagrees with the finite sweep"
