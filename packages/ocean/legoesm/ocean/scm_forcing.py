"""External surface forcing API for the ocean single-column model (SCM).

Closure-style time-dependent surface forcing that mirrors the atmosphere
SCM's :class:`legoesm.atmosphere.forcing.scm.scm_forcing.SCMForcing`: each physical
channel is a callable ``f(t_seconds) -> scalar`` (``None`` disables it),
and the whole object is registered as a **flat-static** pytree so JAX
never treats the Python callables as array leaves.

Channels
--------
``f_c``
    Coriolis parameter [1/s] (``2 Ω sinφ``).  ``0.0`` disables the
    Coriolis turning that, combined with wind stress, produces the
    surface Ekman spiral.
``tau_x``, ``tau_y``
    Surface wind-stress components [Pa].
``q_net``
    Net surface heat flux [W/m²], **positive into the ocean**.  Applied
    in full to the top model layer (no shortwave penetration unless the
    model is configured with a shortwave-penetration scheme, in which
    case the caller is responsible for the non-solar / solar split).
``e_minus_p``
    Evaporation minus precipitation [m/s], positive = net evaporation =
    saltier surface.  Enters as a virtual salt flux on the top layer.
``sw_down``
    Downwelling shortwave at the surface [W/m²].  Only consumed when the
    model enables the shortwave-penetration scheme; otherwise ``q_net``
    already carries the full heating.

Reuse
-----
The top-layer flux→tendency conversion (wind stress → ``du/dt``, heat
flux → ``dT/dt``, virtual salt → ``dS/dt``) is delegated to the existing
:func:`legoesm.ocean.physics.surface_forcing.prescribed.prescribed_surface_forcing`
by rebuilding a cheap :class:`PrescribedForcingConfig` per step — no
numerics are re-derived here.  KPP / shortwave-penetration consume an
:class:`legoesm.ocean.state.OceanSurfaceForcing` struct built by
:func:`build_ocean_surface_forcing_struct` from the *same* callables, so
the boundary-layer friction velocity, surface buoyancy flux, and the
explicitly-applied top-layer fluxes all share one consistent source.
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.ocean.state import (
    OceanState,
    OceanSurfaceForcing,
    OceanTendencies,
)
from legoesm.ocean.vertical import compute_ocean_jacobian
from legoesm.ocean.physics.surface_forcing.config import PrescribedForcingConfig
from legoesm.ocean.physics.surface_forcing.prescribed import (
    prescribed_surface_forcing,
)
from legoesm.ocean.physics.tendencies import wrap_ocean_tendencies


ScalarFn = Callable[[float], jax.Array]


class OceanSCMForcing(NamedTuple):
    """Time-dependent surface forcing for the ocean SCM.

    See the module docstring for channel definitions and sign
    conventions.  Every field is either ``None`` (channel disabled) or a
    callable of simulation time in seconds; ``f_c`` is a static scalar.
    """
    f_c: float = 0.0
    tau_x: Optional[ScalarFn] = None
    tau_y: Optional[ScalarFn] = None
    q_net: Optional[ScalarFn] = None       # [W/m^2], + into ocean
    e_minus_p: Optional[ScalarFn] = None   # [m/s], E - P (virtual salt)
    sw_down: Optional[ScalarFn] = None     # [W/m^2] downwelling shortwave


def default_ocean_forcing() -> OceanSCMForcing:
    """All-zero forcing (no wind, no flux, no Coriolis)."""
    return OceanSCMForcing()


# Register OceanSCMForcing as a flat-static pytree so JAX never treats the
# Python callables as array leaves (mirrors SCMForcing).  All fields live
# in the auxiliary-data slot; the dynamic-leaf list is empty.
def _flatten_ocean_scm_forcing(forcing: "OceanSCMForcing"):
    return (), forcing


def _unflatten_ocean_scm_forcing(aux_data, _children):
    return aux_data


jax.tree_util.register_pytree_node(
    OceanSCMForcing, _flatten_ocean_scm_forcing, _unflatten_ocean_scm_forcing,
)


def validate_ocean_forcing(forcing: OceanSCMForcing) -> None:
    """Raise on a malformed forcing object.

    State-blind checks only: ``f_c`` must be a finite scalar and every
    non-``None`` channel must be callable.  Catches the common mistake of
    passing a constant array where a ``f(t)`` callable is expected (which
    would raise an opaque ``TypeError`` deep inside the step loop).
    """
    import math

    if not math.isfinite(float(forcing.f_c)):
        raise ValueError(
            f"OceanSCMForcing.f_c must be finite, got {forcing.f_c!r}."
        )
    for name in ("tau_x", "tau_y", "q_net", "e_minus_p", "sw_down"):
        fn = getattr(forcing, name)
        if fn is not None and not callable(fn):
            raise ValueError(
                f"OceanSCMForcing.{name} must be a callable f(t)->scalar "
                f"or None, got {type(fn).__name__}. Wrap constants in a "
                f"lambda, e.g. {name}=lambda t: 0.1."
            )


def _eval_scalar(fn: Optional[ScalarFn], t: float) -> float:
    """Evaluate a forcing channel to a Python float (0.0 when disabled).

    Coercing to ``float`` keeps the result host-side and concrete so the
    per-step :class:`PrescribedForcingConfig` it feeds compares cleanly
    against ``0.0`` (the prescribed module uses a Python ``if`` on
    ``E_minus_P``).  The SCM driver is not ``jit``-traced, so every
    callable returns a concrete value.
    """
    if fn is None:
        return 0.0
    return float(fn(t))


def build_ocean_surface_forcing_struct(
    forcing: OceanSCMForcing,
    t: float,
    horiz_shape: tuple[int, ...],
    dtype,
) -> OceanSurfaceForcing:
    """Build an :class:`OceanSurfaceForcing` struct for KPP / shortwave.

    All fields are broadcast to the column's horizontal shape so the
    vertical-mixing integration's ``getattr(surface_forcing, ...)`` reads
    see correctly-shaped arrays.  ``freshwater`` [kg/m²/s, **positive into
    the ocean**] is converted from ``e_minus_p`` [m/s, positive =
    evaporation = water leaving] via ``freshwater = -e_minus_p · ρ_water``
    so a net-evaporation forcing drives a brine-rejection (destabilizing)
    surface buoyancy flux in KPP, consistent with the virtual salt flux
    applied to the top layer.
    """
    def _arr(value: float) -> jax.Array:
        return jnp.full(horiz_shape, value, dtype=dtype)

    tau_x = _arr(_eval_scalar(forcing.tau_x, t))
    tau_y = _arr(_eval_scalar(forcing.tau_y, t))
    q_net = _arr(_eval_scalar(forcing.q_net, t))
    fw = _arr(-_eval_scalar(forcing.e_minus_p, t) * constants.rho_water)
    sw_down = (
        None if forcing.sw_down is None
        else _arr(_eval_scalar(forcing.sw_down, t))
    )
    return OceanSurfaceForcing(
        sw_down=sw_down,
        q_net=q_net,
        tau_x=tau_x,
        tau_y=tau_y,
        freshwater=fw,
    )


def compute_ocean_forcing_tendencies(
    state: OceanState,
    z_coord,
    grid,
    forcing: OceanSCMForcing,
    t: float,
) -> OceanTendencies:
    """Assemble the explicit surface-forcing tendency at time ``t``.

    The top-layer wind-stress / heat / virtual-salt fluxes are computed by
    :func:`prescribed_surface_forcing` (reused verbatim, fed a per-step
    :class:`PrescribedForcingConfig`).  ``K_v`` / ``A_v`` are left ``None``
    — diffusivities come from the physics function, not forcing.

    Planetary Coriolis is **not** included here: forward-Euler integration
    of the rotation operator ``du/dt = f v``, ``dv/dt = −f u`` is
    unconditionally unstable (amplification ``√(1 + (f·dt)²) > 1``), which
    over many inertial periods corrupts the Ekman balance.  The ocean SCM
    driver instead applies Coriolis as a Crank-Nicolson rotation sub-step
    (see ``OceanColumnModel._apply_coriolis_rotation``), which is exact and
    energy-conserving for any ``f·dt``.
    """
    validate_ocean_forcing(forcing)

    cfg = PrescribedForcingConfig(
        tau_x=_eval_scalar(forcing.tau_x, t),
        tau_y=_eval_scalar(forcing.tau_y, t),
        Q_net=_eval_scalar(forcing.q_net, t),
        E_minus_P=_eval_scalar(forcing.e_minus_p, t),
        wind_profile="constant",
    )
    J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)
    out = prescribed_surface_forcing(
        state.u.data, state.v.data, state.T.data, state.S.data,
        z_coord, J, grid, cfg,
    )
    return wrap_ocean_tendencies(out.du_dt, out.dv_dt, out.dT_dt, out.dS_dt, state)
