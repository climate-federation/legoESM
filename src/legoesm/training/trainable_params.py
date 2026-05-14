"""Learnable physics parameters for gradient-based tuning.

Wraps the tunable parameters from ``tuning.py`` into an Equinox module
with constraint transforms (softplus for positive, sigmoid for bounded).
Provides injection into ``build_segment_fn`` kwargs.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import equinox as eqx

from legoesm.tuning import TUNING_PARAMETERS


class ParamConstraint(NamedTuple):
    """Constraint for a single parameter."""
    name: str
    min_val: float
    max_val: float
    transform: str  # "softplus" (positive), "sigmoid" (bounded), "none"


# Default trainable parameters (the ones already traced through build_segment_fn)
# Bounds must match tuning.py validated ranges.
DEFAULT_TRAINABLE = [
    ParamConstraint("tau_equator", 5.0, 10.0, "sigmoid"),
    ParamConstraint("tau_pole", 1.0, 3.0, "sigmoid"),
    ParamConstraint("sbm_tau_c", 3600.0, 14400.0, "sigmoid"),
    ParamConstraint("sbm_RH_ref", 0.6, 0.9, "sigmoid"),
    ParamConstraint("C_H", 0.001, 0.005, "sigmoid"),
    ParamConstraint("C_E", 0.001, 0.005, "sigmoid"),
    ParamConstraint("albedo_ice", 0.4, 0.8, "sigmoid"),
    ParamConstraint("albedo_ocean", 0.03, 0.10, "sigmoid"),
]

# Non-convection trainable parameters (shared by all schemes)
# iter-251 (AIMIP Phase 2.1): added ``albedo_land`` so the AIMIP
# training loop tunes the full surface-albedo triple (ice, ocean,
# land) rather than just two of three.  Bounds match
# ``tuning.py::TUNING_PARAMETERS["albedo_land"]`` (0.1 - 0.4) which
# spans desert -> snow-covered transitions.  AMIP-style runs with
# fixed ``--dataset analytical`` will see no behavioural change
# because ``albedo_land`` is only consulted by the surface module
# when a non-trivial land fraction is in the IC; ERA5 ICs activate it.
_COMMON_TRAINABLE = [
    ParamConstraint("tau_equator", 5.0, 10.0, "sigmoid"),
    ParamConstraint("tau_pole", 1.0, 3.0, "sigmoid"),
    ParamConstraint("C_H", 0.001, 0.005, "sigmoid"),
    ParamConstraint("C_E", 0.001, 0.005, "sigmoid"),
    ParamConstraint("albedo_ice", 0.4, 0.8, "sigmoid"),
    ParamConstraint("albedo_ocean", 0.03, 0.10, "sigmoid"),
    ParamConstraint("albedo_land", 0.1, 0.4, "sigmoid"),
]

# SBM-specific convection parameters
_SBM_TRAINABLE = [
    ParamConstraint("sbm_tau_c", 3600.0, 14400.0, "sigmoid"),
    ParamConstraint("sbm_RH_ref", 0.6, 0.9, "sigmoid"),
]

# Louis-turbulence-specific trainable parameters.
# iter-252 (AIMIP Phase 2.2): added Ri_crit so the AIMIP training
# loop tunes the stable-PBL cutoff in the Louis scheme.  Bounds
# (0.20, 0.50) match tuning.py::TUNING_PARAMETERS["Ri_crit"].  Only
# active when --turbulence louis.
_LOUIS_TURBULENCE_TRAINABLE = [
    ParamConstraint("Ri_crit", 0.20, 0.50, "sigmoid"),
]


def trainable_constraints_for_scheme(
    convection_scheme: str = "sbm",
    turbulence_scheme: str = "none",
) -> list[ParamConstraint]:
    """Return trainable parameter constraints appropriate for the given schemes.

    Parameters
    ----------
    convection_scheme : str
        Convection scheme name: "sbm", "dca", "kuo", "mass_flux", "edmf", "none".
        Only SBM has scheme-specific trainable parameters.
    turbulence_scheme : str
        Turbulence scheme name: "louis", "tke", "smagorinsky", ..., "none".
        Only Louis has scheme-specific trainable parameters (iter-252).

    Returns
    -------
    list[ParamConstraint]
    """
    result = list(_COMMON_TRAINABLE)
    if convection_scheme == "sbm":
        result = result + _SBM_TRAINABLE
    if turbulence_scheme == "louis":
        result = result + _LOUIS_TURBULENCE_TRAINABLE
    return result


def _sigmoid_to_range(raw: jax.Array, lo: float, hi: float) -> jax.Array:
    """Map unconstrained raw value to [lo, hi] via sigmoid."""
    return lo + (hi - lo) * jax.nn.sigmoid(raw)


def _range_to_sigmoid(val: float, lo: float, hi: float) -> float:
    """Inverse: map [lo, hi] value to unconstrained raw (logit)."""
    t = (val - lo) / (hi - lo)
    t = max(min(t, 0.999), 0.001)  # clamp for numerical stability
    import math
    return math.log(t / (1.0 - t))


class TrainablePhysicsParams(eqx.Module):
    """Learnable physics parameters as an Equinox module.

    Stores unconstrained raw values; apply_constraints() returns
    physical values in valid ranges.

    Usage::

        params = TrainablePhysicsParams.from_defaults()
        physical = params.as_dict()  # {name: constrained_value}
        segment_kwargs = params.to_segment_kwargs()  # ready for build_segment_fn
    """
    raw_values: dict[str, jax.Array]
    constraints: list[ParamConstraint] = eqx.field(static=True)

    @staticmethod
    def from_defaults(
        constraints: list[ParamConstraint] | None = None,
    ) -> TrainablePhysicsParams:
        """Create from default values in the tuning registry."""
        if constraints is None:
            constraints = DEFAULT_TRAINABLE

        # Resolve dtype from precision policy (default float32).
        try:
            from legoesm.core.precision import get_policy
            param_dtype = get_policy().compute
        except Exception:
            param_dtype = jnp.float32

        raw = {}
        for c in constraints:
            # Look up default from registry, fallback to midpoint
            if c.name in TUNING_PARAMETERS:
                default = TUNING_PARAMETERS[c.name].default
            else:
                default = (c.min_val + c.max_val) / 2.0

            if c.transform == "sigmoid":
                raw[c.name] = jnp.array(
                    _range_to_sigmoid(default, c.min_val, c.max_val),
                    dtype=param_dtype,
                )
            elif c.transform == "softplus":
                raw[c.name] = jnp.array(
                    float(jnp.log(jnp.exp(default) - 1.0)),
                    dtype=param_dtype,
                )
            else:
                raw[c.name] = jnp.array(default, dtype=param_dtype)

        return TrainablePhysicsParams(raw_values=raw, constraints=constraints)

    def as_dict(self) -> dict[str, jax.Array]:
        """Return constrained physical values."""
        result = {}
        for c in self.constraints:
            raw = self.raw_values[c.name]
            if c.transform == "sigmoid":
                result[c.name] = _sigmoid_to_range(raw, c.min_val, c.max_val)
            elif c.transform == "softplus":
                result[c.name] = jax.nn.softplus(raw)
            else:
                result[c.name] = raw
        return result

    def to_segment_kwargs(self) -> dict[str, jax.Array]:
        """Return kwargs compatible with build_segment_fn."""
        return self.as_dict()
