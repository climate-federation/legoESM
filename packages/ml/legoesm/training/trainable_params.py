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
    """Constraint for a single trainable parameter.

    ``min_val``/``max_val`` are scalars for a scalar parameter, or equal-length
    tuples for per-element bounds on an array parameter. ``shape`` is the resolved
    concrete array shape (``()`` = scalar). ``scheme_key``/``field`` carry the
    parameter's scheme-qualified identity so trained values can be routed back to
    the owning ``*Config`` via :meth:`TrainablePhysicsParams.to_overrides`
    (empty for the legacy flat parameters, which use ``to_segment_kwargs``)."""
    name: str
    min_val: float | tuple[float, ...]
    max_val: float | tuple[float, ...]
    transform: str  # "softplus" (positive), "sigmoid" (bounded), "none"
    shape: tuple[int, ...] = ()
    scheme_key: str = ""
    field: str = ""


# Default trainable parameters (the ones already traced through build_segment_fn)
# Bounds must match tuning.py validated ranges.
# Gray radiation is NOT trained (user directive 2026-08-11): tau_equator /
# tau_pole are gray optical depths and were dropped from this set on the same
# day they were dropped from AIMIPClassicalParams. Gray still RUNS, at its
# documented defaults. The radiative knobs a classical model trains are
# RRTMGP's surface albedo + emissivity (see AIMIPClassicalParams).
DEFAULT_TRAINABLE = [
    ParamConstraint("sbm_tau_c", 3600.0, 14400.0, "sigmoid"),
    ParamConstraint("sbm_RH_ref", 0.6, 0.9, "sigmoid"),
    ParamConstraint("C_H", 0.001, 0.005, "sigmoid"),
    ParamConstraint("C_E", 0.001, 0.005, "sigmoid"),
    ParamConstraint("albedo_ice", 0.4, 0.8, "sigmoid"),
    ParamConstraint("albedo_ocean", 0.03, 0.10, "sigmoid"),
]

# Gray-radiation optical-depth parameters.  The unified pipeline feeds
# tau_equator/tau_pole only into the gray solver; the RRTMGP branch
# explicitly discards them (physics_pipeline.py `del tau_equator,
# tau_pole`), so under rrtmgp they would be dead degrees of freedom.
# Emptied 2026-08-11 with DEFAULT_TRAINABLE above: gray is never trained.
_GRAY_RADIATION_TRAINABLE: list = []

# Surface-albedo parameters.  The blended (ice/ocean/land) albedo
# reaches the radiative heating through BOTH solvers: RRTMGP consumes
# ``albedo_col`` directly, and since 2026-06-10 the gray SW reflection
# takes the blended albedo too (``gray_radiation(sfc_albedo=...)``) —
# only ``radiation="none"`` leaves these without a gradient path.
_ALBEDO_TRAINABLE = [
    ParamConstraint("albedo_ice", 0.4, 0.8, "sigmoid"),
    ParamConstraint("albedo_ocean", 0.03, 0.10, "sigmoid"),
]

# Bulk surface-exchange coefficients.  Live only when no turbulence
# scheme is active: with turbulence on, ``turb_owns_surface`` skips the
# pipeline's bulk boundary-layer kick and the scheme's own
# SurfaceLayerConfig (which these trainables are NOT threaded into)
# supplies the fluxes.
_BULK_SURFACE_TRAINABLE = [
    ParamConstraint("C_H", 0.001, 0.005, "sigmoid"),
    ParamConstraint("C_E", 0.001, 0.005, "sigmoid"),
]

# SBM-specific convection parameters
_SBM_TRAINABLE = [
    ParamConstraint("sbm_tau_c", 3600.0, 14400.0, "sigmoid"),
    ParamConstraint("sbm_RH_ref", 0.6, 0.9, "sigmoid"),
]


def trainable_constraints_for_scheme(
    convection_scheme: str = "sbm",
    radiation_scheme: str = "gray",
    turbulence_scheme: str = "none",
) -> list[ParamConstraint]:
    """Return trainable parameter constraints reachable under the given schemes.

    Filters out parameters that the selected physics configuration can
    never train (zero gradient by construction), so an optimizer is not
    handed dead degrees of freedom:

    - the gray optical depths ``tau_equator``/``tau_pole`` only feed the
      gray solver (rrtmgp explicitly discards them);
    - the blended-albedo pair trains under gray AND rrtmgp (both consume
      the pipeline's blended surface albedo) but not ``radiation="none"``;
    - an active turbulence scheme owns the surface fluxes, so the bulk
      ``C_H``/``C_E`` only train with ``turbulence_scheme="none"``;
    - only SBM has scheme-specific convection parameters.

    Parameters
    ----------
    convection_scheme : str
        Convection scheme name: "sbm", "dca", "kuo", "mass_flux", "edmf",
        "none".
    radiation_scheme : str
        Radiation scheme name: "gray", "rrtmgp" (alias "rrtmg"), "none".
    turbulence_scheme : str
        Turbulence scheme name ("none" enables the bulk surface pair).

    Returns
    -------
    list[ParamConstraint]
    """
    radiation = "rrtmgp" if radiation_scheme == "rrtmg" else radiation_scheme

    constraints: list[ParamConstraint] = []
    if radiation == "gray":
        constraints += _GRAY_RADIATION_TRAINABLE
    if radiation in ("gray", "rrtmgp"):
        constraints += _ALBEDO_TRAINABLE
    if turbulence_scheme == "none":
        constraints += _BULK_SURFACE_TRAINABLE
    if convection_scheme == "sbm":
        constraints += _SBM_TRAINABLE
    return constraints


def sigmoid_to_range(raw: jax.Array, lo, hi) -> jax.Array:
    """Map unconstrained raw value(s) to [lo, hi] via sigmoid.

    ``lo``/``hi`` may be scalars (broadcast over an array ``raw``) or tuples /
    arrays of per-element bounds; conversion through ``jnp.asarray`` keeps the
    transform elementwise and differentiable for array parameters."""
    lo = jnp.asarray(lo, dtype=raw.dtype)
    hi = jnp.asarray(hi, dtype=raw.dtype)
    return lo + (hi - lo) * jax.nn.sigmoid(raw)


def range_to_sigmoid(val: float, lo: float, hi: float) -> float:
    """Inverse (scalar): map a [lo, hi] value to unconstrained raw (logit)."""
    t = (val - lo) / (hi - lo)
    t = max(min(t, 0.999), 0.001)  # clamp for numerical stability
    import math
    return math.log(t / (1.0 - t))


def range_to_sigmoid_array(val: jax.Array, lo, hi) -> jax.Array:
    """Inverse (array-safe): map [lo, hi] value(s) to unconstrained raw (logit),
    elementwise, with the same 0.001/0.999 clamp as the scalar form. Used by the
    spec collector to seed array-valued raw parameters."""
    val = jnp.asarray(val)
    lo = jnp.asarray(lo, dtype=val.dtype)
    hi = jnp.asarray(hi, dtype=val.dtype)
    t = jnp.clip((val - lo) / (hi - lo), 0.001, 0.999)
    return jnp.log(t / (1.0 - t))


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
                    range_to_sigmoid(default, c.min_val, c.max_val),
                    dtype=param_dtype,
                )
            elif c.transform == "softplus":
                # Overflow-safe inverse softplus: log(exp(y)-1) = y + log1p(-exp(-y)).
                # The naive ``float(log(exp(default)-1))`` overflows to inf for
                # default >~ 700 (even in fp64) — large timescales / lengths /
                # concentrations — seeding the trained leaf with inf. Mirrors
                # ``param_collector._seed_raw``.
                y = max(float(default), 1e-6)
                raw[c.name] = jnp.array(
                    y + float(jnp.log1p(-jnp.exp(-y))),
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
                result[c.name] = sigmoid_to_range(raw, c.min_val, c.max_val)
            elif c.transform == "softplus":
                result[c.name] = jax.nn.softplus(raw)
            else:
                result[c.name] = raw
        return result

    def to_segment_kwargs(self) -> dict[str, jax.Array]:
        """Return kwargs compatible with ``build_segment_fn`` (legacy flat path).

        Emits ``{legacy_flat_name: value}``. A spec-collected parameter that has
        no flat ``build_segment_fn`` alias (``scheme_key`` set, no legacy name)
        cannot be injected this way — use :meth:`to_overrides` +
        ``legoesm.core.param_overrides.apply_param_overrides`` instead. Such a
        parameter raises
        ``ValueError`` here rather than being silently dropped."""
        values = self.as_dict()
        scheme_qualified = [c.name for c in self.constraints if c.scheme_key]
        if scheme_qualified:
            raise ValueError(
                "to_segment_kwargs() cannot inject scheme-qualified parameters "
                f"{scheme_qualified}; use to_overrides() with "
                "legoesm.core.param_overrides.apply_param_overrides(). Only legacy "
                "flat "
                "parameters (no scheme_key) are routable as segment kwargs."
            )
        return values

    def to_overrides(self) -> dict[str, dict[str, jax.Array]]:
        """Return ``{scheme_key: {config_field: constrained_value}}`` for the
        spec-collected parameters, ready for
        ``legoesm.core.param_overrides.apply_param_overrides(physics_config,
        overrides)`` to
        splice into the owning ``*Config`` NamedTuples inside the loss. Legacy
        flat parameters (no ``scheme_key``) are skipped (they use
        :meth:`to_segment_kwargs`)."""
        values = self.as_dict()
        out: dict[str, dict[str, jax.Array]] = {}
        for c in self.constraints:
            if not c.scheme_key or not c.field:
                continue
            out.setdefault(c.scheme_key, {})[c.field] = values[c.name]
        return out
