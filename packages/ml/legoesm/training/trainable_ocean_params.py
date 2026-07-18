"""Trainable ocean-closure parameters — GEOMETRIC calibration targets.

Ocean sibling of :mod:`legoesm.training.trainable_params` (the AIMIP
atmosphere set), reusing the same ``ParamConstraint`` + sigmoid-transform
machinery so there is ONE constraint/dead-DOF mechanism in the repo.

Stage 0 of the GEOMETRIC calibration campaign
(``docs/dev-notes/planning/geometric_calibration_campaign.md``): the tunables are the
Torres et al. (2025) closure coefficients of
:class:`legoesm.ocean.physics.lateral_mixing.eke.GeometricConfig`, with the
paper's own sampled/plausible ranges as bounds. Wide-decade parameters use a
LOG-sigmoid transform (sigmoid in log10 space) so the optimizer sees a
roughly symmetric landscape around the calibrated defaults; O(1) parameters
use the plain sigmoid.

Every parameter here has a liveness pin in
``tests/unit/test_trainable_ocean_params.py`` (finite, NONZERO gradient
through the actual consuming formula — the AIMIP dead-DOF lesson,
commit d018a795): a knob that stops reaching the tendencies must fail the
test, not silently waste optimizer iterations.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from legoesm.training.trainable_params import (
    ParamConstraint,
    sigmoid_to_range,
    range_to_sigmoid,
)


class OceanParamSpec(NamedTuple):
    """A ParamConstraint plus the log-space flag and the config default."""
    constraint: ParamConstraint
    log_space: bool
    default: float


# GEOMETRIC closure tunables (GeometricConfig field name -> spec).
# Bounds provenance: GeometricConfig docstrings / Torres et al. (2025)
# Table E1 + Appendix E sampled ranges; defaults are the paper's calibrated
# values. Defaults sit interior to the bounds (saturation lesson: a default
# pinned at a sigmoid bound starts with ~zero gradient).
GEOMETRIC_TRAINABLE: tuple[OceanParamSpec, ...] = (
    # eddy efficiency (Eq. 6): calibrated 0.04; ~x4 span each way, log space
    OceanParamSpec(ParamConstraint("alpha", 0.01, 0.16, "sigmoid"),
                   log_space=True, default=0.04),
    # dissipation coefficient (Eq. 4): calibrated 0.022; paper plausible
    # range 0.001-0.1 (two decades -> log space)
    OceanParamSpec(ParamConstraint("c_eps_geometric", 0.001, 0.1, "sigmoid"),
                   log_space=True, default=0.022),
    # barotropic-production momentum diffusivity (Eq. 3): calibrated 1500,
    # paper sampled 500-5000 (log space)
    OceanParamSpec(ParamConstraint("kappa_u", 500.0, 5000.0, "sigmoid"),
                   log_space=True, default=1500.0),
    # EKE lateral diffusion (Eq. 5): 500, low sensitivity (log space)
    OceanParamSpec(ParamConstraint("kappa_e", 100.0, 2000.0, "sigmoid"),
                   log_space=True, default=500.0),
    # Rossby-radius prefactor (Appendix D): 0.4 (NEMO v3.6) / 0.5 (v4);
    # O(1), linear sigmoid
    OceanParamSpec(ParamConstraint("rossby_factor", 0.2, 0.8, "sigmoid"),
                   log_space=False, default=0.4),
)


def _bounds_in_opt_space(spec: OceanParamSpec) -> tuple[float, float]:
    c = spec.constraint
    if spec.log_space:
        return math.log10(c.min_val), math.log10(c.max_val)
    return c.min_val, c.max_val


def constrain(raw: jax.Array, spec: OceanParamSpec) -> jax.Array:
    """Unconstrained real -> physical value (sigmoid, optionally in log10)."""
    lo, hi = _bounds_in_opt_space(spec)
    val = sigmoid_to_range(raw, lo, hi)
    if spec.log_space:
        val = 10.0 ** val
    return val


def unconstrain(value: float, spec: OceanParamSpec) -> float:
    """Physical value -> unconstrained real (inverse of :func:`constrain`)."""
    lo, hi = _bounds_in_opt_space(spec)
    v = math.log10(value) if spec.log_space else value
    return range_to_sigmoid(v, lo, hi)


class TrainableOceanParams(eqx.Module):
    """Unconstrained parameter vector + specs.

    An ``eqx.Module`` like the sibling ``TrainablePhysicsParams``: ``raw``
    is the single array leaf, ``specs`` is STATIC aux — so the object is
    safe to pass through ``jax.jit`` / ``eqx.filter_value_and_grad`` whole
    (the codex-review F1 finding: a plain NamedTuple flattens the spec
    strings/floats into leaves and breaks under jit). The ETKI path
    operates on plain (n_e, n_p) ensemble arrays and only borrows the
    transforms. Field order is GEOMETRIC_TRAINABLE order throughout.
    """
    raw: jax.Array                                # (n_p,) unconstrained
    specs: tuple[OceanParamSpec, ...] = eqx.field(static=True)

    @classmethod
    def from_defaults(cls, specs=GEOMETRIC_TRAINABLE) -> "TrainableOceanParams":
        raw = jnp.asarray([unconstrain(s.default, s) for s in specs],
                          dtype=jnp.float64)
        return cls(raw=raw, specs=tuple(specs))

    @classmethod
    def from_values(cls, values: dict[str, float],
                    specs=GEOMETRIC_TRAINABLE) -> "TrainableOceanParams":
        raw = jnp.asarray(
            [unconstrain(values[s.constraint.name], s) for s in specs],
            dtype=jnp.float64)
        return cls(raw=raw, specs=tuple(specs))

    def as_dict(self) -> dict[str, jax.Array]:
        """Constrained physical values, name-keyed (traced-safe)."""
        return {s.constraint.name: constrain(self.raw[i], s)
                for i, s in enumerate(self.specs)}

    def to_geometric_config(self, base):
        """Project onto a :class:`GeometricConfig` via ``_replace``.

        The GEOMETRIC formulas accept traced field values by design (see the
        GeometricConfig docstring), so the returned config can carry tracers
        through ``jax.grad``/``jax.jit``. NOTE ``validate_geometric_config``
        runs Python comparisons and must be called on CONCRETE configs only
        (eager construction), never inside a trace on this output.
        """
        return base._replace(**self.as_dict())


def names(specs=GEOMETRIC_TRAINABLE) -> tuple[str, ...]:
    return tuple(s.constraint.name for s in specs)


def ensemble_from_priors(
    key: jax.Array,
    n_ensemble: int,
    specs=GEOMETRIC_TRAINABLE,
    rel_spread: float = 0.25,
) -> jnp.ndarray:
    """Initial ETKI ensemble in the UNCONSTRAINED (logit) space.

    Logit-space sigma = ``4*rel_spread``. NOTE (codex-review F2,
    measured): at the default ``rel_spread=0.25`` this yields a BROAD
    prior — p16-p84 spread factors ~2.9-6.9x for the log-space params
    (alpha 3.6x, c_eps 6.9x, kappa_u 2.9x, kappa_e 3.9x) and ~31% for
    rossby_factor — wider than Perezhogin's ~25% point perturbation.
    Treat ``rel_spread`` as a Stage-0 hyperparameter: shrink it if
    members blow up; the sigmoid guarantees members stay inside bounds
    regardless. Returns the (n_e, n_p) unconstrained array; map members
    through :func:`constrain` per spec before forward runs.
    """
    base = TrainableOceanParams.from_defaults(specs).raw
    noise = jax.random.normal(key, (n_ensemble, base.shape[0]),
                              dtype=jnp.float64)
    # raw is unconstrained (sigmoid-logit) space; perturb there with a
    # spread that maps to roughly rel_spread of the physical span
    return base[None, :] + 4.0 * rel_spread * noise  # logit-space sigma
