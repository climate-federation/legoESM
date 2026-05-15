"""AIMIP Phase-2 capstone: TrainablePhysicsParams integration sentinel.

Locks in the iter-251 through iter-258 work by pinning:

- ``TrainablePhysicsParams.from_defaults`` builds the canonical
  10-param set for ``("sbm", "louis")`` (7 common + 2 sbm + 1 louis).
- Every key in ``to_segment_kwargs()`` is a known kwarg of
  :func:`legoesm.driver.compiled_segments.build_segment_fn`, so the
  AIMIP training loop's parameter pytree maps 1:1 onto the rollout
  entry point with no orphan names.
- Constrained values land inside the documented physical ranges.
- ``eqx.filter_value_and_grad`` flows finite gradients to every raw
  parameter — i.e. each of the 10 trainables is genuinely
  differentiable end-to-end, not just nominal-only.

A regression here is a strong signal that the AIMIP training loop will
break or silently optimise a constant.
"""

from __future__ import annotations

import inspect

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from legoesm.driver.compiled_segments import build_segment_fn
from legoesm.training.trainable_params import (
    TrainablePhysicsParams,
    trainable_constraints_for_scheme,
)


def _sbm_louis_constraints():
    return trainable_constraints_for_scheme(
        convection_scheme="sbm", turbulence_scheme="louis"
    )


_EXPECTED_NAMES = [
    "tau_equator", "tau_pole",
    "C_H", "C_E",
    "albedo_ice", "albedo_ocean", "albedo_land",
    "sbm_tau_c", "sbm_RH_ref", "sbm_CAPE_threshold",
    "l_mix_max",
]


def test_trainable_constraints_for_sbm_louis_count_is_11():
    """Pin the exact count + names of the AIMIP-default trainable set
    (``convection=sbm`` + ``turbulence=louis``).  Any silent drop or
    add changes the count and trips this sentinel."""
    cs = _sbm_louis_constraints()
    names = [c.name for c in cs]
    assert names == _EXPECTED_NAMES, (
        f"trainable_constraints_for_scheme('sbm', 'louis') returned "
        f"{names}, expected {_EXPECTED_NAMES}."
    )


def test_from_defaults_round_trips_through_to_segment_kwargs():
    """``TrainablePhysicsParams.from_defaults`` builds raw_values that
    invert back through the constraint transforms to physical values
    inside the documented bounds.  Pinning catches a regression in
    either ``_range_to_sigmoid`` or ``_sigmoid_to_range``."""
    cs = _sbm_louis_constraints()
    p = TrainablePhysicsParams.from_defaults(constraints=cs)
    kwargs = p.to_segment_kwargs()
    assert set(kwargs.keys()) == set(_EXPECTED_NAMES), (
        f"to_segment_kwargs keys {set(kwargs.keys())} != expected "
        f"{set(_EXPECTED_NAMES)}."
    )
    for c in cs:
        v = float(kwargs[c.name])
        assert c.min_val <= v <= c.max_val, (
            f"{c.name}={v} outside [{c.min_val}, {c.max_val}] after "
            f"from_defaults round-trip."
        )


def test_to_segment_kwargs_keys_are_build_segment_fn_kwargs():
    """Every name produced by ``to_segment_kwargs`` must be a kwarg of
    ``build_segment_fn`` — i.e. the trainable dict can be ``**kw``'d
    straight into the rollout entry point without an extra rename.
    Catches a regression where a new trainable lands in
    ``_COMMON_TRAINABLE`` without the corresponding ``build_segment_fn``
    kwarg.
    """
    sig_kwargs = set(inspect.signature(build_segment_fn).parameters)
    cs = _sbm_louis_constraints()
    p = TrainablePhysicsParams.from_defaults(constraints=cs)
    kw = p.to_segment_kwargs()
    orphan = [k for k in kw if k not in sig_kwargs]
    assert not orphan, (
        f"TrainablePhysicsParams emits keys not present in "
        f"build_segment_fn signature: {orphan}.  See iter-255 wire-"
        f"status sentinel — UNWIRED_TODO is the documented place to "
        f"park names that lack wiring."
    )


def test_filter_grad_returns_finite_grads_for_every_raw_value():
    """Differentiability sentinel: ``eqx.filter_value_and_grad`` flows
    finite gradients into every raw value.  A regression that drops a
    parameter from the autodiff path silently zeros its gradient — this
    sentinel catches that by asserting every gradient is finite AND at
    least one component is non-zero per parameter.
    """
    cs = _sbm_louis_constraints()
    p0 = TrainablePhysicsParams.from_defaults(constraints=cs)

    def _loss(p: TrainablePhysicsParams) -> jnp.ndarray:
        # Toy loss: squared sum of physical (constrained) values.  Each
        # raw value contributes to the loss through its sigmoid /
        # softplus, so a finite gradient must flow back to every leaf.
        d = p.as_dict()
        return sum(jnp.sum(v ** 2) for v in d.values())

    grad = eqx.filter_grad(_loss)(p0)

    for c in cs:
        g = grad.raw_values[c.name]
        assert jnp.all(jnp.isfinite(g)), (
            f"Gradient for {c.name} contains NaN/Inf: {g}."
        )
        assert float(jnp.abs(g)) > 0.0, (
            f"Gradient for {c.name} is identically zero.  The leaf is "
            f"likely orphaned from the autodiff path — check the "
            f"constraint transform in as_dict()."
        )


def test_constraint_bounds_match_tuning_registry():
    """Cross-check: every constraint's (min, max) must match the
    corresponding entry in ``tuning.py::TUNING_PARAMETERS``.  Prevents
    silent drift between the trainable constraint bounds and the
    canonical tuning registry."""
    from legoesm.tuning import TUNING_PARAMETERS

    cs = _sbm_louis_constraints()
    for c in cs:
        if c.name not in TUNING_PARAMETERS:
            pytest.fail(
                f"{c.name} is a trainable constraint but missing from "
                f"TUNING_PARAMETERS — add it to tuning.py."
            )
        reg = TUNING_PARAMETERS[c.name]
        assert c.min_val == pytest.approx(reg.min_val), (
            f"{c.name} min_val mismatch: constraint={c.min_val}, "
            f"registry={reg.min_val}."
        )
        assert c.max_val == pytest.approx(reg.max_val), (
            f"{c.name} max_val mismatch: constraint={c.max_val}, "
            f"registry={reg.max_val}."
        )
