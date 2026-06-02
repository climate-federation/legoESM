"""PrescribedComponent — the inert producer brick of the component seam (Stage B)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.components import AbstractComponent
from legoesm.components.prescribed_component import PrescribedComponent


def test_metadata_and_isinstance() -> None:
    pc = PrescribedComponent({"sst": jnp.full((4,), 290.0)}, provided_fluxes=("sst",))
    assert isinstance(pc, AbstractComponent)
    assert pc.prognostic_variables == ()   # evolves nothing
    assert pc.required_forcing == ()        # needs no partner
    assert pc.provided_fluxes == ("sst",)   # ... it PROVIDES


def test_provide_static_payload() -> None:
    payload = {"sst": jnp.full((4,), 290.0)}
    pc = PrescribedComponent(payload, provided_fluxes=("sst",))
    assert pc.provide() is payload          # static handoff returned as-is


def test_provide_callable_payload() -> None:
    pc = PrescribedComponent(
        lambda grid, state: {"sst": jnp.full((3,), float(state))},
        provided_fluxes=("sst",),
    )
    assert jnp.allclose(pc.provide(None, 285.0)["sst"], 285.0)


def test_tendency_is_a_noop() -> None:
    pc = PrescribedComponent({"sst": jnp.zeros((2,))}, provided_fluxes=("sst",))
    assert pc.tendency(None, None, None, None) == ()
    # forcing/params are ignored (a producer is inert) — still a no-op, no raise
    assert pc.tendency(None, None, {"x": 1.0}, {"y": 2.0}) == ()


def test_provide_is_differentiable() -> None:
    """A gradient flows back through the prescribed payload (producer side, D1)."""
    def loss(sst):
        pc = PrescribedComponent({"sst": sst}, provided_fluxes=("sst",))
        return jnp.sum(pc.provide()["sst"] ** 2)

    x = jnp.array([1.0, 2.0, 3.0])
    assert jnp.allclose(jax.grad(loss)(x), 2.0 * x)
