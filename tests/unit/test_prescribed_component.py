"""PrescribedComponent — the inert producer brick of the component seam (Stage B)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.components import AbstractComponent
from legoesm.components.prescribed_component import PrescribedComponent


def test_metadata_and_isinstance() -> None:
    pc = PrescribedComponent({"sst": jnp.full((4,), 290.0)}, provided_fluxes=("sst",))
    assert isinstance(pc, AbstractComponent)
    assert pc.prognostic_variables == ()   # evolves nothing
    assert pc.required_forcing == ()        # needs no partner
    assert pc.provided_fluxes == ("sst",)   # ... it PROVIDES


def test_provide_routes_by_name() -> None:
    payload = {"sst": jnp.full((4,), 290.0), "sic": jnp.zeros((4,))}
    pc = PrescribedComponent(payload, provided_fluxes=("sst", "sic"))
    out = pc.provide()
    assert jnp.allclose(out["sst"], 290.0) and jnp.allclose(out["sic"], 0.0)


def test_payload_keys_must_match_provided_fluxes() -> None:
    # missing a declared flux -> loud failure (not a silent partial producer)
    with pytest.raises(ValueError, match="provided_fluxes"):
        PrescribedComponent({"sst": jnp.zeros((2,))}, provided_fluxes=("sst", "sic"))
    # a raw array is not a name->value mapping
    with pytest.raises(TypeError, match="Mapping"):
        PrescribedComponent(jnp.zeros((2,)), provided_fluxes=("sst",))


def test_provide_callable_payload() -> None:
    pc = PrescribedComponent(
        lambda grid, state: {"sst": jnp.full((3,), state)},
        provided_fluxes=("sst",),
    )
    assert jnp.allclose(pc.provide(None, jnp.asarray(285.0))["sst"], 285.0)


def test_callable_payload_keys_validated_lazily() -> None:
    """A callable returning the wrong key set fails on provide() (not silently)."""
    pc = PrescribedComponent(
        lambda grid, state: {"wrong": jnp.zeros((2,))}, provided_fluxes=("sst",))
    with pytest.raises(ValueError, match="provided_fluxes"):
        pc.provide()


def test_tendency_is_noop_and_rejects_payloads() -> None:
    pc = PrescribedComponent({"sst": jnp.zeros((2,))}, provided_fluxes=("sst",))
    assert pc.tendency(None, None, None, None) == ()
    # an inert producer rejects forcing/params rather than silently dropping them
    with pytest.raises(ValueError, match="inert producer"):
        pc.tendency(None, None, {"x": 1.0}, None)
    with pytest.raises(ValueError, match="inert producer"):
        pc.tendency(None, None, None, {"y": 2.0})


def test_provide_is_differentiable() -> None:
    """A gradient flows back through the prescribed payload (producer side, D1)."""
    def loss(sst):
        pc = PrescribedComponent({"sst": sst}, provided_fluxes=("sst",))
        return jnp.sum(pc.provide()["sst"] ** 2)

    x = jnp.array([1.0, 2.0, 3.0])
    assert jnp.allclose(jax.grad(loss)(x), 2.0 * x)


def test_callable_payload_is_differentiable() -> None:
    """grad flows through a callable (time-varying) payload closure — a pure JAX
    seam, so it traces and differentiates."""
    def loss(amp):
        pc = PrescribedComponent(
            lambda grid, state: {"sst": amp * jnp.ones((3,))},
            provided_fluxes=("sst",))
        return jnp.sum(pc.provide(None, None)["sst"] ** 2)

    g = jax.grad(loss)(2.0)
    assert jnp.isfinite(g) and g > 0.0
