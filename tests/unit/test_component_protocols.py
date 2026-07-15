"""Conformance for the Stage-A2 component contracts.

Proves the Protocols/ABC match reality (an existing solver class satisfies
DycoreProtocol) and behave correctly (runtime_checkable isinstance, ABC enforces
its abstract methods), so Stage B can rely on them.
"""

from __future__ import annotations

import pytest

from legoesm.components import (
    AbstractComponent,
    DycoreProtocol,
    PhysicsModuleProtocol,
    SurfaceComponentProtocol,
    validate_dycore,
)


def test_dycore_protocol_is_structural() -> None:
    class Duck:
        def step(self, state, dt):
            return state

    class NoStep:
        pass

    assert isinstance(Duck(), DycoreProtocol)
    assert not isinstance(NoStep(), DycoreProtocol)


def test_real_solver_class_conforms_to_dycore_protocol() -> None:
    """An existing dycore exposes the contract's ``step`` — the contract is real."""
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
        CDGridShallowWaterModel,
    )

    assert hasattr(CDGridShallowWaterModel, "step")


def test_validate_dycore_accepts_a_real_step() -> None:
    class Good:
        def step(self, state, dt):
            return state

    validate_dycore(Good())  # no raise


def test_validate_dycore_rejects_non_callable_step() -> None:
    class Bad:
        step = 1  # passes isinstance(DycoreProtocol) but is not callable

    assert isinstance(Bad(), DycoreProtocol)  # runtime_checkable is fooled...
    with pytest.raises(TypeError, match="callable"):
        validate_dycore(Bad())  # ...validate_dycore is not


def test_validate_dycore_rejects_wrong_arity() -> None:
    class Bad:
        def step(self):  # missing (state, dt)
            return None

    with pytest.raises(TypeError, match="state, dt"):
        validate_dycore(Bad())


def test_validate_dycore_accepts_varargs_step() -> None:
    class Flexible:
        def step(self, *args, **kwargs):
            return args

    validate_dycore(Flexible())  # *args satisfies the arity contract


def test_validate_dycore_rejects_extra_required_positional() -> None:
    class Bad:
        def step(self, state, dt, forcing):  # extra REQUIRED arg
            return state

    with pytest.raises(TypeError, match="step.state, dt."):
        validate_dycore(Bad())


def test_validate_dycore_rejects_extra_required_keyword_only() -> None:
    class Bad:
        def step(self, state, dt, *, forcing):  # required keyword-only
            return state

    with pytest.raises(TypeError):
        validate_dycore(Bad())


def test_validate_dycore_accepts_extra_optional_arg() -> None:
    class Good:
        def step(self, state, dt, forcing=None):  # extra OPTIONAL -> still callable
            return state

    validate_dycore(Good())


def test_validate_dycore_rejects_uninspectable_step(monkeypatch) -> None:
    """An uninspectable step is rejected, not silently accepted."""
    import legoesm.components.protocol as proto

    class Weird:
        def step(self, state, dt):
            return state

    def _no_signature(_obj):
        raise ValueError("no signature found")

    monkeypatch.setattr(proto.inspect, "signature", _no_signature)
    with pytest.raises(TypeError, match="could not be inspected"):
        validate_dycore(Weird())


def test_surface_component_protocol_is_structural() -> None:
    class Duck:
        def tendency(self, grid, state, forcing, params):
            return state

    assert isinstance(Duck(), SurfaceComponentProtocol)
    assert not isinstance(object(), SurfaceComponentProtocol)


def test_abstract_component_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        AbstractComponent()


def test_minimal_concrete_component_works() -> None:
    class Toy(AbstractComponent):
        @property
        def prognostic_variables(self):
            return ("T",)

        @property
        def required_forcing(self):
            return ("sw_down",)

        @property
        def provided_fluxes(self):
            return ("lw_up",)

        def tendency(self, grid, state, forcing, params):
            return {"T": 0.0}

    c = Toy()
    assert c.prognostic_variables == ("T",)
    assert c.required_forcing == ("sw_down",)
    assert c.provided_fluxes == ("lw_up",)
    assert c.tendency(None, None, None, None) == {"T": 0.0}


def test_incomplete_component_still_abstract() -> None:
    """Omitting an abstract method keeps the subclass un-instantiable."""

    class Missing(AbstractComponent):
        @property
        def prognostic_variables(self):
            return ()

        # required_forcing / provided_fluxes / tendency not implemented

    with pytest.raises(TypeError):
        Missing()


def test_physics_module_protocol_reexported() -> None:
    from legoesm.core.state import PhysicsModuleProtocol as CorePMP

    # Re-exported (not duplicated) so all component seams import from one place.
    assert PhysicsModuleProtocol is CorePMP
