"""Conformance for the Stage-A2.3 grid operator interface (the B2 target)."""

from __future__ import annotations

import pytest

from legoesm.grids.operator_protocol import GridOperators, validate_grid_operators


class _ToyGrid:
    """A minimal grid exposing the operator interface (what B2 will give grids)."""

    def divergence(self, u, v):
        return u + v

    def gradient(self, scalar):
        return (scalar, scalar)

    def vorticity(self, u, v):
        return u - v

    def interpolate(self, field, src_location, dst_location):
        return field

    def halo_fill(self, field):
        return field


def test_toy_grid_satisfies_protocol() -> None:
    assert isinstance(_ToyGrid(), GridOperators)
    validate_grid_operators(_ToyGrid())  # no raise


def test_existing_namedtuple_grid_does_not_satisfy_yet() -> None:
    """Existing grids are NamedTuples without operator methods (B2 adds them)."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere

    grid = create_cubed_sphere(4)
    assert not isinstance(grid, GridOperators)  # honest: not yet


def test_validate_rejects_missing_operator() -> None:
    class Incomplete:
        def divergence(self, u, v):
            return u

        # gradient / vorticity / interpolate / halo_fill missing

    with pytest.raises(TypeError, match="gradient"):
        validate_grid_operators(Incomplete())


def test_validate_rejects_non_callable_operator() -> None:
    class Bad(_ToyGrid):
        divergence = 1  # passes isinstance(GridOperators) on attr presence...

    assert isinstance(Bad(), GridOperators)  # ...but is not callable
    with pytest.raises(TypeError, match="divergence"):
        validate_grid_operators(Bad())


def test_validate_rejects_wrong_arity() -> None:
    class Bad(_ToyGrid):
        def divergence(self, u):  # missing v
            return u

    with pytest.raises(TypeError, match="divergence"):
        validate_grid_operators(Bad())
