"""Direct unit tests for the four ocean-physics ``output.py`` NamedTuple
containers.

These leaf modules each define a single output container that every scheme in
their package builds and the dynamics step unpacks.  They carry no numerics, so
the contract under test is structural:

  * the field set + order is exactly what callers index by name / position;
  * arrays round-trip through construction with their shapes / dtypes intact;
  * the optional fields (``OceanConvectionOutput.K_v`` / ``A_v`` / momentum
    tendencies) default to ``None`` so momentum-free schemes (plume) can omit
    them;
  * the containers are JAX pytrees (so they flow through ``jax.jit`` / ``scan``
    / ``grad`` like every other carry in the model).

A drift in any of these field names silently breaks every scheme that fills the
container, so pinning them here is the cheap guard.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.ocean.physics.bottom_drag.output import (
    BottomDragOutput,
    bottom_level_drag_output,
)
from legoesm.ocean.physics.convection.output import OceanConvectionOutput
from legoesm.ocean.physics.surface_forcing.output import SurfaceForcingOutput
from legoesm.ocean.physics.vertical_mixing.output import VerticalMixingOutput


@pytest.fixture(autouse=True)
def _x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


# --------------------------------------------------------------------------- #
# BottomDragOutput + the shared bottom-level pad helper.
# --------------------------------------------------------------------------- #
class TestBottomDragOutput:
    def test_fields_and_order(self):
        assert BottomDragOutput._fields == ("du_dt", "dv_dt")

    def test_construct_roundtrip(self):
        a = jnp.ones((6, 4, 4, 5))
        b = jnp.zeros((6, 4, 4, 5))
        out = BottomDragOutput(du_dt=a, dv_dt=b)
        assert out.du_dt.shape == (6, 4, 4, 5)
        assert out[0] is a and out[1] is b

    def test_bottom_level_pad_places_drag_at_deepest_level(self):
        """Per-column bottom drag is padded to the deepest level; all levels
        above are exactly zero (single Pad HLO)."""
        nlev = 5
        drag_u = jnp.full((6, 4, 4), -0.3)
        drag_v = jnp.full((6, 4, 4), 0.1)
        out = bottom_level_drag_output(drag_u, drag_v, nlev)
        assert isinstance(out, BottomDragOutput)
        assert out.du_dt.shape == (6, 4, 4, nlev)
        # Only the bottom level is nonzero.
        assert jnp.allclose(out.du_dt[..., -1], drag_u)
        assert jnp.allclose(out.dv_dt[..., -1], drag_v)
        assert jnp.allclose(out.du_dt[..., :-1], 0.0)
        assert jnp.allclose(out.dv_dt[..., :-1], 0.0)

    def test_bottom_level_pad_arbitrary_leading_rank(self):
        """The pad adapts to the input rank (works for MPAS (nCells,) too)."""
        nlev = 3
        drag_u = jnp.arange(7.0)          # 1-D leading (e.g. nCells)
        drag_v = -jnp.arange(7.0)
        out = bottom_level_drag_output(drag_u, drag_v, nlev)
        assert out.du_dt.shape == (7, nlev)
        assert jnp.allclose(out.du_dt[..., -1], drag_u)
        assert jnp.allclose(out.du_dt[..., :-1], 0.0)

    def test_is_pytree(self):
        out = bottom_level_drag_output(
            jnp.ones((2, 2)), jnp.ones((2, 2)), 4)
        leaves = jax.tree_util.tree_leaves(out)
        assert len(leaves) == 2
        scaled = jax.tree_util.tree_map(lambda x: 2.0 * x, out)
        assert jnp.allclose(scaled.du_dt[..., -1], 2.0)


# --------------------------------------------------------------------------- #
# OceanConvectionOutput — optional momentum / diffusivity slots.
# --------------------------------------------------------------------------- #
class TestOceanConvectionOutput:
    def test_field_order(self):
        assert OceanConvectionOutput._fields == (
            "dT_dt", "dS_dt", "convection_flag", "K_v", "du_dt", "dv_dt", "A_v",
        )

    def test_required_fields_only_optionals_default_none(self):
        dT = jnp.zeros((6, 4, 4, 5))
        dS = jnp.zeros((6, 4, 4, 5))
        flag = jnp.zeros((6, 4, 4, 4))   # interfaces nlev-1
        out = OceanConvectionOutput(dT_dt=dT, dS_dt=dS, convection_flag=flag)
        # Momentum / diffusivity slots default to None (plume-style schemes).
        assert out.K_v is None
        assert out.du_dt is None and out.dv_dt is None
        assert out.A_v is None

    def test_full_construction_shapes(self):
        dT = jnp.zeros((6, 4, 4, 5))
        K = jnp.ones((6, 4, 4, 4))
        out = OceanConvectionOutput(
            dT_dt=dT, dS_dt=dT, convection_flag=K,
            K_v=K, du_dt=dT, dv_dt=dT, A_v=K,
        )
        assert out.K_v.shape == (6, 4, 4, 4)        # interface field
        assert out.dT_dt.shape == (6, 4, 4, 5)      # full-level tendency

    def test_pytree_treats_none_optionals_as_no_leaf(self):
        out = OceanConvectionOutput(
            dT_dt=jnp.zeros(3), dS_dt=jnp.zeros(3),
            convection_flag=jnp.zeros(2),
        )
        leaves = jax.tree_util.tree_leaves(out)
        # None optionals are pytree-empty, so only the 3 real arrays remain.
        assert len(leaves) == 3


# --------------------------------------------------------------------------- #
# SurfaceForcingOutput — all 7 fields required.
# --------------------------------------------------------------------------- #
class TestSurfaceForcingOutput:
    def test_field_order(self):
        assert SurfaceForcingOutput._fields == (
            "du_dt", "dv_dt", "dT_dt", "dS_dt", "Q_net", "tau_x", "tau_y",
        )

    def test_construction_and_diagnostic_shapes(self):
        tend = jnp.zeros((6, 4, 4, 5))
        diag = jnp.zeros((6, 4, 4))
        out = SurfaceForcingOutput(
            du_dt=tend, dv_dt=tend, dT_dt=tend, dS_dt=tend,
            Q_net=diag, tau_x=diag, tau_y=diag,
        )
        assert out.dT_dt.shape == (6, 4, 4, 5)
        assert out.Q_net.shape == (6, 4, 4)
        assert jax.tree_util.tree_leaves(out).__len__() == 7


# --------------------------------------------------------------------------- #
# VerticalMixingOutput — tendencies + K_v / A_v diagnostics at interfaces.
# --------------------------------------------------------------------------- #
class TestVerticalMixingOutput:
    def test_field_order(self):
        assert VerticalMixingOutput._fields == (
            "du_dt", "dv_dt", "dT_dt", "dS_dt", "K_v", "A_v",
        )

    def test_construction_interface_shapes(self):
        tend = jnp.zeros((6, 4, 4, 5))
        iface = jnp.ones((6, 4, 4, 4))   # nlev-1 interfaces
        out = VerticalMixingOutput(
            du_dt=tend, dv_dt=tend, dT_dt=tend, dS_dt=tend,
            K_v=iface, A_v=iface,
        )
        assert out.K_v.shape == (6, 4, 4, 4)
        assert out.du_dt.shape == (6, 4, 4, 5)
        # Diffusivity / viscosity diagnostics are non-negative as built.
        assert bool(jnp.all(out.K_v >= 0.0))
        assert bool(jnp.all(out.A_v >= 0.0))

    def test_pytree_jit_roundtrip(self):
        tend = jnp.zeros((2, 3))
        iface = jnp.ones((2, 2))
        out = VerticalMixingOutput(
            du_dt=tend, dv_dt=tend, dT_dt=tend, dS_dt=tend,
            K_v=iface, A_v=iface,
        )
        identity = jax.jit(lambda o: o)(out)
        assert isinstance(identity, VerticalMixingOutput)
        assert jnp.allclose(identity.K_v, out.K_v)
