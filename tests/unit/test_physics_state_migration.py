"""PR-0 schema-migration tests for ``PhysicsState.conv_prog_profile``.

The convection prognostic carry was renamed from ``conv_prog`` (shape
``(ncol,)``) to ``conv_prog_profile`` (shape ``(ncol, nlev)``) so that
profile-carrying schemes (Tiedtke, Bechtold) and scalar-carrying schemes
(``mass_flux``, ``edmf``) can share the same slot.  These tests pin
that contract:

1. ``init_physics_state`` returns a ``(ncol, nlev)`` shape for every
   scheme.
2. The scalar-carrying ``mass_flux`` and ``edmf`` schemes pack their
   single-scalar initial value at ``[:, -1]`` (cloud-base proxy) with
   zeros aloft.
3. ``update_physics_state`` round-trips through the dict-update
   contract used by the orchestrator.
4. The orchestrator threading (``combined.py``) writes back
   ``conv_prog_profile``-shaped output from the convection bridges.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm.atmosphere.physics.combined import PhysicsConfig
from legoesm.atmosphere.physics.physics_state import (
    PhysicsState,
    init_physics_state,
    update_physics_state,
)
from legoesm.atmosphere.physics.convection.config import ConvectionConfig
from legoesm.atmosphere.physics.radiation.config import RadiationConfig
from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig


NCOL = 24
NLEV = 12


def _make_config(conv_scheme: str = "none") -> PhysicsConfig:
    return PhysicsConfig(
        radiation=RadiationConfig(scheme="none"),
        convection=ConvectionConfig(scheme=conv_scheme),
        turbulence=TurbulenceConfig(scheme="none"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )


# ---------------------------------------------------------------------------
# 1. init_physics_state shape contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "scheme",
    ["none", "sbm", "dca", "kuo", "mass_flux", "edmf"],
)
def test_init_physics_state_conv_prog_profile_shape(scheme):
    cfg = _make_config(scheme)
    ps = init_physics_state(NCOL, NLEV, cfg)
    assert hasattr(ps, "conv_prog_profile"), \
        "PhysicsState must expose conv_prog_profile after PR 0 migration"
    assert ps.conv_prog_profile.shape == (NCOL, NLEV), (
        f"Expected (ncol, nlev) = ({NCOL}, {NLEV}), got "
        f"{ps.conv_prog_profile.shape}"
    )
    # The legacy field name must not still exist.
    assert not hasattr(ps, "conv_prog"), (
        "Stale 'conv_prog' attribute exists; the migration must rename "
        "it to 'conv_prog_profile'"
    )


# ---------------------------------------------------------------------------
# 2. mass_flux / edmf scalar packed at [:, -1]
# ---------------------------------------------------------------------------

def test_mass_flux_init_scalar_at_surface_only():
    """``mass_flux`` packs ``M_c_init`` at the surface-adjacent slot,
    zeros aloft."""
    cfg = _make_config("mass_flux")
    M_c_init = cfg.convection.mass_flux.M_c_init
    ps = init_physics_state(NCOL, NLEV, cfg)
    profile = ps.conv_prog_profile

    # Surface-adjacent slot has the initial value.
    surf_slot = profile[:, -1]
    assert jnp.allclose(surf_slot, M_c_init), (
        f"profile[:, -1] should equal M_c_init={M_c_init}, "
        f"got {jnp.asarray(surf_slot[0])!s}"
    )

    # All slots above the surface are exactly zero.
    aloft = profile[:, :-1]
    assert jnp.all(aloft == 0.0), (
        "Slots above the surface-adjacent index must be zero in the "
        "scalar-carrying scheme convention; got "
        f"max abs = {float(jnp.max(jnp.abs(aloft)))}"
    )


def test_edmf_init_scalar_at_surface_only():
    """``edmf`` packs ``a_u_init`` at ``[:, -1]`` with zeros aloft."""
    cfg = _make_config("edmf")
    a_u_init = cfg.convection.edmf.a_u_init
    ps = init_physics_state(NCOL, NLEV, cfg)
    profile = ps.conv_prog_profile

    assert jnp.allclose(profile[:, -1], a_u_init)
    assert jnp.all(profile[:, :-1] == 0.0)


# ---------------------------------------------------------------------------
# 3. Stateless / "none" schemes initialize to zeros
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scheme", ["none", "sbm", "dca", "kuo"])
def test_stateless_scheme_init_all_zeros(scheme):
    cfg = _make_config(scheme)
    ps = init_physics_state(NCOL, NLEV, cfg)
    assert jnp.all(ps.conv_prog_profile == 0.0), (
        f"scheme={scheme!r}: stateless scheme must init conv_prog_profile "
        "to all-zeros"
    )


# ---------------------------------------------------------------------------
# 4. update_physics_state round-trip
# ---------------------------------------------------------------------------

def test_update_physics_state_round_trip():
    """The dict-update contract used by ``combined.py`` orchestrator
    correctly threads new values through and preserves untouched fields."""
    cfg = _make_config("mass_flux")
    ps = init_physics_state(NCOL, NLEV, cfg)

    new_profile = jnp.full((NCOL, NLEV), 0.42)
    updates = {"conv_prog_profile": new_profile}
    ps_out = update_physics_state(ps, updates)

    assert isinstance(ps_out, PhysicsState)
    assert jnp.allclose(ps_out.conv_prog_profile, new_profile)
    # Other fields must be carried over unchanged (object identity is
    # not required, but values are).
    assert jnp.allclose(ps_out.tke, ps.tke)
    assert jnp.allclose(ps_out.gwd_spectrum, ps.gwd_spectrum)


def test_update_physics_state_partial_update():
    """A partial update dict (e.g., only TKE) leaves conv_prog_profile
    unchanged."""
    cfg = _make_config("mass_flux")
    ps = init_physics_state(NCOL, NLEV, cfg)
    new_tke = jnp.full((NCOL, NLEV), 0.123)

    ps_out = update_physics_state(ps, {"tke": new_tke})

    assert jnp.allclose(ps_out.tke, new_tke)
    assert jnp.allclose(ps_out.conv_prog_profile, ps.conv_prog_profile)


def test_update_physics_state_none_returns_none():
    """A ``None`` input returns ``None`` (orchestrator-without-carry path)."""
    assert update_physics_state(None, {"tke": jnp.zeros(1)}) is None


# ---------------------------------------------------------------------------
# 5. End-to-end orchestrator threading: conv_prog_profile shape preserved
# ---------------------------------------------------------------------------

def test_orchestrator_returns_conv_prog_profile_shape():
    """A single physics step through ``make_physics`` returns a
    ``PhysicsState`` whose ``conv_prog_profile`` matches the
    ``(ncol, nlev)`` contract."""
    from legoesm.core.field import Field
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
    from legoesm.atmosphere.physics.combined import make_physics

    n = 4
    nlev = 8
    grid = create_cubed_sphere(n)
    sigma = create_sigma_coordinate(nlev)
    state = held_suarez_init(grid, sigma)
    tracers = {
        "q_v": Field(
            1e-3 * jnp.ones((6, n, n, nlev)), name="q_v",
            dims=("face", "x", "y", "level"), units="kg/kg",
        ),
        "q_c": Field(
            jnp.zeros((6, n, n, nlev)), name="q_c",
            dims=("face", "x", "y", "level"), units="kg/kg",
        ),
    }
    state = state._replace(tracers=tracers)

    cfg = _make_config("mass_flux")
    ncol = 6 * n * n
    ps_in = init_physics_state(ncol, nlev, cfg)

    physics_fn = make_physics(cfg, model_type="hydrostatic", dt=300.0)
    _, ps_out = physics_fn(state, grid, sigma, phys_state=ps_in)

    assert ps_out is not None
    assert ps_out.conv_prog_profile.shape == (ncol, nlev), (
        f"Orchestrator must preserve (ncol, nlev) shape, got "
        f"{ps_out.conv_prog_profile.shape}"
    )
    # mass_flux scheme keeps writing scalar at [:, -1] only; aloft
    # remains zero.
    assert jnp.all(ps_out.conv_prog_profile[:, :-1] == 0.0), (
        "mass_flux bridge must pack the scalar at [:, -1] with zeros "
        "aloft after the schema migration"
    )
