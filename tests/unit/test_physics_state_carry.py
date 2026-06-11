"""Issue #405 regression: PhysicsState must be seeded and carried.

Root cause: ``update_physics_state(None, ...)`` returns ``None``, so a
run loop that starts the carry at ``None`` (or never threads it) keeps
``None`` forever — every stateful scheme silently reseeds its
prognostic fields each timestep.  Three guarantees pinned here:

1. ``update_physics_state(None, ...)`` returning ``None`` is the
   documented sentinel (the trap the loops must not fall into).
2. The MPAS model step threads a SEEDED carry: the output carry
   depends on the input carry (memory), and a perturbed prognostic
   column survives chained steps.
3. ``ModelDriver._refuse_stateful_physics_unthreaded`` raises for
   prognostic-carry schemes and passes diagnostic ones — the loud
   tripwire on the loops that do not thread the carry yet.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.physics_state import (
    PhysicsState,
    init_physics_state,
    update_physics_state,
)


def _tke_physics_config():
    from legoesm.atmosphere.physics.combined import PhysicsConfig
    from legoesm.atmosphere.physics.turbulence import TurbulenceConfig

    return PhysicsConfig(turbulence=TurbulenceConfig(scheme="tke"))


def test_update_physics_state_none_sentinel():
    ps = init_physics_state(8, 4, _tke_physics_config())
    assert update_physics_state(None, {"tke": ps.tke}) is None
    out = update_physics_state(ps, {})
    assert isinstance(out, PhysicsState)


def test_mpas_step_threads_seeded_carry():
    """Two MPAS steps with a seeded carry: the prognostic TKE evolves
    and step 2 consumes step 1's output (no silent reseed)."""
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.primitive_eq_mpas import (
        MPASPrimitiveEquationModel,
        MPASPrimitiveEquationConfig,
    )
    from legoesm.atmosphere.held_suarez import held_suarez_init_mpas
    from legoesm.atmosphere.physics.combined import make_physics

    mesh = create_voronoi_mesh(3)
    sigma = create_sigma_coordinate(8)
    model = MPASPrimitiveEquationModel(
        mesh, sigma, MPASPrimitiveEquationConfig())
    state = held_suarez_init_mpas(mesh, sigma)

    cfg = _tke_physics_config()
    # model_type="mpas" — the driver's own call (model_driver.py:3067);
    # MPAS states carry edge-normal u with v=None, which the
    # "hydrostatic" pipeline would dereference.
    physics_fn = make_physics(cfg, model_type="mpas", dt=1.0)
    ncol = int(state.T.data.shape[0])
    nlev = int(state.T.data.shape[1])
    seed = init_physics_state(ncol, nlev, cfg)

    # The issue-405 property is MEMORY: the carry fed in must shape the
    # carry coming out.  From a rest state TKE has zero shear production
    # and at dt=300 s the dissipation term removes any perturbation in a
    # single step (both branches legitimately floor at tke_min), so the
    # step uses dt=1 s — the perturbed column must then SURVIVE the step
    # iff the scheme actually consumed the input carry.
    # Same dynamics state, two different input carries -> outputs must
    # differ, and the perturbed column must remember its perturbation.
    seed_pert = seed._replace(tke=seed.tke.at[0, :].set(1e-2))

    _ = model.step(state, 1.0, physics_fn=physics_fn, phys_state=seed)
    ps_a = model._phys_state
    _ = model.step(state, 1.0, physics_fn=physics_fn,
                   phys_state=seed_pert)
    ps_b = model._phys_state

    assert ps_a is not None and ps_b is not None, (
        "MPAS step dropped the seeded carry (model._phys_state is None)"
    )
    assert not np.array_equal(np.asarray(ps_b.tke), np.asarray(ps_a.tke)), (
        "Output carry is independent of the input carry — the scheme "
        "is being reseeded every step (issue #405)"
    )
    assert (float(np.max(np.asarray(ps_b.tke)[0]))
            > float(np.max(np.asarray(ps_a.tke)[0]))), (
        "Perturbed TKE column lost its memory through the step"
    )
    # Chain a second step on the perturbed branch: memory persists.
    s1 = model.step(state, 1.0, physics_fn=physics_fn,
                    phys_state=seed_pert)
    _ = model.step(s1, 1.0, physics_fn=physics_fn,
                   phys_state=model._phys_state)
    ps2 = model._phys_state
    assert ps2 is not None
    assert (float(np.max(np.asarray(ps2.tke)[0]))
            > float(np.max(np.asarray(ps_a.tke)[0])))
    assert np.all(np.isfinite(np.asarray(ps2.tke)))


@pytest.mark.parametrize(
    "turb,conv,gwd,should_raise",
    [
        ("tke", "none", "none", True),
        ("mynn25", "none", "none", True),
        ("none", "bechtold", "none", True),
        ("none", "mass_flux", "none", True),
        ("none", "edmf", "none", True),
        ("none", "none", "prognostic_spectral", True),
        ("louis", "sbm", "none", False),
        ("none", "none", "linear", False),
    ],
)
def test_refusal_guard(turb, conv, gwd, should_raise):
    from legoesm.driver.model_driver import ModelDriver

    class _Cfg:
        turbulence = turb
        convection = conv
        gravity_wave_drag = gwd

    if should_raise:
        with pytest.raises(NotImplementedError, match="405"):
            ModelDriver._refuse_stateful_physics_unthreaded(_Cfg())
    else:
        ModelDriver._refuse_stateful_physics_unthreaded(_Cfg())


def test_refusal_guard_normalizes_scheme_objects():
    """PhysicsConfig-style sub-configs (with .scheme) are normalized."""
    from legoesm.driver.model_driver import ModelDriver

    class _Sub:
        def __init__(self, scheme):
            self.scheme = scheme

    class _Cfg:
        turbulence = _Sub("tke")
        convection = _Sub("none")
        gravity_wave_drag = _Sub("none")

    with pytest.raises(NotImplementedError, match="405"):
        ModelDriver._refuse_stateful_physics_unthreaded(_Cfg())
