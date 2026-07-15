"""Terrain-metric (z*-Jacobian) consistency of NH horizontal continuity.

With a terrain-following z* coordinate the Jacobian J = dz/dz* =
(H - z_s)/H varies horizontally and conservative continuity reads

    drho/dt = -(1/J) [ div_h(J rho v) + d(rho w)/dz* ].

The vertical leg has always carried its 1/J (acoustic substeps); these
tests pin the HORIZONTAL leg to the same metric treatment.  A plain
-div_h(rho v) horizontal leg silently creates/destroys dry mass over
topography (measured WITHOUT the J weighting, 20-step protocol below:
MPAS 2.5e-7 relative drift, cube 5.5e-8; WITH it: 0.0 and 3.9e-16)
while flat terrain (J = 1) is bit-for-bit unaffected.

Protocol notes
--------------
* ``fix_mass=False`` everywhere — the anchored fixer would mask the leak
  (``test_nh_mass_conservation_anchored.py`` gates the fixer instead).
* The Gaussian hill is widened to a 2000 km halfwidth so it is RESOLVED
  on the coarse test grids (the TC1 default 72 km is sub-grid at C12 /
  level-4 Voronoi: z_s rounds to ~0 and the terrain paths never
  activate).  J spans [0.95, 1.0].
* The cube grid uses duogrid so cross-face PPM fluxes are synchronized;
  without it a pre-existing O(2e-7) face-boundary flux mismatch
  (scalar-halo wind basis rotation) dominates the budget and buries the
  J signal.  Measured duogrid flat-terrain floor: ~6e-16.
* The dry-mass integrals ``compute_nh_dry_mass``/``compute_nh_dry_mass_mpas``
  already include J (M = int J*(rho_ref+rho')*dz*dA) — verified, no fix
  was needed on the diagnostic side.
* DCMIP-2025 TC1 init (Gaussian/Schaer mountain, rho'=theta'=0 rest
  perturbation + gentle zonal wind) provides the z_s != 0
  hydrostatically balanced start; ``mountain_height=0`` gives exact
  J = 1.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

DRIFT_TOL = 1e-10
N_STEPS = 20
# 2000 km halfwidth: hill resolved at C12 / level-4 Voronoi (see module docstring).
HILL = {"mountain_height": 2000.0, "mountain_halfwidth": 2000.0e3}
FLAT = {"mountain_height": 0.0}


def _rel_drift(m0: float, m1: float) -> float:
    return abs(m1 - m0) / max(abs(m0), 1.0)


def _mpas_setup(mesh_level: int, terrain: dict):
    from legoesm.atmosphere.dynamics.gcm.compressible_euler_mpas import (
        MPASCompressibleEulerConfig,
        MPASCompressibleEulerModel,
    )
    from legoesm.grids.voronoi import create_voronoi_mesh

    from tests.atmosphere.nonhydrostatic.test_cases.dcmip2025.test_case_1_mpas import (
        dcmip25_tc1_init_mpas,
    )

    mesh = create_voronoi_mesh(mesh_level)
    state, hcoord, tmetric = dcmip25_tc1_init_mpas(
        mesh, n_levels=10, params=terrain,
    )
    cfg = MPASCompressibleEulerConfig(n_acoustic_substeps=10, fix_mass=False)
    model = MPASCompressibleEulerModel(mesh, hcoord, tmetric, cfg)
    return mesh, state, hcoord, tmetric, model


def _cube_setup(n: int, terrain: dict):
    from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
        CDGridCompressibleEulerConfig,
        CDGridCompressibleEulerModel,
    )
    from legoesm.grids.cubed_sphere import create_cubed_sphere

    from tests.atmosphere.nonhydrostatic.test_cases.dcmip2025.test_case_1 import (
        dcmip25_tc1_init,
    )

    grid = create_cubed_sphere(n, use_duogrid=True)
    state, hcoord, tmetric = dcmip25_tc1_init(
        grid, n_levels=10, params=terrain,
    )
    cfg = CDGridCompressibleEulerConfig(
        n_acoustic_substeps=10, semi_implicit_acoustic=True, fix_mass=False,
    )
    model = CDGridCompressibleEulerModel(grid, hcoord, tmetric, cfg)
    return grid, state, hcoord, tmetric, model


# ---------------------------------------------------------------------------
# 1. Dry-mass conservation over a mountain, NO mass fixer.
# ---------------------------------------------------------------------------

def test_mpas_terrain_dry_mass_drift():
    """MPAS NH over a resolved 2 km Gaussian hill: inherent dry-mass
    drift at the fp64 floor (measured 0.0 exactly).  Without the
    J-weighted horizontal continuity this drifts at 2.5e-7
    (dt=60 s, 20 steps, level-4 mesh)."""
    from legoesm.core.conservation import compute_nh_dry_mass_mpas

    mesh, state, hcoord, tmetric, model = _mpas_setup(4, HILL)

    def _mass(s):
        return float(compute_nh_dry_mass_mpas(
            s.rho_prime.data, hcoord, tmetric, mesh,
        ))

    m0 = _mass(state)
    for _ in range(N_STEPS):
        state = model.step(state, 60.0)
    assert jnp.all(jnp.isfinite(state.w.data))
    assert _rel_drift(m0, _mass(state)) < DRIFT_TOL


def test_cube_terrain_dry_mass_drift():
    """Cube NH (duogrid) over a resolved 2 km Gaussian hill: inherent
    dry-mass drift at the fp64 floor (measured 3.9e-16).  Without the
    J-weighted horizontal continuity this drifts at 5.5e-8
    (dt=30 s, 20 steps, C12)."""
    from legoesm.core.conservation import compute_nh_dry_mass

    grid, state, hcoord, tmetric, model = _cube_setup(12, HILL)

    def _mass(s):
        return float(compute_nh_dry_mass(
            s.rho_prime.data, hcoord, tmetric, grid,
        ))

    m0 = _mass(state)
    for _ in range(N_STEPS):
        state = model.step(state, 30.0)
    assert jnp.all(jnp.isfinite(state.w.data))
    assert _rel_drift(m0, _mass(state)) < DRIFT_TOL


# ---------------------------------------------------------------------------
# 2. Rest state (u = 0, rho' = theta' = 0) is a bitwise fixed point —
#    over terrain AND flat.  Known-value guard: the J = 1 (flat) case
#    certifies the J-weighted code path degenerates exactly to the
#    unweighted one; the mountain case certifies a resting atmosphere
#    over a hill feels no spurious metric source.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("terrain", [FLAT, HILL], ids=["flat", "mountain"])
def test_mpas_rest_state_is_stationary(terrain):
    _, state, _, _, model = _mpas_setup(3, terrain)
    state = state._replace(u=state.u.replace(data=jnp.zeros_like(state.u.data)))

    stepped = model.step(state, 60.0)
    assert jnp.array_equal(stepped.u.data, state.u.data)
    assert jnp.array_equal(stepped.w.data, state.w.data)
    assert jnp.array_equal(stepped.theta_prime.data, state.theta_prime.data)
    assert jnp.array_equal(stepped.rho_prime.data, state.rho_prime.data)


@pytest.mark.parametrize("terrain", [FLAT, HILL], ids=["flat", "mountain"])
def test_cube_rest_state_is_stationary(terrain):
    _, state, _, _, model = _cube_setup(8, terrain)
    zeros = jnp.zeros_like(state.u.data)
    state = state._replace(
        u=state.u.replace(data=zeros), v=state.v.replace(data=zeros),
    )

    stepped = model.step(state, 30.0)
    assert jnp.array_equal(stepped.u.data, state.u.data)
    assert jnp.array_equal(stepped.v.data, state.v.data)
    assert jnp.array_equal(stepped.w.data, state.w.data)
    assert jnp.array_equal(stepped.theta_prime.data, state.theta_prime.data)
    assert jnp.array_equal(stepped.rho_prime.data, state.rho_prime.data)
