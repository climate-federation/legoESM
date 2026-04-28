"""GM/Redi isopycnal mixing on the MPAS Voronoi mesh — API skeleton.

**Status (2026-04-28):** This module defines the public API for the
MPAS port of GM/Redi but does not yet implement it.  Calling any
function below raises ``NotImplementedError`` with a pointer to the
implementation plan at ``docs/ocean_experiments/gm_redi_mpas_plan.md``.

The signatures mirror ``gm_redi_tracer_tendency_latlon`` in
``gm_redi_latlon_cgrid.py`` so the dycore hook in
``ocean_model_mpas.py`` is mechanical once the body is filled in:

    if self.config.gm_redi is not None:
        from legoesm.ocean.physics.lateral_mixing.gm_redi_mpas import (
            gm_redi_tracer_tendency_mpas,
        )
        dT_gm, dS_gm = gm_redi_tracer_tendency_mpas(...)
        T_mid = T_mid + dt * dT_gm * mask
        S_mid = S_mid + dt * dS_gm * mask

The skeleton intentionally raises rather than silently no-op'ing —
this matches the lesson learned with the surface-forcing scheme dropout
(`make_mpas_ocean_physics` used to silently no-op on
``scheme='combined'``, producing a bit-frozen 1-yr global overturning
run before the bug was caught).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    from legoesm.grids.voronoi import VoronoiMesh
    from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
    from legoesm.ocean.vertical import OceanZStarCoordinate


_PLAN = "docs/ocean_experiments/gm_redi_mpas_plan.md"


def _not_implemented(name: str) -> None:
    raise NotImplementedError(
        f"GM/Redi on MPAS is not yet implemented (called: {name}). "
        f"See {_PLAN} for the design and phasing."
    )


def compute_isopycnal_slopes_mpas(
    rho: jnp.ndarray,
    mask: jnp.ndarray,
    z_coord: "OceanZStarCoordinate",
    jacobian: jnp.ndarray,
    mesh: "VoronoiMesh",
    cfg: "GMRediConfig",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Edge-normal isopycnal slope and slope-magnitude squared.

    Returns
    -------
    S_n : (nEdges, nlev-1)
        Edge-normal slope at interior interfaces.
    taper : (nEdges, nlev-1)
        Danabasoglu-McWilliams 1995 taper factor [0, 1].

    See Phase 1 in ``docs/ocean_experiments/gm_redi_mpas_plan.md``.
    """
    _not_implemented("compute_isopycnal_slopes_mpas")


def gm_redi_tracer_tendency_centered_mpas(
    q: jnp.ndarray,
    S_n: jnp.ndarray,
    mask: jnp.ndarray,
    edge_mask: jnp.ndarray,
    z_coord: "OceanZStarCoordinate",
    jacobian: jnp.ndarray,
    mesh: "VoronoiMesh",
    kappa_GM,
    kappa_Redi: float,
) -> jnp.ndarray:
    """Centred GM+Redi tracer tendency on Voronoi (Phase 2).

    Mirrors ``gm_redi_tracer_tendency_latlon_cgrid``: horizontal flux
    in the edge-normal direction, vertical flux at cell-centred
    interfaces.  Uses the small-slope tensor.

    See Phase 2 in ``docs/ocean_experiments/gm_redi_mpas_plan.md``.
    """
    _not_implemented("gm_redi_tracer_tendency_centered_mpas")


def gm_redi_tracer_tendency_triads_mpas(
    q: jnp.ndarray,
    rho: jnp.ndarray,
    mask: jnp.ndarray,
    edge_mask: jnp.ndarray,
    z_coord: "OceanZStarCoordinate",
    jacobian: jnp.ndarray,
    mesh: "VoronoiMesh",
    kappa_GM,
    kappa_Redi: float,
    S_max: float,
) -> jnp.ndarray:
    """Triad-discretised GM+Redi tracer tendency (Phase 5).

    The Griffies 1998 triad scheme is the open scientific subproblem
    on Voronoi geometry — see the dedicated phase in the plan doc.
    """
    _not_implemented("gm_redi_tracer_tendency_triads_mpas")


def gm_redi_tracer_tendency_mpas(
    T: jnp.ndarray,
    S: jnp.ndarray,
    eta: jnp.ndarray,
    H_bathy: jnp.ndarray,
    mesh: "VoronoiMesh",
    z_coord: "OceanZStarCoordinate",
    cfg: "GMRediConfig",
    *,
    eos: str = "wright",
    eos_linear=None,
    mask: jnp.ndarray | None = None,
    edge_mask: jnp.ndarray | None = None,
    f_coriolis: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Top-level MPAS GM/Redi entry point.

    Composes density → slopes → optional Visbeck → tracer tendency.
    Signature mirrors ``gm_redi_tracer_tendency_latlon`` so the
    dycore hook in ``ocean_model_mpas.py`` is mechanical once the
    body is filled in.

    Returns
    -------
    dT_dt, dS_dt : (nCells, nlev)
    """
    _not_implemented("gm_redi_tracer_tendency_mpas")
