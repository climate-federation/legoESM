"""NEMO state -> legoESM lat-lon C-grid state bridge (differentiable-NEMO harness).

Assembles a :class:`~legoesm.ocean.state.LatLonCGridOceanState` (plus its
beta-plane geometry and z-coordinate) from the halo-stripped :class:`NemoGrid` /
:class:`NemoState` produced by :mod:`nemo_io`, so legoESM can be handed the exact
state a NEMO run is in for a single-step tendency comparison.

Conventions reconciled here (the model/harness split of
``docs/ocean/fidelity/oracle_recipe_strategy.md`` keeps these in the harness):

* **Geometry.** GYRE (and NEMO idealised beta-plane configs) use a uniform
  Cartesian metric + ``f(y) = f0 + beta*y`` — exactly
  :func:`create_beta_plane_cgrid_geometry`. ``f0``/``beta`` are recovered from
  NEMO's own ``ff_t`` (linear in y) and the build is VERIFIED against it.
* **Velocity staggering.** NEMO ``u(i)`` sits on the **east** face of T-cell
  ``i``; legoESM ``u_face`` is the **west**-face array of length ``n_lon+1``. So
  NEMO ``u`` maps to ``u_face[:, 1:]`` and ``u_face[:, 0]`` is the west wall
  (prepend a zero column) — the OPPOSITE end from the MITgcm bridge (MITgcm ``U``
  is the west face). ``v`` analogously: NEMO north-face ``v`` -> prepend a south
  wall row. GYRE is a closed basin, so all four boundary faces are walls (0).
* **Vertical.** ``z*`` from NEMO ``e3t_1d``. GYRE is ``key_linssh`` (thicknesses
  FIXED, do not move with eta), whereas legoESM z* moves them. The *thickness
  metric* error is tiny (``~eta/H ~ 1e-5``), but the *free-surface / continuity
  formulation* differs (linssh keeps thicknesses fixed in the ssh and surface-w
  equations) — invisible at correlation tier, but it caps a machine-precision
  tier-3 tendency match. A fixed-thickness (linssh) z-coord option is a later
  step; until then this residual is expected and must not be read as a physics bug.

Scope: **flat-bottom only** (``key_vco_1d``). A NEMO config with
topography/partial cells is rejected (see :func:`bridge_nemo_to_legoesm`).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from legoesm.grids.latlon import (
    LatLonCGridGeometry,
    create_beta_plane_cgrid_geometry,
)
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import neumann_fill_cgrid
from legoesm.ocean.fidelity.nemo_io import NemoGrid, NemoState
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanState
from legoesm.ocean.vertical import create_z_star_from_thicknesses


class NemoBridgeOutput(NamedTuple):
    geometry: LatLonCGridGeometry
    z_coord: object
    state: LatLonCGridOceanState
    land_mask: np.ndarray          # (n_lat, n_lon) surface wet mask
    f_match_max_abs: float         # max|geom.f_T - NEMO ff_t| — build self-check


def _beta_plane_params(grid: NemoGrid):
    """Recover (n_lat, n_lon, dx_m, dy_m, f0, beta) from a NEMO beta-plane mesh.

    ``ff_t`` is linear in y and constant along x on a beta-plane; fit f0/beta with
    ``y_origin_m = 0`` and cell-centre ``y_c[j] = (j + 1/2) dy`` so the resulting
    ``create_beta_plane_cgrid_geometry`` reproduces ``ff_t`` exactly.
    """
    n_lat, n_lon = grid.gphit.shape
    dx_m = float(np.mean(grid.e1t))
    dy_m = float(np.mean(grid.e2t))
    ff_col = np.asarray(grid.ff_t)[:, 0]                     # (n_lat,)
    beta = float((ff_col[-1] - ff_col[0]) / ((n_lat - 1) * dy_m))
    f0 = float(ff_col[0] - beta * 0.5 * dy_m)               # f at y_c[0] = ff_col[0]
    return n_lat, n_lon, dx_m, dy_m, f0, beta


def _u_east_to_face(nemo_u: np.ndarray) -> np.ndarray:
    """NEMO east-face u ``(n_lat, n_lon, nz)`` -> legoESM ``u_face (n_lat, n_lon+1, nz)``.

    ``u_face[:, 0]`` = west wall (0); ``u_face[:, 1:]`` = NEMO u (whose last
    column is the east wall). Closed basin -> the west wall is a true wall.
    """
    wall = np.zeros_like(nemo_u[:, :1, :])
    return np.concatenate([wall, nemo_u], axis=1)


def _v_north_to_face(nemo_v: np.ndarray) -> np.ndarray:
    """NEMO north-face v ``(n_lat, n_lon, nz)`` -> legoESM ``v_face (n_lat+1, n_lon, nz)``."""
    wall = np.zeros_like(nemo_v[:1, :, :])
    return np.concatenate([wall, nemo_v], axis=0)


def bridge_nemo_to_legoesm(
    grid: NemoGrid,
    state: NemoState,
    *,
    f_tol: float = 1e-9,
) -> NemoBridgeOutput:
    """Build a legoESM C-grid ocean state from a halo-stripped NEMO grid+state.

    Raises ``ValueError`` if the reconstructed beta-plane Coriolis does not match
    NEMO's ``ff_t`` to ``f_tol`` (a guard against a wrong f0/beta/dy).
    """
    n_lat, n_lon, dx_m, dy_m, f0, beta = _beta_plane_params(grid)

    geom = create_beta_plane_cgrid_geometry(
        n_lat, n_lon, dx_m=dx_m, dy_m=dy_m, f0=f0, beta=beta,
        y_origin_m=0.0, cartesian_pseudo_lat=True,
    )
    f_err = float(np.max(np.abs(np.asarray(geom.f_T) - np.asarray(grid.ff_t))))
    if f_err > f_tol:
        raise ValueError(
            f"beta-plane Coriolis mismatch vs NEMO ff_t: max|Δ|={f_err:.3e} > "
            f"{f_tol:.1e}. f0={f0:.6e}, beta={beta:.6e}, dy={dy_m:.1f}."
        )
    # f_v (vorticity-point Coriolis, used by the rel-vort flux) is f0+beta*y_g by
    # construction; f_T matching confirms the LAW, but verify NEMO's ff_f obeys the
    # SAME (f0, beta) so a y_g stagger error can't slip through the f_T gate.
    ff_f_col = np.asarray(grid.ff_f)[:, 0]
    if ff_f_col.size > 1:
        beta_f = float((ff_f_col[-1] - ff_f_col[0]) / ((ff_f_col.size - 1) * dy_m))
        if abs(beta_f - beta) > max(f_tol, 1e-6 * abs(beta)):
            raise ValueError(
                f"NEMO ff_f slope {beta_f:.6e} != ff_t slope {beta:.6e}; "
                "f-point / T-point Coriolis are inconsistent — check the mesh."
            )

    # Flat-bottom guard: this bridge builds a single-depth basin from the SURFACE
    # mask (key_vco_1d GYRE). "Flat" = every wet column is wet for the TOP n_wet
    # levels and dry below — a UNIFORM bottom depth (GYRE has 30 wet of 31 levels,
    # the 31st below-bottom). Topography/partial cells (varying n_wet, or interior
    # holes) would corrupt PGF/continuity — reject them loudly.
    tmask = np.asarray(grid.tmask) > 0.5                     # (n_lat, n_lon, nlev)
    land_mask = tmask[:, :, 0]                               # surface-wet (n_lat, n_lon)
    n_wet_col = tmask.sum(axis=2)                            # wet levels per column
    n_wet = int(n_wet_col[land_mask].max()) if land_mask.any() else 0
    # (a) every wet column has the SAME bottom depth; (b) wet cells are the top
    # n_wet (contiguous from the surface, no interior/topographic holes).
    top_contig = np.zeros_like(tmask)
    top_contig[:, :, :n_wet] = land_mask[:, :, None]
    if not (np.array_equal(n_wet_col[land_mask], np.full(int(land_mask.sum()), n_wet))
            and np.array_equal(tmask & land_mask[:, :, None], top_contig)):
        raise ValueError(
            "NEMO tmask is not flat-bottomed (varying bottom depth or interior "
            "masked cells): this bridge only supports uniform-depth bathymetry "
            "(key_vco_1d). Add partial-cell/mbathy H_bathy support before "
            "bridging topographic configs."
        )

    z_coord = create_z_star_from_thicknesses(np.asarray(grid.e3t_1d))
    H_max = float(np.sum(np.asarray(grid.e3t_1d)[:n_wet]))   # depth of the n_wet wet cells

    base = rest_state_latlon_cgrid_ocean(
        geom, z_coord, H_max=H_max, land_mask_override=jnp.asarray(land_mask),
    )

    # Neumann-fill T/S over land so the 0.0 NEMO stores on masked cells cannot
    # contaminate legoESM's wide high-order tracer stencils (the #480 T=0 bug).
    mask3 = jnp.asarray(tmask)                               # (n_lat, n_lon, nlev) bool
    T_fill = neumann_fill_cgrid(jnp.asarray(state.T), mask3, geom)
    S_fill = neumann_fill_cgrid(jnp.asarray(state.S), mask3, geom)

    u_face = _u_east_to_face(np.asarray(state.u))
    v_face = _v_north_to_face(np.asarray(state.v))
    st = base._replace(
        T=base.T.replace(data=T_fill),
        S=base.S.replace(data=S_fill),
        u=base.u.replace(data=jnp.asarray(u_face)),
        v=base.v.replace(data=jnp.asarray(v_face)),
        eta=base.eta.replace(data=jnp.asarray(state.ssh)),
    )

    return NemoBridgeOutput(
        geometry=geom, z_coord=z_coord, state=st,
        land_mask=land_mask, f_match_max_abs=f_err,
    )


__all__ = ("NemoBridgeOutput", "bridge_nemo_to_legoesm")
