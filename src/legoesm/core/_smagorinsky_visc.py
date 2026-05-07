"""Smagorinsky-style adaptive Laplacian viscosity (FV3_3D iter 57).

Computes a per-cell ``A_h`` field from the local strain-rate tensor
of the (u, v) wind components on the cubed-sphere C-D grid:

    A_h(i, j) = c_s * dx² * sqrt(D11² + 2*D12² + D22²)

where the strain-rate tensor components are::

    D11 = ∂u/∂x
    D22 = ∂v/∂y
    D12 = 0.5 * (∂u/∂y + ∂v/∂x)

The coefficient ``c_s`` is the Smagorinsky constant (typically
0.1-0.4 for atmosphere; FV3 uses ~0.2 in some configurations).

This helper is NOT YET wired into the dycore.  It's the building
block for an iter-58+ opt-in adaptive A_h that auto-scales with
local flow conditions, addressing the iter-51 codex meta-review
"open generalization gap" — the iter-33 ``10x A_h`` calibration is
case-specific; Smagorinsky generalizes by responding to actual
strain.

Implementation notes
--------------------

* Uses centered finite differences on the corner-staggered C-D
  grid: ``∂u/∂x ≈ (u(i+1) - u(i-1)) / (2 dx)``.

* Halo padding via ``pad_halo`` so that boundary cells use cross-
  panel values rather than edge-mode replication.  This matters
  for FV3 fidelity at panel boundaries.

* Returns A_h at corner-staggered locations (same shape as input
  u, v: ``(6, n+1, n+1)``).

* Single-level (2D) helper; ``vmap`` the trailing axis for 3D.

* Constant-angle approximation: assumes the C-D grid metric is
  locally orthogonal.  For the full FV3 metric (including cosa
  cross-correction), see ``fv3_divergence_corner_2d``.  The
  approximation is acceptable for this STABILITY damping
  application — the goal is "enough viscosity to stabilize
  interior modes", not "metric-perfect Laplacian".

References
----------

* Smagorinsky (1963) — basic deformation-based eddy viscosity.
* FV3 ``sw_core.F90:smag_corner`` — FV3's full implementation
  with sin_sg / cos_sg metric corrections.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.grids.cubed_sphere_cdgrid import CubedSphereCDGrid
from legoesm.grids.halo import pad_halo


def compute_smagorinsky_ah_2d(
    u_corner: jnp.ndarray,
    v_corner: jnp.ndarray,
    cdgrid: CubedSphereCDGrid,
    c_s: float,
) -> jnp.ndarray:
    """Smagorinsky-style adaptive A_h at corner-staggered locations (2D).

    Parameters
    ----------
    u_corner : jnp.ndarray, shape ``(6, n+1, n+1)``
        u-component at C-D corners.
    v_corner : jnp.ndarray, shape ``(6, n+1, n+1)``
        v-component at C-D corners.
    cdgrid : CubedSphereCDGrid
    c_s : float
        Smagorinsky coefficient.  Typical 0.1-0.4.  When 0,
        returns 0 everywhere (allows cheap on/off gating).

    Returns
    -------
    ah : jnp.ndarray, same shape as ``u_corner`` — adaptive A_h
        at each corner [m^2 / s].
    """
    if c_s <= 0.0:
        return jnp.zeros_like(u_corner)

    n = cdgrid.n

    # Halo-pad u, v with cross-panel halo (n+3, n+3 each).
    u_pad = pad_halo(u_corner)
    v_pad = pad_halo(v_corner)

    # Centered finite differences on the (n+1, n+1) corner grid.
    # u_pad has padded indices [0..n+2]; corner (i, j) is at
    # padded (i+1, j+1).  Centered diff of u in i direction:
    #   du_dx(i, j) = (u(i+1, j) - u(i-1, j)) / (2 * dx_corner)
    # Padded indices: u_pad[..., i+2, j+1] - u_pad[..., i, j+1].
    du_di = (
        u_pad[:, 2:n + 3, 1:n + 2] - u_pad[:, 0:n + 1, 1:n + 2]
    )  # (6, n+1, n+1)
    du_dj = (
        u_pad[:, 1:n + 2, 2:n + 3] - u_pad[:, 1:n + 2, 0:n + 1]
    )
    dv_di = (
        v_pad[:, 2:n + 3, 1:n + 2] - v_pad[:, 0:n + 1, 1:n + 2]
    )
    dv_dj = (
        v_pad[:, 1:n + 2, 2:n + 3] - v_pad[:, 1:n + 2, 0:n + 1]
    )

    # Convert finite-difference numerators to physical derivatives
    # by dividing by 2 * dx_corner.  At corner (i, j), the
    # corner-to-corner distance in i is ``cdgrid.base.dx`` (cell
    # width); in j is ``cdgrid.base.dy``.  For a constant-angle
    # approximation, use cdgrid's mean cell width.
    # Use dxc / dyc which are at u/v faces and pad to corners.
    dxc = jnp.pad(
        cdgrid.dxc, [(0, 0), (0, 0), (0, 1)], mode="edge",
    )                                                       # (6, n+1, n+1)
    dyc = jnp.pad(
        cdgrid.dyc, [(0, 0), (0, 1), (0, 0)], mode="edge",
    )                                                       # (6, n+1, n+1)
    # FD numerators above are (u(i+1) - u(i-1)) which spans 2*dx;
    # divide by 2*dxc to get ∂u/∂x.
    du_dx = du_di / (2.0 * dxc)
    du_dy = du_dj / (2.0 * dyc)
    dv_dx = dv_di / (2.0 * dxc)
    dv_dy = dv_dj / (2.0 * dyc)

    # Strain-rate tensor components.
    D11 = du_dx
    D22 = dv_dy
    D12 = 0.5 * (du_dy + dv_dx)

    # |D| = sqrt(D11^2 + 2*D12^2 + D22^2).
    strain_mag = jnp.sqrt(D11 * D11 + 2.0 * D12 * D12 + D22 * D22)

    # dx² weighting using mean of dxc * dyc at corner.
    dx_squared = dxc * dyc

    return c_s * dx_squared * strain_mag


def compute_smagorinsky_ah_3d(
    u_corner_3d: jnp.ndarray,
    v_corner_3d: jnp.ndarray,
    cdgrid: CubedSphereCDGrid,
    c_s: float,
) -> jnp.ndarray:
    """3D wrapper for ``compute_smagorinsky_ah_2d`` over levels.

    Parameters
    ----------
    u_corner_3d : (6, n+1, n+1, nlev)
    v_corner_3d : (6, n+1, n+1, nlev)
    cdgrid : CubedSphereCDGrid
    c_s : float

    Returns
    -------
    ah_3d : (6, n+1, n+1, nlev)
    """
    if c_s <= 0.0:
        return jnp.zeros_like(u_corner_3d)
    return jax.vmap(
        lambda u, v: compute_smagorinsky_ah_2d(u, v, cdgrid, c_s),
        in_axes=-1, out_axes=-1,
    )(u_corner_3d, v_corner_3d)
