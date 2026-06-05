"""FV3_3D iter 606: terrain (phis) del-2 / del-4 smoothing.

Faithful port of FV3 ``del2_cubed_sphere`` and ``del4_cubed_sphere``
(``tools/fv_surf_map.F90:817+``).  Applied at IC load time to smooth
surface geopotential before the dycore reads it — prevents stair-
step terrain artifacts and improves dycore stability.

Usage::

    phis_smooth = terrain_filter(phis, grid, n_iter=4, nord=2)

where:
- ``n_iter`` matches FV3 ``n_zs_filter`` namelist parameter.
- ``nord=2`` → del-2 smoothing; ``nord=4`` → del-4.
- Coefficient defaults to ``0.20 · min(grid.area)`` (FV3
  ``cnst_0p20 · da_min`` per ``external_ic.F90:609``).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.core.operators import laplacian_compact


def terrain_filter(
    phis: jax.Array,
    grid,
    n_iter: int = 4,
    nord: int = 2,
    cd: float | None = None,
) -> jax.Array:
    """Apply FV3-style del-N terrain filter ``n_iter`` times.

    Parameters
    ----------
    phis : jax.Array, shape (6, n, n)
        Surface geopotential or any 2D scalar.
    grid : CubedSphereGrid
    n_iter : int, default 4
        Number of applications.  FV3 ``n_zs_filter``.  0 = no-op.
    nord : int, default 2
        2 = del-2 (Laplacian); 4 = del-4 (biharmonic).  FV3
        ``nord_zs_filter`` namelist parameter.
    cd : float, optional
        Diffusion coefficient.  Default = ``0.20 · min(grid.area)``
        matching FV3 ``cnst_0p20 · da_min`` at ``external_ic.F90:609``.

    Returns
    -------
    phis_smooth : jax.Array, shape (6, n, n)
        Smoothed phis.

    Notes
    -----
    Differentiable end-to-end.  Bit-for-bit identity when n_iter=0.
    The Laplacian uses legoESM ``laplacian_compact`` (compact
    second-difference stencil) — FV3 uses metric-aware flux-form;
    the two agree to discretization order for smooth fields.
    """
    if n_iter <= 0:
        return phis
    if cd is None:
        cd = 0.20 * float(jnp.min(grid.area))

    q = phis
    for _ in range(n_iter):
        if nord == 2:
            lap = laplacian_compact(q, grid)
            q = q + cd * lap
        elif nord == 4:
            # del-4 = -del-2 applied to del-2: smoother but less
            # extra suppression at the boundaries.
            lap1 = laplacian_compact(q, grid)
            lap2 = laplacian_compact(lap1, grid)
            q = q - cd * lap2
        else:
            raise ValueError(
                f"nord must be 2 (del-2) or 4 (del-4); got {nord}"
            )
    return q
