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
import numpy as np

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


def terrain_filter_duo(hs6, gs6, ectx, *, n_iter: int, cd: float | None = None):
    """FV3 ``del2_cubed_sphere`` (tools/fv_surf_map.F90) on the six-face
    duo gridstruct, flux form with the grid's own metrics::

        fx(i,j) = dy(i,j)*sina_u(i,j)*(q(i-1,j) - q(i,j))*rdxc(i,j)
        fy(i,j) = dx(i,j)*sina_v(i,j)*(q(i,j-1) - q(i,j))*rdyc(i,j)
        q(i,j) += cd*rarea(i,j)*(fx(i,j) - fx(i+1,j) + fy(i,j) - fy(i,j+1))

    on the compute window, then the context builder's EXTENDED-geometry
    halo exchange (``ext_scalar_sixface``, the one static exchange
    upstream applies to phis and the one the dycore's geopk halos are
    built on -- the stepper's plain A-grid exchange puts values up to
    1e4 m^2/s^2 apart in those halos, MEASURED 2026-09-27, and fed 8 m/s
    of spurious seam wind into a rest state), ``n_iter`` times (``cd``
    defaults to ``0.20*da_min`` as external_ic.F90:609).
    ``hs6``: ``(6, m, m)`` padded A-grid phis with valid halos; ``gs6``:
    the six gridstructs; ``ectx``: the builder's ext context.  Returns
    ``(6, m, m)``, halo-complete.  The one-ring stencil never reads the
    corner diagonals.
    """
    from legoesm.grids.fv3_native_ext_vector import ext_scalar_sixface
    n, ng = int(gs6[0]["n"]), int(gs6[0]["ng"])
    q = jnp.asarray(hs6, dtype=jnp.float64)
    if n_iter <= 0:
        return q
    if cd is None:
        cd = 0.20 * float(min(float(gs["da_min"]) for gs in gs6))
    st = lambda k: jnp.asarray(np.stack([np.asarray(gs[k]) for gs in gs6]))  # noqa: E731
    dy, sina_u, rdxc = st("dy"), st("sina_u"), st("rdxc")      # (6, m_b, m_a)
    dx, sina_v, rdyc = st("dx"), st("sina_v"), st("rdyc")      # (6, m_a, m_b)
    rarea = st("rarea")                                        # (6, m_a, m_a)
    ci = slice(ng, ng + n)          # compute cells
    ce = slice(ng, ng + n + 1)      # compute-window edges (n+1)
    cm = slice(ng - 1, ng + n)      # cells i-1 .. i
    for _ in range(n_iter):
        fx = (dy[:, ce, ci] * sina_u[:, ce, ci] * rdxc[:, ce, ci]
              * (q[:, cm, ci] - q[:, ce, ci]))
        fy = (dx[:, ci, ce] * sina_v[:, ci, ce] * rdyc[:, ci, ce]
              * (q[:, ci, ng - 1:ng + n] - q[:, ci, ng:ng + n + 1]))
        dq = cd * rarea[:, ci, ci] * (fx[:, :-1, :] - fx[:, 1:, :]
                                      + fy[:, :, :-1] - fy[:, :, 1:])
        q = q.at[:, ci, ci].add(dq)
        faces = [np.array(q[t]) for t in range(6)]     # writable copies
        ext_scalar_sixface(faces, "A", ectx)          # in place
        q = jnp.asarray(np.stack(faces))
    return q
