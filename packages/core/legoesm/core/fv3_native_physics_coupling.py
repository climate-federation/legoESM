"""FV3 physics coupling on the duo grid: A-grid tendency -> D-grid winds.

The first consumer of the port's physics-coupling wind vectors
(``compute_fv3_native_wind_vectors``, certified against the pinned Fortran
gridstruct to the float64 floor).  ``update_dwinds_phys_duo`` is the
``grid_type=0`` bounded-domain (duo) path of FV3's ``update_dwinds_phys``
(``model/fv_grid_utils.F90:3363-3547``): it rotates the A-grid physics wind
tendency into an Earth-frame 3-vector with ``vlon``/``vlat``, averages it onto
the cell edges, and projects onto the D-grid edge tangents ``es(:,:,:,1)`` /
``ew(:,:,:,2)`` to form the D-grid ``u``/``v`` increment.

DUO SCOPE.  ``bounded_domain`` is forced true on a duo grid
(``fv_arrays.F90:1512``), so the four ``.not. bounded_domain`` panel-edge blocks
in the reference (the ``edge_vect_w/e/s/n`` interpolations) are dead here and are
not ported; on an UNBOUNDED cubed sphere they are live and this routine would be
incomplete.  The routine is authored per level (2-D fields); the caller selects
the level, so it is level-agnostic and composes under ``vmap``/``scan``.

The NumPy function is the authority; ``update_dwinds_phys_duo_jax`` is its JAX
twin (same slicing, ``.at[].add`` for the scatter), gated against it in
``tests/grids/test_fv3_physics_coupling.py``.
"""
from __future__ import annotations

import numpy as np


def update_dwinds_phys_duo(u, v, u_dt, v_dt, dt, vlon, vlat, es1, ew2, n, ng):
    """Port of FV3 ``update_dwinds_phys``, duo (grid_type=0, bounded) path, one
    face, one level.  Dead ``edge_vect`` blocks omitted.

    Shapes (one face, full data domain, ``m = n + 2*ng``, ``ng >= 1``):

    =========  ===========  ====================================================
    ``u``      ``(m, m+1)``  D-grid west/east wind (updated in the returned copy)
    ``v``      ``(m+1, m)``  D-grid south/north wind
    ``u_dt``   ``(m, m)``    A-grid zonal wind tendency
    ``v_dt``   ``(m, m)``    A-grid meridional wind tendency
    ``vlon``   ``(m, m, 3)`` Earth-frame east unit vector, cell centred
    ``vlat``   ``(m, m, 3)`` Earth-frame north unit vector, cell centred
    ``es1``    ``(m, m+1,3)``N/S edge tangent ``es(:, i, j, 1)`` (component last)
    ``ew2``    ``(m+1, m,3)``E/W edge tangent ``ew(:, i, j, 2)``
    =========  ===========  ====================================================

    Returns ``(u_new, v_new)``; inputs are not mutated.
    """
    dt5 = 0.5 * dt

    # v3 on (is-1:ie+1, js-1:je+1): A-grid tendency -> Earth-frame 3-vector.
    # local index 0 <-> global ng-1.
    w = slice(ng - 1, ng + n + 1)
    v3 = u_dt[w, w, None] * vlon[w, w, :] + v_dt[w, w, None] * vlat[w, w, :]

    # sum (NOT mean) the two straddling cell vectors onto each edge.
    # ue(i,j) = v3(i, j-1) + v3(i, j) for i in is:ie, j edges js:je+1
    ue = v3[1:n + 1, 0:n + 1, :] + v3[1:n + 1, 1:n + 2, :]      # (n, n+1, 3)
    # ve(i,j) = v3(i-1, j) + v3(i, j) for i edges is:ie+1, j in js:je
    ve = v3[0:n + 1, 1:n + 1, :] + v3[1:n + 2, 1:n + 1, :]      # (n+1, n, 3)

    # project onto the edge tangents -> D-grid increments.
    u_new = u.copy()
    u_new[ng:ng + n, ng:ng + n + 1] += dt5 * (
        ue * es1[ng:ng + n, ng:ng + n + 1, :]).sum(axis=-1)
    v_new = v.copy()
    v_new[ng:ng + n + 1, ng:ng + n] += dt5 * (
        ve * ew2[ng:ng + n + 1, ng:ng + n, :]).sum(axis=-1)
    return u_new, v_new


def update_dwinds_phys_duo_jax(u, v, u_dt, v_dt, dt, vlon, vlat, es1, ew2,
                               n, ng):
    """JAX twin of :func:`update_dwinds_phys_duo` -- identical slicing, ``.at``
    scatter for the non-mutating update.  Differentiable and jittable."""
    import jax.numpy as jnp

    dt5 = 0.5 * dt
    w = slice(ng - 1, ng + n + 1)
    v3 = u_dt[w, w, None] * vlon[w, w, :] + v_dt[w, w, None] * vlat[w, w, :]
    ue = v3[1:n + 1, 0:n + 1, :] + v3[1:n + 1, 1:n + 2, :]
    ve = v3[0:n + 1, 1:n + 1, :] + v3[1:n + 2, 1:n + 1, :]
    du = dt5 * (ue * es1[ng:ng + n, ng:ng + n + 1, :]).sum(axis=-1)
    dv = dt5 * (ve * ew2[ng:ng + n + 1, ng:ng + n, :]).sum(axis=-1)
    u_new = jnp.asarray(u).at[ng:ng + n, ng:ng + n + 1].add(du)
    v_new = jnp.asarray(v).at[ng:ng + n + 1, ng:ng + n].add(dv)
    return u_new, v_new
