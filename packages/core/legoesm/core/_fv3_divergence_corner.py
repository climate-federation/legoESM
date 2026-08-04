"""Faithful port of FV3 ``divergence_corner`` from ``sw_core.F90:2124``.

Computes B-grid corner divergence with the FV3-specific edge stencils
and cube-vertex corner-removal terms.  This is the exact arithmetic
used by GFDL FV3 in ``d_sw5`` (sw_core.F90:1641-1719) before the
adaptive Smagorinsky-style damping is applied to the kinetic-energy
field.

Why this exists
---------------

Our 3D atmospheric path computes divergence at CELL CENTRES via
``cgrid_divergence`` (a simple flux-form ``net_x + net_y / area``)
that does NOT include the FV3-specific:

* sin_sg metric averaging at the j==1 / j==npy boundary rows (which
  picks up the sub-grid-cell metric at panel boundaries instead of
  the cell-edge average);

* cosa-cross-correction ``- 0.25*(va_W+va_E)*(cos_sg_W + cos_sg_E)``
  for the non-orthogonal cubed-sphere grid (the contravariant
  velocity correction);

* sw/se/ne/nw corner removal terms at the 4 cube vertices that
  subtract the spurious 4th-cell contribution where only 3 faces
  meet.

These three FV3 mechanisms give the corner-staggered ``divg_d`` its
faithful magnitude at panel-boundary halo cells — the same cells that
drive the cube-imprint feedback diagnosed in iter-2.

This module provides:

- :func:`fv3_divergence_corner_2d`: the 2D port, faithful to FV3
  ``sw_core.F90:2124-2229`` (non-bounded-domain, ``grid_type < 3``
  branch — i.e., the full sin_sg + cosa-correction + corner-removal
  formulation).

- :func:`fv3_divergence_corner_3d`: 3D wrapper that vmaps over the
  trailing level axis.

The returned ``divg_d`` is at B-grid corners with shape
``(6, n+1, n+1[, nlev])`` — ready to be used by an adaptive damping
that targets cube-vertex amplification.

Reference
---------

GFDL FV3 ``atmos_cubed_sphere-symmetryclean/model/sw_core.F90``
lines 2124-2229 (subroutine ``divergence_corner``).
"""
from __future__ import annotations

import jax.numpy as jnp

from legoesm.grids.cubed_sphere_cdgrid import CubedSphereCDGrid
from legoesm.grids.halo import pad_halo


def corner_div_damp_active(config) -> bool:
    """True when the B-grid corner divergence damping executes at all.

    FV3's d_sw has NO master switch on ``d2_bg`` — sw_core.F90:1641
    branches only on ``nord`` and ``d2_bg`` enters solely as the
    background floor ``max(d2_bg, min(0.20, dddmp*|delpc*dt|))``.
    Gating the whole block on ``d2_bg > 0`` silently disabled the
    del-4 corner damping under the standard FV3 configuration
    (d2_bg=0, nord>=1, d4_bg=0.16) — the NH DCMIP TC2/TC3 cube
    vertex blow-up (2026-07-29).

    legoESM ENABLE SELECTORS (deliberate deviation from FV3's
    always-on d_sw, so that legacy all-zero configs stay inert):
    activation requires ``d2_bg > 0`` OR the del-4 pair
    (``d4_bg > 0`` AND ``nord > 0``).  ``dddmp`` and the top-sponge
    boosts (``d2_bg_k1``/``d2_bg_k2``) are MODIFIERS only — nonzero
    values (dddmp defaults to 0.20) never activate the block by
    themselves.  Duck-typed over the CE/PE CD-grid config
    NamedTuples (both carry the ``corner_div_damp_*`` fields)."""
    return (config.corner_div_damp_d2_bg > 0.0
            or (config.corner_div_damp_d4_bg > 0.0
                and config.corner_div_damp_nord > 0))


def corner_div_damp_higher_order_active(config) -> bool:
    """True when the FV3 ``nord>0`` higher-order branch executes
    (sw_core.F90:1727 ``else`` of ``if (nord==0)``).

    Faithful to FV3, the branch keys on ``nord`` alone once the block
    is active — NOT on ``d4_bg``: with ``d4_bg=0`` the higher-order
    formula still applies (Smagorinsky ``smag_vort`` cap) and the
    del-4 term drops out naturally through
    ``dd8 = (da_min_c*d4_bg)**(nord+1) = 0`` (codex r1 P1)."""
    return corner_div_damp_active(config) and config.corner_div_damp_nord > 0


def _to_fv3_normal_dgrid_2d(
    u_corner: jnp.ndarray, v_corner: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convert C-D corner winds (n+1, n+1) to FV3 normal D-grid layout.

    FV3's ``divergence_corner`` expects:

    - ``u`` shape ``(isd:ied, jsd:jed+1)`` — x-velocity at v-interface
      midpoints, our convention ``(6, n, n+1)``.
    - ``v`` shape ``(isd:ied+1, jsd:jed)`` — y-velocity at u-interface
      midpoints, our convention ``(6, n+1, n)``.

    Our C-D layout has both at corners ``(6, n+1, n+1)``.  The
    FV3 v-interface midpoint at ``(i+0.5, j)`` corresponds to the
    average of corners ``(i, j)`` and ``(i+1, j)``.

    Parameters
    ----------
    u_corner : (6, n+1, n+1) — C-D u_d at corners.
    v_corner : (6, n+1, n+1) — C-D v_d at corners.

    Returns
    -------
    u_fv3 : (6, n, n+1) — FV3 normal D-grid u.
    v_fv3 : (6, n+1, n) — FV3 normal D-grid v.
    """
    u_fv3 = 0.5 * (u_corner[:, :-1, :] + u_corner[:, 1:, :])
    v_fv3 = 0.5 * (v_corner[:, :, :-1] + v_corner[:, :, 1:])
    return u_fv3, v_fv3


def _to_agrid_cell_centre_2d(
    u_corner: jnp.ndarray, v_corner: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convert C-D corner winds to A-grid cell-centre (ua, va) for FV3.

    A-grid cell-centre value at (i, j) = average of the 4 surrounding
    corner values.  This is exactly ``interp_corner_to_center``.
    """
    ua = 0.25 * (
        u_corner[:, :-1, :-1] + u_corner[:, 1:, :-1]
        + u_corner[:, :-1, 1:] + u_corner[:, 1:, 1:]
    )
    va = 0.25 * (
        v_corner[:, :-1, :-1] + v_corner[:, 1:, :-1]
        + v_corner[:, :-1, 1:] + v_corner[:, 1:, 1:]
    )
    return ua, va


def fv3_divergence_corner_2d(
    u_corner: jnp.ndarray,
    v_corner: jnp.ndarray,
    cdgrid: CubedSphereCDGrid,
    *,
    dgrid_ne_halo: bool = False,
) -> jnp.ndarray:
    """Faithful port of FV3 ``sw_core.F90:divergence_corner`` (2D or 4D).

    FV3_3D iter-1044: ndim-polymorphic.  Accepts both 3D inputs
    ``(6, n+1, n+1)`` and 4D inputs ``(6, n+1, n+1, nlev)``.  Under
    MPI the 4D path issues exactly ONE ``pad_halo_4d`` call for
    ``ua`` and one for ``va`` (one ``mpi4jax.sendrecv`` per call,
    batched over levels via the trailing axis), instead of ``nlev``
    separate sendrecvs.  Static metric pads still happen once
    regardless of ``nlev``.  This closes the codex iter-1044 review
    blocker on the per-level Python loop's O(nlev) sendrecv count.

    Direct port of the non-bounded-domain, ``grid_type < 3`` branch
    (lines 2181-2225 in the Fortran source).  Computes B-grid corner
    divergence using the FV3 sin_sg edge metrics, cosa cross-
    correction for non-orthogonality, and the four cube-vertex
    corner-removal terms.

    Parameters
    ----------
    u_corner : (6, n+1, n+1) — C-D u_d at D-grid corners.
    v_corner : (6, n+1, n+1) — C-D v_d at D-grid corners.
    cdgrid : CubedSphereCDGrid

    Returns
    -------
    divg_d : (6, n+1, n+1) — corner-staggered B-grid divergence
        with the FV3-faithful edge metrics and corner-removal terms.

    Notes
    -----
    - ``sin_sg[..., 1]`` (Fortran ``sin_sg(i,j,2)``) is the south-edge
      metric at cell (i, j); ``[..., 3]`` is the north-edge metric;
      ``[..., 0]`` is the west-edge; ``[..., 2]`` is the east-edge.
      We map FV3's 1-based index to our 0-based at the call site.

    - The cosa cross-correction reads ``va`` at cell (i, j-1) and
      (i, j) — both A-grid cell-centre values.  Periodic-aware halo
      padding via ``pad_halo`` is required so that the j-1 read at
      j=0 picks up the south-neighbour-panel cell.

    - Corner-removal: at the 4 cube vertices (sw, se, ne, nw of each
      face), one of the four cells in the divergence stencil
      doesn't physically exist.  FV3 subtracts that cell's
      contribution from the raw difference, giving the correct
      3-cell convergence at the cube vertex.
    """
    # Step 1: bring inputs into FV3 normal D-grid + A-grid layouts.
    # The helpers are shape-polymorphic (axis-1/2 slicing, trailing
    # axes broadcast).
    u_fv3, v_fv3 = _to_fv3_normal_dgrid_2d(u_corner, v_corner)
    ua, va = _to_agrid_cell_centre_2d(u_corner, v_corner)

    n = cdgrid.n
    _is_4d = u_corner.ndim == 4

    if dgrid_ne_halo:
        from legoesm.core.fv3_sw_core import (
            d2a2c_ua_va_halo,
            d2a2c_ua_va_halo_4d,
            sina_u_v_from_sin_sg,
        )
        from legoesm.grids.dgrid_halo import (
            pad_halo_dgrid_scalar_pair_4d,
            pad_halo_dgrid_sg_slots_4d,
            pad_halo_dgrid_vector_4d,
        )
        from legoesm.grids.halo import get_halo_backend

        def _dgrid_ne_halo(u_d, v_d):
            if get_halo_backend() == "mpi":
                from legoesm.grids.dgrid_halo import (
                    pad_halo_dgrid_vector_4d_replicated_mpi,
                )
                from legoesm.grids.halo import get_mpi_topology
                return pad_halo_dgrid_vector_4d_replicated_mpi(
                    u_d, v_d, get_mpi_topology(),
                )
            return pad_halo_dgrid_vector_4d(u_d, v_d)

        # COVARIANT D-point lift (2026-08-04, replaces the mixed-convention
        # 4-corner u_at_v construct).  Depth-binned probe measured the old
        # lift O(1)-wrong exactly on the seam line: it pushed the
        # ORTHONORMAL V.e_i through the COVARIANT-convention DGRID
        # exchange (u_at_v d0 = 0.75, x cosa_seam 0.49 = the 0.36 v_cov4
        # seam error), while its interior converged O(dx^2).  The faithful
        # lift needs NO halo at all: both corner endpoints of every edge
        # are on-face.  corners -> geographic (exact inverse of
        # rotate_winds_geo_to_grid) -> 2-point average along the edge ->
        # project onto the EXACT stagger bases (angle_edge_x/y), with
        # covariant v = cosa*(V.e_i) + sina*(V.e_i_perp) at the v points.
        sina_u, _ = sina_u_v_from_sin_sg(cdgrid)

        def _bcast(m):
            return m[..., None] if _is_4d else m

        ca_c = _bcast(jnp.cos(cdgrid.angle_corner))
        sa_c = _bcast(jnp.sin(cdgrid.angle_corner))
        u_e_c = ca_c * u_corner - sa_c * v_corner
        v_n_c = sa_c * u_corner + ca_c * v_corner
        # u D-points: x-edge midpoints (6, n, n+1) — average along i.
        u_e_ex = 0.5 * (u_e_c[:, :-1, :] + u_e_c[:, 1:, :])
        v_n_ex = 0.5 * (v_n_c[:, :-1, :] + v_n_c[:, 1:, :])
        ca_ex = _bcast(jnp.cos(cdgrid.angle_edge_x))
        sa_ex = _bcast(jnp.sin(cdgrid.angle_edge_x))
        u_cov = ca_ex * u_e_ex + sa_ex * v_n_ex
        # v D-points: y-edge midpoints (6, n+1, n) — average along j.
        u_e_ey = 0.5 * (u_e_c[:, :, :-1] + u_e_c[:, :, 1:])
        v_n_ey = 0.5 * (v_n_c[:, :, :-1] + v_n_c[:, :, 1:])
        ca_ey = _bcast(jnp.cos(cdgrid.angle_edge_y))
        sa_ey = _bcast(jnp.sin(cdgrid.angle_edge_y))
        vei_ey = ca_ey * u_e_ey + sa_ey * v_n_ey
        veip_ey = -sa_ey * u_e_ey + ca_ey * v_n_ey
        v_cov = (_bcast(cdgrid.cosa_u) * vei_ey
                 + _bcast(sina_u) * veip_ey)

        u4 = u_cov if _is_4d else u_cov[..., None]
        v_cov4 = v_cov if _is_4d else v_cov[..., None]

        # real_metric_ghosts: Fortran's ghost-ring ua/va (sw_core.F90:3513)
        # are evaluated with REAL gridstruct halo cosa_s/rsin2, not
        # edge-replicated pads.  The ring cells this lane reads at panel
        # boundaries (va(0,j), ua(i,0), ...) were the last edge-replicated
        # input feeding the O(1/dx) solid-body boundary residual.
        if _is_4d:
            ua_h1, va_h1 = d2a2c_ua_va_halo_4d(
                u4, v_cov4, cdgrid, real_metric_ghosts=True)
        else:
            ua_h1, va_h1 = d2a2c_ua_va_halo(
                u4[..., 0], v_cov4[..., 0], cdgrid,
                real_metric_ghosts=True,
            )
            ua_h1 = ua_h1[..., None]
            va_h1 = va_h1[..., None]

        u_full, v_full = _dgrid_ne_halo(u4, v_cov4)
        u_pad = u_full[:, :, 1:-1, :]
        v_pad = v_full[:, 1:-1, :, :]

        dyc_h, dxc_h = pad_halo_dgrid_scalar_pair_4d(
            cdgrid.dyc[..., None], cdgrid.dxc[..., None],
            axis_swap_sign=+1.0,
        )
        dyc_pad = dyc_h[:, :, 1:-1, :]
        dxc_pad = dxc_h[:, 1:-1, :, :]

        sin_sg_h, cos_sg_h = pad_halo_dgrid_sg_slots_4d(
            cdgrid.sin_sg[..., :4], cdgrid.cos_sg[..., :4],
        )
        sin_uf = 0.5 * (
            sin_sg_h[:, :, :-1, 3] + sin_sg_h[:, :, 1:, 1]
        )
        cos_uf = 0.5 * (
            cos_sg_h[:, :, :-1, 3] + cos_sg_h[:, :, 1:, 1]
        )
        sin_vf = 0.5 * (
            sin_sg_h[:, :-1, :, 2] + sin_sg_h[:, 1:, :, 0]
        )
        cos_vf = 0.5 * (
            cos_sg_h[:, :-1, :, 2] + cos_sg_h[:, 1:, :, 0]
        )

        va_at_jface = 0.5 * (
            va_h1[:, :, :-1, :] + va_h1[:, :, 1:, :]
        )
        ua_at_iface = 0.5 * (
            ua_h1[:, :-1, :, :] + ua_h1[:, 1:, :, :]
        )

        # ORACLE NOTE (2026-08-04): the boundary uf/vf forms below are the
        # AVERAGED slot pairs — u*dyc*0.5*(sin_sg(i,j-1,4)+sin_sg(i,j,2)) at
        # j==1|npy and the vf mirror at i==1|npx.  BOTH reference trees agree:
        # Zenodo symmetryclean sw_core.F90:2190/:2205-2206 AND plain FV3
        # 6f658bd0 divergence_corner.  A codex round-14 review prescribed
        # replacing these with d2a2c_vect's DIRECTIONAL upwind selection
        # (sw_core.F90:3589-3593 etc.) — that selection belongs to the C-wind
        # edge branches of d2a2c_vect, not to divergence_corner, which never
        # consumes uc/vc.  Rejected against both sources; do not "fix" this
        # to directional.
        j_face = jnp.arange(n + 1)
        is_uf_boundary = ((j_face == 0) | (j_face == n))[None, None, :, None]
        uf_boundary = u_pad * dyc_pad * sin_uf[..., None]
        uf_interior = (
            (u_pad - va_at_jface * cos_uf[..., None])
            * dyc_pad * sin_uf[..., None]
        )
        uf = jnp.where(is_uf_boundary, uf_boundary, uf_interior)

        i_face = jnp.arange(n + 1)
        is_vf_boundary = ((i_face == 0) | (i_face == n))[None, :, None, None]
        vf_boundary = v_pad * dxc_pad * sin_vf[..., None]
        vf_interior = (
            (v_pad - ua_at_iface * cos_vf[..., None])
            * dxc_pad * sin_vf[..., None]
        )
        vf = jnp.where(is_vf_boundary, vf_boundary, vf_interior)

        divg_d = (
            vf[:, :, :-1, :] - vf[:, :, 1:, :]
            + uf[:, :-1, :, :] - uf[:, 1:, :, :]
        )
        divg_d = divg_d.at[:, 0, 0, :].add(-vf[:, 0, 0, :])
        divg_d = divg_d.at[:, n, 0, :].add(-vf[:, n, 0, :])
        divg_d = divg_d.at[:, n, n, :].add(vf[:, n, n + 1, :])
        divg_d = divg_d.at[:, 0, n, :].add(vf[:, 0, n + 1, :])
        divg_d = divg_d * cdgrid.rarea_c[..., None]
        return divg_d if _is_4d else divg_d[..., 0]


    # Step 2: pad ua, va with halo=1 so the cosa cross-correction at
    # j-1 / i-1 reads neighbour-panel cells (cross-face values).
    # FV3_3D iter-1044: dispatch pad_halo vs pad_halo_4d by ndim so
    # the 4D path issues one batched sendrecv per array (not nlev).
    if _is_4d:
        from legoesm.grids.halo import pad_halo_4d
        ua_pad = pad_halo_4d(ua)
        va_pad = pad_halo_4d(va)
    else:
        ua_pad = pad_halo(ua)
        va_pad = pad_halo(va)

    # Step 3: halo the normal D-grid pair with a REAL cross-panel
    # DGRID_NE exchange (FV3 dyn_core.F90:376 / :501 -- the vector halo
    # completed before ``c_sw``), NOT edge replication.
    #
    # Why this matters (2026-07-31): FV3's SW-vertex value, after the
    # one-extra-flux removal at sw_core.F90:2209/:2215, is
    #
    #     divg_d(1,1) = -vf(1,1) + uf(0,1) - uf(1,1)
    #
    # so the cross-panel WEST u GHOST ``uf(0,1)`` must survive.  Edge
    # replication sets uf(0,1) == uf(1,1), erasing that difference and
    # leaving ~ -vf(1,1)/area_corner -- an O(v/dx) residual at the 4 face
    # vertices, which grows with resolution.  Across the eight
    # axis-swapping seams u and v exchange WITH SIGNS (face 4's east u
    # halo is -v from face 1); only the vector helper does that.
    # ``pad_halo_dgrid_scalar_4d`` deliberately falls back to edge
    # replication on exactly those seams (dgrid_halo.py:189-228) and is
    # the wrong tool here.
    if not dgrid_ne_halo:
        # DEFAULT (bit-identical to the pre-2026-07-31 behaviour).  The
        # DGRID_NE halo below is the CORRECT velocity exchange and is proven
        # by the exact +8/+5 seam/vertex assertions, but ON ITS OWN it makes
        # the operator INCONSISTENT: real cross-panel neighbour winds get
        # multiplied by still-edge-replicated seam metrics (dyc/sina/cosa at
        # :400,:455, raw sin_sg at :263) and combined with ua/va from a
        # 4-corner average rather than FV3's d2a2c_vect.  Measured
        # consequence: DCMIP TC1 on the cube goes PASS -> BLOWUP (step 3950,
        # max|u| 1004) with the velocity halo alone.  Enable only together
        # with the companion metric halo + d2a2c_vect work; flip the default
        # when the solid-body convergence oracle XPASSes.
        if _is_4d:
            u_fv3_pad = jnp.pad(
                u_fv3, [(0, 0), (1, 1), (0, 0), (0, 0)], mode="edge")
            v_fv3_pad = jnp.pad(
                v_fv3, [(0, 0), (0, 0), (1, 1), (0, 0)], mode="edge")
        else:
            u_fv3_pad = jnp.pad(u_fv3, [(0, 0), (1, 1), (0, 0)], mode="edge")
            v_fv3_pad = jnp.pad(v_fv3, [(0, 0), (0, 0), (1, 1)], mode="edge")
    else:
        _u4 = u_fv3 if _is_4d else u_fv3[..., None]
        _v4 = v_fv3 if _is_4d else v_fv3[..., None]
        from legoesm.grids.halo import get_halo_backend as _ghb_dc
        if _ghb_dc() == "mpi":
            from legoesm.grids.dgrid_halo import (
                pad_halo_dgrid_vector_4d_replicated_mpi,
            )
            from legoesm.grids.halo import get_mpi_topology
            _u_full, _v_full = pad_halo_dgrid_vector_4d_replicated_mpi(
                _u4, _v4, get_mpi_topology(),
            )
        else:
            from legoesm.grids.dgrid_halo import pad_halo_dgrid_vector_4d
            _u_full, _v_full = pad_halo_dgrid_vector_4d(_u4, _v4)
        # The helper halos BOTH axes; this routine differences u only along
        # the cell axis and v only along its own, so trim the other back.
        u_fv3_pad = _u_full[:, :, 1:-1, :]      # (6, n+2, n+1, nlev)
        v_fv3_pad = _v_full[:, 1:-1, :, :]      # (6, n+1, n+2, nlev)
        if not _is_4d:
            u_fv3_pad = u_fv3_pad[..., 0]
            v_fv3_pad = v_fv3_pad[..., 0]

    # Step 4: extract sin_sg / cos_sg at the 4 sub-grid positions used.
    # FV3 indexing convention (0-based here): sg[0]=west, sg[1]=south,
    # sg[2]=east, sg[3]=north.  See cubed_sphere_cdgrid.py:_compute_sin_cos_sg
    # for the derivation.  The Fortran 1-based ``(i, j, 2)`` → south
    # → our index 1; ``(i, j, 4)`` → north → our index 3.
    sin_s = cdgrid.sin_sg[..., 1]     # south edge sin
    sin_n = cdgrid.sin_sg[..., 3]     # north edge sin
    sin_w = cdgrid.sin_sg[..., 0]     # west edge sin
    sin_e = cdgrid.sin_sg[..., 2]     # east edge sin
    cos_s = cdgrid.cos_sg[..., 1]
    cos_n = cdgrid.cos_sg[..., 3]
    cos_w = cdgrid.cos_sg[..., 0]
    cos_e = cdgrid.cos_sg[..., 2]

    sin_s_pad = pad_halo(sin_s)
    sin_n_pad = pad_halo(sin_n)
    sin_w_pad = pad_halo(sin_w)
    sin_e_pad = pad_halo(sin_e)
    cos_s_pad = pad_halo(cos_s)
    cos_n_pad = pad_halo(cos_n)
    cos_w_pad = pad_halo(cos_w)
    cos_e_pad = pad_halo(cos_e)

    # Step 5: ``uf`` at v-interface (i, j) for i ∈ [0, n], j ∈ [0, n].
    # j == 0 (FV3 j==1) and j == n (FV3 j==npy): boundary rows use the
    # simpler sin-only formula.  Other j: full cosa-corrected formula.
    #
    # FV3 line 2190 (j==1 or j==npy):
    #   uf(i,j) = u(i,j) * dyc(i,j) * 0.5*(sin_sg(i,j-1,4)+sin_sg(i,j,2))
    # FV3 line 2194 (interior j):
    #   uf(i,j) = (u(i,j) - 0.25*(va(i,j-1)+va(i,j)) *
    #              (cos_sg(i,j-1,4) + cos_sg(i,j,2)))
    #             * dyc(i,j) * 0.5*(sin_sg(i,j-1,4)+sin_sg(i,j,2))
    #
    # In our 0-based arrays:
    #  - u_fv3_pad[:, i+1, j] is u_FV3(i, j) — added 1 to i for the pad.
    #    Wait, u_fv3 has shape (n, n+1) → axis 1 is i ∈ [0, n-1].
    #    After pad axis 1 to (n+2,): index 0 = west halo, 1..n = interior.
    #    But the Fortran reads u(i, j) for i ∈ [is-1, ie+1] = [-1, n+1] in
    #    0-based, which is the full padded range [0, n+1].
    #
    # Simpler: the divergence_corner output is at j ∈ [0, n] (n+1 corners).
    # Build uf at i ∈ [0, n], j ∈ [0, n] (also n+1 in i).  uf shape: (n+1, n+1).
    # j_face = 0..n; cell-i for sin_sg lookups: i_cell = i_face but we need
    # cell (i, j-1) and cell (i, j) where j is the FACE index — FV3 uses
    # j-1 → south cell, j → north cell of the v-edge at j-face.
    #
    # The FV3 ``uf`` is actually at v-interfaces (j-faces), shape (i, j_face).
    # For our 0-based j_face ∈ [0, n], "south cell" (j-1 in FV3) is cell
    # j_face-1 with index j_face_pad-1 in the halo'd array.
    #
    # For the 6 face dimension we vectorise via leading axis.

    # Build sin/cos at v-edges via south + north cell averages.
    # Padded south cell at j_face = j-1 in FV3 → padded j (when j_face=0).
    # Padded north cell at j_face = j in FV3 → padded j+1 (when j_face=0).
    # In 0-based with halo padded shape (n+2, n+2): cell (i, j_face-1) →
    # padded[i+1, j_face]; cell (i, j_face) → padded[i+1, j_face+1].
    # Wait — `j-1` in FV3 corresponds to "the cell to the south of the
    # j-face".  In our convention, the j-face at index j_face sits BETWEEN
    # cells at row j_face-1 (south) and row j_face (north).  Padded
    # indexing: cell j_face-1 = padded[1+j_face-1] = padded[j_face]; cell
    # j_face = padded[j_face+1].
    # So at j_face = 0: south cell padded index is 0 (the south halo);
    # north cell padded index is 1 (first interior).
    # At j_face = n: south = padded[n] (last interior); north = padded[n+1]
    # (north halo).

    # j_face range: 0..n (n+1 values).  i_face range: 0..n (n+1 values).
    # Use slicing [j_face_idx_start: j_face_idx_end] = [0: n+1].
    # For interior i: cell at column i_cell, padded index i_cell+1.
    # uf(i_face, j_face) reads u_fv3_pad[..., i_face?, j_face? ]
    #
    # u_FV3 at v-interface (i, j_face) — i is which COLUMN of cells you're
    # between, j_face is the face index in j.  u shape (n, n+1) means
    # i ∈ [0, n-1], j_face ∈ [0, n].  Halo-padded to (n+2, n+1) (axis 1 padding):
    # u_fv3_pad[i_axis_1, j_face] for i_axis_1 ∈ [0, n+1].  Map FV3's i
    # cell index to padded: i_cell padded = i_cell + 1.
    #
    # Oh wait: u(i, j) in FV3 with shape (isd:ied, jsd:jed+1) at i=is-1
    # means halo cell.  But it's a v-interface NOT a cell — there's a
    # subtle distinction.  Following the existing legoESM SW path
    # (the legoESM SW path used u_d shape (6, n, n+1) without
    # padding because it computes interior corners only).
    #
    # For a faithful divergence_corner port that includes BOUNDARY corners
    # (i_face = 0 or n, j_face = 0 or n), we need u_fv3 at i = -1 (halo).
    # u_fv3_pad provides that via ``mode='edge'`` — same-face extension.

    # Build south/north cell indices in the halo'd cell-array (n+2 axis)
    # for j_face = 0..n:
    #   south_cell_pad = j_face       (j_face=0 → padded[0] = south halo)
    #   north_cell_pad = j_face + 1   (j_face=n → padded[n+1] = north halo)
    # Use slicing windows.

    # Interpolate sin/cos at v-edges (j-faces) via 0.5 * (south + north).
    sin_s_at_j1 = sin_n_pad[:, 1:n + 1, 0:n + 1]   # cell j-1 at j-face j: padded col j
    sin_n_at_j  = sin_s_pad[:, 1:n + 1, 1:n + 2]   # cell j   at j-face j: padded col j+1
    sin_v_face = 0.5 * (sin_s_at_j1 + sin_n_at_j)  # (6, n, n+1)
    cos_s_at_j1 = cos_n_pad[:, 1:n + 1, 0:n + 1]
    cos_n_at_j  = cos_s_pad[:, 1:n + 1, 1:n + 2]
    cos_v_face = 0.5 * (cos_s_at_j1 + cos_n_at_j)  # (6, n, n+1)

    # va at j-face (south + north cell, axis 2 is j).
    va_s = va_pad[:, 1:n + 1, 0:n + 1]   # (6, n, n+1) cell j-1
    va_n = va_pad[:, 1:n + 1, 1:n + 2]   # (6, n, n+1) cell j
    va_at_j_face = 0.5 * (va_s + va_n)   # (6, n, n+1) — this is 0.25*(va(j-1)+va(j))*2

    # j-face mask: True where j == 0 or j == n (boundary), False elsewhere.
    # j_face indices 0..n.  For 4D input add a trailing nlev broadcast axis.
    j_face_idx = jnp.arange(n + 1)
    is_boundary_j = (j_face_idx == 0) | (j_face_idx == n)        # (n+1,)
    if _is_4d:
        boundary_j_3d = is_boundary_j[None, None, :, None]        # (1, 1, n+1, 1)
    else:
        boundary_j_3d = is_boundary_j[None, None, :]              # (1, 1, n+1)

    # Now compute uf.  But note FV3's uf has shape (i_face, j_face) with
    # i_face ∈ [is-1, ie+1] (so halo+1).  Our u_fv3 (n, n+1) in axis 1
    # is cell-i.  We need uf at i_face, but actually FV3's uf is at
    # i_cell — confusingly named.  Reading the formula again:
    #   uf(i, j) = u(i, j) * stuff
    # where u has shape (isd:ied, jsd:jed+1).  So u(i, j) at i ∈ [-1, n+1]
    # means cell-i halo coverage.  uf has shape (is-2:ie+2, js-1:je+2)
    # so uf is at cell-i, j-face.
    #
    # For divg_d(i_face, j_face) = vf(j-1) - vf(j) + uf(i-1) - uf(i) the
    # uf indices are i-1 and i where i = i_face.  So uf is at cell-i =
    # i_face-1 and i_face.  The output divg_d at i_face uses uf at
    # i_face-1 (left cell) and i_face (right cell).
    #
    # In our convention, uf is at cell-i, j-face → shape (n, n+1) BEFORE
    # halo extension.  Pad axis 1 to (n+2, n+1) for the i_face-1 access.

    # u_fv3 at v-interface (cell-i, j-face) — directly available, but we
    # have it after pad as u_fv3_pad shape (6, n+2, n+1).  Axis 1 padded
    # index 0 = west halo, n+1 = east halo, 1..n = interior cells 0..n-1.

    # Slice u at all relevant (cell-i, j-face) for the boundary-aware
    # formula.  Use the full (n+2, n+1) range — interior cell-i ∈ [0, n-1]
    # in 0-based maps to padded index 1..n; halo at 0 and n+1.

    # Build uf with both branches and select via boundary mask.
    u_full = u_fv3_pad[:, :, :]                   # (6, n+2, n+1[, nlev])
    # va at v-interfaces in cell-i coords: va_at_j_face shape (6, n, n+1[, nlev]).
    # Pad axis 1 (cell-i) to (n+2, n+1) via mode='edge' — keep nlev pad width=0.
    if _is_4d:
        va_at_jface_pad = jnp.pad(
            va_at_j_face, [(0, 0), (1, 1), (0, 0), (0, 0)], mode="edge",
        )
    else:
        va_at_jface_pad = jnp.pad(
            va_at_j_face, [(0, 0), (1, 1), (0, 0)], mode="edge",
        )
    cos_v_pad = jnp.pad(
        cos_v_face, [(0, 0), (1, 1), (0, 0)], mode="edge",
    )                                              # (6, n+2, n+1) — 3D static
    sin_v_pad = jnp.pad(
        sin_v_face, [(0, 0), (1, 1), (0, 0)], mode="edge",
    )

    # FV3's dyc has shape (6, n, n+1) per ``cdgrid.dyc`` definition.
    # Pad cell-i to (n+2, n+1) via mode='edge'.
    dyc_pad = jnp.pad(cdgrid.dyc, [(0, 0), (1, 1), (0, 0)], mode="edge")

    # FV3_3D iter-1044: broadcast 3D static metric pads against 4D data
    # via a trailing nlev singleton axis.
    if _is_4d:
        cos_v_pad_b = cos_v_pad[..., None]
        sin_v_pad_b = sin_v_pad[..., None]
        dyc_pad_b = dyc_pad[..., None]
    else:
        cos_v_pad_b = cos_v_pad
        sin_v_pad_b = sin_v_pad
        dyc_pad_b = dyc_pad

    cross = va_at_jface_pad * cos_v_pad_b
    uf_interior = (u_full - cross) * dyc_pad_b * sin_v_pad_b
    uf_boundary = u_full * dyc_pad_b * sin_v_pad_b
    uf = jnp.where(boundary_j_3d, uf_boundary, uf_interior)

    # Same construction for vf.  FV3 lines 2200-2207:
    #   vf(i,j) = (v(i,j) - 0.25*(ua(i-1,j)+ua(i,j))*(cos_sg(i-1,j,3)+cos_sg(i,j,1))) * dxc(i,j) * 0.5*(sin_sg(i-1,j,3)+sin_sg(i,j,1))
    # at non-edge i (i_face != 0, npx), and a boundary form at i==1 or i==npx.
    # Interpolate ua at u-edges (i-faces) using west + east cell.
    ua_w = ua_pad[:, 0:n + 1, 1:n + 1]            # (6, n+1, n) cell i-1
    ua_e = ua_pad[:, 1:n + 2, 1:n + 1]            # (6, n+1, n) cell i
    ua_at_iface = 0.5 * (ua_w + ua_e)              # (6, n+1, n)

    # cos_sg at i-face: 0.5 * (east_of_i-1, west_of_i).
    cos_e_w = cos_e_pad[:, 0:n + 1, 1:n + 1]      # cell i-1, east edge
    cos_w_e = cos_w_pad[:, 1:n + 2, 1:n + 1]      # cell i, west edge
    cos_u_face = 0.5 * (cos_e_w + cos_w_e)        # (6, n+1, n)
    sin_e_w = sin_e_pad[:, 0:n + 1, 1:n + 1]
    sin_w_e = sin_w_pad[:, 1:n + 2, 1:n + 1]
    sin_u_face = 0.5 * (sin_e_w + sin_w_e)        # (6, n+1, n)

    # Pad to (n+1, n+2) for the cell-j halo on j-axis.
    cos_u_pad = jnp.pad(
        cos_u_face, [(0, 0), (0, 0), (1, 1)], mode="edge",
    )
    sin_u_pad = jnp.pad(
        sin_u_face, [(0, 0), (0, 0), (1, 1)], mode="edge",
    )
    if _is_4d:
        ua_at_iface_pad = jnp.pad(
            ua_at_iface, [(0, 0), (0, 0), (1, 1), (0, 0)], mode="edge",
        )
    else:
        ua_at_iface_pad = jnp.pad(
            ua_at_iface, [(0, 0), (0, 0), (1, 1)], mode="edge",
        )
    dxc_pad = jnp.pad(cdgrid.dxc, [(0, 0), (0, 0), (1, 1)], mode="edge")

    # i-face boundary mask: i_face = 0 or n.  4D-broadcast on nlev.
    i_face_idx = jnp.arange(n + 1)
    is_boundary_i = (i_face_idx == 0) | (i_face_idx == n)
    if _is_4d:
        boundary_i_3d = is_boundary_i[None, :, None, None]
    else:
        boundary_i_3d = is_boundary_i[None, :, None]

    if _is_4d:
        cos_u_pad_b = cos_u_pad[..., None]
        sin_u_pad_b = sin_u_pad[..., None]
        dxc_pad_b = dxc_pad[..., None]
    else:
        cos_u_pad_b = cos_u_pad
        sin_u_pad_b = sin_u_pad
        dxc_pad_b = dxc_pad

    cross_v = ua_at_iface_pad * cos_u_pad_b
    vf_full = v_fv3_pad
    vf_interior = (vf_full - cross_v) * dxc_pad_b * sin_u_pad_b
    vf_boundary = vf_full * dxc_pad_b * sin_u_pad_b
    vf = jnp.where(boundary_i_3d, vf_boundary, vf_interior)

    # Step 6: assemble divg_d at corners (n+1, n+1).  FV3 line 2211:
    #   divg_d(i, j) = vf(i, j-1) - vf(i, j) + uf(i-1, j) - uf(i, j)
    # In our notation: divg_d at corner (i_face, j_face).
    #   uf at (cell_i, j_face) — for divg_d at i_face we use cell_i = i_face-1 (left)
    #   and cell_i = i_face (right).
    #   In padded indexing: padded_i = cell_i + 1, so left = i_face, right = i_face+1.
    #   vf at (i_face, cell_j) — for divg_d at j_face we use cell_j = j_face-1
    #   (south) and cell_j = j_face (north).  Padded j: south = j_face,
    #   north = j_face+1.

    uf_left  = uf[:, 0:n + 1, :]      # cell_i = i_face-1 → padded i_face
    uf_right = uf[:, 1:n + 2, :]      # cell_i = i_face   → padded i_face+1
    vf_south = vf[:, :, 0:n + 1]      # cell_j = j_face-1 → padded j_face
    vf_north = vf[:, :, 1:n + 2]      # cell_j = j_face   → padded j_face+1

    divg_d_raw = vf_south - vf_north + uf_left - uf_right        # (6, n+1, n+1)

    # Step 7: corner-removal at the 4 cube vertices of each face.  FV3
    # lines 2216-2219:
    #   if (sw_corner) divg_d(1, 1)     -= vf(1, 0)         ! SW
    #   if (se_corner) divg_d(npx, 1)   -= vf(npx, 0)       ! SE
    #   if (ne_corner) divg_d(npx, npy) += vf(npx, npy)     ! NE
    #   if (nw_corner) divg_d(1, npy)   += vf(1, npy)       ! NW
    # In our 0-based: sw=(0, 0), se=(n, 0), ne=(n, n), nw=(0, n).  vf at the
    # corresponding (i_face, padded j) = (0, 0), (n, 0), (n, n+1), (0, n+1).
    # All face cube vertices are corner cells in our (6, n+1, n+1)
    # representation (every face has all four).
    sw = vf[:, 0, 0]                        # (6,)
    se = vf[:, n, 0]
    ne = vf[:, n, n + 1]
    nw = vf[:, 0, n + 1]

    divg_d = divg_d_raw
    divg_d = divg_d.at[:, 0, 0].add(-sw)
    divg_d = divg_d.at[:, n, 0].add(-se)
    divg_d = divg_d.at[:, n, n].add(ne)
    divg_d = divg_d.at[:, 0, n].add(nw)

    # Step 8: divide by area_corner (Fortran rarea_c).
    rarea_c = cdgrid.rarea_c                    # (6, n+1, n+1)
    if _is_4d:
        rarea_c = rarea_c[..., None]
    return divg_d * rarea_c


def fv3_divergence_corner_3d(
    u_corner_3d: jnp.ndarray,
    v_corner_3d: jnp.ndarray,
    cdgrid: CubedSphereCDGrid,
    *,
    dgrid_ne_halo: bool = False,
) -> jnp.ndarray:
    """3D wrapper around the now-4D-native :func:`fv3_divergence_corner_2d`.

    FV3_3D iter-1044: ``fv3_divergence_corner_2d`` is shape-polymorphic
    (3D and 4D), so the 3D wrapper is now a single direct call rather
    than a per-level ``jax.vmap`` or Python loop.  Under MPI the dynamic
    ``ua``/``va`` go through one ``pad_halo_4d`` (one batched sendrecv
    per array, not ``nlev`` of them); static metric pads happen once.

    Parameters
    ----------
    u_corner_3d : (6, n+1, n+1, nlev)
    v_corner_3d : (6, n+1, n+1, nlev)
    cdgrid : CubedSphereCDGrid

    Returns
    -------
    divg_d_3d : (6, n+1, n+1, nlev)
    """
    if u_corner_3d.ndim != 4:
        raise ValueError(
            f"fv3_divergence_corner_3d expects 4D input "
            f"(6, n+1, n+1, nlev); got ndim={u_corner_3d.ndim}, "
            f"shape={tuple(u_corner_3d.shape)}.  Use "
            f"fv3_divergence_corner_2d for 3D input."
        )
    return fv3_divergence_corner_2d(
        u_corner_3d, v_corner_3d, cdgrid, dgrid_ne_halo=dgrid_ne_halo)


def fv3_corner_laplacian_iteration(
    divg_d: jnp.ndarray,
    cdgrid: CubedSphereCDGrid,
    apply_vector_corner_fill: bool = False,
) -> jnp.ndarray:
    """Apply one Laplacian iteration on a B-grid corner-staggered field.

    Faithful port of FV3 ``sw_core.F90:1746-1782`` inner block of the
    higher-order ``nord > 0`` divergence-damping path.  One iteration
    (``do n=1, nord`` body) on the corner-staggered ``divg_d`` field.

    The Fortran sequence (sw_core.F90:1748-1785, 1-based)::

        do j=js-nt,je+1+nt
           do i=is-1-nt,ie+1+nt
              vc(i,j) = (divg_d(i+1,j)-divg_d(i,j))*divg_u(i,j)   ! x-flux
           enddo
        enddo
        do j=js-1-nt,je+1+nt
           do i=is-nt,ie+1+nt
              uc(i,j) = (divg_d(i,j+1)-divg_d(i,j))*divg_v(i,j)   ! y-flux
           enddo
        enddo
        do j=js-nt,je+1+nt
           do i=is-nt,ie+1+nt
              divg_d(i,j) = uc(i,j-1) - uc(i,j) + vc(i-1,j) - vc(i,j)
           enddo
        enddo
        ! Cube-vertex corner-removal (sw_core.F90:1773-1776):
        if (sw_corner) divg_d(1,    1)   -= uc(1,    0)     ! uc south halo
        if (se_corner) divg_d(npx,  1)   -= uc(npx,  0)     ! uc south halo
        if (ne_corner) divg_d(npx,npy)   += uc(npx,npy)     ! uc interior NE
        if (nw_corner) divg_d(1,  npy)   += uc(1,  npy)     ! uc interior NW
        divg_d(:,:) = divg_d(:,:) * rarea_c(:,:)

    Parameters
    ----------
    divg_d : jnp.ndarray, shape ``(6, n+1, n+1)``
        Corner-staggered divergence (or its iterated Laplacian).
    cdgrid : CubedSphereCDGrid

    Returns
    -------
    lap_divg : jnp.ndarray, same shape — Laplacian of the input at
        corner staggering.

    Notes
    -----
    Halo provenance:

    * ``divg_d`` is padded with the legoESM halo (``pad_halo``) to
      provide cross-panel values at i = -1 / i = n+1 / j = -1 / j = n+1
      (in our 0-based shifted convention).  The default
      ``set_corner_fill_mode = "avg"`` (iter-7) handles the cube-vertex
      halo of the scalar.

    * ``vc`` is computed at i ∈ [-1, n] (shape ``(6, n+2, n+1)``) so
      that the divergence at the WEST corner (i = 0) sees the
      cross-panel value at i = -1.

    * ``uc`` is computed at j ∈ [-1, n] (shape ``(6, n+1, n+2)``) so
      that the divergence at the SOUTH corner (j = 0) AND the cube-
      vertex corner-removal at SW / SE both pick up the cross-panel
      south-halo value (FV3's ``uc(:, 0)`` row in 1-based).

    Documented fidelity gaps relative to FV3:

    * FV3 calls ``fill_corners(divg_d, ..., FILL=XDir, BGRID=.true.)``
      and ``YDir`` separately before each gradient (sw_core.F90:1746,
      1754) for direction-aware cube-vertex completion.  We use a
      single ``pad_halo`` with the iter-7 average corner mode; this is
      direction-agnostic but consistent with the iter-15 / iter-16
      port philosophy.

    Optional FV3 vector cube-vertex fill (iter 20):

    * ``apply_vector_corner_fill = True`` triggers FV3's
      ``fill_corners(vc, uc, VECTOR=true, DGRID=true)`` between
      gradient and divergence (sw_core.F90:1762, ported in
      ``legoesm.grids._fv3_dgrid_corner_fill``).  This affects the
      4 cube-vertex halo cells of vc / uc.

    * Code-level audit verified the cells written by the vector fill
      are NOT read by the divergence operator OR the corner-removal
      step at ``nt = 0`` (the only iteration when ``nord = 1``).  In
      that regime ``apply_vector_corner_fill = True`` produces output
      bit-for-bit identical to ``apply_vector_corner_fill = False`` —
      see the regression test ``test_corner_laplacian_vector_fill_is_noop_for_nord1``.

    * For ``nord >= 2``, the inner iterations have ``nt > 0`` and the
      divergence operator extends into halo rows that DO read the
      fill-written cells.  In that regime the flag changes output.
      Default off to preserve iter-18 bit-for-bit nord=1 behaviour
      until the wider nord >= 2 restructure (iter 21+) is in place.
    """
    n = cdgrid.n
    # FV3_3D iter-1044 (codex review claim-4): ndim guard.
    if divg_d.ndim not in (3, 4):
        raise ValueError(
            f"fv3_corner_laplacian_iteration expects ndim ∈ {{3, 4}}; "
            f"got ndim={divg_d.ndim}, shape={tuple(divg_d.shape)}."
        )

    # Step 1: cross-panel halo of divg_d.  divg_pad shape (6, n+3, n+3)
    # for 3D input or (6, n+3, n+3, nlev) for 4D input.
    # FV3_3D iter-1044: dispatch by ndim so NH callers can pass the
    # full 4D ``(6, n+1, n+1, nlev)`` corner-staggered ``delpc`` array
    # directly and avoid wrapping this function in ``jax.vmap`` (which
    # under MPI puts ``mpi4jax.sendrecv`` inside a vmap and trips
    # mpi4jax's batch-axis assertion).  Padded index 0 = west halo
    # (i = -1 shifted), padded index n+2 = east halo (i = n+1).
    # Cube-vertex halo follows ``set_corner_fill_mode`` (iter-7 "avg").
    from legoesm.grids.halo import pad_halo, pad_halo_4d
    if divg_d.ndim == 4:
        divg_pad = pad_halo_4d(divg_d)                     # (6, n+3, n+3, nlev)
    else:
        divg_pad = pad_halo(divg_d)                        # (6, n+3, n+3)

    if apply_vector_corner_fill:
        # FV3-faithful path with full halo'd vc / uc (matches FV3
        # D-grid layout) and ``fill_corners(vc, uc, VECTOR=true,
        # DGRID=true)`` applied between gradient and divergence.
        return _laplacian_iteration_with_vector_fill(
            divg_pad, cdgrid, n,
        )

    # Default iter-18 path: vc / uc with the minimum halo needed by
    # the divergence operator + corner removal at nt = 0.
    # FV3_3D iter-1044: when input is 4D, broadcast 3D metric arrays
    # against trailing nlev axis.  3D metric stays 3D when input is 3D.
    _is_4d = divg_d.ndim == 4

    # Step 2: x-flux ``vc(i, j) = (divg_d(i+1, j) - divg_d(i, j)) * divg_u``
    # at i ∈ [-1, n], j ∈ [0, n] — shape (6, n+2, n+1) or (6, n+2, n+1, nlev).
    vc_raw = (
        divg_pad[:, 1:n + 3, 1:n + 2]   # divg_d(i+1, j), i ∈ [-1, n]
        - divg_pad[:, 0:n + 2, 1:n + 2]  # divg_d(i,   j)
    )
    # FV3 ``divg_u = dy / dxc`` at u-face.  Our equivalent:
    # ``dy_edge_x * rdxc``, shape (6, n+1, n) at u-face.  Pad to
    # (6, n+2, n+1) via edge mode (the metric is geometric and varies
    # smoothly across panel boundaries; the i = -1 west-halo strip is
    # well-approximated by the i = 0 interior strip).
    divg_u = cdgrid.dy_edge_x * cdgrid.rdxc                  # (6, n+1, n)
    divg_u_pad = jnp.pad(
        divg_u, [(0, 0), (1, 0), (0, 1)], mode="edge",
    )                                                       # (6, n+2, n+1)
    if _is_4d:
        divg_u_pad = divg_u_pad[..., None]
    vc = vc_raw * divg_u_pad

    # Step 3: y-flux ``uc(i, j) = (divg_d(i, j+1) - divg_d(i, j)) * divg_v``
    # at i ∈ [0, n], j ∈ [-1, n] — shape (6, n+1, n+2) or (..., nlev).
    uc_raw = (
        divg_pad[:, 1:n + 2, 1:n + 3]   # divg_d(i, j+1), j ∈ [-1, n]
        - divg_pad[:, 1:n + 2, 0:n + 2]  # divg_d(i, j)
    )
    divg_v = cdgrid.dx_edge_y * cdgrid.rdyc                  # (6, n, n+1)
    divg_v_pad = jnp.pad(
        divg_v, [(0, 0), (0, 1), (1, 0)], mode="edge",
    )                                                       # (6, n+1, n+2)
    if _is_4d:
        divg_v_pad = divg_v_pad[..., None]
    uc = uc_raw * divg_v_pad

    # Step 4: divergence of (vc, uc) back at corners.
    #   divg_d_new(i, j) = uc(i, j-1) - uc(i, j) + vc(i-1, j) - vc(i, j)
    # In our extended uc (j ∈ [-1, n], padded slice 0..n+1):
    #   uc(i, j-1) for j ∈ [0, n] → uc[..., 0:n+1]
    #   uc(i, j)   for j ∈ [0, n] → uc[..., 1:n+2]
    # In our extended vc (i ∈ [-1, n], padded slice 0..n+1):
    #   vc(i-1, j) for i ∈ [0, n] → vc[..., 0:n+1, :]
    #   vc(i,   j) for i ∈ [0, n] → vc[..., 1:n+2, :]
    lap_divg_raw = (
        uc[:, :, 0:n + 1]
        - uc[:, :, 1:n + 2]
        + vc[:, 0:n + 1, :]
        - vc[:, 1:n + 2, :]
    )                                                       # (6, n+1, n+1)

    # Step 5: corner-removal at the 4 cube vertices — FV3 lines
    # 1773-1776.  In our extended uc (j ∈ [-1, n], shape (n+1, n+2)):
    #   FV3 uc(:, j=js-1=0) = SW/SE south-halo row → our uc[..., 0]
    #   FV3 uc(:, j=npy)    = NE/NW interior row  → our uc[..., n+1]
    sw = uc[:, 0, 0]            # uc at (i = 0,  j = -1) — south halo
    se = uc[:, n, 0]            # uc at (i = n,  j = -1) — south halo
    ne = uc[:, n, n + 1]        # uc at (i = n,  j = n)  — interior
    nw = uc[:, 0, n + 1]        # uc at (i = 0,  j = n)  — interior
    lap_divg = lap_divg_raw
    lap_divg = lap_divg.at[:, 0, 0].add(-sw)
    lap_divg = lap_divg.at[:, n, 0].add(-se)
    lap_divg = lap_divg.at[:, n, n].add(ne)
    lap_divg = lap_divg.at[:, 0, n].add(nw)

    # Step 6: normalise by rarea_c (Fortran line 1782).
    # FV3_3D iter-1044: broadcast 3D rarea_c against 4D lap_divg.
    _rarea_c = cdgrid.rarea_c[..., None] if _is_4d else cdgrid.rarea_c
    return lap_divg * _rarea_c


def _laplacian_iteration_with_vector_fill(
    divg_pad: jnp.ndarray,
    cdgrid: CubedSphereCDGrid,
    n: int,
) -> jnp.ndarray:
    """FV3-fully-faithful Laplacian iteration with vector cube-vertex fill.

    Differs from the default path: vc / uc are computed on the FV3
    D-grid layout shape ((6, n+2, n+3) and (6, n+3, n+2) respectively),
    and ``fv3_fill_corners_dgrid_vector`` is applied between gradient
    and divergence (matching FV3 ``sw_core.F90:1762``).

    For nord = 1 (single iteration with nt = 0) the divergence
    operator and corner-removal access cells that are unaffected by
    the vector fill — the output is bit-for-bit identical to the
    default path.  Verified by ``test_corner_laplacian_vector_fill_is_noop_for_nord1``.

    For nord >= 2, the inner iterations (nt > 0) extend into halo
    rows that DO read fill-written cells, so the flag is genuinely
    effective there.  This function is preparation for a future
    nord >= 2 fidelity fix.
    """
    from legoesm.grids._fv3_dgrid_corner_fill import (
        fv3_fill_corners_dgrid_vector,
    )

    # FV3_3D iter-1044: 4D-broadcast guard for the metric arrays.
    _is_4d = divg_pad.ndim == 4

    # vc at i ∈ [-1, n], j ∈ [-1, n+1] — shape (6, n+2, n+3).
    vc_raw = (
        divg_pad[:, 1:n + 3, 0:n + 3]
        - divg_pad[:, 0:n + 2, 0:n + 3]
    )
    divg_u = cdgrid.dy_edge_x * cdgrid.rdxc                  # (6, n+1, n)
    divg_u_pad = jnp.pad(
        divg_u, [(0, 0), (1, 0), (1, 2)], mode="edge",
    )                                                       # (6, n+2, n+3)
    if _is_4d:
        divg_u_pad = divg_u_pad[..., None]
    vc = vc_raw * divg_u_pad

    # uc at i ∈ [-1, n+1], j ∈ [-1, n] — shape (6, n+3, n+2).
    uc_raw = (
        divg_pad[:, 0:n + 3, 1:n + 3]
        - divg_pad[:, 0:n + 3, 0:n + 2]
    )
    divg_v = cdgrid.dx_edge_y * cdgrid.rdyc                  # (6, n, n+1)
    divg_v_pad = jnp.pad(
        divg_v, [(0, 0), (1, 2), (1, 0)], mode="edge",
    )                                                       # (6, n+3, n+2)
    if _is_4d:
        divg_v_pad = divg_v_pad[..., None]
    uc = uc_raw * divg_v_pad

    # Apply FV3 vector cube-vertex fill (sw_core.F90:1762).
    # This overwrites the 4 cube-vertex halo cells of vc / uc with the
    # sign-flipped diagonal mirror of the OTHER component.
    vc, uc = fv3_fill_corners_dgrid_vector(vc, uc, n)

    # Divergence at interior corners (i, j) ∈ [0, n] × [0, n].
    # In the wider vc shape (n+2, n+3), padded i = 0..n+1 = i_logical = -1..n;
    # padded j = 0..n+2 = j_logical = -1..n+1.
    # vc(i-1, j) for (i, j) ∈ [0, n]² → padded i = 0..n, padded j = 1..n+1.
    # vc(i,   j) for (i, j) ∈ [0, n]² → padded i = 1..n+1, padded j = 1..n+1.
    # uc(i, j-1) for (i, j) ∈ [0, n]² → padded i = 1..n+1, padded j = 0..n.
    # uc(i,   j) for (i, j) ∈ [0, n]² → padded i = 1..n+1, padded j = 1..n+1.
    lap_divg_raw = (
        uc[:, 1:n + 2, 0:n + 1]      # uc(i, j-1)
        - uc[:, 1:n + 2, 1:n + 2]    # uc(i, j)
        + vc[:, 0:n + 1, 1:n + 2]    # vc(i-1, j)
        - vc[:, 1:n + 2, 1:n + 2]    # vc(i, j)
    )                                                       # (6, n+1, n+1)

    # Corner removal — same logical positions as the default path
    # but adjusted indices for the wider arrays.
    # SW: uc(i_logical = 0, j_logical = -1) → padded i = 1, j = 0.
    # SE: uc(i_logical = n, j_logical = -1) → padded i = n+1, j = 0.
    # NE: uc(i_logical = n, j_logical = n)  → padded i = n+1, j = n+1.
    # NW: uc(i_logical = 0, j_logical = n)  → padded i = 1,   j = n+1.
    sw = uc[:, 1, 0]
    se = uc[:, n + 1, 0]
    ne = uc[:, n + 1, n + 1]
    nw = uc[:, 1, n + 1]
    lap_divg = lap_divg_raw
    lap_divg = lap_divg.at[:, 0, 0].add(-sw)
    lap_divg = lap_divg.at[:, n, 0].add(-se)
    lap_divg = lap_divg.at[:, n, n].add(ne)
    lap_divg = lap_divg.at[:, 0, n].add(nw)

    _rarea_c = cdgrid.rarea_c[..., None] if _is_4d else cdgrid.rarea_c
    return lap_divg * _rarea_c
