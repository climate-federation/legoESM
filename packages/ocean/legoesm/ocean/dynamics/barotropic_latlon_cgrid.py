"""Barotropic solver for the lat-lon C-grid FV ocean model.

Forward-backward substeps for 2D free-surface gravity waves on the
Arakawa C-grid:

    d(eta)/dt = -div(H_total * U_bar, H_total * V_bar)   [cell centers]
    d(U_bar)/dt = f * V_bar_at_u - g * d(eta)/dx          [u-points]
    d(V_bar)/dt = -f * U_bar_at_v - g * d(eta)/dy         [v-points]

The C-grid layout uses compact (single-cell) gradient and divergence
stencils, eliminating the 2*dx checkerboard null space of the A-grid.

Parallels barotropic_latlon.py (A-grid) but with staggered variables.

Wide-halo mode (opt-in, ``BarotropicConfig.barotropic_wide_halo``)
------------------------------------------------------------------
The standard substep loop re-dispatches ~4 latitude halo pads per substep
(``pad_ns_zero`` on the column thickness, the eta-PGF ``gradient_y_cgrid``,
``interp_u_to_vface_4pt``, and the eta-diffusion gradient), i.e. O(4 x
n_substeps) messages per baroclinic step on a band decomposition — a pure
latency term at high rank counts.  The wide-halo path instead exchanges ONE
halo of width ``W = k x r`` (``k`` substeps per exchange, ``r`` = the
per-substep stencil reach) and runs ``k`` substeps communication-free on the
extended band: halo garbage creeps inward at most ``r`` rows per substep, so
after ``k`` substeps the owned rows are still exact.  Serial results are
value-identical (the extended band reproduces the serial neighborhood);
MPI/SPMD parity is gated by tests.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.grids.latlon import LatLonGrid
from legoesm.ocean.vertical import OceanZStarCoordinate, compute_layer_thickness
from legoesm.ocean.state import LatLonCGridOceanState, LatLonCGridOceanConfig
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    fold_is_local,
    north_fold_mask,
    apply_north_fold,
    compute_vertex_mask,
    divergence_cgrid,
    fold_vface_row,
    gradient_x_cgrid,
    gradient_y_cgrid,
    interp_u_to_vface_4pt,
    min_cell_to_uface,
    min_cell_to_vface,
    pad_ns_zero,
    pv_flux_al81_partial_cell,
    vertex_coriolis,
)
from legoesm.grids.halo_latlon import zero_polar_lat_ends as _zero_polar_lat_ends
from legoesm.ocean.dynamics.eta_floor import clamp_and_redistribute as _clamp_redistribute
from legoesm.ocean.dynamics.barotropic_common import (
    bebt_blend,
    compute_filter_weights,
    compute_nemo_boxcar_centred_weights,
    compute_power_law_filter_weights,
    coriolis_at_faces,
    maxvel_clip,
)
from legoesm.ocean.dynamics.ocean_tendency_common import (
    depth_average_to_faces,
)

# --- NEMO dynspg_ts nn_bt_flt=3 (Demange 2019 dissipative FB) coefficients ---
# (dynspg_ts.F90:536-553 velocity AB3 extrapolation; ts_bck_interp the 4-level
# backward ssh interpolation; rn_bt_alpha = GYRE namelist_cfg 0.07.)
_NEMO_BT_ALPHA = 0.07              # rn_bt_alpha [1]
_NEMO_AB3_ZA = (1.781105, -1.06221, 0.281105)   # 3/2+bet, -(1/2+2bet), bet


# --- NEMO ts_bck_interp jn>=3 ssh half-step-back weights for rn_bt_alpha=0
#     (dynspg_ts.F90:1698-1701, nn_bt_flt=2 branch): NO Demange formula — NEMO
#     hard-codes these rounded literals (bet=0.281105, eps=0.013, gam=0.088).
_NEMO_TS_BCK_FLT2 = (0.614, 0.285, 0.088, 0.013)   # za0..za3 (sum == 1)


def nemo_ab3am4_coeff_arrays(n_loop: int, alpha: float = _NEMO_BT_ALPHA,
                             ramp: bool = True, flt2: bool = False):
    """Per-substep coefficient arrays for the NEMO AB3-AM4 barotropic substep.

    Returns ``(za, zb)`` with shapes ``(n, 3)`` / ``(n, 4)``: the AB3
    mid-step velocity-extrapolation weights and the AM4 backward ssh
    interpolation weights (temporal dissipation, ts_bck_interp).  The AB3
    velocity weights ``za`` (1.781105, -1.06221, 0.281105) are shared by both
    NEMO barotropic filters (dynspg_ts.F90:540-542).  The ssh weights ``zb``
    are the ``ts_bck_interp`` output:

    * ``flt2=False`` (nn_bt_flt=3, Demange dissipative FB): the Demange formula
      in ``rn_bt_alpha`` (``alpha``).
    * ``flt2=True`` (nn_bt_flt=2, DINO's boxcar-averaged centred scheme): the
      hard-coded ``rn_bt_alpha=0`` literals ``(0.614, 0.285, 0.088, 0.013)``
      (dynspg_ts.F90:1698-1701) — NEMO's built-in AM4 temporal dissipation that
      damps the 2Δx barotropic gravity-wave mode under the (neutral) leapfrog.

    ``ramp=True`` applies NEMO's ``ll_init`` startup on the first two substeps
    (forward, then AB2-AM3) — nn_bt_flt=3 does this ONLY at cold start
    (``ll_bt_av=F``); nn_bt_flt=2 re-inits the barotropic sub-state EVERY
    baroclinic step (``ll_init=ll_bt_av=T``, dynspg_ts.F90:202/469-476), so
    ``flt2`` callers pass ``ramp=True`` every step (no cross-window ``bt_hist``).
    Continuation windows (``state.bt_hist`` carried, nn_bt_flt=3 only) use
    ``ramp=False``: full AB3/AM4 rows from substep 0.
    """
    import numpy as np

    eps = 0.00976186 - 0.13451357 * alpha
    gam = 0.08344500 - 0.51358400 * alpha
    zb0 = 0.5 + gam + 2.0 * alpha + 2.0 * eps
    za = np.tile(np.asarray(_NEMO_AB3_ZA, dtype=np.float64), (n_loop, 1))
    if flt2:
        zb_row = np.asarray(_NEMO_TS_BCK_FLT2, dtype=np.float64)
    else:
        zb_row = np.asarray([zb0, 1.0 - zb0 - gam - eps, gam, eps],
                            dtype=np.float64)
    zb = np.tile(zb_row, (n_loop, 1))
    if ramp:
        # ll_init ramp (dynspg_ts:536-543 + ts_bck_interp jn==1/jn==2 branches)
        za[0] = (1.0, 0.0, 0.0)
        zb[0] = (1.0, 0.0, 0.0, 0.0)
        if n_loop > 1:
            za[1] = (1.0, 0.0, 0.0)
            zb[1] = (1.0833333333333, -0.1666666666666, 0.0833333333333, 0.0)
    return jnp.asarray(za), jnp.asarray(zb)


def _depth_average_to_faces(
    u_3d: jnp.ndarray,
    v_3d: jnp.ndarray,
    h_k: jnp.ndarray,
    min_water_col: jnp.ndarray,
    mask: jnp.ndarray,
    u_mask: jnp.ndarray,
    v_mask: jnp.ndarray,
    grid=None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute depth-averaged velocities at C-grid face points.

    Parameters
    ----------
    u_3d : (n_lat, n_lon+1, nlev)
    v_3d : (n_lat+1, n_lon, nlev)
    h_k : (n_lat, n_lon, nlev) layer thickness at cell centers.
    min_water_col : scalar
    mask : (n_lat, n_lon)
    u_mask : (n_lat, n_lon+1)
    v_mask : (n_lat+1, n_lon)

    Returns
    -------
    U_bar : (n_lat, n_lon+1)
    V_bar : (n_lat+1, n_lon)
    """
    # h at u/v-faces — min-rule (MOM6/MITgcm hFacW convention).
    # Must match the PE tendency and slow-forcing depth-average which
    # both use min_cell_to_uface/min_cell_to_vface.  Arithmetic mean
    # overestimates face depth at topographic steps, creating a
    # barotropic-baroclinic residual that drives spurious currents.
    h_u = min_cell_to_uface(h_k)
    h_v = min_cell_to_vface(h_k, grid)

    # Barotropic-mean face velocities: thickness-weighted depth average
    # masked by the face mask (#517 item 1: shared depth_average_to_faces;
    # floor = min_water_col, passed verbatim → bit-identical).
    U_bar = depth_average_to_faces(u_3d, h_u, u_mask, min_water_col)
    V_bar = depth_average_to_faces(v_3d, h_v, v_mask, min_water_col)

    return U_bar, V_bar


def _substep_stencil_reach(config: LatLonCGridOceanConfig) -> int:
    """Per-substep latitude stencil reach ``r`` of the FB barotropic substep.

    The number of rows one substep's output can be contaminated by per side
    when its input carries garbage beyond the valid region — the wide-halo
    width budget is ``W = k_substeps x r``.

    Term-by-term (see the substep body):

    * continuity (``pad_ns_zero(H_total)`` min-rule + flux divergence): 1;
    * semi-implicit PGF (``gradient_y_cgrid`` of the blended eta, which
      itself depends on the fresh eta): +1  → base 2;
    * ``interp_u_to_vface_4pt`` of the fresh U (whose eta dependence is
      zonal-only) lands within the same 2;
    * flux-form eta diffusion (``barotropic_diffusion_alpha > 0``): the
      grad+div chain acts on the FRESH eta → its total is 2, within base;
    * divergence damping (``barotropic_div_damp > 0``): grad(div(u,v)) on
      the fresh V (reach 2) adds 2 more (div then grad) → 4 total... the
      conservative budget below charges +1 for the V-side asymmetry too.

    The value is deliberately CONSERVATIVE (a too-small reach corrupts
    owned rows; a too-large reach only wastes halo rows) and is pinned
    mechanically by the NaN-sentinel propagation test
    (``tests/ocean/unit/test_barotropic_wide_halo.py``) — trust the test,
    not this comment, when editing the substep body.
    """
    reach = 3
    if config.barotropic.barotropic_div_damp > 0.0:
        reach += 2
    return reach


def _dissipation_coeffs(config, grid, area, dt_s, dtype, mask):
    """Precompute the substep loop's dissipation coefficient arrays.

    Returns ``(nu_face_u, nu_face_v, diff_u_mask, diff_v_mask,
    div_damp_coeff, div_damp_area_u, div_damp_area_v)`` with ``None``
    entries for statically-disabled terms (the loop gates on ``is not
    None``).  Verbatim extraction from the pre-loop block of
    ``barotropic_substeps_latlon_cgrid`` (values bit-identical).
    """
    nu_face_u = nu_face_v = diff_u_mask = diff_v_mask = None
    div_damp_coeff = div_damp_area_u = div_damp_area_v = None

    # Barotropic diffusion — flux-form with face-centered coefficient.
    # Using div(nu_face * grad(eta)) instead of nu_cell * div(grad(eta))
    # ensures exact volume conservation (divergence theorem: sum of
    # div(F)*area = 0 for any flux F with no-flux BCs).
    # The cell-center form nu_cell * laplacian(eta) is non-conservative
    # when nu_cell varies spatially (area varies as cos(lat) on latlon).
    baro_alpha = jnp.asarray(
        config.barotropic.barotropic_diffusion_alpha, dtype=dtype,
    ) * (dt_s / jnp.asarray(config.barotropic.barotropic_diffusion_dt_ref, dtype=dtype))

    # Precompute face-centered diffusion coefficients (grid geometry only,
    # constant across substeps).
    if config.barotropic.barotropic_diffusion_alpha > 0.0:
        # u-face coefficient: average of adjacent cell areas
        nu_face_u = baro_alpha * 0.5 * (jnp.roll(area, 1, axis=1) + area)
        nu_face_u = jnp.concatenate([nu_face_u, nu_face_u[:, 0:1]], axis=1)
        # v-face coefficient: average of adjacent cell areas.  Cell-pad-first
        # (PR357 Bug-2 pattern): pad the cell AREA so the v-face coefficient
        # at a partition cut averages the neighbour rank's adjacent cell area
        # (MPI halo exchange) rather than zero-padding a rank-local interior
        # average.  Pole rows are zero (wall BC) via zero_polar_lat_ends.
        area_p = pad_ns_zero(area)
        nu_face_v = baro_alpha * 0.5 * (area_p[:-1] + area_p[1:])
        # The north fold row stays zero, matching the pre-existing serial
        # behaviour (flux_y = nu_face_v * grad_y * diff_v_mask is therefore
        # zero across the seam regardless of diff_v_mask).  Enabling
        # fold-seam barotropic diffusion (a non-zero fold-row coefficient)
        # is a physics change that requires the gradient_y_cgrid fold fix
        # (PR358) and a tripolar barotropic-diffusion validation case, so it
        # is deferred to PR358 rather than introduced unvalidated here.
        nu_face_v = _zero_polar_lat_ends(nu_face_v).astype(dtype)
        # Face masks for land boundaries (zero flux at coastlines)
        diff_u_mask = mask * jnp.roll(mask, 1, axis=1)
        diff_u_mask = jnp.concatenate(
            [diff_u_mask, diff_u_mask[:, 0:1]], axis=1,
        )
        # Partial-periodic seam wall (NEMO DINO): close the barotropic-
        # diffusion seam u-face (cols 0 and n_lon) on walled rows so
        # smoothing does not leak η across the closed seam.  None → fully
        # periodic (byte-identical).
        _seam = getattr(grid, "seam_wall_rows", None)
        if _seam is not None:
            _open = (1.0 - jnp.asarray(_seam)).astype(diff_u_mask.dtype)
            diff_u_mask = diff_u_mask.at[:, 0].multiply(_open)
            diff_u_mask = diff_u_mask.at[:, -1].multiply(_open)
        # Cell-pad-first (PR357 Bug-2 pattern): pad the cell mask so the
        # v-face mask at a partition cut is the product of the two adjacent
        # cells across the cut (MPI halo exchange) rather than a halo-padded
        # interior face.  pad_ns_zero zero-pads the physical pole on every
        # rank (consistent MPI call count); the seam is overwritten only on
        # the rank that owns it (fold_is_local).
        mask_p = pad_ns_zero(mask)
        diff_v_mask = mask_p[:-1] * mask_p[1:]
        diff_v_mask = _zero_polar_lat_ends(diff_v_mask)
        nmask = north_fold_mask(grid)
        if fold_is_local(grid) or nmask is not None:
            north_dm = mask[-1:] * mask[-1:, grid.fold.perm_T]
            diff_v_mask = apply_north_fold(
                diff_v_mask, north_dm, grid, north_mask=nmask)

    # Divergence damping on barotropic velocity: grad(div(u_bar)).
    # Targets the divergent mode that creates the eta checkerboard,
    # while leaving geostrophic (rotational) flow untouched (#205).
    if config.barotropic.barotropic_div_damp > 0.0:
        div_damp_coeff = jnp.asarray(
            config.barotropic.barotropic_div_damp, dtype=dtype,
        ) * (dt_s / jnp.asarray(config.barotropic.barotropic_diffusion_dt_ref, dtype=dtype))
        # u-face area: average of adjacent cells
        div_damp_area_u = 0.5 * (jnp.roll(area, 1, axis=1) + area)
        div_damp_area_u = jnp.concatenate(
            [div_damp_area_u, div_damp_area_u[:, 0:1]], axis=1,
        )
        # v-face area: average of adjacent cells.  Cell-pad-first so the
        # partition-cut v-face averages the neighbour rank's adjacent cell
        # area (MPI halo); pole rows are zero (wall BC) via
        # zero_polar_lat_ends.
        area_p_dd = pad_ns_zero(area)
        div_damp_area_v = 0.5 * (area_p_dd[:-1] + area_p_dd[1:])
        div_damp_area_v = _zero_polar_lat_ends(div_damp_area_v).astype(dtype)

    return (nu_face_u, nu_face_v, diff_u_mask, diff_v_mask,
            div_damp_coeff, div_damp_area_u, div_damp_area_v)


def _build_een_barotropic_inputs(h_k, grid, mask, u_mask, v_mask, dtype,
                                 metric_complete=False):
    """Precompute the geometry inputs for the EEN barotropic Coriolis (node 16).

    NEMO ``dyn_spg_ts::dyn_cor_2D`` applies an ENSTROPHY-conserving EEN
    (``ln_dynvor_een``) barotropic Coriolis whose ``ffu``/``ffv`` coefficients
    are the DEPTH-INTEGRAL of the Arakawa-Lamb (1981) 12-point triad
    (dynspg_ts.F90 ``dyn_cor_2D_init`` np_EEN, ~L1329-1354).  Rather than
    hand-remap NEMO's staggered ``(i,j)`` indices (offset-by-one from the
    lego C-grid convention), :func:`een_barotropic_coriolis` REUSES the
    already-verified lego-convention EEN operator
    :func:`pv_flux_al81_partial_cell` (``q=(f+ζ)/e3f`` on the 12-point triad,
    energy+enstrophy conserving, ``ζ=0`` here for the pure planetary
    barotropic Coriolis) applied to the depth-broadcast barotropic velocity
    and depth-integrated with the same ``e3u·e3v`` volume weighting NEMO uses.

    Returns a dict of reference (η-independent, like NEMO's frozen arrays)
    3-D face/vertex thicknesses, the vertex Coriolis ``f_vtx``, the vertex and
    3-D face masks, and the column depths ``hu``/``hv`` — all static geometry.
    """
    e3u = min_cell_to_uface(h_k).astype(dtype)      # (nlat, nlon+1, nlev)
    e3v = min_cell_to_vface(h_k, grid).astype(dtype)  # (nlat+1, nlon, nlev)
    hu = jnp.sum(e3u, axis=-1)                       # (nlat, nlon+1)
    hv = jnp.sum(e3v, axis=-1)                       # (nlat+1, nlon)
    # F-point (vertex) thickness = min over the 4 surrounding cells with a
    # BIG_H sentinel at dry cells (MITgcm hFacZ / NEMO e3f_vor; same min-rule
    # the 3-D EEN caller uses in ocean_pe_latlon_cgrid.py:1817).  Any positive
    # e3f keeps the EEN energy conservation (a property of the triad pairing,
    # not the e3f value), so the min-corner choice is a tier-3 detail.
    from legoesm.grids.halo_latlon import pad_with_pole_bc_lat_multi
    BIG = 1.0e30
    ha = jnp.where(h_k > 0.0, h_k, BIG)
    h_sw = jnp.roll(ha, 1, axis=1)                   # west neighbour
    hp, hswp = pad_with_pole_bc_lat_multi(
        (ha, h_sw), halo=1, south_values=(BIG, BIG), north_values=(BIG, BIG))
    h_vtx = jnp.minimum(jnp.minimum(hp[:-1], hp[1:]),
                        jnp.minimum(hswp[:-1], hswp[1:]))
    nmask = north_fold_mask(grid)
    if fold_is_local(grid) or nmask is not None:
        fold = grid.fold
        ha_p = ha[-1:, fold.perm_T, :]
        hsw_p = h_sw[-1:, fold.perm_T, :]
        h_vtx_north = jnp.minimum(
            jnp.minimum(ha[-1:], h_sw[-1:]), jnp.minimum(ha_p, hsw_p))
        h_vtx = apply_north_fold(h_vtx, h_vtx_north, grid, north_mask=nmask)
    h_vtx = jnp.concatenate([h_vtx, h_vtx[:, 0:1, :]], axis=1).astype(dtype)
    f_vtx = vertex_coriolis(grid).astype(dtype)      # (nlat+1, nlon+1)
    vtx_mask = compute_vertex_mask(mask, grid=grid)
    u_mask_3d = (u_mask[..., None] * (e3u > 0)).astype(dtype)
    v_mask_3d = (v_mask[..., None] * (e3v > 0)).astype(dtype)
    out = dict(e3u=e3u, e3v=e3v, hu=hu, hv=hv, h_vtx=h_vtx, f_vtx=f_vtx,
               vtx_mask=vtx_mask, u_mask_3d=u_mask_3d, v_mask_3d=v_mask_3d,
               metric_complete=bool(metric_complete))
    if metric_complete:
        # NEMO horizontal scale factors (dyn_cor_2D_init, dynspg_ts.F90:1349-
        # 1379): e1u/e1v [zonal widths] and e2u/e2v [meridional widths] at the
        # u-/v-points.  lego geometry names them dx_u/dx_v/dy_u/dy_v.  These are
        # the metric factors the per-unit-width AL81 operator DROPS; folding them
        # in (see een_barotropic_coriolis) makes the discrete enstrophy budget
        # exact on the sphere.  NB (twin-verified) it is a ~1% high-lat correction,
        # NOT the cure for the |lat|~68° 2Δx eta runaway.
        from legoesm.grids.latlon import ensure_geometry
        geom = ensure_geometry(grid)   # idempotent: pass-through if already geom
        out.update(e1u=geom.dx_u.astype(dtype), e1v=geom.dx_v.astype(dtype),
                   e2u=geom.dy_u.astype(dtype), e2v=geom.dy_v.astype(dtype))
    return out


def een_barotropic_coriolis(U_bar, V_bar, pre, eps=1.0e-10):
    """EEN barotropic Coriolis tendency (``cor_u``, ``cor_v``) — node 16.

    ``cor_u = (1/hu)·Σ_k e3u_k·diag_u_k``, ``cor_v = (1/hv)·Σ_k e3v_k·diag_v_k``
    where ``(diag_u, diag_v) = pv_flux_al81_partial_cell(ζ=0, f_vtx=f, …)`` with
    ``u,v`` = the (depth-independent) barotropic velocities broadcast over the
    ``nlev`` reference levels.  The ``e3u·e3v`` depth weighting reproduces
    NEMO's ``ffu``/``ffv`` volume weighting; in the flat-bottom limit this
    reduces to ``f·V̄`` (verified to 1e-4).  Both components are evaluated at
    the SAME time level (NEMO ``dyn_cor_2D`` uses ``punb``/``pvnb`` together) —
    required for the EEN energy conservation.

    Two variants (``pre["metric_complete"]``):

    * ``False`` (``barotropic_coriolis="een"``): the per-unit-width AL81 operator,
      which DROPS the ``e1``/``e2`` horizontal metric factors NEMO keeps in
      ``ffu``/``ffv``.  Energy+enstrophy conservation is exact only on a uniform-
      metric grid; on lat-lon it leaves an ``O(Δcos φ)`` work residual (~4e-4),
      the SAME character as the baseline baroclinic AL81.

    * ``True`` (``barotropic_coriolis="een_metric"``): METRIC-COMPLETE — exactly
      NEMO's ``dyn_cor_2D`` ``ffu``/``ffv`` (dynspg_ts.F90:1349-1379 · 1495-1503).
      NEMO's coefficient is
      ``ffu_* = r1_12·r1_e1u·r1_hu·e1v · Σ_k[e3u·e3v·vmask·(Σ_3 f/e3f)]`` and
      ``zu_trd = Σ_* ffu_*·v_neighbor``.  Because every AL81 triad term already
      reads ``v`` at a specific v-point and this operator's triad→v-point pairing
      IS NEMO's EEN pairing, multiplying the meridional velocity FIELD by ``e1v``
      (the v-point zonal width) attaches each term's neighbour ``e1v`` exactly as
      NEMO does; dividing the output by ``e1u`` (the u-point zonal width) supplies
      ``r1_e1u``.  Symmetrically ``v`` uses ``e2u``·``r1_e2v`` (:1376-1379).  The
      metric factors are per-column (2-D), so they factor cleanly through the
      depth integral and the ``1/hu`` normalisation.  This is algebraically the
      materialised ``ffu``/``ffv`` — without hand-transcribing NEMO's offset-by-
      one stencil into the lego index convention.  Retaining ``e1v/e1u`` (∝Δcosφ)
      makes discrete enstrophy conservation exact on the sphere (the coefficients
      NEMO's ``dyn_cor_2D`` actually uses).  NB it is a ~1% high-lat correction:
      twin-verified, it does NOT by itself cure the DINO |lat|~68° 2Δx *eta*
      runaway (a free-surface mode via the barotropic PGF/continuity coupling) —
      it is a fidelity refinement over the metric-less "een".
    """
    nlev = pre["e3u"].shape[-1]
    U_src, V_src = U_bar, V_bar
    if pre.get("metric_complete", False):
        # Fold in the NEMO horizontal metrics (see docstring): scale the mass-flux
        # velocities by the neighbour width, normalise the output by the local
        # width. e2u/e1v are (…,) 2-D face arrays broadcast over levels below.
        U_src = U_bar * pre["e2u"]     # u-flux carries e2u (NEMO ffv e2u factor)
        V_src = V_bar * pre["e1v"]     # v-flux carries e1v (NEMO ffu e1v factor)
    u3 = jnp.broadcast_to(U_src[..., None], U_src.shape + (nlev,))
    v3 = jnp.broadcast_to(V_src[..., None], V_src.shape + (nlev,))
    diag_u, diag_v = pv_flux_al81_partial_cell(
        jnp.zeros_like(pre["h_vtx"]), pre["h_vtx"], pre["e3v"], v3,
        pre["e3u"], u3, pre["u_mask_3d"], pre["v_mask_3d"], pre["vtx_mask"],
        f_vtx=pre["f_vtx"])
    cor_u = jnp.sum(pre["e3u"] * diag_u, axis=-1) / jnp.maximum(pre["hu"], eps)
    cor_v = jnp.sum(pre["e3v"] * diag_v, axis=-1) / jnp.maximum(pre["hv"], eps)
    if pre.get("metric_complete", False):
        # eps floor mirrors the /hu guard: e1u=R·cosφ·dλ→0 only on a pole-covering
        # row (never for DINO); e2v is always positive.
        cor_u = cor_u / jnp.maximum(pre["e1u"], eps)   # NEMO r1_e1u
        cor_v = cor_v / jnp.maximum(pre["e2v"], eps)   # NEMO r1_e2v
    return cor_u, cor_v


def barotropic_coriolis_een_pre_step(u_3d, v_3d, h_k, grid, mask, u_mask,
                                     v_mask, min_water_col, dtype,
                                     metric_complete=False):
    """Pre-step EEN barotropic Coriolis ``(cor_u, cor_v)`` for the live split.

    NEMO ``dynspg_ts.F90:296-300`` subtracts ``dyn_cor_2D(puu_b, pvv_b)`` — the
    barotropic Coriolis of the BEFORE barotropic transport — from ``zu_frc``
    before the substep loop applies the SAME operator live on the evolving
    ``ua_e``/``va_e`` each substep.  This public wrapper builds the identical
    EEN geometry inputs (:func:`_build_een_barotropic_inputs`) and the identical
    thickness-weighted barotropic mean (:func:`_depth_average_to_faces`) the
    substep loop uses internally, then evaluates :func:`een_barotropic_coriolis`
    — so subtracting the result from the slow forcing cancels the substep-0 live
    term exactly (up to float round-off) and leaves ONLY the LIVE, evolving,
    null-mode-restoring EEN barotropic Coriolis inside the window.
    """
    pre = _build_een_barotropic_inputs(h_k, grid, mask, u_mask, v_mask, dtype,
                                       metric_complete=metric_complete)
    U_bar, V_bar = _depth_average_to_faces(
        u_3d, v_3d, h_k, min_water_col, mask, u_mask, v_mask, grid)
    return een_barotropic_coriolis(U_bar, V_bar, pre)


def _run_substep_loop(
    eta, U_bar, V_bar,
    *,
    dt_s, n_loop, w_filter, w_transport,
    grid, config, g, H_bathy, mask, u_mask, v_mask,
    min_water_col, eta_floor, area,
    F_slow_eta, F_slow_u, F_slow_v,
    f_u, f_v, add_barotropic_coriolis,
    coeffs, local_subcycle_clamp,
    linear_free_surface=False,
    ab3_za=None, ab3_zb=None, ab3_hist=None,
    een_pre=None,
    drag_r_u=None, drag_r_v=None,
):
    """The forward-backward substep loop (verbatim extraction).

    Runs ``n_loop`` substeps from the ``(eta, U_bar, V_bar)`` carry and
    returns the raw final carry
    ``(eta_f, U_bar_f, V_bar_f, Hu_sum_f, Hv_sum_f, eta_sum_f, U_sum_f,
    V_sum_f)`` — callers apply the ``w_total`` normalization and any
    post-loop global redistribute themselves (the wide-halo path must do
    both on the CROPPED owned rows, never on the extended band).

    ``local_subcycle_clamp`` replaces the config flag inside the loop: the
    standard path passes the config value; the wide-halo path FORCES True
    (its per-substep clamp must be allreduce-free — a global redistribute
    over the extended band would double-count the halo overlap).

    Function boundaries are invisible to tracing (Python inlining), so this
    extraction is jaxpr-identical to the previous inline loop.
    """
    (nu_face_u, nu_face_v, diff_u_mask, diff_v_mask,
     div_damp_coeff, div_damp_area_u, div_damp_area_v) = coeffs
    use_div_damp = div_damp_coeff is not None
    use_diffusion = nu_face_u is not None

    # BEBT semi-implicit parameter and MAXVEL clipping
    bebt = config.barotropic.bebt
    _maxvel = config.barotropic.maxvel_barotropic
    use_maxvel = _maxvel > 0.0
    dtype = eta.dtype

    # Accumulators for time-averaged barotropic transport (Phase 2a, issue #102).
    # These accumulate the mass fluxes H*U_bar at each substep so the tracer
    # equation can use transport consistent with the barotropic continuity.
    n_lat = eta.shape[0]
    n_lon = eta.shape[1]
    Hu_sum = jnp.zeros((n_lat, n_lon + 1), dtype=dtype)
    Hv_sum = jnp.zeros((n_lat + 1, n_lon), dtype=dtype)
    eta_sum = jnp.zeros((n_lat, n_lon), dtype=dtype)
    U_sum = jnp.zeros((n_lat, n_lon + 1), dtype=dtype)
    V_sum = jnp.zeros((n_lat + 1, n_lon), dtype=dtype)

    def substep_body(wts_i, carry):
        """Single barotropic substep with BEBT, slow forcing, MAXVEL, and cosine filter.

        Parameters
        ----------
        w_i : scalar
            Cosine filter weight for this substep (1.0 for box filter).
        carry : tuple
            (eta, U_bar, V_bar, Hu_sum, Hv_sum, eta_sum, U_sum, V_sum)
        """
        if ab3_za is None:
            w_i, w_tr_i = wts_i
        if ab3_za is not None:
            (eta_c, U_bar_c, V_bar_c,
             Hu_sum_c, Hv_sum_c, eta_sum_c, U_sum_c, V_sum_c,
             Ub_c, Ubb_c, Vb_c, Vbb_c, etab_c, etabb_c) = carry
            w_i, w_tr_i, za_i, zb_i = wts_i
        else:
            (eta_c, U_bar_c, V_bar_c,
             Hu_sum_c, Hv_sum_c, eta_sum_c, U_sum_c, V_sum_c) = carry

        if linear_free_surface:
            # NEMO key_linssh barotropic continuity: FIXED column depth H
            # (deta/dt = -div(H*U), traadv.F90 r1_hu_0 convention) — eta does
            # not feed back into the transport depth.
            H_total_c = H_bathy * mask
        else:
            H_total_c = jnp.maximum(eta_c + H_bathy, min_water_col) * mask

        # Forward: update eta from continuity (C-grid divergence)
        # Min-rule face depth (consistent with implicit solver and PE
        # tendency).  Arithmetic mean overestimates face depth at
        # topographic steps, creating a transport mismatch.
        H_u = jnp.minimum(jnp.roll(H_total_c, 1, axis=1), H_total_c)
        H_u = jnp.concatenate([H_u, H_u[:, 0:1]], axis=1)
        # Pole rows are zero (wall BC) on regular lat-lon; fold min-rule
        # on tripolar.
        # Cell-pad-first (PR357 Bug-2 pattern): pad the cell column thickness
        # so the v-face min at a partition cut uses the neighbour rank's
        # adjacent column (MPI halo exchange).  Every rank calls pad_ns_zero
        # (consistent MPI call count — the previous direct-concat fold branch
        # skipped it and deadlocked against the else branch); the fold seam is
        # overwritten only on the rank that owns it.
        H_total_pad = pad_ns_zero(H_total_c)
        H_v = jnp.minimum(H_total_pad[:-1], H_total_pad[1:])
        H_v = _zero_polar_lat_ends(H_v)
        nmask = north_fold_mask(grid)
        if fold_is_local(grid) or nmask is not None:
            north = jnp.minimum(
                H_total_c[-1:], fold_vface_row(H_total_c, grid),
            )
            H_v = apply_north_fold(H_v, north, grid, north_mask=nmask)

        if ab3_za is not None:
            # NEMO nn_bt_flt=3: mid-step AB3 velocity extrapolation
            # u^{m+1/2} = za1*u^m + za2*u^{m-1} + za3*u^{m-2}
            U_mid = za_i[0] * U_bar_c + za_i[1] * Ub_c + za_i[2] * Ubb_c
            V_mid = za_i[0] * V_bar_c + za_i[1] * Vb_c + za_i[2] * Vbb_c
        else:
            U_mid, V_mid = U_bar_c, V_bar_c
        flux_u = H_u * U_mid * u_mask
        flux_v = H_v * V_mid * v_mask

        # Accumulate transport (always box-filtered for volume conservation)
        Hu_sum_new = Hu_sum_c + w_tr_i * flux_u.astype(dtype)
        Hv_sum_new = Hv_sum_c + w_tr_i * flux_v.astype(dtype)

        div_flux = divergence_cgrid(
            flux_u, flux_v, grid, u_mask=u_mask, v_mask=v_mask,
        ).astype(dtype)
        eta_unfloored = (eta_c - dt_s * div_flux + dt_s * F_slow_eta * mask) * mask
        if local_subcycle_clamp:
            # SOTA-local (MOM6/MPAS-O): LOCAL clamp per substep — NO allreduce.
            # The global mass-conserving redistribute is deferred to ONCE per
            # outer step (post-loop, on the time-averaged eta).
            eta_new = jnp.maximum(eta_unfloored, eta_floor) * mask
        else:
            eta_new = _clamp_redistribute(eta_unfloored, eta_floor, mask, area)

        # --- BEBT: Semi-implicit barotropic PGF (#205) ---
        # Blend new and old eta for the pressure gradient to damp fast
        # barotropic gravity waves.  bebt=0 → forward-backward (current),
        # bebt=0.2 → MOM6 default semi-implicit.
        if ab3_za is not None:
            # NEMO ts_bck_interp: ssh' = zb0*ssh^{m+1} + zb1*ssh^m
            #                          + zb2*ssh^{m-1} + zb3*ssh^{m-2}
            # (the Demange temporal dissipation; replaces the bebt blend).
            eta_pgf = (zb_i[0] * eta_new + zb_i[1] * eta_c
                       + zb_i[2] * etab_c + zb_i[3] * etabb_c)
        else:
            eta_pgf = bebt_blend(eta_new, eta_c, bebt)
        deta_dx = gradient_x_cgrid(eta_pgf, grid).astype(dtype)
        deta_dy = gradient_y_cgrid(eta_pgf, grid).astype(dtype)

        # Average V to u-points for Coriolis.  In ab3am4 mode NEMO applies
        # the 2D Coriolis to the EXTRAPOLATED mid-step velocities (both
        # components simultaneously, dynspg_ts.F90:689); otherwise the FB pair.
        _V_cor_src = V_mid if ab3_za is not None else V_bar_c
        _U_cor_src_cur = U_mid if ab3_za is not None else U_bar_c
        if een_pre is not None:
            # NEMO dyn_spg_ts::dyn_cor_2D enstrophy-conserving EEN (node 16):
            # BOTH components from the SAME time-level velocities (required for
            # energy conservation; NEMO passes punb/pvnb together).  The
            # planetary f rides the depth-integrated AL81 12-point triad, which
            # exerts a restoring on the 2Δx checkerboard the 4-pt avg annihilates.
            _cor_u_een, _cor_v_een = een_barotropic_coriolis(
                _U_cor_src_cur, _V_cor_src, een_pre)
        V_west = jnp.roll(_V_cor_src, 1, axis=1)
        V_at_u = 0.25 * (_V_cor_src[:-1] + _V_cor_src[1:]
                         + V_west[:-1] + V_west[1:])
        V_at_u = jnp.concatenate([V_at_u, V_at_u[:, 0:1]], axis=1)

        # Forward-backward Coriolis (Matsuno) + PGF + slow forcing.  The
        # in-substep Coriolis is gated off when the planetary f×u already
        # reaches the barotropic mode via F_slow (Oceananigans convention) —
        # this removes the C-grid 4-point-average rotational null mode.
        if not add_barotropic_coriolis:
            _cor_u = 0.0
        elif een_pre is not None:
            _cor_u = _cor_u_een
        else:
            _cor_u = f_u * V_at_u
        # NEMO dyn_drg in-subcycle explicit bottom stress (#1226;
        # dynspg_ts.F90:701-705, the .NOT.ll_wd branch — DINO's active path;
        # the implicit division at :764-768 is wetting-drying-only, ll_wd=F
        # for DINO):
        #   zu_trd += zCdU_u * un_e * hur_e ;  ua_e = un_e + rDt_e*(spg+trd+frc)
        # zCdU_u = -r_eff (NEMO rCdU_bot <= 0; lego r_eff >= 0), un_e = the
        # substep-START velocity (NEMO uses un_e here even in AB3 mode, NOT
        # the mid-step extrapolation), hur_e = 1/(u-face column depth at
        # substep level jn) — lego's min-rule H_u/H_v of the carry eta (NEMO
        # updates hu_e per substep from the fresh ssh the same way,
        # dynspg_ts.F90:771-778; min-rule vs ssh-average face depth is lego's
        # documented global face-depth convention).
        # SIGN (positive-r damping convention): dU/dt += -r_eff*U/H opposes
        # U_bar — strictly reduces |U_bar| (drag can never accelerate).
        # Static Python gate (drag_r_u is a closure capture) — no traced
        # control flow, carry unchanged, off ⇒ byte-identical.
        if drag_r_u is not None:
            _drag_u = -drag_r_u * U_bar_c / jnp.maximum(H_u, min_water_col)
        else:
            _drag_u = 0.0
        U_bar_new = (U_bar_c + dt_s * (
            _cor_u + _drag_u - g * deta_dx + F_slow_u
        )) * u_mask

        # U averaged to v-points for the backward Coriolis half-step,
        # cell-pad-first (shared interp_u_to_vface_4pt): the partition-
        # cut v-face uses the exact serial 4-point average instead of
        # the neighbour's adjacent FACE row (one row off — the old
        # interior-then-pad_ns_vector_u pattern).  Serial bit-identical;
        # one cell pad per substep replaces one face pad per substep
        # (same collective count on every rank).
        _U_cor_src = U_mid if ab3_za is not None else U_bar_new
        U_new_at_v = interp_u_to_vface_4pt(_U_cor_src, grid)
        if not add_barotropic_coriolis:
            _cor_v = 0.0
        elif een_pre is not None:
            # EEN cor_v computed above from the SAME-time-level U (not the
            # backward U_bar_new) — the energy-conserving pairing.
            _cor_v = _cor_v_een
        else:
            _cor_v = -f_v * U_new_at_v
        # NEMO dynspg_ts.F90:704: zv_trd += zCdU_v * vn_e * hvr_e — same
        # substep-START velocity + carry-eta face depth as the u-drag above.
        if drag_r_v is not None:
            _drag_v = -drag_r_v * V_bar_c / jnp.maximum(H_v, min_water_col)
        else:
            _drag_v = 0.0
        V_bar_new = (V_bar_c + dt_s * (
            _cor_v + _drag_v - g * deta_dy + F_slow_v
        )) * v_mask

        # Divergence damping: grad(div(u_bar)) (#205)
        if use_div_damp:
            div_uv = divergence_cgrid(
                U_bar_new, V_bar_new, grid,
                u_mask=u_mask, v_mask=v_mask,
            ).astype(dtype)
            grad_div_x = gradient_x_cgrid(div_uv * mask, grid).astype(dtype)
            grad_div_y = gradient_y_cgrid(div_uv * mask, grid).astype(dtype)
            U_bar_new = (
                U_bar_new + div_damp_coeff * div_damp_area_u * grad_div_x
            ) * u_mask
            V_bar_new = (
                V_bar_new + div_damp_coeff * div_damp_area_v * grad_div_y
            ) * v_mask

        # Bottom drag — TWO mutually-exclusive compositions (single owner per
        # config; finding #6 + #1226 dyn_drg):
        # * ``barotropic_drag_substep=False`` (default): the 3D PE tendency
        #   (``_bc_bottom_drag``) applies the full bottom drag ``-r·u_bot/h_bot``
        #   to ``du_dt``; its depth-mean ``-r·u_bot/H`` is carried into the
        #   barotropic mode through ``F_slow_u``/``F_slow_v`` at every substep
        #   above.  Re-applying ``implicit_bottom_drag_factor`` here would make
        #   the effective barotropic-mode drag ``≈ 2·r/H`` (codex iter-2
        #   finding #1) — so no in-loop drag on this path.  (The implicit-CN
        #   solver relies on F_slow alone the same way.)
        # * ``barotropic_drag_substep=True`` (requires ``zdf_drag_in_matrix``,
        #   which SKIPS ``_bc_bottom_drag`` — so F_slow carries no drag): the
        #   NEMO dyn_drg composition instead — the per-substep explicit
        #   ``-r_eff·U/H`` term applied in the updates above
        #   (dynspg_ts.F90:701-705) plus the once-per-step ``pu_RHSi``
        #   baroclinic-residual correction folded into F_slow by the model
        #   (dynspg_ts.F90:1627-1642).  Exactly one owner in each mode.

        # --- MAXVEL clipping: prevent runaway velocities ---
        if use_maxvel:
            U_bar_new = maxvel_clip(U_bar_new, _maxvel)
            V_bar_new = maxvel_clip(V_bar_new, _maxvel)

        # Optional Laplacian damping on eta (flux-form: conservative)
        if use_diffusion:
            grad_x = gradient_x_cgrid(eta_new * mask, grid)
            grad_y = gradient_y_cgrid(eta_new * mask, grid)
            flux_x = nu_face_u * grad_x * diff_u_mask
            flux_y = nu_face_v * grad_y * diff_v_mask
            eta_new = (
                eta_new + divergence_cgrid(flux_x, flux_y, grid).astype(dtype)
            ) * mask
            if local_subcycle_clamp:
                eta_new = jnp.maximum(eta_new, eta_floor) * mask
            else:
                eta_new = _clamp_redistribute(eta_new, eta_floor, mask, area)

        # Accumulate eta, U_bar, V_bar with cosine filter weights
        eta_sum_new = eta_sum_c + w_i * eta_new
        U_sum_new = U_sum_c + w_i * U_bar_new
        V_sum_new = V_sum_c + w_i * V_bar_new

        if ab3_za is not None:
            # rotate the AB3/AM4 histories (dynspg_ts:805-815)
            return (eta_new, U_bar_new, V_bar_new,
                    Hu_sum_new, Hv_sum_new, eta_sum_new, U_sum_new, V_sum_new,
                    U_bar_c, Ub_c, V_bar_c, Vb_c, eta_c, etab_c)
        return (eta_new, U_bar_new, V_bar_new,
                Hu_sum_new, Hv_sum_new, eta_sum_new, U_sum_new, V_sum_new)

    if ab3_za is not None:
        if ab3_hist is not None:
            # NEMO continuation (dynspg_ts ll_init=F): now-values reset to the
            # baroclinic state (ln_bt_fw), b/bb histories carried from the end
            # of the PREVIOUS window — one continuous AB3 series across windows.
            # DEVIATION form: ab3_hist holds (X_final - X_b, X_final - X_bb)
            # of the previous window, reconstructed against THIS window's
            # now-values. NEMO re-imposes the stp2d barotropic mean on the 3D
            # velocity after every stage (stprk3_stg.F90:440 zub correction),
            # so its raw-carried histories never see a window-boundary jump;
            # legoESM's implicit vmix/bottom drag shift the depth mean after
            # the solve, and a raw carry would feed that jump into the AB3
            # extrapolation (x1.78 amplification) every window — pumping a
            # spurious deep barotropic mode. Deviation form is identical to
            # NEMO's raw carry when the mean is preserved (NEMO's case) and
            # jump-transparent when it is not.
            (dU_b, dU_bb, dV_b, dV_bb, deta_b, deta_bb) = (
                h.astype(dtype) for h in ab3_hist)
            Ub0 = U_bar - dU_b
            Ubb0 = U_bar - dU_bb
            Vb0 = V_bar - dV_b
            Vbb0 = V_bar - dV_bb
            etab0 = eta - deta_b
            etabb0 = eta - deta_bb
        else:
            # cold start: histories = window-start values; with the ll_init
            # ramp rows 0-1 these never reach a full-AB3 row (NEMO-exact).
            Ub0 = Ubb0 = U_bar
            Vb0 = Vbb0 = V_bar
            etab0 = etabb0 = eta
        init_carry = (eta, U_bar, V_bar, Hu_sum, Hv_sum, eta_sum, U_sum,
                      V_sum, Ub0, Ubb0, Vb0, Vbb0, etab0, etabb0)
        _xs = (w_filter, w_transport, ab3_za, ab3_zb)
    else:
        init_carry = (eta, U_bar, V_bar, Hu_sum, Hv_sum, eta_sum, U_sum, V_sum)
        _xs = (w_filter, w_transport)

    if config.barotropic.differentiable_barotropic:
        # scan path: pass (averaging, transport) weights as xs per substep
        def scan_body(carry, wts_i):
            new_carry = substep_body(wts_i, carry)
            return new_carry, None

        finals, _ = jax.lax.scan(
            scan_body, init_carry, xs=_xs, length=n_loop,
        )
    else:
        # fori_loop path: index into the filter + transport weights
        def fori_body(i, carry):
            if ab3_za is not None:
                return substep_body(
                    (w_filter[i], w_transport[i], ab3_za[i], ab3_zb[i]), carry)
            return substep_body((w_filter[i], w_transport[i]), carry)

        finals = jax.lax.fori_loop(0, n_loop, fori_body, init_carry)

    return finals


def _compute_weights(config, n_substeps: int, dtype, substep_scale: int = 1):
    """Filter + transport weights for the substep loop (shared verbatim).

    Cosine time filter for time-averaging (replaces box-average).
    Cosine-bell (Hanning) window suppresses the side lobes of the box
    filter that alias barotropic modes into the baroclinic coupling.
    Averaging filter for eta/U/V + the matching transport weights for Hu/Hv.
    "power_law" = Shchepetkin-McWilliams (2005) extended-window filter (ROMS/
    MOM6/Oceananigans), which damps the 2Δx barotropic Coriolis null mode the
    first-order cosine filter excites (docs/issues/barotropic_mode_noise.md).
    For box/cosine the transport weights are NO LONGER a flat 1/n: they are the
    continuity-consistent SM2005 tail-sum ``tail_j/(n·w_total)`` returned by
    compute_filter_weights, so the discrete continuity invariant
    ``div(Hu_avg) == (eta_old - eta_avg)/dt`` — exact when the eta floor does
    not bind (the clamp breaks local telescoping; global mass is restored by
    the post-loop redistribute) — (which the flux-form tracer step
    needs to preserve a uniform tracer) holds for EVERY filter — the flat 1/n
    broke it for both box (~95%) and cosine (~99%).
    """
    if config.barotropic.barotropic_time_filter == "nemo_ab3am4":
        # NEMO nn_bt_flt=3 (ll_bt_av=F): NO time averaging of the state (the
        # final substep IS the answer; dissipation is temporal, via the AM4
        # backward ssh interpolation). Tracer transports are the PLAIN mean
        # (ts_wgt ll_av=F secondary weights are uniform), which telescopes
        # continuity exactly: div(Hu_avg) == (eta_old - eta_final)/dt.
        w_filter = jnp.zeros((n_substeps,), dtype=dtype)      # sums unused
        w_total = jnp.asarray(1.0, dtype=dtype)
        w_transport = jnp.full((n_substeps,), 1.0 / n_substeps, dtype=dtype)
        n_loop = n_substeps
    elif config.barotropic.barotropic_time_filter == "power_law":
        w_filter, w_total, w_transport, n_loop = compute_power_law_filter_weights(
            n_substeps, dtype,
        )
    elif config.barotropic.barotropic_time_filter in (
            "nemo_boxcar_centred", "nemo_boxcar_ab3"):
        # NEMO dynspg_ts ln_bt_fw=F + nn_bt_flt=2 boxcar averaging window (see
        # compute_nemo_boxcar_centred_weights).  "nemo_boxcar_ab3" is the FULL
        # nn_bt_flt=2 substep — the SAME boxcar averaging PLUS the AB3 velocity
        # predictor + ts_bck_interp(α=0) ssh temporal dissipation (activated by
        # the `_ab3` gate in barotropic_substeps_latlon_cgrid); the weights
        # (window + boxcar averaging) are identical.
        w_filter, w_total, w_transport, n_loop = (
            compute_nemo_boxcar_centred_weights(
                n_substeps, dtype, substep_scale=substep_scale))
    else:
        use_cosine_filter = config.barotropic.barotropic_time_filter == "cosine"
        w_filter, w_total, w_transport = compute_filter_weights(
            n_substeps, dtype, use_cosine=use_cosine_filter,
        )
        n_loop = n_substeps
    return w_filter, w_total, w_transport, n_loop


def barotropic_substeps_latlon_cgrid(
    state: LatLonCGridOceanState,
    dt_s: float,
    n_substeps: int,
    grid: LatLonGrid,
    z_coord: OceanZStarCoordinate,
    config: LatLonCGridOceanConfig,
    F_slow_eta=None,
    F_slow_u=None,
    F_slow_v=None,
    add_barotropic_coriolis: bool = True,
    t_seconds=None,
    eta_init=None,
    u_init=None,
    v_init=None,
    substep_scale: int = 1,
) -> LatLonCGridOceanState:
    """Run barotropic substeps on a C-grid lat-lon grid.

    ``eta_init`` / ``u_init`` / ``v_init`` (default ``None``) override the time
    level the split-explicit barotropic INTEGRATION is seeded from — the ssh and
    depth-mean transport that start the substep loop.  ``None`` (every non-MLF
    caller) seeds from ``state`` (the NOW level, forward-frame).  The NEMO
    Modified-Leap-Frog step (``_leapfrog_step``) passes the BEFORE-level
    ``(eta_before, u_before, v_before)`` so the barotropic mode leap-frogs
    ``n-1 → n+1`` exactly as NEMO ``dyn_spg_ts`` under ``ln_bt_fw=.FALSE.``
    (``dynspg_ts.F90:494-503``: ``sshn_e=pssh(Kbb)``, ``un_e=puu_b(Kbb)``,
    ``vn_e=pvv_b(Kbb)``).  The frozen slow forcing ``F_slow_*`` stays at the NOW
    level (NEMO's ``zu_frc``, assembled from the now 3-D RHS), and the 3-D
    depth-mean REPLACEMENT (``u' = u − ū``) still uses ``state``'s NOW velocity —
    only the fast-mode integration is re-seeded.  With ``None`` the two
    depth-means coincide ⇒ byte-identical for every other caller.

    ``add_barotropic_coriolis`` (default True) applies the explicit f×U_bt
    Coriolis term inside each substep.  Set False when the planetary Coriolis
    already reaches the barotropic mode through ``F_slow_u/v`` (its depth-mean,
    via ``coriolis_scheme="explicit_ab2"``) — this is the Oceananigans /
    split-explicit convention (the barotropic equation is ∂_tU = −gH∇η + G^U,
    with NO in-substep Coriolis), and it avoids the C-grid 4-point Coriolis
    rotational null mode (the 2Δx barotropic checkerboard; see
    docs/issues/barotropic_mode_noise.md §A) that otherwise grows under an
    eddy field and blows the eddy-resolving jet.

    Parameters
    ----------
    state : LatLonCGridOceanState
        State after slow tendency application.
    dt_s : float
        Substep size [seconds].
    n_substeps : int
    grid : LatLonGrid
    z_coord : OceanZStarCoordinate
    config : LatLonCGridOceanConfig
    F_slow_eta : jax.Array or None, shape (n_lat, n_lon)
        Slow forcing for eta (e.g., freshwater mass flux) [m/s].
    F_slow_u : jax.Array or None, shape (n_lat, n_lon+1)
        Depth-averaged slow forcing for u (baroclinic PGF + viscosity
        + advection) [m/s^2].  Passed as constant forcing each substep
        for MOM6-style slow-forcing coupling.
    F_slow_v : jax.Array or None, shape (n_lat+1, n_lon)
        Depth-averaged slow forcing for v [m/s^2].

    Returns
    -------
    LatLonCGridOceanState with updated eta and velocity.
    """
    g = jnp.asarray(config.g)
    H_bathy = state.H_bathy.data
    mask = state.land_mask.data
    u_mask = state.u_mask.data
    v_mask = state.v_mask.data
    # NOW-level velocity for the 3-D depth-mean replacement (u' = u − ū); the
    # barotropic INTEGRATION is seeded from u_init/v_init/eta_init when the MLF
    # step supplies the BEFORE level (else these coincide, byte-identical).
    u_corr = state.u.data
    v_corr = state.v.data
    u = u_init if u_init is not None else state.u.data
    v = v_init if v_init is not None else state.v.data
    eta_raw = eta_init if eta_init is not None else state.eta.data
    _seed_override = (eta_init is not None or u_init is not None
                      or v_init is not None)
    # All-or-none: a partial before-level seed would mix (e.g.) the NOW velocity
    # over the BEFORE eta-thickness — a silent inconsistency.  The MLF always
    # passes all three; reject any partial override (static Python check).
    if _seed_override and (eta_init is None or u_init is None or v_init is None):
        raise ValueError(
            "barotropic before-level seed must supply ALL of eta_init/u_init/"
            "v_init or NONE (a partial seed mixes NOW/BEFORE time levels); got "
            f"eta_init={'set' if eta_init is not None else None}, "
            f"u_init={'set' if u_init is not None else None}, "
            f"v_init={'set' if v_init is not None else None}.")
    # Cast all closure-captured arrays to eta's dtype so the fori_loop
    # carry stays in a single precision throughout.  Without this,
    # H_bathy (float64 under x64) promotes H_total → H_u → U_bar_new
    # to float64 while the accumulators (U_sum, V_sum) remain float32,
    # causing a carry-type mismatch in jax.lax.fori_loop.
    _dt = eta_raw.dtype
    min_water_col = jnp.asarray(config.min_water_column_m, dtype=_dt)
    dt_s = jnp.asarray(dt_s, dtype=_dt)
    g = g.astype(_dt)
    H_bathy = H_bathy.astype(_dt)
    mask = mask.astype(_dt)
    u_mask = u_mask.astype(_dt)
    v_mask = v_mask.astype(_dt)
    u = u.astype(_dt)
    v = v.astype(_dt)
    u_corr = u_corr.astype(_dt)
    v_corr = v_corr.astype(_dt)
    eta_floor = min_water_col - H_bathy
    eta = jnp.maximum(eta_raw, eta_floor) * mask
    _area = grid.area.astype(_dt)

    if F_slow_eta is None:
        F_slow_eta = jnp.zeros_like(eta)
    else:
        F_slow_eta = F_slow_eta.astype(eta.dtype)
    if F_slow_u is None:
        F_slow_u = jnp.zeros((eta.shape[0], eta.shape[1] + 1), dtype=eta.dtype)
    else:
        F_slow_u = F_slow_u.astype(eta.dtype)
    if F_slow_v is None:
        F_slow_v = jnp.zeros((eta.shape[0] + 1, eta.shape[1]), dtype=eta.dtype)
    else:
        F_slow_v = F_slow_v.astype(eta.dtype)

    # --- Equilibrium-tide barotropic body force (OPT-IN; #tidal_forcing) -------
    # Add a = +g*grad(eta_eq_eff) to the SLOW forcing so it (a) is applied at
    # every substep as a constant-over-the-baroclinic-step body force (the tide
    # is slowly varying vs the ~s barotropic subcycle), and (b) is MASKED by
    # u_mask/v_mask together with F_slow inside the substep (lines below:
    # ``(... + F_slow_u) * u_mask``) — so closed/land faces receive nothing.
    # Feature-gated on the STATIC config bool (CLAUDE.md feature-gating exception)
    # AND a supplied traced model time: disabled / no-time => bit-identical.
    from legoesm.ocean.physics.tidal_forcing import apply_tidal_forcing
    F_slow_u, F_slow_v = apply_tidal_forcing(
        F_slow_u, F_slow_v, grid, t_seconds,
        getattr(config, "tidal_forcing", None), g=g)

    # Depth-averaged velocity.  Cast h_k to _dt because z_coord.sigma_w
    # may be float64 (jnp.linspace default under x64), which would
    # promote U_bar/V_bar and break the fori_loop carry-type invariant.
    # NEMO key_linssh: depth-average weights use the FIXED reference
    # thicknesses (dynspg_ts.F90:471 ``zhup2_e = hu_0``, r1_hu_0 convention).
    # For z-star this is a defensive no-op (compute_ocean_jacobian already
    # discards eta under linssh); it is load-bearing for the partial-cell
    # coordinate, whose compute_layer_thickness branch ignores the flag
    # (pre-existing inconsistency, vertical.py:685-692).
    _h_eta = (jnp.zeros_like(eta)
              if getattr(z_coord, 'linear_free_surface', False) else eta)
    h_k = compute_layer_thickness(
        _h_eta, H_bathy, z_coord, min_water_column_m=config.min_water_column_m,
    ).astype(_dt)
    U_bar, V_bar = _depth_average_to_faces(
        u, v, h_k, min_water_col, mask, u_mask, v_mask, grid,
    )
    # 3-D depth-mean REPLACEMENT reference (u' = u − ū_corr): the NOW-level
    # barotropic mean over the NOW eta.  With the MLF before-level seed the
    # integration's ū (from u/eta_init) is the BEFORE transport, but the 3-D
    # velocity being corrected is the NOW state, so its old depth-mean must use
    # the NOW velocity + NOW eta (matches the `_split` in `_leapfrog_step`).
    # No override ⇒ identical to (U_bar, V_bar) ⇒ byte-identical.
    if _seed_override:
        _eta_corr = jnp.maximum(state.eta.data.astype(_dt), eta_floor) * mask
        _h_eta_corr = (jnp.zeros_like(_eta_corr)
                       if getattr(z_coord, 'linear_free_surface', False)
                       else _eta_corr)
        _h_k_corr = compute_layer_thickness(
            _h_eta_corr, H_bathy, z_coord,
            min_water_column_m=config.min_water_column_m).astype(_dt)
        U_bar_corr, V_bar_corr = _depth_average_to_faces(
            u_corr, v_corr, _h_k_corr, min_water_col, mask, u_mask, v_mask, grid,
        )
    else:
        U_bar_corr, V_bar_corr = U_bar, V_bar

    # Semi-implicit Coriolis parameter at face points (#517: shared helper,
    # prefers stored grid.f_u/f_v, fold-safe).
    f_u, f_v = coriolis_at_faces(grid, eta.dtype)

    # In-substep barotropic Coriolis discretization (node 16).  "avg" (default)
    # = 4-pt average (legacy, bit-identical); "een" = NEMO enstrophy-conserving
    # EEN (kills the 2Δx checkerboard null mode / deep-eq jet).  Validated at
    # fn entry on the static config value (dispatch hardening).
    _bt_cor = getattr(config.barotropic, "barotropic_coriolis", "avg")
    if _bt_cor not in ("avg", "een", "een_metric"):
        raise ValueError(
            "unknown barotropic_coriolis scheme "
            f"{_bt_cor!r}: must be one of ('avg', 'een', 'een_metric').")
    _een_pre = None
    if _bt_cor in ("een", "een_metric") and add_barotropic_coriolis:
        # "een_metric" (node-16 finale) folds NEMO's e1v/r1_e1u (u) and e2u/
        # r1_e2v (v) horizontal metrics into the EEN coefficients — the factors
        # the per-unit-width "een" operator drops (dynspg_ts.F90:1349-1379).
        _een_pre = _build_een_barotropic_inputs(
            h_k, grid, mask, u_mask, v_mask, eta.dtype,
            metric_complete=(_bt_cor == "een_metric"))

    coeffs = _dissipation_coeffs(config, grid, _area, dt_s, eta.dtype, mask)

    # NEMO dyn_drg_init barotropic drag coefficient (#1226;
    # dynspg_ts.F90:1614-1618, bottom-only branch — DINO has ln_isfcav=F,
    # ln_drgice_imp=F):
    #   pCdU_u(ji,jj) = r1_2*( rCdU_bot(ji+1,jj) + rCdU_bot(ji,jj) )
    # Computed ONCE per baroclinic step (frozen across substeps, exactly like
    # NEMO's zCdU_u closure over the DO jn loop) from the NOW 3-D velocity
    # (zdfdrg's rCdU_bot level).  ``nemo_bottom_drag_rate_faces`` IS that
    # same 0.5-average-at-faces transcription in lego's positive-r
    # convention (``r_eff = -pCdU >= 0``) — shared helper, never re-derived.
    # Static config gate: flag off ⇒ None ⇒ the substep loop's drag branch
    # is not built ⇒ byte-identical.
    _drag_r_u = _drag_r_v = None
    if getattr(config, "barotropic_drag_substep", False):
        from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
            nemo_bottom_drag_rate_faces,
        )
        # NOW-level thickness for the rate (the MLF seed override re-seeds
        # only the fast integration; the drag coef stays at NOW like NEMO's
        # zdfdrg rCdU_bot).
        _hk_now = _h_k_corr if _seed_override else h_k
        _r_u_bt, _r_v_bt, _, _ = nemo_bottom_drag_rate_faces(
            u_corr, v_corr, _hk_now, z_coord, config, grid)
        _drag_r_u = _r_u_bt.astype(_dt)
        _drag_r_v = _r_v_bt.astype(_dt)

    w_filter, w_total, w_transport, n_loop = _compute_weights(
        config, n_substeps, eta.dtype, substep_scale=substep_scale)

    _filter = config.barotropic.barotropic_time_filter
    _boxcar_ab3 = _filter == "nemo_boxcar_ab3"   # NEMO nn_bt_flt=2 (DINO)
    _ab3 = _filter in ("nemo_ab3am4", "nemo_boxcar_ab3")
    if _boxcar_ab3:
        # NEMO nn_bt_flt=2: the barotropic sub-state is re-initialised EVERY
        # baroclinic step (ll_init=ll_bt_av=T, dynspg_ts.F90:202/469-476) ⇒ the
        # ll_init ramp fires every step and there is NO cross-window bt_hist
        # carry.  The ssh half-step-back interpolation uses the rn_bt_alpha=0
        # literals (flt2=True); the boxcar averaging (w_filter above) is applied
        # to the raw per-substep ssh, matching NEMO's pssh(Kaa) accumulation.
        _ab3_hist = None
        _ab3_za, _ab3_zb = nemo_ab3am4_coeff_arrays(
            n_loop, ramp=True, flt2=True)
        _ab3_za = _ab3_za.astype(eta.dtype)
        _ab3_zb = _ab3_zb.astype(eta.dtype)
    elif _ab3:
        # Cross-window AB3/AM4 substep histories (NEMO nn_bt_flt=3 restart state
        # ubb_e/ub_e/vbb_e/vb_e/sshbb_e/sshb_e): None on the first step
        # ⇒ NEMO cold-start ll_init ramp; carried tuple afterwards ⇒ full
        # AB3/AM4 rows continuing the substep series across the window
        # boundary (dynspg_ts.F90:200-226, 806-808). Static Python gate on
        # pytree structure — same pattern as the prognostic state.tke seed.
        _ab3_hist = getattr(state, "bt_hist", None)
        _ab3_za, _ab3_zb = nemo_ab3am4_coeff_arrays(
            n_loop, ramp=_ab3_hist is None)
        # cast to the state dtype: f64 coefficients would silently promote the
        # f32 carry and break the scan carry-type invariant.
        _ab3_za = _ab3_za.astype(eta.dtype)
        _ab3_zb = _ab3_zb.astype(eta.dtype)
    else:
        _ab3_za = _ab3_zb = _ab3_hist = None
    _finals = _run_substep_loop(
        eta, U_bar, V_bar,
        dt_s=dt_s, n_loop=n_loop, w_filter=w_filter, w_transport=w_transport,
        grid=grid, config=config, g=g, H_bathy=H_bathy, mask=mask,
        u_mask=u_mask, v_mask=v_mask, min_water_col=min_water_col,
        eta_floor=eta_floor, area=_area,
        F_slow_eta=F_slow_eta, F_slow_u=F_slow_u, F_slow_v=F_slow_v,
        f_u=f_u, f_v=f_v, add_barotropic_coriolis=add_barotropic_coriolis,
        coeffs=coeffs,
        local_subcycle_clamp=config.barotropic.barotropic_local_subcycle_clamp,
        linear_free_surface=getattr(z_coord, 'linear_free_surface', False),
        ab3_za=_ab3_za, ab3_zb=_ab3_zb, ab3_hist=_ab3_hist,
        een_pre=_een_pre,
        drag_r_u=_drag_r_u, drag_r_v=_drag_r_v,
    )
    (eta_f, U_bar_f, V_bar_f,
     Hu_sum_f, Hv_sum_f, eta_sum_f, U_sum_f, V_sum_f) = _finals[:8]

    # Time-averaged barotropic transport: w_transport already carries the full
    # continuity-consistent normalisation — the SM2005 tail-sum
    # ``tail_j/(n·w_total)`` (box/cosine) or the SM2005 secondary weights
    # (power_law) — so the accumulator IS the time-averaged transport ``Hu_avg``
    # that closes ``div(Hu_avg) == (eta_old - eta_avg)/dt``.
    Hu_avg = Hu_sum_f
    Hv_avg = Hv_sum_f

    if _ab3 and not _boxcar_ab3:
        # NEMO nn_bt_flt=3: the new state is the FINAL substep value (no time
        # averaging); Hu_avg above is the plain substep mean (uniform
        # w_transport), continuity-consistent with eta_f by telescoping.
        eta_avg = eta_f
        U_bar_avg = U_bar_f
        V_bar_avg = V_bar_f
    else:
        # Time-averaged eta and velocity (cosine / box / nemo_boxcar_centred /
        # nemo_boxcar_ab3 = nn_bt_flt=2 boxcar averaging of the raw substep ssh).
        eta_avg = eta_sum_f / w_total
        U_bar_avg = U_sum_f / w_total
        V_bar_avg = V_sum_f / w_total

    # SOTA-local split-explicit: the per-substep clamp was LOCAL (no allreduce);
    # restore GLOBAL mass conservation with ONE redistribute call on the
    # time-averaged eta (the returned SSH).  No-op (bit-identical to the
    # per-substep path) when no cell hit eta_floor; this single call's 3 batched
    # allreduces replace the subcycle's ~3*n_substeps (the outer-step
    # fix_eta_drift fixer is separate, unaffected).
    if config.barotropic.barotropic_local_subcycle_clamp:
        eta_avg = _clamp_redistribute(eta_avg, eta_floor, mask, _area)

    # Correct 3D velocities: preserve baroclinic structure.
    # Use time-averaged barotropic velocity for the 3D correction to ensure
    # consistency with eta_avg (the time-averaged eta used for layer thicknesses).
    u_baro_old = U_bar_corr[..., jnp.newaxis]
    v_baro_old = V_bar_corr[..., jnp.newaxis]
    u_prime = u_corr - u_baro_old
    v_prime = v_corr - v_baro_old
    u_new = (u_prime + U_bar_avg[..., jnp.newaxis]) * u_mask[..., jnp.newaxis]
    v_new = (v_prime + V_bar_avg[..., jnp.newaxis]) * v_mask[..., jnp.newaxis]

    state_new = state._replace(
        eta=state.eta.replace(data=eta_avg),
        u=state.u.replace(data=u_new),
        v=state.v.replace(data=v_new),
    )
    if _ab3 and not _boxcar_ab3 and hasattr(state, "bt_hist"):
        # NEMO nn_bt_flt=3 only (nn_bt_flt=2 re-inits the sub-state each step ⇒
        # no cross-window carry).
        # store the end-of-window (b, bb) histories in DEVIATION form
        # (X_final - X_b, X_final - X_bb) for the next window (finals
        # positions 8-13: Ub, Ubb, Vb, Vbb, etab, etabb after the final
        # dynspg_ts:806-808 rotation; finals 0-2: eta_f, U_bar_f, V_bar_f).
        # See the reconstruction comment in _run_substep_loop.
        state_new = state_new._replace(bt_hist=(
            _finals[1] - _finals[8], _finals[1] - _finals[9],
            _finals[2] - _finals[10], _finals[2] - _finals[11],
            _finals[0] - _finals[12], _finals[0] - _finals[13],
        ))
    return state_new, (Hu_avg, Hv_avg)


# ============================================================================
# Wide-halo split-explicit barotropic (opt-in; scaling-audit item 3)
# ============================================================================

def estimate_barotropic_halo_messages(
    config: LatLonCGridOceanConfig, n_substeps: int,
) -> dict:
    """Analytic per-baroclinic-step lat-halo message counts, standard vs wide.

    The STANDARD substep loop dispatches, per substep: ``pad_ns_zero`` on the
    column thickness, ``gradient_y_cgrid`` on the PGF eta,
    ``interp_u_to_vface_4pt`` on the fresh U (3 pads), +1 pad when the eta
    diffusion is on, +2 when divergence damping is on (its grad_y +
    the diffusion-coeff pads are precomputed once, not counted).  The WIDE
    path replaces them with 2 fused exchanges (cell + v-face groups) for the
    geometry, 2 for the static fields, and 2 per additional chunk.

    Used by the scaling benches to record the communication saving next to
    the timing row (halo-count metric of the wide-halo audit item); counts
    are per interior rank, exchanges not sendrecv pairs.
    """
    pads_per_substep = 3
    if config.barotropic.barotropic_diffusion_alpha > 0.0:
        pads_per_substep += 1
    if config.barotropic.barotropic_div_damp > 0.0:
        pads_per_substep += 2
    _, _, _, n_loop = _compute_weights(config, n_substeps, jnp.float32)
    reach = _substep_stencil_reach(config)
    chunk = int(config.barotropic.barotropic_wide_halo_chunk)
    return {
        "standard_messages": pads_per_substep * int(n_loop),
        "wide_messages_fixed": 4,     # geometry (2) + static fields (2)
        "wide_messages_per_chunk": 2,  # carry (cell + v-face groups)
        "n_loop": int(n_loop),
        "stencil_reach": reach,
        "chunk_config": chunk,
    }


def barotropic_substeps_wide_halo_latlon_cgrid(
    state: LatLonCGridOceanState,
    dt_s: float,
    n_substeps: int,
    grid,
    z_coord: OceanZStarCoordinate,
    config: LatLonCGridOceanConfig,
    F_slow_eta=None,
    F_slow_u=None,
    F_slow_v=None,
    add_barotropic_coriolis: bool = True,
    t_seconds=None,
) -> LatLonCGridOceanState:
    """Wide-halo twin of :func:`barotropic_substeps_latlon_cgrid`.

    Exchange ONE lat halo of width ``W = k x r`` (``k`` substeps per
    exchange, ``r`` = :func:`_substep_stencil_reach`), then run ``k``
    substeps on the extended band with every operator-internal pad forced
    LOCAL (:func:`legoesm.grids.halo.local_halo_pads`): halo garbage creeps
    inward at most ``r`` rows per substep, so the owned rows stay exact.
    Repeat per chunk; crop to the owned band at the end.

    Contract / scope (v1):

    * regular lat-lon band decompositions only — REFUSES an active tripolar
      fold (the fold row needs a permuted sign-flipped wide exchange) and
      assumes wall-BC poles;
    * per-substep eta clamping runs in the LOCAL (allreduce-free) mode
      regardless of ``barotropic_local_subcycle_clamp`` — a per-substep
      global redistribute over the EXTENDED band would double-count the
      halo overlap.  Global volume conservation is restored by the single
      post-loop redistribute on the CROPPED owned rows (always applied on
      this path), exactly the SOTA-local scheme;
    * uneven band layouts: the auto chunk assumes the neighbour band is at
      least as tall as the halo; with strongly uneven bands (``--wet-balance``)
      set ``barotropic_wide_halo_chunk`` so ``chunk x reach <= min band
      height`` across ranks.

    Numerics: identical update operators via the SAME
    :func:`_run_substep_loop`; serial results match the standard local-clamp
    path to re-association tolerance (parity-gated in
    ``tests/ocean/unit/test_barotropic_wide_halo.py``).
    """
    from legoesm.grids.halo import local_halo_pads
    from legoesm.grids.halo_latlon import (
        widen_band_cell_fields,
        widen_band_vface_fields,
        widen_cgrid_geometry_band,
    )

    if not (hasattr(grid, "dx_u") and hasattr(grid, "fold")):
        raise TypeError(
            "barotropic_substeps_wide_halo_latlon_cgrid requires a "
            "LatLonCGridGeometry (the model's ensure_geometry output); got "
            f"{type(grid).__name__}."
        )
    if getattr(config.barotropic, "barotropic_coriolis", "avg") in (
            "een", "een_metric"):
        raise NotImplementedError(
            "barotropic_coriolis='een'/'een_metric' is not wired into the "
            "wide-halo barotropic path (the EEN precompute would need the "
            "extended band); use the standard split-explicit path or 'avg'.")
    if getattr(config, "barotropic_drag_substep", False):
        raise NotImplementedError(
            "barotropic_drag_substep=True is not wired into the wide-halo "
            "barotropic path (the drag-rate faces would need widening to the "
            "extended band); use the standard split-explicit path or disable "
            "barotropic_wide_halo.")

    g = jnp.asarray(config.g)
    H_bathy = state.H_bathy.data
    mask = state.land_mask.data
    u_mask = state.u_mask.data
    v_mask = state.v_mask.data
    u = state.u.data
    v = state.v.data
    eta_raw = state.eta.data
    _dt = eta_raw.dtype
    min_water_col = jnp.asarray(config.min_water_column_m, dtype=_dt)
    dt_s = jnp.asarray(dt_s, dtype=_dt)
    g = g.astype(_dt)
    H_bathy = H_bathy.astype(_dt)
    mask = mask.astype(_dt)
    u_mask = u_mask.astype(_dt)
    v_mask = v_mask.astype(_dt)
    u = u.astype(_dt)
    v = v.astype(_dt)
    eta_floor = min_water_col - H_bathy
    eta = jnp.maximum(eta_raw, eta_floor) * mask

    if F_slow_eta is None:
        F_slow_eta = jnp.zeros_like(eta)
    else:
        F_slow_eta = F_slow_eta.astype(eta.dtype)
    if F_slow_u is None:
        F_slow_u = jnp.zeros((eta.shape[0], eta.shape[1] + 1), dtype=eta.dtype)
    else:
        F_slow_u = F_slow_u.astype(eta.dtype)
    if F_slow_v is None:
        F_slow_v = jnp.zeros((eta.shape[0] + 1, eta.shape[1]), dtype=eta.dtype)
    else:
        F_slow_v = F_slow_v.astype(eta.dtype)

    # Depth average LOCALLY (column-local op): the wide exchange then only
    # moves 2-D fields — never the 3-D (nlev) velocity blocks.
    h_k = compute_layer_thickness(
        eta, H_bathy, z_coord, min_water_column_m=config.min_water_column_m,
    ).astype(_dt)
    U_bar, V_bar = _depth_average_to_faces(
        u, v, h_k, min_water_col, mask, u_mask, v_mask, grid,
    )

    w_filter, w_total, w_transport, n_loop = _compute_weights(
        config, n_substeps, eta.dtype)

    # --- Static chunk / width budget --------------------------------------
    # The v-face widening exchanges ``halo = W + 1`` cell rows (the stagger
    # trick), so the halo must satisfy W + 1 <= band height — the exchange
    # pulls rows from ONE neighbour only.
    #
    # RANK-CONSISTENCY (codex: uneven-band deadlock): chunk/W/n_chunks gate
    # COLLECTIVE calls, so they must be identical on every rank.  Under the
    # MPI band layout they derive from the GLOBAL (n_lat_global, n_ranks) —
    # never from the local band height, which differs across ranks when
    # n_lat % n_ranks != 0.  The local band is only ASSERTED to be within
    # the even-split envelope {base, base+1}: every such band is >= base
    # rows, so a base-derived W fits every neighbour; a custom (wet-balance)
    # band outside the envelope aborts loudly BEFORE any exchange (mpirun
    # kills the world on the nonzero exit).  The 2-D pencil is refused:
    # local_halo_pads would also localize its ZONAL exchanges, silently
    # wrapping E/W inside the lon block.
    reach = _substep_stencil_reach(config)
    nl = int(eta.shape[0])
    from legoesm.grids.halo import get_halo_backend, get_mpi_topology
    band_height_budget = nl
    if get_halo_backend() == "mpi":
        from legoesm.parallel.latlon_mpi import (
            LatLon2DLayout,
            LatLonBandLayout,
        )
        topology = get_mpi_topology()
        if isinstance(topology, LatLon2DLayout):
            raise ValueError(
                "wide-halo barotropic: the 2-D lat-lon pencil layout is not "
                "supported (forcing pads local would also localize the "
                "ZONAL halo exchanges); use the 1-D band layout or disable "
                "barotropic_wide_halo."
            )
        if isinstance(topology, LatLonBandLayout):
            base = int(topology.n_lat_global) // int(topology.n_ranks)
            if nl not in (base, base + 1):
                raise ValueError(
                    f"wide-halo barotropic: local band height {nl} is "
                    f"outside the even-split envelope {{{base}, {base + 1}}} "
                    f"(n_lat_global={topology.n_lat_global}, n_ranks="
                    f"{topology.n_ranks}) — a custom/wet-balanced layout. "
                    f"The wide-halo chunk budget must be rank-consistent, "
                    f"which this path guarantees only for even-split bands; "
                    f"disable barotropic_wide_halo on this layout."
                )
            band_height_budget = base
    w_max = band_height_budget - 1
    chunk_cfg = int(config.barotropic.barotropic_wide_halo_chunk)
    if chunk_cfg > 0:
        chunk = min(chunk_cfg, int(n_loop))
    else:
        chunk = min(int(n_loop), max(w_max // reach, 1))
    W = chunk * reach
    if W > w_max:
        raise ValueError(
            f"wide-halo barotropic: halo width {W} (= chunk {chunk} x reach "
            f"{reach}) exceeds the band-height budget {w_max} (= min band "
            f"height {band_height_budget} - 1, the one-neighbour exchange "
            f"limit); lower barotropic_wide_halo_chunk (or use more "
            f"substeps per exchange only on taller bands)."
        )
    n_chunks = -(-int(n_loop) // chunk)  # ceil

    # --- Extended geometry + static fields (per-step, fused) ---------------
    grid_ext = widen_cgrid_geometry_band(grid, W)
    (H_ext, mask_ext, umask_ext, Fse_ext, Fsu_ext) = widen_band_cell_fields(
        (H_bathy, mask, u_mask, F_slow_eta, F_slow_u), W)
    (vmask_ext, Fsv_ext) = widen_band_vface_fields((v_mask, F_slow_v), W)

    # Tide on the EXTENDED geometry (analytic in the grid — no exchange
    # needed; owned rows are bit-identical to the standard path's values).
    from legoesm.ocean.physics.tidal_forcing import apply_tidal_forcing
    # Wide-halo path: the equilibrium tide is evaluated on the EXTENDED band
    # geometry, so the acceleration call must run under local_halo_pads(). The
    # single-owner wrapper does the enabled/t_seconds gate, masking contract and
    # the carry-invariant dtype cast; only the halo-pad context is band-specific.
    with local_halo_pads():
        Fsu_ext, Fsv_ext = apply_tidal_forcing(
            Fsu_ext, Fsv_ext, grid_ext, t_seconds,
            getattr(config, "tidal_forcing", None), g=g)

    eta_floor_ext = min_water_col - H_ext
    area_ext = grid_ext.area.astype(_dt)
    with local_halo_pads():
        f_u_ext, f_v_ext = coriolis_at_faces(grid_ext, eta.dtype)
        coeffs_ext = _dissipation_coeffs(
            config, grid_ext, area_ext, dt_s, eta.dtype, mask_ext)

    # --- Chunked substep loop ----------------------------------------------
    eta_c, U_c, V_c = eta, U_bar, V_bar
    sums = None
    done = 0
    for _ in range(n_chunks):
        k = min(chunk, int(n_loop) - done)
        # Wide exchange of the carry (2 fused messages) with the REAL backend.
        (eta_x, U_x) = widen_band_cell_fields((eta_c, U_c), W)
        (V_x,) = widen_band_vface_fields((V_c,), W)
        wf = jax.lax.slice_in_dim(w_filter, done, done + k)
        wt = jax.lax.slice_in_dim(w_transport, done, done + k)
        with local_halo_pads():
            finals = _run_substep_loop(
                eta_x, U_x, V_x,
                dt_s=dt_s, n_loop=k, w_filter=wf, w_transport=wt,
                grid=grid_ext, config=config, g=g, H_bathy=H_ext,
                mask=mask_ext, u_mask=umask_ext, v_mask=vmask_ext,
                min_water_col=min_water_col, eta_floor=eta_floor_ext,
                area=area_ext,
                F_slow_eta=Fse_ext, F_slow_u=Fsu_ext, F_slow_v=Fsv_ext,
                f_u=f_u_ext, f_v=f_v_ext,
                add_barotropic_coriolis=add_barotropic_coriolis,
                coeffs=coeffs_ext,
                # NEVER a per-substep global redistribute on the extended
                # band (halo overlap would double-count in the allreduce).
                local_subcycle_clamp=True,
                linear_free_surface=getattr(
                    z_coord, "linear_free_surface", False),
            )
        (eta_ext_f, U_ext_f, V_ext_f,
         Hu_k, Hv_k, eta_sum_k, U_sum_k, V_sum_k) = finals
        # Crop the carry back to the owned band for the next exchange.
        eta_c = eta_ext_f[W:W + nl]
        U_c = U_ext_f[W:W + nl]
        V_c = V_ext_f[W:W + nl + 1]
        # Accumulate the (pointwise) sums on OWNED rows across chunks.
        k_sums = (Hu_k[W:W + nl], Hv_k[W:W + nl + 1],
                  eta_sum_k[W:W + nl], U_sum_k[W:W + nl],
                  V_sum_k[W:W + nl + 1])
        sums = k_sums if sums is None else tuple(
            a + b for a, b in zip(sums, k_sums))
        done += k

    Hu_avg, Hv_avg, eta_sum_f, U_sum_f, V_sum_f = sums

    eta_avg = eta_sum_f / w_total
    U_bar_avg = U_sum_f / w_total
    V_bar_avg = V_sum_f / w_total

    # Global mass conservation: ONE redistribute on the OWNED rows with the
    # REAL backend (its global sums are MPI/SPMD-aware) — the wide path's
    # per-substep clamp was local by construction.
    _area = grid.area.astype(_dt)
    eta_avg = _clamp_redistribute(eta_avg, eta_floor, mask, _area)

    # Correct 3D velocities: preserve baroclinic structure (owned rows only).
    u_baro_old = U_bar[..., jnp.newaxis]
    v_baro_old = V_bar[..., jnp.newaxis]
    u_prime = u - u_baro_old
    v_prime = v - v_baro_old
    u_new = (u_prime + U_bar_avg[..., jnp.newaxis]) * u_mask[..., jnp.newaxis]
    v_new = (v_prime + V_bar_avg[..., jnp.newaxis]) * v_mask[..., jnp.newaxis]

    state_new = state._replace(
        eta=state.eta.replace(data=eta_avg),
        u=state.u.replace(data=u_new),
        v=state.v.replace(data=v_new),
    )
    return state_new, (Hu_avg, Hv_avg)
