"""Barotropic solver for the lat-lon C-grid FV ocean model.

Forward-backward substeps for 2D free-surface gravity waves on the
Arakawa C-grid:

    d(eta)/dt = -div(H_total * U_bar, H_total * V_bar)   [cell centers]
    d(U_bar)/dt = f * V_bar_at_u - g * d(eta)/dx          [u-points]
    d(V_bar)/dt = -f * U_bar_at_v - g * d(eta)/dy         [v-points]

The C-grid layout uses compact (single-cell) gradient and divergence
stencils, eliminating the 2*dx checkerboard null space of the A-grid.

Parallels barotropic_latlon.py (A-grid) but with staggered variables.

Reference for the generic (default) arm
----------------------------------------
Every ``barotropic_*_evaluation`` selector in this module (continuity,
transport accumulation, PGF, seed, EEN-Coriolis) defaults to ``"generic"``;
the alternative ``"nemo_literal"``/``"nemo_ssh_avg"`` values are a literal
transcription of NEMO's ``dyn_spg_ts`` (see the ``nemo_literal_*`` helpers
below and ``docs/ocean/fidelity/nemo_branch_isomorphism_map.md`` S-16). The
``"generic"`` arm is **not** a NEMO port and is not a transcription of any
single paper's equations either: it is legoESM's own in-house
forward-backward split-explicit C-grid solver, introduced with no external
citation by commit ``adbb49f83`` (2026-04-08, "Add C-grid lat-lon ocean
model to fix checkerboard instability", #87). Its later closure choices were
explicitly modeled on the standard MOM6/ROMS split-explicit family, per
those commits' own messages: the substep time-averaging "follow[s] the
standard approach in MOM6, MPAS-Ocean, and ROMS (Hallberg 1997, Shchepetkin
& McWilliams 2005)" (commit ``06f4de939``, 2026-04-11, "Add barotropic
time-averaging for split-explicit stability") and the BEBT semi-implicit
pressure-gradient blend "match[es] MOM6 default" ``bebt=0.2`` (commit
``3d0170d04``, 2026-04-20, #205). USER DECISION 2026-09-02: this arm is kept
as a legitimate non-NEMO fork — it is the dycore of 19 idealized ocean
test-matrix experiments via the ``default_wright_v1``/``legoesm_linear_v1``
catalog recipes, plus ``legoesm_nemo_like_v1`` (renamed from ``nemo_v1``,
which falsely implied this arm was NEMO's ``dyn_spg_ts``; see
``legoesm.ocean.recipes``) — rather than collapsed onto the NEMO-literal arm.

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

from legoesm.core.source_rounding import nemo_source_round
from legoesm.grids.latlon import LatLonGrid, ensure_geometry
from legoesm.ocean.vertical import (
    OceanPartialCellCoordinate,
    OceanZStarCoordinate,
    compute_layer_thickness,
    nemo_qco_card_mesh_operands,
    nemo_qco_live_face_geometry_from_operands,
)
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
    vface_zonal_cos_lat,
    vertex_coriolis,
)
from legoesm.grids.halo_latlon import zero_polar_lat_ends as _zero_polar_lat_ends
from legoesm.ocean.dynamics.eta_floor import clamp_and_redistribute as _clamp_redistribute
from legoesm.ocean.dynamics.barotropic_common import (
    bebt_blend,
    compute_filter_weights,
    compute_nemo_boxcar_centred_weights,
    compute_nemo_boxcar_forward_weights,
    compute_nemo_forward_raw_transport_weights,
    compute_nemo_boxcar_raw_transport_weights,
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


def _nemo_literal_seed_depth_mean(field, h_face, face_mask, r1_live):
    """NEMO ``istate.F90:149-155`` source-ordered barotropic seed mean.

    Keep the vertical recurrence explicit: a stacked ``jnp.sum`` is permitted
    to use a tree reduction, whereas the active NEMO source left-accumulates
    one level at a time before multiplying by the separately constructed live
    reciprocal depth.  The Python loop is static under JIT and differentiable.
    """
    acc = jnp.zeros_like(field[..., 0])
    for jk in range(field.shape[-1]):
        acc = acc + h_face[..., jk] * field[..., jk]
    return (acc * r1_live) * face_mask


def _nemo_literal_seed_from_reference_mesh(
    u_3d, v_3d, h_k, eta_dyn, u_mask, v_mask, z_coord,
):
    """Evaluate the DINO QCO seed from the carried NEMO reference mesh.

    Returns ``None`` when the coordinate was not constructed by the NEMO
    bridge.  The generated-grid fallback in :func:`_depth_average_to_faces`
    then retains the same source-ordered recurrence using its model-native
    face thicknesses.

    DINO's executed ``ln_zco`` branch has ``e3u_0 == e3v_0 == e3t_0`` at the
    corresponding NEMO point.  The bridge carries that exact, unaveraged
    ``e3t_0`` plus ``hu_0/hv_0`` and the three native areas.  Use them without
    reconstructing either the reference ladder or QCO ``r3`` through the
    generic z-star/min-face path; that reconstruction is mathematically
    equivalent but changed the last bits measured in round 19.
    """
    if z_coord is None:
        return None
    names = ("nemo_e3t_0", "nemo_hu_0", "nemo_hv_0", "nemo_e1e2t",
             "nemo_e1e2u", "nemo_e1e2v")
    refs = tuple(getattr(z_coord, name, None) for name in names)
    if any(value is None for value in refs):
        return None
    e3t_ref, hu0, hv0, area_t, area_u, area_v = (
        jnp.asarray(value, dtype=eta_dyn.dtype) for value in refs)
    nlev = u_3d.shape[-1]
    e3t_ref = e3t_ref[..., :nlev]
    expected_t_shape = eta_dyn.shape
    if (e3t_ref.shape[:2] != expected_t_shape
            or hu0.shape != expected_t_shape
            or hv0.shape != expected_t_shape
            or area_t.shape != expected_t_shape
            or area_u.shape != expected_t_shape
            or area_v.shape != expected_t_shape):
        raise ValueError(
            "NEMO reference seed operands must share eta's T-grid shape; got "
            f"eta={expected_t_shape}, e3t={e3t_ref.shape}, hu0={hu0.shape}, "
            f"hv0={hv0.shape}, area_t={area_t.shape}, area_u={area_u.shape}, "
            f"area_v={area_v.shape}.")

    # Work first in NEMO's native east-/north-face arrays (same 2-D extent as
    # T points), then map once to legoESM's redundant west/south face layout.
    # This preserves domqco.F90:166-169 operand order exactly.
    wet_t = h_k[..., :nlev] > 0.0
    wet_u = (wet_t & jnp.roll(wet_t, -1, axis=1)
             & (u_mask[:, 1:, None] > 0.5))
    wet_t_north = jnp.concatenate(
        [wet_t[1:], jnp.zeros_like(wet_t[:1])], axis=0)
    wet_v = (wet_t & wet_t_north & (v_mask[1:, :, None] > 0.5))
    wet2_u = wet_u[..., 0]
    wet2_v = wet_v[..., 0]
    hu_safe = jnp.where(wet2_u, hu0, 1.0)
    hv_safe = jnp.where(wet2_v, hv0, 1.0)
    area_u_safe = jnp.where(wet2_u, area_u, 1.0)
    area_v_safe = jnp.where(wet2_v, area_v, 1.0)
    area_eta = area_t * eta_dyn
    area_eta_east = jnp.roll(area_eta, -1, axis=1)
    area_eta_north = jnp.concatenate(
        [area_eta[1:], jnp.zeros_like(area_eta[:1])], axis=0)
    r3u = (0.5 * (area_eta + area_eta_east) / hu_safe / area_u_safe)
    r3v = (0.5 * (area_eta + area_eta_north) / hv_safe / area_v_safe)
    mask_u3 = wet_u.astype(eta_dyn.dtype)
    mask_v3 = wet_v.astype(eta_dyn.dtype)
    live_u = (e3t_ref * (1.0 + r3u[..., None] * mask_u3)) * mask_u3
    live_v = (e3t_ref * (1.0 + r3v[..., None] * mask_v3)) * mask_v3
    r1u = ((1.0 / hu_safe) / (1.0 + r3u)) * wet2_u
    r1v = ((1.0 / hv_safe) / (1.0 + r3v)) * wet2_v
    un_native = _nemo_literal_seed_depth_mean(
        u_3d[:, 1:, :], live_u, wet2_u, r1u)
    vn_native = _nemo_literal_seed_depth_mean(
        v_3d[1:, :, :], live_v, wet2_v, r1v)
    un = jnp.concatenate([un_native[:, -1:], un_native], axis=1)
    vn = jnp.concatenate([jnp.zeros_like(vn_native[:1]), vn_native], axis=0)
    return un, vn


def _nemo_literal_barotropic_pressure_gradient(eta_pgf, grid, g, u_mask, v_mask):
    """NEMO ``dynspg_ts.F90:776-780`` surface PGF and native face metrics."""
    geom = ensure_geometry(grid)
    east_delta = jnp.roll(eta_pgf, -1, axis=1) - eta_pgf
    pgf_u_native = ((-g * east_delta) * (1.0 / geom.dx_u[:, 1:]))
    pgf_u = jnp.concatenate([pgf_u_native[:, -1:], pgf_u_native], axis=1)
    north_delta = eta_pgf[1:] - eta_pgf[:-1]
    pgf_v = jnp.concatenate([
        jnp.zeros_like(eta_pgf[:1]),
        ((-g * north_delta) * (1.0 / geom.dy_v[1:-1])),
        jnp.zeros_like(eta_pgf[:1]),
    ], axis=0)
    return pgf_u * u_mask, pgf_v * v_mask


def _nemo_literal_seed_from_card_mesh(
    u_3d, v_3d, eta_dyn, H_bathy, min_water_col, u_mask, v_mask, grid, z_coord,
):
    """NEMO ``puu_b(Kmm)`` loop-entry seed for a card WITHOUT NEMO's mesh.

    ``dynspg_ts.F90:487`` seeds the external loop with ``puu_b(Kmm)``, which
    ``stprk3_stg.F90:439-446`` has imposed on ``uu`` at every stage with the
    ``e3u_0/hu_0`` weights; the live face thickness is
    ``e3u(Kmm) = e3u_0*(1 + r3u(Kmm))`` (``domzgr_substitute.h90``) with
    ``r3u`` the ``e1e2t``-weighted ssh mean over ``hu_0``
    (``domqco.F90:219-222``).  Those operands come from the ONE shared kernel
    every NEMO consumer in legoESM already uses (``vertical.py``:
    ``nemo_qco_card_mesh_operands`` -> ``e3u_0`` = min-rule of the REFERENCE
    ladder, ``hu_0 = SUM(e3u_0*umask)`` (``domain.F90:145``);
    ``nemo_qco_live_face_geometry_from_operands`` -> ``e3u``, ``r1_hu``), and
    the sum is the same source-ordered ``istate.F90:149-155`` recurrence the
    reference-mesh seed uses.

    NOT the per-level min of the two STRETCHED T-cell thicknesses rescaled by
    a column-depth ratio (the previous card path): on a face whose two
    columns have different reference depths the per-level min follows the
    less-stretched column while the column min follows the shallower one, so
    the weights summed to ``hu_0*(1+r3t_e)*(1+r3u)/(1+r3t_w)`` and a uniform
    velocity came back scaled by ``(1+r3t_e)/(1+r3t_w)`` -- measured
    ``1.05e-9 -> 3.1e-7 -> 4.3e-6 m/s`` at the OVERFLOW-zps shelf break at
    kt=2..4, the re-injected owner of the kt>=3 SSH walk (receipt
    ``nemo_testcases_l1_ssh_walk_preregister.md``).
    """
    from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks_3d

    if z_coord is None:
        raise ValueError(
            "barotropic_seed_evaluation='nemo_literal' needs the card's "
            "z-coordinate: NEMO's e3u_0 is the min-rule of the reference "
            "ladder (domain.F90:145), which cannot be recovered from a live "
            "thickness alone; pass z_coord (both production call sites do).")
    dtype = jnp.asarray(eta_dyn).dtype
    eta_dyn = jnp.asarray(eta_dyn, dtype=dtype)
    # e3t_0: the reference (eta = 0) ladder, exactly as the WS-RK3 stage
    # transport builds it (ocean_model_latlon_cgrid ``_h_ref_ws``).
    h_ref = compute_layer_thickness(
        jnp.zeros_like(eta_dyn), H_bathy, z_coord,
        min_water_column_m=min_water_col).astype(dtype)
    if isinstance(z_coord, OceanPartialCellCoordinate):
        u_mask3, v_mask3 = compute_face_masks_3d(z_coord.is_active, grid)
        u_mask3 = u_mask3.astype(dtype)
        v_mask3 = v_mask3.astype(dtype)
    else:
        u_mask3 = jnp.asarray(u_mask, dtype=dtype)[..., None]
        v_mask3 = jnp.asarray(v_mask, dtype=dtype)[..., None]
    ops = nemo_qco_card_mesh_operands(h_ref, u_mask3, v_mask3, grid, dtype)
    geom = nemo_qco_live_face_geometry_from_operands(
        eta_dyn, ops.e3u_0, ops.e3v_0, ops.umask3, ops.vmask3,
        ops.hu_0, ops.hv_0, ops.area_t, ops.area_u, ops.area_v,
    )
    wet2_u = (ops.hu_0 > 0.0).astype(dtype)
    wet2_v = (ops.hv_0 > 0.0).astype(dtype)
    un_native = _nemo_literal_seed_depth_mean(
        jnp.asarray(u_3d, dtype=dtype)[:, 1:, :], geom.e3u * ops.umask3,
        wet2_u, geom.r1_hu)
    vn_native = _nemo_literal_seed_depth_mean(
        jnp.asarray(v_3d, dtype=dtype)[1:, :, :], geom.e3v * ops.vmask3,
        wet2_v, geom.r1_hv)
    un = jnp.concatenate([un_native[:, -1:], un_native], axis=1)
    vn = jnp.concatenate([jnp.zeros_like(vn_native[:1]), vn_native], axis=0)
    return un, vn


def _depth_average_to_faces(
    u_3d: jnp.ndarray,
    v_3d: jnp.ndarray,
    h_k: jnp.ndarray,
    min_water_col: jnp.ndarray,
    mask: jnp.ndarray,
    u_mask: jnp.ndarray,
    v_mask: jnp.ndarray,
    grid=None,
    *,
    seed_face_depth: str = "min_rule",
    seed_evaluation: str = "generic",
    eta_dyn: jnp.ndarray | None = None,
    H_bathy: jnp.ndarray | None = None,
    area: jnp.ndarray | None = None,
    z_coord=None,
    legacy_seed_min_rule_faces: bool = False,
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
    seed_face_depth : ``BarotropicConfig.barotropic_seed_face_depth`` value
        (#1226 round 2 item 1).  ``"min_rule"`` (DEFAULT, bit-identical
        legacy): the 3-D min-rule face thickness (unchanged below).
        ``"nemo_ssh_avg"``: rescale that min-rule 3-D face thickness by the
        ratio of the NEMO ssh-averaged TOTAL column face depth
        (:func:`nemo_ssh_avg_face_depth`, reused verbatim — see its
        docstring for the NEMO ``dynspg_ts.F90`` ``un_e = puu_b(Kbb)`` /
        ``puu_b`` finalization this matches) to the min-rule TOTAL column
        face depth, both at ``eta_dyn``.  Exact for z-star (the ratio is a
        single per-column scalar, identical at every level, so it commutes
        with the vertical sum used by ``depth_average_to_faces`` below);
        for partial cells this is the same scalar-column-factor
        approximation the rest of the split-explicit solver already makes.
        NB the per-column ratio cancels in the weighted MEAN (measured
        2026-07-27, DINO Y5 twin: bit-identical seeds) ONLY away from the
        ``min_water_column_m`` floor — the option is INERT there, but NOT
        inert where ``max(sum_k h_face, min_water_column_m)`` binds
        asymmetrically between the two face-depth rules (thin/shelf
        columns): an 11% loop-entry velocity difference was reproduced at
        the production default ``min_water_column_m=0.5`` (see the
        ``BarotropicConfig.barotropic_seed_face_depth`` docstring and
        ``TestBarotropicSeedFaceDepth::
        test_shelf_column_floor_breaks_inertness_at_production_default``).
        It selects the face-depth bookkeeping convention, not the
        velocity, EXCEPT at the floor. Requires ``eta_dyn``/``H_bathy``/
        ``area`` when selected. Unknown value raises (dispatch hardening).
    seed_evaluation : ``BarotropicConfig.barotropic_seed_evaluation`` value.
        ``"generic"`` preserves the legacy shared stacked reduction.
        ``"nemo_literal"`` requires ``seed_face_depth="nemo_ssh_avg"`` and
        evaluates NEMO's live-thickness, source-ordered recurrence followed by
        its separately associated reciprocal depth (``istate.F90:149-155``).
    eta_dyn, H_bathy, area : required only when ``seed_face_depth ==
        "nemo_ssh_avg"``.
    z_coord : optional vertical coordinate.  A NEMO-bridged coordinate carries
        the exact reference ``e3t_0``, face depths, and native areas needed by
        the literal DINO QCO expression. Generated coordinates fall back to
        the model-native face-thickness construction.

    Returns
    -------
    U_bar : (n_lat, n_lon+1)
    V_bar : (n_lat+1, n_lon)
    """
    if seed_face_depth not in ("min_rule", "nemo_ssh_avg"):
        raise ValueError(
            "unknown barotropic_seed_face_depth scheme "
            f"{seed_face_depth!r}: must be one of ('min_rule', 'nemo_ssh_avg').")
    if seed_evaluation not in ("generic", "nemo_literal"):
        raise ValueError(
            "unknown barotropic_seed_evaluation scheme "
            f"{seed_evaluation!r}: must be one of ('generic', 'nemo_literal').")
    if seed_evaluation == "nemo_literal" and seed_face_depth != "nemo_ssh_avg":
        raise ValueError(
            "barotropic_seed_evaluation='nemo_literal' requires "
            "barotropic_seed_face_depth='nemo_ssh_avg'.")

    if seed_evaluation == "nemo_literal":
        mesh_seed = _nemo_literal_seed_from_reference_mesh(
            u_3d, v_3d, h_k, eta_dyn, u_mask, v_mask, z_coord)
        if mesh_seed is not None:
            return mesh_seed
        if not legacy_seed_min_rule_faces:
            # Card without NEMO's mesh: the same NEMO e3u(Kmm) as every
            # other qco consumer.  ``legacy_seed_min_rule_faces`` is the
            # harness-only one-variable control (min-of-stretched-cells
            # rescale below); NEMO has no such switch.
            return _nemo_literal_seed_from_card_mesh(
                u_3d, v_3d, eta_dyn, H_bathy, min_water_col, u_mask, v_mask,
                grid, z_coord)

    # h at u/v-faces — min-rule (MOM6/MITgcm hFacW convention).
    # Must match the PE tendency and slow-forcing depth-average which
    # both use min_cell_to_uface/min_cell_to_vface.  Arithmetic mean
    # overestimates face depth at topographic steps, creating a
    # barotropic-baroclinic residual that drives spurious currents.
    h_u = min_cell_to_uface(h_k)
    h_v = min_cell_to_vface(h_k, grid)

    _literal_r1_u = _literal_r1_v = None
    if seed_face_depth == "nemo_ssh_avg":
        # Rescale the min-rule 3-D face thickness by the NEMO/min-rule TOTAL
        # column face-depth ratio (see docstring above) — the loop-entry
        # analogue of NEMO's puu_b finalization, matching the SAME formula
        # `barotropic_face_depth="nemo_ssh_avg"` already uses in-substep
        # (reused verbatim, not re-derived).
        dtype = h_k.dtype
        H_total_2d = jnp.sum(h_k, axis=-1)
        H_u_minrule = min_cell_to_uface(H_total_2d)
        H_v_minrule = min_cell_to_vface(H_total_2d, grid)
        _prep = _nemo_ssh_avg_prep(H_bathy, mask, grid, dtype)
        H_u_nemo, H_v_nemo, _literal_r1_u, _literal_r1_v = _nemo_ssh_avg_apply(
            eta_dyn, u_mask, v_mask, grid, area, _prep,
            return_literal_inverse=True)
        # `_eps` is a bare numerical divide-by-zero guard for THIS ratio's
        # own denominator (`H_u_minrule`) — deliberately NOT
        # `min_water_col`/`config.min_water_column_m` (the caller's
        # physical wet-cell floor applied later, to `h_u`/`h_v`, by
        # `depth_average_to_faces` below). The ratio only needs to avoid
        # 0/0 on masked-out (land) faces where `H_u_minrule` is exactly 0;
        # using the ~0.5 m physical floor here would itself perturb the
        # ratio on legitimate thin-but-wet columns instead of just guarding
        # division. Matches the existing divergent-floor convention
        # documented in `ocean_tendency_common.column_depth`.
        _eps = jnp.asarray(1.0e-10, dtype=dtype)
        ratio_u = jnp.where(u_mask > 0.5, H_u_nemo / jnp.maximum(H_u_minrule, _eps), 1.0)
        ratio_v = jnp.where(v_mask > 0.5, H_v_nemo / jnp.maximum(H_v_minrule, _eps), 1.0)
        h_u = h_u * ratio_u[..., jnp.newaxis]
        h_v = h_v * ratio_v[..., jnp.newaxis]

    if seed_evaluation == "nemo_literal":
        # NEMO DINO MLF restart initialization, istate.F90:149-155:
        # start from zero, add one live-thickness transport per level in source
        # order, then apply the separately associated live reciprocal depth.
        # A Python loop is static at trace time, JIT/autodiff-safe, and prevents
        # XLA from replacing this source-ordered recurrence by the generic
        # stacked tree reduction that measured 7,302/7,035 bit mismatches.
        return (_nemo_literal_seed_depth_mean(
                    u_3d, h_u, u_mask, _literal_r1_u),
                _nemo_literal_seed_depth_mean(
                    v_3d, h_v, v_mask, _literal_r1_v))

    # Barotropic-mean face velocities: thickness-weighted depth average
    # masked by the face mask (#517 item 1: shared depth_average_to_faces;
    # floor = min_water_col, passed verbatim → bit-identical).
    U_bar = depth_average_to_faces(u_3d, h_u, u_mask, min_water_col)
    V_bar = depth_average_to_faces(v_3d, h_v, v_mask, min_water_col)

    return U_bar, V_bar


def _reciprocal_face_area(face_area):
    """``1/(e1*e2)`` on a face, and exactly zero where that face has no area.

    NEMO's ``r1_e1e2u``/``r1_e1e2v`` (``domain.f90:213``) are reciprocals of
    metrics that are strictly positive everywhere in its own domain, so NEMO
    has no statement to match here.  legoESM represents a CLOSED WALL row as a
    v-face of zero extent -- the ORCA2 tripolar card's southernmost v-row has
    ``e1v == e2v == 0`` on all 180 longitudes -- and a plain reciprocal makes
    that row infinite.  The infinity then met the dry face's exactly-zero
    ``r1_v0`` inside ``r3_v`` (``0 * inf``), so the entry inverse face depth
    ``r1_v_entry`` was NaN on that whole row, the barotropic bottom-drag
    statement multiplied it in, and within two substeps the sea surface, the
    velocities and every downstream N2 divisor were NaN.

    Zero is the only finite value a zero-area face can carry, and it is the
    value the sibling statement already gives that row: ``ssh_avg_v`` is
    zeroed on both polar rows a few lines below.  Wherever the face area is
    positive this returns exactly ``1.0 / face_area``, so every card with a
    non-degenerate metric is bit-identical.  The same guarded shape is already
    used for the single metrics in this module (``r1_e2u``/``r1_e1v``).
    """
    positive = face_area > 0.0
    return jnp.where(positive, 1.0 / jnp.where(positive, face_area, 1.0), 0.0)


def _nemo_ssh_avg_prep(H_bathy, mask, grid, dtype, _nfold_mask=None):
    """Loop-invariant prep for :func:`_nemo_ssh_avg_apply` (geometry-only —
    matches NEMO's frozen ``hu_0``/``hv_0``/``r1_e1e2u``).  Hoisted OUTSIDE
    any per-substep/per-call loop so the reference depth + metric build run
    ONCE, not once per ``eta`` snapshot.  See :func:`nemo_ssh_avg_face_depth`
    for the full formula docstring.
    """
    from legoesm.grids.latlon import ensure_geometry

    if _nfold_mask is None:
        _nfold_mask = north_fold_mask(grid)
    H_u_ref, H_v_ref = _min_rule_face_depths(H_bathy * mask, mask, grid,
                                             _nfold_mask)
    _geom = ensure_geometry(grid)
    _r1_e1e2u = _reciprocal_face_area(
        _geom.dx_u * _geom.dy_u).astype(dtype)
    _r1_e1e2v = _reciprocal_face_area(
        _geom.dx_v * _geom.dy_v).astype(dtype)
    return H_u_ref, H_v_ref, _r1_e1e2u, _r1_e1e2v, _nfold_mask


def _nemo_ssh_avg_apply(eta_dyn, u_mask, v_mask, grid, area, prep, *,
                        return_literal_inverse=False,
                        return_ssh_average=False,
                        return_entry_inverse=False):
    """Apply the NEMO ssh-average face-depth formula at one ``eta`` snapshot,
    given the loop-invariant ``prep = _nemo_ssh_avg_prep(...)`` tuple.  See
    :func:`nemo_ssh_avg_face_depth` for the full formula docstring."""
    H_u_ref, H_v_ref, _r1_e1e2u, _r1_e1e2v, _nfold_mask = prep

    area_w = jnp.roll(area, 1, axis=1)
    eta_w = jnp.roll(eta_dyn, 1, axis=1)
    # dynspg_ts.F90:663-666, in the written gfortran association.  These are
    # source statements in the NEMO external-mode identity, so make every
    # binary operation an observable IEEE result before XLA can reassociate it.
    ssh_u_w = nemo_source_round(area_w * eta_w)
    ssh_u_e = nemo_source_round(area * eta_dyn)
    ssh_u_sum = nemo_source_round(ssh_u_w + ssh_u_e)
    ssh_u_scale = nemo_source_round(0.5 * _r1_e1e2u[:, :-1])
    ssh_avg_u = nemo_source_round(ssh_u_scale * ssh_u_sum)
    ssh_avg_u = nemo_source_round(ssh_avg_u * u_mask[:, :-1])
    ssh_avg_u = jnp.concatenate([ssh_avg_u, ssh_avg_u[:, 0:1]], axis=1)
    H_u = nemo_source_round(H_u_ref + ssh_avg_u)
    r1_u_denom = nemo_source_round(nemo_source_round(H_u + 1.0) - u_mask)
    r1_u = nemo_source_round(u_mask / r1_u_denom)
    r1_u0_denom = nemo_source_round(
        nemo_source_round(H_u_ref + 1.0) - u_mask)
    r1_u0 = nemo_source_round(u_mask / r1_u0_denom)
    r3_u_half_sum = nemo_source_round(0.5 * ssh_u_sum)
    r3_u = nemo_source_round(
        nemo_source_round(r3_u_half_sum * r1_u0[:, :-1])
        * _r1_e1e2u[:, :-1])
    r3_u = jnp.concatenate([r3_u, r3_u[:, 0:1]], axis=1)
    r1_u_entry = nemo_source_round(
        r1_u0 / nemo_source_round(1.0 + r3_u))

    area_pad = pad_ns_zero(area)
    eta_pad = pad_ns_zero(eta_dyn)
    ssh_v_s = nemo_source_round(area_pad[:-1] * eta_pad[:-1])
    ssh_v_n = nemo_source_round(area_pad[1:] * eta_pad[1:])
    ssh_v_sum = nemo_source_round(ssh_v_s + ssh_v_n)
    ssh_v_scale = nemo_source_round(0.5 * _r1_e1e2v)
    ssh_avg_v = nemo_source_round(ssh_v_scale * ssh_v_sum)
    ssh_avg_v = nemo_source_round(ssh_avg_v * v_mask)
    ssh_avg_v = _zero_polar_lat_ends(ssh_avg_v)
    if fold_is_local(grid) or _nfold_mask is not None:
        area_fold = fold_vface_row(area, grid)
        eta_fold = fold_vface_row(eta_dyn, grid)
        ssh_n_local = nemo_source_round(area[-1:] * eta_dyn[-1:])
        ssh_n_fold = nemo_source_round(area_fold * eta_fold)
        ssh_n_sum = nemo_source_round(ssh_n_local + ssh_n_fold)
        ssh_n_scale = nemo_source_round(0.5 * _r1_e1e2v[-1:])
        ssh_avg_north = nemo_source_round(ssh_n_scale * ssh_n_sum)
        ssh_avg_north = nemo_source_round(ssh_avg_north * v_mask[-1:])
        ssh_avg_v = apply_north_fold(
            ssh_avg_v, ssh_avg_north, grid, north_mask=_nfold_mask)
    H_v = nemo_source_round(H_v_ref + ssh_avg_v)
    r1_v_denom = nemo_source_round(nemo_source_round(H_v + 1.0) - v_mask)
    r1_v = nemo_source_round(v_mask / r1_v_denom)
    r1_v0_denom = nemo_source_round(
        nemo_source_round(H_v_ref + 1.0) - v_mask)
    r1_v0 = nemo_source_round(v_mask / r1_v0_denom)
    r3_v_half_sum = nemo_source_round(0.5 * ssh_v_sum)
    r3_v = _zero_polar_lat_ends(nemo_source_round(
        nemo_source_round(r3_v_half_sum * r1_v0) * _r1_e1e2v))
    r1_v_entry = nemo_source_round(
        r1_v0 / nemo_source_round(1.0 + r3_v))
    if return_entry_inverse:
        return H_u, H_v, r1_u_entry, r1_v_entry
    if return_literal_inverse:
        if return_ssh_average:
            return H_u, H_v, r1_u, r1_v, ssh_avg_u, ssh_avg_v
        return H_u, H_v, r1_u, r1_v
    return H_u, H_v


def nemo_ssh_avg_face_depth(eta_dyn, H_bathy, mask, u_mask, v_mask, grid,
                            area, dtype):
    """NEMO zhup2_e/zhvp2_e-style C-grid face depth (dynspg_ts.F90:568-592
    flux depth; :658-666,771-778 drag/update depth; :963-966,978-979 the
    ``puu_b``-finalizing ssh-average — same formula, evaluated on whichever
    ssh time-level the caller supplies):

        H_u = hu_0 + 0.5 * r1_e1e2u * (area[j]*eta[j] + area[j+1]*eta[j+1])

    ``hu_0``/``hv_0`` (the fixed still-water reference depth) is the
    min-rule applied to ``H_bathy`` alone — the lego equivalent of NEMO's
    frozen ``domain.F90:139-147`` reference thickness.  ``r1_e1e2u ==
    1/(e1u*e2u)`` is the face's OWN metric area (domhgr.F90:146-160), not a
    sum of the two adjacent T-cell areas.  Standalone extraction (#1226
    round 2 item 1) of the closure previously private to
    ``_run_substep_loop`` — reused verbatim (not re-derived) so the
    in-substep face-thickness rule and the barotropic loop-entry SEED
    (``barotropic_substeps_latlon_cgrid``) apply the identical NEMO
    formula.  One-shot convenience wrapper around
    :func:`_nemo_ssh_avg_prep` + :func:`_nemo_ssh_avg_apply` — callers that
    evaluate the formula more than once per fixed ``(H_bathy, grid)``
    (e.g. the per-substep loop) should call ``_nemo_ssh_avg_prep`` ONCE and
    reuse its ``prep`` tuple, to avoid rebuilding the geometry every call.

    Parameters
    ----------
    eta_dyn : (n_lat, n_lon) dynamic ssh snapshot the average is taken at.
    H_bathy, mask : (n_lat, n_lon) raw bathymetry / land mask.
    u_mask, v_mask : C-grid face masks.
    grid : LatLonGrid (fold/geometry lookups).
    area : (n_lat, n_lon) T-cell area (``grid.area``, already cast to
        ``dtype`` by the caller).
    dtype : output dtype.

    Returns
    -------
    (H_u, H_v) : NEMO-rule face depths, same shapes as ``min_cell_to_uface``/
        ``min_cell_to_vface`` would return.
    """
    prep = _nemo_ssh_avg_prep(H_bathy, mask, grid, dtype)
    return _nemo_ssh_avg_apply(eta_dyn, u_mask, v_mask, grid, area, prep)


def nemo_literal_metric_transports(
    H_u, H_v, U, V, u_mask, v_mask, grid,
):
    """Assemble NEMO DINO's literal ``zhU``/``zhV`` metric transports."""
    # Rich/fold-aware geometry carries the full 2-D u-face meridional metric.
    # Lean LatLonGrid callers retain the canonical regular-grid construction.
    e2u = (grid.dy_u if hasattr(grid, "dy_u")
           else (grid.dy * 0.5)[:, jnp.newaxis])
    # Rich NEMO-faithful geometry carries e1v directly. Lean LatLonGrid tests
    # and callers reconstruct the same canonical v-face width used by the
    # generic divergence path.
    e1v = (grid.dx_v if hasattr(grid, "dx_v")
           else (grid.radius * grid.dlon * vface_zonal_cos_lat(grid))[:, jnp.newaxis])
    zh_u = ((e2u * U) * H_u) * u_mask
    zh_v = ((e1v * V) * H_v) * v_mask
    return zh_u, zh_v


def nemo_literal_accumulate_transport(
    Hu_sum, Hv_sum, raw_weight, H_u, H_v, U, V, u_mask, v_mask, grid,
):
    """One source-ordered DINO ``un_adv/vn_adv`` accumulation row.

    Preserves ``dynspg_ts.F90:699-704,734-737``: materialise metric
    transports, multiply by raw ``za2``, multiply by the reciprocal face
    metric, then add to the running accumulator.  The metric factors cancel
    algebraically but round 58 measured the cancelled form outside the
    pointwise bar.  Optimization barriers pin the stated association against
    XLA fusion while retaining JIT/autodiff compatibility.
    """
    zh_u, zh_v = nemo_literal_metric_transports(
        H_u, H_v, U, V, u_mask, v_mask, grid)
    dtype = Hu_sum.dtype
    zh_u = zh_u.astype(dtype)
    zh_v = zh_v.astype(dtype)
    raw_weight = jnp.asarray(raw_weight, dtype=dtype)
    e2u = jnp.asarray(
        grid.dy_u if hasattr(grid, "dy_u")
        else (grid.dy * 0.5)[:, jnp.newaxis], dtype=dtype)
    e1v = jnp.asarray(
        grid.dx_v if hasattr(grid, "dx_v")
        else (grid.radius * grid.dlon
              * vface_zonal_cos_lat(grid))[:, jnp.newaxis], dtype=dtype)
    # NEMO's reciprocal metric arrays are zero on masked/polar faces.  A raw
    # divide by the geometric zero would turn the already-masked transport
    # into 0*Inf=NaN on lean-grid tests and at wall halos.
    r1_e2u = jnp.where(u_mask != 0, 1.0 / e2u, 0.0)
    r1_e1v = jnp.where(v_mask != 0, 1.0 / e1v, 0.0)
    # Every ``b`` below is one written operation from dynspg_ts.F90's
    # dyn_cor_2D_init recurrence.  ``optimization_barrier`` alone is stripped
    # by XLA; use the shared IEEE identity that keeps each source result
    # observable to the next statement.
    b = nemo_source_round
    inc_u = b(b(raw_weight * b(zh_u)) * b(r1_e2u))
    inc_v = b(b(raw_weight * b(zh_v)) * b(r1_e1v))
    return b(Hu_sum + inc_u), b(Hv_sum + inc_v)


def nemo_literal_continuity_divergence(
    H_u, H_v, U, V, u_mask, v_mask, grid,
):
    """NEMO DINO's literal QCO metric-transport continuity expression.

    The active oracle first assembles metric transports in operand order
    (``dynspg_ts.F90:699-704``)::

        zhU = e2u * ua_e * zhup2_e
        zhV = e1v * va_e * zhvp2_e

    and then evaluates (:722-724)::

        zhdiv = ((zhU(i)-zhU(i-1)) + (zhV(j)-zhV(j-1))) * r1_e1e2t

    This differs by a few ulps from sending ``H*U`` through the generic
    velocity-divergence operator, which multiplies face metrics after the
    depth/velocity product and divides by area after the flux sum.  Keep this
    specialization beside the NEMO ssh-average depth rule: it returns only the
    continuity divergence and does not change the non-metric ``H*U`` transport
    accumulated for tracer advection.

    Arrays use legoESM's full C-face layout, so ``[:, 1:]-[:, :-1]`` supplies
    NEMO's periodic U east-minus-west difference and
    ``[1:]-[:-1]`` supplies its closed-wall V north-minus-south difference.
    """
    zh_u, zh_v = nemo_literal_metric_transports(
        H_u, H_v, U, V, u_mask, v_mask, grid)
    du = zh_u[:, 1:] - zh_u[:, :-1]
    dv = zh_v[1:] - zh_v[:-1]
    return (du + dv) * (1.0 / grid.area)


def _min_rule_face_depths(H_total, mask, grid, _nfold_mask):
    """Min-rule C-grid face depths of a total column depth field.

    Standalone extraction of the ``_face_depths`` closure body (unchanged;
    see ``_run_substep_loop`` for the original docstring) — the ``mask``
    argument is unused inside the body (kept for call-site symmetry with
    ``nemo_ssh_avg_face_depth``, whose reference-depth branch masks
    ``H_bathy`` before calling this) but is accepted, not applied twice.
    """
    H_u = jnp.minimum(jnp.roll(H_total, 1, axis=1), H_total)
    H_u = jnp.concatenate([H_u, H_u[:, 0:1]], axis=1)
    H_total_pad = pad_ns_zero(H_total)
    H_v = jnp.minimum(H_total_pad[:-1], H_total_pad[1:])
    H_v = _zero_polar_lat_ends(H_v)
    if fold_is_local(grid) or _nfold_mask is not None:
        north = jnp.minimum(
            H_total[-1:], fold_vface_row(H_total, grid),
        )
        H_v = apply_north_fold(H_v, north, grid, north_mask=_nfold_mask)
    return H_u, H_v


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


def _nemo_literal_een_coefficients(
    eta, z_coord, dtype, scheme="een", *, _e3f_test_override=None,
):
    """Materialize NEMO's eight frozen EEN or ENE coefficients.

    ``scheme="een"`` transcribes ``dyn_cor_2D_init``'s 12-point triads;
    ``scheme="ene"`` transcribes its four 2-point Sadourny coefficients.  The
    name is retained because the EEN implementation and bridge bundle predate
    the ENE sibling; both branches consume the same frozen Kmm geometry.
    """
    if scheme not in ("een", "ene"):
        raise ValueError(
            f"unknown literal barotropic PV-flux scheme {scheme!r}")
    raw = getattr(z_coord, "nemo_een_barotropic", None)
    if raw is None:
        raise ValueError(
            "barotropic_een_coefficient_evaluation='nemo_literal' requires "
            "bridge-carried raw NEMO EEN/QCO operands")
    if eta is None or eta.shape != raw.ff_f.shape:
        raise ValueError(
            "literal NEMO EEN coefficients require eta at the native A2D "
            f"shape {raw.ff_f.shape}; got {None if eta is None else eta.shape}")
    # This builder is the literal dynspg_ts coefficient program.  A bare
    # optimization_barrier is stripped from optimized HLO; keep every written
    # binary64 result observable with the shared NEMO source-round identity.
    b = nemo_source_round
    one = jnp.asarray(1.0, dtype=dtype)
    half = jnp.asarray(0.5, dtype=dtype)
    quarter = jnp.asarray(0.25, dtype=dtype)
    leading_scale = jnp.asarray(
        1.0 / 12.0 if scheme == "een" else 0.25, dtype=dtype)
    eta = jnp.asarray(eta, dtype=dtype)
    ff = jnp.asarray(raw.ff_f, dtype=dtype)
    e3u0 = jnp.asarray(raw.e3u_0, dtype=dtype)
    e3v0 = jnp.asarray(raw.e3v_0, dtype=dtype)
    e3f0 = jnp.asarray(raw.e3f_0, dtype=dtype)
    umask = jnp.asarray(raw.umask, dtype=dtype)
    vmask = jnp.asarray(raw.vmask, dtype=dtype)
    fmask = jnp.asarray(raw.fmask, dtype=dtype)

    def recip(depth, wet):
        return b(wet / b(depth + one - wet))

    wet_u = (jnp.asarray(raw.hu_0, dtype=dtype) > 0.0).astype(dtype)
    wet_v = (jnp.asarray(raw.hv_0, dtype=dtype) > 0.0).astype(dtype)
    wet_f = (jnp.asarray(raw.hf_0, dtype=dtype) > 0.0).astype(dtype)
    r1_hu0 = recip(jnp.asarray(raw.hu_0, dtype=dtype), wet_u)
    r1_hv0 = recip(jnp.asarray(raw.hv_0, dtype=dtype), wet_v)
    r1_hf0 = recip(jnp.asarray(raw.hf_0, dtype=dtype), wet_f)
    e1t = jnp.asarray(raw.e1t, dtype=dtype)
    e2t = jnp.asarray(raw.e2t, dtype=dtype)
    e1u = jnp.asarray(raw.e1u, dtype=dtype)
    e2u = jnp.asarray(raw.e2u, dtype=dtype)
    e1v = jnp.asarray(raw.e1v, dtype=dtype)
    e2v = jnp.asarray(raw.e2v, dtype=dtype)
    e1f = jnp.asarray(raw.e1f, dtype=dtype)
    e2f = jnp.asarray(raw.e2f, dtype=dtype)
    area_eta = b(b(e1t * e2t) * eta)
    east = jnp.roll(area_eta, -1, axis=1)
    north = jnp.roll(area_eta, -1, axis=0)
    northeast = jnp.roll(north, -1, axis=1)
    r3u = b(b(half * b(area_eta + east)) * r1_hu0 / b(e1u * e2u))
    r3v = b(b(half * b(area_eta + north)) * r1_hv0 / b(e1v * e2v))
    quad = b(b(area_eta + east) + b(north + northeast))
    r3f = b(b(quarter * quad) * r1_hf0 / b(e1f * e2f))
    e3u = b(e3u0 * b(one + r3u[..., None] * umask) * umask)
    e3v = b(e3v0 * b(one + r3v[..., None] * vmask) * vmask)
    e3f = b(e3f0 * b(one + r3f[..., None] * fmask))
    if _e3f_test_override is not None:
        candidate = jnp.asarray(_e3f_test_override, dtype=dtype)
        if candidate.shape != e3f.shape:
            raise ValueError(
                "literal EEN test e3f override must match the native A2D "
                f"coefficient divisor shape {e3f.shape}; got {candidate.shape}")
        e3f = candidate
    q = b(ff[..., None] / e3f)

    def shift(value, di=0, dj=0):
        out = jnp.roll(value, di, axis=1) if di else value
        return jnp.roll(out, dj, axis=0) if dj else out

    def triad(a, c, d):
        return b(b(a + c) + d)

    def coefficient(face, neighbor, neighbor_mask, q_factor,
                    neighbor_metric, local_metric, r1_h):
        term = b(b(b(face * neighbor) * neighbor_mask) * q_factor)
        acc = jnp.zeros_like(r1_h)
        for jk in range(term.shape[-1]):
            acc = b(acc + term[..., jk])
        return b(b(b(b(leading_scale * b(one / local_metric)) * r1_h)
                     * neighbor_metric) * acc)

    def ene_coefficient(face, neighbor, neighbor_mask, e3f_divisor,
                        f_factor, neighbor_metric, local_metric, r1_h):
        """dynspg_ts.F90:1387-1409, without moving ``ff_f`` into q.

        ENE accumulates ``e3u*e3v*mask/e3f`` first and multiplies the frozen
        Coriolis value only after the vertical sum.  Folding ``ff_f/e3f``
        into the per-level term is algebraically equivalent but not the NEMO
        source program.
        """
        term = b(face * neighbor)
        term = b(term * neighbor_mask)
        term = b(term / e3f_divisor)
        acc = jnp.zeros_like(r1_h)
        for jk in range(term.shape[-1]):
            acc = b(acc + term[..., jk])
        scale = b(quarter * b(one / local_metric))
        scale = b(scale * r1_h)
        scale = b(scale * neighbor_metric)
        scale = b(scale * f_factor)
        return b(scale * acc)

    if scheme == "een":
        uq = {
            "nw": triad(shift(q, 1, 0), q, shift(q, 0, 1)),
            "ne": triad(shift(q, 0, 1), q, shift(q, -1, 0)),
            "sw": triad(q, shift(q, 0, 1), shift(q, 1, 1)),
            "se": triad(shift(q, -1, 1), shift(q, 0, 1), q),
        }
        vq = {
            "se": triad(shift(q, 1, 0), q, shift(q, 0, 1)),
            "sw": triad(shift(q, 1, 1), shift(q, 1, 0), q),
            "ne": triad(shift(q, 0, -1), q, shift(q, 1, 0)),
            "nw": triad(q, shift(q, 1, 0), shift(q, 1, -1)),
        }
    else:
        # dynspg_ts.F90 np_ENE:1383-1410.  The northern U pair shares
        # ff_f/e3f at F(i,j); the southern pair uses F(i,j-1).  The western V
        # pair uses F(i-1,j); the eastern pair uses F(i,j).
        uq = {
            "nw": q,
            "ne": q,
            "sw": shift(q, 0, 1),
            "se": shift(q, 0, 1),
        }
        vq = {
            "nw": shift(q, 1, 0),
            "ne": q,
            "sw": shift(q, 1, 0),
            "se": q,
        }
        ene_u_f = {
            "nw": (e3f, ff),
            "ne": (e3f, ff),
            "sw": (shift(e3f, 0, 1), shift(ff, 0, 1)),
            "se": (shift(e3f, 0, 1), shift(ff, 0, 1)),
        }
        ene_v_f = {
            "nw": (shift(e3f, 1, 0), shift(ff, 1, 0)),
            "ne": (e3f, ff),
            "sw": (shift(e3f, 1, 0), shift(ff, 1, 0)),
            "se": (e3f, ff),
        }
    un = {
        "nw": (e3v, vmask, e1v),
        "ne": (shift(e3v, -1, 0), shift(vmask, -1, 0), shift(e1v, -1, 0)),
        "sw": (shift(e3v, 0, 1), shift(vmask, 0, 1), shift(e1v, 0, 1)),
        "se": (shift(e3v, -1, 1), shift(vmask, -1, 1), shift(e1v, -1, 1)),
    }
    vn = {
        "nw": (shift(e3u, 1, -1), shift(umask, 1, -1), shift(e2u, 1, -1)),
        "ne": (shift(e3u, 0, -1), shift(umask, 0, -1), shift(e2u, 0, -1)),
        "sw": (shift(e3u, 1, 0), shift(umask, 1, 0), shift(e2u, 1, 0)),
        "se": (e3u, umask, e2u),
    }
    r1_hu = b(r1_hu0 / b(one + r3u))
    r1_hv = b(r1_hv0 / b(one + r3v))
    out = {}
    for corner in ("nw", "ne", "sw", "se"):
        neighbor, neighbor_mask, metric = un[corner]
        if scheme == "ene":
            divisor, f_factor = ene_u_f[corner]
            out[f"ffu_{corner}"] = ene_coefficient(
                e3u, neighbor, neighbor_mask, divisor, f_factor,
                metric, e1u, r1_hu)
        else:
            out[f"ffu_{corner}"] = coefficient(
                e3u, neighbor, neighbor_mask, uq[corner], metric, e1u, r1_hu)
        neighbor, neighbor_mask, metric = vn[corner]
        if scheme == "ene":
            divisor, f_factor = ene_v_f[corner]
            out[f"ffv_{corner}"] = ene_coefficient(
                e3v, neighbor, neighbor_mask, divisor, f_factor,
                metric, e2v, r1_hv)
        else:
            out[f"ffv_{corner}"] = coefficient(
                e3v, neighbor, neighbor_mask, vq[corner], metric, e2v, r1_hv)
    return out


def _build_een_barotropic_inputs(h_k, grid, mask, u_mask, v_mask, dtype,
                                 metric_complete=False,
                                 een_q_boundary="neumann_fill",
                                 een_e3f_scheme="min",
                                 dz_ref=None,
                                 coefficient_evaluation="generic",
                                 eta=None,
                                 z_coord=None,
                                 scheme="een"):
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

    ``een_q_boundary`` / ``een_e3f_scheme`` are the SAME two options the 3-D
    EEN caller takes (``ocean_pe_latlon_cgrid._bc_pv_flux``), threaded here so
    a NEMO-faithful card's setting actually reaches the barotropic path:
    NEMO's ``dyn_cor_2D_init`` builds its ``ffu``/``ffv`` coefficients from
    ``ff_f(ji,jj)/e3f_vor(ji,jj,jk)`` RAW (``dynspg_ts.F90:1517-1531``) — the
    same ``e3f_vor`` array ``vor_een`` uses, with no fill and (``DINO
    ln_dynvor_msk=.false.``) no ``fmask``.  That is ``een_q_boundary=
    "nemo_live"`` + ``een_e3f_scheme="nemo_avg"`` (``nn_e3f_typ=1``).
    Defaults keep the legacy ``neumann_fill``/``min`` behaviour bit-identical.

    ``dz_ref`` (``z_coord.dz_ref``) is the SAME per-level reference thickness
    the 3-D EEN caller forwards (``ocean_pe_latlon_cgrid.py`` ``_bc_pv_flux``
    call site): under ``een_e3f_scheme="nemo_avg"`` it selects NEMO's
    ``e3f_0`` fully-dry-vertex fallback instead of the legacy ``BIG_H``
    sentinel (#1226 item 10).  Passing it here is required by this function's
    own "must be the SAME operator as the 3-D EEN" rationale — with
    ``dz_ref=None`` on one path and ``z_coord.dz_ref`` on the other the two
    build DIFFERENT ``e3f`` at fully-dry vertices.  ``None`` (default) keeps
    the legacy behaviour bit-identical, and the ``"min"`` branch is
    ``dz_ref``-independent either way.

    Returns a dict of reference (η-independent, like NEMO's frozen arrays)
    3-D face/vertex thicknesses, the vertex Coriolis ``f_vtx``, the vertex and
    3-D face masks, the column depths ``hu``/``hv``, and the ``q_boundary``
    string the operator is to be called with — all static geometry/config.
    """
    if coefficient_evaluation not in ("generic", "nemo_literal"):
        raise ValueError(
            "unknown barotropic_een_coefficient_evaluation "
            f"{coefficient_evaluation!r}; expected 'generic' or 'nemo_literal'")
    if scheme not in ("ene", "een"):
        raise ValueError(f"unknown barotropic PV-flux scheme {scheme!r}")
    if coefficient_evaluation == "nemo_literal" and not metric_complete:
        raise ValueError(
            "barotropic_een_coefficient_evaluation='nemo_literal' requires "
            "barotropic_coriolis='ene_metric' or 'een_metric'")
    e3u = min_cell_to_uface(h_k).astype(dtype)      # (nlat, nlon+1, nlev)
    e3v = min_cell_to_vface(h_k, grid).astype(dtype)  # (nlat+1, nlon, nlev)
    hu = jnp.sum(e3u, axis=-1)                       # (nlat, nlon+1)
    hv = jnp.sum(e3v, axis=-1)                       # (nlat+1, nlon)
    # F-point (vertex) thickness via the SAME production helper the 3-D EEN
    # caller uses (min-rule = MITgcm hFacZ; "nemo_avg" = NEMO nn_e3f_typ=1
    # masked average).  Never re-derived here: an inline copy of the
    # nemo_avg divisor is exactly the duplication that let a mutated guard
    # pass undetected in #1226 item 10.  Thickness-only call -> Fu/u=None.
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import een_e3f_h_vtx
    if een_e3f_scheme not in ("min", "nemo_avg", "nemo_avg4"):
        raise ValueError(
            f"_build_een_barotropic_inputs: unknown een_e3f_scheme "
            f"{een_e3f_scheme!r}; expected 'min', 'nemo_avg', or "
            "'nemo_avg4'.")
    if een_q_boundary not in ("neumann_fill", "nemo_live"):
        raise ValueError(
            f"_build_een_barotropic_inputs: unknown een_q_boundary "
            f"{een_q_boundary!r}; expected 'neumann_fill' or 'nemo_live'.")
    h_vtx, _, _ = een_e3f_h_vtx(h_k, None, None, grid, een_e3f_scheme,
                                dz_ref=dz_ref)
    h_vtx = h_vtx.astype(dtype)
    f_vtx = vertex_coriolis(grid).astype(dtype)      # (nlat+1, nlon+1)
    vtx_mask = compute_vertex_mask(mask, grid=grid)
    u_mask_3d = (u_mask[..., None] * (e3u > 0)).astype(dtype)
    v_mask_3d = (v_mask[..., None] * (e3v > 0)).astype(dtype)
    out = dict(e3u=e3u, e3v=e3v, hu=hu, hv=hv, h_vtx=h_vtx, f_vtx=f_vtx,
               vtx_mask=vtx_mask, u_mask_3d=u_mask_3d, v_mask_3d=v_mask_3d,
               metric_complete=bool(metric_complete),
               q_boundary=een_q_boundary, scheme=scheme)
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
    if coefficient_evaluation == "nemo_literal":
        out["literal_coefficients"] = _nemo_literal_een_coefficients(
            eta, z_coord, dtype, scheme=scheme)
    out["coefficient_evaluation"] = coefficient_evaluation
    return out


def _nemo_literal_barotropic_coriolis(U_bar, V_bar, coefficients,
                                      *, return_terms=False):
    """Apply frozen NEMO ENE/EEN coefficients in source association order."""
    # Native NEMO arrays name the east/north face of T(i,j); legoESM stores
    # redundant west/south faces.  Apply first, map only afterward.
    ua = U_bar[:, 1:]
    va = V_bar[1:, :]
    east_v = jnp.roll(va, -1, axis=1)
    south_v = jnp.concatenate([jnp.zeros_like(va[:1]), va[:-1]], axis=0)
    southeast_v = jnp.roll(south_v, -1, axis=1)
    north_u = jnp.concatenate([ua[1:], jnp.zeros_like(ua[:1])], axis=0)
    west_u = jnp.roll(ua, 1, axis=1)
    northwest_u = jnp.roll(north_u, 1, axis=1)
    products = {
        "u_nw": nemo_source_round(coefficients["ffu_nw"] * va),
        "u_ne": nemo_source_round(coefficients["ffu_ne"] * east_v),
        "u_sw": nemo_source_round(coefficients["ffu_sw"] * south_v),
        "u_se": nemo_source_round(coefficients["ffu_se"] * southeast_v),
        "v_sw": nemo_source_round(coefficients["ffv_sw"] * west_u),
        "v_se": nemo_source_round(coefficients["ffv_se"] * ua),
        "v_nw": nemo_source_round(coefficients["ffv_nw"] * northwest_u),
        "v_ne": nemo_source_round(coefficients["ffv_ne"] * north_u),
    }
    products["u_north_pair"] = nemo_source_round(
        products["u_nw"] + products["u_ne"])
    products["u_south_pair"] = nemo_source_round(
        products["u_sw"] + products["u_se"])
    products["u_total"] = nemo_source_round(
        products["u_north_pair"] + products["u_south_pair"])
    products["v_south_pair"] = nemo_source_round(
        products["v_sw"] + products["v_se"])
    products["v_north_pair"] = nemo_source_round(
        products["v_nw"] + products["v_ne"])
    products["v_total"] = -nemo_source_round(
        products["v_south_pair"] + products["v_north_pair"])
    cor_u = jnp.concatenate([products["u_total"][:, -1:], products["u_total"]], axis=1)
    cor_v = jnp.concatenate(
        [jnp.zeros_like(products["v_total"][:1]), products["v_total"]], axis=0)
    if return_terms:
        return cor_u, cor_v, products
    return cor_u, cor_v


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
    if pre.get("coefficient_evaluation", "generic") == "nemo_literal":
        return _nemo_literal_barotropic_coriolis(
            U_bar, V_bar, pre["literal_coefficients"])

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
        f_vtx=pre["f_vtx"],
        q_boundary=pre.get("q_boundary", "neumann_fill"))
    cor_u = jnp.sum(pre["e3u"] * diag_u, axis=-1) / jnp.maximum(pre["hu"], eps)
    cor_v = jnp.sum(pre["e3v"] * diag_v, axis=-1) / jnp.maximum(pre["hv"], eps)
    if pre.get("metric_complete", False):
        # eps floor mirrors the /hu guard: e1u=R·cosφ·dλ→0 only on a pole-covering
        # row (never for DINO); e2v is always positive.
        cor_u = cor_u / jnp.maximum(pre["e1u"], eps)   # NEMO r1_e1u
        cor_v = cor_v / jnp.maximum(pre["e2v"], eps)   # NEMO r1_e2v
    return cor_u, cor_v


def ene_barotropic_coriolis(U_bar, V_bar, pre, eps=1.0e-10):
    """NEMO ``np_ENE`` Sadourny barotropic Coriolis at F-points.

    This is the depth-integrated sibling of the already-canonical 3-D
    ``pv_flux_ene`` operator.  It reuses the same frozen Kmm thickness bundle
    as the EEN path but selects NEMO dynspg_ts.F90:1383-1412's 1/4 two-point
    recurrence instead of the 1/12 AL81 triads.  Horizontal metric widths are
    folded in exactly as for :func:`een_barotropic_coriolis`.
    """
    if pre.get("coefficient_evaluation", "generic") == "nemo_literal":
        return _nemo_literal_barotropic_coriolis(
            U_bar, V_bar, pre["literal_coefficients"])

    from legoesm.ocean.dynamics.latlon_cgrid_operators import pv_flux_ene

    nlev = pre["e3u"].shape[-1]
    U_src, V_src = U_bar, V_bar
    if pre.get("metric_complete", False):
        U_src = U_bar * pre["e2u"]
        V_src = V_bar * pre["e1v"]
    u3 = jnp.broadcast_to(U_src[..., None], U_src.shape + (nlev,))
    v3 = jnp.broadcast_to(V_src[..., None], V_src.shape + (nlev,))
    diag_u, diag_v = pv_flux_ene(
        jnp.zeros_like(pre["h_vtx"]), pre["h_vtx"], pre["e3v"], v3,
        pre["e3u"], u3, pre["u_mask_3d"], pre["v_mask_3d"],
        pre["vtx_mask"], f_vtx=pre["f_vtx"],
    )
    cor_u = jnp.sum(pre["e3u"] * diag_u, axis=-1) / jnp.maximum(pre["hu"], eps)
    cor_v = jnp.sum(pre["e3v"] * diag_v, axis=-1) / jnp.maximum(pre["hv"], eps)
    if pre.get("metric_complete", False):
        cor_u = cor_u / jnp.maximum(pre["e1u"], eps)
        cor_v = cor_v / jnp.maximum(pre["e2v"], eps)
    return cor_u, cor_v


def barotropic_coriolis_een_pre_step(u_3d, v_3d, h_k, grid, mask, u_mask,
                                     v_mask, min_water_col, dtype,
                                     metric_complete=False,
                                     een_q_boundary="neumann_fill",
                                     een_e3f_scheme="min",
                                     dz_ref=None,
                                     coefficient_evaluation="generic",
                                     eta=None,
                                     z_coord=None,
                                     pre=None,
                                     return_pre=False,
                                     scheme="een"):
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

    ``dz_ref`` (``z_coord.dz_ref``) is forwarded verbatim — see
    :func:`_build_een_barotropic_inputs`; it must be the SAME value the
    substep loop and the 3-D EEN caller pass, or this subtraction uses a
    different ``e3f`` at fully-dry vertices than the live term it cancels.
    """
    if pre is None:
        pre = _build_een_barotropic_inputs(
            h_k, grid, mask, u_mask, v_mask, dtype,
            metric_complete=metric_complete,
            een_q_boundary=een_q_boundary,
            een_e3f_scheme=een_e3f_scheme,
            dz_ref=dz_ref,
            coefficient_evaluation=coefficient_evaluation,
            eta=eta,
            z_coord=z_coord,
            scheme=scheme)
    U_bar, V_bar = _depth_average_to_faces(
        u_3d, v_3d, h_k, min_water_col, mask, u_mask, v_mask, grid)
    if scheme == "ene":
        cor_u, cor_v = ene_barotropic_coriolis(U_bar, V_bar, pre)
    else:
        cor_u, cor_v = een_barotropic_coriolis(U_bar, V_bar, pre)
    if return_pre:
        return cor_u, cor_v, pre
    return cor_u, cor_v


def _nemo_flux_form_external_velocity_update(
    velocity_entry,
    depth_entry,
    depth_pgf,
    depth_midpoint,
    depth_kmm,
    depth_exit,
    pgf,
    transport_tendency,
    slow_forcing,
    dt_s,
    wet_mask,
    min_water_col,
):
    """NEMO key_qcoTest_FluxForm external-mode transport update.

    This is the literal ``ua_e``/``va_e`` numerator and exit-depth division
    from NEMO 5.0.2 ``dynspg_ts.F90:731-761``.  It is deliberately not a
    selector: NEMO's WS-RK3 flux-form identity has no alternate composition.
    """
    return (
        depth_entry * velocity_entry
        + dt_s * (
            depth_pgf * pgf
            + depth_midpoint * transport_tendency
            + depth_kmm * slow_forcing
        )
    ) / jnp.maximum(depth_exit, min_water_col) * wet_mask


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
    ab3_raw_hist=None,
    een_pre=None,
    drag_r_u=None, drag_r_v=None, drag_r_t=None,
    tide_basis=None, tide_cos=None, tide_sin=None,
    transport_sum_init=None,
    primary_transport_average=False,
    return_trace=False,
    nemo_flux_form_update_test_override=None,
    nemo_continuity_update_test_override=None,
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
    # Function-scope import (deferred, per the core->physics import rule);
    # resolved once per trace, not per substep.
    from legoesm.ocean.physics.tidal_forcing import (
        tidal_acceleration_from_phase)

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
    if transport_sum_init is None:
        Hu_sum = jnp.zeros((n_lat, n_lon + 1), dtype=dtype)
        Hv_sum = jnp.zeros((n_lat + 1, n_lon), dtype=dtype)
    else:
        Hu_sum = jnp.asarray(transport_sum_init[0], dtype=dtype)
        Hv_sum = jnp.asarray(transport_sum_init[1], dtype=dtype)
    eta_sum = jnp.zeros((n_lat, n_lon), dtype=dtype)
    U_sum = jnp.zeros((n_lat, n_lon + 1), dtype=dtype)
    V_sum = jnp.zeros((n_lat + 1, n_lon), dtype=dtype)

    _nfold_mask = north_fold_mask(grid)

    _face_depth_mode = config.barotropic.barotropic_face_depth
    if _face_depth_mode not in ("min_rule", "nemo_ssh_avg"):
        raise ValueError(
            "unknown barotropic_face_depth scheme "
            f"{_face_depth_mode!r}: must be one of ('min_rule', 'nemo_ssh_avg').")
    _continuity_evaluation = config.barotropic.barotropic_continuity_evaluation
    if _continuity_evaluation not in ("generic", "nemo_literal"):
        raise ValueError(
            "unknown barotropic_continuity_evaluation "
            f"{_continuity_evaluation!r}: must be one of "
            "('generic', 'nemo_literal').")
    _transport_evaluation = getattr(
        config.barotropic,
        "barotropic_transport_accumulation_evaluation", "generic")
    if _transport_evaluation not in ("generic", "nemo_literal"):
        raise ValueError(
            "unknown barotropic_transport_accumulation_evaluation "
            f"{_transport_evaluation!r}: must be one of "
            "('generic', 'nemo_literal').")

    def _face_depths(H_total):
        """Min-rule C-grid face depths (module-level ``_min_rule_face_depths``,
        reused verbatim — see that function's docstring)."""
        return _min_rule_face_depths(H_total, mask, grid, _nfold_mask)

    # --- "nemo_ssh_avg" reference depth + metric prep (built ONCE, outside
    # the substep loop — geometry-only, matches NEMO's frozen
    # hu_0/hv_0/r1_e1e2u; module-level ``_nemo_ssh_avg_prep``, reused
    # verbatim — see ``nemo_ssh_avg_face_depth``'s docstring for the formula).
    _ssh_avg_prep = (
        _nemo_ssh_avg_prep(H_bathy, mask, grid, dtype, _nfold_mask)
        if _face_depth_mode == "nemo_ssh_avg" else None)

    def _ssh_avg_face_depths(eta_dyn):
        """NEMO zhup2_e/zhvp2_e-style face depth (module-level
        ``_nemo_ssh_avg_apply`` against the hoisted ``_ssh_avg_prep``,
        reused verbatim — see ``nemo_ssh_avg_face_depth``'s docstring);
        byte-identical to the prior inline closure."""
        return _nemo_ssh_avg_apply(eta_dyn, u_mask, v_mask, grid, area,
                                   _ssh_avg_prep)

    def _ssh_avg_entry_inverse(eta_dyn):
        return _nemo_ssh_avg_apply(
            eta_dyn, u_mask, v_mask, grid, area, _ssh_avg_prep,
            return_entry_inverse=True)[2:]

    _nemo_flux_form_update = nemo_flux_form_update_active(config)
    # Harness-only causal arm.  This is deliberately absent from public
    # configuration: NEMO exposes no switch inside its WS-RK3 flux-form
    # identity.  Production always follows the source-attested branch above.
    if nemo_flux_form_update_test_override is not None:
        _nemo_flux_form_update = bool(nemo_flux_form_update_test_override)
    if _nemo_flux_form_update:
        if _face_depth_mode != "nemo_ssh_avg":
            raise ValueError(
                "NEMO WS-RK3 flux-form external update requires "
                "barotropic_face_depth='nemo_ssh_avg'")
        H_u_kmm, H_v_kmm = _ssh_avg_face_depths(eta)
    if _face_depth_mode == "nemo_ssh_avg":
        r1_H_u_entry, r1_H_v_entry = _ssh_avg_entry_inverse(eta)

    def substep_body(wts_i, carry):
        """Single barotropic substep with BEBT, slow forcing, MAXVEL, and cosine filter.

        Parameters
        ----------
        w_i : scalar
            Cosine filter weight for this substep (1.0 for box filter).
        carry : tuple
            (eta, U_bar, V_bar, Hu_sum, Hv_sum, eta_sum, U_sum, V_sum)
        """
        # Unpack the carry.
        if ab3_za is not None:
            (eta_c, U_bar_c, V_bar_c,
             Hu_sum_c, Hv_sum_c, eta_sum_c, U_sum_c, V_sum_c,
             Ub_c, Ubb_c, Vb_c, Vbb_c, etab_c, etabb_c) = carry
        else:
            (eta_c, U_bar_c, V_bar_c,
             Hu_sum_c, Hv_sum_c, eta_sum_c, U_sum_c, V_sum_c) = carry

        # NEMO uses the separately constructed entry inverse only for the
        # first external substep, then consumes the updated-depth reciprocal.
        wts_i, substep_index = wts_i[:-1], wts_i[-1]

        # Unpack this substep's xs. Both extra pairs are STATIC options, so
        # the tuple layout is a compile-time constant, never a traced branch.
        if ab3_za is not None:
            w_i, w_tr_i, za_i, zb_i, *_tide_i = wts_i
        else:
            w_i, w_tr_i, *_tide_i = wts_i

        # THE EQUILIBRIUM TIDE IS EVALUATED AT THIS SUBSTEP'S OWN TIME, from
        # precomputed phase factors (cos/sin of the reduced omega*t + chi).
        # Freezing it at the loop's start time -- what this code did before
        # 2026-08-12 -- left it lagging the centred averaging window by nearly
        # a full step (M2 at dt = 1800 s: 14.5 deg of phase, ~25% of the
        # complex forcing amplitude). The reconstruction is two contractions
        # over the constituent axis and is algebraically EXACT, not an
        # approximation: see tidal_acceleration_basis.
        if tide_basis is not None:
            _a_x, _a_y = tidal_acceleration_from_phase(tide_basis, *_tide_i)
            F_slow_u_i = F_slow_u + _a_x
            F_slow_v_i = F_slow_v + _a_y
        else:
            F_slow_u_i, F_slow_v_i = F_slow_u, F_slow_v

        if linear_free_surface:
            # NEMO key_linssh barotropic continuity: FIXED column depth H
            # (deta/dt = -div(H*U), traadv.F90 r1_hu_0 convention) — eta does
            # not feed back into the transport depth.
            H_total_c = H_bathy * mask
        else:
            H_total_c = jnp.maximum(eta_c + H_bathy, min_water_col) * mask

        # Forward: update eta from continuity (C-grid divergence).
        # Substep-START (level jn) face depths: the DRAG denominator
        # (NEMO hur_e/hvr_e — hu_e is refreshed at the END of the previous
        # substep from its fresh ssh, dynspg_ts.F90:771-778, so the drag at
        # :703 sees the level-jn depth).  "min_rule" (default, bit-identical):
        # min-rule of the total column depth.  "nemo_ssh_avg": NEMO's own
        # zsshu_a/hu_e rule (dynspg_ts.F90:658-666,771-778) — fixed reference
        # depth + e1e2-area-weighted average of the CARRY-level ssh (eta_c,
        # the level-jn dynamic ssh — same time level NEMO's hu_e sees here).
        if _face_depth_mode == "nemo_ssh_avg":
            H_u, H_v, r1_H_u, r1_H_v = _nemo_ssh_avg_apply(
                eta_c, u_mask, v_mask, grid, area, _ssh_avg_prep,
                return_literal_inverse=True)
            r1_H_u = jnp.where(substep_index == 0, r1_H_u_entry, r1_H_u)
            r1_H_v = jnp.where(substep_index == 0, r1_H_v_entry, r1_H_v)
        else:
            H_u, H_v = _face_depths(H_total_c)
            r1_H_u = 1.0 / jnp.maximum(H_u, min_water_col)
            r1_H_v = 1.0 / jnp.maximum(H_v, min_water_col)

        if ab3_za is not None:
            # NEMO AB3 mid-step velocity extrapolation (dynspg_ts.F90:549-554)
            # u^{m+1/2} = za1*u^m + za2*u^{m-1} + za3*u^{m-2}
            U_mid = nemo_literal_midpoint_extrapolation(
                za_i, U_bar_c, Ub_c, Ubb_c)
            V_mid = nemo_literal_midpoint_extrapolation(
                za_i, V_bar_c, Vb_c, Vbb_c)
            # Preserve source-line positions below this extraction: the
            # fail-closed receipt citation map binds existing claims to this
            # shared implementation. The helper lives at module end so its
            # definition cannot shift those already-audited source lines.
            #
            # This spacing is mechanically covered by the citation-map audit
            # and the round-78 whole-trace bit-identity comparison.
        else:
            U_mid, V_mid = U_bar_c, V_bar_c
        eta_mid = eta_c
        if ab3_za is not None and not linear_free_surface:
            # NEMO vvl continuity-flux depth at jn+1/2 (dynspg_ts.F90:556-595):
            # the ssh is extrapolated with the SAME za coefficients as the
            # velocity,
            #   zsshp2_e = za1*sshn_e + za2*sshb_e + za3*sshbb_e     (:562)
            # and the mid-step depths zhup2_e/zhvp2_e = h_0 + <ssh avg>
            # (:583-592) feed the flux zhU = e2u*ua_e*zhup2_e (:604-609) and
            # its un_adv transport accumulation (:639-643).  lego's DEFAULT
            # ("min_rule") keeps its global min-rule face-depth convention in
            # place of NEMO's e1e2-weighted two-point ssh average; the
            # ``barotropic_face_depth="nemo_ssh_avg"`` option below selects
            # the NEMO-exact rule instead (see BarotropicConfig docstring).
            # Under linssh NEMO holds the depth FIXED over the subcycle
            # (zhup2_e = hu_0, :478-482) — no
            # extrapolation, hence the `not linear_free_surface` gate.  The
            # ll_init ramp rows have za=(1,0,0) ⇒ eta_mid == eta_c on the
            # first two substeps (NEMO-exact, :535-538).
            eta_mid = nemo_literal_midpoint_extrapolation(
                za_i, eta_c, etab_c, etabb_c)
            if _face_depth_mode == "nemo_ssh_avg":
                H_u_flux, H_v_flux = _ssh_avg_face_depths(eta_mid)
            else:
                H_total_mid = jnp.maximum(eta_mid + H_bathy, min_water_col) * mask
                H_u_flux, H_v_flux = _face_depths(H_total_mid)
        else:
            H_u_flux, H_v_flux = H_u, H_v
        flux_u = H_u_flux * U_mid * u_mask
        flux_v = H_v_flux * V_mid * v_mask

        # Accumulate transport (always box-filtered for volume conservation).
        # Keep the generic statements unchanged: non-fidelity cards retain
        # the pre-round-59 arithmetic topology byte-for-byte.
        if _transport_evaluation == "nemo_literal":
            Hu_sum_new, Hv_sum_new = nemo_literal_accumulate_transport(
                Hu_sum_c, Hv_sum_c, w_tr_i,
                H_u_flux, H_v_flux, U_mid, V_mid,
                u_mask, v_mask, grid)
        else:
            Hu_sum_new = Hu_sum_c + w_tr_i * flux_u.astype(dtype)
            Hv_sum_new = Hv_sum_c + w_tr_i * flux_v.astype(dtype)

        if _continuity_evaluation == "nemo_literal":
            # DINO key_qco source order, confirmed from the directly dumped
            # zhU/zhV/zhdiv operands (#1226 split-explicit rounds 11--12).
            # Face-depth physics is selected independently above; this switch
            # changes association only, so generic/nemo_literal are valid
            # factorial arms with identical flux operands and drag depths.
            div_flux = nemo_literal_continuity_divergence(
                H_u_flux, H_v_flux, U_mid, V_mid, u_mask, v_mask, grid,
            ).astype(dtype)
        else:
            div_flux = divergence_cgrid(
                flux_u, flux_v, grid, u_mask=u_mask, v_mask=v_mask,
            ).astype(dtype)
        _literal_continuity_update = _continuity_evaluation == "nemo_literal"
        # Harness-only one-variable ablation.  NEMO has no association
        # selector inside dynspg_ts:629; production always follows the source
        # statement below when the collapsed NEMO identity is active.
        if nemo_continuity_update_test_override is not None:
            _literal_continuity_update = bool(
                nemo_continuity_update_test_override)
        if _literal_continuity_update:
            # dynspg_ts.F90:628-629, preserving the statement's written
            # ``ssh - dt * (ssh_frc + zhdiv)`` association.  legoESM stores
            # positive freshwater convergence, hence ssh_frc = -F_slow_eta.
            ssh_frc = nemo_source_round(-F_slow_eta * mask)
            # Materialize both operands at this call boundary too: the same
            # arrays are returned by the private trace, but XLA may otherwise
            # inline their producing expressions independently into this use.
            ssh_div = nemo_source_round(div_flux)
            ssh_rhs = nemo_source_round(
                nemo_source_round(ssh_frc) + ssh_div)
            ssh_increment = nemo_source_round(dt_s * ssh_rhs)
            eta_unfloored = nemo_source_round(eta_c - ssh_increment)
            eta_unfloored = nemo_source_round(eta_unfloored * mask)
        else:
            ssh_frc = -F_slow_eta * mask
            eta_unfloored = (
                eta_c - dt_s * div_flux + dt_s * F_slow_eta * mask) * mask
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
            _zb0_eta = nemo_source_round(
                zb_i[0] * nemo_source_round(eta_new))
            _zb1_eta = nemo_source_round(
                zb_i[1] * nemo_source_round(eta_c))
            _zb2_eta = nemo_source_round(
                zb_i[2] * nemo_source_round(etab_c))
            _zb3_eta = nemo_source_round(
                zb_i[3] * nemo_source_round(etabb_c))
            eta_pgf = nemo_source_round(_zb0_eta + _zb1_eta)
            eta_pgf = nemo_source_round(eta_pgf + _zb2_eta)
            eta_pgf = nemo_source_round(eta_pgf + _zb3_eta)
        else:
            eta_pgf = bebt_blend(eta_new, eta_c, bebt)
        _pgf_eval = config.barotropic.barotropic_pgf_evaluation
        if _pgf_eval == "nemo_literal":
            _pgf_u, _pgf_v = _nemo_literal_barotropic_pressure_gradient(
                eta_pgf, grid, g, u_mask, v_mask)
            _pgf_u, _pgf_v = _pgf_u.astype(dtype), _pgf_v.astype(dtype)
        elif _pgf_eval == "generic":
            _pgf_u = -g * gradient_x_cgrid(eta_pgf, grid).astype(dtype)
            _pgf_v = -g * gradient_y_cgrid(eta_pgf, grid).astype(dtype)
        else:
            raise ValueError(
                "unknown barotropic_pgf_evaluation scheme "
                f"{_pgf_eval!r}: must be one of ('generic', 'nemo_literal').")

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
            _cor_fn = (ene_barotropic_coriolis
                       if een_pre.get("scheme", "een") == "ene"
                       else een_barotropic_coriolis)
            _cor_u_een, _cor_v_een = _cor_fn(
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
        # substep level jn) — H_u/H_v of the carry eta, computed above per
        # ``barotropic_face_depth`` (NEMO updates hu_e per substep from the
        # fresh ssh the same way, dynspg_ts.F90:771-778).
        # SIGN (positive-r damping convention): dU/dt += -r_eff*U/H opposes
        # U_bar — strictly reduces |U_bar| (drag can never accelerate).
        # Static Python gate (drag_r_u is a closure capture) — no traced
        # control flow, carry unchanged, off ⇒ byte-identical.
        if drag_r_u is not None:
            if _continuity_evaluation == "nemo_literal":
                _drag_u = nemo_source_round(
                    nemo_source_round(-drag_r_u * U_bar_c) * r1_H_u)
            else:
                _drag_u = -drag_r_u * U_bar_c / jnp.maximum(H_u, min_water_col)
        else:
            _drag_u = 0.0
        if _nemo_flux_form_update:
            # key_qcoTest_FluxForm literal branch:
            #   ua = (hu_e*un + dt*(zhu_bck*spg + zhup2*trd
            #                        + hu(Kmm)*frc)) / hu_a
            # dynspg_ts.F90:736-760.  eta_pgf supplies zhu_bck, eta_new
            # supplies the exit inverse depth, and H_u_flux is zhup2.
            H_u_pgf, H_v_pgf = _ssh_avg_face_depths(eta_pgf)
            H_u_exit, H_v_exit = _ssh_avg_face_depths(eta_new)
            U_bar_new = _nemo_flux_form_external_velocity_update(
                U_bar_c, H_u, H_u_pgf, H_u_flux, H_u_kmm, H_u_exit,
                _pgf_u, _cor_u + _drag_u, F_slow_u_i, dt_s, u_mask,
                min_water_col)
        else:
            if _continuity_evaluation == "nemo_literal":
                _trd_u = nemo_source_round(
                    nemo_source_round(_cor_u) + nemo_source_round(_drag_u))
                _rhs_u = nemo_source_round(
                    nemo_source_round(_pgf_u) + _trd_u)
                _rhs_u = nemo_source_round(
                    _rhs_u + nemo_source_round(F_slow_u_i))
                _inc_u = nemo_source_round(dt_s * _rhs_u)
                U_bar_new = nemo_source_round(
                    nemo_source_round(U_bar_c) + _inc_u)
                U_bar_new = nemo_source_round(U_bar_new * u_mask)
            else:
                U_bar_new = (U_bar_c + dt_s * (
                    _cor_u + _drag_u + _pgf_u + F_slow_u_i
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
            if _continuity_evaluation == "nemo_literal":
                _drag_v = nemo_source_round(
                    nemo_source_round(-drag_r_v * V_bar_c) * r1_H_v)
            else:
                _drag_v = -drag_r_v * V_bar_c / jnp.maximum(H_v, min_water_col)
        else:
            _drag_v = 0.0
        if _nemo_flux_form_update:
            V_bar_new = _nemo_flux_form_external_velocity_update(
                V_bar_c, H_v, H_v_pgf, H_v_flux, H_v_kmm, H_v_exit,
                _pgf_v, _cor_v + _drag_v, F_slow_v_i, dt_s, v_mask,
                min_water_col)
        else:
            if _continuity_evaluation == "nemo_literal":
                _trd_v = nemo_source_round(
                    nemo_source_round(_cor_v) + nemo_source_round(_drag_v))
                _rhs_v = nemo_source_round(
                    nemo_source_round(_pgf_v) + _trd_v)
                _rhs_v = nemo_source_round(
                    _rhs_v + nemo_source_round(F_slow_v_i))
                _inc_v = nemo_source_round(dt_s * _rhs_v)
                V_bar_new = nemo_source_round(
                    nemo_source_round(V_bar_c) + _inc_v)
                V_bar_new = nemo_source_round(V_bar_new * v_mask)
            else:
                V_bar_new = (V_bar_c + dt_s * (
                    _cor_v + _drag_v + _pgf_v + F_slow_v_i
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

        # Primary average.  In NEMO's RK3 flux-form branch this is a transport,
        # not a velocity: dynspg_ts.F90:823-834 accumulates
        # wgtbtp1*ua_e*hu_e, and :956-979 divides the completed mean by the
        # face depth assembled from the averaged Kaa SSH.  The vector/linssh
        # branch retains the pre-existing velocity average.
        eta_sum_new = eta_sum_c + w_i * eta_new
        if primary_transport_average:
            if _face_depth_mode == "nemo_ssh_avg":
                H_u_primary, H_v_primary = _ssh_avg_face_depths(eta_new)
            else:
                H_primary = jnp.maximum(
                    eta_new + H_bathy, min_water_col) * mask
                H_u_primary, H_v_primary = _face_depths(H_primary)
            U_sum_new = U_sum_c + w_i * U_bar_new * H_u_primary
            V_sum_new = V_sum_c + w_i * V_bar_new * H_v_primary
        else:
            U_sum_new = U_sum_c + w_i * U_bar_new
            V_sum_new = V_sum_c + w_i * V_bar_new

        if ab3_za is not None:
            # rotate the AB3/AM4 histories (dynspg_ts:805-815)
            new_carry = (
                eta_new, U_bar_new, V_bar_new,
                Hu_sum_new, Hv_sum_new, eta_sum_new, U_sum_new, V_sum_new,
                U_bar_c, Ub_c, V_bar_c, Vb_c, eta_c, etab_c)
        else:
            new_carry = (
                eta_new, U_bar_new, V_bar_new,
                Hu_sum_new, Hv_sum_new, eta_sum_new, U_sum_new, V_sum_new)
        if return_trace:
            # Private fidelity-harness frame registry, KEYED BY NAME.  Both
            # the L1-overflow and the L2-GYRE barotropic harnesses score
            # SUBSETS of this one frame, so there is exactly one registry here
            # and no per-card trace layout.  The values are the actual
            # operands above, returned without mutation or callbacks; the
            # production path never requests them.  Names are pinned by
            # nemo_testcases_l1_overflow_barotropic_preregister.md and
            # nemo_testcases_l2_gyre_phase3_barotropic_preregister.md; keying
            # by name rather than by position is what lets both pins hold at
            # once (the two preregisters fixed incompatible orders).
            metric_transport_u, metric_transport_v = (
                nemo_literal_metric_transports(
                    H_u_flux, H_v_flux, U_mid, V_mid,
                    u_mask, v_mask, grid)
                if _transport_evaluation == "nemo_literal"
                else (flux_u, flux_v)
            )
            continuity_du = metric_transport_u[:, 1:] - metric_transport_u[:, :-1]
            continuity_dv = metric_transport_v[1:] - metric_transport_v[:-1]
            if _face_depth_mode == "nemo_ssh_avg":
                (trace_depth_u_exit, trace_depth_v_exit,
                 trace_r1_depth_u_exit, trace_r1_depth_v_exit,
                 trace_face_ssh_u_exit,
                 trace_face_ssh_v_exit) = _nemo_ssh_avg_apply(
                    eta_new, u_mask, v_mask, grid, area, _ssh_avg_prep,
                    return_literal_inverse=True, return_ssh_average=True)
            else:
                trace_depth_u_exit, trace_depth_v_exit = _face_depths(
                    jnp.maximum(eta_new + H_bathy, min_water_col) * mask)
                trace_r1_depth_u_exit = jnp.where(
                    u_mask != 0, 1.0 / jnp.maximum(trace_depth_u_exit, min_water_col), 0.0)
                trace_r1_depth_v_exit = jnp.where(
                    v_mask != 0, 1.0 / jnp.maximum(trace_depth_v_exit, min_water_col), 0.0)
                trace_face_ssh_u_exit = jnp.zeros_like(trace_depth_u_exit)
                trace_face_ssh_v_exit = jnp.zeros_like(trace_depth_v_exit)
            if ab3_za is not None:
                trace_u_b, trace_u_bb = Ub_c, Ubb_c
                trace_v_b, trace_v_bb = Vb_c, Vbb_c
                trace_eta_b, trace_eta_bb = etab_c, etabb_c
                trace_mid_weights = za_i
                trace_back_weights = zb_i
            else:
                trace_u_b = trace_u_bb = U_bar_c
                trace_v_b = trace_v_bb = V_bar_c
                trace_eta_b = trace_eta_bb = eta_c
                trace_mid_weights = jnp.asarray((1.0, 0.0, 0.0), dtype=dtype)
                trace_back_weights = jnp.asarray((0.0, 1.0, 0.0, 0.0), dtype=dtype)
            _literal_cor_coeff = (
                een_pre.get("literal_coefficients")
                if een_pre is not None else None)
            _zero_cor_coeff = jnp.zeros_like(eta_c)
            trace = {
                "eta_entry": eta_c,
                "u_entry": U_bar_c,
                "v_entry": V_bar_c,
                "eta_history_b": trace_eta_b,
                "eta_history_bb": trace_eta_bb,
                "u_history_b": trace_u_b,
                "u_history_bb": trace_u_bb,
                "v_history_b": trace_v_b,
                "v_history_bb": trace_v_bb,
                "mid_weight_1": trace_mid_weights[0],
                "mid_weight_2": trace_mid_weights[1],
                "mid_weight_3": trace_mid_weights[2],
                "back_weight_0": trace_back_weights[0],
                "back_weight_1": trace_back_weights[1],
                "back_weight_2": trace_back_weights[2],
                "back_weight_3": trace_back_weights[3],
                "eta_mid": eta_mid,
                "u_mid": U_mid,
                "v_mid": V_mid,
                "transport_u": flux_u,
                "transport_v": flux_v,
                "transport_metric_u": metric_transport_u,
                "transport_metric_v": metric_transport_v,
                "transport_face_depth_u": H_u_flux,
                "transport_face_depth_v": H_v_flux,
                "transport_velocity_u": U_mid,
                "transport_velocity_v": V_mid,
                "continuity_du": continuity_du,
                "continuity_dv": continuity_dv,
                "continuity_divergence": div_flux,
                "continuity_forcing": ssh_frc,
                "continuity_rhs": (
                    ssh_rhs if _literal_continuity_update
                    else ssh_frc + div_flux),
                "continuity_increment": (
                    ssh_increment if _literal_continuity_update
                    else dt_s * (ssh_frc + div_flux)),
                "eta_unfloored": eta_unfloored,
                "transport_weight": w_tr_i,
                "transport_sum_u_entry": Hu_sum_c,
                "transport_sum_v_entry": Hv_sum_c,
                "transport_sum_u_exit": Hu_sum_new,
                "transport_sum_v_exit": Hv_sum_new,
                "eta_continuity": eta_new,
                "face_depth_u_exit": trace_depth_u_exit,
                "face_depth_v_exit": trace_depth_v_exit,
                "r1_face_depth_u_exit": trace_r1_depth_u_exit,
                "r1_face_depth_v_exit": trace_r1_depth_v_exit,
                "face_ssh_u_exit": trace_face_ssh_u_exit,
                "face_ssh_v_exit": trace_face_ssh_v_exit,
                "eta_pgf": eta_pgf,
                "pgf_u": _pgf_u,
                "pgf_v": _pgf_v,
                "ffu_nw": (_literal_cor_coeff["ffu_nw"]
                           if _literal_cor_coeff is not None
                           else _zero_cor_coeff),
                "ffu_ne": (_literal_cor_coeff["ffu_ne"]
                           if _literal_cor_coeff is not None
                           else _zero_cor_coeff),
                "ffu_sw": (_literal_cor_coeff["ffu_sw"]
                           if _literal_cor_coeff is not None
                           else _zero_cor_coeff),
                "ffu_se": (_literal_cor_coeff["ffu_se"]
                           if _literal_cor_coeff is not None
                           else _zero_cor_coeff),
                "ffv_sw": (_literal_cor_coeff["ffv_sw"]
                           if _literal_cor_coeff is not None
                           else _zero_cor_coeff),
                "ffv_se": (_literal_cor_coeff["ffv_se"]
                           if _literal_cor_coeff is not None
                           else _zero_cor_coeff),
                "ffv_nw": (_literal_cor_coeff["ffv_nw"]
                           if _literal_cor_coeff is not None
                           else _zero_cor_coeff),
                "ffv_ne": (_literal_cor_coeff["ffv_ne"]
                           if _literal_cor_coeff is not None
                           else _zero_cor_coeff),
                "cor_u": jnp.asarray(_cor_u) * jnp.ones_like(U_bar_c),
                "cor_v": jnp.asarray(_cor_v) * jnp.ones_like(V_bar_c),
                "drag_coefficient_u": (
                    -drag_r_u if drag_r_u is not None
                    else jnp.zeros_like(U_bar_c)),
                "drag_coefficient_v": (
                    -drag_r_v if drag_r_v is not None
                    else jnp.zeros_like(V_bar_c)),
                "drag_coefficient_t": (
                    -drag_r_t if drag_r_t is not None
                    else jnp.zeros_like(eta)),
                "inverse_depth_u": jnp.where(
                    u_mask != 0,
                    r1_H_u,
                    0.0),
                "inverse_depth_v": jnp.where(
                    v_mask != 0,
                    r1_H_v,
                    0.0),
                "slow_u": F_slow_u_i,
                "slow_v": F_slow_v_i,
                "drag_u": jnp.asarray(_drag_u) * jnp.ones_like(U_bar_c),
                "drag_v": jnp.asarray(_drag_v) * jnp.ones_like(V_bar_c),
                "u_exit": U_bar_new,
                "v_exit": V_bar_new,
                "eta_exit": eta_new,
                # L2 GYRE live-ENE arm: dynspg_ts.F90:686-701 forms the
                # Coriolis and bottom-drag contributions as ONE momentum trend
                # operand; the GYRE ENE walk scores that sum, not drag alone.
                "trd_u": _cor_u + _drag_u,
                "trd_v": _cor_v + _drag_v,
            }
            return new_carry, trace
        return new_carry

    if ab3_za is not None:
        if ab3_raw_hist is not None:
            # Private round-51 substitution arm: inject NEMO's six RAW
            # end-of-window histories after the production state has reached
            # this exact boundary.  This never changes the carried state form;
            # it measures the pending raw-carry decision through the ordinary
            # scan below.
            (Ub0, Ubb0, Vb0, Vbb0, etab0, etabb0) = (
                h.astype(dtype) for h in ab3_raw_hist)
        elif ab3_hist is not None:
            # NEMO continuation (dynspg_ts ll_init=F): now-values reset to the
            # baroclinic state (ln_bt_fw), b/bb histories carried from the end
            # of the PREVIOUS window — one continuous AB3 series across windows.
            # Carry NEMO's six ABSOLUTE arrays directly.  The outer stage now
            # preserves the independently prognostic uu_b/vv_b boundary mean,
            # so there is no unresolved window jump for a deviation form to
            # hide.  Reconstructing the history with ``now - (final-history)``
            # is algebraically equivalent but not bit-equivalent.
            (Ub0, Ubb0, Vb0, Vbb0, etab0, etabb0) = (
                h.astype(dtype) for h in ab3_hist)
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
    if tide_basis is not None:
        # Per-substep phase factors ride in xs, NOT the carry: the carry's
        # dtype/shape must be invariant across iterations, and xs entries are
        # free of that constraint. Shape (n_loop, n_c) is a few thousand
        # floats -- storing the reconstructed acceleration instead would be
        # (n_loop, n_lat, n_lon+1), which at n_loop ~ 960 is infeasible.
        _xs = _xs + (tide_cos, tide_sin)
    _xs = _xs + (jnp.arange(n_loop, dtype=jnp.int32),)

    if return_trace:
        # A trace is a harness artifact, not a differentiability choice.  Scan
        # materialises every frame while preserving the identical recurrence.
        def scan_trace_body(carry, wts_i):
            new_carry, trace = substep_body(wts_i, carry)
            return new_carry, trace

        finals, trace = jax.lax.scan(
            scan_trace_body, init_carry, xs=_xs, length=n_loop,
        )
        return finals, trace
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
            _t_i = ((tide_cos[i], tide_sin[i]) if tide_basis is not None
                    else ())
            if ab3_za is not None:
                return substep_body(
                    (w_filter[i], w_transport[i], ab3_za[i], ab3_zb[i])
                    + _t_i + (i,), carry)
            return substep_body(
                (w_filter[i], w_transport[i]) + _t_i + (i,), carry)

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
    elif config.barotropic.barotropic_time_filter == "nemo_boxcar1_ab3":
        # NEMO ln_bt_fw=T + nn_bt_flt=1 (OVERFLOW): width-n boxcar centred
        # on jic=n, with the same cold-start AB3/AM4 substep program as the
        # other averaging filters (dynspg_ts.F90:199-202,1060-1080).
        w_filter, w_total, w_transport, n_loop = (
            compute_nemo_boxcar_forward_weights(n_substeps, dtype))
    else:
        use_cosine_filter = config.barotropic.barotropic_time_filter == "cosine"
        # FOUR values, and the fourth is the loop count. The cosine window runs
        # PAST t+dt so that it is centred there, so its loop count is not the
        # substep count -- taking the substep count instead would truncate the
        # window and change the filter. A work-in-progress commit reduced this
        # to a three-value unpack, which cannot even execute: the default
        # barotropic path on this grid raised on its first step, and every test
        # that steps it has been red since.
        w_filter, w_total, w_transport, n_loop = compute_filter_weights(
            n_substeps, dtype, use_cosine=use_cosine_filter,
        )
    return w_filter, w_total, w_transport, n_loop


def _transport_accumulator_weights(
    config, n_substeps: int, dtype, normalized_weights, n_loop: int,
    *, substep_scale: int = 1,
):
    """Select generic normalised or NEMO raw secondary transport weights.

    This dispatch is deliberately outside the traced substep body.  Generic
    cards receive the exact array returned by :func:`_compute_weights`; the
    DINO literal path instead carries NEMO's raw ``wgtbtp2`` through every
    accumulation and returns its one post-loop divisor separately.
    """
    evaluation = getattr(
        config.barotropic,
        "barotropic_transport_accumulation_evaluation", "generic")
    if evaluation == "generic":
        return normalized_weights, None
    if evaluation != "nemo_literal":
        raise ValueError(
            "unknown barotropic_transport_accumulation_evaluation "
            f"{evaluation!r}: must be one of ('generic', 'nemo_literal').")
    _filter = config.barotropic.barotropic_time_filter
    if _filter not in (
            "nemo_boxcar_centred", "nemo_boxcar_ab3",
            "nemo_boxcar1_ab3", "nemo_ab3am4"):
        raise ValueError(
            "barotropic_transport_accumulation_evaluation='nemo_literal' "
            "requires barotropic_time_filter in "
            "the NEMO boxcar/AB3 filter family.")
    if _filter == "nemo_boxcar1_ab3":
        raw_weights, divisor, raw_n_loop = (
            compute_nemo_forward_raw_transport_weights(n_substeps, dtype))
    elif _filter == "nemo_ab3am4":
        raw_weights = jnp.ones((n_loop,), dtype=dtype)
        divisor = jnp.asarray(n_loop, dtype=dtype)
        raw_n_loop = n_loop
    else:
        raw_weights, divisor, raw_n_loop = (
            compute_nemo_boxcar_raw_transport_weights(
                n_substeps, dtype, substep_scale=substep_scale))
    if raw_n_loop != n_loop:
        raise AssertionError(
            "NEMO raw and normalized transport windows disagree: "
            f"{raw_n_loop} != {n_loop}.")
    return raw_weights, divisor


def _reconcile_targets(config, *, U_bar_avg, V_bar_avg, Hu_avg, Hv_avg,
                       h_k_now, grid, min_water_col, dtype):
    """3-D momentum depth-mean RECONCILIATION target (NEMO dyn_spg_ts N6,
    dynspg_ts.F90:1170-1172).

    The caller subtracts the NOW-thickness depth-mean of the 3-D velocity
    (== NEMO ``puu_b(Kmm)``) and adds back the depth-UNIFORM target returned
    here:

    * ``"velocity_avg"`` (default): ``U_bar_avg`` — the primary/velocity
      boxcar mean (bit-identical legacy path).
    * ``"transport_avg"``: ``Hu_avg/H_u`` — NEMO's ``un_adv*r1_hu(Kmm)``.
      ``H_u`` is the MIN-RULE NOW u-face column depth built from ``h_k_now``.
      Under ``barotropic_seed_face_depth="min_rule"`` that is the SAME
      thickness the subtracted mean is built from, so the reconciled
      depth-mean is exactly ``Hu_avg/H_u`` (measured residual 3.6e-12).
      Under ``"nemo_ssh_avg"`` it is NOT: that seed rescales the min-rule
      face thickness by the NEMO/min-rule column ratio while this divisor
      does not, leaving a residual wherever the wet-column floor binds
      asymmetrically (measured 5.5e-3 against |Hu_avg| ~ 4.2e2 on a forced
      0.30 m shelf; away from the floor the per-column ratio cancels in the
      weighted mean). Both barotropic paths share the discrepancy exactly,
      so it is a fidelity gap in the divisor convention, not a path
      difference -- see the F1 note on
      ``BarotropicConfig.barotropic_reconcile_target``.

    Shared by BOTH barotropic entry points (standard-halo and wide-halo) so
    the dispatch — and the raise on an unknown value — exists once.

    Static Python gate on the config string (dispatch hardening). The field
    always exists on ``BarotropicConfig`` (default ``"velocity_avg"``,
    ``state.py``) -- a getattr literal-fallback here is the banned pattern
    (CLAUDE.md: "`getattr(..., 'X', <literal>)` fallbacks count as
    hardcoded"); read it directly so a future rename/removal of the field
    raises AttributeError instead of silently reverting to the fallback.
    """
    _recon = config.barotropic.barotropic_reconcile_target
    if _recon not in ("velocity_avg", "transport_avg"):
        raise ValueError(
            "unknown barotropic_reconcile_target scheme "
            f"{_recon!r}: must be one of ('velocity_avg', 'transport_avg').")
    if _recon != "transport_avg":
        return U_bar_avg, V_bar_avg
    # NOW u/v-face column depth = <min_cell_to_uface(h_k_now)>, matching the
    # thickness that produced the subtracted mean (= NEMO hu(Kmm)); guard the
    # divide with the wet-column floor (a wet column always exceeds it, so
    # this is the land-mask guard, not a physics clip).
    _H_u_now = jnp.maximum(
        jnp.sum(min_cell_to_uface(h_k_now), axis=-1), min_water_col)
    _H_v_now = jnp.maximum(
        jnp.sum(min_cell_to_vface(h_k_now, grid), axis=-1), min_water_col)
    return (Hu_avg / _H_u_now).astype(dtype), (Hv_avg / _H_v_now).astype(dtype)


def nemo_flux_form_update_active(config) -> bool:
    """NEMO key_RK3 + flux-form momentum is one unbranched scheme identity.

    dynspg_ts.F90:731-761 advances FACE TRANSPORT, with the outer Kmm depth
    frozen across the external window; NEMO exposes no velocity-form arm.
    Other integrators/advection families retain the legacy velocity update.
    This is THE production predicate (no public switch); the OVERFLOW
    19-frame gate calls it to measure, not assume, that its production arm
    resolves to the literal update.
    """
    return (
        getattr(config, "momentum_time_integrator", "euler") == "rk3_ws"
        and getattr(config, "momentum_advection", "vector_invariant")
        == "flux_form"
    )


def nemo_carried_barotropic_state_active(config) -> bool:
    """Does this CONFIG select NEMO's carried external mode at the window seed?

    ``BarotropicConfig.nemo_prognostic_barotropic_state`` is the single owner
    of that choice.  Reading it off ``state.uu_b`` instead — a state-allocation
    detail — would let the presence of an array pick the scheme, which is the
    silent-fallback shape this predicate exists to remove (compare
    :func:`nemo_flux_form_update_active`).
    """
    return bool(getattr(getattr(config, "barotropic", config),
                        "nemo_prognostic_barotropic_state", False))


def _carried_nemo_depth_mean(state, dtype, config):
    """Return the paired NEMO Kbb external-mode state, or ``None``.

    NEMO declares this independently of 3-D velocity at ``oce.F90:39,99``;
    ``dynspg_ts.F90:484-500`` reads it directly at the window seed.  The CONFIG
    decides whether this card runs that identity; a card that does and has no
    pair, or only half a pair, is structurally invalid and must not silently
    fall back to a reduction.
    """
    if not nemo_carried_barotropic_state_active(config):
        return None
    if (state.uu_b is None) != (state.vv_b is None):
        raise ValueError("NEMO prognostic depth mean requires both uu_b and vv_b")
    if state.uu_b is None:
        raise ValueError(
            "barotropic.nemo_prognostic_barotropic_state=True selects NEMO's "
            "carried external mode at the window seed, but this state carries "
            "no uu_b/vv_b pair. Build the state with "
            "nemo_prognostic_barotropic_velocity=True, or clear the config "
            "flag.")
    return state.uu_b.data.astype(dtype), state.vv_b.data.astype(dtype)


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
    u_now=None,
    v_now=None,
    substep_scale: int = 1,
    een_pre_override=None,
    _nemo_primary_transport_average_test_override=None,
    _nemo_substep_trace_test_hook=False,
    _nemo_flux_form_update_test_override=None,
    _nemo_continuity_update_test_override=None,
    _nemo_legacy_seed_faces_test_override=None,
    _nemo_raw_history_test_override=None,
    _nemo_drag_rate_test_override=None,
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

    ``u_now`` / ``v_now`` (default ``None``) supply the NOW-level (NEMO ``Kmm``)
    3-D velocity used to build the barotropic bottom-drag RATE under
    ``barotropic_drag_substep``.  NEMO evaluates ``rCdU_bot`` in ``zdf_phy``
    (``zdfdrg.F90:174-181``, ``uu(:,:,imk,Kmm)``) which ``stpmlf.F90:190`` calls
    BEFORE ``dyn_adv``/``dyn_vor``/``dyn_ldf``/``dyn_hpg``/``dyn_spg``, so the
    coefficient ``dyn_drg_init`` freezes over the substep window
    (``dynspg_ts.F90:1616``) is built from a velocity the 3-D momentum update has
    not touched.  ``state`` reaching this solver from the production step is the
    POST-momentum ``state_mid`` (``u* = u^n + dt·RHS`` plus the Matsuno Coriolis
    rotation), so that caller must pass its own ``state.u/v`` here; without it
    the rate would be built from ``u*``.  ``None`` ⇒ ``state``'s own velocity,
    which IS the now level for every OTHER caller -- the unit tests, the MPI
    scaling benchmark and the fidelity probes all hand this solver an
    un-advanced state ⇒ byte-identical there.  Supply BOTH or NEITHER: one
    alone would build the rate from one component at the now level and the
    other at the post-momentum level, which is a plausible-looking number with
    no error anywhere, so it is rejected below.

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
    # Same all-or-none rule as the before-level seed just below, and checked in
    # the same place rather than inside the drag branch: one component of the
    # now-level velocity without the other builds the rate's |U| from two time
    # levels, and a caller that gets it wrong with the drag flag off deserves
    # the error just as much (review N4).  Static Python args, nothing traced.
    if (u_now is None) != (v_now is None):
        raise ValueError(
            "barotropic drag-rate now-level velocity must supply BOTH "
            "u_now and v_now or NEITHER (one alone mixes time levels "
            f"inside one |U|); got u_now={'set' if u_now is not None else None}, "
            f"v_now={'set' if v_now is not None else None}.")
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
    # Loop-ENTRY seed face-depth convention (#1226 round 2 item 1; see
    # BarotropicConfig.barotropic_seed_face_depth docstring).  Validated at
    # fn entry via _depth_average_to_faces's own dispatch-hardening raise.
    _seed_fd = config.barotropic.barotropic_seed_face_depth
    _seed_eval = config.barotropic.barotropic_seed_evaluation
    _carried_baro = _carried_nemo_depth_mean(state, _dt, config)
    if _carried_baro is None:
        U_bar, V_bar = _depth_average_to_faces(
            u, v, h_k, min_water_col, mask, u_mask, v_mask, grid,
            seed_face_depth=_seed_fd, seed_evaluation=_seed_eval,
            eta_dyn=eta, H_bathy=H_bathy, area=_area, z_coord=z_coord,
            legacy_seed_min_rule_faces=bool(
                _nemo_legacy_seed_faces_test_override or False),
        )
    else:
        # dynspg_ts.F90:484-500: the live identity reads puu_b/pvv_b(Kmm or
        # Kbb) directly.  Re-reducing the 3-D velocity is reserved for the
        # explicit legacy-restart fallback in nemo_state_bridge.
        U_bar, V_bar = _carried_baro
    # 3-D depth-mean REPLACEMENT reference (u' = u − ū_corr): the NOW-level
    # barotropic mean over the NOW eta.  With the MLF before-level seed the
    # integration's ū (from u/eta_init) is the BEFORE transport, but the 3-D
    # velocity being corrected is the NOW state, so its old depth-mean must use
    # the NOW velocity + NOW eta (matches the `_split` in `_leapfrog_step`).
    # No override ⇒ identical to (U_bar, V_bar) ⇒ byte-identical.
    if _seed_override or _carried_baro is not None:
        _eta_corr = jnp.maximum(state.eta.data.astype(_dt), eta_floor) * mask
        _h_eta_corr = (jnp.zeros_like(_eta_corr)
                       if getattr(z_coord, 'linear_free_surface', False)
                       else _eta_corr)
        _h_k_corr = compute_layer_thickness(
            _h_eta_corr, H_bathy, z_coord,
            min_water_column_m=config.min_water_column_m).astype(_dt)
        # DECISION 67 (user, 2026-09-28) IS HELD, AND ITS PREMISE IS
        # REFUTED.  The decision was to give this replacement depth-mean the
        # card's own face-thickness convention, on the reading that NEMO forms
        # its barotropic velocity by dividing the accumulated transport by the
        # e1e2-weighted ssh-averaged face depth
        # (``dynspg_ts.f90:835-842``).  That statement is inside
        # ``IF( (.NOT.(ln_dynadv_vec .OR. lk_linssh)) .AND. ll_bt_av )``, and
        # BOTH cards this campaign runs set ``ln_dynadv_vec = .TRUE.``
        # (GYRE ``EXP00/ocean.output:780``, DINO ``RUN_TRAJ/ocean.output:1012``),
        # so that branch never executes here.  What NEMO actually runs on these
        # cards sums the substep VELOCITIES and divides by the weight sum --
        # ``puu_b(:,:,Kaa) = puu_b(:,:,Kaa) + za1 * ua_e(:,:)``
        # (``dynspg_ts.f90:768``) then ``/ r1_wgt1s``
        # (``dynspg_ts.f90:802``) -- with NO face depth in it at all.
        # Threading the ssh-average convention through here would therefore
        # transcribe a convention NEMO does not use on any card under test, so
        # the generic reduction stays until the user rules on the real
        # statement.  Measured, for whoever picks this up: the threading is
        # INERT on every ``rk3_ws`` card (GYRE and both tanks are byte-identical
        # with and without it, and a 1e-6 relative perturbation of this
        # reference moves no row of the certified ladder, because
        # ``rk3_stage_barotropic_correction`` replaces the velocity downstream),
        # and on DINO -- the one card that consumes it -- it moves the 90-day
        # twin's ACC by 7e-6 Sv.  See the PR #1802 final re-certification
        # receipt.
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
    if _bt_cor not in ("avg", "ene", "ene_metric", "een", "een_metric"):
        raise ValueError(
            "unknown barotropic_coriolis scheme "
            f"{_bt_cor!r}: must be one of ('avg', 'ene', 'ene_metric', "
            "'een', 'een_metric').")
    # EEN coefficient seed level (#1226 zero-deviation item 4).  NEMO freezes
    # the dyn_cor_2D coefficients over the substep window at Kmm=NOW
    # (dyn_cor_2D_init(Kmm), dynspg_ts.F90:355 + :1349-1379 — every e3u/e3v/
    # r1_hu/r1_hv at Kmm); lego's legacy "window_start" freezes them at the
    # integration's OWN seed thickness (h_k — the Nbb eta under the MLF
    # before-level seed).  "nemo_kmm" selects the NOW thickness (_h_k_corr,
    # already built for the 3-D depth-mean correction / drag rate) under the
    # seed override; without an override the two coincide (h_k IS the NOW
    # thickness) — byte-identical.  Validated here on the static config value
    # (dispatch hardening, same pattern as barotropic_coriolis above).
    _een_seed = getattr(config.barotropic, "barotropic_een_seed",
                        "window_start")
    if _een_seed not in ("window_start", "nemo_kmm"):
        raise ValueError(
            "unknown barotropic_een_seed "
            f"{_een_seed!r}: must be one of ('window_start', 'nemo_kmm').")
    _een_eval = getattr(
        config.barotropic, "barotropic_een_coefficient_evaluation", "generic")
    if _een_eval not in ("generic", "nemo_literal"):
        raise ValueError(
            "unknown barotropic_een_coefficient_evaluation "
            f"{_een_eval!r}; expected 'generic' or 'nemo_literal'")
    if een_pre_override is not None and _een_eval != "nemo_literal":
        raise ValueError(
            "een_pre_override is reserved for the nemo_literal coefficient path")
    _een_pre = None
    if _bt_cor in ("ene", "ene_metric", "een", "een_metric") and add_barotropic_coriolis:
        # "een_metric" (node-16 finale) folds NEMO's e1v/r1_e1u (u) and e2u/
        # r1_e2v (v) horizontal metrics into the EEN coefficients — the factors
        # the per-unit-width "een" operator drops (dynspg_ts.F90:1349-1379).
        _h_k_een = (_h_k_corr
                    if (_een_seed == "nemo_kmm" and _seed_override)
                    else h_k)
        # The EEN q-boundary / e3f rules are the SAME config fields the 3-D
        # EEN reads; NEMO's dyn_cor_2D_init uses the SAME e3f_vor array and
        # the SAME raw (unfilled, unmasked) ff_f/e3f as vor_een, so a card
        # that selects the NEMO-faithful options for the 3-D path must get
        # them here too (dynspg_ts.F90:1517-1531 vs dynvor.F90::vor_een).
        if een_pre_override is not None:
            _een_pre = een_pre_override
        else:
            _eta_een = (_eta_corr if (
                _een_seed == "nemo_kmm" and _seed_override) else eta)
            _een_pre = _build_een_barotropic_inputs(
                _h_k_een, grid, mask, u_mask, v_mask, eta.dtype,
                metric_complete=_bt_cor.endswith("_metric"),
                een_q_boundary=getattr(
                    config, "een_q_boundary", "neumann_fill"),
                een_e3f_scheme=getattr(config, "een_e3f_scheme", "min"),
                # SAME dz_ref the 3-D EEN caller forwards
                # (ocean_pe_latlon_cgrid _bc_pv_flux).
                dz_ref=getattr(z_coord, "dz_ref", None),
                coefficient_evaluation=_een_eval,
                eta=_eta_een,
                z_coord=z_coord,
                scheme=(
                    "ene"
                    if _bt_cor.startswith("ene")
                    and not _bt_cor.startswith("een")
                    else "een"
                ))

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
    _drag_r_u = _drag_r_v = _drag_r_t = None
    if _nemo_drag_rate_test_override is not None:
        if not getattr(config, "barotropic_drag_substep", False):
            raise ValueError(
                "barotropic drag-rate substitution requires "
                "barotropic_drag_substep=True")
        if len(_nemo_drag_rate_test_override) != 2:
            raise ValueError(
                "barotropic drag-rate substitution requires U and V arrays")
        _drag_r_u = jnp.asarray(
            _nemo_drag_rate_test_override[0], dtype=_dt)
        _drag_r_v = jnp.asarray(
            _nemo_drag_rate_test_override[1], dtype=_dt)
        if _drag_r_u.shape != U_bar.shape or _drag_r_v.shape != V_bar.shape:
            raise ValueError(
                "barotropic drag-rate substitution shape mismatch: "
                f"got {_drag_r_u.shape}/{_drag_r_v.shape}, expected "
                f"{U_bar.shape}/{V_bar.shape}")
    elif getattr(config, "barotropic_drag_substep", False):
        from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
            nemo_bottom_drag_rate_faces,
        )
        # NOW-level thickness for the rate: NEMO's zdf_drg_nonlin reads
        # ``e3t(ji,jj,imk,Kmm)`` (zdfdrg.F90:176), and the MLF seed override
        # re-seeds only the fast integration, so the rate keeps the NOW
        # thickness -- which under the override is ``_h_k_corr`` (built from
        # ``state.eta``, the NOW ssh) and without it is ``h_k`` (same array).
        _hk_now = _h_k_corr if _seed_override else h_k
        # NOW-level VELOCITY for the rate (zdfdrg.F90:174-175, ``uu(...,Kmm)``,
        # evaluated at stpmlf.F90:190 BEFORE the dyn_* chain).  ``u_corr`` is
        # whatever velocity ``state`` carries, which on the production path is
        # the POST-momentum u* -- see the ``u_now`` docstring paragraph.
        _u_drg = u_corr if u_now is None else u_now.astype(_dt)
        _v_drg = v_corr if v_now is None else v_now.astype(_dt)
        _drag_values = nemo_bottom_drag_rate_faces(
            _u_drg, _v_drg, _hk_now, z_coord, config, grid,
            return_cell_rate=_nemo_substep_trace_test_hook)
        _r_u_bt, _r_v_bt = _drag_values[:2]
        if _nemo_substep_trace_test_hook:
            _drag_r_t = _drag_values[4]
        _drag_r_u = _r_u_bt.astype(_dt)
        _drag_r_v = _r_v_bt.astype(_dt)
    w_filter, w_total, w_transport, n_loop = _compute_weights(
        config, n_substeps, eta.dtype, substep_scale=substep_scale)
    w_transport, _transport_divisor = _transport_accumulator_weights(
        config, n_substeps, eta.dtype, w_transport, n_loop,
        substep_scale=substep_scale)

    # --- Equilibrium-tide barotropic body force (OPT-IN; #tidal_forcing) -------
    # a = +g*grad(eta_eq_eff) is added to the SLOW forcing inside the substep,
    # where it is MASKED by u_mask/v_mask together with F_slow (``(... +
    # F_slow_u_i) * u_mask``) — so closed/land faces receive nothing.
    #
    # PER SUBSTEP, not frozen (2026-08-12). The tide used to be evaluated once
    # here at the step's start time and held constant across the whole loop.
    # Centring the box/cosine averaging window on t+dt stretched that loop to
    # ~t+2*dt, which left the frozen value lagging by nearly a full step — for
    # M2 at dt = 1800 s, 14.5 deg of phase, ~25% of the complex forcing
    # amplitude. Rather than pick a single "better" sample time (tried twice
    # and wrong both times: the window centroid is filter- and frame-specific,
    # and MLF's dt_s = dt_mom/n makes n*dt_s == 2*dt), the freezing itself is
    # removed. Substep i is forced at t + (i+1)*dt_s, which is the time that
    # substep's state actually represents, so no centroid or frame reasoning
    # is required and every filter is correct by construction.
    #
    # Cost is kept off the hot path by splitting the harmonic sum: the spatial
    # basis is built ONCE here and the loop only contracts it against
    # per-substep cos/sin factors. Feature-gated on the STATIC config bool
    # (CLAUDE.md feature-gating exception) AND a supplied traced model time:
    # disabled / no-time => the branch is not traced => bit-identical.
    from legoesm.ocean.physics.tidal_forcing import (
        tidal_acceleration_basis, tidal_phase_factors)
    _tide_cfg = getattr(config, "tidal_forcing", None)
    _tide_basis = _tide_cos = _tide_sin = None
    if (_tide_cfg is not None and _tide_cfg.enabled
            and t_seconds is not None):
        _tide_basis = tidal_acceleration_basis(grid, _tide_cfg, g=g)
        # Cast to the working dtype BEFORE the basis is closed over: the
        # grid geometry can be float64 under x64, and adding a float64
        # acceleration to the float32 predictor would promote the loop carry
        # and break its type invariant.
        _tide_basis = _tide_basis._replace(
            ax_cos=_tide_basis.ax_cos.astype(eta.dtype),
            ax_sin=_tide_basis.ax_sin.astype(eta.dtype),
            ay_cos=_tide_basis.ay_cos.astype(eta.dtype),
            ay_sin=_tide_basis.ay_sin.astype(eta.dtype))
        # Phase reduced mod 2*pi at t's OWN precision inside
        # tidal_phase_factors, before the cast — omega*t reaches ~1e5 rad on a
        # long run, where float32 resolves only ~1e-2 rad, and once cos/sin
        # are taken the reduction cannot be recovered.
        _t_sub = t_seconds + (jnp.arange(1, n_loop + 1,
                                         dtype=jnp.asarray(dt_s).dtype) * dt_s)
        _tide_cos, _tide_sin = tidal_phase_factors(_tide_basis, _t_sub)
        _tide_cos = _tide_cos.astype(eta.dtype)
        _tide_sin = _tide_sin.astype(eta.dtype)


    _filter = config.barotropic.barotropic_time_filter
    _boxcar_ab3 = _filter in (
        "nemo_boxcar_ab3", "nemo_boxcar1_ab3")
    _ab3 = _filter in (
        "nemo_ab3am4", "nemo_boxcar_ab3", "nemo_boxcar1_ab3")
    # NEMO exposes no independent switch here.  Its RK3 + flux-form scheme
    # identity necessarily runs the transport primary at dynspg_ts.F90:
    # 823-834,956-979.  The underscore argument is a private fidelity-harness
    # ablation, never a model configuration selector.
    _primary_transport_average = (
        getattr(config, "momentum_time_integrator", "euler") == "rk3_ws"
        and getattr(config, "momentum_advection", "vector_invariant")
        == "flux_form"
        and _boxcar_ab3
    )
    if _nemo_primary_transport_average_test_override is not None:
        _primary_transport_average = bool(
            _nemo_primary_transport_average_test_override)
    if _boxcar_ab3:
        # NEMO nn_bt_flt=1/2: the barotropic sub-state is re-initialised EVERY
        # baroclinic step (ll_init=ll_bt_av=T, dynspg_ts.F90:202/469-476) ⇒ the
        # ll_init ramp fires every step and there is NO cross-window bt_hist
        # carry.  The ssh half-step-back interpolation uses the rn_bt_alpha=0
        # alpha-zero literals (flt2=True); boxcar averaging is applied
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
        if _ab3_hist is not None and _carried_baro is None:
            raise ValueError(
                "nemo_ab3am4 continuation requires the paired prognostic "
                "uu_b/vv_b boundary mean when bt_hist is present")
        _ab3_za, _ab3_zb = nemo_ab3am4_coeff_arrays(
            n_loop, ramp=_ab3_hist is None)
        # cast to the state dtype: f64 coefficients would silently promote the
        # f32 carry and break the scan carry-type invariant.
        _ab3_za = _ab3_za.astype(eta.dtype)
        _ab3_zb = _ab3_zb.astype(eta.dtype)
    else:
        _ab3_za = _ab3_zb = _ab3_hist = None
    if _nemo_raw_history_test_override is not None:
        if not _ab3:
            raise ValueError(
                "raw barotropic-history substitution requires an AB3/AM4 "
                "barotropic filter")
        if len(_nemo_raw_history_test_override) != 6:
            raise ValueError(
                "raw barotropic-history substitution requires six arrays")
    _loop_result = _run_substep_loop(
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
        ab3_raw_hist=_nemo_raw_history_test_override,
        tide_basis=_tide_basis, tide_cos=_tide_cos, tide_sin=_tide_sin,
        een_pre=_een_pre,
        drag_r_u=_drag_r_u, drag_r_v=_drag_r_v, drag_r_t=_drag_r_t,
        primary_transport_average=_primary_transport_average,
        return_trace=_nemo_substep_trace_test_hook,
        nemo_flux_form_update_test_override=(
            _nemo_flux_form_update_test_override),
        nemo_continuity_update_test_override=(
            _nemo_continuity_update_test_override),
    )
    if _nemo_substep_trace_test_hook:
        _finals, _substep_trace = _loop_result
    else:
        _finals = _loop_result
    (eta_f, U_bar_f, V_bar_f,
     Hu_sum_f, Hv_sum_f, eta_sum_f, U_sum_f, V_sum_f) = _finals[:8]

    # Generic transport weights already carry their full normalization.  The
    # DINO literal path mirrors dynspg_ts.F90:734-737,999-1000 instead: raw
    # wgtbtp2 in every addition and exactly one division after the loop.
    if _transport_divisor is None:
        Hu_avg = Hu_sum_f
        Hv_avg = Hv_sum_f
    else:
        _barrier = jax.lax.optimization_barrier
        Hu_avg = _barrier(
            _barrier(Hu_sum_f) / _barrier(_transport_divisor))
        Hv_avg = _barrier(
            _barrier(Hv_sum_f) / _barrier(_transport_divisor))

    if _ab3 and not _boxcar_ab3:
        # NEMO nn_bt_flt=3: the new state is the FINAL substep value (no time
        # averaging); Hu_avg above is the plain substep mean (uniform
        # w_transport), continuity-consistent with eta_f by telescoping.
        eta_avg = eta_f
        U_bar_avg = U_bar_f
        V_bar_avg = V_bar_f
    else:
        # Time-averaged eta and velocity (cosine / box / nemo_boxcar_centred /
        # nemo_boxcar_ab3 / nemo_boxcar1_ab3 = NEMO boxcar averaging of the
        # raw substep ssh).
        eta_avg = eta_sum_f / w_total
        U_bar_avg = U_sum_f / w_total
        V_bar_avg = V_sum_f / w_total
        if _primary_transport_average:
            if config.barotropic.barotropic_face_depth == "nemo_ssh_avg":
                _H_u_primary, _H_v_primary = _nemo_ssh_avg_apply(
                    eta_avg, u_mask, v_mask, grid, _area,
                    _nemo_ssh_avg_prep(H_bathy, mask, grid, _dt,
                                       north_fold_mask(grid)))
            else:
                _H_primary = jnp.maximum(
                    eta_avg + H_bathy, min_water_col) * mask
                _H_u_primary, _H_v_primary = _min_rule_face_depths(
                    _H_primary, mask, grid, north_fold_mask(grid))
            U_bar_avg = U_bar_avg / jnp.maximum(
                _H_u_primary, min_water_col)
            V_bar_avg = V_bar_avg / jnp.maximum(
                _H_v_primary, min_water_col)

    # SOTA-local split-explicit: the per-substep clamp was LOCAL (no allreduce);
    # restore GLOBAL mass conservation with ONE redistribute call on the
    # time-averaged eta (the returned SSH).  No-op (bit-identical to the
    # per-substep path) when no cell hit eta_floor; this single call's 3 batched
    # allreduces replace the subcycle's ~3*n_substeps (the outer-step
    # fix_eta_drift fixer is separate, unaffected).
    if config.barotropic.barotropic_local_subcycle_clamp:
        eta_avg = _clamp_redistribute(eta_avg, eta_floor, mask, _area)

    # 3-D momentum depth-mean RECONCILIATION target (NEMO dyn_spg_ts N6);
    # ``U_bar_corr`` is the NOW-thickness depth-mean of the 3-D velocity
    # (== NEMO ``puu_b(Kmm)``) the caller subtracts below.  Dispatch +
    # divisor convention live in the shared helper (same call on the
    # wide-halo path).
    recon_u, recon_v = _reconcile_targets(
        config, U_bar_avg=U_bar_avg, V_bar_avg=V_bar_avg,
        Hu_avg=Hu_avg, Hv_avg=Hv_avg,
        h_k_now=_h_k_corr if _seed_override else h_k, grid=grid,
        min_water_col=min_water_col, dtype=_dt)

    # Correct 3D velocities: preserve baroclinic structure.
    # Use the reconciliation target for the 3D correction to ensure
    # consistency with eta_avg (the time-averaged eta used for layer thicknesses).
    u_baro_old = U_bar_corr[..., jnp.newaxis]
    v_baro_old = V_bar_corr[..., jnp.newaxis]
    u_prime = u_corr - u_baro_old
    v_prime = v_corr - v_baro_old
    u_new = (u_prime + recon_u[..., jnp.newaxis]) * u_mask[..., jnp.newaxis]
    v_new = (v_prime + recon_v[..., jnp.newaxis]) * v_mask[..., jnp.newaxis]

    state_new = state._replace(
        eta=state.eta.replace(data=eta_avg),
        u=state.u.replace(data=u_new),
        v=state.v.replace(data=v_new),
    )
    if _carried_baro is not None:
        # dynspg_ts.F90:857-897 commits the external solution into Kaa.  The
        # outer stprk3.F90:213 slot swap makes this returned pair next-step
        # Kbb; no live 3-D re-reduction is involved.
        state_new = state_new._replace(
            uu_b=state.uu_b.replace(data=U_bar_avg),
            vv_b=state.vv_b.replace(data=V_bar_avg),
        )
    if _ab3 and not _boxcar_ab3 and hasattr(state, "bt_hist"):
        # NEMO nn_bt_flt=3 only (nn_bt_flt=2 re-inits the sub-state each step ⇒
        # no cross-window carry).
        # Store NEMO's six absolute end-of-window histories for the next
        # window.  Positions 8-13 are Ub, Ubb, Vb, Vbb, etab, etabb after the
        # final dynspg_ts rotation; no derived or live 3-D mean is involved.
        state_new = state_new._replace(bt_hist=tuple(_finals[8:14]))
    if _nemo_substep_trace_test_hook:
        return state_new, (Hu_avg, Hv_avg), _substep_trace
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
    # The AB3 time filters need the predictor coefficient arrays, the
    # cross-window bt_hist carry and the final-substep state selection that
    # the standard entry point builds at its _ab3 block; this path builds
    # NONE of them and passes no ab3_* argument to _run_substep_loop, so
    # running them here is not an approximation but a wrong answer:
    # 'nemo_ab3am4' zeroes the filter weights while leaving w_total = 1, so
    # the wide path divides a zero accumulator and returns sea surface
    # height IDENTICALLY ZERO (measured, 12 substeps, fp64: standard
    # max|eta| = 4.377e-01 m, wide = 0.000e+00); 'nemo_boxcar_ab3' -- DINO's
    # own filter (nn_bt_flt=2) -- runs the boxcar weights without the
    # predictor and diverges at 2.445e-03 m, nine orders above the 1e-12
    # parity gate. Refuse both, matching this function's precedent for the
    # EEN Coriolis and substep-drag options above.
    if config.barotropic.barotropic_time_filter in (
            "nemo_ab3am4", "nemo_boxcar_ab3"):
        raise NotImplementedError(
            "barotropic_time_filter="
            f"{config.barotropic.barotropic_time_filter!r} is not wired into "
            "the wide-halo barotropic path (the AB3 predictor coefficients, "
            "the cross-window bt_hist carry and the final-substep state "
            "selection are built only by the standard entry point); use the "
            "standard split-explicit path or disable barotropic_wide_halo.")

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
    # Loop-ENTRY seed face-depth convention (#1226 round 2 item 1; see
    # BarotropicConfig.barotropic_seed_face_depth docstring) — same option,
    # same dispatch, as the standard-halo entry point above.
    _seed_fd = config.barotropic.barotropic_seed_face_depth
    _seed_eval = config.barotropic.barotropic_seed_evaluation
    _area_seed = grid.area.astype(_dt)
    _carried_baro = _carried_nemo_depth_mean(state, _dt, config)
    if _carried_baro is None:
        U_bar, V_bar = _depth_average_to_faces(
            u, v, h_k, min_water_col, mask, u_mask, v_mask, grid,
            seed_face_depth=_seed_fd, seed_evaluation=_seed_eval,
            eta_dyn=eta, H_bathy=H_bathy,
            area=_area_seed, z_coord=z_coord,
        )
    else:
        U_bar, V_bar = _carried_baro

    w_filter, w_total, w_transport, n_loop = _compute_weights(
        config, n_substeps, eta.dtype)
    w_transport, _transport_divisor = _transport_accumulator_weights(
        config, n_substeps, eta.dtype, w_transport, n_loop)

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
    # PER SUBSTEP, matching the standard path: the basis is built once on the
    # extended band (so it must run under local_halo_pads(), the only
    # band-specific part) and the loop contracts it against per-substep phase
    # factors. See the standard path for why the tide is no longer frozen.
    from legoesm.ocean.physics.tidal_forcing import (
        tidal_acceleration_basis, tidal_phase_factors)
    _tide_cfg = getattr(config, "tidal_forcing", None)
    _tide_basis = _tide_cos = _tide_sin = None
    if (_tide_cfg is not None and _tide_cfg.enabled
            and t_seconds is not None):
        with local_halo_pads():
            _tide_basis = tidal_acceleration_basis(grid_ext, _tide_cfg, g=g)
        # Carry-invariant dtype cast (the wrapper used to own this).
        _tide_basis = _tide_basis._replace(
            ax_cos=_tide_basis.ax_cos.astype(eta.dtype),
            ax_sin=_tide_basis.ax_sin.astype(eta.dtype),
            ay_cos=_tide_basis.ay_cos.astype(eta.dtype),
            ay_sin=_tide_basis.ay_sin.astype(eta.dtype))
        _t_sub = t_seconds + (jnp.arange(1, n_loop + 1,
                                         dtype=jnp.asarray(dt_s).dtype) * dt_s)
        _tide_cos, _tide_sin = tidal_phase_factors(_tide_basis, _t_sub)
        _tide_cos = _tide_cos.astype(eta.dtype)
        _tide_sin = _tide_sin.astype(eta.dtype)

    eta_floor_ext = min_water_col - H_ext
    area_ext = grid_ext.area.astype(_dt)
    with local_halo_pads():
        f_u_ext, f_v_ext = coriolis_at_faces(grid_ext, eta.dtype)
        coeffs_ext = _dissipation_coeffs(
            config, grid_ext, area_ext, dt_s, eta.dtype, mask_ext)

    # --- Chunked substep loop ----------------------------------------------
    eta_c, U_c, V_c = eta, U_bar, V_bar
    _literal_transport = _transport_divisor is not None
    if _literal_transport:
        # The source-order accumulator is one recurrence over the complete
        # substep window.  Carry it across chunk boundaries; summing completed
        # chunk totals afterwards would introduce an association NEMO has no
        # analogue for.
        Hu_c = jnp.zeros_like(U_bar)
        Hv_c = jnp.zeros_like(V_bar)
    sums = None
    done = 0
    for _ in range(n_chunks):
        k = min(chunk, int(n_loop) - done)
        # Wide exchange of the carry (2 fused messages) with the REAL backend.
        # Literal Hu/Hv rides in those SAME messages; a second exchange pair
        # would silently erase the wide-halo communication budget.
        if _literal_transport:
            eta_x, U_x, Hu_x = widen_band_cell_fields(
                (eta_c, U_c, Hu_c), W)
            V_x, Hv_x = widen_band_vface_fields((V_c, Hv_c), W)
            _transport_sum_init = (Hu_x, Hv_x)
        else:
            (eta_x, U_x) = widen_band_cell_fields((eta_c, U_c), W)
            (V_x,) = widen_band_vface_fields((V_c,), W)
            _transport_sum_init = None
        wf = jax.lax.slice_in_dim(w_filter, done, done + k)
        wt = jax.lax.slice_in_dim(w_transport, done, done + k)
        # Slice the tide phases with the SAME [done, done+k) window as the
        # weights: that IS the global substep offset, so a chunked run forces
        # each substep at the same time an unchunked one would. Slicing them
        # independently (or restarting at 0 per chunk) would repeat the first
        # chunk's tide in every chunk.
        tcos = (None if _tide_cos is None
                else jax.lax.slice_in_dim(_tide_cos, done, done + k))
        tsin = (None if _tide_sin is None
                else jax.lax.slice_in_dim(_tide_sin, done, done + k))
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
                tide_basis=_tide_basis, tide_cos=tcos, tide_sin=tsin,
                transport_sum_init=_transport_sum_init,
            )
        (eta_ext_f, U_ext_f, V_ext_f,
         Hu_k, Hv_k, eta_sum_k, U_sum_k, V_sum_k) = finals
        # Crop the carry back to the owned band for the next exchange.
        eta_c = eta_ext_f[W:W + nl]
        U_c = U_ext_f[W:W + nl]
        V_c = V_ext_f[W:W + nl + 1]
        # The literal Hu/Hv recurrence was seeded with the previous chunk's
        # owned result, so crop it as the next carry and do NOT add chunk
        # totals. Generic mode retains the pre-round-59 tuple/add statements
        # byte-for-byte.
        if _literal_transport:
            Hu_c = Hu_k[W:W + nl]
            Hv_c = Hv_k[W:W + nl + 1]
            k_sums = (eta_sum_k[W:W + nl], U_sum_k[W:W + nl],
                      V_sum_k[W:W + nl + 1])
        else:
            k_sums = (Hu_k[W:W + nl], Hv_k[W:W + nl + 1],
                      eta_sum_k[W:W + nl], U_sum_k[W:W + nl],
                      V_sum_k[W:W + nl + 1])
        sums = k_sums if sums is None else tuple(
            a + b for a, b in zip(sums, k_sums))
        done += k

    if _literal_transport:
        eta_sum_f, U_sum_f, V_sum_f = sums
        _barrier = jax.lax.optimization_barrier
        Hu_avg = _barrier(_barrier(Hu_c) / _barrier(_transport_divisor))
        Hv_avg = _barrier(_barrier(Hv_c) / _barrier(_transport_divisor))
    else:
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
    # Same N6 reconciliation dispatch as the standard-halo path (shared
    # helper).  ``h_k`` / ``grid`` are the OWNED-band NOW thickness and
    # geometry, matching the ``U_bar`` subtracted just below; this path has
    # no before-level seed override, so there is no ``_h_k_corr`` variant.
    recon_u, recon_v = _reconcile_targets(
        config, U_bar_avg=U_bar_avg, V_bar_avg=V_bar_avg,
        Hu_avg=Hu_avg, Hv_avg=Hv_avg,
        h_k_now=h_k, grid=grid, min_water_col=min_water_col, dtype=_dt)

    u_baro_old = U_bar[..., jnp.newaxis]
    v_baro_old = V_bar[..., jnp.newaxis]
    u_prime = u - u_baro_old
    v_prime = v - v_baro_old
    u_new = (u_prime + recon_u[..., jnp.newaxis]) * u_mask[..., jnp.newaxis]
    v_new = (v_prime + recon_v[..., jnp.newaxis]) * v_mask[..., jnp.newaxis]

    state_new = state._replace(
        eta=state.eta.replace(data=eta_avg),
        u=state.u.replace(data=u_new),
        v=state.v.replace(data=v_new),
    )
    if _carried_baro is not None:
        state_new = state_new._replace(
            uu_b=state.uu_b.replace(data=U_bar_avg),
            vv_b=state.vv_b.replace(data=V_bar_avg),
        )
    return state_new, (Hu_avg, Hv_avg)


# Public promotions (CLAUDE.md cross-module private-import ratchet):
# these symbols are imported by sibling modules; expose a public alias
# so importers use the sanctioned public name (definitions keep the
# original underscore name for in-module callers).
#
# Distinct name (NOT plain ``depth_average_to_faces``): this module already
# imports the canonical ``depth_average_to_faces`` from ocean_tendency_common
# (line 65); the barotropic C-grid has its OWN face-averaging variant, so the
# public alias is namespaced to avoid shadowing that import (codex).
barotropic_depth_average_to_faces = _depth_average_to_faces


def nemo_literal_midpoint_extrapolation(
    coefficients, now, before, before_before,
):
    """NEMO AB3 external-mode midpoint in written source association.

    ``dynspg_ts.F90`` forms ``ua_e`` and ``va_e`` as the left-associated
    three-term sum.  Materialize each multiply and add so the shared helper is
    usable both by the production loop and by oracle-input fidelity gates.
    """
    first = nemo_source_round(
        coefficients[0] * nemo_source_round(now))
    second = nemo_source_round(
        coefficients[1] * nemo_source_round(before))
    third = nemo_source_round(
        coefficients[2] * nemo_source_round(before_before))
    value = nemo_source_round(first + second)
    return nemo_source_round(value + third)
