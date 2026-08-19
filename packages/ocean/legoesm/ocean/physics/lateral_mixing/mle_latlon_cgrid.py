"""Fox-Kemper mixed-layer-eddy (MLE) restratification — lat-lon / tripole C-grid.

Faithful port of NEMO 5.0.1 ``src/OCE/TRA/tramle.F90`` ``tra_mle_trp_RK3`` for the
ORCA1 setting ``ln_mle=.true., nn_mle=1, nn_conv=1`` (the "new formulation").
The submesoscale mixed-layer eddies are represented by an eddy-induced (bolus)
overturning streamfunction that restratifies the mixed layer — flattening ML
isopycnals and shoaling the mixed-layer depth, especially in the subtropical-gyre
mode-water regions south of the western boundary currents where coarse models
lack the resolved eddies.

SANCTIONED DEPARTURE FROM NEMO (codex MLE-vertfix review, 2026-08-11): NEMO
adds the three MLE transport components to the ADVECTING transport and lets
the tracer advection scheme (ORCA1: FCT) carry them, inheriting its
anti-diffusive limiter.  This port applies a STANDALONE centered-tracer flux
divergence instead: conservative and divergence-free per cell, but
non-monotone — centered explicit advection can disperse and create local
extrema at sharp fronts.  If plume-front overshoots appear in production with
the vertical branch present, route the MLE transports through the advection
scheme rather than bolting a limiter on here.

This is the Arakawa C-grid adapter (regular lat-lon + ORCA tripole, array layout
``(n_lat, n_lon, nlev)``).  It reuses the grid-AGNOSTIC core in
:mod:`legoesm.ocean.physics.lateral_mixing.mle` (``mle_coefficient``,
``mle_vertical_structure``, ``mle_mld_and_buoyancy``, ``face_mld``) and the shared
C-grid operators (``gradient_x_cgrid``, ``gradient_y_cgrid``,
``neumann_fill_cgrid``, ``compute_face_masks_3d``, ``ensure_geometry``) — it adds
NO new numerics beyond the face streamfunction assembly and the conservative
bolus-flux divergence, mirroring ``gm_redi_latlon_cgrid``.

NEMO nn_mle=1 streamfunction MAGNITUDE (per u-face), using
``(bm_E - bm_W) / e1u = dbm/dx`` so ``e2_e1u·(bm_E - bm_W) = e2u·dbm/dx``::

    Psi_u = rc_f · H_u² · e2u · (dbm/dx) · min(111 km, e1u)
    Psi_v = rc_f · H_v² · e1v · (dbm/dy) · min(111 km, e2v)

with ``rc_f = rn_ce / (5 km · 2Ω sin(rn_lat))`` (constant ⇒ no equatorial
singularity), ``H_u``/``H_v`` the FACE mixed-layer depth (``face_mld`` of the two
neighbour MLDs), and ``bm`` the vertically-averaged ML buoyancy.  The W-grid
streamfunction is ``Psi_uw[k] = Psi_u · mu(gdepw_uw[k]/H_u) · u_mask`` (zero at
the surface and at/below the ML base), the bolus volume transport at the T-level
is the adjacent-W difference ``Utr_u[k] = Psi_uw[k] - Psi_uw[k+1]`` [m³/s], the
centered tracer flux is ``Utr_u·½(T_W+T_E)``, and the conservative tendency is the
flux divergence over the LIVE cell volume.

References
----------
Fox-Kemper, Ferrari & Hallberg (2008), JPO 38, 1145-1165.
Fox-Kemper & Ferrari (2008), JPO 38, 1166-1179.
NEMO 5.0.1 TRA/tramle.F90 (ORCA1 RUN_REF: ln_mle, nn_mle=1, rn_ce=0.06, rn_lat=20,
nn_mld_uv=0, nn_conv=1, rn_rho_c_mle=0.01).
"""
from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.grids.latlon import ensure_geometry
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    compute_face_masks_3d,
    gradient_x_cgrid,
    gradient_y_cgrid,
    interp_cell_to_uface,
    interp_cell_to_vface,
)
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import neumann_fill_cgrid
from legoesm.ocean.physics.lateral_mixing.mle import (
    MLEConfig,
    face_mld,
    mle_coefficient,
    mle_mld_and_buoyancy,
    mle_streamfunction_magnitude,
    mle_vertical_structure,
)
from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
    EPS_DIV as _EPS_DIV,
)

# Division-guard epsilon (shared _gm_redi_common.EPS_DIV = 1e-10, #518 item 11):
# larger than float32 machine eps to prevent intermediate blow-up in the
# backward pass (matches gm_redi_latlon_cgrid).


__physics_contract__ = {
    "summary": (
        "Fox-Kemper mixed-layer-eddy (MLE) restratification on the lat-lon / "
        "tripole Arakawa C-grid (NEMO tramle nn_mle=1): a submesoscale bolus "
        "overturning streamfunction Psi = rc_f·H^2·e2u·(dbm/dx)·min(111km,e1u)·"
        "mu(z), evaluated at the mixed-layer FACE depth H_u/H_v, drives a "
        "down-gradient bolus transport Utr = dk[Psi] whose centered tracer flux "
        "is applied as a conservative divergence. It flattens mixed-layer "
        "isopycnals (restratifying), shoaling the MLD in mode-water regions a "
        "coarse model cannot resolve. No MLE where the column is statically "
        "unstable (N^2 < 0; nn_conv=1)."
    ),
    "inputs": {
        "T": "degC", "S": "PSU", "rho_pot": "kg/m^3 (surface-referenced potential density, NEMO rhop)", "N2": "1/s^2",
        "mask": "1", "u_mask": "1", "v_mask": "1", "jacobian": "1",
        "ce": "1", "lat_ref_deg": "deg", "rho_c_mle": "kg/m^3",
        "ref_depth_m": "m",
    },
    "outputs": {"dT_dt": "degC/s", "dS_dt": "PSU/s"},
    "sign_convention": (
        "The bolus streamfunction magnitude carries the sign of the ML "
        "buoyancy gradient dbm/dx (lighter water on one side), so the bolus "
        "transport advects light water over dense — DOWN the buoyancy gradient "
        "— restratifying the mixed layer (flattening isopycnals, shoaling the "
        "MLD). The tendency is the negative divergence of the centered bolus "
        "tracer flux over the live cell volume."
    ),
    # The bolus transport is a CLOSED overturning cell: horizontal transports
    # are adjacent-W differences of the streamfunction, and the VERTICAL
    # transport is reconstructed from continuity (NEMO zw_mle =
    # -di[psi_uw]-dj[psi_vw]), so the 3-D divergence vanishes per cell and a
    # uniform tracer has EXACTLY zero tendency.  Global conservation
    # (sum(dT·area·dz_live) = 0 to roundoff) follows; the horizontal-only
    # variant satisfied the global sum while corrupting river-plume fronts
    # (-54 psu in 30 d) and is the regression the uniform-tracer test locks.
    "conserves": ["tracer"],
    "differentiable": True,
    "reference": (
        "Fox-Kemper, Ferrari & Hallberg (2008) JPO 38 1145-1165; Fox-Kemper & "
        "Ferrari (2008) JPO 38 1166-1179; NEMO 5.0.1 TRA/tramle.F90 "
        "(ORCA1 RUN_REF: ln_mle, nn_mle=1, rn_ce=0.06, rn_lat=20, nn_conv=1)."
    ),
    "idealized_test": (
        "tests/ocean/unit/test_mle.py: vertical structure mu(0)=mu(1)=0 with a "
        "mid-ML peak; rc_f>0 and equator guard; 2-layer MLD + buoyancy sign; "
        "EXACT tracer conservation sum(dT·area·dz)~0 on a buoyancy-front "
        "domain; restratifying flux sign across a light/dense cell pair."
    ),
}


def _wet_cell_3d(mask: jnp.ndarray, z_coord, nlev: int) -> jnp.ndarray:
    """Per-cell wet mask ``(n_lat, n_lon, nlev)``.

    Uses the partial-cell ``is_active`` field when present (realistic
    bathymetry), else broadcasts the 2-D ``mask`` over all levels (pure z*,
    flat bottom).  Mirrors ``gm_redi_latlon_cgrid._per_level_face_acts`` /
    ``_fill_dry_cells_columnwise`` dispatch on ``getattr(z_coord, 'is_active')``.
    """
    is_active = getattr(z_coord, "is_active", None)
    if is_active is not None:
        return is_active.astype(mask.dtype)
    return jnp.broadcast_to(mask[:, :, jnp.newaxis], (*mask.shape, nlev))


def _gdepw_w(dz_live: jnp.ndarray) -> jnp.ndarray:
    """Cumulative LIVE w-interface depths ``(n_lat, n_lon, nlev+1)`` [m, positive].

    ``gdepw[..., 0] = 0`` (surface), ``gdepw[..., k] = sum_{l<k} dz_live[l]`` so
    ``gdepw[..., nlev]`` is the live column depth.  Matches NEMO ``gdepw`` (the
    w-level depths used in the ``mu(z)`` structure function), partial-cell aware
    because ``dz_live`` already carries the Jacobian + dry-cell zeroing.
    """
    horiz = dz_live.shape[:-1]
    zero = jnp.zeros((*horiz, 1), dtype=dz_live.dtype)
    return jnp.concatenate([zero, jnp.cumsum(dz_live, axis=-1)], axis=-1)


def mle_tracer_tendency_latlon_cgrid(
    T: jnp.ndarray,
    S: jnp.ndarray,
    rho_pot: jnp.ndarray,
    N2: jnp.ndarray,
    mask: jnp.ndarray,
    u_mask: jnp.ndarray,
    v_mask: jnp.ndarray,
    z_coord,
    jacobian: jnp.ndarray,
    grid,
    cfg: MLEConfig,
    h_k: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Fox-Kemper MLE bolus tracer tendency on the lat-lon C-grid (NEMO nn_mle=1).

    Parameters
    ----------
    T, S : (n_lat, n_lon, nlev)
        Potential temperature [degC] / salinity [PSU] at cell centres.
    rho_pot : (n_lat, n_lon, nlev)
        SURFACE-REFERENCED POTENTIAL density [kg/m^3] at cell centres
        (NEMO ``rhop``: the EOS evaluated at zero pressure — e.g.
        ``make_eos_fn()(T, S, 0)``).  NOT in-situ density: compressibility
        alone exceeds the 0.01 kg/m^3 Delta-rho threshold between adjacent
        levels and collapses the diagnosed mixed layer (see
        ``mle_mld_and_buoyancy``).
    N2 : (n_lat, n_lon, nlev-1)
        Brunt-Vaisala frequency squared [1/s^2] at interior interfaces —
        drives the convection gate.  Use the LOCALLY-REFERENCED
        adiabatic-parcel form (``compute_buoyancy_frequency_adiabatic``,
        NEMO's ``rn2`` analogue): the in-situ-difference form carries the
        compressibility between reference pressures (~6x too stable) and
        reads deep unstable columns as stable, leaking transport through
        the nn_conv gate.
    mask : (n_lat, n_lon)
        2-D ocean mask (1 = ocean, 0 = land).
    u_mask : (n_lat, n_lon+1)
        u-face mask (1 if both adjacent cells are ocean).
    v_mask : (n_lat+1, n_lon)
        v-face mask.
    z_coord : OceanZStarCoordinate or OceanPartialCellCoordinate
        Vertical coordinate.  ``is_active`` (partial cells) is used for the wet
        domain when present.
    jacobian : (n_lat, n_lon)
        z-star / partial-cell Jacobian (eta + H) / H.
    grid : LatLonGrid or LatLonCGridGeometry
        Horizontal grid (``ensure_geometry`` supplies the face metrics).
    cfg : MLEConfig

    Returns
    -------
    (dT_dt, dS_dt) : each (n_lat, n_lon, nlev) [degC/s], [PSU/s].
    """
    nlev = T.shape[-1]
    geom = ensure_geometry(grid)
    # Face metrics (NEMO e1u/e2u/e1v/e2v) — bit-exact with the operator metrics.
    e1u = geom.dx_u                                   # (n_lat, n_lon+1) zonal spacing
    e2u = geom.dy_u                                   # (n_lat, n_lon+1) meridional extent
    e1v = geom.dx_v                                   # (n_lat+1, n_lon) zonal extent
    e2v = geom.dy_v                                   # (n_lat+1, n_lon) meridional spacing
    area = geom.area_T                                # (n_lat, n_lon) cell area

    # Live layer thickness: prefer the actual partial-cell thickness h_k (passed
    # from the caller via compute_layer_thickness, IDENTICAL to what
    # compute_ocean_rho uses) so MLD/bm/gdepw/volume are consistent over real
    # bathymetry; fall back to dz_ref*J only for pure z* (no partial cells).
    if h_k is not None:
        dz_live = h_k                                       # (n_lat, n_lon, nlev)
    else:
        dz_live = z_coord.dz_ref * jacobian[:, :, jnp.newaxis]   # pure z* fallback
    wet3d = _wet_cell_3d(mask, z_coord, nlev)                # (n_lat, n_lon, nlev)
    # Reference W-INTERFACE depths [m, positive down] for the NEMO nla10
    # reference-level pick (z_half_ref is 0 at the surface, -H_max at the
    # bottom); the 0.01 criterion is robust to the small z* stretch.
    z_faces = jnp.abs(z_coord.z_half_ref)                    # (nlev+1,)

    # --- MLE mixed-layer depth + ML-mean buoyancy (shared core) ---
    # SEPARATE 0.01 in-situ criterion (NOT the 0.03 dBM diagnostic), per codex.
    # Exact NEMO gdept_1d when the coordinate carries it (t_depth_ref, the
    # same pattern the PGF uses): the nla10 tolerance depends on e3w_1d,
    # which is NOT the face-midpoint spacing on a partial-cell ladder.
    _t_ref = getattr(z_coord, "t_depth_ref", None)
    zmld, bm, in_ml = mle_mld_and_buoyancy(
        rho_pot, dz_live, wet3d,
        z_faces=z_faces,
        z_centers_ref=(None if _t_ref is None else jnp.abs(jnp.asarray(_t_ref))),
        rho_c_mle=cfg.rho_c_mle,
        ref_depth_m=cfg.ref_depth_m,
        rho0=constants.rho_ocean,
        grav=constants.g,
    )                                                 # zmld, bm both (n_lat, n_lon)

    # Neumann-fill bm / zmld so the face gradients and face MLD across a
    # coastline take an ocean-neighbour value (the flux is still killed by the
    # face masks downstream).  2-D fields -> fill in place.
    bm_filled = neumann_fill_cgrid(bm, mask)
    zmld_filled = neumann_fill_cgrid(zmld, mask)

    # --- Face mixed-layer depth H_u / H_v (NEMO nn_mld_uv) ---
    # u-face j is between cell west=(j-1) mod n_lon and east=j (periodic);
    # v-face i is between south=(i-1) and north=i.
    zmld_w = jnp.roll(zmld_filled, 1, axis=1)         # west neighbour at each u-face
    H_u_int = face_mld(zmld_w, zmld_filled, cfg.mld_uv)        # (n_lat, n_lon)
    H_u = jnp.concatenate([H_u_int, H_u_int[:, 0:1]], axis=1)  # (n_lat, n_lon+1) periodic
    # v-face MLD: interior faces from south/north neighbours; pole rows inert
    # (v_mask zeroes the flux there).
    H_v_int = face_mld(zmld_filled[:-1], zmld_filled[1:], cfg.mld_uv)   # (n_lat-1, n_lon)
    H_v = jnp.concatenate(
        [zmld_filled[0:1], H_v_int, zmld_filled[-1:]], axis=0)          # (n_lat+1, n_lon)

    # --- Streamfunction magnitude (NEMO nn_mle=1) ---
    rc_f = mle_coefficient(cfg.ce, cfg.lat_ref_deg)   # constant scalar [s/m]
    dbm_dx_u = gradient_x_cgrid(bm_filled, grid)      # (n_lat, n_lon+1) = (bm_E-bm_W)/e1u
    dbm_dy_v = gradient_y_cgrid(bm_filled, grid)      # (n_lat+1, n_lon) = (bm_N-bm_S)/e2v
    cap_u = jnp.minimum(cfg.max_grid_scale_m, e1u)    # min(111 km, e1u)
    cap_v = jnp.minimum(cfg.max_grid_scale_m, e2v)
    # Psi = rc_f · H² · width · dbm · cap (shared kernel, #518 item 9).
    psim_u = mle_streamfunction_magnitude(rc_f, H_u, e2u, dbm_dx_u, cap_u)  # (n_lat, n_lon+1)
    psim_v = mle_streamfunction_magnitude(rc_f, H_v, e1v, dbm_dy_v, cap_v)  # (n_lat+1, n_lon)

    # --- Convection gate (NEMO nn_conv=1): no MLE where a neighbour column is
    # statically unstable.  NEMO gates on the ML-INTEGRATED N^2 (zn2), NOT the
    # whole-column min, so a deep unstable interface BELOW the mixed layer does
    # not spuriously kill the restratification.  Sum N^2 over the mixed-layer
    # interfaces (in_ml restricts to the ML), then face-local min of the two
    # columns. ---
    if cfg.no_mle_in_convection:
        iface_in_ml = in_ml[:, :, :-1]                # (n_lat,n_lon,nlev-1): iface k in ML if cell k is
        col_n2 = jnp.sum(iface_in_ml * N2, axis=-1)   # (n_lat, n_lon) NEMO zn2 (ML-integrated N^2)
        col_n2_filled = neumann_fill_cgrid(col_n2, mask)
        n2_w = jnp.roll(col_n2_filled, 1, axis=1)
        face_n2_u_int = jnp.minimum(n2_w, col_n2_filled)
        face_n2_u = jnp.concatenate(
            [face_n2_u_int, face_n2_u_int[:, 0:1]], axis=1)    # (n_lat, n_lon+1)
        face_n2_v_int = jnp.minimum(col_n2_filled[:-1], col_n2_filled[1:])
        face_n2_v = jnp.concatenate(
            [col_n2_filled[0:1], face_n2_v_int, col_n2_filled[-1:]], axis=0)
        psim_u = jnp.where(face_n2_u < 0.0, 0.0, psim_u)
        psim_v = jnp.where(face_n2_v < 0.0, 0.0, psim_v)

    # --- W-grid structure function and streamfunction (length nlev+1) ---
    # gdepw at cell centres -> interpolate to u/v faces (matches NEMO's
    # 0.5*(gdepw[i+1]+gdepw[i]) face depth), then mu(gdepw_face / H_face).
    gdepw = _gdepw_w(dz_live)                          # (n_lat, n_lon, nlev+1)
    gdepw_u = interp_cell_to_uface(gdepw)              # (n_lat, n_lon+1, nlev+1)
    gdepw_v = interp_cell_to_vface(gdepw, grid)        # (n_lat+1, n_lon, nlev+1)
    inv_Hu = 1.0 / jnp.maximum(H_u, _EPS_DIV)
    inv_Hv = 1.0 / jnp.maximum(H_v, _EPS_DIV)
    # Per-level face wet masks (partial-cell aware): a face is wet at level k iff
    # both adjacent cells are wet there.  Build from the 3-D wet mask; AND with
    # the supplied 2-D face masks so a dynamically-closed face is also killed.
    u_face_act3, v_face_act3 = compute_face_masks_3d(wet3d, grid)   # (...,nlev)
    u_face_act3 = u_face_act3 * u_mask[:, :, jnp.newaxis]
    v_face_act3 = v_face_act3 * v_mask[:, :, jnp.newaxis]
    # mu at the nlev+1 W-interfaces (mu=0 at the surface k=0 and at/below ML base).
    mu_u = mle_vertical_structure(gdepw_u * inv_Hu[:, :, jnp.newaxis])  # (n_lat, n_lon+1, nlev+1)
    mu_v = mle_vertical_structure(gdepw_v * inv_Hv[:, :, jnp.newaxis])  # (n_lat+1, n_lon, nlev+1)
    # Psi_uw[k] = psim_u · mu_u[k] · (wet u-face at the level BELOW the interface).
    # Interface k sits above T-level k, so it is active iff T-level k is a wet
    # face (NEMO multiplies by wumask(jk+1)·wumask(1)).  Surface interface k=0
    # has mu=0 so it is zero regardless; the deepest interface k=nlev has mu=0
    # too (gdepw=H at the ML base / column bottom).
    zcol_u = jnp.zeros((*u_face_act3.shape[:2], 1), dtype=u_face_act3.dtype)
    wface_u = jnp.concatenate([u_face_act3, zcol_u], axis=-1)  # (n_lat, n_lon+1, nlev+1)
    zcol_v = jnp.zeros((*v_face_act3.shape[:2], 1), dtype=v_face_act3.dtype)
    wface_v = jnp.concatenate([v_face_act3, zcol_v], axis=-1)  # (n_lat+1, n_lon, nlev+1)
    psi_uw = psim_u[:, :, jnp.newaxis] * mu_u * wface_u            # (n_lat, n_lon+1, nlev+1)
    psi_vw = psim_v[:, :, jnp.newaxis] * mu_v * wface_v            # (n_lat+1, n_lon, nlev+1)

    # --- Bolus volume transport at the T-level (NEMO dk[Psi]) [m^3/s] ---
    utr_u = psi_uw[:, :, :-1] - psi_uw[:, :, 1:]      # (n_lat, n_lon+1, nlev)
    vtr_v = psi_vw[:, :, :-1] - psi_vw[:, :, 1:]      # (n_lat+1, n_lon, nlev)

    # --- Centered tracer flux at the faces; mask closed faces BEFORE divergence ---
    T_u = interp_cell_to_uface(T)                     # (n_lat, n_lon+1, nlev) = 0.5(T_W+T_E)
    T_v = interp_cell_to_vface(T, grid)               # (n_lat+1, n_lon, nlev)
    S_u = interp_cell_to_uface(S)
    S_v = interp_cell_to_vface(S, grid)
    # Apply the per-level wet face masks to the TRANSPORTS (codex: mask the
    # transports/fluxes, not the final tendency).  utr already vanishes where mu
    # or psim is zero, but a partial-cell step face must be hard-zeroed.
    utr_u = utr_u * u_face_act3
    vtr_v = vtr_v * v_face_act3
    FxT = utr_u * T_u                                 # (n_lat, n_lon+1, nlev) [degC·m^3/s]
    FyT = vtr_v * T_v
    FxS = utr_u * S_u
    FyS = vtr_v * S_v

    # --- VERTICAL bolus transport from continuity (NEMO tramle.F90:
    #        zw_mle = - di[ zpsi_uw ] - dj[ zpsi_vw ]
    # applied to pFw).  The FK overturning is a CLOSED cell in the vertical
    # plane: the horizontal branch alone is globally conservative but locally
    # wrong -- a cell where the horizontal bolus transports converge
    # accumulates tracer without the compensating vertical export, which
    # measured as -54 psu / -31 degC extremes at equatorial river-plume fronts
    # after 30 days (results/omip_nemo/mle_psi_diag_d30).  W is reconstructed
    # from the MASKED horizontal transports so the 3-D divergence closes per
    # cell: W(k) - W(k+1) = -div_h(k) with W = 0 at the sea floor, positive
    # toward the surface.
    div_h = ((utr_u[:, 1:, :] - utr_u[:, :-1, :])
             + (vtr_v[1:, :, :] - vtr_v[:-1, :, :]))   # (n_lat, n_lon, nlev)
    W_int = -jnp.flip(jnp.cumsum(jnp.flip(div_h, axis=-1), axis=-1), axis=-1)
    zcol_c = jnp.zeros((*div_h.shape[:2], 1), dtype=div_h.dtype)
    # Interfaces 0..nlev; the bottom is exactly 0 and the surface is zeroed
    # (its residual is column-telescoping roundoff): tracer conservation is
    # exact and the per-cell continuity error stays at roundoff.
    W_w = jnp.concatenate([W_int, zcol_c], axis=-1)   # (n_lat, n_lon, nlev+1)
    W_w = W_w.at[..., 0].set(0.0)
    # Centered tracer at the interior W-interfaces (0.5(T_above + T_below));
    # the end interfaces carry zero transport so their tracer value is inert.
    T_w = jnp.concatenate(
        [zcol_c, 0.5 * (T[..., :-1] + T[..., 1:]), zcol_c], axis=-1)
    S_w = jnp.concatenate(
        [zcol_c, 0.5 * (S[..., :-1] + S[..., 1:]), zcol_c], axis=-1)
    FzT = W_w * T_w                                   # (n_lat, n_lon, nlev+1)
    FzS = W_w * S_w

    # --- Conservative flux divergence over the LIVE cell volume ---
    # The transports already carry the face cross-distance (e2u/e1v), so the
    # divergence is the raw telescoping difference of face fluxes (NOT
    # divergence_cgrid, which would re-multiply by the face length).  Divide by
    # the live cell volume area·dz_live -> tracer-units/s.
    vol = area[:, :, jnp.newaxis] * dz_live           # (n_lat, n_lon, nlev)
    inv_vol = jnp.where(vol > 0.0, 1.0 / jnp.maximum(vol, _EPS_DIV), 0.0)

    def _div(Fx_u, Fy_v, Fz_w):
        # cell (i,j,k): east u-face j+1 minus west u-face j; north v-face i+1
        # minus south v-face i; top interface k (W positive toward the
        # surface, so outflow through the top) minus bottom interface k+1.
        net_x = Fx_u[:, 1:, :] - Fx_u[:, :-1, :]      # (n_lat, n_lon, nlev)
        net_y = Fy_v[1:, :, :] - Fy_v[:-1, :, :]      # (n_lat, n_lon, nlev)
        net_z = Fz_w[..., :-1] - Fz_w[..., 1:]        # (n_lat, n_lon, nlev)
        return net_x + net_y + net_z

    dT_dt = -_div(FxT, FyT, FzT) * inv_vol
    dS_dt = -_div(FxS, FyS, FzS) * inv_vol

    # Optional bolus vertical-Courant cap (cfg.bolus_cfl_cap > 0): NOT
    # implemented here because the physics_fn signature does not expose ``dt``
    # (the cap is |Utr/(area·dz)|·dt <= fraction).  Default off (0.0) =
    # oracle behaviour; raise loudly rather than silently no-op if requested.
    # TODO(dt-plumbing): thread dt into the lateral-mixing physics_fn to enable
    # the cap, mirroring bbl_adv's host-applied transport-CFL guard.
    if cfg.bolus_cfl_cap > 0.0:
        raise NotImplementedError(
            "MLEConfig.bolus_cfl_cap > 0 needs the timestep dt, which is not "
            "available in the lateral-mixing physics_fn signature. Leave it at "
            "0.0 (the NEMO-oracle default); the bolus magnitude is already "
            "bounded by H^2 and the mu(z) structure. (TODO: plumb dt to enable "
            "the cap.)"
        )

    # The tendency is already zero on dry cells (inv_vol = 0 there); no final
    # mask multiply (codex: divide by the live volume, do not mask the result).
    return dT_dt, dS_dt
