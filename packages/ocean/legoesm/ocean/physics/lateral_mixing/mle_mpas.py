"""Fox-Kemper mixed-layer-eddy (MLE) restratification on the MPAS Voronoi mesh.

Faithful port of NEMO 5.0.1 ``TRA/tramle.F90`` (``ln_mle=.true., nn_mle=1``) to
the icosahedral/Voronoi grid, the MPAS analogue of
``mle_latlon_cgrid.mle_tracer_tendency_latlon_cgrid``.  The grid-AGNOSTIC pieces
(config, ``rc_f`` coefficient, the MLE mixed-layer depth + mean buoyancy, the
``mu(z)`` vertical structure, the face-MLD rule) are reused verbatim from
``mle.py`` — only the edge streamfunction assembly and the conservative bolus
tracer-flux divergence are grid-specific and live here.

Streamfunction (NEMO nn_mle=1), per Voronoi edge ``e`` between cells ``c1,c2``::

    psim_e = rc_f * H_e^2 * dvEdge_e * (bm[c2]-bm[c1])/dcEdge_e * min(111 km, dcEdge_e)
    psi_ew[k] = psim_e * mu(gdepw_e[k] / H_e) * (wet edge below interface k)

with ``rc_f = rn_ce / (5 km * 2*Omega*sin(rn_lat))`` (constant -> no equatorial
singularity), ``H_e`` the face mixed-layer depth, ``bm`` the ML-mean buoyancy,
and ``dvEdge`` the edge (face) length — the Voronoi analogue of NEMO's ``e2u``
cross-face width, so ``psim_e`` is a VOLUME streamfunction [m^3/s] exactly as the
lat-lon ``psim_u`` (which carries ``e2u``).  The bolus volume transport at a
T-level is ``dk[psi]`` and advects T,S; the cell tendency is the conservative
divergence of the edge tracer transports over the live cell volume — exactly
conservative because every interior edge contributes to its two cells with
opposite sign (``Sum_c Vol*dq = -Sum_c Sum_e sign*F_e = 0``).

References
----------
Fox-Kemper, Ferrari & Hallberg (2008), JPO 38, 1145-1165.
NEMO 5.0.1 TRA/tramle.F90 (ORCA1 RUN_REF: ln_mle, nn_mle=1, rn_ce=0.06, rn_lat=20).
See also ``docs/ocean/experiments/mle_mpas_port_plan.md``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from legoesm import constants
from legoesm.core.operators_voronoi import cell_to_edge_avg_3d
from legoesm.ocean.eos import (
    compute_buoyancy_frequency_adiabatic,
    make_eos_fn,
    rho_0 as _RHO_0,
)
from legoesm.ocean.physics.lateral_mixing.gm_redi_mpas import voronoi_neumann_fill
from legoesm.ocean.physics.lateral_mixing.mle import (
    MLEConfig,
    face_mld,
    mle_coefficient,
    mle_mld_and_buoyancy,
    mle_streamfunction_magnitude,
    mle_vertical_structure,
)
from legoesm.ocean.dynamics.ocean_tendency_common import (
    iterate_eos_and_pressure_anomaly,
)
from legoesm.ocean.vertical import compute_layer_thickness, compute_ocean_jacobian
from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
    EPS_DIV as _EPS_DIV,
)

if TYPE_CHECKING:
    from legoesm.grids.voronoi import VoronoiMesh
    from legoesm.ocean.vertical import OceanZStarCoordinate

# Safety floor for divisions: shared _gm_redi_common.EPS_DIV = 1e-10
# (#518 item 11; matches the C-grid MLE module).


__physics_contract__ = {
    "summary": (
        "Fox-Kemper mixed-layer-eddy (MLE) restratification on the MPAS Voronoi "
        "mesh (NEMO tramle nn_mle=1): a submesoscale bolus overturning "
        "streamfunction psim_e = rc_f·H_e^2·dvEdge·(bm[c2]-bm[c1])/dcEdge·"
        "min(111km,dcEdge)·mu(z), evaluated at the edge mixed-layer FACE depth "
        "H_e, drives a down-gradient bolus VOLUME transport Utr = dk[psi] whose "
        "centered tracer flux is applied as a conservative cell divergence. It "
        "flattens mixed-layer isopycnals (restratifying), shoaling the MLD in "
        "mode-water regions a coarse model cannot resolve. No MLE where the "
        "column is statically unstable (ML-integrated N^2 < 0; nn_conv=1). The "
        "grid-agnostic MLD/buoyancy/mu(z)/rc_f pieces are reused from mle.py; "
        "only the edge streamfunction + divergence are Voronoi-specific."
    ),
    "inputs": {
        "T": "degC", "S": "PSU", "eta": "m", "H_bathy": "m",
        "mask": "1", "ce": "1", "lat_ref_deg": "deg",
        "rho_c_mle": "kg/m^3", "ref_depth_m": "m",
    },
    "outputs": {"dT_dt": "degC/s", "dS_dt": "PSU/s"},
    "sign_convention": (
        "psim_e carries the sign of the edge ML buoyancy gradient "
        "(bm[c2]-bm[c1])/dcEdge (lighter water on one side), so the bolus "
        "transport advects light water over dense — DOWN the buoyancy gradient "
        "— restratifying the mixed layer (flattening isopycnals, shoaling the "
        "MLD). The tendency is the NEGATIVE signed-edge divergence of the "
        "centered bolus tracer transport over the live cell volume."
    ),
    # psi is built at W-interfaces (zero at the surface k=0 and at the ML base /
    # column bottom via mu(z)=0), the T-level transport is the adjacent-W
    # difference, and the edge transports already carry the face width dvEdge;
    # since every interior edge contributes to its two cells with opposite
    # edgeSignOnCell, sum_c(Vol·dq) = -sum_c sum_e sign·F_e = 0 to roundoff.
    "conserves": ["tracer"],
    "differentiable": True,
    "reference": (
        "Fox-Kemper, Ferrari & Hallberg (2008) JPO 38 1145-1165; Fox-Kemper & "
        "Ferrari (2008) JPO 38 1166-1179; NEMO 5.0.1 TRA/tramle.F90 "
        "(ORCA1 RUN_REF: ln_mle, nn_mle=1, rn_ce=0.06, rn_lat=20, nn_conv=1)."
    ),
    "idealized_test": (
        "tests/ocean/unit/test_mle_mpas.py: EXACT tracer conservation "
        "sum(dT·area·dz)~0 on an ico buoyancy-front mesh; restratification "
        "reduces the ML-mean-buoyancy horizontal variance; finite at the "
        "equator (rn_lat=20 floor); convection gate zeroes MLE in a statically "
        "unstable column; partial-cell conservation; jit-stable."
    ),
}


def _bolus_divergence_cell(F_edge: jnp.ndarray, mesh: "VoronoiMesh") -> jnp.ndarray:
    """Signed sum of an edge VOLUME-transport flux over each cell's edges.

    ``F_edge`` already carries the face width (it is a transport, [X*m^3/s]), so
    — unlike :func:`operators_voronoi.divergence_cell_3d`, which re-multiplies by
    ``dvEdge`` — this only applies the per-cell edge orientation and sums::

        net[c, k] = Sum_{e in cell c} edgeSignOnCell[e, c] * F_edge[e, k]

    The caller divides by the live cell volume.  Because every interior edge
    appears in exactly two cells with opposite ``edgeSignOnCell`` the global sum
    of ``net`` vanishes -> exact tracer conservation.
    """
    eoc = mesh.edgesOnCell                       # (maxEdges, nCells)
    sign = mesh.edgeSignOnCell                   # (maxEdges, nCells)
    valid = (eoc >= 0).astype(F_edge.dtype)      # (maxEdges, nCells)
    eoc_safe = jnp.maximum(eoc, 0)
    F_g = F_edge[eoc_safe]                        # (maxEdges, nCells, nlev)
    net = jnp.sum(
        (sign * valid)[:, :, None] * F_g, axis=0
    )                                            # (nCells, nlev)
    return net


def mle_tracer_tendency_mpas(
    T: jnp.ndarray,
    S: jnp.ndarray,
    eta: jnp.ndarray,
    H_bathy: jnp.ndarray,
    mesh: "VoronoiMesh",
    z_coord: "OceanZStarCoordinate",
    cfg: MLEConfig,
    *,
    eos: str = "wright",
    eos_linear=None,
    mask: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Fox-Kemper MLE bolus tracer tendency on the MPAS Voronoi mesh.

    Signature mirrors :func:`gm_redi_mpas.gm_redi_tracer_tendency_mpas` so the
    dycore hook in ``ocean_model_mpas.py`` is mechanical (it is added right after
    the GM/Redi block, same additive forward-Euler pattern).

    Parameters
    ----------
    T, S : (nCells, nlev)
        Potential temperature [degC] / salinity [PSU] at cell centres.
    eta : (nCells,)        sea-surface height [m].
    H_bathy : (nCells,)    bottom depth [m, positive].
    mesh : VoronoiMesh
    z_coord : OceanZStarCoordinate or partial-cell coordinate.
    cfg : MLEConfig
    eos : str              "wright" or "linear".
    eos_linear : LinearEOSConfig or None.
    mask : (nCells,)       ocean mask (default all ocean).

    Returns
    -------
    (dT_dt, dS_dt) : each (nCells, nlev)  [degC/s], [PSU/s].
    """
    if cfg.bolus_cfl_cap > 0.0:
        # The vertical-Courant cap needs dt, which the lateral-mixing hook does
        # not expose (identical limitation to the C-grid port).  Raise rather
        # than silently no-op; the bolus is already bounded by H^2 and mu(z).
        raise NotImplementedError(
            "MLEConfig.bolus_cfl_cap > 0 needs the timestep dt, which is not "
            "available in the MPAS lateral-mixing hook. Leave it at 0.0 (the "
            "NEMO-oracle default)."
        )

    nlev = T.shape[1]
    if mask is None:
        mask = jnp.ones((mesh.nCells,), dtype=T.dtype)
    c1 = mesh.cellsOnEdge[0]
    c2 = mesh.cellsOnEdge[1]
    edge_mask = mask[c1] * mask[c2]                      # (nEdges,)

    jacobian = compute_ocean_jacobian(eta, H_bathy, z_coord)   # (nCells,)
    # ACTUAL live layer thickness (partial-cell aware: h_partial·(eta+H)/H_bathy;
    # = dz_ref·jacobian for pure z*).  Used for the MLD/buoyancy criterion, the
    # gdepw(z) for mu, AND the cell volume — so the bolus tendency is EXACTLY
    # conservative against the model's true tracer mass areaCell·h_k (not just
    # against dz_ref·jacobian).  Mirrors the C-grid sibling's h_k.
    dz_live = compute_layer_thickness(eta, H_bathy, z_coord)   # (nCells, nlev)

    # Per-level wet mask (partial cells: level active AND ocean column).
    if hasattr(z_coord, "is_active"):
        wet3d = z_coord.is_active.astype(T.dtype) * mask[:, jnp.newaxis]
    else:
        wet3d = jnp.broadcast_to(mask[:, jnp.newaxis], T.shape).astype(T.dtype)

    # Sub-seafloor T/S fill before EOS (same rationale as GM-MPAS: avoid a
    # spurious rho from the zero-fill convention at inactive levels).
    T_fill, S_fill = T, S
    if hasattr(z_coord, "is_active") and hasattr(z_coord, "bottom_level"):
        _active = z_coord.is_active
        _bot = jnp.clip(z_coord.bottom_level, 0, nlev - 1)
        _row = jnp.arange(T.shape[0])
        T_fill = jnp.where(_active, T, T[_row, _bot][:, None])
        S_fill = jnp.where(_active, S, S[_row, _bot][:, None])

    eos_fn = make_eos_fn(eos, eos_linear)
    fill_fn = lambda field: voronoi_neumann_fill(field, mask, mesh)
    rho_insitu, _rp, _pp = iterate_eos_and_pressure_anomaly(
        T_fill, S_fill, mask, fill_fn, eos_fn,
        z_coord.dz_ref, _RHO_0, constants.g, n_iter=2,
    )                                                   # (nCells, nlev)

    # Reference W-INTERFACE depths [m, positive down] for the NEMO nla10
    # reference-level pick: cumulative reference thicknesses, surface first.
    z_faces = jnp.concatenate([
        jnp.zeros((1,), z_coord.dz_ref.dtype),
        jnp.cumsum(z_coord.dz_ref),
    ])                                                              # (nlev+1,)

    # NEMO rhop: SURFACE-REFERENCED POTENTIAL density (the EOS at zero
    # pressure) drives BOTH the Delta-rho MLD criterion and zbm (tramle.F90;
    # eosbn2 prhop).  In-situ rho here collapsed the ML to the top layer by
    # pure compressibility (~0.14 kg/m^3 per ~30 m >> the 0.01 threshold),
    # zeroing the MLE transport — the test_mle_mpas restratification
    # regression.  The N^2 convection gate below keeps in-situ rho (a local
    # vertical gradient, the standard N^2 approximation).
    rho_pot = eos_fn(T_fill, S_fill, jnp.zeros_like(T_fill))

    # --- MLE mixed-layer depth + ML-mean buoyancy (shared grid-agnostic core) ---
    zmld, bm, in_ml = mle_mld_and_buoyancy(
        rho_pot, dz_live, wet3d,
        z_faces=z_faces,
        rho_c_mle=cfg.rho_c_mle,
        ref_depth_m=cfg.ref_depth_m,
        rho0=constants.rho_ocean,
        grav=constants.g,
    )                                                   # zmld, bm: (nCells,)

    # Neumann-fill bm / zmld so coastal edges see an ocean-neighbour value (the
    # transport is still killed by edge_mask downstream).
    bm_f = voronoi_neumann_fill(bm, mask, mesh)
    zmld_f = voronoi_neumann_fill(zmld, mask, mesh)

    # --- Edge face MLD H_e (NEMO nn_mld_uv) and edge buoyancy gradient ---
    H_e = face_mld(zmld_f[c1], zmld_f[c2], cfg.mld_uv)          # (nEdges,)
    dbm_de = (bm_f[c2] - bm_f[c1]) / mesh.dcEdge                # (nEdges,)
    cap_e = jnp.minimum(cfg.max_grid_scale_m, mesh.dcEdge)     # min(111 km, dcEdge)

    rc_f = mle_coefficient(cfg.ce, cfg.lat_ref_deg)            # constant [s/m]
    # psim_e = rc_f * H_e^2 * dvEdge * dbm/dn * min(111km, dcEdge)  [m^3/s]
    # (dvEdge is the cross-face width, the Voronoi analogue of e2u.)
    # Shared kernel (#518 item 9).
    psim_e = mle_streamfunction_magnitude(
        rc_f, H_e, mesh.dvEdge, dbm_de, cap_e)                  # (nEdges,)

    # --- Convection gate (NEMO nn_conv=1): no MLE where the ML-integrated N^2
    # of either neighbour column is negative (statically unstable). ---
    if cfg.no_mle_in_convection:
        # NEMO's gate sums rn2 — the PROPER (locally-referenced,
        # compressibility-free) buoyancy frequency (tramle.F90:
        # ``zn2 = zn2 + zc*(rn2(jk)+rn2(jk+1))*0.5``).  Differencing
        # IN-SITU rho here carried the compressibility between reference
        # pressures (~6x too stable; the shared helper's docstring) and
        # read deep unstable columns as stable, leaking transport through
        # the gate.  Shared adiabatic-parcel helper; reference pressure =
        # hydrostatic estimate at the reference centre depths (the Veros
        # press = |zt| convention the helper documents).
        z_centers_ref = jnp.cumsum(z_coord.dz_ref) - 0.5 * z_coord.dz_ref
        p_cell = jnp.broadcast_to(
            (_RHO_0 * constants.g) * z_centers_ref[None, :],
            T_fill.shape,
        )
        N2 = compute_buoyancy_frequency_adiabatic(
            T_fill, S_fill, p_cell, z_coord.dz_ref, jacobian,
            eos_fn=eos_fn,
        )                                                          # (nCells, nlev-1)
        iface_in_ml = in_ml[:, :-1]                               # iface k in ML if cell k is
        col_n2 = jnp.sum(iface_in_ml * N2, axis=-1)              # (nCells,) NEMO zn2
        col_n2_f = voronoi_neumann_fill(col_n2, mask, mesh)
        face_n2 = jnp.minimum(col_n2_f[c1], col_n2_f[c2])        # (nEdges,)
        psim_e = jnp.where(face_n2 < 0.0, 0.0, psim_e)

    # --- W-interface depths at edges and mu(z) ---
    # gdepw at cell centres (cumulative live thickness), then averaged to edges.
    ztop = jnp.zeros((T.shape[0], 1), dtype=dz_live.dtype)
    gdepw_cell = jnp.concatenate([ztop, jnp.cumsum(dz_live, axis=1)], axis=1)  # (nCells, nlev+1)
    gdepw_e = cell_to_edge_avg_3d(gdepw_cell, mesh)             # (nEdges, nlev+1)
    inv_He = 1.0 / jnp.maximum(H_e, _EPS_DIV)                   # (nEdges,)
    mu_e = mle_vertical_structure(gdepw_e * inv_He[:, jnp.newaxis])   # (nEdges, nlev+1)

    # Per-level edge wet mask: edge active at level k iff both cells active there.
    if hasattr(z_coord, "is_active"):
        act_e = (z_coord.is_active[c1] * z_coord.is_active[c2])      # (nEdges, nlev)
    else:
        act_e = jnp.ones((mesh.nEdges, nlev), dtype=T.dtype)
    act_e = act_e * edge_mask[:, jnp.newaxis]
    # The interface k sits above T-level k -> active iff T-level k is a wet edge.
    # Surface (k=0) and bottom (k=nlev) interfaces have mu=0, so the deepest
    # interface is forced zero by padding the level mask with a zero column.
    zcol = jnp.zeros((mesh.nEdges, 1), dtype=act_e.dtype)
    wface_e = jnp.concatenate([act_e, zcol], axis=1)            # (nEdges, nlev+1)

    psi_ew = psim_e[:, jnp.newaxis] * mu_e * wface_e            # (nEdges, nlev+1)

    # --- Bolus volume transport at each T-level: dk[psi]  [m^3/s] ---
    utr_e = psi_ew[:, :-1] - psi_ew[:, 1:]                      # (nEdges, nlev)
    utr_e = utr_e * act_e                                       # hard-zero closed/step faces

    # --- Centred tracer face value, flux, conservative divergence ---
    T_face = 0.5 * (T_fill[c1] + T_fill[c2])                    # (nEdges, nlev)
    S_face = 0.5 * (S_fill[c1] + S_fill[c2])
    FxT = utr_e * T_face                                        # (nEdges, nlev) [degC*m^3/s]
    FxS = utr_e * S_face

    vol = mesh.areaCell[:, jnp.newaxis] * dz_live              # (nCells, nlev)
    inv_vol = jnp.where(vol > 0.0, 1.0 / jnp.maximum(vol, _EPS_DIV), 0.0)
    dT_dt = -_bolus_divergence_cell(FxT, mesh) * inv_vol        # (nCells, nlev)
    dS_dt = -_bolus_divergence_cell(FxS, mesh) * inv_vol

    # Tendency already vanishes on dry cells (inv_vol = 0); mask explicitly so a
    # sub-seafloor level carries exactly zero.
    if hasattr(z_coord, "is_active"):
        _active = z_coord.is_active.astype(dT_dt.dtype)
        dT_dt = dT_dt * _active
        dS_dt = dS_dt * _active
    else:
        dT_dt = dT_dt * mask[:, jnp.newaxis]
        dS_dt = dS_dt * mask[:, jnp.newaxis]
    return dT_dt, dS_dt
