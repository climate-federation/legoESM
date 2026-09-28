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

import logging
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from legoesm.grids.latlon import (
    LatLonCGridGeometry,
    create_beta_plane_cgrid_geometry,
    create_latlon_geometry,
)
from legoesm.ocean.constants_config import NEMO_CONSTANTS_CONFIG
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import neumann_fill_cgrid
from legoesm.ocean.fidelity.nemo_io import NemoBeforeState, NemoGrid, NemoState
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanState
from legoesm.ocean.vertical import (
    NemoEENBarotropicOperands,
    nemo_fe3mask_from_tmask,
    create_full_step_coordinate,
    create_z_star_from_thicknesses,
)

from legoesm import constants


class NemoBridgeOutput(NamedTuple):
    geometry: LatLonCGridGeometry
    z_coord: object
    state: LatLonCGridOceanState
    land_mask: np.ndarray          # (n_lat, n_lon) surface wet mask
    # max|geom.f_T - selected reference ff_t|. The selected reference is NEMO
    # by default; the registered bridge-Omega counterfactual scales NEMO ff_t
    # by selected_omega/NEMO_OMEGA without relaxing the guard.
    f_match_max_abs: float


def _nemo_een_barotropic_operands(grid: NemoGrid):
    """Return the raw dyn_cor_2D_init inputs when the mesh carries all of them."""
    required = (grid.ff_f, grid.e3u_0, grid.e3v_0, grid.e3f_0,
                grid.umask, grid.vmask, grid.fmask, grid.hu_0, grid.hv_0,
                grid.e1t, grid.e2t, grid.e1u, grid.e2u,
                grid.e1v, grid.e2v, grid.e1f, grid.e2f)
    if any(value is None for value in required):
        return None
    hf_0 = (np.asarray(grid.e3f_0) * np.asarray(grid.fmask)).sum(axis=-1)
    # dommsk.F90:146-198 freezes fe3mask before lateral-slip fmask changes.
    tmask = np.asarray(grid.tmask, dtype=bool)
    fe3mask = np.asarray(
        nemo_fe3mask_from_tmask(tmask.astype(np.float64)), dtype=np.float64
    )
    values = (grid.ff_f, grid.e3u_0, grid.e3v_0, grid.e3f_0,
              grid.umask, grid.vmask, grid.fmask, fe3mask,
              grid.hu_0, grid.hv_0, hf_0,
              grid.e1t, grid.e2t, grid.e1u, grid.e2u,
              grid.e1v, grid.e2v, grid.e1f, grid.e2f)
    return NemoEENBarotropicOperands(*values)


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


def _u_east_to_face_periodic(nemo_u: np.ndarray) -> np.ndarray:
    """NEMO east-face u -> legoESM ``u_face`` for an i-PERIODIC (re-entrant) grid.

    DINO and other ``ln_Iperio=T`` configs are zonally re-entrant: the west face
    of cell 0 is the east face of the last cell (periodic wrap), NOT a wall.  So
    ``u_face[:, 1:] = nemo_u`` and ``u_face[:, 0] = nemo_u[:, -1]`` (the periodic
    image).  Contrast :func:`_u_east_to_face`, which prepends a zero wall for a
    closed basin (GYRE).  Longitude is axis 1 for both 2-D ``(lat,lon)`` masks and
    3-D ``(lat,lon,nz)`` fields.
    """
    nemo_u = np.asarray(nemo_u)
    return np.concatenate([nemo_u[:, -1:], nemo_u], axis=1)


def _restart_depth_mean_velocity(
    state: NemoState,
    grid: NemoGrid,
    *,
    periodic_i: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return restart ``uu_n/vv_n`` on legoESM U/V faces.

    The identity route reads the pair written at ``restart.F90:175-182`` and
    read at :304-314.  NEMO's :315-323 compatibility branch re-derives it only
    when an older restart lacks ``uu_n``; this helper mirrors that exception,
    logs it, and is never called by the live model step.  Re-derivation is not
    an alternative numerical identity path.
    """
    umap = _u_east_to_face_periodic if periodic_i else _u_east_to_face
    if (state.uu_b is None) != (state.vv_b is None):
        raise ValueError("NEMO restart must carry both uu_n/vv_n or neither")
    if state.uu_b is not None:
        return (
            umap(np.asarray(state.uu_b)[..., None])[..., 0],
            _v_north_to_face(np.asarray(state.vv_b)[..., None])[..., 0],
        )

    required = (grid.e3u_0, grid.e3v_0, grid.umask, grid.vmask,
                grid.hu_0, grid.hv_0)
    if any(value is None for value in required):
        raise ValueError(
            "legacy NEMO restart lacks uu_n/vv_n and mesh_mask lacks the "
            "e3u_0/e3v_0/mask/hu_0/hv_0 operands required by restart.F90:316-323")
    logging.getLogger(__name__).warning(
        "NEMO restart lacks uu_n/vv_n; using restart.F90:316-323 legacy "
        "depth-mean reconstruction (not the NEMO-identity live-step path)")
    u = np.asarray(state.u)
    v = np.asarray(state.v)
    e3u = np.asarray(grid.e3u_0)
    e3v = np.asarray(grid.e3v_0)
    umask = np.asarray(grid.umask)
    vmask = np.asarray(grid.vmask)
    ub = u[..., 0] * e3u[..., 0] * umask[..., 0]
    vb = v[..., 0] * e3v[..., 0] * vmask[..., 0]
    for jk in range(1, u.shape[-1] - 1):
        ub = ub + u[..., jk] * e3u[..., jk] * umask[..., jk]
        vb = vb + v[..., jk] * e3v[..., jk] * vmask[..., jk]
    hu = np.asarray(grid.hu_0)
    hv = np.asarray(grid.hv_0)
    r1_hu = np.divide(1.0, hu, out=np.zeros_like(hu), where=hu > 0.0)
    r1_hv = np.divide(1.0, hv, out=np.zeros_like(hv), where=hv > 0.0)
    return (
        umap((ub * r1_hu)[..., None])[..., 0],
        _v_north_to_face((vb * r1_hv)[..., None])[..., 0],
    )


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

    # Supply NEMO's EXACT analytic T-point depths (gdept_1d) so the
    # nemo_trapezoid PGF quadrature reconstructs NEMO's e3w W-spacings
    # (e3w(1)=2·gdept(1), e3w(k)=gdept(k)−gdept(k−1)) to roundoff.  These
    # differ from the interface-midpoint z_full_ref by up to ~3.6 m on the
    # stretched MI96 grid, which is the entire ~0.5% depth-signed PGF gap.
    z_coord = create_z_star_from_thicknesses(
        np.asarray(grid.e3t_1d),
        t_depth_ref_m=np.asarray(grid.gdept_1d).ravel(),
        nemo_gdept_0_m=grid.gdept_0,
        nemo_gdepw_0_m=grid.gdepw_0,
        nemo_e3t_0_m=grid.e3t_0,
        nemo_e3w_0_m=grid.e3w_0,
        nemo_hu_0_m=grid.hu_0, nemo_hv_0_m=grid.hv_0,
        nemo_e1e2t_m=np.asarray(grid.e1t) * np.asarray(grid.e2t),
        nemo_e1e2u_m=(None if grid.e2u is None else
                      np.asarray(grid.e1u) * np.asarray(grid.e2u)),
        nemo_e1e2v_m=(None if grid.e1v is None else
                      np.asarray(grid.e1v) * np.asarray(grid.e2v)),
        nemo_e2u_m=grid.e2u, nemo_e1v_m=grid.e1v,
        nemo_een_barotropic_m=_nemo_een_barotropic_operands(grid),
        nemo_e3w_source="mesh_reference",
    )
    H_max = float(np.sum(np.asarray(grid.e3t_1d)[:n_wet]))   # depth of the n_wet wet cells

    base = rest_state_latlon_cgrid_ocean(
        geom, z_coord, H_max=H_max, land_mask_override=jnp.asarray(land_mask),
        nemo_prognostic_barotropic_velocity=True,
    )

    # Neumann-fill T/S over land so the 0.0 NEMO stores on masked cells cannot
    # contaminate legoESM's wide high-order tracer stencils (the #480 T=0 bug).
    mask3 = jnp.asarray(tmask)                               # (n_lat, n_lon, nlev) bool
    T_fill = neumann_fill_cgrid(jnp.asarray(state.T), mask3, geom)
    S_fill = neumann_fill_cgrid(jnp.asarray(state.S), mask3, geom)

    u_face = _u_east_to_face(np.asarray(state.u))
    v_face = _v_north_to_face(np.asarray(state.v))
    uu_b_face, vv_b_face = _restart_depth_mean_velocity(
        state, grid, periodic_i=False)
    st = base._replace(
        T=base.T.replace(data=T_fill),
        S=base.S.replace(data=S_fill),
        u=base.u.replace(data=jnp.asarray(u_face)),
        v=base.v.replace(data=jnp.asarray(v_face)),
        eta=base.eta.replace(data=jnp.asarray(state.ssh)),
        uu_b=base.uu_b.replace(data=jnp.asarray(uu_b_face)),
        vv_b=base.vv_b.replace(data=jnp.asarray(vv_b_face)),
    )

    return NemoBridgeOutput(
        geometry=geom, z_coord=z_coord, state=st,
        land_mask=land_mask, f_match_max_abs=f_err,
    )


#: The vertical-ladder selections :func:`effective_vertical_scale_factors`
#: accepts, and the only values ``LEGOESM_NEMO_E3T`` may take. Exported so a
#: harness validates against THIS tuple instead of re-listing the literals and
#: drifting out of step with the function that raises on them.
NEMO_E3T_MODES = ("off", "e3t_only", "gdept_only", "both")


def _level_value(v):
    """One level's reference value: the EXACT value where the level is uniform.

    ``np.mean`` over N identical float64 values is a sum-then-divide, and it
    does not return the value.  Measured on NEMO's DINO R1 mesh before this
    was written: the mean lands 1 ulp off the cell value on 3 of 35 ``e3t_0``
    levels and on 18 of 35 ``gdept_0`` levels -- our reduction rounding the
    ORACLE's own number, which a gate whose bar is "0 cells unequal" refuses.

    DINO's ``e3t_0``/``gdept_0`` are horizontally uniform by construction:
    ``usr_def_zgr`` calls ``zgr_sco_mi96`` on a FLAT column
    (``zflat(:,:) = zHmax``, ``cfgs/DINO/MY_SRC/usrdef_zgr.F90:107-118``), so
    there is a single value per level to take and averaging can only lose it.
    A level that is NOT uniform keeps the mean it has always had, so no mesh
    with real horizontal variation changes behaviour here.
    """
    lo = v.min()
    return float(lo) if lo == v.max() else float(v.mean())


def effective_vertical_scale_factors(grid, tmask, mode=None):
    """Per-level thickness + T-depth the NEMO run ACTUALLY integrates with.

    NEMO integrates with the 3-D scale factors ``e3t_0`` (``key_vco_3d``).
    ``e3t_1d`` is a DIFFERENT, unstretched reference ladder. For DINO the two agree
    in the upper ocean and part company below the ~1000 m re-anchor: they agree to
    roundoff (2.8e-14 m) through k=24, and k=25 is the first level where they
    differ AT ALL, by 3.317 m, its top face sitting at 982.4 m -- DINO's
    ``rn_hco = 1000 m``.  Below that the per-level thickness
    difference REVERSES SIGN once, running -2.1% at k=25 through -8.5% at k=28
    to +14.8% at k=34.  All percentages here are relative to ``e3t_1d``; the
    same deepest-level gap is 70.389 m, which is 14.8% of ``e3t_1d`` and 12.9%
    of ``e3t_0`` -- an earlier version of this docstring "corrected" 12.9% to
    14.8% as an understatement, which was wrong: they are one measurement under
    two denominators, and a percentage here without its denominator is not a
    number.  (All figures measured 2026-08-21 from RUN_TRAJ/mesh_mask.nc.)

    What the redistribution costs, measured rather than asserted:

    * TOTAL column depth is unchanged -- both ladders sum to exactly 4000.000 m
      over the 35 wet levels and to 4506.375 m over all 36.  That holds for the
      ladder THIS FUNCTION BUILDS; a reader who sums raw ``e3t_0`` over all 36
      levels gets 4617.462 m instead, because level 36 has no wet cell anywhere
      and the loop below leaves ``e3t_1d``'s 506.375 m there rather than
      ``e3t_0``'s 617.462.  The model never integrates that level.
    * PER-COLUMN depth is NOT.  It is identical only in the 7442 of 9920 wet
      columns that reach the full 35 levels (75%).  In the other 25% the 1-D
      ladder puts the bottom 70.4-104.2 m too DEEP, and the mean over all wet
      columns is 21.9 m -- so this docstring's long-standing "~22 m too deep"
      is CORRECT and stands; a 2026-08-21 attempt to withdraw it was itself
      withdrawn after measurement.
    * WET VOLUME differs by 4.70e-03 relative to the 1-D ladder's own volume
      (2.546363e+17 vs 2.534390e+17 m3 -- 4.72e-03 against the other
      denominator), which is the "volume 4.7e-03 -> 6.0e-09" the body comment
      below already records.

    Thermal wind integrates density x THICKNESS, and bottom-referenced
    transport integrates it over the column depth, so both the sign-reversing
    per-level error and the 25% of columns whose bottom is misplaced feed
    straight into it.  That is where #1226's ACC deficit is sourced (80% of the
    missing thermal wind below 2000 m).

    Falls back to the 1-D ladder when the mesh_mask predates ``e3t_0`` (GYRE,
    ``key_linssh``, where the two coincide -- which is why this went unnoticed).

    Raises
    ------
    ValueError
        If ``e3t_0`` varies horizontally over wet cells, i.e. the config has
        PARTIAL CELLS (``ln_zps``), which this bridge does not support. Silently
        averaging a thinned bottom cell into a full one would yield a
        plausible-looking but wrong bathymetry.
    """
    e3t = np.asarray(grid.e3t_1d).ravel().astype(np.float64)
    t_depth = np.asarray(grid.gdept_1d).ravel().astype(np.float64)
    # DIAGNOSTIC (#1226, temporary): LEGOESM_NEMO_E3T isolates which half of
    # NEMO's 3-D geometry drives a regression -- the thickness ladder or the
    # T-depth ladder.  "off" (this function's default, see below) | "e3t_only"
    # | "gdept_only" | "both"
    import os as _os
    # DEFAULT IS "off" -- i.e. the KNOWN-WRONG 1-D ladder. This is deliberate
    # and temporary. Adopting NEMO's true e3t_0 thicknesses is CORRECT (it makes
    # legoESM's geometry match NEMO to roundoff: volume 4.7e-03 -> 6.0e-09) but
    # it DESTABILISES the model: from a bit-exact NEMO restart, max|u| grows
    # 0.66 -> 2.2 m/s over 20 days and saturates near 3 m/s, where the 1-D
    # ladder holds 0.60-0.69 indefinitely. Isolated to the THICKNESS ladder --
    # "gdept_only" (NEMO T-depths, 1-D thicknesses) is stable at 0.61, so the
    # depth ladder is innocent.
    # => legoESM is UNSTABLE ON NEMO'S ACTUAL GRID and was stable only because
    #    it ran on a wrong one. That second defect must be found before this can
    #    default to "both". Do NOT flip this default to hide the instability.
    #
    # 2026-08-20/21, #1455: THE INSTABILITY ABOVE DID NOT REPRODUCE, and the
    # same measurements show this default is expensive.
    #
    # Non-reproduction. Four 90-day DINO twin arms from the day-180 restart
    # (corrected seasonal clock, --bridge-before), differing ONLY in this mode
    # and bit-identical at day 0, were ALL STABLE over 2880 steps: day-90
    # max|u| = 0.6332 ("off"), 0.6341 ("e3t_only"), 0.6344 ("gdept_only"),
    # 0.6347 ("both"). Re-measured under an fp64 precision policy on branch
    # fidelity/dino-step-walk, the two end arms give max|u| = 0.6332 ("off")
    # and 0.6350 ("both") -- so 4/4 arms stable at 0.633-0.635 m/s, agreeing to
    # three decimals, none growing, nothing near the 2.2-3 m/s above. The
    # comment does not record the state or configuration its measurement came
    # from, so this is a NON-REPRODUCTION under ONE configuration, not proof
    # that it was never true -- but it IS the stated blocker, and it did not
    # fire.
    #
    # Cost of this default, day-90 circumpolar (channel-band) transport error
    # vs NEMO, fp64, mean reduction over longitudes 2..-2: "off" +2.93 Sv
    # against "both" +0.29 Sv. Full-section ACC error over the same pair:
    # +1.87 Sv against -0.60 Sv. The four-arm fp32 sweep that first ranked them
    # put the halves at "e3t_only" +0.69 and "gdept_only" +2.89 Sv; those two
    # numbers have NOT been re-measured at fp64 and are quoted only for the
    # split they show -- the THICKNESS ladder fixes the barotropic component
    # and the DEPTH ladder the baroclinic one, so neither half alone is the
    # answer, and "gdept_only" is in any case an internally INCONSISTENT grid
    # (cells from one ladder, T-points from the other -- 110 m off the cell
    # centre at k=32).
    #
    # This default is UNCHANGED and still resolves to the 1-D reference ladder,
    # because the note above is a non-reproduction rather than a refutation and
    # this function serves every caller, not only oracle twins. What DID change
    # (2026-08-21) is scoped strictly to bridged DINO twin runs:
    # scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py resolves this
    # variable to "both" when nothing sets it, on the argument that a twin only
    # isolates SCHEME differences if both models stand on the same grid. It does
    # NOT write this environment variable -- it passes the resolved mode down as
    # the ``e3t_mode`` argument and stamps it into its output. An explicit
    # LEGOESM_NEMO_E3T still wins there. See
    # kamm_twin_90d.resolve_ladder_mode and
    # scripts/validate/ocean_fidelity/dino_1226/d180_step_walk.py for the
    # measurements, the retractions attached to them, and the discriminating run
    # that is still owed (pin the EOS depth back to the 1-D ladder on the
    # "gdept_only" arm to find out whether the equation of state, or N^2/the
    # pressure-gradient geometry, owns the response).
    _mode = (mode if mode is not None
             else _os.environ.get("LEGOESM_NEMO_E3T", "off"))
    if _mode not in NEMO_E3T_MODES:
        raise ValueError(
            f"unknown vertical-scale-factor mode {_mode!r}; expected one of "
            + ", ".join(repr(m) for m in NEMO_E3T_MODES))
    e3t3 = getattr(grid, "e3t_0", None)
    if e3t3 is None or _mode == "off":
        return e3t, t_depth, "e3t_1d"
    e3t3 = np.asarray(e3t3)
    nlev = e3t3.shape[-1]
    # EVERY level takes NEMO's own value, the permanently-dry one included.
    # A level with no wet cell used to keep the OTHER ladder's number, which
    # left the deepest level 111.088 m away from NEMO's e3t_0 and its T-depth
    # 55.544 m away.  Nothing on this card integrates that level, but it is
    # still carried state -- it sets H_max and the three deepest reference
    # interfaces -- and "NEMO's grid except one level" is not NEMO's grid.
    # NEMO's e3t_0/gdept_0 are horizontally uniform at every level here, dry
    # levels included, so a dry level HAS a well-defined NEMO value; the
    # uniformity check below now covers those levels too rather than skipping
    # them, so a partial-cell grid cannot slip through on a dry level.
    # tmask arrives as float from the mesh_mask reader and as bool from some
    # callers; a float array used as an index raises, so normalise once.
    wet = np.asarray(tmask) > 0.5
    lev_any = wet.any(axis=(0, 1))
    everywhere = np.ones(wet.shape[:2], dtype=bool)

    def _sel(k):
        return wet[:, :, k] if lev_any[k] else everywhere

    spread = np.zeros(nlev)
    for k in range(nlev):
        v = e3t3[:, :, k][_sel(k)]
        spread[k] = float(v.max() - v.min())
    if spread.max() > 1.0e-6:
        raise ValueError(
            f"mesh_mask e3t_0 varies horizontally (max spread {spread.max():.3e} "
            "m over a level): this is a PARTIAL-CELL (ln_zps) grid, which "
            "bridge_nemo_to_legoesm_topo does not support."
        )
    out_e3t = e3t.copy()
    for k in range(nlev):
        out_e3t[k] = _level_value(e3t3[:, :, k][_sel(k)])
    if _mode == "gdept_only":
        out_e3t = e3t.copy()            # keep the 1-D thickness ladder
    gd3 = getattr(grid, "gdept_0", None)
    out_td = t_depth.copy()
    if gd3 is not None and _mode in ("both", "gdept_only"):
        gd3 = np.asarray(gd3)
        gspread = max(float(np.ptp(gd3[:, :, k][_sel(k)])) for k in range(nlev))
        if gspread > 1.0e-6:
            raise ValueError(
                f"mesh_mask gdept_0 varies horizontally (max spread "
                f"{gspread:.3e} m over a level): partial cells are not "
                "supported by this bridge.")
        for k in range(nlev):
            out_td[k] = _level_value(gd3[:, :, k][_sel(k)])
    return out_e3t, out_td, "e3t_0"


_FP32_BRIDGE_WARNED = False


def _warn_if_not_fp64() -> None:
    """One-shot loud warning when an oracle bridge is built in single precision.

    NEMO/MITgcm/Veros are fp64.  RUNNING legoESM in fp32 is a legitimate
    performance choice, so this is NOT an error -- but COMPARING against an
    oracle in fp32 measures our own rounding (f32 eps = 1.19e-7), and that is
    how the f32 depth ladder silently corrupted every #1226 measurement for
    weeks (it rounded NEMO's f64 gdept_1d to ~7 digits; median |rel| 2.555e-8 =
    0.21 x f32 eps).  ``JAX_ENABLE_X64=1`` does NOT change the policy.

    Comparison probes should use the HARD gate
    :func:`legoesm.ocean.fidelity.precision_gate.require_fp64` instead of
    relying on this warning.
    """
    global _FP32_BRIDGE_WARNED
    if _FP32_BRIDGE_WARNED:
        return
    import jax.numpy as _jnp
    from legoesm.core.precision import get_policy
    if _jnp.dtype(get_policy().control) == _jnp.float64:
        return
    _FP32_BRIDGE_WARNED = True
    import warnings
    warnings.warn(
        "NEMO oracle bridge built under a NON-fp64 precision policy "
        f"(control={_jnp.dtype(get_policy().control).name}). Model RUNS in "
        "fp32 are fine, but any COMPARISON against the oracle is then "
        "measuring float32 rounding, not physics (f32 eps = 1.19e-7). "
        "JAX_ENABLE_X64=1 does NOT change this -- set "
        "PrecisionPolicy.fp64() via legoesm.core.precision.set_policy, and "
        "use ocean.fidelity.precision_gate.require_fp64 for a hard gate.",
        RuntimeWarning, stacklevel=3,
    )


def detect_metric_convention(grid, rtol: float = 1e-12) -> str:
    """Read the oracle's OWN grid to decide its horizontal metric convention.

    NEMO's ``mesh_mask`` carries both ``e1t`` and ``e2t``, so the convention is
    OBSERVABLE rather than something to assume:

    * DINO's ``usr_def_hgr.F90:111-119`` sets ``pe2t = pe1t =
      ra*rad*COS(phi)*rn_e1_deg`` — Mercator conformality imposed analytically
      — so ``e1t == e2t`` EXACTLY  → ``"nemo_isotropic"``.
    * A config carrying a genuine finite-difference meridional metric has
      ``e1t != e2t``  → ``"exact"``.

    Detecting beats a hard default because this bridge also serves
    non-isotropic NEMO configs (GYRE), where forcing ``"nemo_isotropic"`` would
    be wrong.  It also beats trusting a recipe card, which can drift out of
    step with the mesh_mask actually being read.

    Getting this wrong is not subtle in its consequences: the shipped
    ``"exact"`` default mismatched NEMO's ``e2v`` by median 2.798e-05, which
    ``ldf_slp``'s ``vslp`` divides by — worth 4 orders of magnitude on that row
    (#1226).
    """
    e1t = np.asarray(grid.e1t, dtype=np.float64)
    e2t = np.asarray(grid.e2t, dtype=np.float64)
    rel = np.abs(e1t - e2t) / np.maximum(np.abs(e2t), 1e-30)
    return "nemo_isotropic" if float(rel.max()) <= rtol else "exact"


def bridge_nemo_to_legoesm_topo(
    grid: NemoGrid,
    state: NemoState,
    *,
    periodic_i: bool = True,
    # A NEMO bridge defaults to NEMO'S EARTH.  It used to default to
    # ``legoesm.constants.Omega``, a four-significant-figure rounding of the
    # same physical constant, and no DINO caller overrode it -- so the ENTIRE
    # oracle-matching lane (the 90-day twin and every fidelity probe in
    # scripts/validate/ocean_fidelity/dino_1226/) built its Coriolis arrays on
    # a planet 1.578e-05 away from the one the oracle integrated, while the
    # card's own pinned rate reached the config-side consumers.  Two Earths in
    # one run, and the config-vs-geometry split is exactly the one an oracle
    # harness must not have.  Measured, per site, by
    # coriolis_omega_routing_audit.py (#1455 next-action 1, STEP 0).
    omega: float = NEMO_CONSTANTS_CONFIG.Omega,
    radius: float = constants.R_earth,
    # Tightened from 1e-3 once the rate above stopped being wrong.  The old
    # bound was 63x too loose to see the 1.578e-05 rotation-rate gap, which is
    # how a whole lane ran on the wrong planet with a green guard.  The
    # built f_T now matches NEMO's ff_t to ~1e-16 relative on DINO; 1e-9 keeps
    # four orders of headroom for a different NEMO mesh's own roundoff while
    # still failing on any real constant or latitude error.
    f_rtol: float = 1e-9,
    # Counterfactual-only validation axis. ``nemo`` is the historical default
    # and remains bit-identical. ``selected_omega`` validates against NEMO's
    # own ff_t scaled by the explicitly supplied omega, so a registered old-
    # Earth reproduction remains fail-closed instead of loosening f_rtol.
    f_reference_mode: str = "nemo",
    full_step: bool = False,
    metric_convention: str = "auto",
    vface_zonal_metric_evaluation: str = "nemo_vpoint",
    coriolis_placement: str = "cell_average",
    e3t_mode: str | None = None,
    nemo_e3w_source: str = "mesh_reference",
    carry_native_lat_deg: bool = False,
) -> NemoBridgeOutput:
    """Bridge a NEMO **Mercator + topography** config (e.g. DINO) to legoESM.

    Unlike :func:`bridge_nemo_to_legoesm` — which reconstructs a *beta-plane*
    geometry and rejects non-flat bathymetry (GYRE, ``key_vco_1d``) — this builds
    the legoESM geometry directly from NEMO's own mesh arrays and carries the
    column-varying bottom depth, so it handles:

    * **Mercator geometry** with real ``f(φ) = 2Ω sin φ`` via
      :func:`create_latlon_geometry` (``lat_1d``/``lon_1d`` from ``gphit``/``glamt``,
      exact meridional faces from ``gphiv``).  The built ``dx_T``/``f_T`` match
      NEMO's ``e1t``/``ff_t`` by construction (verified to ``f_rtol``).
    * **Full-step-z topography** (``ln_zco``, ``ln_zps=F``): a per-column bottom
      depth ``H_bathy`` from the 3-D ``tmask`` (no partial cells), so bowl / ridge
      / sill bathymetry is represented.  The guard below checks mask TOPOLOGY only
      (no interior holes); it CANNOT detect ``ln_zps`` partial cells (their mask is
      identical to full-step), and ``H_bathy`` uses the 1-D reference ``e3t_1d`` —
      so the **caller must guarantee ``ln_zps=F``**.  A per-cell ``e3t`` path
      (not read by :mod:`nemo_io`) would be needed for partial cells.
    * **i-periodic** (``ln_Iperio``) zonal boundaries via
      :func:`_u_east_to_face_periodic`; set ``periodic_i=False`` for a closed
      basin.

    Certified against a NEMO DINO 12-step + 2000-step trend dump: the interior
    hydrostatic-PGF, Coriolis, EEN-vorticity and Hollingsworth-KE tendencies match
    to correlation 1.000 (see ``docs/ocean/fidelity``).  KNOWN LIMITATIONS: (a) the
    single redundant periodic-wrap u-face (columns 0 / n_lon, which are the same
    physical face) uses legoESM's closed-basin face-storage convention rather than
    the periodic roll — exclude it from a full-domain face comparison; (b) the
    caller MUST set ``eos_depth="geometric"`` on the probe/model config for the
    NEMO S-EOS depth argument to match (the ``insitu`` default gives a
    depth-proportional density error via the thermobaric ``μ1·zh`` term); (c) z*
    thickness metric moves with η whereas NEMO ``key_qco`` differs at the
    machine-precision tier (same caveat as the flat-bottom bridge).

    Parameters
    ----------
    grid, state : NemoGrid, NemoState
        Halo-stripped NEMO mesh + restart from :mod:`nemo_io`.  ``grid.gphiv``
        (V-point latitudes) is required for exact meridional faces.
    periodic_i : bool
        ``True`` for a zonally re-entrant grid (``ln_Iperio``); ``False`` closes
        the west/east boundaries with walls.
    carry_native_lat_deg : bool
        Opt in to carrying NEMO's native degree-valued ``gphit`` array for a
        literal oracle consumer. The default is ``False`` so generic bridge
        geometry retains its historical pytree structure.
    f_rtol : float
        Max relative error tolerance between the built ``f_T`` and NEMO ``ff_t``.
    f_reference_mode : {"nemo", "selected_omega"}, optional
        Reference used by the ``f_T`` guard. ``"nemo"`` (default) compares
        directly with NEMO ``ff_t`` and is bit-identical to the prior bridge.
        ``"selected_omega"`` compares with ``ff_t`` scaled by
        ``omega / NEMO_CONSTANTS_CONFIG.Omega``. It exists only for explicit,
        hash-stamped counterfactual reproduction and does not relax
        ``f_rtol``.
    metric_convention : {"exact", "nemo_isotropic"}, optional (#1226)
        Forwarded to :func:`create_latlon_geometry`. Default ``"exact"``
        (the true finite-difference T/u-face metric legoESM has always
        built here — BIT-IDENTICAL for every existing caller of this
        bridge). ``"nemo_isotropic"`` reproduces NEMO's own
        ``usr_def_hgr.F90`` DINO closed-form T/u-face metric
        (``pe1t = pe2t``) instead of the exact one this bridge computes
        from ``gphiv`` -- lets a fidelity probe compare against NEMO on
        NEMO's OWN metric convention rather than legoESM's (geometrically
        more exact but less NEMO-faithful) reconstruction. Does not touch
        the v-face metric (#516) or the Coriolis/``f_rtol`` check below,
        which reads ``geom.f_T`` (unaffected by this flag).
    vface_zonal_metric_evaluation : {"legacy_tracer_midpoint", "nemo_vpoint"}
        Forwarded to :func:`create_latlon_geometry`.  This isolates only the
        #1455 V-face zonal-width reconstruction while holding the detected
        T/u metric convention and all other geometry fixed.

    ``e3t_mode`` selects which vertical ladder to build on, forwarded verbatim to
    :func:`effective_vertical_scale_factors`. ``None`` (the default, and the
    behaviour every existing caller keeps) falls back to the ``LEGOESM_NEMO_E3T``
    environment variable and, failing that, to the 1-D reference ladder. Passing
    it EXPLICITLY is strictly better for a caller that knows what it wants: it
    removes the need to mutate a process-global variable that this function and
    every concurrent caller share, and it lets two different ladders be built in
    one process without either one silently inheriting the other's setting.

    PARTIAL-CELL CAVEAT, worth knowing before selecting a mode: the
    horizontal-spread guard that rejects an ``ln_zps`` grid lives past the
    ``"off"`` early return, so ``"off"`` accepts a partial-cell mesh_mask
    silently while ``"e3t_only"``, ``"gdept_only"`` and ``"both"`` all raise on
    it. Selecting NEMO's own ladders therefore ADDS a guard rather than removing
    one; ``"off"`` is the mode that still relies on the caller's ``ln_zps=F``
    promise.

    Raises
    ------
    ValueError
        If ``gphiv`` is missing, the bathymetry is not full-step, or the built
        Coriolis does not match NEMO ``ff_t`` to ``f_rtol``.
        Also if ``e3t_mode`` is not ``None`` or one of ``NEMO_E3T_MODES``
        (checked at entry, before any geometry is built).
    """
    # Validate HERE, on the static argument, rather than ~180 lines further in
    # when the vertical grid is built: a typo should stop the call, not surface
    # after the geometry has been constructed.
    if nemo_e3w_source not in ("mesh_reference", "depth_difference"):
        raise ValueError(
            f"unknown nemo_e3w_source {nemo_e3w_source!r}; expected "
            "'mesh_reference' or 'depth_difference'")
    if e3t_mode is not None and e3t_mode not in NEMO_E3T_MODES:
        raise ValueError(
            f"unknown e3t_mode {e3t_mode!r}; expected None or one of "
            + ", ".join(repr(m) for m in NEMO_E3T_MODES))
    if f_reference_mode not in ("nemo", "selected_omega"):
        raise ValueError(
            "f_reference_mode must be 'nemo' or 'selected_omega', got "
            f"{f_reference_mode!r}")
    # metric_convention="auto" (DEFAULT): ASK THE ORACLE instead of assuming.
    # NEMO's mesh_mask carries e1t and e2t, so the convention is observable:
    # DINO's usr_def_hgr sets pe2t = pe1t (Mercator conformality imposed
    # analytically), giving e1t == e2t EXACTLY, whereas a config with a genuine
    # finite-difference meridional metric has e1t != e2t.  Detecting beats a
    # hard default because this bridge also serves non-isotropic NEMO configs
    # (GYRE), where forcing "nemo_isotropic" would be wrong.
    if metric_convention == "auto":
        metric_convention = detect_metric_convention(grid)

    _warn_if_not_fp64()
    if grid.gphiv is None:
        raise ValueError(
            "bridge_nemo_to_legoesm_topo requires grid.gphiv (V-point latitudes) "
            "for exact meridional cell faces; read the mesh_mask with a build that "
            "carries gphiv (read_nemo_mesh_mask populates it when present)."
        )

    gphit = np.asarray(grid.gphit)
    glamt = np.asarray(grid.glamt)
    n_lat, n_lon = gphit.shape
    lat_1d = np.deg2rad(gphit[:, 0])            # Mercator: lat varies with j only
    lon_1d = np.deg2rad(glamt[0, :])            #           lon varies with i only
    # Face latitudes: gphiv[j] = north face of cell j -> face[j+1]; the south face
    # of cell 0 by half-cell reflection about the cell centre.
    gphiv = np.deg2rad(np.asarray(grid.gphiv)[:, 0])
    lat_face = np.concatenate([[2.0 * lat_1d[0] - gphiv[0]], gphiv])  # (n_lat+1,)

    geom = create_latlon_geometry(
        n_lat, n_lon, radius=radius, omega=omega,
        lat_1d=jnp.asarray(lat_1d), lon_1d=jnp.asarray(lon_1d),
        lat_face_1d=jnp.asarray(lat_face),
        metric_convention=metric_convention,
        vface_zonal_metric_evaluation=vface_zonal_metric_evaluation,
        # Where the vertex Coriolis is EVALUATED.  Default "cell_average" is
        # bit-identical to every bridge caller; "face_latitude" reproduces
        # NEMO's own ff_f convention (2*omega*sin(gphif)).  See
        # create_latlon_geometry's docstring for the measured gap.
        coriolis_placement=coriolis_placement,
    )
    # Preserve native degrees only for an explicitly selected oracle card.
    # Generic NEMO bridges retain the historical geometry pytree exactly;
    # reconstructing degrees(lat_T) is nevertheless too lossy for DINO's
    # literal latitude-dependent etau profile.
    if carry_native_lat_deg:
        geom = geom._replace(native_lat_T_deg=jnp.asarray(gphit))
    # Partial-periodic seam wall (NEMO DINO): ALL interior cells are wet,
    # but the zonal seam u-face is closed outside the ACC channel — carried
    # on the geometry so every mask derivation (2-D/3-D face, vertex,
    # barotropic diffusion) reads it via ``getattr(grid, "seam_wall_rows")``.
    # Only meaningful for a re-entrant (periodic_i) grid; a closed basin
    # already has real west/east walls.  None on grids without a partial
    # seam → fully periodic (byte-identical).
    seam_wall_rows = getattr(grid, "seam_wall_rows", None) if periodic_i else None
    if seam_wall_rows is not None:
        seam_wall_rows = jnp.asarray(seam_wall_rows)
        if seam_wall_rows.shape != (n_lat,):
            raise ValueError(
                f"NEMO seam_wall_rows shape {seam_wall_rows.shape} != (n_lat,)="
                f"({n_lat},); the halo-derived seam profile must span the "
                "interior latitude rows."
            )
        geom = geom._replace(seam_wall_rows=seam_wall_rows)
    # Verify the built metrics + Coriolis reproduce NEMO's own arrays (guards a
    # wrong omega/lat/lon/radius/face build).  Relative because Mercator f/e1
    # span the whole latitude range.  dx_T (= R·dλ·cos φ) matches NEMO e1t to
    # roundoff; dy_T (from the reconstructed cell faces) matches e2t only to the
    # Mercator centre-vs-face residual (~0.4% on the stretched grid), so its guard
    # is loose — tight enough to catch a face sign-flip / off-by-one (which is
    # O(100%)), loose enough to pass the reconstruction residual.
    f_built = np.asarray(geom.f_T)
    f_nemo = np.asarray(grid.ff_t)
    if f_reference_mode == "nemo":
        f_reference = f_nemo
    else:
        f_reference = f_nemo * (float(omega) / NEMO_CONSTANTS_CONFIG.Omega)
    f_scale = float(np.max(np.abs(f_reference)))
    f_err = float(np.max(np.abs(f_built - f_reference)))
    # PRECISION-AWARE BOUND.  ``f_rtol`` is tight enough (1e-9) to catch a
    # rotation-rate or latitude error in fp64, which is the precision every
    # oracle comparison runs at.  The geometry is stored at the PRECISION
    # POLICY's dtype, though, and in fp32 the array itself only carries ~1.2e-07
    # relative -- so a fixed 1e-9 would be unsatisfiable by construction and
    # would fail on rounding rather than on physics.  The effective bound is
    # therefore the looser of the two, and it is REPORTED so nobody reads an
    # fp32 pass as an fp64 one.
    _eps = float(np.finfo(f_built.dtype).eps) if np.issubdtype(
        f_built.dtype, np.floating) else 0.0
    _bound = max(f_rtol, 8.0 * _eps)
    if f_err > _bound * f_scale:
        raise ValueError(
            f"Mercator Coriolis mismatch vs {f_reference_mode} ff_t reference: "
            f"max|Δ|={f_err:.3e} > "
            f"{_bound:.1e}·{f_scale:.3e} (relative {f_err / f_scale:.3e}; "
            f"f_rtol={f_rtol:.1e}, dtype={f_built.dtype}, 8*eps="
            f"{8.0 * _eps:.1e}). Check gphit/omega -- a relative gap near "
            "1.58e-05 is legoESM's rounded constants.Omega against NEMO's own "
            "2*pi/rsiday, which is what this bound was tightened to catch."
        )
    if f_reference_mode == "selected_omega":
        f_nemo_scale = float(np.max(np.abs(f_nemo)))
        unscaled_err = float(np.max(np.abs(f_built - f_nemo)))
        print(
            "BRIDGE OMEGA COUNTERFACTUAL: selected_omega validation PASS; "
            f"omega={float(omega):.17g} reference_scale="
            f"{float(omega) / NEMO_CONSTANTS_CONFIG.Omega:.17g} "
            f"max|f_T-NEMO ff_t|={unscaled_err:.17g} "
            f"relative={unscaled_err / f_nemo_scale:.17g}",
            flush=True,
        )
    # SEPARATE BOUND, deliberately.  ``f_rtol`` was tightened from 1e-3 to 1e-9
    # to catch a rotation-rate error in the CORIOLIS check above, and it was
    # shared with THIS check (the ``dy_T`` guard below always carried its own
    # hardcoded 5e-2 and was never affected -- an earlier version of this
    # comment said "two others", which adversarial review corrected to one).
    # 1e-9 is far tighter than the reconstruction residual this guard tolerates
    # by design, so it keeps the bound it was calibrated on.
    _metric_rtol = 1e-3
    dx_err = float(np.max(np.abs(np.asarray(geom.dx_T) - grid.e1t)))
    if dx_err > _metric_rtol * float(np.max(np.abs(grid.e1t))):
        raise ValueError(
            f"Mercator dx_T mismatch vs NEMO e1t: max|Δ|={dx_err:.3e}. Check "
            "glamt (lon-separable?) / radius."
        )
    dy_err = float(np.max(np.abs(np.asarray(geom.dy_T) - grid.e2t)))
    if dy_err > 5e-2 * float(np.max(np.abs(grid.e2t))):   # loose: catches face flip
        raise ValueError(
            f"Mercator dy_T mismatch vs NEMO e2t: max|Δ|={dy_err:.3e}. Check "
            "gphiv / lat_face reflection."
        )

    # --- Full-step-z topography from the 3-D tmask -------------------------
    # NB this checks tmask TOPOLOGY only — that every wet column is wet for its
    # top k_bot cells with no interior holes / dry-surface-over-wet.  It does NOT
    # (and cannot) detect ``ln_zps`` PARTIAL cells: a partial cell keeps tmask=1
    # on the thinned bottom cell, so its mask is byte-identical to a full-step
    # column.  H_bathy here is built from the 1-D reference e3t_1d, i.e. the
    # FULL-STEP bottom depth — a real ln_zps config would silently get the wrong
    # bathymetry.  The caller MUST guarantee ln_zps=F / ln_zco=T (DINO is ln_zco).
    # Detecting/correcting partial cells needs the 3-D e3t (not read by nemo_io).
    tmask = np.asarray(grid.tmask) > 0.5                 # (n_lat, n_lon, nlev)
    e3t_1d = np.asarray(grid.e3t_1d).ravel()
    land_mask = tmask[:, :, 0]                           # surface wet
    k_bot = tmask.sum(axis=2).astype(int)               # wet levels per column
    kk = np.arange(tmask.shape[2])[None, None, :]
    top_contig = (kk < k_bot[:, :, None]) & land_mask[:, :, None]
    if not np.array_equal(tmask, top_contig):
        raise ValueError(
            "NEMO tmask topology is not full-step (interior masked cells / "
            "dry-surface-over-wet): bridge_nemo_to_legoesm_topo assumes ln_zco "
            "full-step-z. NB partial cells (ln_zps) are NOT detectable from the "
            "mask — the caller must guarantee ln_zps=F."
        )
    # NEMO integrates with e3t_0, not the 1-D ladder e3t_1d -- see
    # effective_vertical_scale_factors for why this matters (#1226).
    e3t_1d, _t_depth, _e3t_src = effective_vertical_scale_factors(
        grid, tmask, mode=e3t_mode)

    depth_cum = np.cumsum(e3t_1d)                        # bottom-interface depth
    H_bathy = np.where(
        k_bot > 0, depth_cum[np.clip(k_bot - 1, 0, len(e3t_1d) - 1)], 0.0)

    z_coord = create_z_star_from_thicknesses(
        e3t_1d, t_depth_ref_m=_t_depth,
        nemo_gdept_0_m=grid.gdept_0,
        nemo_gdepw_0_m=grid.gdepw_0,
        nemo_e3t_0_m=grid.e3t_0,
        nemo_e3w_0_m=grid.e3w_0,
        nemo_hu_0_m=grid.hu_0, nemo_hv_0_m=grid.hv_0,
        nemo_e1e2t_m=np.asarray(grid.e1t) * np.asarray(grid.e2t),
        nemo_e1e2u_m=(None if grid.e2u is None else
                      np.asarray(grid.e1u) * np.asarray(grid.e2u)),
        nemo_e1e2v_m=(None if grid.e1v is None else
                      np.asarray(grid.e1v) * np.asarray(grid.e2v)),
        nemo_e2u_m=grid.e2u, nemo_e1v_m=grid.e1v,
        nemo_een_barotropic_m=_nemo_een_barotropic_operands(grid),
        nemo_e3w_source=nemo_e3w_source,
    )

    # NEMO ln_zco FULL-STEP-z: fixed reference levels everywhere + a
    # STAIRCASE of dry bottom cells below k_bot (usrdef_zgr.F90 zgr_zco_3d
    # e3t=pe3t_1d + zgr_msk_top_bot k_bot).  Wrap the plain z* coord into a
    # full-step OceanPartialCellCoordinate built DIRECTLY from NEMO's own
    # tmask column count (k_bot) — bit-faithful to the staircase, no float
    # rounding at the level interfaces.  Default off ⇒ the legacy pure-z*
    # (all levels stretched, no dry cells) path is byte-identical.
    if full_step:
        z_coord = create_full_step_coordinate(
            z_coord, bottom_level=jnp.asarray(k_bot - 1, dtype=jnp.int32),
        )

    base = rest_state_latlon_cgrid_ocean(
        geom, z_coord,
        land_mask_override=jnp.asarray(land_mask),
        H_bathy_override=jnp.asarray(H_bathy.astype(np.float64)),
        nemo_prognostic_barotropic_velocity=True,
    )

    # Surface face masks from NEMO umask/vmask (periodic-wrap for re-entrant i).
    umap = _u_east_to_face_periodic if periodic_i else _u_east_to_face
    umask_s = (np.asarray(grid.umask)[:, :, 0] > 0.5).astype(np.float64)
    vmask_s = (np.asarray(grid.vmask)[:, :, 0] > 0.5).astype(np.float64)
    umask_face = umap(umask_s[:, :, None])[:, :, 0]
    vmask_face = _v_north_to_face(vmask_s[:, :, None])[:, :, 0]
    # Close the seam u-face on walled rows so the bridge's own state is
    # self-consistent with the geometry seam wall (NEMO's interior umask
    # is filled wet at the seam by the periodic lbc_lnk — the wall lives
    # only in the halo tmask, so re-impose it here).
    if seam_wall_rows is not None:
        _open = (1.0 - np.asarray(seam_wall_rows)).astype(umask_face.dtype)
        umask_face[:, 0] *= _open
        umask_face[:, -1] *= _open

    # Neumann-fill T/S over land (NEMO stores 0.0 on masked cells; the wide
    # high-order tracer stencils must not see it — the #480 T=0 bug).
    mask3 = jnp.asarray(tmask)
    T_fill = neumann_fill_cgrid(jnp.asarray(state.T), mask3, geom)
    S_fill = neumann_fill_cgrid(jnp.asarray(state.S), mask3, geom)
    u_face = umap(np.asarray(state.u))
    v_face = _v_north_to_face(np.asarray(state.v))
    uu_b_face, vv_b_face = _restart_depth_mean_velocity(
        state, grid, periodic_i=periodic_i)

    st = base._replace(
        T=base.T.replace(data=T_fill),
        S=base.S.replace(data=S_fill),
        u=base.u.replace(data=jnp.asarray(u_face)),
        v=base.v.replace(data=jnp.asarray(v_face)),
        eta=base.eta.replace(data=jnp.asarray(state.ssh)),
        uu_b=base.uu_b.replace(data=jnp.asarray(uu_b_face)),
        vv_b=base.vv_b.replace(data=jnp.asarray(vv_b_face)),
        u_mask=base.u_mask.replace(data=jnp.asarray(umask_face)),
        v_mask=base.v_mask.replace(data=jnp.asarray(vmask_face)),
    )

    return NemoBridgeOutput(
        geometry=geom, z_coord=z_coord, state=st,
        land_mask=land_mask, f_match_max_abs=f_err,
    )


def bridge_before_state_topo(
    br: NemoBridgeOutput,
    grid: NemoGrid,
    before: NemoBeforeState,
    *,
    periodic_i: bool = True,
) -> LatLonCGridOceanState:
    """Populate ``br.state``'s leap-frog BEFORE fields (Nbb) from a NEMO restart.

    NEMO's Modified-Leap-Frog restart always carries a THIRD time level
    (``tb/sb/ub/vb`` (+``sshb``/``utau_b``/``vtau_b``)), one full step behind
    the now-level (``tn/sn/...``) fields :func:`bridge_nemo_to_legoesm_topo`
    already bridges onto ``br.state``. This populates
    ``state.{T,S,u,v,eta}_before`` (+ ``tau_x_prev``/``tau_y_prev`` when the
    restart carries ``utau_b``/``vtau_b``) with the SAME face-staggering /
    Neumann-land-fill conventions as the now-level bridge, so a twin using
    this state is an EXACT leap-frog entry state (matches NEMO's own three
    time levels), not a forward-Euler cold start.

    Must be called with the SAME ``grid``/``periodic_i`` used to build
    ``br`` (no independent re-derivation of the mesh/mask).

    Parameters
    ----------
    br : NemoBridgeOutput
        Output of :func:`bridge_nemo_to_legoesm_topo` on the SAME ``grid``.
    grid : NemoGrid
        The mesh_mask this ``br`` was bridged from (for ``tmask``).
    before : NemoBeforeState
        From :func:`nemo_io.read_nemo_restart_before` on the SAME restart
        file ``br.state`` was bridged from.
    """
    tmask = np.asarray(grid.tmask) > 0.5
    mask3 = jnp.asarray(tmask)
    umap = _u_east_to_face_periodic if periodic_i else _u_east_to_face

    T_fill = neumann_fill_cgrid(jnp.asarray(before.T), mask3, br.geometry)
    S_fill = neumann_fill_cgrid(jnp.asarray(before.S), mask3, br.geometry)
    u_face = umap(np.asarray(before.u))
    v_face = _v_north_to_face(np.asarray(before.v))

    st = br.state
    replacements = dict(
        T_before=st.T.replace(data=T_fill),
        S_before=st.S.replace(data=S_fill),
        u_before=st.u.replace(data=jnp.asarray(u_face)),
        v_before=st.v.replace(data=jnp.asarray(v_face)),
        eta_before=st.eta.replace(
            data=jnp.asarray(before.ssh if before.ssh is not None else st.eta.data)),
    )
    # utau_b/vtau_b (T-point, before-level wind stress): only set when the
    # restart carries them. When absent (older NEMO builds without these
    # fields), leave tau_x_prev/tau_y_prev at their None default --
    # ``_leapfrog_step``'s own forward-Euler-start branch (state.u_before is
    # the ONLY None-gate it checks) then seeds "before := now" on step 1
    # (NEMO nit000 convention, sbcmod.F90:568-573) exactly as an un-bridged
    # cold start would, so barotropic_forcing_centred=True still gets a
    # defined ½(before+now) average rather than an AttributeError.
    #
    # SIGN CONVENTION (#1455 fix): ``OceanSurfaceForcing.tau_x`` is stored
    # internally in the ATMOSPHERIC (negated) convention everywhere it is
    # built from a raw NEMO-convention stress array -- ``dino.py:3383``
    # (``tau_x=-forcing["tau_u_cell_2d"]``) and the sibling
    # ``nemo_recipe.py:768`` (``tau_x=-utau``) both negate, with the same
    # documented reason: ``surface_stress_faces``/``_bc_external_surface_
    # forcing`` (ocean_pe_latlon_cgrid.py:3465-3466) applies the ocean
    # REACTION ``-tau_x`` a SECOND time, so the net stress the ocean feels
    # equals NEMO's raw ``utau``.  ``before.tau_x`` here is NEMO's raw
    # ``utau_b`` (nemo_io.py:259, no sign manipulation) -- the SAME
    # NEMO-convention quantity ``dino.py``/``nemo_recipe.py`` negate before
    # storing.  Without the matching negation, ``_leapfrog_step``'s
    # ``barotropic_forcing_centred`` average (``0.5*(state.tau_x_prev +
    # surface_forcing.tau_x)``, ocean_model_latlon_cgrid.py:2754) mixes
    # opposite-sign-convention operands: measured corr(tau_x_prev, tau_x_now)
    # = -0.98 on the DINO y5 restart before this fix, collapsing the
    # centred average to near-zero at 2/3 of wet u-faces (own-RMS ratio
    # 0.039 vs NEMO's dumped wnd_dump_zu_frc_inc.bin increment,
    # zu_frc_write_ledger.py STEP 3).
    if before.tau_x is not None:
        replacements["tau_x_prev"] = -jnp.asarray(before.tau_x)
    if before.tau_y is not None:
        replacements["tau_y_prev"] = -jnp.asarray(before.tau_y)
    return st._replace(**replacements)


__all__ = (
    "NemoBridgeOutput",
    "bridge_nemo_to_legoesm",
    "bridge_nemo_to_legoesm_topo",
    "bridge_before_state_topo",
)
