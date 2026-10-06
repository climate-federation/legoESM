"""Ocean z-star vertical coordinate.

z* = H_max * (z + H) / (eta + H)

where H is the local ocean depth (bathymetry) and eta is the
time-varying sea surface height.

Unlike the atmosphere's z-star (static terrain Jacobian), the ocean
z-star has a DYNAMIC Jacobian J = (eta + H) / H that is recomputed
at every timestep as eta evolves.

Level convention: k=0 is surface, k=nlev-1 is deepest.
Reference z values are negative (below sea level).
"""

from __future__ import annotations

import math
from typing import NamedTuple

from jax import lax
import jax.numpy as jnp
import numpy as np

from legoesm.core.precision import get_policy
from legoesm.core.source_rounding import nemo_source_round
from legoesm.timestepping.tridiagonal import thomas_solve

# Shchepetkin (2015) adaptive-implicit vertical-advection Courant
# thresholds (NEMO ``ln_zad_Aimp`` PARAMETERs).  Below ``CU_MIN`` the
# vertical advection is fully explicit; above ``CU_CUT = 2*CU_MAX - CU_MIN``
# it is fully implicit; in between a smooth ramp blends the two.  These
# are scheme constants (not physical constants), so they live here as
# documented module defaults rather than in ``constants.py``; expose as
# kwargs on the wrapper so a single edit retunes the scheme.
_AIMP_CU_MIN = 0.15
_AIMP_CU_MAX = 0.30
# Layer-thickness floor [m] for advective-tendency / Courant denominators
# (matches ``flux_form_vertical_momentum_advection``).
_H_FLOOR = 1.0e-10


def nemo_qco_live_t_thickness(
    eta: jnp.ndarray,
    H_bathy: jnp.ndarray,
    z_coord,
    dtype,
    *,
    e3t_0: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """NEMO ``e3t(Kmm)`` from the literal QCO T-point statements.

    ``domain.F90:158`` materialises ``r1_ht_0`` as
    ``ssmask / (ht_0 + 1 - ssmask)``; ``domqco.F90:159-161`` then forms
    ``r3t = ssh*r1_ht_0``.  Finally ``domzgr_substitute.h90:45-51,126``
    expands ``e3t`` as ``e3t_0*(1+r3t*tmask)``.  These stored-expression
    boundaries matter on partial-cell, non-uniform meshes and are shared by
    every NEMO WS-RK3 card.

    This is deliberately fail-closed: a NEMO-identity caller must supply or
    carry the oracle's reference ``e3t_0``, and -- on the MOVING-THICKNESS
    path only -- the active-cell mask.  The linear-free-surface arm returns
    ``e3t_0`` untouched and reads no mask, so it does not require one
    (decision 17).  It never reconstructs a midpoint ladder.
    """
    if e3t_0 is None:
        e3t_0 = getattr(z_coord, "nemo_e3t_0", None)
    active = getattr(z_coord, "is_active", None)
    if e3t_0 is None:
        raise ValueError(
            "literal NEMO QCO e3t requires explicit/reference nemo_e3t_0")
    sr = nemo_source_round
    eta = jnp.asarray(eta, dtype=dtype)
    H = jnp.asarray(H_bathy, dtype=dtype)
    e3t_0 = jnp.asarray(e3t_0, dtype=dtype)
    if getattr(z_coord, "linear_free_surface", False):
        return e3t_0
    # DECISION 17, asked and answered by the user ("Yes"): the MASK half of
    # the guard sits BELOW the linear-free-surface early return, because that
    # path returns e3t_0 untouched and never reads a mask -- refusing a caller
    # for an operand its own arm never consumes is a defect, not a check.  The
    # e3t_0 half stays ABOVE, because the early return needs it.  The
    # moving-thickness path below is unchanged and still refuses a missing
    # mask, which is the whole point of the guard.
    if active is None:
        raise ValueError(
            "literal NEMO QCO e3t requires is_active on the moving-thickness "
            "path")
    tmask = jnp.asarray(active, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    ssmask = tmask[..., 0]
    denominator = sr(sr(H + one) - ssmask)
    r1_ht_0 = sr(ssmask / denominator)
    r3t = sr(eta * r1_ht_0)
    factor = sr(one + sr(r3t[..., None] * tmask))
    return sr(e3t_0 * factor)


class NemoAdaptiveImplicitPartition(NamedTuple):
    """NEMO RK3 ``wAimp`` transport split on T-point interfaces."""

    w_explicit: jnp.ndarray
    w_implicit: jnp.ndarray
    fraction: jnp.ndarray
    courant_horizontal: jnp.ndarray
    courant_vertical: jnp.ndarray


def nemo_aimp_fraction(cu_v_int, cu_h, w_int):
    """NEMO ``wAimp_RK3_t`` implicit fraction on INTERIOR interfaces (any mesh).

    ``cu_v_int`` = dt*|w|/e3w on the nlev-1 interior interfaces, ``cu_h`` =
    the horizontal OUTFLOW Courant number of each T cell (nlev levels; NEMO
    ``Cu_adv``, sshwzv.F90:779-797), ``w_int`` = interior w (sign picks the
    upstream cell: w>0 takes the lower cell, sshwzv.F90:816-820).  Thresholds
    0.8/1.1 scaled by ``1 - cu_h/1.1`` and the Shchepetkin ramp
    (sshwzv.F90:744-748, 826-836).  Shared by the C-grid and Voronoi lanes;
    each lane builds its own ``cu_h``.
    """
    dtype = cu_v_int.dtype
    cu_h_int = jnp.where(w_int > 0.0, cu_h[..., 1:], cu_h[..., :-1])
    one = jnp.asarray(1.0, dtype=dtype)
    cu_min = jnp.asarray(0.8, dtype=dtype) * (one - cu_h_int / 1.1)
    cu_max = jnp.asarray(1.1, dtype=dtype) * (one - cu_h_int / 1.1)
    cu_cut = 2.0 * cu_max - cu_min
    in_mid = (cu_v_int > cu_min) & (cu_v_int < cu_cut)
    in_high = (cu_v_int > cu_min) & (cu_v_int >= cu_cut)
    # Double-where: each branch sees a safe operand outside its own range, so
    # the unselected branch cannot put inf/NaN into the reverse pass.
    delta = jnp.where(in_mid, cu_v_int - cu_min, one)
    # NEMO's 1/(1+F/d^2) (sshwzv.F90:833-834) as d^2/(d^2+F): same value,
    # no 1/d^2 overflow in the fp32 reverse pass when both d and F are tiny.
    d2 = delta * delta
    mid = d2 / (d2 + 4.0 * cu_max * (cu_max - cu_min))
    cu_hi = jnp.where(in_high & (cu_v_int > 0.0), cu_v_int, one)
    high = (cu_hi - cu_max) / cu_hi
    frac_int = jnp.where(in_mid, mid,
                         jnp.where(in_high, high, jnp.zeros_like(cu_v_int)))
    return jnp.clip(frac_int, 0.0, 1.0)


def nemo_aimp_midstep_geometry(h_old, h_new):
    """Step-midpoint T thickness and w-cell spacing for the Aimp Courant
    (NEMO stage-3 Kmm ~ N+1/2): ``e3w`` = mean of the adjacent thicknesses,
    half a cell at the surface and bottom interfaces (nlev+1)."""
    h_mid = 0.5 * (h_old + h_new)
    e3w = jnp.concatenate(
        [0.5 * h_mid[..., :1], 0.5 * (h_mid[..., :-1] + h_mid[..., 1:]),
         0.5 * h_mid[..., -1:]], axis=-1)
    return h_mid, e3w


def cgrid_outflow_courant(mass_flux_u, mass_flux_v, h_t, area_t, dy_u, dx_v, dt):
    """Horizontal OUTFLOW Courant number of each C-grid T cell (NEMO
    ``Cu_adv``, sshwzv.F90:779-797): positive east/north face transports minus
    negative west/south ones, times dt over the cell volume."""
    dtype = mass_flux_u.dtype
    hu_transport = mass_flux_u * jnp.asarray(dy_u, dtype=dtype)[..., None]
    hv_transport = mass_flux_v * jnp.asarray(dx_v, dtype=dtype)[..., None]
    outflow = (
        jnp.maximum(hu_transport[:, 1:, :], 0.0)
        - jnp.minimum(hu_transport[:, :-1, :], 0.0)
        + jnp.maximum(hv_transport[1:, :, :], 0.0)
        - jnp.minimum(hv_transport[:-1, :, :], 0.0)
    )
    area = jnp.asarray(area_t, dtype=dtype)
    return jnp.asarray(dt, dtype=dtype) * outflow / jnp.maximum(
        area[..., None] * h_t, jnp.asarray(_H_FLOOR, dtype=dtype))


def nemo_aimp_implicit_w_columns(cu_h, w, e3w, active, dt, return_cu_v=False):
    """Mesh-agnostic NEMO ``wAimp_RK3_t`` implicit share ``wi`` (nlev+1
    interfaces) from a per-cell horizontal Courant ``cu_h`` (each mesh builds
    its own) and the w-cell spacing ``e3w``.  Interior interfaces are zeroed
    unless both adjacent cells are ``active``."""
    dtype = w.dtype
    nlev = cu_h.shape[-1]
    w_int = w[..., 1:nlev]
    cu_v_int = jnp.asarray(dt, dtype=dtype) * jnp.abs(w_int) / jnp.maximum(
        e3w[..., 1:nlev], jnp.asarray(_H_FLOOR, dtype=dtype))
    frac_int = nemo_aimp_fraction(cu_v_int, cu_h, w_int)
    pad = ((0, 0),) * (frac_int.ndim - 1) + ((1, 1),)
    act = jnp.broadcast_to(active, cu_h.shape).astype(dtype)
    gate = jnp.pad(act[..., :-1] * act[..., 1:], pad)
    wi = (jnp.pad(frac_int, pad) * w) * gate
    if return_cu_v:
        return wi, jnp.pad(cu_v_int, pad) * gate
    return wi


def nemo_wicker_aimp_partition_transport(
    mass_flux_u: jnp.ndarray,
    mass_flux_v: jnp.ndarray,
    w: jnp.ndarray,
    h_t_kmm: jnp.ndarray,
    e3w_kmm: jnp.ndarray,
    area_t: jnp.ndarray,
    dy_u: jnp.ndarray,
    dx_v: jnp.ndarray,
    dt: float,
) -> NemoAdaptiveImplicitPartition:
    """Literal NEMO 5.0.2 RK3 transport-form adaptive partition.

    This transcribes ``sshwzv.F90:wAimp_RK3_t``'s live
    ``np_transport`` arm (lines 773--843).  ``mass_flux_u/v`` are legoESM's
    thickness transports [m2/s] on redundant west/east and south/north C-grid
    faces; multiplication by the face width reconstructs NEMO's ``zFu/zFv``
    volume transports [m3/s].  ``w`` is the T-point vertical velocity [m/s]
    with surface and bottom interfaces included.

    NEMO stores a bottom-up maximum vertical Courant number for diagnostics,
    but that value does not enter the split coefficient: every coefficient
    branch overwrites ``zcff`` from the current interface ``zCu_v``.  This
    routine therefore has no artificial cross-interface recurrence.
    """
    dtype = w.dtype
    dt_a = jnp.asarray(dt, dtype=dtype)
    cu_h = cgrid_outflow_courant(mass_flux_u, mass_flux_v, h_t_kmm, area_t,
                                 dy_u, dx_v, dt)

    nlev = h_t_kmm.shape[-1]
    if w.shape[-1] != nlev + 1 or e3w_kmm.shape[-1] != nlev + 1:
        raise ValueError("w and e3w_kmm must contain nlev+1 interfaces")
    w_int = w[..., 1:nlev]
    cu_v_int = dt_a * jnp.abs(w_int) / jnp.maximum(
        e3w_kmm[..., 1:nlev], jnp.asarray(_H_FLOOR, dtype=dtype))
    frac_int = nemo_aimp_fraction(cu_v_int, cu_h, w_int)
    pad = ((0, 0),) * (frac_int.ndim - 1) + ((1, 1),)
    fraction = jnp.pad(frac_int, pad)
    cu_v = jnp.pad(cu_v_int, pad)
    return NemoAdaptiveImplicitPartition(
        (1.0 - fraction) * w,
        fraction * w,
        fraction,
        cu_h,
        cu_v,
    )


class NemoQCOLiveFaceGeometry(NamedTuple):
    """Executed DINO-QCO face thickness and reciprocal operands."""

    e3u: jnp.ndarray
    e3v: jnp.ndarray
    r1_hu: jnp.ndarray
    r1_hv: jnp.ndarray
    r3u: jnp.ndarray
    r3v: jnp.ndarray


def nemo_qco_live_face_geometry_from_operands(
    eta,
    e3u_0,
    e3v_0,
    umask3,
    vmask3,
    hu_0,
    hv_0,
    area_t,
    area_u,
    area_v,
):
    """Build live native-face QCO geometry in NEMO source association.

    All horizontal inputs use NEMO's native A2D layout: U/V store the east/
    north face of each T cell.  The returned thicknesses and reciprocals are
    the coupled operands materialized by ``dom_qco_r3c.F90:160-181`` and the
    ``key_qco`` substitutions.  Keeping the reciprocal separate is necessary
    even though its free-surface factor cancels algebraically against the
    thickness: round 49 measured that executing both sides changes the final
    few ULPs in ``mlf_baro_corr``.
    """
    dtype = jnp.asarray(e3u_0).dtype
    # Source statements, not a scheduling hint: XLA strips
    # ``optimization_barrier`` before optimized HLO.  The IEEE identity in
    # ``nemo_source_round`` preserves the written domqco association used by
    # both the live thickness and its coupled reciprocal.
    b = nemo_source_round
    one = jnp.asarray(1.0, dtype=dtype)
    half = jnp.asarray(0.5, dtype=dtype)
    eta = jnp.asarray(eta, dtype=dtype)
    e3u_0 = jnp.asarray(e3u_0, dtype=dtype)
    e3v_0 = jnp.asarray(e3v_0, dtype=dtype)
    umask3 = jnp.asarray(umask3, dtype=dtype)
    vmask3 = jnp.asarray(vmask3, dtype=dtype)
    hu_0 = jnp.asarray(hu_0, dtype=dtype)
    hv_0 = jnp.asarray(hv_0, dtype=dtype)
    area_t = jnp.asarray(area_t, dtype=dtype)
    area_u = jnp.asarray(area_u, dtype=dtype)
    area_v = jnp.asarray(area_v, dtype=dtype)

    weighted_eta = b(area_t * eta)
    num_u = b(half * b(weighted_eta + jnp.roll(weighted_eta, -1, axis=1)))
    num_v = b(half * b(weighted_eta + jnp.roll(weighted_eta, -1, axis=0)))
    wet_u = (hu_0 > 0.0).astype(dtype)
    wet_v = (hv_0 > 0.0).astype(dtype)
    r1_hu0 = b(wet_u / (hu_0 + one - wet_u))
    r1_hv0 = b(wet_v / (hv_0 + one - wet_v))
    r1_area_u = b(one / area_u)
    r1_area_v = b(one / area_v)
    r3u = b(b(num_u * r1_hu0) * r1_area_u)
    r3v = b(b(num_v * r1_hv0) * r1_area_v)
    one_plus_r3u = b(one + r3u)
    one_plus_r3v = b(one + r3v)
    e3u = b(e3u_0 * b(one + r3u[..., None] * umask3))
    e3v = b(e3v_0 * b(one + r3v[..., None] * vmask3))
    r1_hu = b(r1_hu0 / one_plus_r3u)
    r1_hv = b(r1_hv0 / one_plus_r3v)
    return NemoQCOLiveFaceGeometry(e3u, e3v, r1_hu, r1_hv, r3u, r3v)


def nemo_t_fold_f_owned(field, grid):
    """Apply NEMO's owned-row T-pivot/F-point north-fold overwrite."""
    fold = getattr(grid, "fold", None)
    if fold is None or not bool(getattr(fold, "is_active", False)):
        return field
    from legoesm.grids.operators_latlon_cgrid import (
        fold_is_local, north_fold_mask,
    )
    # lbc_nfd_generic.h90, c_NFtype='T', cd_nat='F': the final owned
    # row takes the preceding row with ii2=Ni0glo-ji+1, i.e. (-1-i) mod N
    # after stripping the two NEMO halos.  This is an F-origin permutation;
    # it is intentionally independent of the T-origin convention detected
    # for generic scalar halo exchange.
    perm_f = jnp.arange(field.shape[1] - 1, -1, -1, dtype=jnp.int32)
    folded = field[-2, perm_f]
    if fold_is_local(grid):
        return field.at[-1].set(folded)
    nmask = north_fold_mask(grid)
    if nmask is not None:
        return field.at[-1].set(jnp.where(nmask, folded, field[-1]))
    return field


def nemo_fe3mask_from_tmask(tmask, *, grid=None):
    """Return NEMO's frozen QCO thickness mask at native F points.

    ``dommsk.F90:146-198`` forms the product of the four surrounding T masks,
    applies the F-point lateral boundary condition, and copies that result to
    ``fe3mask``.  Later slip and strait edits change ``fmask`` only.
    """
    active = jnp.asarray(tmask)
    east = jnp.roll(active, -1, axis=1)
    north = jnp.concatenate([active[1:], jnp.zeros_like(active[:1])], axis=0)
    northeast = jnp.roll(north, -1, axis=1)
    fe3mask = active * east * north * northeast
    return nemo_t_fold_f_owned(fe3mask, grid)


def nemo_dynvor_e3f_0vor(e3t_0, tmask, *, grid, dtype, nn_e3f_typ=0,
                         substitute_e3f=None, return_stages=False):
    """``dyn_vor_init``'s frozen vertex thickness ``e3f_0vor``.

    Three compiled statements, in NEMO's order
    (``dynvor.f90:914-919``, ``:935``, ``:937``):

    1. the masked four-cell reference average, ``nn_e3f_typ=0`` dividing by a
       literal four (``:918``) and ``=1`` by the wet-mask sum (``:929``);
    2. ``CALL lbc_lnk( 'dynvor', e3f_0vor, 'F', 1._wp )`` (``:935``) --
       the F-point north-fold exchange, a no-op off a folded grid;
    3. ``WHERE( e3f_0vor == 0 ) e3f_0vor = e3f_3d`` (``:937``), the zero
       substitution, which takes the MESH reference F thickness.

    ``substitute_e3f`` supplies statement 3's operand.  ``None`` keeps the
    historical unmasked four-cell ``e3t_0`` average, which is what every
    certified card was built and measured on; it is NOT ``e3f_3d`` and the
    difference is reported in round 31's receipt.

    ``return_stages`` additionally returns the array after statement 1 and
    after statement 2, so a gate can score the three compiled statements
    separately instead of re-deriving them.
    """
    b = lax.optimization_barrier
    one = jnp.asarray(1.0, dtype=dtype)
    quarter = jnp.asarray(0.25, dtype=dtype)
    e3t0 = jnp.asarray(e3t_0, dtype=dtype)
    tmask = jnp.asarray(tmask, dtype=dtype)

    def east(value):
        return jnp.roll(value, -1, axis=1)

    def north(value):
        return jnp.concatenate([value[1:], jnp.zeros_like(value[:1])], axis=0)

    masked = b(e3t0 * tmask)
    masked_n = north(masked)
    ref_sum = b(b(masked + east(masked)) + b(masked_n + east(masked_n)))
    tmask_n = north(tmask)
    wet_sum = b(b(tmask + east(tmask)) + b(tmask_n + east(tmask_n)))
    divisor = (jnp.asarray(4.0, dtype=dtype) if nn_e3f_typ == 0
               else jnp.maximum(wet_sum, one))
    e3f0vor = b(ref_sum / divisor)
    after_average = e3f0vor
    if substitute_e3f is None:
        ref_n = north(e3t0)
        fill = b(quarter * b(b(e3t0 + east(e3t0))
                             + b(ref_n + east(ref_n))))
        e3f0vor = jnp.where(e3f0vor == 0.0, fill, e3f0vor)
        # ORCA T-pivot north fold, F-point field.  Regular/closed grids
        # retain the historical path byte-for-byte.  NOTE the order: this
        # substitutes BEFORE the exchange, where NEMO exchanges first.
        after_fold = nemo_t_fold_f_owned(e3f0vor, grid)
        result = after_fold
    else:
        after_fold = nemo_t_fold_f_owned(e3f0vor, grid)
        fill = jnp.asarray(substitute_e3f, dtype=dtype)
        result = jnp.where(after_fold == 0.0, fill, after_fold)
    if return_stages:
        return result, after_average, after_fold
    return result


def nemo_ldf_reference_e3f(z_coord):
    """dyn_ldf's OWN frozen F thickness operand, ``e3f_3d``.

    ``dynldf_lev.f90:123`` stretches the MESH reference F thickness, not
    ``dyn_vor_init``'s masked four-cell ``e3f_0vor``; ``e3f_3d`` is ``e3f_0``
    by ``domzgr_substitute.h90:100`` and is read from the domain file at
    ``domzgr.F90:173``.  Fail closed rather than fall back to the vorticity
    array: silently reusing it is exactly the defect this separates.
    """
    raw = getattr(z_coord, "nemo_een_barotropic", None)
    reference = None if raw is None else getattr(raw, "e3f_0", None)
    if reference is None:
        raise ValueError(
            "NEMO's e3-weighted lateral diffusion reads the mesh reference F "
            "thickness e3f_3d (dynldf_lev.f90:123); the card must carry it "
            "as z_coord.nemo_een_barotropic.e3f_0")
    return reference


def _nemo_qco_r1_area_f(raw, geom_grid, dtype):
    b, one = lax.optimization_barrier, jnp.asarray(1.0, dtype=dtype)
    e1f = None if raw is None else getattr(raw, "e1f", None)
    e2f = None if raw is None else getattr(raw, "e2f", None)
    if e1f is None or e2f is None:
        area_f = b(jnp.asarray(geom_grid.area_q[1:, 1:], dtype=dtype))
    else:
        area_f = b(b(jnp.asarray(e1f, dtype=dtype))
                   * b(jnp.asarray(e2f, dtype=dtype)))
    return b(one / area_f)


def nemo_e3f_0vor_from_tmask(e3t_0, tmask, dry_vertex_fill, *,
                             nn_e3f_typ=0, grid=None):
    """Return NEMO's frozen vorticity thickness ``e3f_0vor`` at native F points.

    ``dyn_vor_init`` allocates and freezes this array for EVERY curl-point
    vorticity scheme -- ENS, ENE, EEN and MIX share one ``SELECT CASE`` arm
    (``dynvor.f90:890``) -- and ``dyn_cor_2D_init`` divides ``ff_f`` by
    it in every one of its branches: EEN at ``dynspg_ts.f90:960``, ENE/MIX
    at ``dynspg_ts.f90:1016``, ENS at ``dynspg_ts.f90:1046``.  It is NOT the
    F-point reference thickness ``e3f_0``.

    At ``nn_e3f_typ = 0`` (the shipped ``namelist_ref`` value for every card
    on this lane) it is the four surrounding T cells' MASKED reference
    thickness divided by FOUR -- by four, not by the number of wet cells,
    which is the ``nn_e3f_typ = 1`` branch (``dynvor.f90:897`` against
    ``dynvor.f90:905``).  NEMO's own ``((N + NE) + (C + E))`` association is
    kept: the compiled source brackets the north pair first "for
    reproducibility around NP".  The lateral boundary condition runs before
    the "insure e3f_0vor /= 0" sweep, which restores ``dry_vertex_fill`` at a
    fully dry vertex.  The unpreprocessed ``dynvor.F90:945-951`` restores
    ``e3f_0`` in BOTH ``nn_e3f_typ`` arms; ``domzgr_substitute.h90`` then
    expands that name to ``e3t_1d(jk)`` under key_vco_1d (the VORTEX build's
    ``dynvor.f90:920``) and to ``e3f_3d`` under key_vco_3d (GYRE's
    ``dynvor.f90:936``), so the caller passes its own card's array.

    Sibling implementations of the same NEMO statement, both in a DIFFERENT
    layout and neither interchangeable with this one: the live baroclinic
    ``nemo_qco_live_vorticity_e3f_cgrid`` below (padded F storage, and it
    brackets ``((C + E) + (N + NE))`` -- the opposite association, an
    unmeasured finding), and ``een_e3f_h_vtx``'s ``nemo_avg4`` arm (halo-
    padded live thicknesses).  This one is the native-A2D frozen operand the
    split-explicit coefficient builder needs.
    """
    if nn_e3f_typ not in (0, 1):
        raise ValueError("nn_e3f_typ must be 0 or 1")
    e3t0 = jnp.asarray(e3t_0)
    dtype = e3t0.dtype
    active = jnp.asarray(tmask, dtype=dtype)
    # A bare optimization_barrier is stripped from optimized HLO, so the
    # bracketing above would not survive JIT on a card whose e3t_0 varies
    # horizontally.  Hold every written binary64 result the way the literal
    # coefficient builder does.
    b = nemo_source_round

    def east(value):
        return jnp.roll(value, -1, axis=1)

    def north(value):
        # Closed north wall; an ORCA T fold is applied below instead.
        return jnp.concatenate([value[1:], jnp.zeros_like(value[:1])], axis=0)

    masked = b(e3t0 * active)
    masked_n = north(masked)
    ref_sum = b(b(masked_n + east(masked_n)) + b(masked + east(masked)))
    if nn_e3f_typ == 0:
        e3f0vor = b(ref_sum * jnp.asarray(0.25, dtype=dtype))
    else:
        active_n = north(active)
        wet = b(b(active_n + east(active_n)) + b(active + east(active)))
        e3f0vor = jnp.where(
            wet != 0.0, b(ref_sum / jnp.where(wet == 0.0, 1.0, wet)), 0.0)
    e3f0vor = nemo_t_fold_f_owned(e3f0vor, grid)
    fill = jnp.asarray(dry_vertex_fill, dtype=dtype)
    return jnp.where(e3f0vor == 0.0, fill, e3f0vor)


def nemo_qco_live_vorticity_e3f_cgrid(
    eta, z_coord, dtype, nn_e3f_typ=0, *, grid=None, e3t_0=None, tmask=None,
    reference_e3f=None,
):
    """Build literal NEMO ``e3f_vor(Kmm)`` from the card's own mesh.

    ``dyn_vor_init`` freezes ``e3f_0vor`` from masked reference T-cell
    thicknesses (``dynvor.F90:918-950``); ``dom_qco_r3c_RK3`` builds live
    ``r3f`` (``domqco.F90:233-246``); and
    ``domzgr_substitute.h90:130`` applies it through ``fe3mask``.

    ``e3t_0`` and ``tmask`` default to the coordinate's bridge fields, but
    callers may provide the same operands from their own state.  Horizontal
    areas and F-depth are always rebuilt from that mesh; bridge-carried ENE
    operands are an oracle check, not a production dependency.

    ``reference_e3f`` selects WHICH frozen F thickness the live ``r3f``
    stretch multiplies, because NEMO's two consumers do not share one.  The
    vorticity operator takes ``dyn_vor_init``'s own masked four-cell array
    ``e3f_0vor`` (``dynvor.f90:734-738`` over ``:914-937``) and is the
    default here.  The lateral-diffusion operator instead takes the MESH
    reference thickness ``e3f_3d`` -- ``e3f_0`` by the compiled macro in
    ``domzgr_substitute.h90:100``, read from the domain file at
    ``domzgr.F90:173`` -- at ``dynldf_lev.f90:123``::

        zwf(ji-1,jj-1) = ahmf(ji-1,jj-1,jk) * (e3f_3d(ji-1,jj-1,jk)          &
           &   *(1._wp+r3f(ji-1,jj-1)*fe3mask(ji-1,jj-1,jk))) * r1_e1e2f(...)

    Passing the card's carried ``e3f_0`` here therefore gives ``dyn_ldf`` its
    own consumer-local reference while ``r3f`` and ``fe3mask``, which NEMO
    genuinely shares between the two operators, stay the same arrays.  The
    array is expected on the native A2D F layout, like the carried mesh.
    """
    if nn_e3f_typ not in (0, 1):
        raise ValueError("nn_e3f_typ must be 0 or 1")
    if e3t_0 is None:
        e3t_0 = getattr(z_coord, "nemo_e3t_0", None)
    if tmask is None:
        tmask = getattr(z_coord, "is_active", None)
    if e3t_0 is None or tmask is None or grid is None:
        raise ValueError(
            "literal NEMO e3f_vor requires e3t_0, tmask, and grid operands")
    from legoesm.grids.latlon import ensure_geometry
    geom_grid = ensure_geometry(grid)
    b = lax.optimization_barrier
    one = jnp.asarray(1.0, dtype=dtype)
    quarter = jnp.asarray(0.25, dtype=dtype)
    eta = jnp.asarray(eta, dtype=dtype)
    e3t0 = jnp.asarray(e3t_0, dtype=dtype)
    tmask = jnp.asarray(tmask, dtype=dtype)

    def east(value):
        return jnp.roll(value, -1, axis=1)

    def north(value):
        # The certified GYRE use is a closed beta-plane box.
        return jnp.concatenate([value[1:], jnp.zeros_like(value[:1])], axis=0)

    raw = getattr(z_coord, "nemo_een_barotropic", None)
    mesh_e3f = None if raw is None else getattr(raw, "e3f_0", None)
    e3f0vor = nemo_dynvor_e3f_0vor(
        e3t0, tmask, grid=grid, dtype=dtype, nn_e3f_typ=nn_e3f_typ,
        substitute_e3f=mesh_e3f)

    area_eta = b(jnp.asarray(geom_grid.area_T, dtype=dtype) * eta)
    area_eta_n = north(area_eta)
    quad = b(b(area_eta + east(area_eta))
             + b(area_eta_n + east(area_eta_n)))
    fe3mask = nemo_fe3mask_from_tmask(tmask, grid=grid)
    hf0 = _nemo_qco_hf0(raw, e3f0vor, fe3mask, dtype)
    wet_f = (hf0 > 0.0).astype(dtype)
    r1_hf0 = b(wet_f / b(hf0 + one - wet_f))
    # Preserve NEMO's native F-layout operand and stored-reciprocal boundary
    # on cards that carry its raw mesh.
    r1_area_f = _nemo_qco_r1_area_f(raw, geom_grid, dtype)
    r3f = b(b(b(quarter * quad) * r1_hf0) * r1_area_f)
    # dom_qco_zgr applies the F-point lateral boundary condition to r3f
    # (domqco.F90:124-135) before domzgr_substitute.h90:130 consumes it.
    # On ORCA's T fold this is the same F-origin permutation as e3f_0vor.
    r3f = nemo_t_fold_f_owned(r3f, grid)
    # dommsk.F90:146-198 freezes fe3mask from the four-T-cell free-slip
    # mask.  The later lateral-slip/strait changes at :207-243 affect fmask
    # only.  domzgr_substitute.h90:48,130 therefore consumes fe3mask here;
    # using the vorticity fmask silently stretches partial-cell bottom faces.
    # dynldf_lev.f90:123 stretches e3f_3d, dynvor.f90:734-738 stretches
    # e3f_0vor.  Only the reference differs; r3f and fe3mask above are the
    # single shared pair NEMO builds once.
    reference = (e3f0vor if reference_e3f is None
                 else jnp.asarray(reference_e3f, dtype=dtype))
    if reference.shape != e3f0vor.shape:
        raise ValueError(
            "reference_e3f must be the native A2D F-point thickness with "
            f"shape {e3f0vor.shape}; got {reference.shape}")
    e3f_native = b(reference * b(one + r3f[..., None] * fe3mask))

    # NEMO native F(i,j) maps to legoESM vertex [j+1,i+1].  The added
    # south/west rows are inert walls for this closed-box identity.
    with_south = jnp.concatenate([e3f_native[:1], e3f_native], axis=0)
    return jnp.concatenate([with_south[:, -1:], with_south], axis=1)


def _nemo_qco_hf0(raw, e3f0vor, fe3mask, dtype):
    """Select NEMO's carried mesh F column, retaining the generic fallback."""
    # domain.f90:203-215 builds hf_0 from mesh e3f_3d and two V masks, not
    # from dyn_vor_init's distinct e3f_0vor reference thickness.
    carried = None if raw is None else getattr(raw, "hf_0", None)
    if carried is None:
        return jnp.sum(e3f0vor * fe3mask, axis=-1)
    return jnp.asarray(carried, dtype=dtype)


def nemo_qco_vorticity_f_cgrid(z_coord, dtype):
    """Map bridge-carried NEMO ``ff_f(i,j)`` to lego's vertex storage."""
    raw = getattr(z_coord, "nemo_een_barotropic", None)
    if raw is None:
        raise ValueError("literal NEMO F-point Coriolis requires raw ff_f")
    from legoesm.grids.latlon import nemo_ff_f_to_vertex
    return nemo_ff_f_to_vertex(jnp.asarray(raw.ff_f, dtype=dtype))


def nemo_qco_mesh_operands(z_coord, dtype):
    """The raw NEMO ``hu_0/hv_0`` and ``e1e2t/e1e2u/e1e2v`` QCO operands.

    ``dom_qco_r3c`` needs exactly these five fields; the cards that carry
    NEMO's own ``mesh_mask`` expose them on the z-coordinate.  Extracted so
    the lookup (and its fail-closed error) is written once.
    """
    refs = tuple(getattr(z_coord, name, None) for name in (
        "nemo_hu_0", "nemo_hv_0", "nemo_e1e2t", "nemo_e1e2u",
        "nemo_e1e2v",
    ))
    if any(value is None for value in refs):
        missing = [name for name, value in zip((
            "nemo_hu_0", "nemo_hv_0", "nemo_e1e2t", "nemo_e1e2u",
            "nemo_e1e2v"), refs, strict=True) if value is None]
        raise ValueError(
            "NEMO qco face thicknesses need the raw NEMO mesh operands on "
            f"the z-coordinate; missing: {', '.join(missing)}")
    return tuple(jnp.asarray(value, dtype=dtype) for value in refs)


class NemoQCOMeshOperands(NamedTuple):
    """The qco mesh operands ``dom_qco_r3c``/``div_hor``/``wzv`` consume.

    Native NEMO A2D horizontal extent throughout (U/V carry the EAST/NORTH
    face of each T cell).
    """

    e3t_0: jnp.ndarray
    e3u_0: jnp.ndarray
    e3v_0: jnp.ndarray
    umask3: jnp.ndarray
    vmask3: jnp.ndarray
    hu_0: jnp.ndarray
    hv_0: jnp.ndarray
    area_t: jnp.ndarray
    area_u: jnp.ndarray
    area_v: jnp.ndarray
    e2u: jnp.ndarray
    e1v: jnp.ndarray


def nemo_qco_card_mesh_operands(h_ref, u_mask_3d, v_mask_3d, grid, dtype):
    """The same qco operands, built from the card's OWN grid + ladder.

    NEMO computes these in ``domain.F90``/``domzgr.F90`` from the mesh it
    just built; the cards that read NEMO's own ``mesh_mask.nc`` carry them on
    ``z_coord.nemo_*`` (see :func:`nemo_qco_mesh_operands`), and every other
    card -- LOCK_EXCHANGE, OVERFLOW, ORCA1 -- reconstructs the identical
    quantities here rather than being locked out of the NEMO arm:

    * ``e3u_0``/``e3v_0`` are the shallower neighbour's REFERENCE thickness
      (``tests/OVERFLOW/MY_SRC/usrdef_zgr.F90:179-186``; on the certified
      OVERFLOW-zps ``mesh_mask.nc`` this holds exactly on all 16900 wet U
      faces, and on a full-step ``ln_zco`` mesh it is trivially ``e3t_0``);
    * ``hu_0 = SUM(e3u_0*umask)`` (``domain.F90:145``), so a closed face adds
      zero;
    * ``e1e2t``/``e1e2u``/``e1e2v`` and ``e2u``/``e1v`` are the C-grid
      horizontal metrics, which on a lat-lon mesh are exactly the grid's own
      cell/face lengths (``domhgr.F90`` builds them the same way).

    Inputs use legoESM's redundant west/south face layout; the returned
    operands use NEMO's native east/north extent, matching
    :func:`nemo_qco_mesh_operands`.
    """
    from legoesm.grids.latlon import ensure_geometry
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        min_cell_to_uface,
        min_cell_to_vface,
    )

    geom_grid = ensure_geometry(grid)
    e3t_0 = jnp.asarray(h_ref, dtype=dtype)
    e3u_0 = min_cell_to_uface(e3t_0)[:, 1:, :]
    e3v_0 = min_cell_to_vface(e3t_0, grid)[1:, :, :]
    umask3 = jnp.asarray(u_mask_3d, dtype=dtype)[:, 1:, :]
    vmask3 = jnp.asarray(v_mask_3d, dtype=dtype)[1:, :, :]
    hu_0 = jnp.sum(e3u_0 * umask3, axis=-1)
    hv_0 = jnp.sum(e3v_0 * vmask3, axis=-1)
    area_u = jnp.asarray(geom_grid.dx_u * geom_grid.dy_u, dtype=dtype)[:, 1:]
    area_v = jnp.asarray(geom_grid.dx_v * geom_grid.dy_v, dtype=dtype)[1:, :]
    return NemoQCOMeshOperands(
        e3t_0=e3t_0,
        e3u_0=e3u_0,
        e3v_0=e3v_0,
        umask3=umask3,
        vmask3=vmask3,
        hu_0=hu_0,
        hv_0=hv_0,
        area_t=jnp.asarray(geom_grid.area_T, dtype=dtype),
        area_u=jnp.where(hu_0 > 0.0, area_u, 1.0),
        area_v=jnp.where(hv_0 > 0.0, area_v, 1.0),
        e2u=jnp.asarray(geom_grid.dy_u, dtype=dtype)[:, 1:],
        e1v=jnp.asarray(geom_grid.dx_v, dtype=dtype)[1:, :],
    )


def nemo_qco_resolved_mesh_operands(
    z_coord, grid, u_mask_3d, v_mask_3d, dtype, nlev,
):
    """One qco operand set, from NEMO's own mesh when the card carries it.

    The NEMO arm of ``wzv``/``div_hor`` needs exactly these fields.  Cards
    built from NEMO's ``mesh_mask.nc`` (DINO, GYRE, ORCA2, VORTEX) hand over
    the raw arrays unchanged -- including the reference FACE thicknesses
    ``e3u_0``/``e3v_0``, which NEMO carries separately from ``e3t_0`` and
    which this routine must not invent; cards that are not
    (LOCK_EXCHANGE, OVERFLOW, ORCA1) get the identical quantities rebuilt
    from their own grid and reference ladder by
    :func:`nemo_qco_card_mesh_operands`.  This is what makes the NEMO arm a
    CONFIG choice on every card instead of a mesh-file privilege.
    """
    raw_names = (
        "nemo_e3t_0", "nemo_hu_0", "nemo_hv_0", "nemo_e1e2t", "nemo_e1e2u",
        "nemo_e1e2v", "nemo_e2u", "nemo_e1v",
    )
    raw = tuple(getattr(z_coord, name, None) for name in raw_names)
    umask3 = jnp.asarray(u_mask_3d, dtype=dtype)[:, 1:, :]
    vmask3 = jnp.asarray(v_mask_3d, dtype=dtype)[1:, :, :]
    if all(value is not None for value in raw):
        e3t0, hu0, hv0, area_t, area_u, area_v, e2u, e1v = (
            jnp.asarray(value, dtype=dtype) for value in raw)
        e3t0 = e3t0[..., :nlev]
        # THE REFERENCE FACE THICKNESS IS THE SHALLOWER NEIGHBOUR'S, NOT THE
        # T THICKNESS.  NEMO builds it once, in the domain builder, as
        #     pe3u(ji,jj,jk) = MIN( pe3t(ji,jj,jk), pe3t(ji+1,jj,jk) )
        #     pe3v(ji,jj,jk) = MIN( pe3t(ji,jj,jk), pe3t(ji,jj+1,jk) )
        #     CALL lbc_lnk( ..., pe3u,'U', pe3v,'V', kfillmode=jpfillcopy )
        # (the executing statements on these builds are
        # tests/VORTEX_SMT_R3{,_VEC_R8}_OMIP_L1_P3/MY_SRC/usrdef_zgr.F90:225
        # and :228, with their :231 lbc_lnk, transcribed from
        # tools/DOMAINcfg/src/domzgr.F90::zgr_zps:1166-1167 and its :1177-1178
        # exchange; `E3u_0 -> e3u_3d` under key_vco_1d3d by
        # src/OCE/DOM/domzgr_substitute.h90:94-95).  Aliasing
        # it to e3t_0 is exact ONLY on a full-step mesh; over partial cells it
        # is wrong on every stepped face: 686 wet U faces of VORTEX_SMT in
        # the nlev=10 operand this routine actually returns (1 164 cells of
        # the jpk=11 mesh array it is sliced from), by up to 170.38 m, and
        # 18 803 U / 18 300 V cells of ORCA2 by up to 917 / 949 m.
        #
        # It is NOT re-derived here.  Re-deriving needs NEMO's mask, halo and
        # north-fold conventions, and a round-213 attempt to do that zeroed a
        # whole northern row.  The card already carries the arrays NEMO
        # itself built, verified against its `mesh_mask.nc` at zero ULP, on
        # the EEN barotropic operand bundle; read them, the way
        # `nemo_ldf_reference_e3f` above reads `e3f_0` from the same bundle,
        # and fail closed rather than silently fall back to the alias.
        raw_een = getattr(z_coord, "nemo_een_barotropic", None)
        e3u0 = None if raw_een is None else getattr(raw_een, "e3u_0", None)
        e3v0 = None if raw_een is None else getattr(raw_een, "e3v_0", None)
        if e3u0 is None or e3v0 is None:
            raise ValueError(
                "a card carrying NEMO's raw qco mesh operands must also carry "
                "NEMO's own reference face thicknesses e3u_0/e3v_0 on "
                "z_coord.nemo_een_barotropic: they are the shallower "
                "neighbour's thickness (usrdef_zgr.F90:225,228; DOMAINcfg "
                "zgr_zps:1166-1167), not e3t_0, and re-deriving them here "
                "would need NEMO's mask/halo conventions")
        e3u0 = jnp.asarray(e3u0, dtype=dtype)[..., :nlev]
        e3v0 = jnp.asarray(e3v0, dtype=dtype)[..., :nlev]
        # The bundle is NEMO-native (U/V hold the EAST/NORTH face of each T
        # cell), the same extent as e3t0 and the sliced masks.  A card that
        # attached a redundant west/south layout would BROADCAST silently
        # here rather than fail, so say the shape out loud.
        if e3u0.shape != e3t0.shape or e3v0.shape != e3t0.shape:
            raise ValueError(
                "z_coord.nemo_een_barotropic.e3u_0/.e3v_0 must be on NEMO's "
                f"native extent {e3t0.shape} (U/V on the east/north face of "
                f"each T cell); got {e3u0.shape} and {e3v0.shape}")
        return NemoQCOMeshOperands(
            e3t_0=e3t0, e3u_0=e3u0, e3v_0=e3v0, umask3=umask3, vmask3=vmask3,
            hu_0=hu0, hv_0=hv0, area_t=area_t, area_u=area_u, area_v=area_v,
            e2u=e2u, e1v=e1v)
    if any(value is not None for value in raw):
        # A PARTIAL NEMO mesh is a bridge defect, not a card without one:
        # falling through to the card-built source here would silently mix
        # two operand provenances inside one wzv call.
        missing = [name for name, value in zip(raw_names, raw, strict=True)
                   if value is None]
        raise ValueError(
            "the z-coordinate carries some but not all of NEMO's qco mesh "
            f"operands; missing: {', '.join(missing)}. Attach the whole set "
            "or none of it")
    if not isinstance(z_coord, OceanPartialCellCoordinate):
        raise ValueError(
            "the NEMO qco wzv arm needs either the raw NEMO mesh operands "
            f"({', '.join(raw_names)}) on the z-coordinate or an "
            "OceanPartialCellCoordinate carrying h_partial to rebuild them")
    h_ref = jnp.asarray(z_coord.h_partial, dtype=dtype)
    if h_ref.ndim == 1:
        h_ref = jnp.broadcast_to(
            h_ref, (*umask3.shape[:2], h_ref.shape[-1]))
    return nemo_qco_card_mesh_operands(
        h_ref[..., :nlev], u_mask_3d, v_mask_3d, grid, dtype)


def nemo_qco_live_face_geometry_cgrid(
    eta,
    e3u_0,
    e3v_0,
    umask3,
    vmask3,
    hu_0,
    hv_0,
    area_t,
    area_u,
    area_v,
    *,
    include_reciprocals=False,
):
    """NEMO ``e3u/e3v(Kmm)`` on legoESM's redundant west/south C-grid faces.

    THE single face-thickness rule for BOTH time-stepping lanes.  NEMO's
    Modified-Leap-Frog entry ``dom_qco_r3c`` (``domqco.F90:166-169``) and its
    RK3 entry ``dom_qco_r3c_RK3`` (``domqco.F90:219-222``) carry the
    CHARACTER-IDENTICAL ``pr3u``/``pr3v`` statement --

        pr3u(ji,jj) = 0.5_wp * (  e1e2t(ji  ,jj) * pssh(ji  ,jj)
           &                    + e1e2t(ji+1,jj) * pssh(ji+1,jj)  )
           &                  * r1_hu_0(ji,jj) * r1_e1e2u(ji,jj)

    -- differing only in loop extent (``DO_2D(nn_hls, nn_hls-1, ...)`` versus
    ``DO_2D(0,0,0,0)``) and in the ``key_qcoTest_FluxForm`` alternative
    (``domqco.F90:227``) that is not compiled in either configuration here.
    ``domzgr_substitute.h90:127`` then gives
    ``e3u(i,j,k,t) = e3u_0(i,j,k)*(1 + r3u(i,j,t)*umask(i,j,k))``.  Because
    NEMO shares the routine, a second legoESM implementation would be an
    artificial branch point, so both lanes call this one:

    * WS-RK3 stage transport -- ``stprk3_stg.F90:273-274`` consumes
      ``e3u(Kmm)``/``e3v(Kmm)`` in the single ``zFu``/``zFv`` pair that feeds
      both ``dyn_adv`` and ``tra_adv`` (``zFw`` is built separately at
      ``:301`` and carries no ``e3``);
    * MLF tracer transport -- ``traadv.F90:329-330``, inside the
      ``#if ! defined key_RK3`` branch opened at ``:313``, builds
      ``zuu = e2u*e3u(ji,jj,jk,Kmm)*zptu`` from the same ``e3u(Kmm)``.

    Note what this is NOT: the ``min`` of the two STRETCHED T-cell
    thicknesses, which is first order wrong in the ssh difference across the
    face, and not the mean of the two ``r3t`` (each of those divides by its
    own column's ``ht_0``, not by ``hu_0``).

    ``eta``/``e3u_0``/``e3v_0``/the masks and the ``hu_0``..``area_v``
    operands arrive on NEMO's native A2D extent (U/V store the EAST/NORTH
    face of each T cell); the returned fields carry legoESM's redundant
    west/south face layout, so that native->redundant map is written once
    here instead of at each call site.

    Returns ``(e3u, e3v, one_plus_r3u, one_plus_r3v)`` with the thicknesses
    3-D on ``(n_lat, n_lon+1, nlev)`` / ``(n_lat+1, n_lon, nlev)`` and the
    ratios 2-D on the matching face shapes.
    """
    geom = nemo_qco_live_face_geometry_from_operands(
        eta, e3u_0, e3v_0, umask3, vmask3,
        hu_0, hv_0, area_t, area_u, area_v,
    )
    one = jnp.asarray(1.0, dtype=geom.e3u.dtype)
    result = (
        jnp.concatenate([geom.e3u[:, -1:, :], geom.e3u], axis=1),
        jnp.concatenate([jnp.zeros_like(geom.e3v[:1]), geom.e3v], axis=0),
        jnp.concatenate([one + geom.r3u[:, -1:], one + geom.r3u], axis=1),
        jnp.concatenate(
            [jnp.ones_like(geom.r3v[:1]), one + geom.r3v], axis=0),
    )
    if not include_reciprocals:
        return result
    # Same native-east/native-north -> redundant-west/redundant-south map as
    # the coupled thicknesses.  stprk3_stg.F90:265-278 consumes these stored
    # domqco reciprocals; recomputing 1/SUM(e3) is real-equivalent but not
    # source-identical on ORCA2's non-uniform, partial-cell mesh.
    return result + (
        jnp.concatenate([geom.r1_hu[:, -1:], geom.r1_hu], axis=1),
        jnp.concatenate([jnp.zeros_like(geom.r1_hv[:1]), geom.r1_hv], axis=0),
    )


def nemo_qco_live_face_thicknesses(
    eta,
    z_coord,
    e3u_0,
    e3v_0,
    umask3,
    vmask3,
):
    """Build NEMO QCO live ``e3u/e3v`` on its native east/north faces.

    This is the common ``dom_qco_r3c.F90:160-181`` operand builder used by
    both ldfslp and dynzad.  Inputs and outputs retain NEMO's native A2D
    horizontal extent; callers map once to legoESM's redundant west/south
    face layout when required.  The explicit barriers preserve the executed
    source association measured by the DINO fidelity instruments.
    """
    dtype = jnp.asarray(e3u_0).dtype
    geom = nemo_qco_live_face_geometry_from_operands(
        eta, e3u_0, e3v_0, umask3, vmask3,
        *nemo_qco_mesh_operands(z_coord, dtype),
    )
    return geom.e3u, geom.e3v


class NemoEENBarotropicOperands(NamedTuple):
    """Raw native-A2D operands for NEMO's frozen barotropic EEN builder."""
    ff_f: jnp.ndarray
    e3u_0: jnp.ndarray
    e3v_0: jnp.ndarray
    e3f_0: jnp.ndarray
    umask: jnp.ndarray
    vmask: jnp.ndarray
    fmask: jnp.ndarray
    fe3mask: jnp.ndarray
    hu_0: jnp.ndarray
    hv_0: jnp.ndarray
    hf_0: jnp.ndarray
    e1t: jnp.ndarray
    e2t: jnp.ndarray
    e1u: jnp.ndarray
    e2u: jnp.ndarray
    e1v: jnp.ndarray
    e2v: jnp.ndarray
    e1f: jnp.ndarray
    e2f: jnp.ndarray


class OceanZStarCoordinate(NamedTuple):
    """Static vertical grid definition (independent of eta).

    Levels indexed surface-to-bottom: k=0 is surface, k=nlev-1 is deepest.
    Reference layer thicknesses assume eta=0 and flat bottom H_max.

    Fields
    ------
    n_levels : int
        Number of vertical levels.
    H_max : float
        Maximum ocean depth [m] (positive).
    z_full_ref : array
        Reference z* at full (cell center) levels [m], shape (nlev,).
        Negative values (below sea level). z_full_ref[0] is shallowest.
    z_half_ref : array
        Reference z* at half (interface) levels [m], shape (nlev+1,).
        z_half_ref[0] = 0 (surface), z_half_ref[-1] = -H_max (bottom).
    dz_ref : array
        Reference layer thickness [m], shape (nlev,). Positive.
    dz_half_ref : array
        Distance between adjacent full levels [m], shape (nlev-1,).
    linear_free_surface : bool
        NEMO ``key_linssh``: thicknesses frozen at the eta=0 reference
        (J eta-independent), diagnosed w without the z-star sigma
        correction. Default False (full z*).
    t_depth_ref : array or None
        Optional EXACT positive T-point reference depths [m], shape
        (nlev,).  ``None`` (default) means ``|z_full_ref|`` (the
        cell-centre midpoint) is the T-point depth ladder — correct for
        legoESM's own z* grid.  A fidelity bridge that must reproduce an
        external model whose T-points are NOT the interface midpoints
        (e.g. NEMO's analytic MI96 ``gdept_1d`` ≠ midpoint of
        ``gdepw_1d``) supplies that model's exact T-depths here so the
        ``nemo_trapezoid`` PGF quadrature reconstructs the identical
        ``e3w`` (W-spacing) recurrence.  Read ONLY by the hydrostatic
        pressure quadrature; ``dz_half_ref`` and every other operator
        keep using the midpoint ``z_full_ref``.
    """
    n_levels: int
    H_max: float
    z_full_ref: jnp.ndarray
    z_half_ref: jnp.ndarray
    dz_ref: jnp.ndarray
    dz_half_ref: jnp.ndarray
    t_depth_ref: jnp.ndarray | None = None
    # NEMO key_linssh (LINEAR free surface): freeze the geometry at eta=0 —
    # layer thicknesses NEVER stretch (J = H_bathy/H_max, eta-independent) and
    # the diagnosed w skips the z-star sigma redistribution of deta/dt (NEMO
    # sshwzv.F90:190-193 fixed-e3t continuity; w[0]=deta/dt, w[bottom]=0).
    # eta still evolves via the barotropic solver and drives g*grad(eta).
    # STATIC Python bool — gates are `if` branches (never jnp.where); the
    # coordinate is constructor-captured, not traced.
    linear_free_surface: bool = False
    # NEMO fidelity geometry.  These retain the RAW mesh_mask fields, including
    # their last-bit horizontal variation; averaging them to a 1-D ladder and
    # differencing changes the operation order in eosbn2.F90.  ``mesh_reference``
    # is the faithful/default construction.  ``depth_difference`` is the
    # explicit legacy opt-in for consumers without a NEMO mesh.
    nemo_gdept_0: jnp.ndarray | None = None
    nemo_gdepw_0: jnp.ndarray | None = None
    nemo_e3t_0: jnp.ndarray | None = None
    nemo_e3w_0: jnp.ndarray | None = None
    # Exact interior-face reference scale factors consumed by NEMO's BBL
    # initializer.  They are intentionally distinct from masked live face
    # thickness: trabbl.F90:529-531 gathers e3u_0/e3v_0 at both adjacent
    # bottom indices, including an index below the shallower wet column.
    nemo_bbl_e3u_0: jnp.ndarray | None = None
    nemo_bbl_e3v_0: jnp.ndarray | None = None
    nemo_e3w_mesh_reference: bool = False
    nemo_hu_0: jnp.ndarray | None = None
    nemo_hv_0: jnp.ndarray | None = None
    nemo_e1e2t: jnp.ndarray | None = None
    nemo_e1e2u: jnp.ndarray | None = None
    nemo_e1e2v: jnp.ndarray | None = None
    nemo_e2u: jnp.ndarray | None = None
    nemo_e1v: jnp.ndarray | None = None
    nemo_een_barotropic: NemoEENBarotropicOperands | None = None


def create_ocean_z_star(
    n_levels: int = 50,
    H_max: float = 5500.0,
    dz_surface: float = 10.0,
    dz_deep: float = 200.0,
    *,
    nemo_e3w_source: str = "depth_difference",
) -> OceanZStarCoordinate:
    """Create a stretched ocean z-star coordinate.

    Uses hyperbolic tangent stretching: fine resolution near surface
    (~dz_surface m), coarse at depth (~dz_deep m).

    Parameters
    ----------
    n_levels : int
        Number of vertical levels.
    H_max : float
        Maximum ocean depth [m].
    dz_surface : float
        Target layer thickness near surface [m].
    dz_deep : float
        Target layer thickness at depth [m].
    nemo_e3w_source : {"depth_difference"}
        Generic coordinates have no raw NEMO mesh field and therefore select
        the explicit legacy construction.  Use
        :func:`create_z_star_from_thicknesses` for ``"mesh_reference"``.

    Returns
    -------
    OceanZStarCoordinate : The vertical coordinate.
    """
    if nemo_e3w_source != "depth_difference":
        raise ValueError(
            "create_ocean_z_star has no raw NEMO mesh operand; "
            "nemo_e3w_source must be 'depth_difference' (use "
            "create_z_star_from_thicknesses for 'mesh_reference')")
    if n_levels < 1:
        raise ValueError(
            f"n_levels must be >= 1, got {n_levels!r}",
        )
    if H_max <= 0.0:
        raise ValueError(f"H_max must be > 0, got {H_max!r}")
    if dz_surface <= 0.0:
        raise ValueError(f"dz_surface must be > 0, got {dz_surface!r}")
    if dz_deep <= 0.0:
        raise ValueError(f"dz_deep must be > 0, got {dz_deep!r}")

    # Stretched grid: dz grows smoothly from dz_surface to dz_deep.
    # Use a normalized distribution then scale to match H_max.
    k = jnp.arange(n_levels, dtype=get_policy().control)

    # Layer thickness profile: linear growth from dz_surface to dz_deep
    dz_raw = dz_surface + k * (dz_deep - dz_surface) / jnp.maximum(n_levels - 1.0, 1.0)

    # Normalize so total thickness matches H_max
    scale = H_max / jnp.sum(dz_raw)
    dz_ref = dz_raw * scale

    # Snap the last layer so ``sum(dz_ref) == H_max`` to bit-precision.
    # Without this, the cumsum round-trip below leaves a float-drift
    # residue of order ``H_max * eps_dtype`` (~2e-4 m for H_max=4000m
    # in fp32) that breaks the Hallberg-Adcroft 2009 column-sum
    # identity in partial-cell models — ``sum_k(h_partial)`` would
    # differ from the user-supplied ``H_bathy`` by that residue.
    dz_ref = dz_ref.at[-1].set(H_max - jnp.sum(dz_ref[:-1]))

    # Interface depths from cumulative sum (surface=0, bottom=-H_max)
    z_half_ref = jnp.concatenate([
        jnp.array([0.0], dtype=dz_ref.dtype),
        -jnp.cumsum(dz_ref),
    ])
    # Snap the bottom interface to exactly -H_max (kills cumsum drift).
    z_half_ref = z_half_ref.at[-1].set(-H_max)

    # Full level depths (cell centers)
    z_full_ref = 0.5 * (z_half_ref[:-1] + z_half_ref[1:])

    # Layer thicknesses (positive). Recover from the snapped interfaces.
    dz_ref = z_half_ref[:-1] - z_half_ref[1:]

    # Distance between full levels
    dz_half_ref = z_full_ref[:-1] - z_full_ref[1:]  # positive

    return OceanZStarCoordinate(
        n_levels=n_levels,
        H_max=H_max,
        z_full_ref=z_full_ref,
        z_half_ref=z_half_ref,
        dz_ref=dz_ref,
        dz_half_ref=dz_half_ref,
        nemo_e3w_mesh_reference=False,
    )


def create_z_star_from_thicknesses(
    dz_ref_m, t_depth_ref_m=None, *, nemo_gdept_0_m=None,
    nemo_gdepw_0_m=None, nemo_e3t_0_m=None, nemo_e3w_0_m=None,
    nemo_e3w_source="mesh_reference",
    nemo_hu_0_m=None, nemo_hv_0_m=None, nemo_e1e2t_m=None,
    nemo_e1e2u_m=None, nemo_e1e2v_m=None,
    nemo_e2u_m=None, nemo_e1v_m=None,
    nemo_een_barotropic_m=None,
) -> OceanZStarCoordinate:
    """Build a z* coordinate from EXPLICIT reference layer thicknesses.

    Reproduces an external model's vertical grid EXACTLY -- pass another model's
    1-D reference thicknesses (e.g. NEMO ``e3t_1d`` [m], surface ~1 m growing to
    ~200 m for ORCA L75) and get back the identical level interfaces / centres,
    with no stretching-parameter guessing.  Used by the OMIP runner's
    ``--nemo-vertical`` to match NEMO ORCA1's 75-level grid so vertical gradients
    (thermocline, mixed layer) are resolved comparably.

    ``n_levels`` and ``H_max`` are inferred from the input (``len(dz)`` and
    ``sum(dz)``).  Construction mirrors :func:`create_ocean_z_star` after its
    thickness profile is fixed -- the same snap-to-``H_max`` and interface
    recovery so the Hallberg-Adcroft column-sum identity holds to bit precision.

    Parameters
    ----------
    dz_ref_m : 1-D array-like
        Reference layer thicknesses [m], top -> bottom, all > 0.
    t_depth_ref_m : 1-D array-like or None
        Optional EXACT positive T-point depths [m], shape (nlev,), stored
        on the coordinate's ``t_depth_ref`` field for the fidelity PGF
        quadrature (see :class:`OceanZStarCoordinate`).  ``None`` (default)
        leaves ``t_depth_ref=None`` → the midpoint ``z_full_ref`` is used.
        Pass an external model's true T-depths (e.g. NEMO ``gdept_1d``)
        when they differ from the interface midpoint.

    Returns
    -------
    OceanZStarCoordinate
    """
    if nemo_e3w_source not in ("mesh_reference", "depth_difference"):
        raise ValueError(
            f"unknown nemo_e3w_source {nemo_e3w_source!r}; expected "
            "'mesh_reference' or 'depth_difference'")

    # Check ndim on the ORIGINAL array BEFORE any ravel -- a 2-D array would
    # otherwise be silently flattened and accepted as 1-D (codex HIGH).
    dz_np = np.asarray(dz_ref_m, dtype=np.float64)
    if dz_np.ndim != 1 or dz_np.size < 2:
        raise ValueError(
            f"dz_ref_m must be a 1-D array of >= 2 thicknesses, got shape "
            f"{dz_np.shape}")
    if not np.all(dz_np > 0.0):
        raise ValueError("dz_ref_m thicknesses must all be > 0")
    n_levels = int(dz_np.size)
    H_max = float(dz_np.sum())

    dz_ref = jnp.asarray(dz_np, dtype=get_policy().control)
    # Interfaces from cumulative sum (surface=0, bottom=-H_max); snap the bottom
    # to exactly -H_max to kill cumsum drift, then recover dz from the snapped
    # interfaces (identical pattern to create_ocean_z_star).
    z_half_ref = jnp.concatenate([
        jnp.array([0.0], dtype=dz_ref.dtype),
        -jnp.cumsum(dz_ref),
    ])
    z_half_ref = z_half_ref.at[-1].set(-H_max)
    z_full_ref = 0.5 * (z_half_ref[:-1] + z_half_ref[1:])
    dz_ref = z_half_ref[:-1] - z_half_ref[1:]
    dz_half_ref = z_full_ref[:-1] - z_full_ref[1:]

    t_depth_ref = None
    if t_depth_ref_m is not None:
        t_np = np.asarray(t_depth_ref_m, dtype=np.float64)
        if t_np.ndim != 1 or t_np.size != n_levels:
            raise ValueError(
                f"t_depth_ref_m must be a 1-D array of length n_levels="
                f"{n_levels}, got shape {t_np.shape}")
        if not np.all(t_np > 0.0):
            raise ValueError("t_depth_ref_m depths must all be > 0")
        # Monotone-increasing: the PGF e3w recurrence uses gdept(k)-gdept(k-1) as
        # a positive W-spacing; a non-monotone ladder would give a negative e3w
        # and a silently nonphysical pressure gradient.
        if not np.all(np.diff(t_np) > 0.0):
            raise ValueError("t_depth_ref_m depths must be strictly increasing")
        t_depth_ref = jnp.asarray(t_np, dtype=get_policy().control)

    def _raw_mesh_field(value, name, *, allow_surface_zero=False):
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim < 1 or arr.shape[-1] != n_levels:
            raise ValueError(
                f"{name} must have trailing dimension n_levels={n_levels}, "
                f"got shape {arr.shape}")
        valid = np.all(arr >= 0.0) if allow_surface_zero else np.all(arr > 0.0)
        if not np.all(np.isfinite(arr)) or not valid:
            relation = ">= 0" if allow_surface_zero else "> 0"
            raise ValueError(
                f"{name} must contain only finite values {relation}")
        return jnp.asarray(arr, dtype=get_policy().control)

    nemo_gdept_0 = _raw_mesh_field(nemo_gdept_0_m, "nemo_gdept_0_m")
    nemo_gdepw_0 = _raw_mesh_field(
        nemo_gdepw_0_m, "nemo_gdepw_0_m", allow_surface_zero=True)
    nemo_e3t_0 = _raw_mesh_field(nemo_e3t_0_m, "nemo_e3t_0_m")
    nemo_e3w_0 = _raw_mesh_field(nemo_e3w_0_m, "nemo_e3w_0_m")
    def _raw_horizontal(value, name):
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim != 2 or not np.all(np.isfinite(arr)) or not np.all(arr >= 0.0):
            raise ValueError(f"{name} must be a finite nonnegative 2-D field")
        return jnp.asarray(arr, dtype=get_policy().control)

    nemo_hu_0 = _raw_horizontal(nemo_hu_0_m, "nemo_hu_0_m")
    nemo_hv_0 = _raw_horizontal(nemo_hv_0_m, "nemo_hv_0_m")
    nemo_e1e2t = _raw_horizontal(nemo_e1e2t_m, "nemo_e1e2t_m")
    nemo_e1e2u = _raw_horizontal(nemo_e1e2u_m, "nemo_e1e2u_m")
    nemo_e1e2v = _raw_horizontal(nemo_e1e2v_m, "nemo_e1e2v_m")
    nemo_e2u = _raw_horizontal(nemo_e2u_m, "nemo_e2u_m")
    nemo_e1v = _raw_horizontal(nemo_e1v_m, "nemo_e1v_m")
    nemo_een_barotropic = None
    if nemo_een_barotropic_m is not None:
        if not isinstance(nemo_een_barotropic_m, NemoEENBarotropicOperands):
            raise TypeError(
                "nemo_een_barotropic_m must be NemoEENBarotropicOperands")
        raw = nemo_een_barotropic_m
        two_d = (raw.ff_f, raw.hu_0, raw.hv_0, raw.hf_0,
                 raw.e1t, raw.e2t, raw.e1u, raw.e2u,
                 raw.e1v, raw.e2v, raw.e1f, raw.e2f)
        shape2 = np.asarray(raw.ff_f).shape
        if len(shape2) != 2 or any(np.asarray(x).shape != shape2 for x in two_d):
            raise ValueError(
                "NEMO EEN 2-D operands must have one common native A2D shape")
        three_d = (raw.e3u_0, raw.e3v_0, raw.e3f_0,
                   raw.umask, raw.vmask, raw.fmask, raw.fe3mask)
        if any(np.asarray(x).shape != shape2 + (n_levels,) for x in three_d):
            raise ValueError(
                "NEMO EEN 3-D operands must have native A2D+n_levels shape")
        if any(not np.all(np.isfinite(np.asarray(x)))
               for x in (*two_d, *three_d)):
            raise ValueError("NEMO EEN operands must be finite")
        nemo_een_barotropic = NemoEENBarotropicOperands(*(
            jnp.asarray(x, dtype=get_policy().control) for x in raw))
    if nemo_gdept_0 is not None and nemo_e3w_0 is not None:
        # Validate the caller's fp64 oracle operands before policy casting;
        # a float32 runtime policy must not manufacture a source-recurrence
        # failure during construction.
        gdept_np = np.asarray(nemo_gdept_0_m, dtype=np.float64)
        e3w_np = np.asarray(nemo_e3w_0_m, dtype=np.float64)
        if gdept_np.shape != e3w_np.shape:
            raise ValueError(
                "nemo_gdept_0_m and nemo_e3w_0_m must have identical shapes, "
                f"got {gdept_np.shape} and {e3w_np.shape}")
        gdept_spacing = np.diff(gdept_np, axis=-1)
        e3w_interior = e3w_np[..., 1:]
        # NEMO's shipped GYRE MI96 literals contain one four-ULP recurrence
        # miss (level 23) because gdept and e3w are independently evaluated
        # source arrays.  Preserve both oracle operands; reject discrepancies
        # larger than that demonstrated fp64 arithmetic envelope.
        recurrence_tol = 4.0 * np.spacing(
            np.maximum(np.abs(gdept_spacing), np.abs(e3w_interior))
        )
        if np.any(np.abs(gdept_spacing - e3w_interior) > recurrence_tol):
            raise ValueError(
                "nemo_gdept_0_m differences must equal interior "
                "nemo_e3w_0_m within the four-ULP NEMO mesh-reference "
                "arithmetic envelope")
    return OceanZStarCoordinate(
        n_levels=n_levels,
        H_max=H_max,
        z_full_ref=z_full_ref,
        z_half_ref=z_half_ref,
        dz_ref=dz_ref,
        dz_half_ref=dz_half_ref,
        t_depth_ref=t_depth_ref,
        nemo_gdept_0=nemo_gdept_0,
        nemo_gdepw_0=nemo_gdepw_0,
        nemo_e3t_0=nemo_e3t_0,
        nemo_e3w_0=nemo_e3w_0,
        nemo_e3w_mesh_reference=(nemo_e3w_source == "mesh_reference"),
        nemo_hu_0=nemo_hu_0, nemo_hv_0=nemo_hv_0,
        nemo_e1e2t=nemo_e1e2t, nemo_e1e2u=nemo_e1e2u,
        nemo_e1e2v=nemo_e1e2v,
        nemo_e2u=nemo_e2u, nemo_e1v=nemo_e1v,
        nemo_een_barotropic=nemo_een_barotropic,
    )


def _levy_stretching_coefficients(
    K_formula: int,
    H: float,
    dz_min: float,
    k_th: float,
    a_cr: float,
) -> tuple[float, float, float]:
    """Compute (a₀, a₁, a₂) for the Lévy (2010) tanh+ln(cosh) vertical
    stretching used by NEMO's mi96_1d routine (Madec-Imbard 1996).

    The stretching function is::

        z(k) = a₂ + a₁·k + a₀·a_cr·ln(cosh((k - k_th)/a_cr))

    Coefficients are determined by three constraints:

      z(k=1)         = 0     (surface interface)
      z(k=K_formula) = H     (bottom interface)
      dz/dk at k=1   = dz_min  (derivative-based top-layer scale)

    Parameters
    ----------
    K_formula : int
        Index of the bottom interface in the formula's k-coordinate.
        In NEMO terminology this is ``jpk`` (interface count). For
        ``n_levels`` cells the value is ``n_levels + 1``.
    H : float
        Total ocean depth [m] (positive).
    dz_min : float
        Target top-layer derivative ``dz/dk`` at k=1 [m].
    k_th : float
        Inflection-level index. Layers ``k > k_th`` are thicker than
        ``k < k_th``. Typically ``k_th = n_levels - 1``.
    a_cr : float
        Stretching width parameter. Smaller = sharper transition near
        ``k_th``. Typically 5-15.

    Returns
    -------
    (a0, a1, a2) : tuple of float
    """
    Km1 = K_formula - 1
    th = math.tanh((1 - k_th) / a_cr)
    log_cosh_K = math.log(math.cosh((K_formula - k_th) / a_cr))
    log_cosh_1 = math.log(math.cosh((1 - k_th) / a_cr))
    denom = th - (a_cr / Km1) * (log_cosh_K - log_cosh_1)
    a0 = (dz_min - H / Km1) / denom
    a1 = dz_min - a0 * th
    a2 = -a1 - a0 * a_cr * log_cosh_1
    return a0, a1, a2


def _levy_depth_at_k(k, a0, a1, a2, k_th, a_cr) -> float:
    """Evaluate the Lévy stretching formula at arbitrary k (positive)."""
    return a2 + a1 * k + a0 * a_cr * math.log(math.cosh((k - k_th) / a_cr))


def create_levy_stretched_z_star(
    n_levels: int,
    H_max: float,
    dz_min: float,
    k_th: float,
    a_cr: float,
    analytic_t_depths: bool = False,
) -> OceanZStarCoordinate:
    """Construct a Lévy (2010) / Madec-Imbard (1996) stretched z* grid.

    Used by NEMO's ``mi96_1d`` routine and many idealized NEMO configs
    (Neverworld 2, DINO, Munday-Marshall-Johnson). Top-layer derivative
    is ``dz_min``; layer thickness grows smoothly to ~H/K_formula·tanh
    in the deep abyss.

    Returns
    -------
    OceanZStarCoordinate
        With ``n_levels`` cells, surface interface at 0, bottom
        interface snapped to exactly ``-H_max``.

    Notes
    -----
    NEMO note on indexing: the formula's "K" in the literature is
    the interface count (= n_levels + 1), NOT the cell count.
    See Madec & Imbard 1996 / Lévy et al. 2010.

    ``analytic_t_depths=False`` (default, bit-identical legacy): cell
    centres are interface midpoints.  ``True``: cell centres are the
    ANALYTIC stretching formula at k+0.5 — exactly NEMO's ``mi96_1d``
    ``pdept_1d`` (zgr_lib.F90: ``zt = jk + 0.5``), which on a stretched
    grid is NOT the interface midpoint (up to ~4.6 m difference on the
    DINO 36-interface ladder).  NEMO jpk convention: NEMO's ``jpk``
    counts INTERFACE indices (its level jpk is a permanently-masked
    dummy), so a NEMO config with jpk=36 maps to ``n_levels=35`` here.
    """
    if n_levels < 2:
        raise ValueError(f"n_levels must be >= 2, got {n_levels!r}")
    if H_max <= 0.0:
        raise ValueError(f"H_max must be > 0, got {H_max!r}")
    if dz_min <= 0.0:
        raise ValueError(f"dz_min must be > 0, got {dz_min!r}")

    K_formula = n_levels + 1  # interface count (NEMO jpk convention)

    a0, a1, a2 = _levy_stretching_coefficients(
        K_formula=K_formula,
        H=H_max,
        dz_min=dz_min,
        k_th=float(k_th),
        a_cr=a_cr,
    )

    # Interfaces at integer k = 1, 2, ..., K_formula → n_levels+1 interfaces
    z_half_pos = [
        _levy_depth_at_k(float(k), a0, a1, a2, float(k_th), a_cr)
        for k in range(1, K_formula + 1)
    ]

    # legoESM convention: z negative below surface
    z_half_list = [-z for z in z_half_pos]
    z_half_list[0] = 0.0     # snap surface
    z_half_list[-1] = -H_max # snap bottom (kills sub-meter formula residue)
    z_half_ref = jnp.asarray(z_half_list)

    dz_ref = z_half_ref[:-1] - z_half_ref[1:]
    if analytic_t_depths:
        # NEMO mi96_1d pdept_1d: the SAME stretching formula at k+0.5
        # (one centre per cell, k = 1..n_levels; NEMO's dummy jpk-th
        # centre below the last interface is not represented).
        t_pos = [
            _levy_depth_at_k(k + 0.5, a0, a1, a2, float(k_th), a_cr)
            for k in range(1, n_levels + 1)
        ]
        z_full_ref = jnp.asarray([-t for t in t_pos])
    else:
        z_full_ref = 0.5 * (z_half_ref[:-1] + z_half_ref[1:])
    dz_half_ref = z_full_ref[:-1] - z_full_ref[1:]

    return OceanZStarCoordinate(
        n_levels=n_levels,
        H_max=H_max,
        z_full_ref=z_full_ref,
        z_half_ref=z_half_ref,
        dz_ref=dz_ref,
        dz_half_ref=dz_half_ref,
    )


class OceanPartialCellCoordinate(NamedTuple):
    """z* + partial bottom cells (Adcroft, Hill, Marshall 1997;
    Adcroft & Campin 2004).

    Same z*-style reference levels as ``OceanZStarCoordinate``, but
    augmented with per-cell layer thicknesses ``h_partial(..., k)``
    that account for the seafloor cutting through the deepest active
    level.

    For each column with bathymetry depth ``H_bathy(i,j)``:
      - Levels k < bottom_level(i,j): full cells, h_partial[k] = dz_ref[k]
      - Level k = bottom_level(i,j): partial cell,
                                       h_partial[k] = H_bathy - |z_half_ref[k]|
      - Levels k > bottom_level(i,j): below seafloor, h_partial[k] = 0

    The reference fields (``z_full_ref``, ``z_half_ref``, ``dz_ref``,
    ``dz_half_ref``) match ``OceanZStarCoordinate`` exactly so callers
    that need only the reference grid can treat both coords
    uniformly.

    Fields specific to partial cells
    --------------------------------
    h_partial : array, shape (..., nlev)
        Per-cell at-rest layer thickness [m], with H_bathy folded in.
        Sum over k of h_partial[..., k] equals H_bathy(i, j) per column.
    bottom_level : array of int32, shape (...)
        Index of the deepest active level for each column.  ``-1`` for
        dry columns (H_bathy <= 0).
    is_active : array of bool, shape (..., nlev)
        True for cells at or above bottom_level.  Cells below the
        seafloor are False.
    """
    n_levels: int
    H_max: float
    z_full_ref: jnp.ndarray
    z_half_ref: jnp.ndarray
    dz_ref: jnp.ndarray
    dz_half_ref: jnp.ndarray
    h_partial: jnp.ndarray
    bottom_level: jnp.ndarray
    is_active: jnp.ndarray
    # Exact reference T-level depths (NEMO ``gdept_1d``; z*-only fidelity
    # field) propagated from the wrapped z* coordinate so non-midpoint
    # reference ladders keep their true centre geometry under partial
    # cells (MLE nla10 + gate-N2 consumers; codex MLE-rhop r2), and so the
    # fidelity PGF quadrature keeps the exact ladder (mirrors
    # ``OceanZStarCoordinate.t_depth_ref``).  ``None`` on the model's own
    # midpoint grids → HPG reverts to interface-midpoint depths. Trailing +
    # defaulted so existing constructions stay backward-compatible.
    t_depth_ref: jnp.ndarray | None = None
    nemo_gdept_0: jnp.ndarray | None = None
    nemo_gdepw_0: jnp.ndarray | None = None
    nemo_e3t_0: jnp.ndarray | None = None
    nemo_e3w_0: jnp.ndarray | None = None
    nemo_bbl_e3u_0: jnp.ndarray | None = None
    nemo_bbl_e3v_0: jnp.ndarray | None = None
    nemo_e3w_mesh_reference: bool = False
    nemo_hu_0: jnp.ndarray | None = None
    nemo_hv_0: jnp.ndarray | None = None
    nemo_e1e2t: jnp.ndarray | None = None
    nemo_e1e2u: jnp.ndarray | None = None
    nemo_e1e2v: jnp.ndarray | None = None
    nemo_e2u: jnp.ndarray | None = None
    nemo_e1v: jnp.ndarray | None = None
    nemo_een_barotropic: NemoEENBarotropicOperands | None = None
    # NEMO lateral-viscosity fields (nn_ahm_ijk_t=-30, ldfdyn.F90:288-331) on
    # the model's own stagger, attached by the driver from eddy_viscosity_3D.nc:
    # ahmt at T (n_lat, n_lon, nlev) already * tmask; ahmf at the vertex/F
    # stagger (n_lat+1, n_lon+1, nlev) already * the rn_shlat fmask
    # (dommsk.F90:207-210), so a coastal F point carries 2*ahmf for no-slip;
    # e3f_0 at the vertex stagger (domain_cfg) for the coastal zcur thickness.
    nemo_ahmt_3d: jnp.ndarray | None = None
    nemo_ahmf_3d: jnp.ndarray | None = None
    nemo_e3f_0: jnp.ndarray | None = None
    # NEMO ldf_dyn_init's READ coefficient (nn_ahm_ijk_t = -30,
    # ldfdyn.f90:348-353): the whole 3-D lateral momentum viscosity, already
    # exchanged and masked as that routine leaves it.  ``nemo_ldf_ahmt`` is on
    # the T grid, ``nemo_ldf_ahmf`` on legoESM's vertex layout.  Only the
    # lateral_viscosity_coefficient_source="nemo_ahm_3d_file" arm reads them;
    # every other card leaves them None and is bit-identical.
    nemo_ldf_ahmt: jnp.ndarray | None = None
    nemo_ldf_ahmf: jnp.ndarray | None = None


def create_partial_cell_coordinate(
    z_coord: OceanZStarCoordinate,
    H_bathy: jnp.ndarray,
    *,
    bottom_index_rule: str = "interface",
    min_partial_thickness: float | None = None,
) -> OceanPartialCellCoordinate:
    """Build an ``OceanPartialCellCoordinate`` from a z* coord + bathymetry.

    Parameters
    ----------
    z_coord : OceanZStarCoordinate
        Reference vertical grid (sets H_max, dz_ref, etc.).  Used to
        derive partial-cell thicknesses.
    H_bathy : array
        Per-column bathymetry depth [m], positive downward.  Land
        cells should have H_bathy <= 0; they're flagged as
        ``bottom_level = -1`` and ``is_active = False`` everywhere.
    bottom_index_rule : {"interface", "nemo_tpoint"}
        ``"interface"`` is the legacy Adcroft rule: the deepest active cell
        is the one whose top interface is shallower than the supplied depth.
        ``"nemo_tpoint"`` reproduces NEMO ``zgr_zps`` user domains that set
        ``k_bot`` from ``pdept_1d(k) < H <= pdept_1d(k+1)`` and then clip the
        bottom thickness at the reference bottom interface.  The latter also
        retains near-full last-bit thicknesses instead of applying legoESM's
        legacy near-full snap.  It requires an explicit ``z_coord.t_depth_ref``;
        no arithmetic-midpoint fallback is allowed because that changes
        ``k_bot`` on stretched external grids.
        ``"nemo_zps_e3min"`` is the OTHER rule NEMO's shipped zps user domains
        run, and it is NOT equivalent to ``"nemo_tpoint"``:
        ``tests/OVERFLOW/MY_SRC/usrdef_zgr.F90:204,209-212`` sets
        ``ze3min = 0.1*rn_dz`` and then

            k_bot = jpkm1
            DO jk = jpkm1,1,-1 ; WHERE( zht < pdepw_1d(jk)+ze3min ) k_bot = jk-1

        i.e. ``k_bot`` counts the W interfaces that clear the floor, so the
        thinnest bottom cell is a TENTH of the reference thickness where the
        T-point rule's is a HALF.  On a uniform ladder the two disagree for
        every column whose bathymetry lands in
        ``[pdepw+ze3min, pdepw+0.5*dz)``.  The bottom thickness is then
        ``MIN(H, pdepw_1d(k+1)) - pdepw_1d(k)``, in NEMO's own association
        (``OVERFLOW:221-225``).  It is written the source's way because the
        source writes it that way; no ladder has yet been found on which it
        differs in the bits from the other rules' clipped
        ``MIN(H - pdepw(k), dz_ref[k])``, and the unit test says so rather
        than implying a difference it does not demonstrate.  It requires
        ``min_partial_thickness``.
    min_partial_thickness : float, optional
        ``ze3min`` for ``bottom_index_rule="nemo_zps_e3min"``; refused (and
        required) for exactly that rule, so neither rule can silently run with
        the other's floor.

    Returns
    -------
    OceanPartialCellCoordinate

    Notes
    -----
    The factory is differentiable w.r.t. continuous ``H_bathy`` only
    while ``bottom_level`` does not change — i.e., piecewise smooth
    with discontinuities at every reference-level interface.  This is
    the documented limitation of partial-cell schemes (see
    ``docs/ocean/experiments/partial_cells_plan.md`` Differentiability
    Contract); not specific to this implementation.
    """
    H = jnp.asarray(H_bathy)
    nlev = z_coord.n_levels
    abs_z_half = jnp.abs(z_coord.z_half_ref)        # (nlev+1,) positive depths

    if bottom_index_rule not in {"interface", "nemo_tpoint", "nemo_zps_e3min"}:
        raise ValueError(
            "bottom_index_rule must be 'interface', 'nemo_tpoint' or "
            f"'nemo_zps_e3min', got {bottom_index_rule!r}"
        )
    if (bottom_index_rule == "nemo_zps_e3min") != (min_partial_thickness
                                                   is not None):
        raise ValueError(
            "min_partial_thickness is required by, and only by, "
            'bottom_index_rule="nemo_zps_e3min" (NEMO\'s ze3min); got rule '
            f"{bottom_index_rule!r} with {min_partial_thickness!r}"
        )

    # Number of reference points strictly shallower than H_bathy.  The legacy
    # rule counts W interfaces.  NEMO's zps user-domain rule counts T points;
    # on a uniform grid this guarantees a half-cell minimum before k_bot moves
    # down, exactly as usrdef_zgr.F90:140-143/157-160 executes.
    n_lead = H.ndim
    H_exp = H[..., jnp.newaxis]                     # (..., 1)
    if bottom_index_rule == "nemo_tpoint":
        if z_coord.t_depth_ref is None:
            raise ValueError(
                'bottom_index_rule="nemo_tpoint" requires an explicit '
                "t_depth_ref; an arithmetic-midpoint fallback can select "
                "the wrong NEMO bottom level on a stretched grid"
            )
        index_depths = jnp.abs(z_coord.t_depth_ref)
    elif bottom_index_rule == "nemo_zps_e3min":
        # OVERFLOW:209-212.  The loop runs jk = jpkm1..1 and the LAST write
        # wins, so k_bot is the count of jk in 1..jpkm1 whose
        # pdepw_1d(jk)+ze3min does NOT exceed zht -- the complement of the
        # loop's strict ``<``, hence ``<=`` here.
        index_depths = abs_z_half[:nlev] + min_partial_thickness
    else:
        index_depths = abs_z_half
    view = index_depths[(jnp.newaxis,) * n_lead + (slice(None),)]
    interfaces_above = jnp.sum(
        (view <= H_exp) if bottom_index_rule == "nemo_zps_e3min"
        else (view < H_exp),
        axis=-1,
    )                                                # (...) integer
    bottom_level = interfaces_above.astype(jnp.int32) - 1
    # Dry columns (H <= 0): mark bottom_level = -1 (no active cells).
    bottom_level = jnp.where(H > 0.0, bottom_level, -1)
    # Cap at the deepest possible level (when H exceeds H_max).
    bottom_level = jnp.minimum(bottom_level, nlev - 1)

    # Per-cell active mask.
    k_idx = jnp.arange(nlev, dtype=jnp.int32)
    k_view = k_idx.reshape((1,) * n_lead + (nlev,))
    bottom_view = bottom_level[..., jnp.newaxis]    # (..., 1)
    is_active = (k_view <= bottom_view) & (bottom_view >= 0)

    # Per-cell layer thickness.
    # Start with dz_ref broadcast to (..., nlev).
    dz_ref_view = z_coord.dz_ref.reshape((1,) * n_lead + (nlev,))
    h_full = jnp.broadcast_to(dz_ref_view, H.shape + (nlev,))

    # Partial thickness at bottom level: H - |z_half_ref[bottom_level]|,
    # capped above by the full-cell reference thickness ``dz_ref[bottom]``,
    # AND snapped to ``dz_ref[bottom]`` when within float32 precision.
    #
    # The cap matters when ``H_bathy >= H_max`` (caller passes a column
    # at or beyond reference depth — full cells, no extra thickness).
    #
    # The snap matters because ``abs_z_half`` (cumsum-built in float32)
    # has ~1e-7 relative error → for a column at exactly the reference
    # depth, ``H - abs_z_half[bottom]`` differs from ``dz_ref[bottom]``
    # by sub-millimetre.  Without the snap, ``use_partial_cells=False``
    # backwards-compat regression in Phase 6 breaks — the partial path
    # gives a slightly different thickness than the legacy path.
    # We snap when the values agree to ~1e-5 relative, well below any
    # physically meaningful column-thickness variation.
    #
    # For dry columns, bottom_level = -1 and we use 0 (the value gets
    # masked out by is_active anyway).
    safe_bottom = jnp.maximum(bottom_level, 0)
    abs_z_at_bottom = abs_z_half[safe_bottom]       # (...)
    dz_at_bottom = z_coord.dz_ref[safe_bottom]      # (...)
    if bottom_index_rule == "nemo_zps_e3min":
        # OVERFLOW:222 -- MIN(zht, pdepw_1d(ik+1)) - pdepw_1d(ik), with the
        # MIN inside the subtraction as the source writes it.  safe_bottom is
        # at most nlev-1, so safe_bottom+1 indexes abs_z_half (nlev+1 long)
        # in range without a clamp.
        partial_thickness = (jnp.minimum(H, abs_z_half[safe_bottom + 1])
                             - abs_z_at_bottom)
    elif bottom_index_rule == "nemo_tpoint":
        partial_thickness = jnp.minimum(H - abs_z_at_bottom, dz_at_bottom)
    else:
        capped = jnp.minimum(H - abs_z_at_bottom, dz_at_bottom)
        near_full = jnp.abs(capped - dz_at_bottom) < dz_at_bottom * 1e-5
        partial_thickness = jnp.where(near_full, dz_at_bottom, capped)

    is_bottom = (k_view == bottom_view) & (bottom_view >= 0)
    h_partial = jnp.where(is_bottom, partial_thickness[..., jnp.newaxis], h_full)
    h_partial = jnp.where(is_active, h_partial, 0.0)

    # Carry the exact NEMO ``gdept_1d`` T-depth ladder through the wrap so the
    # fidelity PGF quadrature keeps using it (full-step-z NEMO fidelity, #dino);
    # ``None`` on a coord without it → HPG reverts to interface-midpoint depths.
    return OceanPartialCellCoordinate(
        n_levels=nlev,
        H_max=z_coord.H_max,
        z_full_ref=z_coord.z_full_ref,
        z_half_ref=z_coord.z_half_ref,
        dz_ref=z_coord.dz_ref,
        dz_half_ref=z_coord.dz_half_ref,
        h_partial=h_partial,
        bottom_level=bottom_level,
        is_active=is_active,
        # Propagate the z*-only exact NEMO gdept so partial-cell wraps of a
        # NEMO reference ladder keep true centre depths (nla10 tolerance,
        # gate-N2 pressure geometry, and any future partial-cell PGF
        # fidelity).  ``getattr``: plain midpoint z* coords carry None.
        t_depth_ref=getattr(z_coord, "t_depth_ref", None),
        nemo_gdept_0=getattr(z_coord, "nemo_gdept_0", None),
        nemo_gdepw_0=getattr(z_coord, "nemo_gdepw_0", None),
        nemo_e3t_0=getattr(z_coord, "nemo_e3t_0", None),
        nemo_e3w_0=getattr(z_coord, "nemo_e3w_0", None),
        nemo_bbl_e3u_0=getattr(z_coord, "nemo_bbl_e3u_0", None),
        nemo_bbl_e3v_0=getattr(z_coord, "nemo_bbl_e3v_0", None),
        nemo_e3w_mesh_reference=getattr(
            z_coord, "nemo_e3w_mesh_reference", False),
        nemo_hu_0=getattr(z_coord, "nemo_hu_0", None),
        nemo_hv_0=getattr(z_coord, "nemo_hv_0", None),
        nemo_e1e2t=getattr(z_coord, "nemo_e1e2t", None),
        nemo_e1e2u=getattr(z_coord, "nemo_e1e2u", None),
        nemo_e1e2v=getattr(z_coord, "nemo_e1e2v", None),
        nemo_e2u=getattr(z_coord, "nemo_e2u", None),
        nemo_e1v=getattr(z_coord, "nemo_e1v", None),
        nemo_een_barotropic=getattr(
            z_coord, "nemo_een_barotropic", None),
    )


def create_full_step_coordinate(
    z_coord: OceanZStarCoordinate,
    bottom_level,
) -> OceanPartialCellCoordinate:
    """Build a NEMO ``ln_zco`` FULL-STEP-z coordinate from a z* reference
    grid + an explicit per-column deepest-wet-level index.

    NEMO full-step z (``usrdef_zgr.F90`` ``zgr_zco_3d`` +
    ``zgr_msk_top_bot``): every wet cell keeps the FIXED 1-D reference
    thickness ``pe3t_1d(jk)`` (``e3t(:,:,jk)=pe3t_1d(jk)`` — no partial
    thinning), and bathymetry is a STAIRCASE — column ``(i,j)`` is wet for
    levels ``k <= k_bot(i,j)-1`` and DRY below (``mbkt``/tmask). This is
    exactly an ``OceanPartialCellCoordinate`` whose bottom cell is a FULL
    cell (``h_partial = dz_ref``, never the ``ln_zps`` partial thickness).

    Unlike :func:`create_partial_cell_coordinate` (which derives the mask +
    a *partial* bottom thickness from a continuous ``H_bathy``), this takes
    the wet-level count DIRECTLY (e.g. NEMO's exact 3-D ``tmask`` column
    sum), so it is bit-faithful to NEMO's staircase with no float rounding
    at the level interfaces.

    Parameters
    ----------
    z_coord : OceanZStarCoordinate
        Reference vertical grid (fixed ``dz_ref`` = NEMO ``e3t_1d``).
    bottom_level : int array, shape (...)
        Index of the deepest WET level per column (NEMO ``k_bot - 1``).
        ``-1`` marks a dry column (no active cells), matching the
        :func:`create_partial_cell_coordinate` convention.

    Returns
    -------
    OceanPartialCellCoordinate
        With ``h_partial ∈ {0, dz_ref}`` (full cells + dry cells) and
        ``Σ_k h_partial = Σ_{k<=bottom_level} dz_ref`` per column (the
        staircase depth ``gdepw(k_bot)``).
    """
    nlev = z_coord.n_levels
    bl = jnp.asarray(bottom_level).astype(jnp.int32)
    # Cap at the deepest level (parity with create_partial_cell_coordinate);
    # a caller passing an over-deep index gets a full column, not OOB.
    bl = jnp.minimum(bl, nlev - 1)
    n_lead = bl.ndim
    k_view = jnp.arange(nlev, dtype=jnp.int32).reshape((1,) * n_lead + (nlev,))
    bottom_view = bl[..., jnp.newaxis]                       # (..., 1)
    is_active = (k_view <= bottom_view) & (bottom_view >= 0)

    dz_ref_view = z_coord.dz_ref.reshape((1,) * n_lead + (nlev,))
    h_full = jnp.broadcast_to(dz_ref_view, bl.shape + (nlev,))
    h_partial = jnp.where(is_active, h_full, 0.0)

    return OceanPartialCellCoordinate(
        n_levels=nlev,
        H_max=z_coord.H_max,
        z_full_ref=z_coord.z_full_ref,
        z_half_ref=z_coord.z_half_ref,
        dz_ref=z_coord.dz_ref,
        dz_half_ref=z_coord.dz_half_ref,
        h_partial=h_partial,
        bottom_level=bl,
        is_active=is_active,
        t_depth_ref=getattr(z_coord, "t_depth_ref", None),
        nemo_gdept_0=getattr(z_coord, "nemo_gdept_0", None),
        nemo_gdepw_0=getattr(z_coord, "nemo_gdepw_0", None),
        nemo_e3t_0=getattr(z_coord, "nemo_e3t_0", None),
        nemo_e3w_0=getattr(z_coord, "nemo_e3w_0", None),
        nemo_bbl_e3u_0=getattr(z_coord, "nemo_bbl_e3u_0", None),
        nemo_bbl_e3v_0=getattr(z_coord, "nemo_bbl_e3v_0", None),
        nemo_e3w_mesh_reference=getattr(
            z_coord, "nemo_e3w_mesh_reference", False),
        nemo_hu_0=getattr(z_coord, "nemo_hu_0", None),
        nemo_hv_0=getattr(z_coord, "nemo_hv_0", None),
        nemo_e1e2t=getattr(z_coord, "nemo_e1e2t", None),
        nemo_e1e2u=getattr(z_coord, "nemo_e1e2u", None),
        nemo_e1e2v=getattr(z_coord, "nemo_e1e2v", None),
        nemo_e2u=getattr(z_coord, "nemo_e2u", None),
        nemo_e1v=getattr(z_coord, "nemo_e1v", None),
        nemo_een_barotropic=getattr(
            z_coord, "nemo_een_barotropic", None),
    )


def extrapolate_below_seafloor(
    field: jnp.ndarray,
    z_coord: OceanPartialCellCoordinate,
) -> jnp.ndarray:
    """Fill below-seafloor (inactive) cells of a per-column field with the
    deepest ACTIVE value of that column (constant downward extrapolation).

    Active cells are a surface-down prefix (``k <= bottom_level``), so the
    inactive cells are the suffix below the seafloor.  Filling them with the
    deepest-active value gives the horizontal stencils (FC spectral or C-D
    Arakawa-Lamb) a smooth, physically-defined value at the seafloor step, so
    an active cell adjacent to a shallower column cannot import a stale/poison
    rock-cell value (a tracer or, via the EOS, density).  This is the
    per-level analogue of the cd-grid ``fill_land_cells`` horizontal rock
    fill, and is grid-neutral (touches only the trailing level axis).

    Dry columns (``bottom_level == -1``) become a constant column; they are
    removed by the 2D land mask downstream.

    Parameters
    ----------
    field : array, shape (..., nlev)
        Per-column field (e.g. T or S) to extrapolate below the seafloor.
    z_coord : OceanPartialCellCoordinate
        Provides ``bottom_level`` (deepest active level) and ``is_active``.

    Returns
    -------
    array, same shape as ``field``, with below-seafloor cells filled by the
    deepest active value of their column.
    """
    bl = jnp.maximum(z_coord.bottom_level, 0)[..., jnp.newaxis]   # (..., 1)
    deepest = jnp.take_along_axis(field, bl, axis=-1)             # (..., 1)
    return jnp.where(z_coord.is_active, field, deepest)


def compute_centroid_depth(
    eta: jnp.ndarray,
    H_bathy: jnp.ndarray,
    z_coord,
    min_water_column_m: float | None = None,
) -> jnp.ndarray:
    """Compute the geometric centroid depth (positive downward) of each
    cell, accounting for eta and partial cells.

    For each (column, level k):
      centroid_depth[..., k] = sum_{j<k} h_actual[..., j] + 0.5 * h_actual[..., k]

    For pure z\\* coord (full cells everywhere): centroid is at
    ``|z_full_ref[k]| * (eta + H_bathy) / H_max`` — uniform across columns.
    For partial-cell coord: centroid varies per column at the partial
    bottom.  Cells below the seafloor have h_actual=0 and inherit the
    seafloor depth from above (no further increment).

    Used by the Adcroft-Campin face PGF correction (Phase 3b): the
    horizontal pressure gradient between two cells with different
    centroid depths is corrected by shifting each cell's pressure to a
    common face-reference depth.

    Parameters
    ----------
    eta : array
        Sea surface height [m], shape (...).
    H_bathy : array
        Local bathymetry depth [m], shape (...).  Positive.
    z_coord : OceanZStarCoordinate or OceanPartialCellCoordinate
        Vertical coordinate.
    min_water_column_m : float or None
        Optional water-column floor (passed to ``compute_layer_thickness``).

    Returns
    -------
    array : Centroid depth [m], shape (..., nlev).  Positive downward.
    """
    h = compute_layer_thickness(
        eta, H_bathy, z_coord, min_water_column_m=min_water_column_m,
    )
    # cumsum gives interface depths at the *bottom* of each layer.
    # Centroid is half a layer above the bottom interface.
    cum = jnp.cumsum(h, axis=-1)
    return cum - 0.5 * h


def compute_layer_thickness(
    eta: jnp.ndarray,
    H_bathy: jnp.ndarray,
    z_coord,
    min_water_column_m: float | None = None,
) -> jnp.ndarray:
    """Compute actual layer thickness incorporating eta and bathymetry.

    Dispatches on the coordinate type:

    - ``OceanZStarCoordinate``: pure z\\*.
      ``h_k = dz_ref[k] * (eta + H_bathy) / H_max``.
      All layers compressed uniformly by the column Jacobian.
    - ``OceanPartialCellCoordinate``: z\\* + partial bottom cell.
      ``h_k = h_partial[..., k] * (eta + H_bathy) / H_bathy``.
      Same uniform Jacobian, but applied to the per-cell partial-cell
      thicknesses.  Cells below the seafloor stay zero (h_partial = 0).

    For the flat-bottom case (``H_bathy = H_max`` everywhere),
    both formulas yield identical layer thicknesses — the partial-cell
    coord has ``h_partial = dz_ref`` for every column, and the
    Jacobian becomes ``(eta + H_max) / H_max`` either way.  This
    backwards-compat property is guaranteed by the snap-to-dz_ref logic
    in ``create_partial_cell_coordinate``.

    Parameters
    ----------
    eta : array
        Sea surface height [m], shape (...).
    H_bathy : array
        Local bathymetry depth [m], shape (...). Positive.
    z_coord : OceanZStarCoordinate or OceanPartialCellCoordinate
        Vertical coordinate.
    min_water_column_m : float or None
        Optional lower bound for local water-column thickness
        ``eta + H_bathy`` [m]. When set, Jacobian/thickness values are
        clipped to avoid dry or negative columns.

    Returns
    -------
    array : Layer thickness [m], shape (..., nlev). Positive.
    """
    J = compute_ocean_jacobian(
        eta, H_bathy, z_coord, min_water_column_m=min_water_column_m,
    )
    if isinstance(z_coord, OceanPartialCellCoordinate):
        # ONE implementation of NEMO's stretch, not two.  This branch used to
        # form ``(eta + H_bathy)/H_bathy`` inline, so after the round-40 fix
        # to compute_ocean_jacobian the model carried TWO different roundings
        # of the SAME NEMO statement -- and this one feeds the momentum RHS
        # through _bc_geometry_and_density.  Dry columns are unchanged: the
        # helper returns J = 1 there and h_partial is already 0.
        return z_coord.h_partial * J[..., jnp.newaxis]
    # Pure z\\* path (legacy, unchanged).
    return z_coord.dz_ref * J[..., jnp.newaxis]


def compute_ocean_jacobian(
    eta: jnp.ndarray,
    H_bathy: jnp.ndarray,
    z_coord,
    min_water_column_m: float | None = None,
) -> jnp.ndarray:
    """Compute the dynamic vertical-coordinate Jacobian.

    Dispatches on coord type:

    - ``OceanZStarCoordinate``: ``J = (eta + H_bathy) / H_max``.  Used
      with ``dz_ref`` to get per-cell thickness.
    - ``OceanPartialCellCoordinate``: ``J = (eta + H_bathy) / H_bathy``.
      Used with ``h_partial`` to get per-cell thickness (the partial
      cell, full cells, and below-seafloor zero cells all scale with
      the same Jacobian).

    For backwards-compat on flat-bottom (H_bathy = H_max everywhere),
    both formulas give the same Jacobian.

    Parameters
    ----------
    eta : array
        Sea surface height [m], shape (...).
    H_bathy : array
        Local bathymetry depth [m], shape (...). Positive.
    z_coord : OceanZStarCoordinate or OceanPartialCellCoordinate
        Vertical coordinate.
    min_water_column_m : float or None
        Optional lower bound for local water-column thickness
        ``eta + H_bathy`` [m].

    Returns
    -------
    array : Jacobian, shape (...).
    """
    linssh = bool(getattr(z_coord, "linear_free_surface", False))
    if linssh:
        # NEMO key_linssh: the column NEVER stretches — J is the eta=0
        # reference (H_bathy/H_max; ==1 on a flat bottom where H_bathy==H_max).
        # No min-column clip: the fixed column is positive by construction.
        water_col = jnp.broadcast_to(
            jnp.asarray(H_bathy, dtype=jnp.asarray(eta).dtype), jnp.shape(eta))
    else:
        water_col = eta + H_bathy
        if min_water_column_m is not None:
            min_col = jnp.asarray(min_water_column_m, dtype=water_col.dtype)
            water_col = jnp.maximum(water_col, min_col)
    if isinstance(z_coord, OceanPartialCellCoordinate):
        # Fully-DRY columns (H_bathy == 0, e.g. the DINO land-wall continent
        # on the true 199x52 frame) get the inert reference J = 1, not
        # eta/1e-10: J -> 0 there makes every downstream 1/(dz*J) a 0/0 NaN
        # (N^2 -> Treguier/EKE chain, #1226 full-frame). All cells of a dry
        # column are masked, so the value is physically inert; wet columns
        # (H_bathy > 0) are bit-identical.
        H_safe = jnp.maximum(H_bathy, 1.0e-10)
        # NEMO forms the RATIO first and adds ONE.  It never forms
        # (ssh + ht_0)/ht_0:
        #   r3t(i,j)  = ssh(i,j) * r1_ht_0(i,j)              domqco.F90:209
        #   r1_ht_0   = ssmask / (ht_0 + 1 - ssmask), i.e. exactly 1/ht_0 on
        #               a wet column                          domain.F90:158
        #   e3t(i,j,k,t) = e3t_0(i,j,k) * (1 + r3t(i,j,t))
        #                                        domzgr_substitute.h90:139
        # Rounding the SUM first loses the low bits of the small ratio to
        # cancellation.  Measured on GYRE's kt=1 stage-3 ssh against NEMO's
        # own dumped arrays: (eta+H)/H differs from NEMO's 1+r3t_Kaa on 221
        # of 600 wet columns at 2.220446e-16 and its e3t on 5207 of 18000
        # cells at 1.136868e-13, while 1 + eta*(1/ht_0) reproduces both at
        # 0 cells unequal.  The clip below is legoESM's own and NEMO has
        # none; expressing it on J rather than on the column keeps every
        # clipped cell bit-identical to the pre-round-40 value.
        # ``r1_ht_0 = ssmask/(ht_0 + 1 - ssmask)`` (domain.F90:158) is built
        # ONCE and MULTIPLIED at domqco.F90:209; NEMO never divides by ht_0
        # there, and ``a/b`` and ``a*(1/b)`` are not the same double.
        #
        # RULE 1c.  The reciprocal is taken in the PROMOTED dtype of the two
        # operands, not in the bathymetry's storage dtype.  NEMO is fp64
        # throughout so the question does not arise there; here a f32 ladder
        # under a f64 ssh would round 1/ht_0 to single and cost seven digits
        # of the stretch -- caught by test_jacobian_column_sums_to_water_column
        # at a relative 3.0e-10, which is 0.7/H times f32 eps and not roundoff.
        # ``OceanPartialCellCoordinate`` carries no ``linear_free_surface``
        # field, so ``linssh`` is False on every path that reaches here; the
        # fixed-column arm belongs to the z* branch below and is not
        # duplicated as an unreachable one.
        _dt = jnp.promote_types(jnp.asarray(eta).dtype,
                                jnp.asarray(H_safe).dtype)
        r1_h = jnp.where(H_bathy > 0.0,
                         jnp.asarray(1.0, _dt) / jnp.asarray(H_safe, _dt),
                         jnp.asarray(0.0, _dt))
        jac = 1.0 + jnp.asarray(eta, _dt) * r1_h
        if min_water_column_m is not None:
            # legoESM's own floor; NEMO has none, so there is no faithful form
            # to match and this keeps every clipped cell at its pre-round-40
            # value.
            jac = jnp.maximum(jac, min_col / H_safe)
        return jnp.where(H_bathy > 0.0, jac, 1.0)
    return water_col / z_coord.H_max


def upwind_vertical_gradient(
    field: jnp.ndarray,
    dz_half: jnp.ndarray,
    w: jnp.ndarray,
    *,
    eps: float = 1.0e-12,
) -> jnp.ndarray:
    """Compute first-order upwind d(field)/dz at full levels.

    Assumes levels are indexed surface-to-bottom (k=0 at surface).
    Caller must use consistent coordinates: both ``dz_half`` and ``w``
    should be in the same vertical coordinate (physical z or z*).

    Parameters
    ----------
    field : array
        Field at full levels, shape (..., nlev).
    dz_half : array
        Full-level spacing [m], shape (..., nlev-1). Positive.
    w : array
        Vertical velocity [m/s], shape (..., nlev).
        Only its sign is used (upwind direction). Positive = upward.
    eps : float
        Small denominator guard for spacing.

    Returns
    -------
    array : Upwind vertical gradient d(field)/dz, shape (..., nlev).
    """
    inv_dz_half = 1.0 / jnp.maximum(dz_half, eps)
    df = (field[..., :-1] - field[..., 1:]) * inv_dz_half

    # Pad along trailing axis instead of allocating a fresh ``zeros``
    # buffer + concatenate.  Single Pad HLO op each.  This helper
    # fires once per scan step inside ``vertical_advection_ocean`` for
    # u, v, T, S, and every tracer — so 4-6 zero-broadcast concats
    # per RHS evaluation in the hot loop.
    pad_axes = ((0, 0),) * (df.ndim - 1)
    # Upward flow (w>0): donor is deeper cell -> (f[k] - f[k+1]) / dz.
    grad_up = jnp.pad(df, (*pad_axes, (0, 1)))
    # Downward flow (w<0): donor is shallower cell -> (f[k-1] - f[k]) / dz.
    grad_down = jnp.pad(df, (*pad_axes, (1, 0)))

    return jnp.where(w > 0.0, grad_up, grad_down)


# ---------------------------------------------------------------------------
# Vertical velocity diagnosis and advection (shared across ocean dycores)
# ---------------------------------------------------------------------------

def diagnose_w_from_flux_div(flux_div_k, z_coord=None,
                              thickness_weighted=False):
    """Diagnose z-star transport velocity from flux divergence.

    Performs a bottom-up cumulative sum of the horizontal flux divergence
    and optionally applies the z-star sigma correction so that
    ẇ = 0 at both surface and bottom.

    Parameters
    ----------
    flux_div_k : array, shape (..., nlev)
        Horizontal flux divergence at each layer.
        If ``thickness_weighted=False`` (legacy), this is ``div(u)`` and
        will be multiplied by ``dz_ref`` before integration.
        If ``thickness_weighted=True``, this is ``div(h*u)`` [m/s] and
        already has layer thickness folded in; no dz multiplication.
    z_coord : OceanZStarCoordinate or None
        When provided, applies the z-star correction.
    thickness_weighted : bool
        If True, ``flux_div_k`` already includes layer thickness
        (i.e. it was computed from thickness-weighted velocity).
        Default False for backward compatibility.

    Returns
    -------
    w : array, shape (..., nlev+1)
        Vertical velocity on half levels (surface first, bottom last = 0).
    """
    # From continuity: w(k) = w(k+1) + div_h(h_k * u_k)
    # If flux_div_k already includes layer thickness (thickness_weighted=True),
    # we cumsum directly. Otherwise, multiply by dz_ref first.
    if thickness_weighted:
        fd_integrated = flux_div_k
    elif z_coord is not None:
        dz_ref = z_coord.dz_ref  # Layer thicknesses
        fd_integrated = flux_div_k * dz_ref[jnp.newaxis, jnp.newaxis, :]
    else:
        # Fallback for testing (assume unit thickness)
        fd_integrated = flux_div_k

    fd_rev = fd_integrated[..., ::-1]
    cumsum_rev = jnp.cumsum(fd_rev, axis=-1)
    w_inner = -cumsum_rev[..., ::-1]
    # Pad along the trailing axis instead of allocating a fresh
    # ``(..., 1)`` zero buffer and concatenating.
    pad_axes_w = ((0, 0),) * (w_inner.ndim - 1)
    w_euler = jnp.pad(w_inner, (*pad_axes_w, (0, 1)))

    if z_coord is None:
        return w_euler
    if getattr(z_coord, "linear_free_surface", False):
        # NEMO key_linssh w (sshwzv.F90:190-193): fixed-e3t continuity —
        # w[..., 0] = deta/dt at the fixed z=0 surface, w[..., -1] = 0, NO
        # sigma redistribution of deta/dt through the column (that z-star
        # term is what pumps the surface tendency into the abyss).
        return w_euler

    deta_dt = w_euler[..., 0:1]
    if isinstance(z_coord, OceanPartialCellCoordinate):
        # Partial cells: sigma must use each column's actual seafloor
        # depth, not the reference H_max.  Otherwise w at the partial
        # seafloor (k = bottom_level + 1, not k = nlev) is non-zero by
        # ``sigma_zstar - sigma_partial`` * deta_dt — a spurious vertical
        # mass flux at the seafloor that breaks tracer mass conservation.
        # z_half_actual[k] = -cumsum(h_partial[0..k-1]) from surface;
        # H_bathy_per_column = sum(h_partial).  sigma_per_cell[k] =
        # (z_half_actual + H_bathy)/H_bathy is 1 at surface, 0 at the
        # column's own seafloor (where h_partial = 0 below).
        h_p = z_coord.h_partial                                  # (..., nlev)
        z_half_actual_inner = -jnp.cumsum(h_p, axis=-1)          # (..., nlev)
        pad_axes = ((0, 0),) * (z_half_actual_inner.ndim - 1)
        z_half_actual = jnp.pad(
            z_half_actual_inner, (*pad_axes, (1, 0)),
        )                                                          # (..., nlev+1)
        H_col = jnp.sum(h_p, axis=-1, keepdims=True)             # (..., 1)
        H_col_safe = jnp.maximum(H_col, 1e-10)
        sigma = (z_half_actual + H_col) / H_col_safe
    else:
        sigma = (z_coord.z_half_ref + z_coord.H_max) / z_coord.H_max
    return w_euler - sigma * deta_dt


def vertical_advection_ocean(field, w_half, z_coord, jacobian):
    """Vertical advection ``-w * d(field)/dz`` with upwind scheme.

    Parameters
    ----------
    field : array, shape (..., nlev)
        Quantity being advected.
    w_half : array, shape (..., nlev+1)
        Vertical velocity on half levels.
    z_coord : OceanZStarCoordinate
        Vertical coordinate (provides ``dz_half_ref``).
    jacobian : array, shape (...)
        Dynamic z-star Jacobian.

    Returns
    -------
    tendency : array, shape (..., nlev)
    """
    w_full = 0.5 * (w_half[..., :-1] + w_half[..., 1:])
    jac_safe = jnp.maximum(jacobian[..., jnp.newaxis], 1.0e-10)
    dz_half = z_coord.dz_half_ref * jac_safe
    grad = upwind_vertical_gradient(field, dz_half, w_full)

    # Vertical advection calculation

    return -w_full * grad


def flux_form_vertical_momentum_advection(
    u: jnp.ndarray,
    w_half: jnp.ndarray,
    h_u: jnp.ndarray,
    face_active: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Flux-form vertical momentum advection as a per-thickness tendency.

    Computes the interface-upwind vertical momentum flux
    ``F[k] = w_half[k] * u_face[k]`` with

    - ``F[0] = F[nlev] = 0``  (rigid-lid / no-flux boundary)
    - ``u_face[k] = u[k]``     when ``w_half[k] > 0``  (upward, from below)
    - ``u_face[k] = u[k-1]``   when ``w_half[k] <= 0`` (downward, from above)

    and returns ``-(F_top - F_bot) / h_u`` at each level.  This is the
    tracer-path pattern (``flux_form_vertical_tracer_advection``)
    converted back to a per-thickness advective tendency so that
    callers can add it directly to ``du/dt``.

    Properties
    ----------
    1. Interior interface-upwind (consistent with the tracer path).
    2. Rigid-lid boundary by construction: ``F[0] = F[nlev] = 0``.
       No artificial momentum injection from the boundary via the
       "hard zero at k=0 and k=nlev-1" pathology that the old
       ``vertical_advection_ocean`` cell-upwind gradient has.
    3. Column momentum flux identity: for any ``w_half`` with
       ``w_half[0] = w_half[nlev] = 0`` (closed column), the sum of
       ``(tendency * h_u)`` over the column is exactly zero.

    Partial-fix status (issue #171)
    -------------------------------
    This is a **Level-1** fix.  Dividing by ``h_u_old`` instead of
    doing a full ``(h·u)_new = (h·u)_old - dt * flux_div`` / ``u_new =
    (h·u)_new / h_u_new`` update leaves a residual
    ``O(dt · u · dh_u/dt / h_u)`` error under dynamic z-star.  A full
    flux-form momentum update requires restructuring the model step
    function (Level 2 in the #171 discussion) and is still open.

    Parameters
    ----------
    u : array, shape (..., nlev)
        Velocity at full levels at the momentum points (u-face, v-face,
        or edge — caller's choice, as long as ``w_half`` and ``h_u``
        are interpolated to the same points).
    w_half : array, shape (..., nlev+1)
        Vertical velocity on half (interface) levels, at the same
        momentum points as ``u``.  Positive = upward.  Must be zero
        at the surface and bottom interfaces.
    h_u : array, shape (..., nlev)
        Layer thickness at the momentum points.  Used only as the
        advective-form denominator.
    face_active : array | None, shape (..., nlev)
        Optional per-level face-activity mask (1 = wet face, 0 = closed
        face below the partial seafloor).  When provided, the vertical
        flux at any interface bordering an inactive face level is
        gated to exactly zero — same purpose as the ``cell_active``
        argument of the tracer helper, applied here to momentum.

    Returns
    -------
    tendency : array, shape (..., nlev)
        ``-(F_top - F_bot) / h_u`` — a per-thickness momentum tendency
        ready to add to ``du/dt``.
    """
    vert_flux_div = flux_form_vertical_tracer_advection(
        u, w_half, cell_active=face_active,
    )
    h_u_safe = jnp.maximum(h_u, 1.0e-10)
    return -vert_flux_div / h_u_safe


def flux_form_vertical_momentum_advection_centered(
    u: jnp.ndarray,
    w_half: jnp.ndarray,
    h_u: jnp.ndarray,
    face_active: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Veros-faithful 2nd-order CENTERED vertical momentum advection.

    Same per-thickness advective-tendency interface as
    :func:`flux_form_vertical_momentum_advection` (the 1st-order upwind
    version), but the interface velocity is the UNLIMITED 2-cell average

        ``F[k] = w_half[k] * 0.5*(u[k-1] + u[k])``   (1 <= k <= nlev-1)
        ``F[0] = F[nlev] = 0``                        (rigid-lid / no-flux)

    and the tendency is ``-(F_top - F_bot) / h_u`` per level.  This is the
    vertical part of Veros ``core/momentum.py`` ``momentum_advection``
    (``flux_top = 0.25*(u[k+1]+u[k])*(wtr+wtr_east)``: the ``0.25`` is
    ``0.5`` for the 2-cell ``u`` average times ``0.5`` for the
    interpolation of ``w`` to the momentum point — the latter is handled
    by the caller, which passes ``w_half`` already interpolated to the
    u/v face).  Veros leaves ``flux_top`` zero at both the surface and the
    bottom interface (``flux_top[..., :-1]`` set; bottom skipped in the
    ``du_adv[1:] += flux_top[:-1]`` update), matching the zero-pad at both
    ends here.

    Energy property
    ---------------
    The centered (skew-symmetric) flux conserves the vertical-advection
    contribution to column kinetic energy to machine precision for a
    non-divergent column ``w`` (``w_half[0]=w_half[nlev]=0``): the discrete
    ``sum_k u[k] * (F[k]-F[k+1])`` telescopes to a boundary term that
    vanishes.  The 1st-order upwind flux is strictly KE-dissipative
    (implicit vertical viscosity ``~|w|*dz/2``), which damps baroclinic
    shear; the centered scheme removes that damping.

    DISPERSION / STABILITY
    ----------------------
    The centered face value is UNLIMITED, so this scheme is dispersive (no
    monotonicity, no implicit viscosity).  Stability rests on the same
    ingredients Veros relies on: a short momentum time step (``dt_mom``)
    and explicit/implicit vertical friction (background ``A_v`` + TKE/KPP
    ``kappaM``).  Use only with those in place (the ACC recipe).

    Caller convention
    ------------------
    To reproduce Veros, pass the FULL face velocity ``u`` (barotropic +
    baroclinic).  This restores the depth-integral-zero redistribution
    term ``-d/dz(w * U_bar)`` that advecting the perturbation ``u' =
    u - U_bar`` alone omits — the term that vertically redistributes
    barotropic momentum into shear.

    Parameters
    ----------
    u : array, shape (..., nlev)
        Velocity at full levels at the momentum point.  Pass the FULL
        velocity for the Veros-faithful behaviour.
    w_half : array, shape (..., nlev+1)
        Vertical velocity on half (interface) levels at the SAME momentum
        point as ``u``.  Positive = upward; zero at surface and bottom.
    h_u : array, shape (..., nlev)
        Layer thickness at the momentum point (advective-form denominator).
    face_active : array | None, shape (..., nlev)
        Optional per-level face-activity mask (1 = wet, 0 = closed below
        the partial seafloor).  Gates the flux at any interface bordering
        an inactive face to exactly zero (same role as in the upwind /
        tracer helpers).

    Returns
    -------
    tendency : array, shape (..., nlev)
        ``-(F_top - F_bot) / h_u`` — per-thickness momentum tendency.
    """
    vert_flux_div = flux_form_vertical_tracer_advection_centered(
        u, w_half, cell_active=face_active,
    )
    h_u_safe = jnp.maximum(h_u, 1.0e-10)
    return -vert_flux_div / h_u_safe


def nemo_up3_vertical_momentum_advection(
    velocity: jnp.ndarray,
    w_half_at_face: jnp.ndarray,
    h_face: jnp.ndarray,
    face_active: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """NEMO ``dynadv_up3`` vertical momentum-flux transcription.

    This is the vertical part of the live flux-form ``ln_dynadv_up3`` arm,
    not ``dynzad`` (which ``dynadv.F90:79-91`` calls only for vector form).
    It transcribes NEMO 5.0.2 ``dynadv_up3.F90:239-365`` with
    ``gamma1=1/3``: second differences come from the before/stage-base
    velocity, their upwind choice uses the sign of the vertical transport,
    and the transported value is the centered pair minus the UP3 correction.

    ``w_half_at_face`` is the arithmetic T-to-u/v-face average.  NEMO writes
    ``0.25*(zFw_left+zFw_right)*(u_below+u_above-gamma1*lap)``; because the
    supplied average is half that sum, the coefficient below is ``0.5``.
    Surface and bottom fluxes are exactly zero.  Final tendencies are masked
    at their own wet velocity cell, matching NEMO's subsequent time update.
    """
    nlev = velocity.shape[-1]
    if nlev < 2:
        return jnp.zeros_like(velocity)
    gamma1 = 1.0 / 3.0
    if face_active is None:
        active = jnp.ones_like(velocity)
    else:
        active = jnp.broadcast_to(face_active, velocity.shape)
    interface_active = active[..., :-1] * active[..., 1:]
    first_difference = (
        (velocity[..., :-1] - velocity[..., 1:]) * interface_active
    )
    pad_axes = ((0, 0),) * (velocity.ndim - 1)
    next_difference = jnp.pad(
        first_difference[..., 1:], (*pad_axes, (0, 1))
    )
    second_difference = first_difference - next_difference
    previous_second = jnp.pad(
        second_difference[..., :-1], (*pad_axes, (1, 0))
    )
    w_interior = w_half_at_face[..., 1:nlev]
    up3_correction = jnp.where(
        w_interior > 0.0, second_difference, previous_second
    )
    interior_flux = 0.5 * w_interior * (
        velocity[..., :-1] + velocity[..., 1:]
        - gamma1 * up3_correction
    )
    flux = jnp.pad(interior_flux, (*pad_axes, (1, 1)))
    divergence = flux[..., :-1] - flux[..., 1:]
    return -divergence / jnp.maximum(h_face, _H_FLOOR) * active


#: ``bottom_face_mask_mode`` literals for
#: :func:`nemo_advective_vertical_momentum_advection` (#1226 level-29-onset
#: finding, ``zad_level29_onset_walk.py``, commit ``b6d0d9877``):
#:   "min_rule" (default, bit-identical) -- G at interface k is zeroed
#:     whenever EITHER adjacent cell (k-1 or k) is inactive
#:     (``active_above * active_below``).  NOT what NEMO does; see below.
#:   "nemo_faithful" -- transcribes ``dynzad.F90:86-119`` exactly: the
#:     ``DO_3D(0,0,0,0,1,jpk-2)`` loop has NO per-face umask/vmask guard on
#:     ``ww`` or on the assembled flux, so G is built COMPLETELY UNMASKED
#:     (a straddling u-face's still-wet deeper T-neighbour contributes its
#:     genuine nonzero ``ww`` into the shallower face's last-wet cell).
#:     NEMO defers ALL masking to the velocity update
#:     (``dynzdf.F90:121``, ``puu(Kaa) = (puu(Kbb)+rDt*Krhs) * umask(jk)``)
#:     which zeroes the FINAL TENDENCY using only the cell's OWN
#:     ``face_active`` at that level jk -- not an AND of interface
#:     neighbours.  Below any face's own seafloor both neighbouring cells
#:     (and therefore u, w) are architecturally zero, so this produces the
#:     same zero there as "min_rule"; the two modes differ ONLY at a
#:     face's last-wet cell when its deeper T-neighbour is still wet.
VALID_ZAD_BOTTOM_FACE_MASK = frozenset({"min_rule", "nemo_faithful"})


def nemo_advective_vertical_momentum_advection(
    u: jnp.ndarray,
    w_area_half: jnp.ndarray,
    h_u: jnp.ndarray,
    face_area: jnp.ndarray,
    face_active: jnp.ndarray | None = None,
    bottom_face_mask_mode: str = "min_rule",
) -> jnp.ndarray:
    """NEMO-faithful ADVECTIVE-form vertical momentum advection (dynzad.F90).

    #1226 finding: NEMO's ``dyn_zad`` (``src/OCE/DYN/dynzad.F90:86-118``)
    discretizes vertical momentum advection as the ADVECTIVE form
    ``w * du/dz`` (a CENTERED DIFFERENCE of ``u`` weighted by an
    area-weighted-interpolated ``w``), NOT the flux-divergence form
    ``d(w*u)/dz`` that :func:`flux_form_vertical_momentum_advection_centered`
    computes.  The two differ by ``u * dw/dz`` at every interior level
    (measured: predicted-vs-observed residual corr -0.9992, ratio 0.998
    against NEMO's own dumped ``dyn_zad`` trend — the whole ZAD mismatch).

    NEMO transcription (``dynzad.F90:86-118``, Fortran 1-indexed ``jk``,
    ``Kmm`` = "now" time level, ``ln_vortex_force=.FALSE.`` — no Stokes
    drift, the DINO/GYRE default)::

        DO jk = 1, jpk-2
           zWf  = e1e2t(i  ,j) * ww(i  ,j,jk+1)
           zWfi = e1e2t(i+1,j) * ww(i+1,j,jk+1)
           zzWfu = zWfi + zWf                        ! = 2 * mean(e1e2t*ww) at u-face, interface jk+1
           zzWdzU = zzWfu * (uu(i,j,jk) - uu(i,j,jk+1))
           puu(i,j,jk) -= 0.25 * r1_e1e2u(i,j) / e3u(i,j,jk) * (zWdzU(i,j) + zzWdzU)
           zWdzU(i,j) = zzWdzU                        ! carried to interface jk+1's "top" term
        jk = jpkm1
           puu(i,j,jk) -= 0.25 * r1_e1e2u(i,j) / e3u(i,j,jk) * zWdzU(i,j)   ! bottom: only the top-interface term

    with ``zWdzU`` initialized to 0 at the surface (``dynzad.F90:83-84``,
    ``jk=1``) and the bottom interface flux (``jk=jpk``) is architecturally
    zero (``pww(jpk)=0``, ``sshwzv.F90:182``) and never read.

    Transcribed here in the array (0-indexed, ``k=0`` surface) convention
    shared by the rest of this module: define, at EVERY interface
    ``k=0..nlev`` (``G[0]=G[nlev]=0`` by construction, matching NEMO's
    zeroed top/bottom)::

        G[k] = 2 * w_area_half[k] * (u[k-1] - u[k])   for k=1..nlev-1

    where ``w_area_half`` is ``w`` interpolated to the momentum-point face
    with NEMO's e1e2t-AREA-WEIGHTED interpolation (``interp_cell_to_uface``/
    ``interp_cell_to_vface`` applied to ``area_T * w``, i.e. the SAME plain
    2-cell average NEMO uses for ``e1e2t*ww`` — the area weighting is
    entirely inside the product, not a separate weighted-mean).  Then::

        tend[k] = -0.25 / (face_area[k] * h_u[k]) * (G[k] + G[k+1])

    reproducing ``r1_e1e2u/e3u`` (NEMO's u/v-point area is its OWN metric
    ``e1u*e2u``, not derived from ``e1e2t`` — ``domhgr.F90:148`` — so
    ``face_area`` must be the caller's u/v-face area, e.g.
    ``grid.dx_u*grid.dy_u``, NOT ``area_T`` interpolated).  On the regular
    lon-lat DINO/GYRE grid the u-face area equals ``area_T`` at that row
    exactly (``e1``/``e2`` independent of longitude), so the explicit
    division is a no-op there, but it is kept general (correct on tripolar/
    curvilinear grids too) rather than relying on that coincidence.

    Secondary #1226 finding: at v-faces ``area_T`` genuinely varies between
    the two neighboring cells (``dy`` depends on latitude), so the caller
    MUST pass the true e1e2t-area-weighted interpolation of ``w`` at the
    v-face (not the unweighted ``interp_to_v_points``) for internal
    consistency with NEMO — see the caller in ``ocean_pe_latlon_cgrid.py``.

    Parameters
    ----------
    u : array, shape (..., nlev)
        FULL velocity (barotropic + baroclinic) at the momentum point —
        NEMO advects ``uu``/``vv`` directly, not a perturbation.
    w_area_half : array, shape (..., nlev+1)
        ``interp_cell_to_uface`` / ``interp_cell_to_vface`` of
        ``area_T * w`` (cell-center vertical velocity times T-cell area),
        at the SAME momentum point as ``u``.  Zero at the surface and
        bottom interfaces (rigid-lid / no-flux, matching NEMO's
        ``zWdzU`` init and ``pww(jpk)=0``).
    h_u : array, shape (..., nlev)
        Layer thickness at the momentum point (NEMO ``e3u``/``e3v``).
    face_area : array, shape matching ``u``'s leading dims (broadcastable)
        The momentum-point's OWN face area (NEMO ``e1e2u``/``e1e2v``),
        e.g. ``grid.dx_u * grid.dy_u``.  NOT ``area_T`` interpolated.
    face_active : array | None, shape (..., nlev)
        Optional per-level face-activity mask (1 = wet, 0 = closed below
        the partial seafloor).
    bottom_face_mask_mode : {"min_rule", "nemo_faithful"}
        See :data:`VALID_ZAD_BOTTOM_FACE_MASK` above.  ``"min_rule"``
        (default) gates ``G`` at any interface bordering an inactive face
        to exactly zero (AND of the two neighbouring cells).
        ``"nemo_faithful"`` (#1226 level-29-onset fix) leaves ``G``
        UNMASKED (matching ``dynzad.F90`` having no umask/vmask call at
        all) and instead masks only the FINAL per-cell tendency by that
        cell's own ``face_active[..., k]`` -- reproducing
        ``dynzdf.F90:121``'s ``* umask(ji,jj,jk)`` on the velocity update,
        which is the ONLY masking NEMO ever applies to this term.

    Returns
    -------
    tendency : array, shape (..., nlev)
        ``-0.25/(face_area*h_u) * (G[k] + G[k+1])`` — a per-thickness
        momentum tendency ready to add to ``du/dt``.
    """
    if bottom_face_mask_mode not in VALID_ZAD_BOTTOM_FACE_MASK:
        raise ValueError(
            f"bottom_face_mask_mode must be one of "
            f"{sorted(VALID_ZAD_BOTTOM_FACE_MASK)}, "
            f"got {bottom_face_mask_mode!r}",
        )
    nlev = u.shape[-1]
    # Interior interface k (1 <= k <= nlev-1): G[k] = 2*w_area_half[k]*(u[k-1]-u[k]).
    w_interior = w_area_half[..., 1:nlev]           # (..., nlev-1)
    du_interior = u[..., :-1] - u[..., 1:]           # u[k-1] - u[k], k=1..nlev-1
    G_interior = 2.0 * w_interior * du_interior

    if bottom_face_mask_mode == "min_rule" and face_active is not None:
        active_above = face_active[..., :-1]    # cell k-1 for k=1..nlev-1
        active_below = face_active[..., 1:]      # cell k   for k=1..nlev-1
        G_interior = G_interior * (active_above * active_below)
    # "nemo_faithful": G_interior stays UNMASKED here (dynzad.F90:86-119 has
    # no per-face umask/vmask guard on ww or on the assembled flux) -- the
    # face_active gate is applied below, to the final tendency only.

    # Zero surface/bottom interface G via a single Pad HLO op (matches
    # NEMO's zWdzU=0 init at the surface and the never-read, architecturally
    # -zero bottom interface).
    pad_axes = ((0, 0),) * (G_interior.ndim - 1)
    G = jnp.pad(G_interior, (*pad_axes, (1, 1)))     # (..., nlev+1)

    # tend[k] uses G at the interface ABOVE (top, k) and BELOW (bottom, k+1)
    # the cell: G[..., :-1] = G[k] for k=0..nlev-1, G[..., 1:] = G[k+1].
    G_top = G[..., :-1]
    G_bot = G[..., 1:]
    h_u_safe = jnp.maximum(h_u, 1.0e-10)
    # Floor the face area too: a degenerate (zero-width) pole-row v-face has
    # face_area=0 with v itself masked to zero there, so the physical
    # tendency is zero — but 0 (masked G) * inf (1/0 area) is nan, not 0,
    # without the floor.
    face_area_safe = jnp.maximum(face_area, 1.0e-10)
    tend = -0.25 / (face_area_safe * h_u_safe) * (G_top + G_bot)
    if bottom_face_mask_mode == "nemo_faithful" and face_active is not None:
        # dynzdf.F90:121 -- ``puu(Kaa) = (puu(Kbb) + rDt*Krhs) * umask(ji,jj,jk)``:
        # the ONLY masking NEMO applies to this term, keyed on the cell's
        # OWN level jk (not an interface-neighbour AND).
        tend = tend * face_active
    return tend


# ---------------------------------------------------------------------------
# Adaptive-implicit vertical momentum advection
# (Shchepetkin 2015 / NEMO ``ln_zad_Aimp``)
# ---------------------------------------------------------------------------
#
# Explicit first-order-upwind vertical momentum advection (above) is only
# stable while the vertical Courant number ``Cw = |w| dt / h`` stays below
# 1.  In an OMIP cold-start the spurious equatorial pressure-gradient seed
# drives a transient convergence -> spurious ``w`` -> ``Cw > 1`` in thin
# cells, and the explicit scheme then amplifies it super-exponentially (the
# documented "vertadv is the residual amplifier" runaway).  NEMO removes
# this CFL limit with Shchepetkin's adaptive-implicit scheme: at each
# interface the vertical velocity is split ``w = w_exp + w_imp`` by a
# Courant-dependent fraction; ``w_exp`` (Courant-capped) goes through the
# normal explicit flux-form scheme, and ``w_imp`` is handled by a
# backward-Euler first-order-upwind solve that is unconditionally stable,
# monotone, and conservative (an M-matrix tridiagonal).
#
# Reference: A.F. Shchepetkin (2015), "An adaptive, Courant-number-dependent
# implicit scheme for vertical advection in oceanic modeling", Ocean
# Modelling 91, 38-69.  NEMO impl: sshwzv.F90 (split), trazdf.F90 /
# dynzdf.F90 (implicit solve).


def shchepetkin_implicit_fraction(
    cu: jnp.ndarray,
    cu_min: float = _AIMP_CU_MIN,
    cu_max: float = _AIMP_CU_MAX,
) -> jnp.ndarray:
    """Implicit fraction ``zcff(Cu)`` of Shchepetkin (2015) / NEMO wAimp.

    Maps a (non-negative) vertical Courant number ``cu`` to the fraction
    of the vertical velocity that is treated implicitly:

    - ``cu <= cu_min``                : ``0``  (fully explicit, high order)
    - ``cu_min < cu < cu_cut``        : ``d² / (Fcu + d²)``, ``d = cu-cu_min``
    - ``cu >= cu_cut``                : ``(cu - cu_max) / cu``
    - then clipped to ``<= 1``

    with ``cu_cut = 2 cu_max - cu_min`` and ``Fcu = 4 cu_max (cu_max-cu_min)``.
    The two interior branches join continuously at ``cu_cut`` (both give
    ``(cu_max-cu_min)/(2 cu_max-cu_min)``) and the ramp is monotone
    increasing from 0 to 1.  Pure ``jnp.where`` (no Python control flow) so
    it is safe on traced Courant numbers and differentiable.
    """
    cu_cut = 2.0 * cu_max - cu_min
    fcu = 4.0 * cu_max * (cu_max - cu_min)
    d = cu - cu_min
    mid = (d * d) / (fcu + d * d)
    # Guard the division in the high branch; cu >= cu_cut > 0 there.
    high = (cu - cu_max) / jnp.maximum(cu, 1.0e-30)
    zcff = jnp.where(
        cu <= cu_min,
        jnp.zeros_like(cu),
        jnp.where(cu < cu_cut, mid, high),
    )
    return jnp.minimum(zcff, 1.0)


def implicit_vertical_advection_ocean(
    field: jnp.ndarray,
    w_imp_half: jnp.ndarray,
    h: jnp.ndarray,
    dt: float,
    face_active: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Backward-Euler first-order-upwind vertical advection (one solve).

    Solves, per column, the unconditionally-stable implicit update

        field_new[k] + (dt/h[k]) * (F[k] - F[k+1]) = field[k]

    with the interface-upwind flux ``F[k] = max(w[k],0) field_new[k] +
    min(w[k],0) field_new[k-1]`` (``w`` positive = upward, interface ``k``
    sits above level ``k``; ``w[0] = w[nlev] = 0``).  Grouping by unknown
    gives the tridiagonal system

        a[k] =  dt * min(w_top[k], 0) / h[k]            (sub-diagonal)
        b[k] =  1 + dt*(max(w_top[k],0) - min(w_bot[k],0)) / h[k]  (diag)
        c[k] = -dt * max(w_bot[k], 0) / h[k]            (super-diagonal)
        d[k] =  field[k]                                (rhs)

    where ``w_top = w_imp_half[..., :-1]`` and ``w_bot = w_imp_half[..., 1:]``.
    This is an M-matrix (``b >= 1``, off-diagonals ``<= 0``) so the
    backward-Euler step is unconditionally stable and monotone, and the
    telescoping flux form conserves the column integral ``sum_k h[k]*field[k]``
    to machine precision (with the zero-flux top/bottom boundaries).

    Solver-conditioning note
    ------------------------
    ``thomas_solve`` adds a ``finfo(float32).tiny`` (~1.18e-38) guard to
    its pivots.  Because ``b >= 1`` here, ``b + tiny`` underflows back to
    ``b`` at both float32 and float64 (``tiny`` is below the ULP of any
    number ``>= 1``), and the forward-elimination pivots of a
    diagonally-dominant M-matrix stay ``O(1)`` so the ``|denom| < tiny``
    fallback is never selected.  The guard is therefore inert for this
    matrix class: the solve is *exactly* the conservative finite-volume
    solve, a ``w = 0`` system is bit-exact identity, and an inactive
    (``a = c = 0, b = 1``) row reproduces its input to round-off.
    (Verified: ``test_w_zero_is_exact_identity``,
    ``test_conservation_machine_precision_high_courant``.)

    Parameters mirror :func:`flux_form_vertical_momentum_advection`.
    ``face_active`` (1 = wet, 0 = below the partial seafloor) gates the
    implicit flux at interfaces bordering any inactive cell to exactly
    zero, so rock cells decouple (``a = c = 0``, ``b = 1`` -> identity) and
    cannot mix spurious values up the column.
    """
    nlev = field.shape[-1]
    if nlev < 2:
        return field

    w = w_imp_half
    if face_active is not None:
        # Interior interface i (1..nlev-1) is between cells i-1 and i;
        # gate it to zero unless both are active.  Surface/bottom
        # interfaces are zero by construction (w_half[0]=w_half[nlev]=0).
        active_above = face_active[..., :-1]   # cells 0..nlev-2
        active_below = face_active[..., 1:]    # cells 1..nlev-1
        face_int = active_above * active_below
        pad_axes = ((0, 0),) * (face_int.ndim - 1)
        gate = jnp.pad(face_int, (*pad_axes, (1, 1)))   # (..., nlev+1)
        w = w * gate

    w_top = w[..., :-1]    # interface above each cell, (..., nlev); w_top[0]=0
    w_bot = w[..., 1:]     # interface below each cell, (..., nlev); w_bot[-1]=0
    inv_h = 1.0 / jnp.maximum(h, _H_FLOOR)

    a = dt * jnp.minimum(w_top, 0.0) * inv_h
    c = -dt * jnp.maximum(w_bot, 0.0) * inv_h
    b = 1.0 + dt * (jnp.maximum(w_top, 0.0) - jnp.minimum(w_bot, 0.0)) * inv_h

    return thomas_solve(a, b, c, field)


def implicit_vertical_advection_ocean_advective(
    field: jnp.ndarray,
    w_imp_half: jnp.ndarray,
    h: jnp.ndarray,
    dt: float,
    face_active: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Backward-Euler upwind vertical advection in ADVECTIVE form (one solve).

    Transcribes the ``ln_zad_Aimp`` vector-invariant block of NEMO
    ``dynzdf.F90`` (~:235-250): with ``W_k = dt*(w[k]+w[k+1])/2`` (the cell
    mean of the implicit interface velocity, already area-averaged onto the
    momentum point by the caller) the row is

        field_new[k] + min(W_k,0)*(field_new[k-1]-field_new[k])/e3w[k]
                     + max(W_k,0)*(field_new[k]-field_new[k+1])/e3w[k+1]
        = field[k]

    with ``e3w[k] = (h[k-1]+h[k])/2`` the interface spacing (surface row: no
    ``k-1`` term).  Row sums of the added terms are zero, so a uniform field
    is invariant wherever the upstream neighbour is wet.  As in NEMO
    (dynzdf.F90:237-243, loop to jpkm1 against the masked level below), an
    upward ``W`` in the deepest wet cell couples to ZERO below it, so that
    level relaxes toward 0.  Deviation: the bottom row's ``e3w[k+1]`` is
    ``h[k]/2`` (NEMO: ``e3uw`` of the masked w-level).  Inactive rows return
    ``field`` unchanged.
    """
    nlev = field.shape[-1]
    if nlev < 2:
        return field
    act = (jnp.ones_like(h) if face_active is None
           else jnp.broadcast_to(face_active, h.shape).astype(h.dtype))
    hw = h * act
    w = w_imp_half
    W = dt * 0.5 * (w[..., :-1] + w[..., 1:]) * act          # (..., nlev)
    e3w_top = jnp.concatenate(
        [jnp.ones_like(hw[..., :1]), 0.5 * (hw[..., :-1] + hw[..., 1:])], axis=-1)
    e3w_bot = jnp.concatenate(
        [0.5 * (hw[..., :-1] + hw[..., 1:]), 0.5 * hw[..., -1:]], axis=-1)
    e3w_top = jnp.maximum(e3w_top, _H_FLOOR)
    e3w_bot = jnp.maximum(e3w_bot, _H_FLOOR)
    wn = jnp.minimum(W, 0.0) / e3w_top
    wn = wn.at[..., 0].set(0.0)                     # no k-1 above the surface
    wp = jnp.maximum(W, 0.0) / e3w_bot
    a = wn
    c = -wp
    b = 1.0 - wn + wp
    x = thomas_solve(a, b, c, field * act)
    return jnp.where(act > 0.0, x, field)


def adaptive_implicit_vertical_momentum_advection(
    u: jnp.ndarray,
    w_half: jnp.ndarray,
    h: jnp.ndarray,
    dt: float,
    *,
    cu_min: float = _AIMP_CU_MIN,
    cu_max: float = _AIMP_CU_MAX,
    face_active: jnp.ndarray | None = None,
    explicit_scheme: str = "upwind",
    w_area_half: jnp.ndarray | None = None,
    face_area: jnp.ndarray | None = None,
    bottom_face_mask_mode: str = "min_rule",
    w_imp_half: jnp.ndarray | None = None,
    w_imp_area_half: jnp.ndarray | None = None,
    implicit_form: str = "flux",
) -> jnp.ndarray:
    """Adaptive-implicit vertical momentum advection (Shchepetkin 2015).

    ``w_imp_half`` (with ``w_imp_area_half`` for the ``nemo_advective``
    explicit arm) supplies a PRECOMPUTED implicit share (NEMO
    ``wAimp_RK3_t``, area-averaged to the face) instead of the
    vertical-only Shchepetkin 0.15/0.30 fraction; ``implicit_form`` picks
    the implicit solve: ``"flux"`` (legacy) or ``"advective"`` (NEMO
    dynzdf vector form, :func:`implicit_vertical_advection_ocean_advective`).

    Returns the velocity field ``u`` after one step of vertical advection
    ``-w du/dz``, split into a Courant-capped explicit part and an
    unconditionally-stable backward-Euler implicit part so the explicit
    flux-form scheme can never violate the vertical CFL limit (NEMO
    ``ln_zad_Aimp``).  Reduces *exactly* to the explicit
    :func:`flux_form_vertical_momentum_advection` wherever the vertical
    Courant number stays below ``cu_min`` (``w_imp = 0`` there), so it is a
    drop-in robustness upgrade that only changes the answer in the
    high-Courant cells where the explicit scheme is unstable anyway.

    Parameters
    ----------
    u : array, shape ``(..., nlev)``
        Velocity at the momentum point (u-face or v-face).
    w_half : array, shape ``(..., nlev+1)``
        Vertical velocity on interfaces at the same momentum point,
        positive up, zero at surface and bottom.
    h : array, shape ``(..., nlev)``
        Layer thickness at the momentum point.
    dt : float
        Time step [s].
    cu_min, cu_max : float
        Shchepetkin Courant thresholds (see
        :func:`shchepetkin_implicit_fraction`).
    face_active : array | None, shape ``(..., nlev)``
        Per-level face-activity mask (1 = wet, 0 = below seafloor).
    """
    nlev = u.shape[-1]
    if nlev < 2:
        return u

    inv_h = 1.0 / jnp.maximum(h, _H_FLOOR)
    # Per-cell vertical (outflow) Courant number: the upward outflow
    # through the top interface plus the downward outflow through the
    # bottom interface, normalised by the cell thickness.  This is the
    # quantity the vertical-advection CFL limit constrains.  (NEMO also
    # folds in the horizontal flux divergence to form a single combined
    # Courant number; we use the vertical-only Courant because only the
    # vertical advection is being made implicit here -- the horizontal
    # CFL is governed separately by dt and the grid spacing.)
    w_top = w_half[..., :-1]
    w_bot = w_half[..., 1:]
    cu_cell = dt * (jnp.maximum(w_top, 0.0) - jnp.minimum(w_bot, 0.0)) * inv_h

    # Interface Courant number = max over the two adjacent cells
    # (interior interfaces 1..nlev-1); pad the surface/bottom interfaces
    # with zero (w_half is zero there anyway).
    cu_iface = jnp.maximum(cu_cell[..., :-1], cu_cell[..., 1:])  # (..., nlev-1)
    if implicit_form not in ("flux", "advective"):
        raise ValueError(
            f"implicit_form must be 'flux' or 'advective', got {implicit_form!r}")
    if w_imp_half is None:
        zcff_int = shchepetkin_implicit_fraction(cu_iface, cu_min, cu_max)
        pad_axes = ((0, 0),) * (zcff_int.ndim - 1)
        zcff = jnp.pad(zcff_int, (*pad_axes, (1, 1)))            # (..., nlev+1)
        w_imp = zcff * w_half
        w_exp = (1.0 - zcff) * w_half
        w_area_exp = None if w_area_half is None else (1.0 - zcff) * w_area_half
    else:
        if explicit_scheme == "nemo_advective" and w_imp_area_half is None:
            raise ValueError("w_imp_half with nemo_advective needs w_imp_area_half")
        w_imp = w_imp_half
        w_exp = w_half - w_imp_half
        w_area_exp = (None if w_area_half is None
                      else w_area_half - w_imp_area_half)

    # Explicit (Courant-capped) part.  ``nemo_up3`` selects the live
    # ln_dynadv_up3 vertical flux; the default is unchanged.
    if explicit_scheme == "upwind":
        tend_exp = flux_form_vertical_momentum_advection(
            u, w_exp, h, face_active=face_active,
        )
    elif explicit_scheme == "nemo_up3":
        tend_exp = nemo_up3_vertical_momentum_advection(
            u, w_exp, h, face_active=face_active,
        )
    elif explicit_scheme == "nemo_advective":
        # NEMO ln_dynadv_vec + ln_zad_Aimp: dynzad (centered advective, full u)
        # on the explicit share of the e1e2t-weighted w; the implicit share
        # goes to the upwind solve below, as dynzdf does with wi.
        if w_area_half is None or face_area is None:
            raise ValueError(
                "explicit_scheme='nemo_advective' needs w_area_half and face_area")
        tend_exp = nemo_advective_vertical_momentum_advection(
            u, w_area_exp, h, face_area, face_active=face_active,
            bottom_face_mask_mode=bottom_face_mask_mode,
        )
    else:
        raise ValueError(
            "explicit_scheme must be 'upwind', 'nemo_up3' or 'nemo_advective', got "
            f"{explicit_scheme!r}")
    u_exp = u + dt * tend_exp

    # Implicit part: unconditionally-stable backward-Euler upwind solve.
    if implicit_form == "advective":
        return implicit_vertical_advection_ocean_advective(
            u_exp, w_imp, h, dt, face_active=face_active)
    return implicit_vertical_advection_ocean(
        u_exp, w_imp, h, dt, face_active=face_active,
    )


def flux_form_vertical_tracer_advection(
    field: jnp.ndarray,
    w_half: jnp.ndarray,
    cell_active: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Flux-form vertical tracer advection with first-order upwind.

    Computes the vertical flux divergence  F_top[k] - F_bot[k]  for each
    level k, where F = w * T_face is the upward tracer flux on interfaces.

    Level convention
    ----------------
    k = 0 is surface, k = nlev-1 is bottom.
    Interface k sits ABOVE level k:
      - interface 0  = sea surface  (top of level 0)
      - interface k  = between level k-1 (above) and level k (below), k=1..nlev-1
      - interface nlev = ocean bottom (below level nlev-1)
    w positive = upward.

    Upwind at interior interface k (k = 1 .. nlev-1):
      - w[k] > 0  (upward):  fluid from level k  (below) → T_face = field[k]
      - w[k] <= 0 (downward): fluid from level k-1 (above) → T_face = field[k-1]

    Surface and bottom fluxes are zero (w[0] = w[nlev] = 0 by construction).

    Parameters
    ----------
    field : array, shape (..., nlev)
        Tracer at full levels (e.g. temperature [degC]).
    w_half : array, shape (..., nlev+1)
        Vertical velocity on half (interface) levels [m/s].
    cell_active : array | None, shape (..., nlev)
        Optional per-cell activity mask (1 = wet, 0 = below seafloor).
        When provided, the flux at any interface bordering an inactive
        cell is gated to exactly zero — needed on partial-cell grids
        where ``w_half`` may carry float-precision noise (~1e-10 m/s)
        at inactive interfaces.  Without the gate, that noise produces
        a tiny spurious ``vert_flux_div`` at inactive cells; combined
        with the ``max(h_k_new, 1e-10)`` floor at the caller, this can
        amplify into ~1e3 spurious tracer values inside the rock, then
        propagate into the EOS as huge density and break the model.
        Also makes adjoint sensitivities through inactive cells exactly
        zero (under ``jax.grad``), instead of poorly-conditioned values
        depending on the float noise.

    Returns
    -------
    vert_flux_div : array, shape (..., nlev)
        Vertical flux divergence  F_top[k] - F_bot[k]  for each level.
        Units are [tracer] * [m/s]  (NOT divided by layer thickness).
        The caller uses:  h_new*T_new = h_old*T_old - dt*vert_flux_div - dt*horiz_flux_div
    """
    nlev = field.shape[-1]

    # --- Compute upwind tracer flux at each interface ---
    # F has shape (..., nlev+1).  F[..., 0] = 0, F[..., nlev] = 0.
    # For interior interface k (1 <= k <= nlev-1):
    #   F[k] = w[k] * T_face[k]
    #   where T_face[k] = field[k]   if w[k] > 0   (upward, from below)
    #                    = field[k-1] if w[k] <= 0  (downward, from above)

    # Interior w values: w_half[..., 1:nlev] has shape (..., nlev-1)
    w_interior = w_half[..., 1:nlev]  # (..., nlev-1)

    # Upwind selection at interior interfaces
    # Interface k (1-indexed) is between level k-1 (above) and level k (below)
    T_below = field[..., 1:]    # field[k]   for k=1..nlev-1 → (..., nlev-1)
    T_above = field[..., :-1]   # field[k-1] for k=1..nlev-1 → (..., nlev-1)

    T_face_interior = jnp.where(w_interior > 0.0, T_below, T_above)
    F_interior = w_interior * T_face_interior  # (..., nlev-1)

    # Mask the flux at interfaces that border any inactive cell.  An
    # interior interface k (k=1..nlev-1) is between cells k-1 and k —
    # both must be active for the flux there to be physical.  The
    # surface (k=0) and bottom (k=nlev) interfaces are already zero by
    # the pad below.
    if cell_active is not None:
        active_above = cell_active[..., :-1]   # cells k-1 for k=1..nlev-1
        active_below = cell_active[..., 1:]    # cells k   for k=1..nlev-1
        face_active_interior = active_above * active_below
        F_interior = F_interior * face_active_interior

    # Full flux array with zero boundaries — single Pad HLO op vs
    # alloc fresh ``(..., 1)`` zero buffer and 3-array concatenate.
    pad_axes_f = ((0, 0),) * (F_interior.ndim - 1)
    F = jnp.pad(F_interior, (*pad_axes_f, (1, 1)))  # (..., nlev+1)

    # Flux divergence: F_top[k] - F_bot[k] = F[k] - F[k+1]
    vert_flux_div = F[..., :-1] - F[..., 1:]  # (..., nlev)

    return vert_flux_div


def flux_form_vertical_tracer_advection_centered(
    field: jnp.ndarray,
    w_half: jnp.ndarray,
    cell_active: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Flux-form vertical tracer advection with UNLIMITED centered 2nd order.

    Veros ``adv_flux_2nd`` vertical flux (``veros/core/advection.py``):
    the interface tracer flux uses the plain 2-cell average
    ``T_face[k] = 0.5*(field[k-1] + field[k])`` rather than the upwind /
    TVD-limited value.  This is the Veros ACC tracer scheme
    (``enable_superbee_advection=False``).

    Same level convention and output semantics as
    :func:`flux_form_vertical_tracer_advection` (the 1st-order upwind
    version): k=0 surface, interface k sits ABOVE level k, surface and
    bottom interface fluxes are zero, and the returned ``vert_flux_div``
    is ``F_top[k] - F_bot[k]`` in units ``[tracer]*[m/s]`` (NOT divided
    by layer thickness).

    DISPERSION
    ----------
    The centered face value is UNLIMITED, so this scheme is dispersive:
    it can produce over/undershoots (new local extrema, locally negative
    tracer) near sharp gradients.  It carries zero implicit diapycnal
    diffusion in smooth regions (unlike upwind/TVD).  Used for the
    Veros-faithful ACC comparison (``tracer_advection="centered"``);
    legoESM's production default stays TVD (Van Leer), which is monotone.

    Parameters
    ----------
    field : array, shape (..., nlev)
        Tracer at full levels.
    w_half : array, shape (..., nlev+1)
        Vertical velocity on half (interface) levels [m/s]; positive =
        upward; zero at surface and bottom.
    cell_active : array | None, shape (..., nlev)
        Optional per-cell activity mask (1 = wet, 0 = below seafloor).
        When provided, the flux at any interface bordering an inactive
        cell is gated to exactly zero (same role as in
        :func:`flux_form_vertical_tracer_advection`).

    Returns
    -------
    vert_flux_div : array, shape (..., nlev)
        Vertical flux divergence ``F_top[k] - F_bot[k]`` for each level.
    """
    nlev = field.shape[-1]

    # Interior interface k (1 <= k <= nlev-1) is between level k-1 (above)
    # and level k (below).  Centered face value = unlimited 2-cell average.
    w_interior = w_half[..., 1:nlev]    # (..., nlev-1)
    T_below = field[..., 1:]            # field[k]   for k=1..nlev-1
    T_above = field[..., :-1]           # field[k-1] for k=1..nlev-1
    T_face_interior = 0.5 * (T_above + T_below)
    F_interior = w_interior * T_face_interior  # (..., nlev-1)

    # Gate the flux at interfaces bordering any inactive cell (matches the
    # upwind version: both bordering cells must be active to be physical).
    if cell_active is not None:
        active_above = cell_active[..., :-1]   # cells k-1 for k=1..nlev-1
        active_below = cell_active[..., 1:]    # cells k   for k=1..nlev-1
        F_interior = F_interior * (active_above * active_below)

    # Zero surface/bottom interface fluxes via a single Pad HLO op.
    pad_axes_f = ((0, 0),) * (F_interior.ndim - 1)
    F = jnp.pad(F_interior, (*pad_axes_f, (1, 1)))  # (..., nlev+1)

    # Flux divergence: F_top[k] - F_bot[k] = F[k] - F[k+1]
    vert_flux_div = F[..., :-1] - F[..., 1:]
    return vert_flux_div


# Canonical Van Leer limiter from core (redundancy audit), aliased to the local
# vertical-advection private name so call sites are unchanged.
from legoesm.core.flux_limiters import (
    grad_safe_ratio as _grad_safe_ratio,
    ratio_grad_floor as _ratio_grad_floor,
    van_leer_limiter as _van_leer_limiter_vert,
)


def flux_form_vertical_tracer_advection_tvd(
    field: jnp.ndarray,
    w_half: jnp.ndarray,
    h_k: jnp.ndarray,
    dt: float,
    cell_active: jnp.ndarray | None = None,
    limiter_fn=_van_leer_limiter_vert,
) -> jnp.ndarray:
    """Flux-form vertical tracer advection with TVD scheme.

    ``limiter_fn`` selects the flux limiter family; defaults to Van Leer
    (used by ``tracer_advection="tvd"``). Pass
    :func:`legoesm.ocean.dynamics._flux_limiters.sweby_limiter` for
    Veros-compatible superbee (``tracer_advection="superbee"``).

    Second-order accurate in smooth regions, falls back to first-order
    upwind at discontinuities.  Monotone (no new extrema).  The implicit
    numerical diffusivity is dramatically reduced compared to first-order
    upwind: K_num ~ 0 in smooth regions vs K_num ~ |w|*dz/2 for upwind.

    Same output semantics as flux_form_vertical_tracer_advection.

    Parameters
    ----------
    field : array, shape (..., nlev)
        Tracer at full levels.
    w_half : array, shape (..., nlev+1)
        Vertical velocity on half (interface) levels [m/s].
        Positive = upward. Zero at surface and bottom.
    h_k : array, shape (..., nlev)
        Layer thickness [m] at full levels (z-star actual thickness).
    dt : float
        Time step [s], for CFL computation.
    cell_active : array, shape (..., nlev), optional
        Per-level active mask (1=ocean, 0=sub-seafloor).  When provided,
        sub-seafloor ghost values in the upwind-of-upwind stencil are
        replaced with the boundary active value, preventing the TVD
        limiter from seeing T=0/S=0 below the seafloor.

    Returns
    -------
    vert_flux_div : array, shape (..., nlev)
        Vertical flux divergence F_top[k] - F_bot[k] for each level.
        Units: [tracer]*[m/s] (NOT divided by layer thickness).
    """
    eps = 1e-30
    nlev = field.shape[-1]

    # On partial cells, replace sub-seafloor values with the nearest
    # active value above.  This prevents the TVD upwind-of-upwind
    # stencil from seeing T=0/S=0 below the seafloor.
    if cell_active is not None:
        # Propagate bottom active value downward through inactive levels.
        # Scan from top to bottom: if level k is inactive, copy from k-1.
        def _fill_down(carry, k):
            prev = carry
            cur = field[..., k]
            active_k = cell_active[..., k] > 0.5
            filled = jnp.where(active_k, cur, prev)
            return filled, filled
        import jax.lax
        _, filled_cols = jax.lax.scan(
            _fill_down, field[..., 0], jnp.arange(nlev))
        # filled_cols is (nlev, ...) — transpose back to (..., nlev)
        field_safe = jnp.moveaxis(filled_cols, 0, -1)
    else:
        field_safe = field

    # Interior interface values: k = 1..nlev-1
    w_interior = w_half[..., 1:nlev]   # (..., nlev-1)
    T_below = field_safe[..., 1:]      # field[k]   for k=1..nlev-1
    T_above = field_safe[..., :-1]     # field[k-1] for k=1..nlev-1

    # --- First-order upwind flux ---
    T_upwind = jnp.where(w_interior > 0.0, T_below, T_above)
    F_upwind = w_interior * T_upwind

    # --- CFL number at each interface ---
    h_below = h_k[..., 1:]            # h[k]   for k=1..nlev-1
    h_above = h_k[..., :-1]           # h[k-1] for k=1..nlev-1
    h_donor = jnp.where(w_interior > 0.0, h_below, h_above)
    t_grad = _ratio_grad_floor(field.dtype)
    CFL = _grad_safe_ratio(
        jnp.abs(w_interior) * dt, jnp.maximum(h_donor, eps), h_donor > t_grad)
    CFL = jnp.minimum(CFL, 1.0)

    # --- Smoothness ratio r ---
    # Local gradient across interface k:
    delta = T_above - T_below          # field[k-1] - field[k]

    # Upwind-of-upwind gradient:
    # For upward flow (w>0), donor=k(below): need field[k]-field[k+1]
    # For downward flow (w<=0), donor=k-1(above): need field[k-2]-field[k-1]
    # Ghost cells at boundaries copy boundary value → delta=0 → r=0 → upwind.
    # Using field_safe ensures sub-seafloor ghost = bottom active value.
    field_bot_ghost = jnp.concatenate(
        [field_safe, field_safe[..., -1:]], axis=-1)     # ghost at bottom
    field_top_ghost = jnp.concatenate(
        [field_safe[..., :1], field_safe], axis=-1)      # ghost at top

    # Upwind gradient for upward flow: field[k] - field[k+1]
    delta_upwind_up = field_bot_ghost[..., 1:nlev] - field_bot_ghost[..., 2:nlev + 1]
    # Upwind gradient for downward flow: field[k-2] - field[k-1]
    delta_upwind_down = field_top_ghost[..., :nlev - 1] - field_top_ghost[..., 1:nlev]

    delta_upwind = jnp.where(w_interior > 0.0, delta_upwind_up, delta_upwind_down)

    # r = upwind_gradient / local_gradient
    r = _grad_safe_ratio(
        delta_upwind,
        jnp.where(jnp.abs(delta) > eps, delta, eps),
        jnp.abs(delta) > t_grad,
    )

    # --- Flux limiter and TVD correction ---
    phi = limiter_fn(r)
    F_interior = F_upwind + 0.5 * jnp.abs(w_interior) * (1.0 - CFL) * phi * delta

    # Full flux array with zero boundaries — single Pad HLO op.
    pad_axes_t = ((0, 0),) * (F_interior.ndim - 1)
    F = jnp.pad(F_interior, (*pad_axes_t, (1, 1)))

    # Flux divergence: F_top[k] - F_bot[k] = F[k] - F[k+1]
    vert_flux_div = F[..., :-1] - F[..., 1:]

    return vert_flux_div
