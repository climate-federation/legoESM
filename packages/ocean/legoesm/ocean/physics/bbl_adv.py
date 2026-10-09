"""Bottom boundary layers from NEMO ``trabbl``.

This module carries the selectable diffusive ``nn_bbl_ldf=1`` and advective
Campin--Goosse ``nn_bbl_adv=2`` arms in one shared implementation.

At coarse resolution (1 deg), dense shelf/overflow water (Mediterranean at
Gibraltar, Denmark Strait, Antarctic shelves) cannot descend the continental
slope: the staircase topography mixes it horizontally at sill depth and the
overflow stalls.  NEMO ORCA1 solves this with an ADVECTIVE bottom-boundary-
layer scheme (``ln_trabbl``, ``nn_bbl_adv=2``, ``rn_gambbl=20 s``): wherever
the up-slope (shelf) BOTTOM cell is denser than the down-slope (deep) BOTTOM
cell at a common reference depth, a down-slope transport

    tr_bbl = (face width) * e3_bbl * g * gamma * max(0, drho/rho0)

carries the dense water from the shelf bottom to the DEEP column's bottom,
with an upward return flow through the deep column and a horizontal return
at shelf level — a closed, exactly tracer-conserving circulation cell
(NEMO ``tra_bbl_adv``):

    shelf bottom  (iis, ks):  += |tr| * (pt[deep, ks]  - pt[shelf, ks]) / V
    deep interior (iid, k) :  += |tr| * (pt[k+1]       - pt[k])         / V
                                          for ks <= k < kd
    deep bottom   (iid, kd):  += |tr| * (pt[shelf, ks] - pt[deep, kd])  / V

(V = cell area * thickness; the weighted sum telescopes to zero.)

Geometry is STATIC (NEMO tra_bbl_init): per interior u/v face,
``mgrh = sign(bottom-depth difference)`` (0 when flat), shelf/deep bottom
level indices, and ``e3_bbl = min`` of the two columns' bottom-cell
thicknesses.

This module implements the lat-lon C-grid family version (regular lat-lon +
ORCA tripole, array layout ``(n_lat, n_lon, nlev)``) as pure JAX with STATIC
unrolled level loops — jit/grad safe.  East-west periodicity: faces are built
for column pairs ``(i, i+1)`` for ``i in 0..n_lon-2`` (interior faces only;
on the tripole the cyclic halo columns are slaved by ``ew_cyclic_overlap``
so the seam exchange is represented through the overlap, and on a regular
grid the wrap face is omitted — one face of ~360 at 1 deg, negligible and
safe).  References: Beckmann & Doscher (1997) JPO; Campin & Goosse (1999)
Tellus; NEMO 5.0.2 ``TRA/trabbl.F90``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.core.source_rounding import nemo_source_round


__physics_contract__ = {
    "summary": (
        "Selectable NEMO trabbl identities: nn_bbl_ldf=1 applies the source "
        "bottom-cell diffusive flux divergence through gated ahu_bbl/"
        "ahv_bbl; nn_bbl_adv=2 applies the Campin-Goosse down-slope "
        "transport through a closed three-leg tracer exchange. Both are "
        "shared implementations selected by the NEMO namelist identity."
    ),
    "inputs": {
        "T": "degC", "S": "PSU", "h_ref": "m", "land_mask": "1",
        "area": "m^2", "dy_u_faces": "m", "dx_v_faces": "m",
        "gamma_s": "s", "rho_0": "kg/m^3", "dt": "s",
    },
    "outputs": {
        "utr_bbl": "m^3/s", "vtr_bbl": "m^3/s",
        "dT_dt": "degC/s", "dS_dt": "PSU/s",
    },
    "sign_convention": (
        "Transports are signed down-slope (positive toward the larger "
        "index when the neighbour is deeper, like NEMO mgrh), and are "
        "non-zero only when the shelf bottom is DENSER; the tendency "
        "moves the deep bottom cell toward the shelf water properties."
    ),
    # The 3-leg exchange telescopes to zero under the area*h volume
    # weights: total heat and salt are conserved exactly (unit-tested to
    # fp tolerance on a random staircase domain).
    "conserves": ["tracer", "salt"],
    "differentiable": True,
    "reference": (
        "Campin & Goosse (1999) Tellus 51A 412-430; Beckmann & Doscher "
        "(1997) JPO 27 581-591; NEMO 5.0.2 TRA/trabbl.F90:187-200, "
        "342-380, 507-537 (ORCA2 nn_bbl_ldf=1) and :243-284 "
        "(ORCA1/OVERFLOW nn_bbl_adv=2)"
    ),
    "idealized_test": (
        "tests/ocean/unit/test_bbl_adv.py: diffusive source coefficient EOS "
        "selector and JIT/grad-safe bottom RHS; analytic two-column "
        "dense-shelf overflow, exact conservation, and inactive-face gates."
    ),
}


class BBLGeometry(NamedTuple):
    """Static BBL face geometry (NEMO tra_bbl_init).

    All arrays are FACE-indexed: i-faces ``(n_lat, n_lon-1)`` between columns
    ``(j, i)`` and ``(j, i+1)``; j-faces ``(n_lat-1, n_lon)`` between
    ``(j, i)`` and ``(j+1, i)``.
    """
    mgrhu: jnp.ndarray      # i-face slope sign: +1 deeper at i+1, -1 deeper at i, 0 flat
    mgrhv: jnp.ndarray      # j-face slope sign
    ku_s: jnp.ndarray       # i-face SHELF (shallow) bottom level index
    ku_d: jnp.ndarray       # i-face DEEP bottom level index
    kv_s: jnp.ndarray       # j-face shelf bottom level
    kv_d: jnp.ndarray       # j-face deep bottom level
    e3u_bbl: jnp.ndarray    # i-face BBL thickness = min(bottom e3 of the 2 columns)
    e3v_bbl: jnp.ndarray    # j-face BBL thickness
    dep_bot: jnp.ndarray    # per-CELL bottom mid-cell depth [m] (n_lat, n_lon)
    u_active: jnp.ndarray   # i-face both-columns-wet AND sloped (float 0/1)
    v_active: jnp.ndarray   # j-face mask
    bot_k: jnp.ndarray      # per-CELL bottom level index (n_lat, n_lon)
    h_ref: jnp.ndarray      # per-cell reference thicknesses (n_lat, n_lon, nlev)


class BBLDiffusiveGeometry(NamedTuple):
    """Static full-domain operands for NEMO ``nn_bbl_ldf=1``.

    U/V arrays use NEMO's native convention: element ``(j,i)`` is the east
    or north face of T cell ``(j,i)``.  This differs from the redundant-edge
    representation used by generic C-grid operators and is intentional.
    """

    bot_k: jnp.ndarray
    dep_bot_ref: jnp.ndarray
    mgrhu: jnp.ndarray
    mgrhv: jnp.ndarray
    ahu_bbl_0: jnp.ndarray
    ahv_bbl_0: jnp.ndarray
    t_active: jnp.ndarray
    u_active: jnp.ndarray
    v_active: jnp.ndarray


def _north_cell(value: jnp.ndarray, grid) -> jnp.ndarray:
    """T-point value immediately north, including an ORCA T-fold."""
    north = jnp.concatenate([value[1:], value[-1:]], axis=0)
    fold = getattr(grid, "fold", None)
    if fold is not None and bool(getattr(fold, "is_active", False)):
        north = north.at[-1].set(value[-1, fold.perm_T])
    return north


def _gather_level(field: jnp.ndarray, level: jnp.ndarray) -> jnp.ndarray:
    return jnp.take_along_axis(field, level[..., None], axis=-1)[..., 0]


def nemo_bbl_diffusive_geometry(
    h_ref: jnp.ndarray,
    land_mask: jnp.ndarray,
    gdept_0: jnp.ndarray,
    e3u_0: jnp.ndarray,
    e3v_0: jnp.ndarray,
    e1u: jnp.ndarray,
    e2u: jnp.ndarray,
    e1v: jnp.ndarray,
    e2v: jnp.ndarray,
    umask: jnp.ndarray,
    vmask: jnp.ndarray,
    *,
    aht_m2_s: float,
    grid,
) -> BBLDiffusiveGeometry:
    """Transcribe ``tra_bbl_init`` for the diffusive BBL arm.

    Source: NEMO 5.0.2 ``trabbl.F90:507-537``.  The supplied reference
    arrays are the native NEMO T/U/V arrays read from ``domain_cfg.nc``;
    no face reconstruction is substituted.
    """
    b = nemo_source_round
    h = jnp.asarray(h_ref)
    dtype = h.dtype
    active3 = h > jnp.asarray(1.0e-3, dtype=dtype)  # coeff-ok: wet-cell thickness floor [m]
    t_active = jnp.asarray(land_mask, dtype=dtype) > 0.5
    bot_k = jnp.maximum(jnp.sum(active3.astype(jnp.int32), axis=-1) - 1, 0)
    depth = jnp.asarray(gdept_0, dtype=dtype)
    e3u = jnp.asarray(e3u_0, dtype=dtype)
    e3v = jnp.asarray(e3v_0, dtype=dtype)
    if depth.shape != h.shape or e3u.shape != h.shape or e3v.shape != h.shape:
        raise ValueError("diffusive BBL reference depth/e3 arrays must match h_ref")

    dep_bot = _gather_level(depth, bot_k)
    east_bot = jnp.roll(bot_k, -1, axis=1)
    north_bot = _north_cell(bot_k, grid)
    east_dep = jnp.roll(dep_bot, -1, axis=1)
    north_dep = _north_cell(dep_bot, grid)
    mgrhu = jnp.sign(b(east_dep - dep_bot)).astype(jnp.int32)
    mgrhv = jnp.sign(b(north_dep - dep_bot)).astype(jnp.int32)

    e3u_here = _gather_level(e3u, bot_k)
    e3u_there = _gather_level(e3u, east_bot)
    e3v_here = _gather_level(e3v, bot_k)
    e3v_there = _gather_level(e3v, north_bot)
    e3u_bbl = jnp.minimum(e3u_here, e3u_there)
    e3v_bbl = jnp.minimum(e3v_here, e3v_there)

    um = jnp.asarray(umask, dtype=dtype)
    vm = jnp.asarray(vmask, dtype=dtype)
    if um.ndim == 3:
        um = jnp.max(um, axis=-1)
    if vm.ndim == 3:
        vm = jnp.max(vm, axis=-1)
    aht = jnp.asarray(aht_m2_s, dtype=dtype)
    # trabbl.F90:535-537.  e2_e1u/e1_e2v are the source divisions stored by
    # NEMO's domain initialization before these products are evaluated.
    e2_e1u = b(jnp.asarray(e2u, dtype=dtype) / jnp.asarray(e1u, dtype=dtype))
    e1_e2v = b(jnp.asarray(e1v, dtype=dtype) / jnp.asarray(e2v, dtype=dtype))
    ahu0 = b(b(b(aht * e2_e1u) * e3u_bbl) * um)
    ahv0 = b(b(b(aht * e1_e2v) * e3v_bbl) * vm)
    return BBLDiffusiveGeometry(
        bot_k=bot_k, dep_bot_ref=dep_bot, mgrhu=mgrhu, mgrhv=mgrhv,
        ahu_bbl_0=ahu0, ahv_bbl_0=ahv0, t_active=t_active,
        u_active=um > 0.5, v_active=vm > 0.5,
    )


def nemo_bbl_diffusive_coefficients(
    T: jnp.ndarray,
    S: jnp.ndarray,
    geom: BBLDiffusiveGeometry,
    *,
    bottom_depth_m: jnp.ndarray,
    rho_0: float,
    grid,
    eos_form: str = "teos10",
    seos_cfg=None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate NEMO ``bbl``'s diffusive face gate.

    Source: ``trabbl.F90:342-380``.  In particular this preserves NEMO's
    two-times-alpha/beta sums and Fortran ``SIGN`` zero convention; it does
    not replace the gate with a direct density comparison.  ``eos_form=
    "nemo_seos"`` takes the ``np_seos`` branch of ``eos_rab`` (eosbn2.F90
    rab_2d) with the deck's ``seos_cfg`` coefficients, which it requires.
    """
    from legoesm.ocean.eos import nemo_roquet_alpha_beta, nemo_seos_alpha_beta

    b = nemo_source_round
    dtype = jnp.asarray(T).dtype
    half = jnp.asarray(0.5, dtype=dtype)
    Tb = _gather_level(jnp.asarray(T), geom.bot_k)
    Sb = _gather_level(jnp.asarray(S), geom.bot_k)
    depth = jnp.asarray(bottom_depth_m, dtype=dtype)
    if eos_form == "nemo_seos":
        if seos_cfg is None:
            raise ValueError(
                "eos_form='nemo_seos' needs the deck's NemoSEOSConfig")
        alpha, beta = nemo_seos_alpha_beta(Tb, Sb, depth, cfg=seos_cfg)
    else:
        alpha, beta = nemo_roquet_alpha_beta(
            Tb, Sb, depth, rho0=rho_0, eos_form=eos_form)

    def sign_half(argument):
        # GFortran's SIGN result for a zero second argument is +ABS(first),
        # including when the arithmetic expression carries a -0 sign bit.
        return jnp.where(argument >= 0.0, half, -half)

    Te, Se = jnp.roll(Tb, -1, axis=1), jnp.roll(Sb, -1, axis=1)
    ae, be = jnp.roll(alpha, -1, axis=1), jnp.roll(beta, -1, axis=1)
    za = b(ae + alpha)
    zb = b(be + beta)
    zgdrho_u = b(b(b(za * b(Te - Tb)) - b(zb * b(Se - Sb)))
                   * geom.u_active.astype(dtype))
    arg_u = b(b(-zgdrho_u) * geom.mgrhu.astype(dtype))
    zsign_u = sign_half(arg_u)
    ahu = b(b(half - zsign_u) * geom.ahu_bbl_0)

    Tn, Sn = _north_cell(Tb, grid), _north_cell(Sb, grid)
    an, bn = _north_cell(alpha, grid), _north_cell(beta, grid)
    za = b(an + alpha)
    zb = b(bn + beta)
    zgdrho_v = b(b(b(za * b(Tn - Tb)) - b(zb * b(Sn - Sb)))
                   * geom.v_active.astype(dtype))
    arg_v = b(b(-zgdrho_v) * geom.mgrhv.astype(dtype))
    zsign_v = sign_half(arg_v)
    ahv = b(b(half - zsign_v) * geom.ahv_bbl_0)
    return ahu, ahv


def apply_bbl_diffusive_tendency(
    dT_dt: jnp.ndarray,
    dS_dt: jnp.ndarray,
    T: jnp.ndarray,
    S: jnp.ndarray,
    h_k: jnp.ndarray,
    area: jnp.ndarray,
    geom: BBLDiffusiveGeometry,
    ahu_bbl: jnp.ndarray,
    ahv_bbl: jnp.ndarray,
    *,
    grid,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Add ``tra_bbl_dif`` to the bottom-cell tracer RHS.

    This is the literal ``trabbl.F90:187-200`` association: U and V flux
    pairs are grouped separately, their two divergences are added, the result
    is multiplied by ``r1_e1e2t/e3t(Kmm)``, then added to Krhs.
    """
    b = nemo_source_round
    dtype = jnp.asarray(T).dtype
    area = jnp.asarray(area, dtype=dtype)
    h_k = jnp.asarray(h_k, dtype=dtype)
    active = geom.t_active
    bottom_h = _gather_level(h_k, geom.bot_k)
    safe_area = jnp.where(active, area, jnp.asarray(1.0, dtype=dtype))
    safe_h = jnp.where(active, bottom_h, jnp.asarray(1.0, dtype=dtype))
    r1_area = b(jnp.asarray(1.0, dtype=dtype) / safe_area)

    ahu = jnp.asarray(ahu_bbl, dtype=dtype)
    ahv = jnp.asarray(ahv_bbl, dtype=dtype)

    def apply(rhs, tracer):
        zptb = _gather_level(jnp.asarray(tracer), geom.bot_k)
        east, west = jnp.roll(zptb, -1, axis=1), jnp.roll(zptb, 1, axis=1)
        north = _north_cell(zptb, grid)
        south = jnp.concatenate([zptb[:1], zptb[:-1]], axis=0)
        ahu_w = jnp.roll(ahu, 1, axis=1)
        ahv_s = jnp.concatenate([jnp.zeros_like(ahv[:1]), ahv[:-1]], axis=0)
        u_pair = b(b(ahu * b(east - zptb))
                   - b(ahu_w * b(zptb - west)))
        v_pair = b(b(ahv * b(north - zptb))
                   - b(ahv_s * b(zptb - south)))
        # trabbl.F90:194-200 evaluates the completed horizontal divergence,
        # multiplies by the stored r1_e1e2t, and then performs one ordinary
        # division by e3t(Kmm).  Materialize BOTH operands so XLA cannot
        # reassociate ``numerator / e3t`` to ``numerator * (1/e3t)`` or fuse
        # the preceding product into the divide.  The quotient itself stays a
        # plain IEEE division: compiler control belongs here; nextafter search
        # over oracle-adjacent values does not.
        numerator = b(b(u_pair + v_pair) * r1_area)
        denominator = b(safe_h)
        increment = b(b(numerator) / b(denominator))
        before = _gather_level(jnp.asarray(rhs), geom.bot_k)
        after = jnp.where(active, b(before + increment), before)
        return jnp.asarray(rhs).at[
            jnp.arange(rhs.shape[0])[:, None],
            jnp.arange(rhs.shape[1])[None, :], geom.bot_k,
        ].set(after)

    return apply(dT_dt, T), apply(dS_dt, S)


def bbl_static_geometry(h_ref: jnp.ndarray, land_mask: jnp.ndarray
                        ) -> BBLGeometry:
    """Build the static BBL geometry from per-cell REFERENCE layer
    thicknesses ``h_ref`` (n_lat, n_lon, nlev; 0 below the seafloor —
    partial-cell aware) and the 2-D ocean ``land_mask``.

    NEMO equivalents: ``mbkt`` (bottom level), ``gdept_0`` at the bottom
    (here: cumulative mid-cell depth), ``mgrhu/v = sign(d_bot(i+1)-d_bot(i))``,
    ``mbku_d = max(mbkt, mbkt_neighbour)``, ``e3u_bbl_0 = min`` of the two
    bottom thicknesses.
    """
    h = jnp.asarray(h_ref, dtype=jnp.float64)
    mask = jnp.asarray(land_mask, dtype=jnp.float64)
    wet3 = h > 1.0e-3  # coeff-ok: wet-cell thickness floor [m]
    n_active = jnp.sum(wet3.astype(jnp.int32), axis=-1)          # (ny, nx)
    bot_k = jnp.maximum(n_active - 1, 0)                          # bottom index
    # mid-cell depths + bottom-cell depth/thickness per column
    cum = jnp.cumsum(h, axis=-1)
    z_mid = cum - 0.5 * h                                         # (ny,nx,nl)
    dep_bot = jnp.take_along_axis(z_mid, bot_k[..., None], axis=-1)[..., 0]
    e3_bot = jnp.take_along_axis(h, bot_k[..., None], axis=-1)[..., 0]

    def _faces(a_l, a_r):
        return a_l, a_r

    # --- i-faces: columns (j, i) vs (j, i+1) -------------------------------
    dL, dR = dep_bot[:, :-1], dep_bot[:, 1:]
    mgrhu = jnp.sign(dR - dL)
    ku_s = jnp.where(mgrhu >= 0, bot_k[:, :-1], bot_k[:, 1:])     # shallow col
    ku_d = jnp.maximum(bot_k[:, :-1], bot_k[:, 1:])               # NEMO mbku_d
    e3u_bbl = jnp.minimum(e3_bot[:, :-1], e3_bot[:, 1:])
    u_active = ((mask[:, :-1] > 0.5) & (mask[:, 1:] > 0.5)
                & (mgrhu != 0)).astype(jnp.float64)

    # --- j-faces: columns (j, i) vs (j+1, i) -------------------------------
    dS_, dN = dep_bot[:-1, :], dep_bot[1:, :]
    mgrhv = jnp.sign(dN - dS_)
    kv_s = jnp.where(mgrhv >= 0, bot_k[:-1, :], bot_k[1:, :])
    kv_d = jnp.maximum(bot_k[:-1, :], bot_k[1:, :])
    e3v_bbl = jnp.minimum(e3_bot[:-1, :], e3_bot[1:, :])
    v_active = ((mask[:-1, :] > 0.5) & (mask[1:, :] > 0.5)
                & (mgrhv != 0)).astype(jnp.float64)

    return BBLGeometry(
        mgrhu=mgrhu, mgrhv=mgrhv, ku_s=ku_s.astype(jnp.int32),
        ku_d=ku_d.astype(jnp.int32), kv_s=kv_s.astype(jnp.int32),
        kv_d=kv_d.astype(jnp.int32), e3u_bbl=e3u_bbl, e3v_bbl=e3v_bbl,
        dep_bot=dep_bot, u_active=u_active, v_active=v_active,
        bot_k=bot_k.astype(jnp.int32), h_ref=h,
    )


def nemo_bbl_static_geometry(
    h_ref: jnp.ndarray,
    land_mask: jnp.ndarray,
    gdept_0: jnp.ndarray,
    e3u_0: jnp.ndarray,
    e3v_0: jnp.ndarray,
) -> BBLGeometry:
    """NEMO ``tra_bbl_init`` geometry from its reference mesh operands.

    ``trabbl.F90:507-533`` does *not* infer slope from the continuous water
    column or the partial-cell centroid.  It gathers ``gdept_0`` at each
    column's ``mbkt`` and signs that reference-depth difference; its BBL
    thickness is the minimum of the supplied U/V-face ``e3*_0`` evaluated at
    the two adjacent bottom indices.  Those face fields remain defined below
    the shallower column's wet mask and therefore cannot be reconstructed from
    masked ``h_ref`` with a cell-to-face minimum.

    This is an operand builder, not a second Campin--Goosse implementation:
    :func:`bbl_transports` and :func:`apply_bbl_adv_tendency` remain the only
    transport and three-leg exchange arithmetic.  ``gdept_0`` may be a 1-D
    reference ladder or a T-cell field; ``e3u_0`` and ``e3v_0`` are the exact
    interior-face reference arrays, shaped ``(ny,nx-1,nlev)`` and
    ``(ny-1,nx,nlev)``.
    """
    h = jnp.asarray(h_ref)
    mask = jnp.asarray(land_mask, dtype=h.dtype)
    if h.ndim != 3 or mask.shape != h.shape[:2]:
        raise ValueError("h_ref must be (ny,nx,nlev) and land_mask (ny,nx)")
    ny, nx, nlev = h.shape
    gu = jnp.asarray(e3u_0, dtype=h.dtype)
    gv = jnp.asarray(e3v_0, dtype=h.dtype)
    if gu.shape != (ny, nx - 1, nlev):
        raise ValueError(
            f"e3u_0 must be {(ny, nx - 1, nlev)}, got {gu.shape}")
    if gv.shape != (ny - 1, nx, nlev):
        raise ValueError(
            f"e3v_0 must be {(ny - 1, nx, nlev)}, got {gv.shape}")
    depth = jnp.asarray(gdept_0, dtype=h.dtype)
    if depth.ndim == 1:
        if depth.shape != (nlev,):
            raise ValueError(f"1-D gdept_0 must have {nlev} levels")
        depth = jnp.broadcast_to(depth, h.shape)
    elif depth.shape != h.shape:
        raise ValueError(f"gdept_0 must be {(nlev,)} or {h.shape}, got {depth.shape}")

    wet3 = h > jnp.asarray(1.0e-3, dtype=h.dtype)  # coeff-ok: wet-cell thickness floor [m]
    n_active = jnp.sum(wet3.astype(jnp.int32), axis=-1)
    bot_k = jnp.maximum(n_active - 1, 0)
    dep_bot = jnp.take_along_axis(depth, bot_k[..., None], axis=-1)[..., 0]

    bot_l, bot_r = bot_k[:, :-1], bot_k[:, 1:]
    dep_l, dep_r = dep_bot[:, :-1], dep_bot[:, 1:]
    mgrhu = jnp.sign(dep_r - dep_l)
    ku_s = jnp.where(mgrhu >= 0, bot_l, bot_r)
    ku_d = jnp.maximum(bot_l, bot_r)
    e3u_l = jnp.take_along_axis(gu, bot_l[..., None], axis=-1)[..., 0]
    e3u_r = jnp.take_along_axis(gu, bot_r[..., None], axis=-1)[..., 0]
    e3u_bbl = jnp.minimum(e3u_l, e3u_r)
    u_active = (
        (mask[:, :-1] > 0.5)
        & (mask[:, 1:] > 0.5)
        & (mgrhu != 0)
    ).astype(h.dtype)

    bot_s, bot_n = bot_k[:-1, :], bot_k[1:, :]
    dep_s, dep_n = dep_bot[:-1, :], dep_bot[1:, :]
    mgrhv = jnp.sign(dep_n - dep_s)
    kv_s = jnp.where(mgrhv >= 0, bot_s, bot_n)
    kv_d = jnp.maximum(bot_s, bot_n)
    e3v_s = jnp.take_along_axis(gv, bot_s[..., None], axis=-1)[..., 0]
    e3v_n = jnp.take_along_axis(gv, bot_n[..., None], axis=-1)[..., 0]
    e3v_bbl = jnp.minimum(e3v_s, e3v_n)
    v_active = (
        (mask[:-1, :] > 0.5)
        & (mask[1:, :] > 0.5)
        & (mgrhv != 0)
    ).astype(h.dtype)

    return BBLGeometry(
        mgrhu=mgrhu,
        mgrhv=mgrhv,
        ku_s=ku_s.astype(jnp.int32),
        ku_d=ku_d.astype(jnp.int32),
        kv_s=kv_s.astype(jnp.int32),
        kv_d=kv_d.astype(jnp.int32),
        e3u_bbl=e3u_bbl,
        e3v_bbl=e3v_bbl,
        dep_bot=dep_bot,
        u_active=u_active,
        v_active=v_active,
        bot_k=bot_k.astype(jnp.int32),
        h_ref=h,
    )


def _bottom_ts(T, S, bot_k):
    """Gather bottom-cell T, S per column."""
    Tb = jnp.take_along_axis(T, bot_k[..., None], axis=-1)[..., 0]
    Sb = jnp.take_along_axis(S, bot_k[..., None], axis=-1)[..., 0]
    return Tb, Sb


def bbl_transports(T: jnp.ndarray, S: jnp.ndarray, geom: BBLGeometry,
                   dy_u: jnp.ndarray, dx_v: jnp.ndarray, *,
                   gamma_s: float, rho_0: float,
                   bottom_depth_m: jnp.ndarray | None = None):
    """Campin-Goosse down-slope transports per face [m^3/s].

        tr = facewidth * e3_bbl * (g*gamma) * max(0, zgdrho) * mgrh

    with the EXACT NEMO trabbl gating (eos_rab form): alpha/beta evaluated
    per column at ITS OWN bottom pressure (thermobaricity preserved — the
    canonical Wright-EOS derivatives, ``thermal_expansion_coeff`` /
    ``haline_contraction_coeff``), AVERAGED across the face, then

        zgdrho = max(0, abar*(T_deep - T_shelf) - bbar*(S_deep - S_shelf))

    which is the linearized (rho_shelf - rho_deep)/rho0 — positive only when
    the shelf bottom cell is denser.  A naive direct-density difference at
    the face-mean pressure misses the compressibility asymmetry across
    steep shelf-to-deep faces (codex HIGH) — exactly the overflow faces
    this scheme exists for.
    """
    from legoesm.ocean.eos import nemo_roquet_alpha_beta
    g_gamma = constants.g * gamma_s

    Tb, Sb = _bottom_ts(T, S, geom.bot_k)
    depth = geom.dep_bot if bottom_depth_m is None else bottom_depth_m
    # NEMO option 2 calls eos_rab on Kbb bottom T/S at each column's Kmm
    # geometric depth, then averages alpha/beta across the face before the
    # density gate (trabbl.F90:342-353,415-454).  The Roquet helper is the
    # literal eosbn2 polynomial, including NEMO's rho0=1026 normalization.
    alpha, beta = nemo_roquet_alpha_beta(Tb, Sb, depth, rho0=rho_0)

    def _face_tr(axis):
        if axis == 0:   # j-faces
            sl1 = (slice(0, -1), slice(None)); sl2 = (slice(1, None), slice(None))
            mgrh, e3, act, width = (geom.mgrhv, geom.e3v_bbl,
                                    geom.v_active, dx_v)
        else:           # i-faces
            sl1 = (slice(None), slice(0, -1)); sl2 = (slice(None), slice(1, None))
            mgrh, e3, act, width = (geom.mgrhu, geom.e3u_bbl,
                                    geom.u_active, dy_u)
        T1, T2, S1, S2 = Tb[sl1], Tb[sl2], Sb[sl1], Sb[sl2]
        a_bar = 0.5 * (alpha[sl1] + alpha[sl2])
        b_bar = 0.5 * (beta[sl1] + beta[sl2])
        # shelf = up-slope column, deep = down-slope column (by mgrh)
        T_sh = jnp.where(mgrh >= 0, T1, T2)
        T_dp = jnp.where(mgrh >= 0, T2, T1)
        S_sh = jnp.where(mgrh >= 0, S1, S2)
        S_dp = jnp.where(mgrh >= 0, S2, S1)
        zgdrho = jnp.maximum(
            a_bar * (T_dp - T_sh) - b_bar * (S_dp - S_sh), 0.0)
        return width * e3 * g_gamma * zgdrho * mgrh * act

    return _face_tr(1), _face_tr(0)


def apply_bbl_adv_tendency(dT_dt, dS_dt, T, S, h_k, area, geom: BBLGeometry,
                           utr, vtr, *, nlev: int):
    """Add the NEMO ``tra_bbl_adv`` 3-leg exchange to the tracer tendencies.

    ``nlev`` must be a static Python int (the per-level loop is unrolled).
    All scatter updates use ``.at[].add`` — pure JAX, jit/grad safe.
    """
    a3 = area[..., None] * jnp.maximum(h_k, 1.0e-3)   # cell volumes — coeff-ok: thickness floor [m]
    inv_v = 1.0 / a3

    def _apply(dpt, pt, tr, axis):
        if axis == 1:   # i-faces between (j,i) and (j,i+1)
            mgrh, ks, kd, act = geom.mgrhu, geom.ku_s, geom.ku_d, geom.u_active
            sl_L = (slice(None), slice(0, -1))
            sl_R = (slice(None), slice(1, None))
        else:           # j-faces
            mgrh, ks, kd, act = geom.mgrhv, geom.kv_s, geom.kv_d, geom.v_active
            sl_L = (slice(0, -1), slice(None))
            sl_R = (slice(1, None), slice(None))
        zu = jnp.abs(tr) * act
        up = mgrh >= 0   # True: LEFT column is shelf, RIGHT is deep

        # gathered shelf/deep columns (face-shaped, full nlev)
        pt_L, pt_R = pt[sl_L], pt[sl_R]
        iv_L, iv_R = inv_v[sl_L], inv_v[sl_R]
        pt_sh = jnp.where(up[..., None], pt_L, pt_R)
        pt_dp = jnp.where(up[..., None], pt_R, pt_L)
        iv_sh = jnp.where(up[..., None], iv_L, iv_R)
        iv_dp = jnp.where(up[..., None], iv_R, iv_L)

        ks_ = ks.astype(jnp.int32)
        kd_ = kd.astype(jnp.int32)
        pt_sh_b = jnp.take_along_axis(pt_sh, ks_[..., None], axis=-1)[..., 0]
        pt_dp_at_ks = jnp.take_along_axis(pt_dp, ks_[..., None], axis=-1)[..., 0]
        pt_dp_b = jnp.take_along_axis(pt_dp, kd_[..., None], axis=-1)[..., 0]

        # face-shaped tendency contributions for shelf + deep columns
        d_sh = jnp.zeros_like(pt_sh)
        d_dp = jnp.zeros_like(pt_dp)
        # (1) shelf bottom: exchange with deep column at shelf level
        ztra_sh = zu * (pt_dp_at_ks - pt_sh_b) * jnp.take_along_axis(
            iv_sh, ks_[..., None], axis=-1)[..., 0]
        d_sh = d_sh.at[
            jnp.arange(d_sh.shape[0])[:, None], jnp.arange(d_sh.shape[1])[None, :],
            ks_].add(ztra_sh)
        # (2) deep interior return flow ks <= k < kd  (static unrolled loop)
        for k in range(nlev - 1):
            in_rng = ((k >= ks_) & (k < kd_)).astype(pt.dtype)
            ztra = zu * in_rng * (pt_dp[..., k + 1] - pt_dp[..., k]) \
                * iv_dp[..., k]
            d_dp = d_dp.at[..., k].add(ztra)
        # (3) deep bottom: receives the shelf bottom water
        ztra_dp = zu * (pt_sh_b - pt_dp_b) * jnp.take_along_axis(
            iv_dp, kd_[..., None], axis=-1)[..., 0]
        d_dp = d_dp.at[
            jnp.arange(d_dp.shape[0])[:, None], jnp.arange(d_dp.shape[1])[None, :],
            kd_].add(ztra_dp)

        # un-gather face contributions back onto LEFT/RIGHT cell columns
        d_L = jnp.where(up[..., None], d_sh, d_dp)
        d_R = jnp.where(up[..., None], d_dp, d_sh)
        dpt = dpt.at[sl_L].add(d_L.astype(dpt.dtype))
        dpt = dpt.at[sl_R].add(d_R.astype(dpt.dtype))
        return dpt

    for axis in (1, 0):
        dT_dt = _apply(dT_dt, T, utr if axis == 1 else vtr, axis)
        dS_dt = _apply(dS_dt, S, utr if axis == 1 else vtr, axis)
    return dT_dt, dS_dt


__all__ = [
    "BBLDiffusiveGeometry",
    "BBLGeometry",
    "apply_bbl_diffusive_tendency",
    "apply_bbl_adv_step",
    "apply_bbl_adv_tendency",
    "bbl_static_geometry",
    "nemo_bbl_diffusive_coefficients",
    "nemo_bbl_diffusive_geometry",
    "nemo_bbl_static_geometry",
    "bbl_transports",
]


def apply_bbl_adv_step(state, geom: BBLGeometry, dt: float, *,
                       gamma_s: float, rho_0: float,
                       area_2d, dy_u_faces, dx_v_faces, nlev: int):
    """HOST post-step BBL application (the faithful-runner pattern, like the
    SSS restoring / ice-thermo nudges): recompute the Campin-Goosse
    transports from the CURRENT bottom T/S and integrate one forward-Euler
    exchange step ``pt += dt * d(pt)/dt``.

    COMPOSITION, stated honestly (branch-isomorphism audit S-42): this is NOT
    NEMO's composition.  NEMO adds ``tra_bbl``'s exchange into the tracer
    right-hand side of the step itself (``stprk3_stg.F90:468,498,588`` on the
    RK3 lane, ``stpmlf.F90`` on the MLF lane), reading the BEFORE-level
    tracers; this wrapper applies the SAME operator as a separate forward-
    Euler update after the step, on the updated tracers and the reference
    thicknesses.  The arithmetic is shared: ``bbl_transports`` and
    ``apply_bbl_adv_tendency`` below are the single transcription of
    ``trabbl.F90:243-284``, and this function adds no arithmetic of its own
    beyond one forward-Euler update.  It does feed them DIFFERENT operands
    though: the reference ladder ``geom.h_ref`` / ``geom.dep_bot``, where the
    in-model site passes the live stage thickness and a recomputed live bottom
    depth (``ocean_model_latlon_cgrid.py:1290-1298``) -- an O(eta/H) ~ 3e-4
    difference.  So the placement is a host operator split with no NEMO arm,
    and its operands are the reference ones.  It exists
    because the only in-model BBL site lives in the WS-RK3 tracer lane
    (``ocean_model_latlon_cgrid.py``, ``tracer_time_integrator="rk3_ws"``)
    and the OMIP driver runs the forward-Euler tracer lane, which itself has
    no NEMO arm.  Collapsing the two onto one site is an OPEN item.

    Stability -- the criterion, not one example.  ``e3_bbl`` cancels between
    transport and volume (``tr`` ~ ``width*e3_bbl``, ``V`` ~ ``area*e3_bot``,
    and ``e3_bbl`` is the min of the two bottom thicknesses), so the per-step
    exchange fraction reduces to

        |tr|*dt/V  =  g * gamma_s * (drho/rho_0) * dt / e1t

    which is thickness-INDEPENDENT.  The deleted ``0.25`` cap could therefore
    bind only for ``drho/rho_0 > e1t / (4*g*gamma_s*dt)``.  MEASURED on ORCA1's
    own metrics at 60N, dt=3600 s, with a Denmark-Strait-exceeding contrast
    (dT=13.5 degC, dS=2.5 PSU): max fraction 0.048, transports up to 4.1 Sv; at
    an absurd drho/rho_0 = 1e-2 with dt=5400 s it is 0.19.  The cap never bound
    in any production configuration.  NEMO clamps neither ``utr_bbl`` nor
    ``vtr_bbl`` (``trabbl.F90:243-284``), so neither does this.

    Parameters
    ----------
    state : LatLonCGridOceanState-like (T, S Fields with (..., nlev) data)
    geom : BBLGeometry from :func:`bbl_static_geometry` (STATIC, build once)
    dy_u_faces : (n_lat, n_lon-1) i-face widths [m] (NEMO e2u at the face)
    dx_v_faces : (n_lat-1, n_lon) j-face widths [m] (NEMO e1v)
    nlev : static Python int.

    Returns the state with T, S updated.
    """
    T = jnp.asarray(state.T.data, dtype=jnp.float64)
    S = jnp.asarray(state.S.data, dtype=jnp.float64)
    # actual thicknesses for the volume weights: reuse the static reference
    # h embedded in the geometry via the caller (h_k passed implicitly when
    # the geometry was built from h_partial; eta-induced J deviation is
    # O(eta/H) and irrelevant for an exchange tendency).
    h_k = geom.h_ref
    utr, vtr = bbl_transports(
        T, S, geom, dy_u_faces, dx_v_faces,
        gamma_s=gamma_s, rho_0=rho_0)
    # No transport cap.  NEMO's tra_bbl_adv (trabbl.F90:243-284) clamps
    # neither utr_bbl nor vtr_bbl, and no non-NEMO recipe selects a capped
    # arm (recipes.py never mentions BBL), so a cap here would be a
    # stabilizer the oracle lacks — removed 2026-09-02 under the
    # branch-isomorphism audit (S-42).
    zero = jnp.zeros_like(T)
    dT, dS = apply_bbl_adv_tendency(
        zero, jnp.zeros_like(S), T, S, h_k, jnp.asarray(area_2d), geom,
        utr, vtr, nlev=nlev)
    T_new = (T + dt * dT).astype(state.T.data.dtype)
    S_new = (S + dt * dS).astype(state.S.data.dtype)
    return state._replace(
        T=state.T.replace(data=T_new),
        S=state.S.replace(data=S_new),
    )
