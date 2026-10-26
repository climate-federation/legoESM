"""DCMIP16_BC initial condition, assembled on one cubed-sphere face.

Turns the pointwise profiles in ``fv3_native_dcmip16_bc`` into the D-grid
state the acoustic units consume. ORACLE: ``tools/test_cases.F90:6528-6712``.

WHAT THIS COVERS: ps, delp, the cell-centre Newton height march, pt from
hydrostatic balance, and the D-grid u/v. It does NOT cover pe/peln/pk/pkz
(recomputed by ``p_var``), the tracers, or the halo fills -- those are the
caller's, and the oracle does them outside this span too.

THE THREE THINGS THAT ARE EASY TO GET WRONG HERE, all load-bearing:

1. ``pt`` IS NOT THE ANALYTIC TEMPERATURE AT THE LAYER MIDPOINT.
   ``:6602-6608`` back-derives it from the Newton-solved interface heights:
   ``pt = (g/Rd)*(z_top - z_bot)/(ln p_bot - ln p_top)``. Evaluating
   ``DCMIP16_BC_temperature`` at the midpoint instead gives a field that looks
   entirely reasonable and is wrong everywhere.

2. ``ps_u``/``ps_v`` ARE SET TO p0 AND NEVER UPDATED (``:6625``, ``:6673``),
   so the wind-point interface pressure at ``:6639``/``:6687`` is always
   ``ak(k) + p0*bk(k)`` -- it is NOT read from the ``pe`` built for the cell
   centres, and it is NOT special-cased at k=1. Reproduced literally.

3. BOTH u AND v PROJECT ONTO ``ex`` = e_lon (``:6631``, ``:6679``). The
   analytic wind is purely zonal; ``v`` is that same zonal wind projected onto
   the +j edge tangent, which is non-zero wherever a v-edge is not due north.
   ``u2``/``v2`` (the e_lat projections) are COMPUTED AND NEVER USED upstream,
   so they are not computed here.

The Newton march runs k = npz down to 1 with each level seeded by the
interface below, and the running state (gz, peln) carries between levels --
so the k loop is sequential by construction and cannot be vectorised over k.
"""
from __future__ import annotations

import numpy as np

from legoesm.core.fv3_native_dcmip16_bc import (
    P0_PA,
    ZCONV_M,
    DCMIP16Constants,
    pressure,
    temperature,
    uwind,
    uwind_pert,
)
from legoesm.grids.fv3_native_metrics import (
    get_latlon_vector,
    get_unit_vect2,
    inner_prod,
    mid_pt_sphere,
)

_N_NEWTON = 30          # test_cases.F90:6584, :6644, :6692


def _newton_march(p_target, lat, z_start, c: DCMIP16Constants):
    """One interface of the height march -- :6644-6650 / :6584-6590.

    Scalar-per-point loop, matching the Fortran's per-point ``exit``. The
    vectorised variant in ``fv3_native_dcmip16_bc.newton_height`` keeps
    iterating converged points; here the control flow is the oracle's.
    """
    rdgrav = c.rdgas / c.grav                    # :6506
    z = float(z_start)
    for _ in range(_N_NEWTON):
        ziter = z
        piter = float(pressure(ziter, lat, c))
        titer = float(temperature(ziter, lat, c))
        z = ziter + (piter - p_target) * rdgrav * titer / piter
        if abs(z - ziter) < ZCONV_M:
            break
    return z


def _edge_winds(corner_lonlat, pair_shift, shape, ak, bk, km,
                c: DCMIP16Constants, do_pert, great_circle_dist):
    """The u- and v-point loops, which are identical except for the stencil.

    ``pair_shift`` selects which corner pair spans the edge: ``(1, 0)`` gives
    the u-points (edge from grid(i,j) to grid(i+1,j), :6626) and ``(0, 1)``
    the v-points (grid(i,j) to grid(i,j+1), :6674).

    Factored rather than written twice because the two Fortran loops are
    character-identical apart from that stencil and the array extents -- and
    a copy-paste pair is exactly where an index or sign drifts unnoticed.
    """
    di, dj = pair_shift
    ni, nj = shape
    out = np.zeros((ni, nj, km), dtype=np.float64)

    lon = corner_lonlat[..., 0]
    lat = corner_lonlat[..., 1]
    rrdgrav = c.grav / c.rdgas                   # :6509

    for j in range(nj):
        for i in range(ni):
            p1 = np.array([lon[i, j], lat[i, j]])
            p2 = np.array([lon[i + di, j + dj], lat[i + di, j + dj]])

            pa = mid_pt_sphere(p1, p2)           # :6626 / :6674
            lat_e, lon_e = pa[1], pa[0]
            e = get_unit_vect2(p1, p2)           # :6629 / :6677
            ex, _ey = get_latlon_vector(pa)      # :6630 / :6678
            proj = float(inner_prod(e, ex))      # :6631 / :6679 -- e_lon ONLY

            gz = 0.0                             # :6622 / :6670
            peln = np.log(P0_PA)                 # :6624 / :6672
            for k in range(km - 1, -1, -1):      # k = npz down to 1
                # :6639 / :6687 -- ps_e is pinned at p0, never updated
                p = ak[k] + P0_PA * bk[k]
                pl = np.log(p)
                z0 = gz
                z = _newton_march(p, lat_e, gz, c)
                # :6652 / :6700 -- layer-mean virtual T from hydro balance
                pt_e = rrdgrav * (z - gz) / (peln - pl)
                uu = float(uwind(0.5 * (z + z0), pt_e, lat_e, c))
                if do_pert:                      # :6656 / :6703
                    uu += float(uwind_pert(0.5 * (z + z0), lat_e, lon_e,
                                           c, great_circle_dist))
                out[i, j, k] = proj * uu         # :6659 / :6706
                gz, peln = z, pl
    return out


def dcmip16_bc_face(grid_corner_lonlat, agrid_lonlat, ak, bk, km, *,
                    constants: DCMIP16Constants, do_pert: bool,
                    great_circle_dist) -> dict:
    """Build ``ps``, ``delp``, ``pt``, ``u``, ``v`` for ONE face.

    ``grid_corner_lonlat`` is ``(ni+1, nj+1, 2)`` cell CORNERS (radians,
    lon then lat) and ``agrid_lonlat`` is ``(ni, nj, 2)`` cell CENTRES, over
    whatever window the caller wants filled -- the oracle writes the compute
    domain only (``is:ie``) and leaves halos to later exchanges, so pass the
    compute window and fill halos separately.

    ``do_pert=True`` is test_case = -13; ``False`` is -12.
    """
    c = constants
    ak = np.asarray(ak, dtype=np.float64)
    bk = np.asarray(bk, dtype=np.float64)
    if ak.shape != (km + 1,) or bk.shape != (km + 1,):
        raise ValueError(
            f"ak/bk must have km+1={km + 1} entries, got {ak.shape}/{bk.shape}")

    agrid = np.asarray(agrid_lonlat, dtype=np.float64)
    ni, nj = agrid.shape[0], agrid.shape[1]
    corners = np.asarray(grid_corner_lonlat, dtype=np.float64)
    if corners.shape[:2] != (ni + 1, nj + 1):
        raise ValueError(
            f"corner array {corners.shape[:2]} must be one larger in each "
            f"direction than the centre array {(ni, nj)} -- a mismatch here "
            f"silently shifts every wind by half a cell")

    # --- ps, delp: :6528-6541 ------------------------------------------
    ps = np.full((ni, nj), P0_PA, dtype=np.float64)
    delp = np.empty((ni, nj, km), dtype=np.float64)
    for k in range(km):
        delp[:, :, k] = (ak[k + 1] - ak[k]) + ps * (bk[k + 1] - bk[k])

    # --- cell-centre Newton march + pt from hydrostatic balance --------
    # :6579-6608. gz(npz+1) = 0 at the surface (:6577) and the march runs
    # upward, each interface seeded by the one below.
    lat_c = agrid[..., 1]
    gz = np.zeros((ni, nj, km + 1), dtype=np.float64)
    pt = np.empty((ni, nj, km), dtype=np.float64)
    rrdgrav = c.grav / c.rdgas
    for j in range(nj):
        for i in range(ni):
            lat_ij = lat_c[i, j]
            peln_below = np.log(P0_PA)
            for k in range(km - 1, -1, -1):
                p = ak[k] + P0_PA * bk[k]
                pl = np.log(p)
                gz[i, j, k] = _newton_march(p, lat_ij, gz[i, j, k + 1], c)
                pt[i, j, k] = (rrdgrav * (gz[i, j, k] - gz[i, j, k + 1])
                               / (peln_below - pl))
                peln_below = pl

    # --- D-grid winds: :6620-6712 --------------------------------------
    u = _edge_winds(corners, (1, 0), (ni, nj + 1), ak, bk, km, c, do_pert,
                    great_circle_dist)
    v = _edge_winds(corners, (0, 1), (ni + 1, nj), ak, bk, km, c, do_pert,
                    great_circle_dist)

    return {"ps": ps, "delp": delp, "pt": pt, "gz": gz, "u": u, "v": v}


# --- DCMIP16_BC moisture parameters (test_cases.F90:6497-6502) ---------
Q0_BC = 0.018            # q0   -- surface specific-humidity amplitude
QT_BC = 1.0e-12          # qt   -- stratospheric background
PTROP_BC_PA = 1.0e4      # ptrop
PW_BC_PA = 34000.0       # pW   -- vertical moisture decay pressure
PHIW_BC_RAD = 2.0 * np.pi / 9.0   # phiW -- meridional moisture width


# --- DCMIP16 terminator toy-chemistry IC (test_cases.F90:4136-4205) ---
# Filled by the oracle's cold start whenever the field_table lists
# tracers named cl and cl2 (:6746-6750).  The REACTION lives in fv_phys
# (:4155 "you will have to change it both here and in fv_phys") and never
# runs on an adiabatic deck, so on it cl/cl2 are passive, longitude-
# dependent, nonzero passengers with Cl + 2 Cl2 = qcly exactly.
TERM_QCLY = 4.0e-6            # :4157
TERM_LC = 5.0 * np.pi / 3.0   # :4158
TERM_THC = np.pi / 9.0        # :4159
TERM_K2 = 1.0                 # :4160


def dcmip16_terminator_cl_cl2(agrid_lon, agrid_lat):
    """``(cl, cl2)`` on cell centres, one level -- the oracle copies
    level 1 to every level (:4174-4180).  Radians in, mixing ratio out."""
    lon = np.asarray(agrid_lon, dtype=np.float64)
    lat = np.asarray(agrid_lat, dtype=np.float64)
    k1 = np.maximum(0.0, np.sin(lat) * np.sin(TERM_THC)
                    + np.cos(lat) * np.cos(TERM_THC) * np.cos(lon - TERM_LC))
    r = k1 / TERM_K2 * 0.25
    d = np.sqrt(r * r + 2.0 * r * TERM_QCLY)
    cl = d - r
    cl2 = 0.5 * (TERM_QCLY - cl)
    return cl, cl2


def dcmip16_terminator_six_face(ctx: dict, km: int) -> list:
    """Six ``[cl, cl2]`` padded ``(m_a, m_a, km)`` arrays, compute window
    filled, halos zero (the caller's convention for every tracer)."""
    from legoesm.core.fv3_native_state_3d import field_shape
    n, ng = int(ctx["n"]), int(ctx["ng"])
    cs = slice(ng, ng + n)
    out = []
    for t in range(6):
        gs = ctx["gs6"][t]
        cl, cl2 = dcmip16_terminator_cl_cl2(
            np.asarray(gs["agrid_lon"])[cs, cs],
            np.asarray(gs["agrid_lat"])[cs, cs])
        pair = []
        for f in (cl, cl2):
            q = np.zeros(field_shape("delp", n, ng, km), dtype=np.float64)
            q[cs, cs, :] = f[:, :, None]
            pair.append(q)
        out.append(pair)
    return out


def dcmip16_bc_sphum(ak, bk, agrid_lat, km: int) -> np.ndarray:
    """``sphum`` on the DCMIP16_BC column -- test_cases.F90:6737-6744.

    The oracle computes the layer pressure from its own hydrostatic
    column (``ps == p0`` exactly on this IC)::

        pe(1) = ptop = ak(1);       peln(1) = log(ptop)
        pe(k) = ak(k) + p0*bk(k);   peln(k) = log(pe(k))     (:6540-6560)
        p     = delp(k) / (peln(k+1) - peln(k))              (:6740)

    and then ``DCMIP16_BC_sphum(p, ps, lat, lon)`` (:6840-6852)::

        eta = p/ps
        sphum = qt                                    if p <= ptrop
              = q0 * exp(-(lat/phiW)**4)
                   * exp(-(((eta-1)*p0/pW)**2))       otherwise

    ``lon`` is dead in the Fortran function body; it is not taken here.
    The oracle fills the COMPUTE window only and zeroes the halo
    (:6728-6735; the ``mpp_update_domains(q)`` at :6749 is inside the
    terminator-tracer branch, absent on this deck with no cl/cl2), so
    the caller pads with zeros.

    Returns ``(ni, nj, km)`` over whatever centre-latitude window
    ``agrid_lat`` covers (radians).
    """
    ak = np.asarray(ak, dtype=np.float64)
    bk = np.asarray(bk, dtype=np.float64)
    lat = np.asarray(agrid_lat, dtype=np.float64)
    if ak.shape != (km + 1,) or bk.shape != (km + 1,):
        raise ValueError(
            f"ak/bk must have km+1={km + 1} entries, got "
            f"{ak.shape}/{bk.shape}")

    # ps == p0 uniformly, so pe/peln/delp/p are scalars per level --
    # same arithmetic as the Fortran, just not repeated per column.
    pe = np.empty(km + 1, dtype=np.float64)
    pe[0] = ak[0]                       # pe(i,1,j) = ptop  (:6540)
    for k in range(1, km + 1):
        pe[k] = ak[k] + P0_PA * bk[k]
    peln = np.log(pe)
    lat_term = np.exp(-((lat / PHIW_BC_RAD) ** 4))

    out = np.empty(lat.shape + (km,), dtype=np.float64)
    for k in range(km):
        delp_k = (ak[k + 1] - ak[k]) + P0_PA * (bk[k + 1] - bk[k])
        p = delp_k / (peln[k + 1] - peln[k])
        if p > PTROP_BC_PA:
            eta = p / P0_PA
            out[..., k] = (Q0_BC * lat_term
                           * np.exp(-(((eta - 1.0) * P0_PA / PW_BC_PA)
                                      ** 2)))
        else:
            out[..., k] = QT_BC
    return out


def dcmip16_bc_six_face_state(ctx: dict, ak, bk, km: int, *,
                              hydrostatic: bool = True,
                              do_pert: bool = True,
                              constants: DCMIP16Constants | None = None,
                              with_sphum: bool = True,
                              zvir: float = 0.0):
    """The test_case = -13/-12 IC on ALL SIX duo faces, state_3d layout.

    Committed home of the assembly that
    ``scripts/validate/fv3_native/full_step_oracle_parity.py::build_port_ic``
    (and ``build_port_tracer_ic``) established against the oracle: window
    IC only (``is:ie``), halos left at zero exactly as the oracle leaves
    them to the in-step exchanges; NH adds ``make_nh``'s state
    (``init_hydro.F90:147-158``): ``w = 0`` and
    ``delz = -(rdgas/grav) * pt * dpeln`` from the IC's own hydrostatic
    column (``zvir = 0`` on the adiabatic deck).

    ``zvir != 0`` selects the MOIST IC.  ``test_cases.F90:6760-6768``
    runs ``pt = pt/(1. + zvir*q(sphum))`` -- labelled "Convert pt to
    non-virtual temperature" -- whenever the deck is NOT adiabatic, and
    in the solo driver ``adiabatic = .false.`` is exactly what sets
    ``zvir = rvgas/rdgas - 1`` (``atmosphere.F90:156-161``), so the one
    flag decides both.  A moist deck's restart therefore holds DRY
    temperature: omitting this leaves the IC ~3.8 K warm where
    ``zvir*q`` peaks at 0.0128, which is the 1.06e-02 relative the
    parity harness's instrument control refused on 2026-08-20.

    THE ORDER IS THE ORACLE'S, and it matters on the NH arm: ``delz``
    is built at ``:6721`` from the VIRTUAL ``pt``, ``sphum`` is filled
    at ``:6737``, and only then is ``pt`` divided at ``:6760``.  The
    division is therefore LAST in the per-face body, after ``delz``.
    ``zvir != 0`` requires ``with_sphum`` for the obvious reason.

    ``ctx`` is a ``build_six_face_duo_context`` dict (the NumPy lane's);
    ``constants=None`` selects ``GFS_CONSTANTS`` — the FMS set the pinned
    oracle binary links (parity runner asserts it from the run's own
    log), NOT ``legoesm.constants``; the two differ at ~1e-4 and mixing
    them was the IC-parity confound of 2026-08-07.

    Returns ``(state, sphum)``: ``state`` is ``build_state_3d``'s list of
    six per-face dicts and ``sphum`` a list of six padded
    ``(m_a, m_a, km)`` tracer arrays (zero halos, per :6728-6735), or
    ``None`` when ``with_sphum=False``.
    """
    from legoesm.core.fv3_native_dcmip16_bc import GFS_CONSTANTS
    from legoesm.core.fv3_native_state_3d import build_state_3d, field_shape
    from legoesm.grids.fv3_native_gridstruct import FV3_GRAV, FV3_RDGAS
    from legoesm.grids.fv3_native_metrics import great_circle_dist as _gcd

    if constants is None:
        constants = GFS_CONSTANTS
    if zvir != 0.0 and not with_sphum:
        raise ValueError(
            "zvir != 0 needs with_sphum=True: the moist IC divides pt by "
            "(1 + zvir*q(sphum)) (test_cases.F90:6762) and there is no "
            "sphum to divide by.")

    def gcdr(p1, p2, r):
        # the oracle's 3-arg form: unit-sphere angle times radius
        return _gcd(np.asarray(p1, float), np.asarray(p2, float)) * r

    n, ng = int(ctx["n"]), int(ctx["ng"])
    ak = np.asarray(ak, dtype=np.float64)
    bk = np.asarray(bk, dtype=np.float64)
    st = build_state_3d(n, ng, km, remap_follows=True,
                        hydrostatic=hydrostatic)
    sphum = [] if with_sphum else None
    cs, cc = slice(ng, ng + n), slice(ng, ng + n + 1)
    for t in range(6):
        gs = ctx["gs6"][t]
        co = np.stack([np.asarray(gs["grid_lon"])[cc, cc],
                       np.asarray(gs["grid_lat"])[cc, cc]], -1)
        ce = np.stack([np.asarray(gs["agrid_lon"])[cs, cs],
                       np.asarray(gs["agrid_lat"])[cs, cs]], -1)
        o = dcmip16_bc_face(co, ce, ak, bk, km, do_pert=do_pert,
                            constants=constants, great_circle_dist=gcdr)
        st[t]["delp"][cs, cs, :] = o["delp"]
        st[t]["pt"][cs, cs, :] = o["pt"]
        st[t]["u"][cs, cc, :] = o["u"]
        st[t]["v"][cc, cs, :] = o["v"]
        if not hydrostatic:
            # make_nh (init_hydro.F90:147-158): w = 0 (build_state_3d's
            # fill) and delz from the IC's own hydrostatic column
            pe = np.full((n, n), float(ak[0]))
            for k in range(km):
                dp = st[t]["delp"][cs, cs, k]
                dpeln = np.log(pe + dp) - np.log(pe)
                st[t]["delz"][:, :, k] = (-(FV3_RDGAS / FV3_GRAV)
                                          * st[t]["pt"][cs, cs, k] * dpeln)
                pe = pe + dp
        if with_sphum:
            q = np.zeros(field_shape("delp", n, ng, km), dtype=np.float64)
            q[cs, cs, :] = dcmip16_bc_sphum(
                ak, bk, np.asarray(gs["agrid_lat"])[cs, cs], km)
            sphum.append(q)
            if zvir != 0.0:
                # :6760-6768, LAST -- after delz above, which the oracle
                # builds from the still-virtual pt at :6721.
                st[t]["pt"][cs, cs, :] = (
                    st[t]["pt"][cs, cs, :]
                    / (1.0 + zvir * q[cs, cs, :]))
    return st, sphum
