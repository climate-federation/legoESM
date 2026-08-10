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
