"""Discretely consistent prescribed air-mass fluxes for flux-form tracer
transport with STEADY prescribed winds and a steady prescribed air mass
(DCMIP 2012 test 1-3).

DCMIP 2012 (Ullrich et al. 2012, test-case document v1.7, Sec. 1 eqs. 6-9)
prescribes the density and holds it constant (``d rho / dt = 0``), the
3-D mass flux ``rho v`` being analytically divergence free.  On a grid the
discrete horizontal mass-flux divergence and the analytic vertical
(coordinate-crossing, "perceived", Appendix B) mass flux do not cancel, so
a flux-form tracer on the steady air mass neither keeps a uniform tracer
uniform nor its layer air mass steady.  Here the analytic vertical mass
flux is KEPT (the terrain crossing of 1-3 lives in it) and the horizontal
mass flux receives the minimal correction that closes the discrete
continuity layer by layer::

    F_k = F0_k + grad(chi_k),   div(F_k) + (W_{k+1/2} - W_{k-1/2}) = 0,

i.e. ``L chi_k = -(div F0_k + delta W_k)`` with ``L = div grad`` built from
the core's OWN discrete divergence and the matching gradient (the
Helmholtz / least-squares projection; ``grad chi`` is the smallest
correction in the face-volume-weighted norm).  ``W`` is the caller's
analytic ``p_s sigma_dot`` [Pa/s, positive toward larger sigma] at the
nlev+1 interfaces.  Each interface's area-weighted mean is first shifted
to a common target: the discrete global integral of ``delta W`` must
vanish for ``L chi = r`` to be solvable (analytically it does; the shift
is discretisation-size and is reported).  The target is zero when the top
or the surface is closed (``W`` identically zero there), so a closed
boundary stays closed, else the top interface's mean (a uniform
throughflow is kept); elsewhere the shift applies at every column, including columns where the
caller's ``W`` was zero.  The matrix DCMIP 1-3 rows pass the flow restricted
to ``h0 <= z <= z_top`` (``dcmip13_flow_fraction``) on the Kent et al.
(2014, App. B) hybrid levels, whose ``W`` vanishes at the top and the surface; a nonzero boundary ``W`` is an open boundary,
where the tracer flux takes the adjacent layer's value.

The tracer flux of the corrected air fluxes then keeps ``q = 1`` to
round-off and conserves ``sum area dp q`` to round-off up to the tracer
flux through an open top / surface (none with closed boundaries).

Each core: lat-lon C-grid (``core.operators_fv_latlon_3d`` face fluxes),
MPAS Voronoi (TRiSK ``divergence_cell_3d`` / ``gradient_edge_3d``),
spectral (transform divergence of ``spectral_tracer_transport``; there the
air mass ``p_s`` and ``W`` are the Orszag-band projections, as a spectral
model carries its surface pressure spectrally, so that the uniform tracer
and the air mass are representable in the truncated state).  The Poisson
problems are solved once (steady winds) with a sparse direct solve on the
grid-point cores (scipy, bordered for the constant null space) and
diagonally on the spectral core.  ``refine`` is the TOTAL number of solve
passes, each against the residual of the core's own operators (the first
pass is the solve, the others refine it); ``refine=0`` returns the
uncorrected fluxes.  The horizontal correction is not confined to the
caller's flow support (``grad chi`` may be nonzero where ``F0`` is zero).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


def _require_x64():
    if not jax.config.jax_enable_x64:
        raise ValueError("consistent mass fluxes close the continuity to "
                         "round-off in float64: enable JAX x64")


def _layer_dp(sigma_coord, p_s):
    """Steady layer air mass [Pa] (cells..., nlev) of surface pressure
    ``p_s`` (cells...): ``p_s dsigma`` on a sigma coordinate, ``dA p_ref +
    dB p_s`` (:func:`legoesm.grids.vertical.dp_from_hybrid`) on a hybrid
    sigma-pressure one.  Float64."""
    from legoesm.grids.vertical import HybridSigmaPressureCoordinate, dp_from_hybrid
    if isinstance(sigma_coord, HybridSigmaPressureCoordinate):
        from legoesm.grids.vertical import hybrid_min_valid_surface_pressure
        # dp = dA p_ref + dB p_s > 0 only for p_s strictly above the shared
        # threshold: refuse a collapsed/inverted layer
        p_min = hybrid_min_valid_surface_pressure(
            sigma_coord.A_half, sigma_coord.B_half, sigma_coord.p_ref)
        dp = dp_from_hybrid(sigma_coord, p_s).astype(jnp.float64)
        if not bool(jnp.all(jnp.isfinite(p_s)) and jnp.all(p_s > p_min)
                    and jnp.all(jnp.isfinite(dp)) and jnp.all(dp > 0)):
            raise ValueError(f"hybrid layer air mass dA p_ref + dB p_s must be strictly "
                             f"positive: p_s must exceed {p_min:.6g} Pa")
        return dp
    return p_s[..., None] * jnp.asarray(np.asarray(sigma_coord.dsigma, np.float64))


def _face_air_flux(sigma_coord, p_face, wind):
    """``wind`` (faces..., nlev) times the layer air mass of the face surface
    pressure ``p_face`` (faces...): sigma ``p_face wind dsigma`` (the
    original operation order), hybrid ``wind (dA p_ref + dB p_face)``."""
    from legoesm.grids.vertical import HybridSigmaPressureCoordinate
    if isinstance(sigma_coord, HybridSigmaPressureCoordinate):
        return wind * _layer_dp(sigma_coord, p_face)
    return p_face[..., None] * wind * jnp.asarray(np.asarray(sigma_coord.dsigma, np.float64))


def _require_positive(p_s):
    if not bool(jnp.all(jnp.isfinite(p_s)) and jnp.all(p_s > 0)):
        raise ValueError("the surface pressure of the steady air mass must be finite "
                         "and strictly positive (spectral: its band projection)")


# ponytail: steady winds only (one solve at build time, host-side scipy, not
# differentiable); time-dependent prescribed winds would need a per-stage
# solve inside the step.
class PrescribedMassFlux(NamedTuple):
    """Steady, discretely consistent prescribed air-mass fluxes.

    ``dp`` (cells..., nlev) steady layer air mass [Pa]; ``flux_h`` the
    core's horizontal air-mass flux per layer (lat-lon ``(Fu, Fv)`` on the
    u-faces (n_lat, n_lon+1, nlev) / v-faces (n_lat+1, n_lon, nlev); MPAS
    (nEdges, nlev); spectral ``(Fu cos(lat), Fv cos(lat))`` on the Gaussian
    grid) [Pa m/s]; ``w`` (cells..., nlev+1) interface air-mass flux [Pa/s],
    positive toward larger sigma (down); ``w_mean_removed`` (nlev+1,) the
    area-weighted mean shift taken out of each interface of the analytic
    ``W`` (its mean minus the top interface's)."""
    dp: jnp.ndarray
    flux_h: object
    w: jnp.ndarray
    w_mean_removed: jnp.ndarray


# ponytail: first-order donor cell (the rows' existing "upwind" vertical
# choice, codex vote); a higher-order limited vertical flux is the ceiling.
def donor_cell_vertical_flux(q, w):
    """First-order donor-cell (upwind) vertical tracer flux at the nlev+1
    interfaces, ``w`` (..., nlev+1) positive down, ``q`` (..., nlev, ntr):
    ``w q_above`` for ``w > 0``, ``w q_below`` otherwise; at the open top
    and surface the adjacent layer's value (zero-gradient), so a uniform
    tracer stays uniform when the air fluxes close the continuity."""
    w = w[..., None]
    qa = jnp.concatenate([q[..., :1, :], q], axis=-2)     # above each interface
    qb = jnp.concatenate([q, q[..., -1:, :]], axis=-2)    # below each interface
    return jnp.where(w > 0, w * qa, w * qb)


def van_leer_vertical_flux(q, w, sigma_coord):
    """Monotone van-Leer vertical tracer flux at the nlev+1 interfaces: ``w``
    times the upwind face value of the shared sigma reconstruction
    :func:`legoesm.grids.vertical.van_leer_face_values_sigma` (the lat-lon
    1-3 row, codex vote: halves the donor-cell centroid drift).  Its weights
    are ratios of ``sigma_coord`` spacings: exact for a sigma column and for
    the DCMIP 1-3 hybrid levels, whose pressure is affine in eta in every
    column (``create_dcmip13_hybrid``); the top and
    surface take the adjacent layer's value (as
    :func:`donor_cell_vertical_flux`).  ``q`` (..., nlev, ntr)."""
    from legoesm.grids.vertical import van_leer_face_values_sigma
    if q.shape[-2] < 4:      # the 4-cell stencil needs 4 layers
        return donor_cell_vertical_flux(q, w)
    q_pos, q_neg = (jnp.moveaxis(f, -1, -2) for f in
                    van_leer_face_values_sigma(jnp.moveaxis(q, -1, -2), sigma_coord))
    # the top and surface layers donate their own value (first order): the
    # shared reconstruction's linear-extrapolation ghost would let an empty
    # boundary layer export tracer (e.g. q = 0 over q = 1 gives 0.5)
    q_pos = q_pos.at[..., :2, :].set(q[..., :2, :] * 0 + q[..., :1, :])
    q_neg = q_neg.at[..., -2:, :].set(q[..., -2:, :] * 0 + q[..., -1:, :])
    q_pos = q_pos.at[..., -1, :].set(q[..., -1, :])
    q_neg = q_neg.at[..., 0, :].set(q[..., 0, :])
    w = w[..., None]
    return jnp.where(w > 0, w * q_pos, w * q_neg)


def _closed_boundary(w):
    w = np.asarray(w)
    return not np.any(w[..., 0]) or not np.any(w[..., -1])


def _area_mean_removed(w, area):
    """``W`` with every interface's area mean set to the common target (the
    solvability condition: equal means; module docstring), and the shift."""
    a = np.asarray(area, np.float64)
    m = np.tensordot(a, np.asarray(w, np.float64), axes=(tuple(range(a.ndim)),) * 2) / a.sum()
    m = m - (0.0 if _closed_boundary(w) else m[0])
    return np.asarray(w, np.float64) - m, m


def _bordered_poisson_solver(div_mat, grad_mat, area):
    """Factorise ``area * div grad`` (symmetric, constant null space) bordered
    by the constant vector; returns ``solve(rhs)`` for (ncell, k) right-hand
    sides with zero area-weighted mean (mean-zero ``chi``)."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    a = np.asarray(area, np.float64).ravel()
    lap = sp.diags(a) @ (div_mat @ grad_mat)
    n = lap.shape[0]
    one = sp.csr_matrix(np.ones((n, 1)))
    big = sp.bmat([[lap, one], [one.T, None]], format="csc")
    lu = spla.splu(big)

    def solve(rhs):
        rhs = np.asarray(rhs, np.float64).reshape(n, -1)
        b = np.vstack([a[:, None] * rhs, np.zeros((1, rhs.shape[1]))])
        return lu.solve(b)[:n]
    return solve


# ----------------------------------------------------------------------------
# lat-lon C-grid
# ----------------------------------------------------------------------------

def _latlon_metrics(grid):
    R = float(grid.radius)
    dlon = float(grid.dlon)
    hy = np.asarray(grid.dy, np.float64) * 0.5                       # (n_lat,)
    hx = R * dlon * np.asarray(grid.cos_lat_v, np.float64)           # (n_lat+1,)
    area = np.asarray(grid.area, np.float64)                         # (n_lat, n_lon)
    lat = np.asarray(grid.lat, np.float64)
    dxu = R * dlon * np.cos(lat)                                     # (n_lat,)
    dyv = R * np.diff(lat)                                           # (n_lat-1,)
    return hy, hx, area, dxu, dyv


def divergence_latlon(fu, fv, grid):
    """``div F`` (n_lat, n_lon, nlev) of C-grid face fluxes, the operator of
    ``cgrid_fv_flux_divergence_latlon_3d`` at ``q = 1`` (the core's own)."""
    from legoesm.core.operators_fv_latlon_3d import cgrid_velocity_divergence_3d
    return cgrid_velocity_divergence_3d(fu, fv, grid)


def _gradient_latlon(chi, grid):
    """C-grid face gradient of cell values ``chi`` (n_lat, n_lon, k): u-faces
    (west face of each cell, periodic wrap column appended), v-faces (zero at
    the two pole points)."""
    _, _, _, dxu, dyv = _latlon_metrics(grid)
    gu = (chi - jnp.roll(chi, 1, axis=1)) / dxu[:, None, None]
    gu = jnp.concatenate([gu, gu[:, :1]], axis=1)
    gv_in = (chi[1:] - chi[:-1]) / dyv[:, None, None]
    z = jnp.zeros_like(chi[:1])
    gv = jnp.concatenate([z, gv_in, z], axis=0)
    return gu, gv


def _latlon_matrices(grid):
    import scipy.sparse as sp
    hy, hx, area, dxu, dyv = _latlon_metrics(grid)
    n_lat, n_lon = area.shape
    nc = n_lat * n_lon
    cid = np.arange(nc).reshape(n_lat, n_lon)
    # unknown faces: u-faces (n_lat, n_lon) [west faces], interior v-faces (n_lat-1, n_lon)
    nu, nv = nc, (n_lat - 1) * n_lon
    uid = np.arange(nu).reshape(n_lat, n_lon)
    vid = nu + np.arange(nv).reshape(n_lat - 1, n_lon)
    rows, cols, vals = [], [], []
    # div: cell (j,i) = [hy_j (Fu_{j,i+1} - Fu_{j,i}) + hx_{j+1} Fv_{j+1} - hx_j Fv_j] / area
    jj, ii = np.meshgrid(np.arange(n_lat), np.arange(n_lon), indexing="ij")
    c = cid.ravel()
    rows += [c, c]
    cols += [uid[jj, (ii + 1) % n_lon].ravel(), uid.ravel()]
    vals += [(hy[jj] / area).ravel(), (-hy[jj] / area).ravel()]
    m_n = jj < n_lat - 1          # north v-face interior
    rows.append(cid[m_n]); cols.append(vid[jj[m_n], ii[m_n]])
    vals.append((hx[jj + 1] / area)[m_n])
    m_s = jj > 0
    rows.append(cid[m_s]); cols.append(vid[jj[m_s] - 1, ii[m_s]])
    vals.append((-hx[jj] / area)[m_s])
    div = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                        shape=(nc, nu + nv))
    # grad: u-face (j,i) = (chi_{j,i} - chi_{j,i-1}) / dxu_j ; v-face (j+1/2) = (chi_{j+1}-chi_j)/dyv_j
    gr, gc, gvv = [], [], []
    gr += [uid.ravel(), uid.ravel()]
    gc += [cid.ravel(), cid[jj, (ii - 1) % n_lon].ravel()]
    gvv += [(1.0 / dxu[jj]).ravel(), (-1.0 / dxu[jj]).ravel()]
    jv, iv = np.meshgrid(np.arange(n_lat - 1), np.arange(n_lon), indexing="ij")
    gr += [vid.ravel(), vid.ravel()]
    gc += [cid[jv + 1, iv].ravel(), cid[jv, iv].ravel()]
    gvv += [(1.0 / dyv[jv]).ravel(), (-1.0 / dyv[jv]).ravel()]
    grad = sp.csr_matrix((np.concatenate(gvv), (np.concatenate(gr), np.concatenate(gc))),
                         shape=(nu + nv, nc))
    return div, grad, area


def continuity_residual_latlon(mf: PrescribedMassFlux, grid):
    """``div F_k + delta W_k`` [Pa/s] (n_lat, n_lon, nlev) with the core's
    own divergence: the steady air mass's discrete tendency (negated)."""
    fu, fv = mf.flux_h
    return divergence_latlon(fu, fv, grid) + (mf.w[..., 1:] - mf.w[..., :-1])


def consistent_mass_flux_latlon(grid, sigma_coord, wind_geo_fn, p_s, *, t=0.0,
                                refine=3) -> PrescribedMassFlux:
    """Corrected air fluxes on the lat-lon C-grid (module docstring).

    ``wind_geo_fn(t, lon, lat, sigma_coord) -> (u, v, sigma_dot)`` STEADY
    (evaluated once at ``t``); ``p_s`` (n_lat, n_lon) [Pa] the steady
    surface pressure.  ``F0`` = two-cell mean ``p_s`` x face wind x dsigma
    (v-faces at the two pole points carry no flux)."""
    _require_x64()
    from legoesm.atmosphere.dynamics.gcm.tracer_transport_latlon import (
        uface_coords, vface_coords)
    f64 = jnp.float64
    ps = jnp.asarray(np.asarray(p_s, np.float64))
    _require_positive(ps)
    lon_u, lat_u = uface_coords(grid)
    u_int = jnp.asarray(wind_geo_fn(t, lon_u, lat_u, sigma_coord)[0], f64)
    ps_u = 0.5 * (ps + jnp.roll(ps, 1, axis=1))
    fu = _face_air_flux(sigma_coord, ps_u, u_int)
    fu = jnp.concatenate([fu, fu[:, :1]], axis=1)
    lon_v, lat_v = vface_coords(grid)
    v_face = jnp.asarray(wind_geo_fn(t, lon_v, lat_v, sigma_coord)[1], f64)
    ps_v = 0.5 * (ps[1:] + ps[:-1])
    fv_in = _face_air_flux(sigma_coord, ps_v, v_face[1:-1])
    z = jnp.zeros_like(fv_in[:1])
    fv = jnp.concatenate([z, fv_in, z], axis=0)
    sd = np.asarray(wind_geo_fn(t, grid.lon2d, grid.lat2d, sigma_coord)[2], np.float64)
    w_raw = np.asarray(ps)[..., None] * sd
    w, wmean = _area_mean_removed(w_raw, grid.area)
    w = jnp.asarray(w)
    dp = _layer_dp(sigma_coord, ps)
    mf = PrescribedMassFlux(dp, (fu, fv), w, jnp.asarray(wmean))
    if refine == 0:
        return mf
    div_m, grad_m, area = _latlon_matrices(grid)
    solve = _bordered_poisson_solver(div_m, grad_m, area)
    n_lat, n_lon = area.shape
    for _ in range(refine):
        r = -np.asarray(continuity_residual_latlon(mf, grid), np.float64)
        r = r - np.tensordot(area, r, axes=((0, 1), (0, 1))) / area.sum()
        chi = jnp.asarray(solve(r.reshape(n_lat * n_lon, -1)).reshape(r.shape))
        gu, gv = _gradient_latlon(chi, grid)
        fu, fv = mf.flux_h
        mf = mf._replace(flux_h=(fu + gu, fv + gv))
    return mf


# ----------------------------------------------------------------------------
# MPAS Voronoi (TRiSK)
# ----------------------------------------------------------------------------

def _mpas_matrices(mesh):
    import scipy.sparse as sp
    eoc = np.asarray(mesh.edgesOnCell)            # (maxEdges, nCells)
    sgn = np.asarray(mesh.edgeSignOnCell, np.float64)
    dv = np.asarray(mesh.dvEdge, np.float64)
    dc = np.asarray(mesh.dcEdge, np.float64)
    area = np.asarray(mesh.areaCell, np.float64)
    nC = area.shape[0]
    nE = dv.shape[0]
    ok = eoc >= 0
    cc = np.broadcast_to(np.arange(nC)[None, :], eoc.shape)
    e = np.maximum(eoc, 0)
    div = sp.csr_matrix(((sgn * dv[e] / area[None, :])[ok], (cc[ok], e[ok])), shape=(nC, nE))
    c1 = np.asarray(mesh.cellsOnEdge[0])
    c2 = np.asarray(mesh.cellsOnEdge[1])
    ar = np.arange(nE)
    grad = sp.csr_matrix((np.concatenate([1.0 / dc, -1.0 / dc]),
                          (np.concatenate([ar, ar]), np.concatenate([c2, c1]))), shape=(nE, nC))
    return div, grad, area


def continuity_residual_mpas(mf: PrescribedMassFlux, mesh):
    """``div F_k + delta W_k`` [Pa/s] (nCells, nlev), TRiSK divergence."""
    from legoesm.core.operators_voronoi import divergence_cell_3d
    return divergence_cell_3d(mf.flux_h, mesh) + (mf.w[..., 1:] - mf.w[..., :-1])


def consistent_mass_flux_mpas(mesh, sigma_coord, u_edge, sigma_dot, p_s, *,
                              refine=3) -> PrescribedMassFlux:
    """Corrected air fluxes on an MPAS mesh.  ``u_edge`` (nEdges, nlev) the
    STEADY edge-normal wind, ``sigma_dot`` (nCells, nlev+1), ``p_s``
    (nCells,) [Pa].  ``F0 = u dp_e`` with ``dp_e`` the two-cell mean (MPAS
    ``ru``, as :func:`tracer_transport_mpas._mpas_stage_fluxes`)."""
    _require_x64()
    from legoesm.core.operators_voronoi import gradient_edge_3d
    ps = jnp.asarray(np.asarray(p_s, np.float64))
    _require_positive(ps)
    dp = _layer_dp(sigma_coord, ps)
    c1, c2 = mesh.cellsOnEdge[0], mesh.cellsOnEdge[1]
    fm = jnp.asarray(u_edge, jnp.float64) * (0.5 * (dp[c1] + dp[c2]))
    w_raw = np.asarray(ps)[:, None] * np.asarray(sigma_dot, np.float64)
    w, wmean = _area_mean_removed(w_raw, mesh.areaCell)
    mf = PrescribedMassFlux(dp, fm, jnp.asarray(w), jnp.asarray(wmean))
    if refine == 0:
        return mf
    div_m, grad_m, area = _mpas_matrices(mesh)
    solve = _bordered_poisson_solver(div_m, grad_m, area)
    for _ in range(refine):
        r = -np.asarray(continuity_residual_mpas(mf, mesh), np.float64)
        r = r - (area @ r) / area.sum()
        chi = jnp.asarray(solve(r))
        mf = mf._replace(flux_h=mf.flux_h + gradient_edge_3d(chi, mesh))
    return mf


# ----------------------------------------------------------------------------
# spectral (Gaussian grid, transform method)
# ----------------------------------------------------------------------------

def spectral_flux_divergence_hat(grid, fu_cos, fv_cos):
    """Transform divergence (n_sh, k) of ``(F cos, F cos)`` grid fluxes, the
    Bourke operator of ``tracer_transport_spectral`` (unmasked)."""
    from legoesm.grids.gaussian import sh_analysis_dmu_3d, sh_analysis_oc2_3d
    ims = 1j * grid.ms.astype(jnp.float64)[:, None] / grid.radius
    return ims * sh_analysis_oc2_3d(grid, fu_cos) - sh_analysis_dmu_3d(grid, fv_cos) / grid.radius


def continuity_residual_spectral(mf: PrescribedMassFlux, grid, mask):
    """Grid-space ``mask (div F_k + delta W_k)`` [Pa/s]: the (negated)
    air-mass tendency the spectral model would apply."""
    from legoesm.grids.gaussian import sh_analysis_3d, sh_synthesis_3d
    fu, fv = mf.flux_h
    hat = (spectral_flux_divergence_hat(grid, fu, fv)
           + sh_analysis_3d(grid, mf.w[..., 1:] - mf.w[..., :-1]))
    return sh_synthesis_3d(grid, mask[:, None] * hat)


def consistent_mass_flux_spectral(grid, sigma_coord, wind_geo_fn, p_s, mask, *,
                                  t=0.0, refine=3) -> PrescribedMassFlux:
    """Corrected air fluxes on the Gaussian grid.  The steady air mass is
    ``dp = p_T dsigma`` with ``p_T`` the ``mask`` (Orszag band) projection
    of ``p_s`` (n_lat, n_lon) [Pa]; ``W`` is the band projection of
    ``p_s sigma_dot`` with its (0,0) coefficient set to the common target (module docstring);
    ``F0 = p_T dsigma V cos(lat)`` and ``chi`` from the spectral inverse
    Laplacian of the in-band divergence defect (``uv_from_vordiv_3d``)."""
    _require_x64()
    from legoesm.grids.gaussian import (sh_analysis_3d, sh_synthesis_3d,
                                        uv_from_vordiv_3d)
    f64 = jnp.float64
    dsig = jnp.asarray(np.asarray(sigma_coord.dsigma, np.float64))
    mask = jnp.asarray(mask, f64)
    ps = jnp.asarray(np.asarray(p_s, np.float64))
    p_t = sh_synthesis_3d(grid, mask[:, None] * sh_analysis_3d(grid, ps[..., None]))[..., 0]
    _require_positive(p_t)
    dp = _layer_dp(sigma_coord, p_t)
    u, v, sd = wind_geo_fn(t, grid.lon2d, grid.lat2d, sigma_coord)
    cosl = grid.cos_lat[:, None, None]
    fu = dp * jnp.asarray(u, f64) * cosl
    fv = dp * jnp.asarray(v, f64) * cosl
    w_hat = mask[:, None] * sh_analysis_3d(grid, ps[..., None] * jnp.asarray(sd, f64))
    # the (0,0) coefficient (index 0, the global mean) of every interface set
    # to the common target (solvability; zero with a closed boundary, else
    # the top interface's, as the grid-point builders)
    target = 0.0 if _closed_boundary(sd) else w_hat[0, 0]
    wmean = np.real(np.asarray(w_hat[0] - target))
    w_hat = w_hat.at[0].set(target)
    w = sh_synthesis_3d(grid, w_hat)
    mf = PrescribedMassFlux(dp, (fu, fv), w, jnp.asarray(wmean))
    zero = jnp.zeros((grid.n_sh, dsig.shape[0]), jnp.complex128)
    for _ in range(refine):
        fu, fv = mf.flux_h
        defect = mask[:, None] * (spectral_flux_divergence_hat(grid, fu, fv)
                                  + sh_analysis_3d(grid, w[..., 1:] - w[..., :-1]))
        gu, gv = uv_from_vordiv_3d(grid, zero, -defect)
        mf = mf._replace(flux_h=(fu + gu, fv + gv))
    return mf
