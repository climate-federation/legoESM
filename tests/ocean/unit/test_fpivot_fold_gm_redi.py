"""F-pivot fold (eORCA1 layout, halo row stripped): GM/Redi across the fold.

NEMO computes the ldf_slp slopes and the traldf_iso / ldf_eiv fluxes on the
halo from lbc_lnk-filled inputs (nn_hls=2), so the top row's ``jj+1``
neighbour is its F-pivot fold image.  The reference here is an INDEPENDENT
construction of that: a regular (fold-free) domain with ``K`` extra rows that
ARE the fold images (row ``n_lat+k`` = row ``n_lat-1-k`` permuted by ``P_T``),
run through the legacy (non-fold) stencils.  The F-pivot slopes, eiv transport
and tendency on the stored rows must match it to round-off.

Grid: regular lat-lon band -60..80 N with the F-pivot descriptor (every row
uniform in longitude, so the geometry is fold-symmetric), partial cells,
random bathymetry, land on two top-row cells, random stratified T/S.
"""
from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from legoesm import constants  # noqa: E402
from legoesm.grids.latlon import (  # noqa: E402
    create_latlon_geometry, ensure_geometry)
from legoesm.grids.tripole import (  # noqa: E402
    create_synthetic_tripole, create_synthetic_tripole_fpivot)
from legoesm.ocean.dynamics.latlon_cgrid_operators import (  # noqa: E402
    compute_face_masks)
from legoesm.ocean.eos import make_eos_fn  # noqa: E402
from legoesm.ocean.physics.lateral_mixing import (  # noqa: E402
    gm_redi_latlon_cgrid as gm)
from legoesm.ocean.physics.lateral_mixing.config import (  # noqa: E402
    GMRediConfig)
from legoesm.ocean.vertical import (  # noqa: E402
    create_ocean_z_star, create_partial_cell_coordinate)

N_LAT, N_LON, NLEV, K = 12, 16, 6, 4
H_MAX = 4000.0
DT = 3600.0
CFG = GMRediConfig(slope_scheme="nemo_iso_lap", slope_positions="nemo_native")
EOS = make_eos_fn("nemo_seos")


def _fpivot_grid():
    lat = np.deg2rad(np.linspace(-60.0, 80.0, N_LAT))
    return create_synthetic_tripole_fpivot(
        N_LAT, N_LON, lat_1d=jnp.asarray(lat), dtype=jnp.float64)


def _fields(seed=0):
    rng = np.random.default_rng(seed)
    land = np.ones((N_LAT, N_LON))
    land[0:2] = 0.0
    land[5, 3:6] = 0.0
    land[-1, 2] = 0.0
    land[-1, 9] = 0.0
    H = (H_MAX - 2500.0 * rng.random((N_LAT, N_LON))) * land
    z = create_ocean_z_star(n_levels=NLEV, H_max=H_MAX)
    depth = np.abs(np.asarray(z.z_full_ref))
    T = (4.0 + 18.0 * np.exp(-depth / 800.0))[None, None, :] \
        + 1.5 * rng.standard_normal((N_LAT, N_LON, NLEV))
    S = 35.0 + 0.3 * rng.standard_normal((N_LAT, N_LON, NLEV))
    q = rng.standard_normal((N_LAT, N_LON, NLEV))
    kap = 500.0 + 800.0 * rng.random((N_LAT, N_LON))
    return land, H, T, S, q, kap


def _extend_rows(a, perm):
    """Append K fold-image rows: row n_lat+k = row n_lat-1-k under perm."""
    a = np.asarray(a)
    img = [a[a.shape[0] - 1 - k][perm] for k in range(K)]
    return np.concatenate([a, np.stack(img)], axis=0)


def _reference_grid(g_fp):
    """Fold-free geometry of n_lat+K rows whose extra rows mirror the stored
    top rows (the synthetic metrics are uniform in longitude)."""
    base = create_latlon_geometry(N_LAT + K, N_LON, dtype=jnp.float64)
    ext_c = lambda f: jnp.asarray(_extend_rows(
        np.asarray(f), np.arange(np.asarray(f).shape[1])))
    dxv = np.asarray(g_fp.dx_v)
    dyv = np.asarray(g_fp.dy_v)
    # v faces n_lat+1+k mirror face n_lat-1-k across the fold line (face n_lat).
    ext_v = lambda f: jnp.asarray(np.concatenate(
        [f, np.stack([f[N_LAT - 1 - k] for k in range(K)])], axis=0))
    return base._replace(
        dx_T=ext_c(g_fp.dx_T), dy_T=ext_c(g_fp.dy_T),
        dx_u=ext_c(g_fp.dx_u), dy_u=ext_c(g_fp.dy_u),
        dx_v=ext_v(dxv), dy_v=ext_v(dyv))


def _run(grid, land, H, T, S, q, kap_redi, kap_gm, *, msc=True):
    z = create_partial_cell_coordinate(
        create_ocean_z_star(n_levels=NLEV, H_max=H_MAX), jnp.asarray(H))
    land = jnp.asarray(land)
    u_mask, v_mask = compute_face_masks(land, grid)
    act = gm._nemo_native_active_3d(land, z, jnp.asarray(H), jnp.float64)
    p = (constants.rho_ocean * constants.g
         * jnp.abs(z.z_full_ref))[None, None, :] * jnp.ones_like(jnp.asarray(T))
    T, S, q = map(jnp.asarray, (T, S, q))
    rho = EOS(T, S, p)
    slopes = gm.compute_nemo_native_slopes(
        rho, T, S, land, u_mask, v_mask, z, grid, CFG, EOS, active_3d=act)
    J = jnp.ones(land.shape)
    tend, bolus = gm.nemo_iso_lap_tracer_tendency_latlon_cgrid(
        q, None, None, land, u_mask, v_mask, z, J, grid,
        jnp.asarray(kap_redi), act, native_slopes=slopes,
        msc_stabilize=msc, dt=DT, kappa_GM=jnp.asarray(kap_gm),
        gm_bolus_advection="centred", return_bolus=True)
    e3t = z.dz_ref[None, None, :] * J[:, :, None]
    geom = ensure_geometry(grid)
    vol = (geom.dx_T * geom.dy_T)[:, :, None] * e3t * act
    return slopes, tend, bolus, vol, v_mask


def _assert_close(a, b, name):
    a, b = np.asarray(a), np.asarray(b)
    scale = max(np.abs(b).max(), 1e-300)
    err = np.abs(a - b).max() / scale
    assert err < 1e-10, f"{name}: max rel diff {err:.3e} (scale {scale:.3e})"


def test_fold_matches_unfolded_reference():
    """Slopes, eiv transport and Redi+GM tendency on the stored rows equal the
    same stencils run on a domain whose extra rows are the fold images.
    Uniform kappa (the reference evaluates the fold face from BOTH sides)."""
    g_fp = _fpivot_grid()
    P = np.asarray(g_fp.fold.perm_T)
    land, H, T, S, q, _ = _fields()
    fp = _run(g_fp, land, H, T, S, q, 900.0, 700.0)
    ext = lambda a: _extend_rows(a, P)
    ref = _run(_reference_grid(g_fp), ext(land), ext(H), ext(T), ext(S),
               ext(q), 900.0, 700.0)
    assert np.asarray(fp[4])[-1].sum() > 0          # fold line open
    for name, a, b in zip(("uslp", "vslp", "wslpi", "wslpj"), fp[0], ref[0]):
        _assert_close(a, np.asarray(b)[:N_LAT], name)
    for name, a, b in zip(("u_eiv", "v_eiv", "w_eiv"), fp[2], ref[2]):
        _assert_close(a, np.asarray(b)[:N_LAT], name)
    _assert_close(fp[1], np.asarray(ref[1])[:N_LAT], "tendency")
    # the fold actually carries flux (else the comparison is vacuous)
    assert np.abs(np.asarray(fp[2][1])[-1]).max() > 0
    assert np.abs(np.asarray(fp[0][1])[-1]).max() > 0


def test_fold_line_fields_antisymmetric():
    """vslp and the eiv v-transport on the fold line obey NEMO's lbc 'V',-1
    identity v(i) = -v(P_V i) (one face stored twice)."""
    g_fp = _fpivot_grid()
    PV = np.asarray(g_fp.fold.perm_v)
    land, H, T, S, q, kap = _fields(1)
    slopes, _, bolus, _, _ = _run(g_fp, land, H, T, S, q, kap, kap)
    for a in (slopes[1], bolus[1]):
        top = np.asarray(a)[-1]
        np.testing.assert_allclose(top, -top[PV], atol=1e-14 * np.abs(top).max())


def test_uniform_tracer_stays_uniform():
    g_fp = _fpivot_grid()
    land, H, T, S, _, kap = _fields(2)
    q = np.full(T.shape, 7.25)
    _, tend, _, _, _ = _run(g_fp, land, H, T, S, q, kap, kap)
    assert np.abs(np.asarray(tend)).max() < 1e-18


def test_global_tracer_integral_conserved():
    """Non-uniform (fold-asymmetric) kappa: the fold-line flux must still
    cancel between its two stored copies."""
    g_fp = _fpivot_grid()
    land, H, T, S, q, kap = _fields(3)
    _, tend, _, vol, _ = _run(g_fp, land, H, T, S, q, kap, kap[::-1])
    t = np.asarray(tend) * np.asarray(vol)
    assert np.abs(t).max() > 0
    assert abs(t.sum()) / np.abs(t).sum() < 1e-13


@pytest.mark.parametrize("make", ["regular", "legacy_tripole"])
def test_non_fpivot_layouts_use_plain_roll(make):
    """Off the F-pivot layout every fold helper is the historical roll /
    identity (bitwise)."""
    g = (create_latlon_geometry(N_LAT, N_LON, dtype=jnp.float64)
         if make == "regular"
         else create_synthetic_tripole(N_LAT, N_LON, dtype=jnp.float64))
    a = jnp.asarray(np.random.default_rng(4).standard_normal((N_LAT, N_LON, 3)))
    up, dn = jnp.roll(a, -1, axis=0), jnp.roll(a, 1, axis=0)
    for f in (gm._jp1_t, gm._jp1_ue, gm._jp1_vn):
        np.testing.assert_array_equal(f(a, g, -1.0), up)
    np.testing.assert_array_equal(gm._jm1_vn(a, g), dn)
    np.testing.assert_array_equal(gm._fold_line_vn(a, g, -1.0), a)


def test_production_dispatch_matches_unfolded_reference():
    """The eORCA1 card path through the public dispatcher (nemo_iso_lap +
    native slopes + Treguier GM + nemo21 Redi + MSC + through_fct bolus) and
    the implicit-K33 getter match the unfolded reference on the stored rows."""
    from legoesm.ocean.physics.lateral_mixing.config import TreguierConfig
    g_fp = _fpivot_grid()
    P = np.asarray(g_fp.fold.perm_T)
    land, H, T, S, _, _ = _fields(6)
    ext = lambda a: _extend_rows(a, P)
    f_fp = np.asarray(g_fp.f_T)                       # (n_lat, n_lon)
    g_ref = _reference_grid(g_fp)._replace(f_T=jnp.asarray(ext(f_fp)))

    def run(grid, land, H, T, S, f2d):
        z = create_partial_cell_coordinate(
            create_ocean_z_star(n_levels=NLEV, H_max=H_MAX), jnp.asarray(H))
        land = jnp.asarray(land)
        u_mask, v_mask = compute_face_masks(land, grid)
        cfg = GMRediConfig(
            slope_scheme="nemo_iso_lap", slope_positions="nemo_native",
            treguier=TreguierConfig(enabled=True, aei0=900.0, kappa_min=0.0),
            redi_coefficient="nemo21", redi_aht0=900.0,
            redi_f_f=jnp.asarray(f2d), msc_stabilize=True, implicit_K33=True,
            gm_bolus_advection="through_fct",
            gm_bolus_kappa_face_average=True)
        kw = dict(eos="nemo_seos", mask=land, u_mask=u_mask, v_mask=v_mask,
                  dt=DT)
        eta = jnp.zeros(land.shape)
        dT, dS, bolus = gm.gm_redi_tracer_tendency_latlon(
            jnp.asarray(T), jnp.asarray(S), eta, jnp.asarray(H), grid, z, cfg,
            f_coriolis=jnp.asarray(f2d), return_bolus_transport=True, **kw)
        k33 = gm.compute_isoneutral_K33_latlon(
            jnp.asarray(T), jnp.asarray(S), eta, jnp.asarray(H), grid, z, cfg,
            **kw)
        return (dT, dS) + tuple(bolus) + (k33,)

    fp = run(g_fp, land, H, T, S, f_fp)
    ref = run(g_ref, ext(land), ext(H), ext(T), ext(S), ext(f_fp))
    assert np.abs(np.asarray(fp[3])[-1]).max() > 0     # v_eiv crosses the fold
    for name, a, b in zip(("dT", "dS", "u_eiv", "v_eiv", "w_eiv", "K33"),
                          fp, ref):
        _assert_close(a, np.asarray(b)[:N_LAT], name)
