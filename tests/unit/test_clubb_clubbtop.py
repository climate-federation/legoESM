"""CAM's chemistry tropopause and CLUBB's ``clubbtop`` (user decisions 2026-10-15).

1a: ``cam_chem_tropopause_level`` ports ``tropopause_findChemTrop`` (twmo
lapse-rate locator + the polar rule); 2: the analytic ``25000 - 15000
cos^2(lat)`` Pa surface stands in for CAM's climatology file; 3: the
resulting ``clubbtop`` (pushed down over quiet levels) bounds the water
reset, the returned water and the energy fixer, as in clubb_intr.F90.
Where CLUBB itself runs (the ``trop_cloud_top_press`` taper) is unchanged.

The oracle for the vectorised port is ``_twmo_ref``: a line-by-line
transcription of tropopause.F90 ``twmo`` (cam_cesm2_1_rel 720-812), 1-based
like the Fortran.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants  # noqa: E402
from legoesm.atmosphere.physics._shared import (  # noqa: E402
    compute_heights_from_sigma,
    compute_rho,
)
from legoesm.atmosphere.physics.clouds.cloud_fraction import (  # noqa: E402
    cam6_tropopause_analytic,
    cam_chem_tropopause_level,
)
from legoesm.atmosphere.physics.turbulence import clubb as clubb_mod  # noqa: E402
from legoesm.atmosphere.physics.turbulence.clubb import (  # noqa: E402
    CLUBBConfig,
    clubb_step,
    clubb_top_level,
    clubb_turbulence_prognostic,
    init_clubb_moments,
    integrate_clubb_column,
    pack_clubb_moments,
    unpack_clubb_moments,
)

_CUT_PA = 15000.0                       # the production CLUBB top (150 hPa)
_ON = CLUBBConfig(prognostic=True, clubb_dt=300.0, trop_cloud_top_press=_CUT_PA)
_OFF = CLUBBConfig(prognostic=True, clubb_dt=300.0)
_DT = 300.0
_TROP_LAPSE = 6.5e-3                    # test profile: tropospheric lapse [K/m]
_STRAT_LAPSE = -2.5e-3                  # test profile: stratospheric warming [K/m]


# --------------------------------------------------------------------------
# Fortran transcription (oracle)
# --------------------------------------------------------------------------
def _twmo_ref(t, p, plimu=45000.0, pliml=7500.0, gam=-0.002):
    kap = constants.kappa
    faktor = -constants.g / constants.R_d
    deltaz = 2000.0
    t = [None] + [float(x) for x in t]
    p = [None] + [float(x) for x in p]
    level = len(t) - 1

    def fit(j):                                     # pair (j-1, j)
        pmk = 0.5 * (p[j - 1] ** kap + p[j] ** kap)
        pm = pmk ** (1.0 / kap)
        a = (t[j - 1] - t[j]) / (p[j - 1] ** kap - p[j] ** kap)
        b = t[j] - a * p[j] ** kap
        tm = a * pmk + b
        dtdp = a * kap * pm ** (kap - 1.0)
        return pmk, pm, tm, faktor * dtdp * pm / tm

    pmk, pm, tm, dtdz = fit(level)
    for j in range(level - 1, 1, -1):              # main_loop
        pmk0, dtdz0 = pmk, dtdz
        pmk, pm, tm, dtdz = fit(j)
        if dtdz <= gam:
            continue
        if pm > plimu:
            continue
        if dtdz0 < gam:
            ag = (dtdz - dtdz0) / (pmk - pmk0)
            bg = dtdz0 - ag * pmk0
            ptph = math.exp(math.log((gam - bg) / ag) / kap)
        else:
            ptph = pm
        if ptph < pliml or ptph > plimu:
            continue
        p2km = ptph + deltaz * (pm / tm) * faktor
        asum, icount, rejected = 0.0, 0, False
        for jj in range(j, 1, -1):                  # in_loop
            _, pm2, _, dtdz2 = fit(jj)
            if pm2 > ptph:
                continue
            if pm2 < p2km:
                break
            asum += dtdz2
            icount += 1
            if asum / icount <= gam:
                rejected = True                     # cycle main_loop
                break
        if rejected:
            continue
        return ptph
    return -99.0


def _level_ref(tp, pint):
    """do k = pver, 2, -1: if (tP >= pint(k)) -> k; 0-based, -1 = NOTFOUND."""
    for k in range(len(pint) - 1, 1, -1):          # Fortran k = pver..2
        if tp >= pint[k - 1]:
            return k - 1
    return -1


def _chemtrop_ref(t, pmid, pint, lat):
    tp = _twmo_ref(t, pmid)
    lev = _level_ref(tp, pint) if tp > 0 else -1
    if abs(math.degrees(lat)) > 50.0 and lev >= 0 and pmid[lev] <= 12500.0:
        lev = -1
    if lev < 0:
        lev = _level_ref(25000.0 - 15000.0 * math.cos(lat) ** 2, pint)
    return lev


# --------------------------------------------------------------------------
# Columns
# --------------------------------------------------------------------------
def _temperature(p, p_trop, T_sfc=300.0):
    """Lapse 6.5 K/km to ``p_trop``, warming 2.5 K/km above it."""
    T = T_sfc * (p / 1.0e5) ** (constants.R_d * _TROP_LAPSE / constants.g)
    T_t = T_sfc * (p_trop / 1.0e5) ** (constants.R_d * _TROP_LAPSE / constants.g)
    T_s = T_t * (p / p_trop) ** (constants.R_d * _STRAT_LAPSE / constants.g)
    return np.where(p >= p_trop, T, T_s)


def _grid(ncol, nlev=40, p_top=1.0e3):
    p_half = np.geomspace(p_top, 1.0e5, nlev + 1)[None, :] * np.ones((ncol, nlev + 1))
    return p_half, 0.5 * (p_half[:, :-1] + p_half[:, 1:])


def _column(lat_deg, p_trop=1.0e4, ncol=2, nlev=40, seed=0):
    """Moist, sheared column with a lapse-rate tropopause at ``p_trop``;
    heights and density from the lane's helpers."""
    rng = np.random.default_rng(seed)
    p_half, p_full = _grid(ncol, nlev)
    T = _temperature(p_full, p_trop) + 0.2 * rng.standard_normal((ncol, nlev))
    q_v = 1.5e-2 * (p_full / 1.0e5) ** 3 + 3.0e-6
    q_c = np.where(np.abs(p_full - 8.5e4) < 4.0e3, 2.0e-4, 0.0)
    u = 5.0 + 25.0 * (1.0 - p_full / 1.0e5) + rng.standard_normal((ncol, nlev))
    v = 2.0 + rng.standard_normal((ncol, nlev))
    a = lambda x: jnp.asarray(x, dtype=jnp.float64)  # noqa: E731
    T, q_v, p_half, p_full = a(T), a(q_v), a(p_half), a(p_full)
    z_full, z_half = compute_heights_from_sigma(T, p_half, q_v=q_v)
    return dict(u=a(u), v=a(v), T=T, q_v=q_v, q_c=a(q_c), p_full=p_full,
                p_half=p_half, z_full=z_full, z_half=z_half,
                T_sfc=T[:, -1] + 2.0, q_sfc=1.1 * q_v[:, -1],
                rho=compute_rho(T, p_full, q_v=q_v),
                lat=a(np.full(ncol, np.deg2rad(lat_deg))))


def _args(c):
    return (c["u"], c["v"], c["T"], c["q_v"])


def _spun_up_moments(c, n=4):
    """Moments mixed up to the lid (no cutoff), so CLUBB moves water aloft."""
    ncol, nlev = c["T"].shape
    carry = pack_clubb_moments(init_clubb_moments(ncol, nlev, _OFF))
    for _ in range(n):
        _, carry = clubb_turbulence_prognostic(
            *_args(c), carry, c["p_full"], c["p_half"], c["z_full"],
            c["z_half"], c["T_sfc"], c["q_sfc"], c["rho"], _DT, _OFF,
            q_c=c["q_c"], lat=c["lat"])
    return unpack_clubb_moments(carry)


def _step(c, cfg, moments, **kw):
    return clubb_step(*_args(c), moments, c["p_full"], c["p_half"],
                      c["z_full"], c["z_half"], c["T_sfc"], c["q_sfc"],
                      c["rho"], _DT, cfg, q_c=c["q_c"], **kw)


def _taper(p_full):
    """The production taper w (where CLUBB runs), CLUBBConfig fields."""
    return jax.nn.sigmoid((jnp.log(p_full) - jnp.log(_ON.trop_cloud_top_press))
                          / _ON.trop_cloud_taper_lnp_width)


# --------------------------------------------------------------------------
# 1a / 2: the finder
# --------------------------------------------------------------------------
def test_finder_matches_the_fortran_transcription():
    """Random columns, every branch (twmo accepted, polar rejection,
    fallback when twmo finds nothing): level-for-level equal to the scalar
    Fortran transcription, eager and under jit."""
    rng = np.random.default_rng(1)
    ncol, nlev = 96, 40
    p_half, p_full = _grid(ncol, nlev)
    p_trop = rng.uniform(7.0e3, 3.0e4, ncol)
    p_trop[:12] = 1.0e3                      # no tropopause below the lid
    T = (_temperature(p_full, p_trop[:, None])
         + 0.3 * rng.standard_normal((ncol, nlev)))
    lat = np.deg2rad(rng.uniform(-90.0, 90.0, ncol))
    ref = np.array([_chemtrop_ref(T[i], p_full[i], p_half[i], lat[i])
                    for i in range(ncol)])
    args = tuple(jnp.asarray(x) for x in (T, p_full, p_half, lat))
    got = np.asarray(cam_chem_tropopause_level(*args))
    np.testing.assert_array_equal(got, ref)
    np.testing.assert_array_equal(
        np.asarray(jax.jit(cam_chem_tropopause_level)(*args)), ref)
    # Every branch was exercised.
    tw = np.array([_twmo_ref(T[i], p_full[i]) for i in range(ncol)])
    tw_lev = np.array([_level_ref(x, p_half[i]) if x > 0 else -1
                       for i, x in enumerate(tw)])
    polar = (np.abs(np.degrees(lat)) > 50.0) & (tw_lev >= 0)
    rejected = polar & (p_full[np.arange(ncol), np.maximum(tw_lev, 0)] <= 12500.0)
    assert (tw <= 0).sum() >= 5 and rejected.sum() >= 3
    assert ((tw > 0) & ~rejected).sum() >= 30


def test_tropical_tropopause_lies_above_150_hpa():
    c = _column(0.0)
    lev = np.asarray(cam_chem_tropopause_level(c["T"], c["p_full"], c["p_half"], c["lat"]))
    p_full = np.asarray(c["p_full"])
    for i, k in enumerate(lev):
        assert p_full[i, k] < _CUT_PA
        assert abs(p_full[i, k] - 1.0e4) < 2.0e3, p_full[i, k]


def test_polar_rule_rejects_a_high_tropopause_and_falls_back_to_the_formula():
    """Same profile (tropopause near 100 hPa): kept at 40 deg, rejected
    poleward of 50 deg (pmid <= 125 hPa) -> level of 250 - 150 cos^2(lat)."""
    c40, c70 = _column(40.0), _column(-70.0)
    f = lambda c: np.asarray(cam_chem_tropopause_level(  # noqa: E731
        c["T"], c["p_full"], c["p_half"], c["lat"]))
    p_full = np.asarray(c40["p_full"])
    lev40, lev70 = f(c40), f(c70)
    assert np.all(p_full[np.arange(2), lev40] <= 12500.0)    # twmo kept
    p_fb = np.asarray(cam6_tropopause_analytic(c70["lat"]))  # 232.5 hPa
    want = [_level_ref(p_fb[i], np.asarray(c70["p_half"])[i]) for i in range(2)]
    np.testing.assert_array_equal(lev70, want)
    assert np.all(p_full[np.arange(2), lev70] > _CUT_PA)


def test_fallback_formula_when_twmo_finds_nothing():
    """No inversion below the lid: the formula's level, at any latitude."""
    p_half, p_full = _grid(3)
    T = _temperature(p_full, 1.0e2)                 # troposphere to the lid
    lat = np.deg2rad(np.array([0.0, 30.0, 60.0]))
    got = np.asarray(cam_chem_tropopause_level(
        *(jnp.asarray(x) for x in (T, p_full, p_half, lat))))
    p_fb = 25000.0 - 15000.0 * np.cos(lat) ** 2
    np.testing.assert_array_equal(
        got, [_level_ref(p_fb[i], p_half[i]) for i in range(3)])


# --------------------------------------------------------------------------
# 3: clubbtop
# --------------------------------------------------------------------------
def _clubbtop_ref(troplev, rtp2, rcm):
    """clubb_intr.F90:2042-2045 after the documented clamp, 0-based."""
    nlev = len(rcm)
    ct = min(max(troplev, 1), nlev - 2)
    while rtp2[ct] <= 1.0e-15 and rcm[ct] == 0.0 and ct < nlev - 2:
        ct += 1
    return ct


def test_push_down_matches_the_fortran_loop():
    rng = np.random.default_rng(2)
    ncol, nlev = 400, 12
    troplev = rng.integers(-1, nlev, ncol)
    quiet_rtp2 = rng.random((ncol, nlev + 1)) < 0.7
    quiet_rcm = rng.random((ncol, nlev)) < 0.8
    rtp2 = np.where(quiet_rtp2, 1.0e-16, 1.0e-12)
    rtp2[:, 5] = 1.0e-15                            # the .le. boundary
    rcm = np.where(quiet_rcm, 0.0, 1.0e-9)
    got = np.asarray(clubb_top_level(jnp.asarray(troplev), jnp.asarray(rtp2),
                                     jnp.asarray(rcm)))
    ref = [_clubbtop_ref(troplev[i], rtp2[i], rcm[i]) for i in range(ncol)]
    np.testing.assert_array_equal(got, ref)
    assert np.sum(got > np.clip(troplev, 1, nlev - 2)) >= 50   # pushed down


def test_tropical_clubbtop_resets_aloft_and_spreads_water_above_150_hpa():
    """Tropopause near 100 hPa: water is reset (zero tendency) at and above
    clubbtop and the removed water lands uniformly per unit mass on EVERY
    level below it -- including those between clubbtop and 150 hPa, which
    the fixed 150 hPa set never reached."""
    c = _column(0.0)
    m = _spun_up_moments(c)
    troplev = cam_chem_tropopause_level(c["T"], c["p_full"], c["p_half"], c["lat"])
    _, _, _, dqv_on, _, d_on = _step(c, _ON, m, troplev=troplev)
    _, _, _, dqv_off, _, d_off = _step(c, _OFF, m)
    ct = np.asarray(d_on["clubbtop"])
    p_full = np.asarray(c["p_full"])
    k = np.arange(p_full.shape[1])[None, :]
    above, below = k <= ct[:, None], k > ct[:, None]
    assert np.all(ct >= np.asarray(troplev))
    assert np.all((below & (p_full < _CUT_PA)).sum(axis=1) >= 2), ct
    on = np.asarray(dqv_on + d_on["dq_c_dt"])
    raw_v, raw = np.asarray(dqv_off), np.asarray(dqv_off + d_off["dq_c_dt"])
    assert np.abs(raw[above]).max() > 1.0e-16       # CLUBB did move water there
    assert np.all(on[above] == 0.0)                 # CAM 2099-2103 reset
    dp_g = np.asarray(c["p_half"][:, 1:] - c["p_half"][:, :-1]) / constants.g
    w = np.asarray(_taper(c["p_full"]))
    removed = np.sum(dp_g * (1.0 - w * below) * raw, axis=1)
    share = removed / np.sum(dp_g * below, axis=1)
    assert np.all(np.abs(share) > 0.0)
    np.testing.assert_allclose(
        (np.asarray(dqv_on) - w * below * raw_v)[below],
        np.broadcast_to(share[:, None], below.shape)[below], rtol=1e-9, atol=1e-22)


def test_polar_clubbtop_resets_water_below_150_hpa():
    """Poleward of 50 deg the high tropopause is rejected and the formula
    puts clubbtop near 230 hPa: between 150 hPa and there CLUBB runs at full
    weight, yet its water tendency is reset, as in CAM."""
    c = _column(70.0)
    m = _spun_up_moments(c)
    troplev = cam_chem_tropopause_level(c["T"], c["p_full"], c["p_half"], c["lat"])
    _, _, _, dqv_on, _, d_on = _step(c, _ON, m, troplev=troplev)
    _, _, _, dqv_off, _, d_off = _step(c, _OFF, m)
    ct = np.asarray(d_on["clubbtop"])
    p_full = np.asarray(c["p_full"])
    k = np.arange(p_full.shape[1])[None, :]
    band = (k <= ct[:, None]) & (p_full > _CUT_PA)
    assert np.all(band.sum(axis=1) >= 2), ct
    w = np.asarray(_taper(c["p_full"]))
    assert w[band].min() > 0.5
    raw = np.asarray(dqv_off + d_off["dq_c_dt"])
    assert np.abs(raw[band]).max() > 1.0e-16
    assert np.all(np.asarray(dqv_on + d_on["dq_c_dt"])[band] == 0.0)


def test_energy_fixer_acts_only_below_clubbtop(monkeypatch):
    """CAM clubb_intr.F90:2138-2150: se_dis spread uniformly per unit mass
    over k = clubbtop+1..pver.  One CLUBB step (no sub-cycle), so the
    clubbtop of the matching clubb_step call is the one the fixer used."""
    c = _column(70.0)
    m = _spun_up_moments(c)
    seen = {}
    real = clubb_mod.column_energy_fixer_heating

    def rec(resid, dp_g, weight):
        seen["w"] = weight
        return real(resid, dp_g, weight)

    monkeypatch.setattr(clubb_mod, "column_energy_fixer_heating", rec)
    clubb_turbulence_prognostic(
        *_args(c), pack_clubb_moments(m), c["p_full"], c["p_half"],
        c["z_full"], c["z_half"], c["T_sfc"], c["q_sfc"], c["rho"], _DT, _ON,
        q_c=c["q_c"], lat=c["lat"])
    troplev = cam_chem_tropopause_level(c["T"], c["p_full"], c["p_half"], c["lat"])
    ct = np.asarray(_step(c, _ON, m, troplev=troplev)[5]["clubbtop"])
    k = np.arange(c["T"].shape[1])[None, :]
    np.testing.assert_array_equal(np.asarray(seen["w"]),
                                  (k > ct[:, None]).astype(float))
    assert np.any((k > ct[:, None]) != (np.asarray(c["p_full"]) > _CUT_PA))


def test_bottom_layer_tropopause_keeps_a_nonzero_spread_depth():
    """CAM leaves troplev = pver unclamped and divides by a zero depth; here
    clubbtop stops at pver-1, so everything returns into the bottom layer and
    the column water is still conserved."""
    c = _column(0.0)
    m = _spun_up_moments(c)
    ncol, nlev = c["T"].shape
    out_on = _step(c, _ON, m, troplev=jnp.full((ncol,), nlev - 1))
    out_off = _step(c, _OFF, m)
    assert np.all(np.asarray(out_on[5]["clubbtop"]) == nlev - 2)
    for x in out_on[:4]:
        assert np.all(np.isfinite(np.asarray(x)))
    on = np.asarray(out_on[3] + out_on[5]["dq_c_dt"])
    raw = np.asarray(out_off[3] + out_off[5]["dq_c_dt"])
    assert np.all(on[:, :-1] == 0.0)
    dp_g = np.asarray(c["p_half"][:, 1:] - c["p_half"][:, :-1]) / constants.g
    np.testing.assert_allclose(np.sum(dp_g * on, axis=1),
                               np.sum(dp_g * raw, axis=1), rtol=1e-10)


def test_gradient_through_a_clubb_column_stays_finite():
    c = _column(60.0)
    ncol, nlev = c["T"].shape
    carry = pack_clubb_moments(init_clubb_moments(ncol, nlev, _ON))
    dp_g = (c["p_half"][:, 1:] - c["p_half"][:, :-1]) / constants.g

    def loss(T):
        out, _ = clubb_turbulence_prognostic(
            c["u"], c["v"], T, c["q_v"], carry, c["p_full"], c["p_half"],
            c["z_full"], c["z_half"], c["T_sfc"], c["q_sfc"], c["rho"],
            2.0 * _DT, _ON, q_c=c["q_c"], lat=c["lat"])
        return jnp.sum(dp_g * (out.dq_v_dt + out.dq_c_dt + 1.0e-3 * out.dT_dt))

    g = np.asarray(jax.grad(loss)(c["T"]))
    assert np.all(np.isfinite(g)) and np.abs(g).max() > 0.0


def test_missing_latitude_or_troplev_raises():
    c = _column(0.0)
    ncol, nlev = c["T"].shape
    carry = pack_clubb_moments(init_clubb_moments(ncol, nlev, _ON))
    with pytest.raises(ValueError, match="latitudes"):
        clubb_turbulence_prognostic(
            *_args(c), carry, c["p_full"], c["p_half"], c["z_full"],
            c["z_half"], c["T_sfc"], c["q_sfc"], c["rho"], _DT, _ON)
    with pytest.raises(ValueError, match="latitudes"):
        integrate_clubb_column(
            *_args(c), c["p_full"], c["p_half"], c["z_full"], c["z_half"],
            c["T_sfc"], c["q_sfc"], _DT, 1, _ON)
    with pytest.raises(ValueError, match="troplev"):
        _step(c, _ON, unpack_clubb_moments(carry))


# --------------------------------------------------------------------------
# Every lane hands prognostic CLUBB its own latitudes
# --------------------------------------------------------------------------
def _record_lat(monkeypatch):
    seen = {}
    real = clubb_mod.clubb_turbulence_prognostic

    def rec(*a, lat=None, **k):
        seen["lat"] = lat
        return real(*a, lat=lat, **k)

    monkeypatch.setattr(clubb_mod, "clubb_turbulence_prognostic", rec)
    return seen


def _turb_cfg():
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    return TurbulenceConfig(scheme="clubb", clubb=_ON)


def test_cube_lane_passes_grid_latitudes(monkeypatch):
    from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
    from legoesm.atmosphere.physics.turbulence.integration import (
        make_turbulence_physics,
    )
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_sigma_coordinate
    seen = _record_lat(monkeypatch)
    grid, sigma = create_cubed_sphere(4), create_sigma_coordinate(10)
    fn = make_turbulence_physics(_turb_cfg(), model_type="hydrostatic", dt=_DT)
    fn(held_suarez_init(grid, sigma), grid, sigma)
    np.testing.assert_array_equal(np.asarray(seen["lat"]),
                                  np.asarray(grid.lat).reshape(-1))


def test_mpas_lane_passes_cell_latitudes(monkeypatch):
    from legoesm.atmosphere.forcing.idealized.held_suarez import (
        held_suarez_init_mpas,
    )
    from legoesm.atmosphere.physics.turbulence.integration import (
        _make_mpas_turbulence,
    )
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh
    seen = _record_lat(monkeypatch)
    mesh = create_voronoi_mesh(subdivision_level=1, lloyd_iterations=2)
    sigma = create_sigma_coordinate(8, sigma_top=0.05)
    state = held_suarez_init_mpas(mesh, sigma, T_init=290.0,
                                  perturbation_amplitude=0.0)
    _make_mpas_turbulence(_turb_cfg(), _DT)(state, mesh, sigma)
    np.testing.assert_array_equal(np.asarray(seen["lat"]),
                                  np.asarray(mesh.latCell))


def test_spectral_lane_passes_gaussian_latitudes(monkeypatch):
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
        isothermal_rest_state_spectral,
    )
    from legoesm.atmosphere.physics.combined import PhysicsConfig
    from legoesm.atmosphere.physics.physics_state import init_physics_state
    from legoesm.atmosphere.physics.turbulence.integration import (
        make_turbulence_physics,
    )
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    seen = _record_lat(monkeypatch)
    grid, sigma = create_gaussian_grid(8, dealiasing="quadratic"), create_sigma_coordinate(8)
    state = isothermal_rest_state_spectral(grid, sigma, T_init=280.0, p_s_init=1.0e5)
    ncol = int(grid.n_lat) * int(grid.n_lon)
    ps = init_physics_state(ncol, 8, PhysicsConfig(turbulence=_turb_cfg()))
    make_turbulence_physics(_turb_cfg(), "spectral_pe", dt=_DT)(
        state, grid, sigma, phys_state=ps)
    lat = np.asarray(seen["lat"]).reshape(int(grid.n_lat), int(grid.n_lon))
    np.testing.assert_array_equal(lat, np.broadcast_to(
        np.asarray(grid.lat)[:, None], lat.shape))
