"""CAM-FV PPM (ppm2m, kord 4 and 7) flux-form vertical transport on the
hybrid lane: pass tests of results/legoesm/scm/ppm_remap_proposal.md
section 5 that run as unit tests.

Oracle: the unmodified CAM-FV map1_ppm (cam_cesm2_1_rel_60), run offline by
scripts/validate/oracle/cam_fv_ppm2m/make_fixture.py into
tests/grids/oracle_fixtures/cam_fv_ppm2m.npz (12 C1 columns, 12 RF02 states).
"""
from __future__ import annotations

import importlib.util
import json
import math
import pathlib

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from legoesm import constants
from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import MPASPrimitiveEquationConfig
from legoesm.grids.vertical import (
    compute_mass_flux_from_cumsum, dp_from_hybrid,
    make_cam6_l32_levels, ppm2m_profile, ppm_flux_tendency,
    ppm_swept_face_values, ppm_tv_lnp_start, ppm_tv_lnp_tendency,
    vertical_advection_hybrid_ppm,
    vertical_advection_tv_hybrid_ppm)

FIXTURE = pathlib.Path(__file__).with_name("oracle_fixtures") / "cam_fv_ppm2m.npz"
DEFAULT_SHARPNESS = MPASPrimitiveEquationConfig._field_defaults["vert_ppm_limiter_sharpness"]
HARD = math.inf


def _kind(name):
    return (1, 4) if name == "theta" else (0, 7)          # (iv, kord)


def _cases():
    d = np.load(FIXTURE)
    meta = json.loads(str(d["meta"]))
    out = []
    for j, c in enumerate(meta["columns"]):
        for nm in c["fields"]:
            out.append(dict(col=j, group=c["group"], name=nm, edges=d[f"col{j}_edges"],
                            F=d[f"col{j}_F"], q=d[f"col{j}_{nm}"],
                            sub=d[f"col{j}_{nm}_sub"], face=d[f"col{j}_{nm}_face"],
                            dt=float(d["dt"])))
    return out


CASES = _cases()


def _profile(c, sharpness):
    iv, kord = _kind(c["name"])
    dp = jnp.asarray(np.diff(c["edges"]))
    return tuple(np.asarray(x[0]) for x in ppm2m_profile(
        jnp.asarray(c["q"])[None], dp, iv=iv, kord=kord, sharpness=sharpness))


def _sub_means(al, ar, a6):
    """Means of each parabola over thirds of its layer (map1_ppm's formula)."""
    out = []
    for k in range(al.size):
        for pl, pr in ((0.0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.0)):
            out.append(al[k] + 0.5 * (a6[k] + ar[k] - al[k]) * (pr + pl)
                       - a6[k] / 3.0 * (pr * (pr + pl) + pl ** 2))
    return np.array(out)


def _face_check(c, sharpness):
    """Our swept-mass face values against CAM's remap to edges moved upstream
    by F dt (see make_fixture.py); faces with Courant >= 1e-3 only."""
    e, F, q, dt = c["edges"], c["F"], c["q"], c["dt"]
    dp = np.diff(e)
    al, ar, a6 = _profile(c, sharpness)
    face = np.asarray(ppm_swept_face_values(
        *(jnp.asarray(x)[None] for x in (al, ar, a6)), jnp.asarray(F), jnp.asarray(dp), dt))[0]
    G = np.cumsum(q * dp - c["face"] * (dp - dt * (F[1:] - F[:-1]))) / dt
    ok = np.abs(F[1:-1]) * dt / np.minimum(dp[:-1], dp[1:]) >= 1e-3
    return face[1:-1][ok], G[:-1][ok] / F[1:-1][ok]


def _gaps(c, sharpness):
    scale = np.max(np.abs(c["q"]))
    sub = np.max(np.abs(_sub_means(*_profile(c, sharpness)) - c["sub"])) / scale
    ours, cam = _face_check(c, sharpness)
    return sub, np.max(np.abs(ours - cam)) / scale


def _ids():
    return [f"{c['group']}{c['col']}-{c['name']}" for c in CASES]


# --------------------------------------------------------------- 1. oracle
@pytest.mark.parametrize("c", CASES, ids=_ids())
def test_hard_limit_parabolas_match_cam_fv_ppm2m(c):
    """sharpness = inf: CAM-FV's kord 4 / kord 7 parabolas to 1e-12 of the
    column maximum (proposal 5.1, reconstruction check)."""
    sub, _ = _gaps(c, HARD)
    assert sub <= 1e-12


@pytest.mark.parametrize("c", CASES, ids=_ids())
def test_hard_limit_face_values_match_cam_fv_swept_mass(c):
    """sharpness = inf: the flux-form face values equal CAM's remap of the same
    data over the swept masses, Courant up to 0.05, to 1e-6 relative
    (proposal 5.1, face-value check)."""
    _, face = _gaps(c, HARD)
    assert face <= 1e-6


# --------------------------------------- 1b. the temperature path (te_map)
COLS = sorted({c["col"] for c in CASES})


def _tv_case(j):
    d = np.load(FIXTURE)
    return dict(edges=d[f"col{j}_edges"], F=d[f"col{j}_F"], T=d[f"col{j}_T"],
                q_v=d[f"col{j}_q_v"], tv_star=d[f"col{j}_tv_star"],
                sub=d[f"col{j}_tv_star_sub"], face=d[f"col{j}_tv_star_face"],
                dt=float(d["dt"]))


@pytest.mark.parametrize("j", COLS)
def test_tv_star_is_te_map_pt_times_lagrangian_pkz(j):
    """The temperature the PPM path maps is te_map's: T_v times pkz of the
    Lagrangian edges over pkz of the Eulerian ones (the fixture computes it
    in NumPy, independently)."""
    c = _tv_case(j)
    tvs, _, _ = ppm_tv_lnp_start(jnp.asarray(c["T"]), jnp.asarray(c["q_v"]),
                                 jnp.asarray(c["edges"]), jnp.asarray(c["F"]), c["dt"])
    assert float(jnp.max(jnp.abs(tvs - c["tv_star"]))) <= 1e-14 * float(np.max(c["tv_star"]))


def _oracle_T_tendency(c, dq_v):
    """te_map's temperature step from the map1_ppm output: face fluxes in
    ln p recovered from the remap to upstream-moved edges, then the
    face-minus-cell increment and the division by 1 + zvir q_v of the vapour
    after its step, q_v + dt dq_v."""
    e, F, dt, tvs = c["edges"], c["F"], c["dt"], c["tv_star"]
    x, xt = np.log(e), np.log(e - F * dt)
    dx, dxt = np.diff(x), np.diff(xt)
    G = np.concatenate([[0.0], np.cumsum(tvs * dx - c["face"] * dxt)])   # w * face
    w = x - xt
    incr = -((G[1:] - G[:-1]) - tvs * (w[1:] - w[:-1])) / dx
    zvir = 1.0 / constants.epsilon - 1.0
    return ((tvs + incr) / (1.0 + zvir * (c["q_v"] + dt * dq_v)) - c["T"]) / dt, G, w


@pytest.mark.parametrize("j", COLS)
def test_hard_limit_T_path_parabolas_match_map1_ppm_on_log_pe(j):
    """kord 4, iv 1 parabolas of tv_star over ln pe equal map1_ppm's on log pe
    to 1e-12 of the column maximum."""
    c = _tv_case(j)
    dx = jnp.asarray(np.diff(np.log(c["edges"])))
    al, ar, a6 = (np.asarray(x[0]) for x in ppm2m_profile(
        jnp.asarray(c["tv_star"])[None], dx, iv=1, kord=4, sharpness=HARD))
    err = np.max(np.abs(_sub_means(al, ar, a6) - c["sub"])) / np.max(np.abs(c["tv_star"]))
    assert err <= 1e-12


@pytest.mark.parametrize("moist_step", [False, True])
@pytest.mark.parametrize("j", COLS)
def test_hard_limit_T_path_matches_map1_ppm_on_log_pe(j, moist_step):
    """The model's temperature path at the hard limit against te_map's remap
    of tv_star on log pe (map1_ppm, Courant up to 0.05): face values to 1e-6
    of the column maximum, the whole temperature tendency to 1e-6 of its
    maximum; with and without a vapour tendency in the T_v to T division."""
    c = _tv_case(j)
    dt = c["dt"]
    dq_v = (1e-7 * np.sin(np.arange(c["T"].size)) if moist_step
            else np.zeros(c["T"].shape))
    want, G, w = _oracle_T_tendency(c, dq_v)
    tvs, x, wj = ppm_tv_lnp_start(jnp.asarray(c["T"]), jnp.asarray(c["q_v"]),
                                  jnp.asarray(c["edges"]), jnp.asarray(c["F"]), dt)
    dx = jnp.diff(x)
    al, ar, a6 = ppm2m_profile(tvs[None], dx, iv=1, kord=4, sharpness=HARD)
    face = np.asarray(ppm_swept_face_values(al, ar, a6, wj, dx, 1.0))[0]
    ok = np.abs(w[1:-1]) / np.minimum(np.diff(np.log(c["edges"]))[:-1],
                                      np.diff(np.log(c["edges"]))[1:]) >= 1e-3
    err_face = np.max(np.abs(face[1:-1][ok] - G[1:-1][ok] / w[1:-1][ok])) / np.max(c["tv_star"])
    assert err_face <= 1e-6
    got = np.asarray(ppm_tv_lnp_tendency(
        jnp.asarray(c["T"]), jnp.asarray(c["q_v"]), jnp.asarray(dq_v),
        jnp.asarray(c["F"]), jnp.asarray(c["edges"]), dt, sharpness=HARD))
    assert np.max(np.abs(got - want)) <= 1e-6 * np.max(np.abs(want))


@pytest.mark.parametrize("j", COLS)
def test_hybrid_lane_T_path_matches_map1_ppm_on_log_pe(j):
    """The hybrid-lane entry the dycore calls (vertical_advection_tv_hybrid_ppm)
    against the same te_map oracle, on a pure-pressure coordinate whose
    edges are the fixture column's: fails if the lane's temperature path is
    the theta form."""
    from legoesm.grids.vertical import create_hybrid_coordinate
    c = _tv_case(j)
    km = c["T"].size
    # A_half in Pa with p_ref = 1, so the coordinate's edges are the
    # fixture's bit for bit (A / p_ref * p_ref is not)
    coord = create_hybrid_coordinate(km, jnp.asarray(c["edges"]), jnp.zeros(km + 1), 1.0)
    dq_v = 1e-7 * np.sin(np.arange(km))
    want, _, _ = _oracle_T_tendency(c, dq_v)
    got = np.asarray(vertical_advection_tv_hybrid_ppm(
        jnp.asarray(c["T"])[None], jnp.asarray(c["q_v"])[None], jnp.asarray(dq_v)[None],
        jnp.asarray(c["F"])[None], jnp.asarray([c["edges"][-1]]), coord, c["dt"],
        sharpness=HARD))[0]
    assert np.max(np.abs(got - want)) <= 1e-6 * np.max(np.abs(want))


def test_T_path_step_zero_is_the_limit_of_small_steps():
    """dt = 0 returns the derivative in the step length (method-of-lines
    limit), which a small finite step approaches.  At the default sharpness:
    at the hard limit a limiter branch (kmppm's flat-cell test) can flip
    between the two steps at the RF02 inversion."""
    c = _tv_case(COLS[-1])
    args = (jnp.asarray(c["T"]), jnp.asarray(c["q_v"]), jnp.zeros(c["T"].shape),
            jnp.asarray(c["F"]), jnp.asarray(c["edges"]))
    t0 = ppm_tv_lnp_tendency(*args, 0.0, sharpness=DEFAULT_SHARPNESS)
    t1 = ppm_tv_lnp_tendency(*args, 1e-3, sharpness=DEFAULT_SHARPNESS)
    assert bool(jnp.all(jnp.isfinite(t0)))
    assert float(jnp.max(jnp.abs(t0 - t1))) <= 1e-4 * float(jnp.max(jnp.abs(t0)))


# Measured mismatch to CAM-FV at the default sharpness 1e4, per fixture
# column and field: (parabola sub-layer means, face values), relative to the
# column maximum.  Re-measure when the smooth form or its default changes.
GAPS_DEFAULT = [
    (1.5903674766e-16, 6.0274927362e-14),  # c1 col 0 theta
    (1.0204871699e-05, 2.2885833434e-05),  # c1 col 0 q_v
    (5.6162657363e-07, 1.8330949426e-09),  # c1 col 0 q_c
    (3.2180020460e-12, 6.9349791972e-12),  # c1 col 0 q_r
    (7.9575754352e-17, 5.3634058433e-14),  # c1 col 1 theta
    (1.3127648917e-07, 2.6812569803e-07),  # c1 col 1 q_v
    (8.6389999258e-16, 9.9866839142e-14),  # c1 col 1 q_c
    (1.7122792014e-15, 2.0642477039e-14),  # c1 col 1 q_r
    (7.9281944449e-17, 2.4577402779e-14),  # c1 col 2 theta
    (8.0008719044e-09, 1.7027718291e-08),  # c1 col 2 q_v
    (1.9253320787e-14, 3.8720567361e-14),  # c1 col 2 q_c
    (1.1519508754e-12, 8.8778212794e-13),  # c1 col 2 q_r
    (8.9989420425e-07, 1.8389563433e-06),  # c1 col 3 theta
    (5.8809628322e-09, 8.6533416758e-11),  # c1 col 3 q_v
    (2.0057958147e-05, 1.0529475132e-06),  # c1 col 3 q_c
    (8.1695631937e-16, 7.0024827375e-14),  # c1 col 3 q_r
    (1.5757622133e-16, 1.8676721633e-13),  # c1 col 4 theta
    (4.0763458559e-09, 3.6093212096e-11),  # c1 col 4 q_v
    (8.4376324755e-16, 1.8791812899e-13),  # c1 col 4 q_c
    (1.3142322303e-14, 4.0699731704e-14),  # c1 col 4 q_r
    (1.5963453445e-16, 1.0376244740e-14),  # c1 col 5 theta
    (1.1960493141e-05, 2.6294185177e-05),  # c1 col 5 q_v
    (9.0672207257e-11, 1.8699407901e-10),  # c1 col 5 q_c
    (1.3799538572e-15, 3.4038861810e-14),  # c1 col 5 q_r
    (2.9168899823e-14, 5.3938685679e-14),  # c1 col 6 theta
    (1.8291747676e-05, 3.6870337090e-05),  # c1 col 6 q_v
    (3.5786576196e-16, 9.3850296074e-14),  # c1 col 6 q_c
    (5.1061311598e-13, 7.8886143051e-13),  # c1 col 6 q_r
    (7.9311293740e-17, 1.7583313822e-13),  # c1 col 7 theta
    (2.0524517227e-05, 1.6159533063e-13),  # c1 col 7 q_v
    (1.4154510052e-15, 3.1611739117e-14),  # c1 col 7 q_c
    (1.4588308088e-16, 2.3885923110e-14),  # c1 col 7 q_r
    (1.5893416034e-16, 1.9151566321e-14),  # c1 col 8 theta
    (1.3703826290e-06, 2.2983455773e-14),  # c1 col 8 q_v
    (1.1869828739e-15, 3.2932180402e-13),  # c1 col 8 q_c
    (2.1504877592e-15, 5.0894876968e-14),  # c1 col 8 q_r
    (3.2924025525e-09, 7.3373988572e-09),  # c1 col 9 theta
    (2.1326164282e-05, 1.9434843267e-06),  # c1 col 9 q_v
    (1.1221710401e-09, 2.2875571964e-09),  # c1 col 9 q_c
    (9.4515178592e-16, 1.3238426015e-13),  # c1 col 9 q_r
    (1.5746335433e-16, 1.1258629834e-13),  # c1 col 10 theta
    (3.5701158420e-07, 2.9979833598e-08),  # c1 col 10 q_v
    (1.5512203019e-15, 4.4125166586e-13),  # c1 col 10 q_c
    (2.3647979766e-15, 3.4289570661e-14),  # c1 col 10 q_r
    (1.5723563313e-16, 1.0660575926e-13),  # c1 col 11 theta
    (6.7285734837e-06, 6.7518910048e-07),  # c1 col 11 q_v
    (2.3358071437e-15, 6.4113878841e-14),  # c1 col 11 q_c
    (2.0535206216e-15, 6.6611075163e-14),  # c1 col 11 q_r
    (3.7585415911e-05, 8.8802690733e-05),  # rf02 col 12 theta
    (2.1040502363e-08, 1.4399445843e-09),  # rf02 col 12 q_v
    (2.5205761258e-16, 3.0080462452e-13),  # rf02 col 12 q_c
    (2.8721718649e-05, 3.2343707883e-05),  # rf02 col 13 theta
    (7.1317105607e-16, 3.3340746871e-14),  # rf02 col 13 q_v
    (2.2073077968e-16, 3.3349024951e-13),  # rf02 col 13 q_c
    (1.6285441054e-05, 2.7348602140e-05),  # rf02 col 14 theta
    (7.1110814036e-16, 3.6088738123e-14),  # rf02 col 14 q_v
    (1.7141603176e-16, 2.2507962841e-13),  # rf02 col 14 q_c
    (5.4072038090e-05, 3.4325445940e-05),  # rf02 col 15 theta
    (2.0089756363e-05, 3.0206680539e-13),  # rf02 col 15 q_v
    (2.0741119821e-16, 6.8984630319e-13),  # rf02 col 15 q_c
    (5.3434040464e-05, 3.7021882001e-05),  # rf02 col 16 theta
    (7.9453842329e-14, 4.0711305052e-13),  # rf02 col 16 q_v
    (7.2587513292e-16, 1.3727851707e-12),  # rf02 col 16 q_c
    (5.3510173431e-05, 9.9262399758e-05),  # rf02 col 17 theta
    (7.8884188234e-16, 7.7657100861e-14),  # rf02 col 17 q_v
    (2.6657853080e-16, 2.2205991615e-13),  # rf02 col 17 q_c
    (5.3524454088e-05, 9.9767113486e-05),  # rf02 col 18 theta
    (5.9662488844e-15, 1.0791891364e-13),  # rf02 col 18 q_v
    (7.6103580866e-17, 2.1131182339e-13),  # rf02 col 18 q_c
    (5.3175452270e-05, 9.3215338845e-05),  # rf02 col 19 theta
    (2.6731509824e-12, 1.5062979665e-13),  # rf02 col 19 q_v
    (1.2855311955e-16, 1.9450086988e-13),  # rf02 col 19 q_c
    (4.7355604533e-05, 8.7105117875e-05),  # rf02 col 20 theta
    (4.3214350049e-07, 9.6642761636e-07),  # rf02 col 20 q_v
    (1.5705958533e-13, 1.2849165096e-13),  # rf02 col 20 q_c
    (2.9429157831e-05, 3.2543586025e-05),  # rf02 col 21 theta
    (1.9840026161e-05, 4.1825320360e-05),  # rf02 col 21 q_v
    (2.7308604452e-08, 5.7569961537e-08),  # rf02 col 21 q_c
    (2.4452555006e-05, 2.6749852790e-05),  # rf02 col 22 theta
    (3.0252788494e-06, 6.2103064059e-14),  # rf02 col 22 q_v
    (2.4257194605e-06, 1.9602812254e-13),  # rf02 col 22 q_c
    (2.7894619988e-05, 5.7369887517e-05),  # rf02 col 23 theta
    (2.8000738344e-08, 6.1713655847e-08),  # rf02 col 23 q_v
    (1.7064855229e-05, 3.7610928808e-05),  # rf02 col 23 q_c
]


def test_smooth_default_gap_is_the_recorded_one():
    """RULE 5: at the default sharpness, each case's mismatch to CAM-FV equals
    the stored measurement (1e-6 relative, 1e-12 absolute floor)."""
    assert DEFAULT_SHARPNESS == 1.0e4, "default moved: re-measure GAPS_DEFAULT"
    for c, (sub0, face0) in zip(CASES, GAPS_DEFAULT):
        sub, face = _gaps(c, DEFAULT_SHARPNESS)
        for got, ref in ((sub, sub0), (face, face0)):
            assert abs(got - ref) <= max(1e-6 * abs(ref), 1e-12), (c["col"], c["name"], got, ref)


def test_gradient_through_the_smooth_limiter():
    """RULE 5: finite second-order gradients at the default sharpness (check
    against finite differences), and the smooth switch is on the gradient
    path: the derivative differs from the hard limit's on an RF02 column
    whose mixed layer puts limiter comparisons inside the transition."""
    c = next(x for x in CASES if x["group"] == "rf02" and x["name"] == "q_v")
    dp = jnp.asarray(np.diff(c["edges"]))
    F = jnp.asarray(c["F"])

    def tend(q, s):
        return jnp.sum(ppm_flux_tendency(q, F, dp, c["dt"], kord=7, iv=0, sharpness=s) ** 2)

    q = jnp.asarray(c["q"])
    check_grads(lambda x: tend(x, DEFAULT_SHARPNESS), (q,), order=2, eps=1e-9)
    g_s = jax.grad(tend)(q, DEFAULT_SHARPNESS)
    g_h = jax.grad(tend)(q, HARD)
    assert bool(jnp.all(jnp.isfinite(g_s))) and float(jnp.max(jnp.abs(g_s))) > 0
    assert float(jnp.max(jnp.abs(g_s - g_h))) > 1e-6 * float(jnp.max(jnp.abs(g_h)))


def test_gradient_through_the_temperature_path():
    """The heat path (T_v over ln p, kord 4) is differentiable at the default
    sharpness, in T and in the vapour."""
    coord, p_s, F, q = _l32(5, ncol=1)
    T = jnp.asarray(np.linspace(210.0, 295.0, coord.A_full.shape[0]))[None]
    dq = vertical_advection_hybrid_ppm(q, F, p_s, coord, 300.0, kord=7, iv=0,
                                       sharpness=DEFAULT_SHARPNESS)
    f = lambda x, qv: jnp.sum(vertical_advection_tv_hybrid_ppm(  # noqa: E731
        x, qv, dq, F, p_s, coord, 300.0, sharpness=DEFAULT_SHARPNESS) ** 2)
    check_grads(f, (T, q), order=1, eps=1e-7)


def test_temperature_path_refuses_a_zero_pressure_top():
    from legoesm.grids.vertical import create_hybrid_coordinate
    nlev = 8
    coord = create_hybrid_coordinate(nlev, jnp.zeros(nlev + 1),
                                     jnp.linspace(0.0, 1.0, nlev + 1), constants.p_ref)
    p_s = jnp.array([1.0e5])
    z = jnp.zeros((1, nlev))
    with pytest.raises(ValueError, match="model top above zero pressure"):
        vertical_advection_tv_hybrid_ppm(z + 250.0, z, z, jnp.zeros((1, nlev + 1)),
                                         p_s, coord, 300.0, sharpness=HARD)


# ------------------------------------------------- 2. constants, conservation
def _l32(seed, ncol=6):
    """CAM L32 columns with varying p_s and a continuity-built mass flux
    (as tests/unit/test_vertical_advection_hybrid_van_leer.py)."""
    rng = np.random.default_rng(seed)
    coord = make_cam6_l32_levels()
    nlev = coord.A_full.shape[0]
    p_s = jnp.asarray(1.0e5 + 3.0e3 * rng.standard_normal(ncol))
    div = jnp.asarray(1.0e-5 * rng.standard_normal((ncol, nlev)))
    cums = jnp.cumsum(div * dp_from_hybrid(coord, p_s), axis=-1)
    F = compute_mass_flux_from_cumsum(cums, cums[..., -1:], coord)
    q = jnp.asarray(1.0e-2 * np.exp(-0.5 * ((np.arange(nlev) - 24) / 3.0) ** 2)
                    * (1.0 + 0.3 * rng.random((ncol, nlev))))
    return coord, p_s, F, q


@pytest.mark.parametrize("sharpness", [HARD, DEFAULT_SHARPNESS])
@pytest.mark.parametrize("iv,kord", [(1, 4), (0, 7)])
def test_constant_field_stays_constant(iv, kord, sharpness):
    coord, p_s, F, _ = _l32(1)
    c = jnp.full((p_s.shape[0], coord.A_full.shape[0]), 7.3e-3)
    t = vertical_advection_hybrid_ppm(c, F, p_s, coord, 300.0, kord=kord, iv=iv,
                                      sharpness=sharpness)
    rate = float(jnp.max(jnp.abs(F))) * 7.3e-3 / float(jnp.min(dp_from_hybrid(coord, p_s)))
    assert float(jnp.max(jnp.abs(t))) <= 1e-14 * rate


@pytest.mark.parametrize("sharpness", [HARD, DEFAULT_SHARPNESS])
@pytest.mark.parametrize("iv,kord", [(1, 4), (0, 7)])
def test_column_mass_is_conserved(iv, kord, sharpness):
    """sum_k d(q dp)/dt = sum_k (dp dq/dt - q (F_{k+1/2} - F_{k-1/2})) = 0
    to 1e-14 of the interfacial exchange (proposal 5.3)."""
    coord, p_s, F, q = _l32(0)
    dp = dp_from_hybrid(coord, p_s)
    t = vertical_advection_hybrid_ppm(q, F, p_s, coord, 300.0, kord=kord, iv=iv,
                                      sharpness=sharpness)
    col = jnp.sum(dp * t - q * (F[..., 1:] - F[..., :-1]), axis=-1)
    scale = jnp.sum(jnp.abs(F[..., 1:-1] * jnp.diff(q, axis=-1)), axis=-1)
    assert float(jnp.max(jnp.abs(col) / scale)) <= 1e-14


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_kord7_column_stays_in_range_under_steady_transport(sign):
    """A non-negative, non-constant column under steady descent (ascent) for
    2000 forward steps at a face Courant number of 0.4 stays inside its
    starting range: the boundary layers' cubic and constraint are stable
    (the van Leer operator needed donor faces there)."""
    coord = make_cam6_l32_levels()
    nlev = coord.A_full.shape[0]
    p_s = jnp.array([1.0e5])
    dp = dp_from_hybrid(coord, p_s)
    q0 = jnp.asarray(1.13 + 0.86 * np.random.default_rng(3).random((1, nlev)))
    F = jnp.concatenate([jnp.zeros((1, 1)), jnp.full((1, nlev - 1), sign),
                         jnp.zeros((1, 1))], axis=-1)
    dt = 0.4 / float(jnp.max((jnp.abs(F[..., :-1]) + jnp.abs(F[..., 1:])) / dp))
    step = jax.jit(lambda x: x + dt * vertical_advection_hybrid_ppm(
        x, F, p_s, coord, dt, kord=7, iv=0, sharpness=HARD))
    q = q0
    for _ in range(2000):
        q = step(q)
    assert float(jnp.min(q)) >= float(jnp.min(q0)) - 1e-12
    assert float(jnp.max(q)) <= float(jnp.max(q0)) + 1e-12


# -------------------------------------------------------- 4. the MPAS lane
def _mpas_helpers():
    path = pathlib.Path(__file__).parents[1] / "unit" / "test_mpas_atmosphere.py"
    spec = importlib.util.spec_from_file_location("_mpas_atm_helpers_ppm", path)
    h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)
    return h


def _mpas_state():
    from legoesm.core.field import Field
    h = _mpas_helpers()
    coord = make_cam6_l32_levels()
    nlev = coord.A_full.shape[0]
    mesh = h._make_mesh(level=2)
    state = h._add_perturbation_hydro(h._make_hydrostatic_state(mesh, nlev), mesh, nlev)
    qv = jnp.asarray(np.linspace(1e-6, 1.5e-2, nlev)[None, :].repeat(mesh.nCells, 0))
    tr = {"q_v": Field(data=qv, name="q_v", dims=("nCells", "nlev"), units="kg/kg")}
    return mesh, coord, state._replace(tracers=tr)


def test_mpas_lane_runs_ppm_for_tracers(monkeypatch):
    """Names the symbol the MPAS hybrid lane calls under scheme 'ppm': q_v
    with kord 7, iv 0.  Fails if the lane goes back to the van Leer
    operator.  (Temperature: next test.)"""
    from legoesm.atmosphere.dynamics.gcm import primitive_eq_mpas as pe
    from legoesm.grids import vertical as vt
    mesh, coord, state = _mpas_state()
    calls = []
    real = vt.vertical_advection_hybrid_ppm

    def spy(*a, **k):
        calls.append((k["kord"], k["iv"]))
        return real(*a, **k)

    monkeypatch.setattr(pe, "vertical_advection_hybrid_ppm", spy)
    monkeypatch.setattr(vt, "vertical_advection_hybrid_ppm", spy)
    cfg = pe.MPASPrimitiveEquationConfig(
        nu_del2=1.0e5, vert_advection_scheme="ppm")
    out = pe.mpas_legacy_pe_tendencies(state, mesh, coord, cfg, dt=300.0)
    assert (7, 0) in calls
    assert np.all(np.isfinite(np.asarray(out.tracer_tendencies["q_v"].data)))
    assert np.all(np.isfinite(np.asarray(out.dT_dt.data)))


def test_mpas_lane_temperature_is_the_te_map_path():
    """Under 'ppm' the lane's vertical temperature term (F part of advection
    plus adiabatic term) is exactly the T_v over ln p path that the oracle
    tests pin to map1_ppm, with the vapour's own kord 7 tendency; it fails
    with the theta form restored."""
    from legoesm.atmosphere.dynamics.gcm import primitive_eq_mpas as pe
    mesh, coord, state = _mpas_state()
    s = DEFAULT_SHARPNESS
    _, terms = pe.mpas_legacy_pe_tendencies(
        state, mesh, coord, pe.MPASPrimitiveEquationConfig(
            nu_del2=1.0e5, vert_advection_scheme="ppm"), dt=300.0,
        return_thermo_terms=True)
    # the same mass flux the lane builds (continuity of the flux-form div)
    p_s = state.p_s.data
    qv = state.tracers["q_v"].data
    _, terms_sb = pe.mpas_legacy_pe_tendencies(
        state, mesh, coord, pe.MPASPrimitiveEquationConfig(
            nu_del2=1.0e5, vert_advection_scheme="sb"), dt=300.0,
        return_thermo_terms=True)
    F = terms.sigma_dot * jnp.maximum(p_s[:, None], 1.0)
    dqv = vertical_advection_hybrid_ppm(qv, F, p_s, coord, 300.0, kord=7, iv=0, sharpness=s)
    want = vertical_advection_tv_hybrid_ppm(state.T.data, qv, dqv, F, p_s, coord, 300.0,
                                            sharpness=s)
    scale = float(jnp.max(jnp.abs(want)))
    assert float(jnp.max(jnp.abs(terms.vert_adv - want))) <= 1e-12 * scale
    assert float(jnp.max(jnp.abs(terms_sb.vert_adv - want))) > 1e-6 * scale


def test_mpas_lane_ppm_differs_from_sb_only_where_wired():
    """The flag changes the heat and tracer vertical transport and nothing
    else: the momentum and surface pressure tendencies are identical."""
    from legoesm.atmosphere.dynamics.gcm import primitive_eq_mpas as pe
    mesh, coord, state = _mpas_state()
    outs = {s: pe.mpas_legacy_pe_tendencies(
        state, mesh, coord, pe.MPASPrimitiveEquationConfig(
            nu_del2=1.0e5, vert_advection_scheme=s), dt=300.0) for s in ("sb", "ppm")}
    a, b = outs["sb"], outs["ppm"]
    assert bool(jnp.array_equal(a.du_dt.data, b.du_dt.data))
    assert bool(jnp.array_equal(a.dp_s_dt.data, b.dp_s_dt.data))
    assert not bool(jnp.array_equal(a.dT_dt.data, b.dT_dt.data))
    assert not bool(jnp.array_equal(a.tracer_tendencies["q_v"].data,
                                    b.tracer_tendencies["q_v"].data))


def test_ppm_is_refused_on_the_sigma_lane():
    from legoesm.atmosphere.dynamics.gcm import primitive_eq_mpas as pe
    from legoesm.grids.vertical import create_sigma_coordinate
    h = _mpas_helpers()
    mesh = h._make_mesh(level=2)
    state = h._make_hydrostatic_state(mesh, 8)
    sigma = create_sigma_coordinate(8)
    with pytest.raises(ValueError, match="HYBRID lane only"):
        pe.mpas_legacy_pe_tendencies(
            state, mesh, sigma, pe.MPASPrimitiveEquationConfig(vert_advection_scheme="ppm"))


@pytest.mark.parametrize("bad", [dict(kord=5), dict(iv=-1), dict(sharpness=0.0)])
def test_unported_options_raise(bad):
    kw = dict(kord=7, iv=0, sharpness=HARD) | bad
    with pytest.raises(ValueError):
        ppm2m_profile(jnp.ones((1, 1, 8)), jnp.ones((1, 8)), **kw)
