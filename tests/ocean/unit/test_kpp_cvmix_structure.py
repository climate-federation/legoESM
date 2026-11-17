"""KPP ``bld_structure="cvmix"`` against CVMix / MOM6 (oracle-pinned values).

Oracles: CVMix fce422195a (MOM6 0e9a561 submodule), driven standalone; MOM6
0e9a561 single-column KPP runs at its defaults.  Pins and their generator live in
the KPP oracle tier (oracle_audit/kpp_oracle/pins_rebuild.py).

RULE 5 (smooth first-crossing selection, ``crossing_sharpness``):
1. hard limit (sharpness 1e12) == CVMix compute_OBL_depth (quadratic), rel 1e-10;
2. default sharpness: per-case depth stored (the recorded gap to CVMix), rel 1e-6;
3. gradient through the switch at the default (check_grads).
Known difference: at Ri_b == Ri_c exactly the sigmoid gives 1/2, CVMix's strict
``>`` gives 0 (measure-zero; no case here sits on it).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from legoesm import constants
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.eos import rho_0, wright_eos
from legoesm.ocean.physics.vertical_mixing import kpp as K
from legoesm.ocean.physics.vertical_mixing.config import KPPConfig
from legoesm.ocean.vertical import create_z_star_from_thicknesses

CFG = KPPConfig(bld_structure="cvmix")
HARD = CFG._replace(crossing_sharpness=1e12)


@pytest.fixture(autouse=True)
def _x64_fp64_policy():
    x64 = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(prev)
        jax.config.update("jax_enable_x64", x64)


def _centres(dz):
    dz = np.asarray(dz, float)
    return np.cumsum(dz) - 0.5 * dz


# (dz [m], Ri_b at centres, CVMix OBL depth [m]); CVMix cfg: Ri_c 0.3, quadratic.
_U = [1.0] * 8
_CROSSING = {
    "first_cell": (_U, [0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 2.0], 0.5),
    "bracket1": (_U, [0.05, 0.5, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4], 1.24535599249993),
    "interior": (_U, [0.0, 0.02, 0.05, 0.1, 0.25, 0.6, 1.0, 1.5], 4.750000000000431),
    "concave": (_U, [0.0, 0.1, 0.25, 0.29, 0.31, 0.32, 0.33, 0.34], 3.7928932188126563),
    "none": (_U, [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25], 7.5),
    "multiple": (_U, [0.0, 0.1, 0.4, 0.2, 0.1, 0.5, 0.9, 1.2], 2.280776406404716),
    "stretched": ([1.0, 1.5, 2.0, 3.0, 4.5, 6.0, 8.0, 10.0],
                  [0.0, 0.03, 0.08, 0.15, 0.27, 0.45, 0.8, 1.3], 10.675837747823431),
}
# Port depth at the DEFAULT sharpness (20) per case: the recorded RULE-5 gap.
# Re-measured after the bracket end values became smooth clamps (continuity fix).
_DEFAULT_DEPTH = {
    "bracket1": 1.244438105647975,     # CVMix 1.24535599 -> gap -0.07 %
    "concave": 3.5441008216794967,     # CVMix 3.79289322 -> gap -6.56 %
    "first_cell": 0.5010296317418871,  # CVMix 0.5        -> gap +0.21 %
    "interior": 4.654105580899147,     # CVMix 4.75       -> gap -2.02 %
    "multiple": 2.5619802008046015,    # CVMix 2.28077641 -> gap +12.33 %
    "none": 7.144097939211112,         # CVMix 7.5        -> gap -4.75 %
    "stretched": 10.355106613637513,   # CVMix 10.6758377 -> gap -3.00 %
}


def _h(name, cfg):
    dz, ri, _ = _CROSSING[name]
    return float(K._cvmix_first_crossing_depth(jnp.asarray(ri), jnp.asarray(_centres(dz)), cfg))


@pytest.mark.parametrize("name", sorted(_CROSSING))
def test_crossing_hard_limit_matches_cvmix(name):
    assert _h(name, HARD) == pytest.approx(_CROSSING[name][2], rel=1e-10, abs=0.0)


@pytest.mark.parametrize("name", sorted(_CROSSING))
def test_crossing_default_records_gap(name):
    h20, h_cv = _h(name, CFG), _CROSSING[name][2]
    assert h20 == pytest.approx(_DEFAULT_DEPTH[name], rel=1e-6, abs=1e-12), (
        f"{name}: default depth {h20!r} (CVMix {h_cv!r}); re-measure the gap")


def test_crossing_gradient_at_default():
    dz, ri, _ = _CROSSING["interior"]
    d = jnp.asarray(_centres(dz))
    check_grads(lambda r: K._cvmix_first_crossing_depth(r, d, CFG),
                (jnp.asarray(ri),), order=1, modes=["rev"])
    g = jax.grad(lambda r: K._cvmix_first_crossing_depth(r, d, CFG))(jnp.asarray(ri))
    assert bool(jnp.all(jnp.isfinite(g))) and float(jnp.abs(g).max()) > 0.0


_GRAD_EXTRA = {"constant_sub": (_U, [0.1] * 8, None), "constant_super": (_U, [0.5] * 8, None),
               "constant_far": (_U, [50.0] * 8, None)}   # clamp underflows: sqrt(0) path


@pytest.mark.parametrize("name", sorted(_CROSSING) + sorted(_GRAD_EXTRA))
def test_crossing_gradient_finite_every_case(name):
    dz, ri, _ = {**_CROSSING, **_GRAD_EXTRA}[name]
    d = jnp.asarray(_centres(dz))
    g = jax.grad(lambda r: K._cvmix_first_crossing_depth(r, d, CFG))(jnp.asarray(ri))
    assert bool(jnp.all(jnp.isfinite(g))), name


@pytest.mark.parametrize("ri,dtype", [([0.0, 20.3, 20.3, -19.7], jnp.float64),
                                      ([0.0, 4.3, 4.3, -3.7], jnp.float32),
                                      ([0.0, 60.0, 60.0, -60.0], jnp.float64),
                                      ([19.0, -19.0, 0.0], jnp.float64),
                                      ([2.2, -2.2, 0.0], jnp.float32)])
def test_crossing_gradient_finite_extreme_ri(ri, dtype):
    """Large |Ri - Ri_c| (review case): clamped end values ~1e-175 (fp64) or
    clamp underflow must not give NaN adjoints, eager or jit."""
    d = jnp.asarray([0.5, 1.5, 2.5, 3.5][: len(ri)], dtype=dtype)
    f = lambda r: K._cvmix_first_crossing_depth(r, d, CFG)
    r = jnp.asarray(ri, dtype=dtype)
    for g in (jax.grad(f)(r), jax.jit(jax.grad(f))(r)):
        assert bool(jnp.all(jnp.isfinite(g))), g


def test_crossing_gradient_finite_flushed_q0_fp32():
    """Round-5b review repro: normalisation flushes Q0 with tiny M > 0 (fp32)."""
    r = np.float32(5.3)
    ri = jnp.asarray([0.0, np.nextafter(r, np.float32(0.0)), r, 1e7], dtype=jnp.float32)
    dz = np.array([1.0, 999.0, 2.0 ** -13, 2.0 ** -13], dtype=np.float32)
    z = jnp.asarray(np.cumsum(dz) - dz / 2)
    f = lambda v: K._cvmix_first_crossing_depth(v, z, CFG)
    for g in (jax.grad(f)(ri), jax.jit(jax.grad(f))(ri)):
        assert bool(jnp.all(jnp.isfinite(g))), g


def test_bracket_offset_saturated_negative_slope():
    """f0 >> 0 with m < 0 (q0 clamp underflows): offset tends to the second
    root -m H / (qH - m H), not 0 (round-5 review)."""
    f = lambda f0: K._cvmix_bracket_offset(f0, 0.0, -40.0, 1.0, 20.0)
    lim = 40.0 / (np.log(2.0) / 20.0 + 40.0)
    for f0 in (30.0, 40.0, 80.0):
        assert abs(float(f(f0)) - lim) < 1e-9, f0
        assert bool(jnp.isfinite(jax.grad(f)(f0)))
    # Both clamps underflow, m = 0 (k = 1 bracket): still a finite root.
    g = lambda f0: K._cvmix_bracket_offset(f0, -f0, 0.0, 1.0, 20.0)
    assert 0.0 <= float(g(80.0)) <= 1.0
    assert bool(jnp.isfinite(jax.grad(g)(80.0)))


def test_bulk_richardson_matches_cvmix():
    """CVMix compute_bulk_Richardson (N=0 -> V_t^2 = minVtsqr 1e-10)."""
    zc = np.array([-0.5, -1.5, -2.5])
    db = np.array([0.0, 2e-4, 5e-4])
    dv2 = np.array([0.0, 1e-4, 4e-6])
    ri_cvmix = [0.0, 2.84999715000285, 296.8675783105422]
    ri = K._cvmix_bulk_richardson(jnp.asarray(db * rho_0 / constants.g), jnp.asarray(dv2),
                                  jnp.full(3, 1e-10), jnp.asarray(-zc), CFG, constants.g)
    np.testing.assert_allclose(np.asarray(ri), ri_cvmix, rtol=1e-12, atol=0.0)
    # zero denominator -> num * 1e10 (CVMix)
    r0 = K._cvmix_bulk_richardson(jnp.array([1e-3]), jnp.zeros(1), jnp.zeros(1),
                                  jnp.array([2.0]), CFG, constants.g)
    num = 0.95 * constants.g * 1e-3 * 2.0 / rho_0
    assert float(r0[0]) == pytest.approx(num * 1e10, rel=1e-12)


# MOM6 KPP_BulkDrho at step 12 (inputs: end-of-step-11 T, S), first 12 levels.
_DRHO = {
    "F1conv": dict(dz=[1.0] * 12, T=[19.825796348911247, 19.804908963962014, 19.75456607059842, 19.650210333278263, 19.55000084340464, 19.4500000039158, 19.350000000019733, 19.250000000000114, 19.150000000000002, 19.050000000000043, 18.950000000000042, 18.85],
                   S=[35.00000000000002, 35.00000000000002, 35.00000000000003, 35.00000000000006, 35.00000000000007, 35.000000000000085, 35.00000000000009, 35.00000000000009, 35.00000000000009, 35.00000000000009, 35.00000000000009, 35.00000000000009],
                   drho=[0.0, 0.005471624007668652, 0.018645472815933317, 0.04588789363174328, 0.07196509012555907, 0.0979069170546154, 0.1237673942487163, 0.14954659777686174, 0.17524438803025078, 0.20086062379277791, 0.2261343823970492, 0.25113352699054303]),
    "F4deep": dict(dz=[10.0] * 12, T=[19.954148798653247, 19.933367265720587, 19.900227234760475, 19.860000013683052, 19.820000000000746, 19.78, 19.74, 19.700000000000003, 19.66, 19.620000000000005, 19.58, 19.539999999999996],
                   S=[34.99999999999999, 35.00000000000001, 35.00000000000002, 34.99999999999999, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0],
                   drho=[0.0, 0.005469249826091982, 0.014192179840165409, 0.02477929857855088, 0.03530615441741247, 0.04583250142150064, 0.056358351191875045, 0.06688371174459462, 0.07740859110913334, 0.08793299732610649, 0.098194988554269, 0.10826249334991189]),
}


@pytest.mark.parametrize("case", sorted(_DRHO))
def test_surface_layer_delta_rho_matches_mom6(case):
    c = _DRHO[case]
    dz = jnp.asarray(c["dz"])[None]
    T, S = jnp.asarray(c["T"])[None], jnp.asarray(c["S"])[None]
    z = jnp.asarray(_centres(c["dz"]))[None]
    drho, _ = K._mom6_surface_layer_deltas(T, S, jnp.zeros_like(T), jnp.zeros_like(T),
                                           dz, z, CFG, wright_eos, constants.g)
    np.testing.assert_allclose(np.asarray(drho[0]), c["drho"], rtol=1e-10, atol=1e-12)


def test_surface_layer_delta_v2_matches_overlap_average():
    """Delta-V^2 against MOM6's explicit thickness-overlap loop (MOM_CVMix_KPP.F90
    1342-1364, transcribed independently) on a stretched grid."""
    dz = np.array([1.0, 1.5, 2.0, 3.0, 4.5, 6.0])
    z = _centres(dz)
    z_top = np.cumsum(dz) - dz
    u = 0.1 - 0.01 * z + 0.003 * np.sin(z)
    want = np.empty_like(z)
    for k in range(z.size):
        sl = 0.1 * max(z[k], dz[0])
        overlap = np.clip(sl - z_top, 0.0, dz)
        want[k] = (u[k] - np.sum(overlap * u) / np.sum(overlap)) ** 2
    T = jnp.full((1, dz.size), 15.0)
    S = jnp.full((1, dz.size), 35.0)
    _, dv2 = K._mom6_surface_layer_deltas(T, S, jnp.asarray(u)[None], jnp.zeros((1, dz.size)),
                                          jnp.asarray(dz)[None], jnp.asarray(z)[None], CFG,
                                          wright_eos, constants.g)
    np.testing.assert_allclose(np.asarray(dv2[0]), want, rtol=1e-10, atol=1e-20)


# CVMix cvmix_coeffs_kpp (SimpleShapes, enhanced diffusion off) at MOM6 step-12 OBL,
# u*, surface buoyancy flux (CVMix sign: >0 stabilising); interfaces 0..n.
# "stretched" (prescribed OBL) separates true interfaces from centre midpoints.
_K = {
    "stretched": dict(dz=[1.0, 1.5, 2.0, 3.0, 4.5, 6.0, 8.0, 10.0], obl=9.0, ustar=0.01, B_cvmix=5e-08,
                      Kt=[0.0, 0.0028995356212481588, 0.004785366406161512, 0.004128440366972476,
                          0.0007645259938837916, 0.0, 0.0, 0.0, 0.0]),
    "F3stab": dict(obl=3.3695208329088526, ustar=0.0069843029576957824, B_cvmix=1.2300324986800555e-07,
                   Kt=[0.0, 0.0011111974366066913, 0.0007423991224678597, 8.107169985895808e-05] + [0.0] * 13),
    "F2wind": dict(obl=8.303897863020344, ustar=0.013968605915391565, B_cvmix=-0.0,
                   Kt=[0.0, 0.00432273330147212, 0.0064401708767341595, 0.006838496330524518,
                       0.006003893267581591, 0.004422545292643782, 0.0025806360104494927,
                       0.0009643490257371166, 5.986794324504344e-05] + [0.0] * 8),
}


@pytest.mark.parametrize("case", sorted(_K))
def test_interface_diffusivity_matches_cvmix(case, monkeypatch):
    c = _K[case]
    dz = np.asarray(c.get("dz", [1.0] * 16))
    n = dz.size
    zc = create_z_star_from_thicknesses(dz)
    T = jnp.asarray(20.0 - 0.1 * _centres(dz))[None]
    S = jnp.full((1, n), 35.0)
    rho = wright_eos(T, S, jnp.zeros_like(T))
    monkeypatch.setattr(K, "_boundary_layer_depth", lambda *a, **k: jnp.array([c["obl"]]))
    tau = np.sqrt((c["ustar"] ** 2 * rho_0) ** 2 - K._EPS)  # port u* = sqrt(sqrt(tau^2+eps)/rho0)
    out = K.kpp_vertical_mixing(jnp.zeros((1, n)), jnp.zeros((1, n)), T, S, rho, jnp.zeros(1), zc,
                                jnp.ones(1), CFG, constants.g, tau_x=jnp.array([tau]),
                                tau_y=jnp.zeros(1), B_f=jnp.array([-c["B_cvmix"]]),
                                apply_diffusion=False)
    inbl = np.cumsum(dz)[:-1] < c["obl"]
    kt = np.asarray(c["Kt"])[1:n]
    np.testing.assert_allclose(np.asarray(out.K_v[0] - CFG.K_bg)[inbl], kt[inbl], rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(np.asarray(out.A_v[0] - CFG.A_bg)[inbl], kt[inbl], rtol=1e-12, atol=0.0)


def test_unknown_bld_structure_raises():
    n = 4
    zc = create_z_star_from_thicknesses(np.ones(n))
    T = jnp.full((1, n), 15.0)
    with pytest.raises(ValueError, match="bld_structure"):
        K._boundary_layer_depth(T, T, T, T, T, zc, jnp.ones(1), jnp.array([0.01]),
                                jnp.zeros(1), KPPConfig(bld_structure="cvmx"))


def test_cvmix_mode_batched_and_dry_columns():
    """(2, 3, nlev) block with per-column thickness (jacobian 0.9-1.1) and one dry
    column: finite K and tendencies, every wet column equals the same column run
    alone (row-wise searchsorted / gathers), finite gradients on wet columns."""
    n = 10
    dzr = np.array([1.0, 1.5, 2.0, 3.0, 4.5, 6.0, 8.0, 10.0, 12.0, 15.0])
    zc = create_z_star_from_thicknesses(dzr)
    zr = np.cumsum(dzr) - 0.5 * dzr
    rng = np.random.default_rng(0)
    T = jnp.asarray(20.0 - 0.05 * zr + 0.02 * rng.standard_normal((2, 3, n)))
    S = jnp.broadcast_to(jnp.asarray(35.0 + 0.001 * zr), (2, 3, n))
    u = jnp.asarray(0.2 * np.exp(-zr / 5.0) * rng.uniform(0.5, 1.5, (2, 3, 1)))
    jac = jnp.asarray([[1.0, 0.9, 1.1], [1.05, 0.95, 0.0]])
    tau = jnp.asarray(rng.uniform(0.02, 0.1, (2, 3)))
    Bf = jnp.asarray(rng.uniform(-1e-7, 1e-7, (2, 3)))

    def run(T_, S_, jac_, tau_, Bf_, u_):
        rho = wright_eos(T_, S_, jnp.zeros_like(T_))
        return K.kpp_vertical_mixing(u_, jnp.zeros_like(u_), T_, S_, rho, jnp.zeros(T_.shape[:-1]),
                                     zc, jac_, CFG, constants.g, tau_x=tau_,
                                     tau_y=jnp.zeros_like(tau_), B_f=Bf_,
                                     Q_sfc_T=jnp.ones(T_.shape[:-1]),
                                     Q_sfc_S=jnp.zeros(T_.shape[:-1]), apply_diffusion=False)

    out = run(T, S, jac, tau, Bf, u)
    for f in (out.K_v, out.A_v, out.dT_dt):
        assert bool(jnp.all(jnp.isfinite(f)))
    np.testing.assert_allclose(np.asarray(out.K_v[1, 2]), CFG.K_bg, rtol=0, atol=1e-15)
    for i, j in ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1)):
        one = run(T[i, j][None], S[i, j][None], jac[i, j][None], tau[i, j][None], Bf[i, j][None],
                  u[i, j][None])
        for got, want in ((out.K_v, one.K_v), (out.A_v, one.A_v), (out.dT_dt, one.dT_dt)):
            np.testing.assert_allclose(np.asarray(got[i, j]), np.asarray(want[0]), rtol=1e-13, atol=0)
    g = jax.grad(lambda t: jnp.sum(run(t, S, jac, tau, Bf, u).K_v))(T)
    wet = np.asarray(jac) > 0.0
    # Wet columns only: a fully dry column's h is NaN in BOTH structures (shared
    # N^2 / V_t^2 path divides 0/0 at zero thickness; pre-existing, not cvmix).
    assert bool(jnp.all(jnp.isfinite(g[wet]))) and float(jnp.abs(g[wet]).max()) > 0.0


def test_crossing_continuous_on_supercritical_plateau():
    """Brackets that only carry soft weight must not jump when the plateau
    curvature changes sign (review case)."""
    d = jnp.asarray([0.5, 1.5, 2.5, 3.5, 4.5])
    ri = jnp.asarray([0.0, 0.31, 0.31, 0.31, 0.8])
    h0 = float(K._cvmix_first_crossing_depth(ri, d, CFG))
    for k in (2, 3):
        for e in (1e-8, -1e-8):
            h1 = float(K._cvmix_first_crossing_depth(ri.at[k].add(e), d, CFG))
            assert abs(h1 - h0) < 1e-6, (k, e, h0, h1)
    check_grads(lambda r: K._cvmix_first_crossing_depth(r, d, CFG), (ri,), order=1, modes=["rev"])


@pytest.mark.parametrize("f0,fH,m", [(0.0, 0.1, -1.0),    # top at Ri_c, dipping slope
                                     (-0.1, 0.0, 1.0),    # bottom at Ri_c, convex
                                     (0.0, 0.0, 0.0),
                                     (0.2, -0.2, 0.0)])   # downward bracket, m sign flip
def test_bracket_offset_continuous_at_review_seams(f0, fH, m):
    """Offsets on both sides of each seam reviewers found agree to O(perturbation)."""
    H = jnp.asarray(1.0)
    base = float(K._cvmix_bracket_offset(jnp.asarray(f0), jnp.asarray(fH), jnp.asarray(m), H,
                                         CFG.crossing_sharpness))
    for df0, dfH, dm in ((1e-9, 0, 0), (-1e-9, 0, 0), (0, 1e-9, 0), (0, -1e-9, 0),
                         (0, 0, 1e-9), (0, 0, -1e-9)):
        x = float(K._cvmix_bracket_offset(jnp.asarray(f0 + df0), jnp.asarray(fH + dfH),
                                          jnp.asarray(m + dm), H, CFG.crossing_sharpness))
        assert abs(x - base) < 1e-6, (f0, fH, m, df0, dfH, dm, base, x)


def test_bracket_offset_continuous_random():
    """Random brackets: a 1e-9 perturbation of any input moves the offset < 1e-6."""
    rng = np.random.default_rng(1)
    f0, fH, m = (jnp.asarray(rng.uniform(-0.5, 0.5, 4000)) for _ in range(3))
    H = jnp.asarray(rng.uniform(0.5, 10.0, 4000))
    off = lambda *v: np.asarray(K._cvmix_bracket_offset(*v, H, CFG.crossing_sharpness))
    x0 = off(f0, fH, m)
    for v in ((f0 + 1e-9, fH, m), (f0, fH + 1e-9, m), (f0, fH, m + 1e-9)):
        assert np.max(np.abs(off(*v) - x0)) < 1e-6
    g = jax.grad(lambda a: jnp.sum(K._cvmix_bracket_offset(a, fH, m, H, CFG.crossing_sharpness)))(f0)
    assert bool(jnp.all(jnp.isfinite(g)))


# CVMix cvmix_coeffs_kpp nonlocal term (C_s * G(sigma), CVMix C_s = MOM6 CS2
# 6.32739901508) on the stretched grid, OBL 9 m, u* 0.01, B_cvmix -5e-8 (unstable).
_CVMIX_CS2 = 6.32739901508
_NL_STRETCHED = [0.0, 0.5554918202538277, 0.9167784924111023, 0.7909248768848446,
                 0.14646756979348968, 0.0, 0.0, 0.0, 0.0]


def test_nonlocal_shape_at_true_interfaces(monkeypatch):
    """cvmix mode evaluates G(sigma) of the nonlocal flux at the true interfaces
    (stretched grid: centre midpoints would differ).  Shapes compared (the C_s
    constants differ by the LMD94 rounding, a recorded known difference)."""
    dz = np.array([1.0, 1.5, 2.0, 3.0, 4.5, 6.0, 8.0, 10.0])
    n = dz.size
    zc = create_z_star_from_thicknesses(dz)
    T = jnp.asarray(20.0 - 0.05 * _centres(dz))[None]
    S = jnp.full((1, n), 35.0)
    monkeypatch.setattr(K, "_boundary_layer_depth", lambda *a, **k: jnp.array([9.0]))
    out = K.kpp_vertical_mixing(jnp.zeros((1, n)), jnp.zeros((1, n)), T, S,
                                wright_eos(T, S, jnp.zeros_like(T)), jnp.zeros(1), zc, jnp.ones(1),
                                CFG, constants.g, tau_x=jnp.array([0.01 ** 2 * rho_0]),
                                tau_y=jnp.zeros(1), B_f=jnp.array([5e-8]),
                                Q_sfc_T=jnp.ones(1), Q_sfc_S=jnp.zeros(1), apply_diffusion=False)
    dT = np.asarray(out.dT_dt[0])
    F = -np.cumsum(dT * dz)[:-1]            # flux at interior interfaces
    got = F / CFG.gamma_T
    want = np.asarray(_NL_STRETCHED)[1:n] / _CVMIX_CS2
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-15)


# Legacy structure pinned at the stack base 7818b0e18d (PR #2560 head):
# 2 columns, stretched grid, apply_diffusion=True, dt=600 s.
_LEGACY_K_V = [[0.0028613390168094653, 0.004455298478307277, 0.003432405391351673, 0.0010864835042452773, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05], [0.0005128825621043311, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05]]
_LEGACY_A_V = [[0.002591620563793403, 0.003929723781049334, 0.0030484785733987225, 0.0010274145473546054, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001], [0.0006028825621043311, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]]
_LEGACY_DT_DT = [[-4.371291449051502e-05, -3.33158948160732e-05, -3.1705760647775854e-05, 3.4093446170271146e-05, 1.2070653222080686e-05, -5.646886315028669e-22, 3.5734202462290796e-22, -1.4823076576950255e-22, 2.2940475654803966e-22, 3.3333333333333195e-08], [-3.94117477205255e-05, 2.587969485358914e-05, 7.986360435284163e-08, 4.129244142456893e-08, 1.6024165992988293e-08, 4.694099954881841e-09, 8.886105206707923e-10, 1.1087110016905693e-10, 8.671289543722097e-12, 1.3333727321538709e-08]]
_LEGACY_DS_DT = [[1.37129144905082e-06, 8.315894816084423e-07, 5.455760647757201e-07, -8.579892893535707e-07, -2.501910962120453e-07, -1.6915843526709416e-21, 4.2310288332087366e-22, 2.152323953862978e-22, -1.793603294885815e-22, -6.666666666666921e-10], [2.0515302484154584e-06, -1.3410201656102868e-06, -1.8270765370071275e-20, 3.16093284743597e-21, 0.0, -5.635857240194629e-22, 4.2268929301459714e-22, -4.314574075076593e-22, 3.5954783958971605e-22, -2.6666666666667683e-09]]
_LEGACY_DU_DT = [[-2.4017919398069545e-05, -1.0744126210987942e-05, -3.379876668229163e-06, 1.2728587545212194e-05, 1.856837327675612e-06, 4.086013134606541e-08, 1.0549690131577407e-08, 1.949210750009046e-09, 2.4752300459740787e-10, 2.071654324533266e-11], [-2.9581651571874514e-06, 1.695000890657672e-06, 4.343945162761329e-08, 3.183866423708344e-08, 2.017831729672153e-08, 1.1589502737532698e-08, 5.307029730153358e-09, 2.023736618579618e-09, 6.275161612187131e-10, 1.8037886749171123e-10]]


def test_legacy_structure_unchanged():
    dz = np.array([1.0, 1.5, 2.0, 3.0, 4.5, 6.0, 8.0, 10.0, 12.0, 15.0])
    z = _centres(dz)
    T = jnp.asarray(np.stack([20.0 - 0.05 * z, 18.0 - 0.02 * z + 0.3 * np.exp(-z / 4.0)]))
    S = jnp.asarray(np.stack([35.0 + 0.001 * z, 34.5 + 0.004 * z]))
    u = jnp.asarray(np.stack([0.2 * np.exp(-z / 5.0), 0.05 * np.exp(-z / 9.0)]))
    v = jnp.asarray(np.stack([0.02 * np.exp(-z / 3.0), -0.04 * np.exp(-z / 6.0)]))
    out = K.kpp_vertical_mixing(u, v, T, S, wright_eos(T, S, jnp.zeros_like(T)), jnp.zeros(2),
                                create_z_star_from_thicknesses(dz), jnp.ones(2), KPPConfig(),
                                constants.g, tau_x=jnp.array([0.05, 0.1]),
                                tau_y=jnp.array([0.0, 0.02]), B_f=jnp.array([5e-8, -3e-8]),
                                Q_sfc_T=jnp.array([1e-5, -2e-5]), Q_sfc_S=jnp.array([-1e-6, 1e-6]),
                                apply_diffusion=True, dt=600.0)
    for got, want in ((out.K_v, _LEGACY_K_V), (out.A_v, _LEGACY_A_V), (out.dT_dt, _LEGACY_DT_DT),
                      (out.dS_dt, _LEGACY_DS_DT), (out.du_dt, _LEGACY_DU_DT)):
        w = np.asarray(want)
        np.testing.assert_allclose(np.asarray(got), w, rtol=1e-13, atol=1e-13 * np.abs(w).max())
