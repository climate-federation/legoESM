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
_DEFAULT_DEPTH = {
    "bracket1": 1.244916972515393,     # CVMix 1.24535599 -> gap -0.04 %
    "concave": 3.402727637058351,      # CVMix 3.79289322 -> gap -10.29 %
    "first_cell": 0.5000444751382389,  # CVMix 0.5        -> gap +0.01 %
    "interior": 4.626519199654815,     # CVMix 4.75       -> gap -2.60 %
    "multiple": 2.568907329687243,     # CVMix 2.28077641 -> gap +12.63 %
    "none": 7.165248276145145,         # CVMix 7.5        -> gap -4.46 %
    "stretched": 10.123849217594948,   # CVMix 10.6758377 -> gap -5.17 %
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


@pytest.mark.parametrize("name", sorted(_CROSSING))
def test_crossing_gradient_finite_every_case(name):
    dz, ri, _ = _CROSSING[name]
    d = jnp.asarray(_centres(dz))
    g = jax.grad(lambda r: K._cvmix_first_crossing_depth(r, d, CFG))(jnp.asarray(ri))
    assert bool(jnp.all(jnp.isfinite(g))), name


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


def test_surface_layer_delta_v2_linear_profile():
    """u linear in depth on a stretched grid: the [0, SL_k] mean is u(SL_k / 2)."""
    dz = np.array([1.0, 1.5, 2.0, 3.0, 4.5, 6.0])
    z = _centres(dz)
    u = 0.1 - 0.01 * z
    sl = 0.1 * np.maximum(z, dz[0])
    want = (u - (0.1 - 0.01 * sl / 2)) ** 2
    T = jnp.full((1, dz.size), 15.0)
    S = jnp.full((1, dz.size), 35.0)
    _, dv2 = K._mom6_surface_layer_deltas(T, S, jnp.asarray(u)[None], jnp.zeros((1, dz.size)),
                                          jnp.asarray(dz)[None], jnp.asarray(z)[None], CFG,
                                          wright_eos, constants.g)
    np.testing.assert_allclose(np.asarray(dv2[0]), want, rtol=1e-12, atol=1e-30)


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
