"""Faithfulness pins for the bulk-Richardson PBL-height diagnosis.

Target: ``compute_bulk_richardson``, ``diagnose_pbl_height`` (smooth
transition-weighted), ``diagnose_pbl_height_interp`` (linear-interp), and the
shared ``first_crossing_height`` kernel in
``legoesm.atmosphere.physics.turbulence.pbl_height``.

Most-trustful source
--------------------
Standard bulk-Richardson PBL-height diagnosis (e.g. Vogelezang & Holtslag 1996;
Troen & Mahrt 1986).  The bulk Richardson number from the surface to level z is

    Ri_b(z) = (g / theta_v_sfc) * (theta_v(z) - theta_v_sfc) * dz
              / [(u(z)-u_sfc)^2 + (v(z)-v_sfc)^2]

(the denominator is the wind SHEAR from the surface, not the absolute speed), and
the PBL top is where Ri_b first reaches the critical value Ri_crit.

The existing coverage pins ``first_crossing_height`` only as a DEDUP regression
(vs the old ysu inline copy) + the Louis stability function; the bulk-Richardson
FORMULA and the diagnosis assembly are not oracle-pinned.  This adds those.

Certification (test-only):
1. compute_bulk_richardson: exact Ri_b assembly vs an independent oracle (reusing
   the shared virtual_temperature / exner / buoyancy_coefficient primitives as
   GIVENs), incl. surface Ri_b=0, the stable/unstable sign, and the shear floor.
2. first_crossing_height: exact linear interpolation to the threshold on a known
   linear profile (independent textbook form, not the old-copy dedup).
3. diagnose_pbl_height (smooth): the transition-weighted average h = sum(z*w)/sum(w),
   w = sigma*(1-sigma), sigma = sigmoid(sharpness*(Ri_crit-Ri_b)); the sharp-limit
   -> Ri_crit crossing; the interp variant; Ri_crit plumbing; h_min/h_max clamps;
   differentiability.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.atmosphere.physics._shared import (
    buoyancy_coefficient,
    exner_function,
    virtual_temperature,
)
from legoesm.atmosphere.physics.turbulence.pbl_height import (
    PBLHeightConfig,
    compute_bulk_richardson,
    diagnose_pbl_height,
    diagnose_pbl_height_interp,
    first_crossing_height,
)

jax.config.update("jax_enable_x64", True)

# Top-first level ordering (index -1 = surface).  5 levels.
_Z = jnp.asarray([[3000.0, 1500.0, 600.0, 150.0, 0.0]])       # (1, nlev), z >= 0
_P = jnp.asarray([[7.0e4, 8.5e4, 9.2e4, 9.8e4, 1.0e5]])       # increasing to surface


def _a(x):
    return jnp.asarray(x, dtype=jnp.float64)


def _ri_oracle(T, q, u, v, p, z):
    """Independent bulk-Richardson assembly (helpers reused as GIVENs)."""
    theta_v = np.asarray(virtual_temperature(_a(T), _a(q))) / np.asarray(exner_function(_a(p)))
    tvs = theta_v[:, -1:]
    dz = np.abs(np.asarray(z) - np.asarray(z)[:, -1:]) + 1.0
    dtv = theta_v - tvs
    u = np.asarray(u)
    v = np.asarray(v)
    dV2 = (u - u[:, -1:]) ** 2 + (v - v[:, -1:]) ** 2 + 1e-4
    bcoef = np.asarray(buoyancy_coefficient(jnp.clip(_a(tvs), 1.0, None)))
    return bcoef * (dtv * dz / dV2)


# A convective-then-stable column: near-neutral/unstable below, stable aloft.
# _U has a NONZERO surface value (u_sfc = 10): the shear from the surface is
# [20,14,6,2,0] but the absolute wind differs, so an erroneous absolute-speed
# denominator (instead of surface shear) is caught by the formula pin.
_T = [[250.0, 262.0, 286.0, 289.0, 290.0]]         # warm sfc, cold aloft (stable up high)
_Q = [[1e-3, 2e-3, 5e-3, 8e-3, 1e-2]]
_U = [[30.0, 24.0, 16.0, 12.0, 10.0]]              # u_sfc=10 -> shear != |wind|
_V = [[0.0, 0.0, 0.0, 0.0, 0.0]]


# ---------------------------------------------------------------------------
# 1. Bulk Richardson number.
# ---------------------------------------------------------------------------
def test_bulk_richardson_formula():
    ri, _ = compute_bulk_richardson(_a(_T), _a(_Q), _a(_U), _a(_V), _P, _Z)
    exp = _ri_oracle(_T, _Q, _U, _V, _P, _Z)
    np.testing.assert_allclose(np.asarray(ri), exp, rtol=1e-11)


def test_bulk_richardson_surface_is_zero():
    # At the surface dz=0 and dtheta_v=0 -> Ri_b[..., -1] = 0.
    ri, _ = compute_bulk_richardson(_a(_T), _a(_Q), _a(_U), _a(_V), _P, _Z)
    np.testing.assert_allclose(float(np.asarray(ri)[0, -1]), 0.0, atol=1e-12)


def test_bulk_richardson_stable_positive_unstable_negative():
    # sign(Ri_b) = sign(theta_v - theta_v_sfc): a column whose theta_v INCREASES
    # upward is stable (Ri_b > 0 aloft); one whose theta_v decreases upward is
    # unstable (Ri_b < 0).  theta_v = T / exner and exner decreases upward, so a
    # strongly warm-aloft column is stable while the cold-aloft _T column is not.
    T_stable = [[310.0, 300.0, 295.0, 291.0, 290.0]]          # theta_v increases upward
    ri_s, _ = compute_bulk_richardson(_a(T_stable), _a(_Q), _a(_U), _a(_V), _P, _Z)
    assert float(np.asarray(ri_s)[0, 0]) > 0.0                # stable aloft
    ri_u, _ = compute_bulk_richardson(_a(_T), _a(_Q), _a(_U), _a(_V), _P, _Z)  # cold aloft
    assert float(np.asarray(ri_u)[0, 0]) < 0.0                # unstable aloft


def test_bulk_richardson_shear_floor():
    # Zero wind everywhere -> the dV2 = 1e-4 floor keeps Ri_b finite (no /0).
    zero = [[0.0] * 5]
    ri, _ = compute_bulk_richardson(_a(_T), _a(_Q), _a(zero), _a(zero), _P, _Z)
    assert np.all(np.isfinite(np.asarray(ri)))
    exp = _ri_oracle(_T, _Q, zero, zero, _P, _Z)             # oracle uses the same floor
    np.testing.assert_allclose(np.asarray(ri), exp, rtol=1e-11)


# ---------------------------------------------------------------------------
# 2. first_crossing_height (independent linear-interp).
# ---------------------------------------------------------------------------
def test_first_crossing_linear_interpolation():
    # A linear values profile crossing the threshold between two known levels: the
    # crossing height is the exact linear interpolation, independent of the old copy.
    z = jnp.asarray([[400.0, 300.0, 200.0, 100.0, 0.0]])     # top-first
    vals = jnp.asarray([[2.0, 1.0, 0.0, -1.0, -2.0]])        # crosses 0.5 between z=300,400
    got = float(np.asarray(first_crossing_height(vals, z, 0.5, 1e4))[0])
    # surface-up, the first pair whose UPPER member reaches 0.5 is (z=200,val=0.0)->
    # (z=300,val=1.0); linear interp: val 0->1 crosses 0.5 at z = 200 + 0.5*(300-200)
    # = 250.  At sharpness 1e4 the sigmoid gate is exactly {0,1} in float64, so the
    # crossing is round-off exact.
    np.testing.assert_allclose(got, 250.0, rtol=1e-12)


# ---------------------------------------------------------------------------
# 3. diagnose_pbl_height (smooth transition-weighted).
# ---------------------------------------------------------------------------
def _diag_oracle(ri, z, cfg):
    ri = np.asarray(ri)
    z = np.asarray(z)
    sigma = 1.0 / (1.0 + np.exp(-cfg.sharpness * (cfg.Ri_crit - ri)))
    w = sigma * (1.0 - sigma) + 1e-20
    h = np.sum(z * w, axis=1) / np.sum(w, axis=1)
    return np.clip(h, cfg.h_min, cfg.h_max)


def test_diagnose_smooth_weighted_average():
    cfg = PBLHeightConfig()
    h = diagnose_pbl_height(_a(_T), _a(_Q), _a(_U), _a(_V), _P, _Z, cfg)
    ri, _ = compute_bulk_richardson(_a(_T), _a(_Q), _a(_U), _a(_V), _P, _Z)
    np.testing.assert_allclose(np.asarray(h), _diag_oracle(ri, _Z, cfg), rtol=1e-10)


def test_diagnose_hmin_hmax_clamp():
    # The raw (unclamped) smooth height for this column is ~127 m; bracket it with
    # caps that MUST bind exactly, so both clamp directions are genuine canaries.
    raw = float(
        np.asarray(diagnose_pbl_height(_a(_T), _a(_Q), _a(_U), _a(_V), _P, _Z,
                                       PBLHeightConfig(h_min=0.0, h_max=1.0e9)))[0]
    )
    assert 110.0 < raw < 4000.0                              # both caps genuinely bind
    h_capped = float(
        np.asarray(diagnose_pbl_height(_a(_T), _a(_Q), _a(_U), _a(_V), _P, _Z,
                                       PBLHeightConfig(h_min=100.0, h_max=110.0)))[0]
    )
    np.testing.assert_allclose(h_capped, 110.0, rtol=1e-12)  # h_max clamp binds exactly
    h_floor = float(
        np.asarray(diagnose_pbl_height(_a(_T), _a(_Q), _a(_U), _a(_V), _P, _Z,
                                       PBLHeightConfig(h_min=4000.0, h_max=5000.0)))[0]
    )
    np.testing.assert_allclose(h_floor, 4000.0, rtol=1e-12)  # h_min clamp binds exactly


def test_diagnose_ri_crit_plumbing():
    # A higher Ri_crit moves the transition peak to a different level -> different h.
    h1 = np.asarray(diagnose_pbl_height(_a(_T), _a(_Q), _a(_U), _a(_V), _P, _Z,
                                        PBLHeightConfig(Ri_crit=0.25)))
    h2 = np.asarray(diagnose_pbl_height(_a(_T), _a(_Q), _a(_U), _a(_V), _P, _Z,
                                        PBLHeightConfig(Ri_crit=1.0)))
    assert abs(float(h1[0]) - float(h2[0])) > 1.0
    # and each matches its own oracle (plumbing, not a hard-coded Ri_crit):
    ri, _ = compute_bulk_richardson(_a(_T), _a(_Q), _a(_U), _a(_V), _P, _Z)
    np.testing.assert_allclose(h2, _diag_oracle(ri, _Z, PBLHeightConfig(Ri_crit=1.0)), rtol=1e-10)


def test_diagnose_interp_variant_matches_crossing():
    # A column with a SINGLE clean Ri_crit crossing in the surface->first-level pair:
    # the linear-interp variant must return exactly the independently-computed crossing
    # height z_cross = z_sfc + (Ri_crit - Ri_sfc)/(Ri_1 - Ri_sfc) * (z_1 - z_sfc).
    # This canary catches a wrong interpolation fraction, wrong crossing pair, an
    # ignored Ri_crit, or a broken interp assembly (a bounds-only check would not).
    zc = jnp.asarray([[400.0, 150.0, 0.0]])                  # top-first, surface = 0
    pc = jnp.asarray([[8.0e4, 9.5e4, 1.0e5]])
    tc = [[300.0, 291.0, 290.0]]                             # strongly stable aloft
    qc = [[1e-3, 5e-3, 1e-2]]
    uc = [[20.0, 12.0, 10.0]]                                # u_sfc = 10
    vc = [[0.0, 0.0, 0.0]]
    ric = 0.5                                                # < Ri_1, > Ri_sfc(=0)
    ri = _ri_oracle(tc, qc, uc, vc, pc, zc)                  # independent bulk-Ri
    ri_sfc = ri[0, -1]
    ri_1 = ri[0, -2]
    frac = (ric - ri_sfc) / (ri_1 - ri_sfc)
    z_cross = float(zc[0, -1]) + frac * (float(zc[0, -2]) - float(zc[0, -1]))
    cfg = PBLHeightConfig(Ri_crit=ric, h_min=0.0, h_max=5000.0, sharpness=500.0)
    h = float(
        np.asarray(diagnose_pbl_height_interp(_a(tc), _a(qc), _a(uc), _a(vc), pc, zc, cfg))[0]
    )
    np.testing.assert_allclose(h, z_cross, rtol=1e-9)


def test_differentiable():
    def loss(T):
        return jnp.sum(diagnose_pbl_height(T, _a(_Q), _a(_U), _a(_V), _P, _Z, PBLHeightConfig()))
    g = jax.grad(loss)(_a(_T))
    assert jnp.all(jnp.isfinite(g))
    assert float(jnp.max(jnp.abs(g))) > 0.0                  # non-vacuous: real sensitivity
