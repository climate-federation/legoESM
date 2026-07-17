"""Faithfulness pins for the CATKE vertical-mixing closure.

Target: the column physics in
``legoesm.ocean.physics.vertical_mixing.catke`` — ``catke_diffusivities``,
``catke_dissipation_rate``, ``catke_surface_tke_flux`` and their building-block
closed forms.

Most-trustful source
--------------------
Wagner et al. (2025, JAMES, doi:10.1029/2024MS004522), faithful to the
Oceananigans ``CATKEVerticalDiffusivity`` reference.  CATKE derives eddy
diffusivities ``K_X = l_X * w*`` from a prognostic TKE field, with
``w* = sqrt(max(e, e_min))``, a Richardson-number stability function blending
negative / low / high-Ri regimes, and a dynamic Deardorff convective mixing
length ``l = c_conv * w*^3 / Jb`` (plus an entrainment branch).

The existing test_catke*.py are behavioral (0 exact-magnitude assertions); this
pins the building-block closed forms, the full K_X = l_X * w* assembly (per
variable), the dissipation rate, and the surface TKE flux against an independent
reimplementation typed from the documented forms + the config coefficients.

Certification (test-only):
1. Building blocks: w* floor, Ri = N2/max(S2,eps), the piecewise-linear step, the
   three-regime stability function, the stable (min-of-three) length, and the
   Deardorff convective / entrainment length (all three branches).
2. Public K_u/K_c/K_e = min(l_X * w*, cap) with the per-variable coefficients,
   over stable / convective / entraining regimes; caps; K >= 0; convective
   enhancement.
3. Dissipation rate (physical sqrt|e|/l_D and the e<0 numerical-damping branch);
   surface TKE flux -c_w_ustar*u*^3 - c_w_conv*wConv^3.
4. Config plumbing (per-variable + cap + surface coefficients); differentiability
   through the convective (N2<0) branch.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.ocean.physics.vertical_mixing import catke as _catke
from legoesm.ocean.physics.vertical_mixing.catke import (
    catke_diffusivities,
    catke_dissipation_rate,
    catke_surface_tke_flux,
)
from legoesm.ocean.physics.vertical_mixing.config import CATKEConfig

jax.config.update("jax_enable_x64", True)

_CFG = CATKEConfig()
_EPS_LEN = 1.0e-20
_EPS_S2 = 1.0e-20


def _a(x):
    return jnp.asarray(x, dtype=jnp.float64)


# --- Independent oracle (typed from the Wagner-2025 / Oceananigans forms) -----
def _w_star(e, cfg):
    return np.sqrt(np.maximum(e, cfg.minimum_tke))


def _ri(N2, S2):
    return N2 / np.maximum(S2, _EPS_S2)


def _step(x, c, w):
    return np.clip((x - c) / w, 0.0, 1.0)


def _sigma(Ri, c_un, c_lo, c_hi, cfg):
    pos = c_lo + (c_hi - c_lo) * _step(Ri, cfg.c_ri_lower, cfg.c_ri_width)
    return np.where(Ri < 0.0, c_un, pos)


def _l_stable(w, N2, depth, hab, cfg):
    d = np.minimum(cfg.c_surface_shear * depth, cfg.c_bottom_shear * hab)
    N2p = np.maximum(N2, 0.0)
    ell_N = np.where(N2p > 0.0, w / np.sqrt(N2p + _EPS_LEN), np.inf)
    return np.minimum(d, ell_N)


def _l_conv(w, N2, N2a, S2, depth, Jb, c_conv, c_entr, cfg):
    jb_eps = cfg.minimum_convective_buoyancy_flux
    convecting = (Jb > jb_eps) & (N2 < 0.0)
    entraining = (Jb > jb_eps) & (N2 > 0.0) & (N2a < 0.0)
    conv_den = np.where(convecting, Jb + jb_eps, 1.0)
    Ri_f = depth * S2 * w / conv_den
    ell_c = np.maximum(c_conv * w ** 3 / conv_den * (1.0 - cfg.c_sheared_plume * Ri_f), 0.0)
    ent_den = np.where(entraining, w * N2 + jb_eps, 1.0)
    ell_e = c_entr * Jb / ent_den
    return np.where(convecting, ell_c, np.where(entraining, ell_e, 0.0))


def _l_var(w, Ri, N2, N2a, S2, depth, hab, H, Jb, c_un, c_lo, c_hi, c_conv, c_entr, cfg):
    sigma = _sigma(Ri, c_un, c_lo, c_hi, cfg)
    ell_star = sigma * _l_stable(w, N2, depth, hab, cfg)
    ell_h = _l_conv(w, N2, N2a, S2, depth, Jb, c_conv, c_entr, cfg)
    return np.minimum(H, np.maximum(ell_star, ell_h))


def _k_oracle(e, N2, N2a, S2, depth, hab, H, Jb, cfg):
    w = _w_star(e, cfg)
    Ri = _ri(N2, S2)
    lu = _l_var(w, Ri, N2, N2a, S2, depth, hab, H, Jb,
                cfg.c_un_u, cfg.c_lo_u, cfg.c_hi_u, cfg.c_conv_u, cfg.c_entr_u, cfg)
    lc = _l_var(w, Ri, N2, N2a, S2, depth, hab, H, Jb,
                cfg.c_un_c, cfg.c_lo_c, cfg.c_hi_c, cfg.c_conv_c, cfg.c_entr_c, cfg)
    le = _l_var(w, Ri, N2, N2a, S2, depth, hab, H, Jb,
                cfg.c_un_e, cfg.c_lo_e, cfg.c_hi_e, cfg.c_conv_e, cfg.c_entr_e, cfg)
    return (np.minimum(lu * w, cfg.maximum_viscosity),
            np.minimum(lc * w, cfg.maximum_tracer_diffusivity),
            np.minimum(le * w, cfg.maximum_tke_diffusivity))


# Discriminating column profiles (interface fields; scalar H, Jb per column).
def _stable_profile():
    # stratified, sheared, no surface convection.
    return dict(e=_a([1e-3, 5e-4]), N2=_a([1e-4, 2e-4]), N2_above=_a([1e-4, 1e-4]),
                S2=_a([1e-3, 1e-3]), depth=_a([10.0, 30.0]),
                height_above_bottom=_a([90.0, 70.0]), H=_a(100.0), Jb=_a(-1e-8))


def _convective_profile():
    # destabilising surface flux + unstable stratification (Jb > jb_eps, N2 < 0).
    return dict(e=_a([1e-2, 5e-3]), N2=_a([-1e-5, -2e-5]), N2_above=_a([-1e-5, -1e-5]),
                S2=_a([1e-5, 1e-5]), depth=_a([5.0, 15.0]),
                height_above_bottom=_a([95.0, 85.0]), H=_a(100.0), Jb=_a(1e-3))


def _call_k(p, cfg=_CFG):
    return catke_diffusivities(p["e"], p["N2"], p["N2_above"], p["S2"], p["depth"],
                               p["height_above_bottom"], p["H"], p["Jb"], cfg)


# ---------------------------------------------------------------------------
# 1. Building blocks.
# ---------------------------------------------------------------------------
def test_turbulent_velocity_floor():
    e = _a([4.0e-2, 1e-12])                        # second below minimum_tke
    w = np.asarray(_catke._turbulent_velocity(e, _CFG.minimum_tke))
    np.testing.assert_allclose(w[0], 0.2, rtol=1e-12)                     # sqrt(0.04)
    np.testing.assert_allclose(w[1], np.sqrt(_CFG.minimum_tke), rtol=1e-12)  # floored


def test_richardson_and_shear_floor():
    np.testing.assert_allclose(float(_catke._richardson(_a(2e-4), _a(1e-3))), 0.2, rtol=1e-12)
    # at rest (S2 -> 0) the shear floor is EXACTLY N2 / _EPS_S2 (= N2 / 1e-20).
    got = float(_catke._richardson(_a(1e-4), _a(0.0)))
    np.testing.assert_allclose(got, 1e-4 / _EPS_S2, rtol=1e-12)


def test_step_ramp():
    x = _a([-1.0, 0.254, 0.254 + 0.51, 0.254 + 1.02, 5.0])
    out = np.asarray(_catke._step(x, 0.254, 1.02))
    np.testing.assert_allclose(out, [0.0, 0.0, 0.5, 1.0, 1.0], rtol=1e-12)


def test_stability_function_three_regimes():
    f = _catke._stability_function
    # Ri < 0 -> c_un.
    np.testing.assert_allclose(float(f(_a(-1.0), 0.37, 0.36, 0.24, 0.254, 1.02)),
                               0.37, rtol=1e-12)
    # Ri way above c_ri_lower+c_ri_width -> c_hi.
    np.testing.assert_allclose(float(f(_a(10.0), 0.37, 0.36, 0.24, 0.254, 1.02)),
                               0.24, rtol=1e-12)
    # midway -> c_lo + (c_hi-c_lo)*0.5.
    np.testing.assert_allclose(
        float(f(_a(0.254 + 0.51), 0.37, 0.36, 0.24, 0.254, 1.02)),
        0.36 + (0.24 - 0.36) * 0.5, rtol=1e-12)


def test_stable_length_min_of_three():
    w, N2, depth, hab = 0.2, 1e-4, 10.0, 90.0
    got = float(_catke._stable_length(_a(w), _a(N2), _a(depth), _a(hab),
                                 _CFG.c_surface_shear, _CFG.c_bottom_shear))
    d = min(_CFG.c_surface_shear * depth, _CFG.c_bottom_shear * hab)
    ell_n = w / np.sqrt(N2 + _EPS_LEN)
    np.testing.assert_allclose(got, min(d, ell_n), rtol=1e-12)  # surface-distance wins (11.31)

    def _sl(w_, n2_, dep_, hab_):
        return float(_catke._stable_length(_a(w_), _a(n2_), _a(dep_), _a(hab_),
                                           _CFG.c_surface_shear, _CFG.c_bottom_shear))
    # BOTTOM-distance wins: small height-above-bottom (c_bottom*hab is smallest).
    np.testing.assert_allclose(_sl(0.2, 1e-4, 100.0, 2.0), _CFG.c_bottom_shear * 2.0, rtol=1e-12)
    # STRATIFICATION length wins: strong N2 (w*/sqrt(N2) is smallest).
    np.testing.assert_allclose(_sl(0.2, 1e-2, 100.0, 100.0),
                               0.2 / np.sqrt(1e-2 + _EPS_LEN), rtol=1e-12)
    # N2 <= 0 -> stratification length is infinite, so min is the distance bound.
    np.testing.assert_allclose(_sl(w, -1e-5, depth, hab), d, rtol=1e-12)


def test_convective_length_deardorff_and_entrainment():
    jb_eps = _CFG.minimum_convective_buoyancy_flux
    w, S2, depth = 0.3, 1e-4, 8.0
    # convecting: Jb>jb_eps & N2<0.
    Jb, N2 = 1e-3, -1e-5
    got = float(_catke._convective_length(_a(w), _a(N2), _a(-1e-5), _a(S2), _a(depth), _a(Jb),
                                     _CFG.c_conv_c, _CFG.c_entr_c, _CFG.c_sheared_plume, jb_eps))
    Ri_f = depth * S2 * w / (Jb + jb_eps)
    exp = max(_CFG.c_conv_c * w ** 3 / (Jb + jb_eps) * (1 - _CFG.c_sheared_plume * Ri_f), 0.0)
    np.testing.assert_allclose(got, exp, rtol=1e-12)
    # entraining: Jb>jb_eps & N2>0 & N2_above<0.
    N2e = 1e-6
    got_e = float(_catke._convective_length(_a(w), _a(N2e), _a(-1e-5), _a(S2), _a(depth), _a(Jb),
                                       _CFG.c_conv_c, _CFG.c_entr_c, _CFG.c_sheared_plume, jb_eps))
    np.testing.assert_allclose(got_e, _CFG.c_entr_c * Jb / (w * N2e + jb_eps), rtol=1e-12)
    # neither (stable, Jb<0) -> 0.
    got0 = float(_catke._convective_length(_a(w), _a(1e-4), _a(1e-4), _a(S2), _a(depth), _a(-1e-8),
                                      _CFG.c_conv_c, _CFG.c_entr_c, _CFG.c_sheared_plume, jb_eps))
    assert got0 == 0.0


# ---------------------------------------------------------------------------
# 2. Public diffusivities.
# ---------------------------------------------------------------------------
def test_diffusivities_stable_full_chain():
    p = _stable_profile()
    K_u, K_c, K_e = (np.asarray(x) for x in _call_k(p))
    o_u, o_c, o_e = _k_oracle(np.asarray(p["e"]), np.asarray(p["N2"]),
                              np.asarray(p["N2_above"]), np.asarray(p["S2"]),
                              np.asarray(p["depth"]), np.asarray(p["height_above_bottom"]),
                              float(p["H"]), float(p["Jb"]), _CFG)
    np.testing.assert_allclose(K_u, o_u, rtol=1e-12)
    np.testing.assert_allclose(K_c, o_c, rtol=1e-12)
    np.testing.assert_allclose(K_e, o_e, rtol=1e-12)


def test_diffusivities_convective_full_chain():
    p = _convective_profile()
    K_u, K_c, K_e = (np.asarray(x) for x in _call_k(p))
    o_u, o_c, o_e = _k_oracle(np.asarray(p["e"]), np.asarray(p["N2"]),
                              np.asarray(p["N2_above"]), np.asarray(p["S2"]),
                              np.asarray(p["depth"]), np.asarray(p["height_above_bottom"]),
                              float(p["H"]), float(p["Jb"]), _CFG)
    np.testing.assert_allclose(K_u, o_u, rtol=1e-12)
    np.testing.assert_allclose(K_c, o_c, rtol=1e-12)
    np.testing.assert_allclose(K_e, o_e, rtol=1e-12)


def test_diffusivities_nonnegative_and_convective_enhances():
    ks = _call_k(_stable_profile())
    kc = _call_k(_convective_profile())
    for x in ks + kc:
        assert np.all(np.asarray(x) >= 0.0)
    # convective column mixes tracers more strongly than the stable one (top iface).
    assert float(kc[1][0]) > float(ks[1][0])


def test_diffusivity_caps_fire():
    # Huge TKE -> l*w* would be large; each finite cap clamps its OWN output.
    p = dict(_convective_profile(), e=_a([10.0, 10.0]))    # w* ~ 3.16 m/s
    cfg = _CFG._replace(maximum_viscosity=0.03,
                        maximum_tracer_diffusivity=0.05,
                        maximum_tke_diffusivity=0.07)
    K_u, K_c, K_e = (np.asarray(x) for x in _call_k(p, cfg))
    for K, cap in ((K_u, 0.03), (K_c, 0.05), (K_e, 0.07)):
        assert np.all(K <= cap + 1e-15)
        assert np.any(K >= cap - 1e-12)               # each cap actually reached


def _entrainment_profile():
    # entraining: Jb>jb_eps, N2>0 (stable here), N2_above<0 (unstable above).
    return dict(e=_a([1e-2, 1e-2]), N2=_a([1e-4, 1e-4]), N2_above=_a([-1e-5, -1e-5]),
                S2=_a([1e-5, 1e-5]), depth=_a([5.0, 5.0]),
                height_above_bottom=_a([95.0, 95.0]), H=_a(100.0), Jb=_a(1e-3))


def test_diffusivities_entrainment_regime_and_gate():
    p = _entrainment_profile()
    K_u, K_c, K_e = (np.asarray(x) for x in _call_k(p))
    o = _k_oracle(np.asarray(p["e"]), np.asarray(p["N2"]), np.asarray(p["N2_above"]),
                  np.asarray(p["S2"]), np.asarray(p["depth"]),
                  np.asarray(p["height_above_bottom"]), float(p["H"]), float(p["Jb"]), _CFG)
    np.testing.assert_allclose(K_c, o[1], rtol=1e-12)          # tracer entrainment path
    # GATE: N2_above >= 0 disables the entrainment length -> tracer K drops.
    p_off = dict(p, N2_above=_a([1e-6, 1e-6]))
    K_c_off = np.asarray(_call_k(p_off)[1])
    assert np.all(K_c > K_c_off + 1e-12)                       # entrainment enhanced K_c


def test_h_caps_mixing_length():
    # Shallow column: min(H, ell) must clamp l_X to H, so K_X = H * w*.  Removing
    # the min(H, ...) would give a larger K.
    p = dict(_convective_profile(), H=_a(2.0), e=_a([1.0, 1.0]))   # w* = 1
    K_u, K_c, K_e = (np.asarray(x) for x in _call_k(p))
    w = float(np.sqrt(1.0))
    for K in (K_u, K_c, K_e):
        np.testing.assert_allclose(K, 2.0 * w, rtol=1e-12)        # l_X capped at H=2


def test_per_variable_convective_coefficient_routing():
    # Shallow depth -> the convective length wins for ALL of u/c/e (ell_conv >
    # sigma*ell_stable), so each c_conv_X is the ACTIVE coefficient for its own
    # variable.  Doubling c_conv_X raises K_X and leaves the OTHER two unchanged.
    p = dict(_convective_profile(), depth=_a([0.5, 1.0]))
    base = [np.asarray(x) for x in _call_k(p)]
    for idx, field in ((0, "c_conv_u"), (1, "c_conv_c"), (2, "c_conv_e")):
        cfg_hi = _CFG._replace(**{field: 2.0 * getattr(_CFG, field)})
        hi = [np.asarray(x) for x in _call_k(p, cfg_hi)]
        assert np.all(hi[idx][0] > base[idx][0] + 1e-12)          # this variable's K rose
        for other in (0, 1, 2):
            if other != idx:
                np.testing.assert_allclose(hi[other], base[other], rtol=1e-12)  # others unchanged


# ---------------------------------------------------------------------------
# 3. Dissipation + surface flux.
# ---------------------------------------------------------------------------
def test_dissipation_rate_physical_branch():
    p = _stable_profile()
    omega = np.asarray(catke_dissipation_rate(
        p["e"], p["N2"], p["N2_above"], p["S2"], p["depth"],
        p["height_above_bottom"], p["H"], p["Jb"], _CFG))
    # Independent l_D = min(H, max(l_stable/sigma_D, l_conv_diss)); omega = sqrt|e|/l_D.
    e = np.asarray(p["e"])
    N2 = np.asarray(p["N2"])
    S2 = np.asarray(p["S2"])
    w = _w_star(e, _CFG)
    Ri = _ri(N2, S2)
    sig = _sigma(Ri, _CFG.c_un_diss, _CFG.c_lo_diss, _CFG.c_hi_diss, _CFG)
    ell_star = _l_stable(w, N2, np.asarray(p["depth"]),
                         np.asarray(p["height_above_bottom"]), _CFG) / np.maximum(sig, _EPS_LEN)
    ell_h = _l_conv(w, N2, np.asarray(p["N2_above"]), S2, np.asarray(p["depth"]),
                    float(p["Jb"]), _CFG.c_conv_diss, _CFG.c_entr_diss, _CFG)
    ell_D = np.minimum(float(p["H"]), np.maximum(ell_star, ell_h))
    np.testing.assert_allclose(omega, np.sqrt(np.abs(e)) / np.maximum(ell_D, _EPS_LEN), rtol=1e-12)


def test_dissipation_negative_tke_numerical_branch():
    p = _stable_profile()
    p = dict(p, e=_a([-1e-4, 5e-4]))              # first cell has spurious e<0
    omega = np.asarray(catke_dissipation_rate(
        p["e"], p["N2"], p["N2_above"], p["S2"], p["depth"],
        p["height_above_bottom"], p["H"], p["Jb"], _CFG))
    np.testing.assert_allclose(omega[0], 1.0 / _CFG.negative_tke_damping_time_s, rtol=1e-12)


def test_surface_tke_flux_form_and_signs():
    u_star, w_conv3 = 0.01, 2e-6
    q = float(catke_surface_tke_flux(_a(u_star), _a(w_conv3), _CFG))
    np.testing.assert_allclose(
        q, -_CFG.c_w_ustar * u_star ** 3 - _CFG.c_w_conv * w_conv3, rtol=1e-12)
    assert q < 0.0                                # flux is INTO the column (downward)


# ---------------------------------------------------------------------------
# 4. Config plumbing + differentiability.
# ---------------------------------------------------------------------------
def test_config_plumbing():
    p = _convective_profile()
    base = _call_k(p, _CFG)
    # 2x tracer convective coefficient -> larger tracer convective length -> larger K_c.
    hi = _call_k(p, _CFG._replace(c_conv_c=2.0 * _CFG.c_conv_c))
    assert float(hi[1][0]) > float(base[1][0])
    # surface-flux coefficients plumb through:
    q1 = float(catke_surface_tke_flux(_a(0.02), _a(1e-6), _CFG))
    cfg_w = _CFG._replace(c_w_ustar=2.0 * _CFG.c_w_ustar)
    q2 = float(catke_surface_tke_flux(_a(0.02), _a(1e-6), cfg_w))
    assert q2 < q1                               # stronger shear-driven downward flux


def test_differentiable_through_convective_branch():
    p = _convective_profile()

    def loss(e):
        K_u, K_c, K_e = catke_diffusivities(
            e, p["N2"], p["N2_above"], p["S2"], p["depth"],
            p["height_above_bottom"], p["H"], p["Jb"], _CFG)
        return jnp.sum(K_u + K_c + K_e)
    g = jax.grad(loss)(p["e"])
    assert jnp.all(jnp.isfinite(g))
