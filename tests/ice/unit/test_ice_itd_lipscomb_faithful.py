"""Lipscomb (2001) ITD piecewise-linear remap FAITHFULNESS + conservation tests.

``ice/itd.py::lipscomb_2001_remap`` implements the Lipscomb (2001) linear
remapping of the ice-thickness distribution: it fits a linear sub-distribution
``g(h)`` within each category, displaces the inter-category boundaries by the
interpolated growth rate ``dt·dh/dt``, and re-integrates ``g`` over the FIXED
category bins.  It was entirely UNTESTED (no ``tests/ice/.../*itd*`` suite).

What these tests establish, in order of authority:

1. TRUTH TIER — CONSERVATION.  The central published claim of Lipscomb (2001)
   is that the remap conserves ice area, ice volume, salt mass, snow volume, and
   pond volume to machine precision.  Pinned to rel 1e-12 across grow / melt /
   mixed / clip-saturating growth (this outranks form-matching per CLAUDE.md).

2. ZERO-GROWTH IDENTITY.  ``h_new == h_old`` ⇒ ``dh/dt = 0`` ⇒ no boundary
   displacement ⇒ each category re-integrates to itself, so the remap is the
   identity to round-off.

3. CROSS-IMPLEMENTATION REFERENCE.  A plain-NumPy re-derivation of the column
   kernel (``_lipscomb_ref``) pins ALL SIX outputs (h, a, T, S, V_snow, V_pond)
   per FIXED bin, so a wrong per-bin SPLIT is caught where global conservation
   alone (which a degenerate "dump everything in one bin" also satisfies) cannot.
   It is EXACT center-anchored Lipscomb in the central third; in saturation it
   mirrors the kernel's η-clip + renormalisation (see tier 4), so it is a
   NumPy-vs-JAX regression oracle for THIS kernel, not a full published-spec
   oracle there — complementary to the algorithm-agnostic conservation +
   directional pins.

4. CENTRAL-THIRD RECONSTRUCTION + DEPARTURE.  ``_LIPSCOMB_G1_COEFF == 12`` is the
   Lipscomb first-moment slope (``g1 = 12·a·η/H³``).  ``g(h) = a/H + g1·(h −
   centre)`` is anchored at the BIN CENTRE, so in the central third (``|η| ≤ H/6``)
   BOTH moments are preserved analytically and the ``±H/6`` clip is the correct
   positivity bound (``g`` touches zero at ``η = ±H/6``, never negative) —
   ``test_lipscomb_reconstruction_moments_and_positivity`` pins this.  DEPARTURE
   (documented in the module Faithfulness note, tested via the ``clip`` saturating
   conservation case): outside the central third the kernel clips ``η`` and
   renormalises area/volume rather than using Lipscomb's exact cutoff-support
   triangle — total conservation holds but the transfers are not moments of one
   ``g`` in saturation; the clip is kept for robustness against degenerate
   displaced bins.  (A still earlier mean-anchored ``G0 = a/H`` at ``h̄`` broke the
   zeroth moment and let ``g`` go negative near ``η = H/6``; fixed to center
   anchoring during codex review.)
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

_ENTRY_X64 = jax.config.read("jax_enable_x64")


@pytest.fixture(autouse=True)
def _force_x64():
    """Per-test float64 for the conservation/reference pins; restore the
    process-entry state in finally.  ``category_bounds`` is lru_cached and its
    result dtype follows the x64 flag, so clear it both on entry (a prior float32
    run must not leak float32 bounds into these x64 pins) and on teardown
    (float32 AD checks must not inherit float64 bounds).  ``upper_bounds`` has no
    separate cache — it delegates to ``category_bounds``."""
    from legoesm.ice.itd import category_bounds
    jax.config.update("jax_enable_x64", True)
    category_bounds.cache_clear()
    try:
        yield
    finally:
        category_bounds.cache_clear()
        jax.config.update("jax_enable_x64", _ENTRY_X64)


from legoesm import constants                                            # noqa: E402
from legoesm.ice.itd import (                                            # noqa: E402
    lipscomb_2001_remap, category_bounds, upper_bounds, _LIPSCOMB_G1_COEFF,
)  # upper_bounds imported for the bounds canary below

_O_G1_COEFF = 12.0   # Lipscomb (2001) first-moment linear-slope coefficient
_EPS_WIDTH = 1.0e-6  # displaced-bin numerical width guard (mirrors the kernel)


def _lipscomb_ref(h_old, a_old, h_new, a_new, T_new, S_new, Vs, Vp, lo, hi, dt):
    """NumPy re-derivation of the kernel: center-anchored Lipscomb (2001)
    g(h) = a/H + g1*(h - centre), g1 = 12*a*eta/H^3, integrated over the fixed
    bins, with the η-clip + per-source renormalisation.  EXACT Lipscomb in the
    central third; in saturation it mirrors the kernel's clip+renormalise (a
    documented approximation of Lipscomb's cutoff support).  A NumPy-vs-JAX
    regression oracle for the per-bin split, complementary to the
    algorithm-agnostic conservation pins.  1-D arrays."""
    n = len(h_new)
    Vnew = h_new * a_new
    dhdt = (h_new - h_old) / dt
    has = a_new > 1e-10
    cf = 0.5 * (lo + hi)
    hbar = np.where(has, h_new, cf)
    hl, hr = hbar[:-1], hbar[1:]
    dl, dr = dhdt[:-1], dhdt[1:]
    il, ir = has[:-1], has[1:]
    den = hr - hl
    sd = np.where(np.abs(den) > 1e-10, den, 1.0)
    w = np.clip(np.where(np.abs(den) > 1e-10, (lo[1:] - hl) / sd, 0.5), 0.0, 1.0)
    both = il & ir
    ol = il & (~ir)
    orr = (~il) & ir
    dbi = np.where(both, (1 - w) * dl + w * dr,
                   np.where(ol, dl, np.where(orr, dr, 0.0)))
    dbd = np.concatenate([[0.0], dbi, [0.0]])
    hL = lo + dt * dbd[:n]
    hR = hi + dt * dbd[1:]
    hL[0] = 0.0
    hR[-1] = max(hR[-1], hi[-1])
    dlo, dhi = lo[0], hi[-1]
    hL = np.clip(hL, dlo, dhi - _EPS_WIDTH)
    hR = np.clip(hR, dlo + _EPS_WIDTH, dhi)
    hR = np.maximum(hR, hL + _EPS_WIDTH)
    hL = np.minimum(hL, hR - _EPS_WIDTH)
    H = hR - hL
    cd = 0.5 * (hL + hR)
    eta = np.clip(h_new - cd, -H / 6.0, H / 6.0)
    anchor = cd                                  # BIN CENTRE (exact Lipscomb)
    G0 = np.where(has, a_new / H, 0.0)
    G1 = np.where(has, _O_G1_COEFF * eta * a_new / H ** 3, 0.0)
    loj, hij = lo[:, None], hi[:, None]
    hLk, hRk = hL[None, :], hR[None, :]
    ao = np.maximum(loj, hLk)
    bo = np.minimum(hij, hRk)
    ow = np.maximum(bo - ao, 0.0)
    G0k, G1k, hbk = G0[None, :], G1[None, :], anchor[None, :]
    ia = np.where(ow > 0, np.maximum(G0k * ow + G1k * ((bo - hbk) ** 2 - (ao - hbk) ** 2) / 2, 0.0), 0.0)
    iv = np.where(ow > 0, np.maximum((G0k - G1k * hbk) * (bo ** 2 - ao ** 2) / 2 + G1k * (bo ** 3 - ao ** 3) / 3, 0.0), 0.0)
    Vks, Aks = np.sum(iv, axis=0), np.sum(ia, axis=0)
    Vsc = np.where((Vks > 1e-30) & has, Vnew / np.where(Vks > 1e-30, Vks, 1), 0.0)
    Asc = np.where((Aks > 1e-30) & has, a_new / np.where(Aks > 1e-30, Aks, 1), 0.0)
    iv, ia = iv * Vsc[None, :], ia * Asc[None, :]
    ar = np.clip(np.sum(ia, axis=1), 0.0, 1.0)
    Vr = np.maximum(np.sum(iv, axis=1), 0.0)
    asf = np.where(ar > 1e-30, ar, 1.0)
    hr_ = np.where(ar > 1e-30, Vr / asf, 0.0)
    invV = np.where(Vr > 1e-30, 1 / np.where(Vr > 1e-30, Vr, 1), 0.0)
    Tr = np.where(Vr > 1e-30, np.sum(T_new[None, :] * iv, axis=1) * invV, constants.T_freeze_ocean)
    Sr = np.where(Vr > 1e-30, np.sum(S_new[None, :] * iv, axis=1) * invV, 0.0)
    Vsf = np.where(Vnew > 1e-30, Vs / np.where(Vnew > 1e-30, Vnew, 1), 0.0)
    Vpf = np.where(Vnew > 1e-30, Vp / np.where(Vnew > 1e-30, Vnew, 1), 0.0)
    Vsr = np.sum(iv * Vsf[None, :], axis=1)
    Vpr = np.sum(iv * Vpf[None, :], axis=1)
    return hr_, ar, Tr, Sr, Vsr, Vpr


def _displaced_bins_eta(h_old, a_new, h_new, lo, hi, dt, eps=1e-6):
    """Replicate the kernel's displaced-bin construction and return
    (hL, hR, H, eta, centre) so tests can assert which regime (central |eta|<=H/6
    vs saturated) each populated category lands in."""
    N = len(h_new)
    dhdt = (h_new - h_old) / dt
    has = a_new > 1e-10
    hb = np.where(has, h_new, 0.5 * (lo + hi))
    hl, hr = hb[:-1], hb[1:]
    dl, dr = dhdt[:-1], dhdt[1:]
    il, ir = has[:-1], has[1:]
    den = hr - hl
    sd = np.where(np.abs(den) > 1e-10, den, 1.0)
    w = np.clip(np.where(np.abs(den) > 1e-10, (lo[1:] - hl) / sd, 0.5), 0.0, 1.0)
    both = il & ir
    dbi = np.where(both, (1 - w) * dl + w * dr,
                   np.where(il & (~ir), dl, np.where((~il) & ir, dr, 0.0)))
    dbd = np.concatenate([[0.0], dbi, [0.0]])
    hL = lo + dt * dbd[:N]
    hR = hi + dt * dbd[1:]
    hL[0] = 0.0
    hR[-1] = max(hR[-1], hi[-1])
    hL = np.clip(hL, lo[0], hi[-1] - eps)
    hR = np.clip(hR, lo[0] + eps, hi[-1])
    hR = np.maximum(hR, hL + eps)
    hL = np.minimum(hL, hR - eps)
    H = hR - hL
    c = 0.5 * (hL + hR)
    return hL, hR, H, h_new - c, c


_N = 5
# CICE standard 5-category bounds as explicit float64 (import-time x64 state must
# not decide the reference's precision); matches category_bounds/upper_bounds.
_LO = np.array([0.0, 0.6, 1.4, 2.4, 3.6], dtype=np.float64)
_HI = np.array([0.6, 1.4, 2.4, 3.6, 100.0], dtype=np.float64)
_DT = 86400.0
_A = np.array([0.2, 0.25, 0.2, 0.1, 0.05])
_S = np.array([5.0, 5.0, 4.0, 3.0, 2.0])
_T = np.array([260.0, 255.0, 252.0, 250.0, 248.0])
_VSN = np.array([0.02, 0.03, 0.02, 0.01, 0.005])
_VPN = np.array([0.01, 0.005, 0.0, 0.0, 0.0])

_CASES = {
    "grow":  ([0.3, 1.0, 1.9, 3.0, 5.0], [0.45, 1.2, 2.1, 3.3, 5.4]),
    "melt":  ([0.5, 1.2, 2.1, 3.3, 5.4], [0.35, 1.0, 1.9, 3.0, 5.0]),
    "mixed": ([0.4, 1.0, 2.0, 3.0, 5.0], [0.55, 0.9, 2.2, 2.8, 5.5]),
    "clip":  ([0.3, 1.0, 1.9, 3.0, 5.0], [0.59, 1.39, 2.39, 3.59, 9.0]),  # big growth -> eta hits +-H/6
}


def _run(h_old, h_new, a=_A):
    r = lipscomb_2001_remap(
        jnp.array(np.asarray(h_old)[None]), jnp.array(np.asarray(a)[None]),
        jnp.array(np.asarray(h_new)[None]), jnp.array(np.asarray(a)[None]),
        _N, _DT, T_new=jnp.array(_T[None]), S_new=jnp.array(_S[None]),
        V_snow_new=jnp.array(_VSN[None]), V_pond_new=jnp.array(_VPN[None]))
    return {k: np.array(v)[0] for k, v in r.items()}


# --- tier 1: conservation truth tier -------------------------------------------

@pytest.mark.parametrize("case", list(_CASES))
def test_lipscomb_conserves_area_volume_salt_snow_pond(case):
    """Lipscomb (2001)'s central claim: the remap conserves ice area, ice volume,
    salt mass, snow volume, and pond volume to machine precision."""
    ho, hn = _CASES[case]
    ho, hn = np.array(ho), np.array(hn)
    r = _run(ho, hn)
    assert np.sum(r["a"]) == pytest.approx(np.sum(_A), rel=1e-12)
    assert np.sum(r["a"] * r["h"]) == pytest.approx(np.sum(_A * hn), rel=1e-12)
    assert np.sum(r["S"] * r["a"] * r["h"]) == pytest.approx(np.sum(_S * _A * hn), rel=1e-12)
    assert np.sum(r["V_snow"]) == pytest.approx(np.sum(_VSN), rel=1e-12)
    assert np.sum(r["V_pond"]) == pytest.approx(np.sum(_VPN), rel=1e-12)


# --- tier 2: zero-growth identity ----------------------------------------------

def test_zero_growth_is_identity():
    """h_new == h_old -> dh/dt = 0 -> boundaries do not displace -> each category
    re-integrates to itself: the remap is the identity to round-off."""
    hn = np.array([0.45, 1.2, 2.1, 3.3, 5.4])
    r = _run(hn, hn)
    assert r["h"] == pytest.approx(hn, rel=1e-12, abs=1e-13)
    assert r["a"] == pytest.approx(_A, rel=1e-12, abs=1e-13)
    assert r["S"] == pytest.approx(_S, rel=1e-12, abs=1e-13)
    assert r["T"] == pytest.approx(_T, rel=1e-12, abs=1e-13)


# --- tier 3: separate reference implementation (per-bin split) ------------------

@pytest.mark.parametrize("case", list(_CASES))
def test_matches_separate_reference_impl(case):
    """All six per-bin outputs match a separate NumPy re-derivation of the column
    kernel — pins the per-FIXED-bin SPLIT, not just the conserved totals."""
    ho, hn = _CASES[case]
    ho, hn = np.array(ho), np.array(hn)
    r = _run(ho, hn)
    ref = _lipscomb_ref(ho, _A, hn, _A, _T, _S, _VSN, _VPN, _LO, _HI, _DT)
    for key, rf in zip(("h", "a", "T", "S", "V_snow", "V_pond"), ref):
        assert r[key] == pytest.approx(rf, rel=1e-10, abs=1e-12), f"{case}:{key}"


def test_central_third_matches_independent_quadrature():
    """Exercise the KERNEL's actual anchor + overlap in the exact (central-third)
    regime against an INDEPENDENT integration method (fine trapezoidal quadrature
    of the exact-Lipscomb g over each fixed bin, vs the kernel's closed-form
    integral).  Two adjacent populated categories with differential growth so both
    land central (|eta| <= H/6) and their displaced bins CROSS fixed-bin
    boundaries (ice splits across >=2 fixed bins)."""
    a = np.array([0.0, 0.3, 0.25, 0.0, 0.0])
    ho = np.array([0.0, 0.95, 2.0, 0.0, 0.0])
    hn = np.array([0.0, 1.12, 2.12, 0.0, 0.0])
    hL, hR, H, eta, c = _displaced_bins_eta(ho, a, hn, _LO, _HI, _DT)
    # both populated categories are strictly in the central third, with nonzero eta
    for k in (1, 2):
        assert 0.02 < abs(eta[k]) <= H[k] / 6.0, f"cat{k} eta/H={eta[k]/H[k]}"
    assert hL[1] < 1.4 < hR[1] and hL[2] < 2.4 < hR[2]        # cross fixed-bin edges
    a_remap = _run(ho, hn, a=a)["a"]
    expected = np.zeros(_N)
    for k in (1, 2):                                          # exact-Lipscomb g, quadrature
        g1 = _O_G1_COEFF * eta[k] * a[k] / H[k] ** 3
        g0 = a[k] / H[k] - g1 * c[k]
        for j in range(_N):
            lo_o, hi_o = max(_LO[j], hL[k]), min(_HI[j], hR[k])
            if hi_o > lo_o:
                xs = np.linspace(lo_o, hi_o, 40001)
                ys = g0 + g1 * xs
                # manual trapezoidal sum (portable across numpy 1.x/2.x; a different
                # expression than the kernel's analytic (b^2-a^2)/2 integral)
                expected[j] += float(np.sum(0.5 * (ys[:-1] + ys[1:]) * np.diff(xs)))
    assert a_remap == pytest.approx(expected, rel=1e-6, abs=1e-8)


def test_clip_case_saturates():
    """Guard the reference/departure coverage: the ``clip`` case must genuinely
    drive at least one populated category OUT of the central third (|eta| > H/6),
    exercising the eta-clip + renormalisation path (else it silently tests only the
    central-third form)."""
    ho, hn = np.array(_CASES["clip"][0]), np.array(_CASES["clip"][1])
    _, _, H, eta, _ = _displaced_bins_eta(ho, _A, hn, _LO, _HI, _DT)
    saturated = [k for k in range(_N) if _A[k] > 1e-10 and abs(eta[k]) > H[k] / 6.0]
    assert saturated, f"clip case did not saturate: eta/H={[eta[k]/H[k] for k in range(_N)]}"


def test_degenerate_bins_still_conserve():
    """Deferral evidence (regression pin): divergent extreme melt produces
    DEGENERATE displaced bins (H -> 0) with h_new outside them — the case where an
    exact cutoff-support triangle (g1 ~ 1/H^3) would lose mass.  The retained
    eta-clip + renormalisation must keep all outputs FINITE and conserve area,
    volume, salt, snow, and pond to machine precision."""
    ho = np.array([0.55, 1.35, 2.35, 3.55, 7.0])
    hn = np.array([0.3, 0.65, 1.5, 2.5, 3.7])
    # confirm the geometry really is degenerate (some populated cat has H ~ 0)
    _, _, H, _, _ = _displaced_bins_eta(ho, _A, hn, _LO, _HI, _DT)
    assert np.min(H) < 1e-3, f"expected a collapsed displaced bin, min H={np.min(H)}"
    r = _run(ho, hn)
    for k in ("h", "a", "T", "S", "V_snow", "V_pond"):
        assert np.all(np.isfinite(r[k])), f"{k} not finite"
    assert np.sum(r["a"]) == pytest.approx(np.sum(_A), rel=1e-12)
    assert np.sum(r["a"] * r["h"]) == pytest.approx(np.sum(_A * hn), rel=1e-12)
    assert np.sum(r["S"] * r["a"] * r["h"]) == pytest.approx(np.sum(_S * _A * hn), rel=1e-12)
    assert np.sum(r["V_snow"]) == pytest.approx(np.sum(_VSN), rel=1e-12)
    assert np.sum(r["V_pond"]) == pytest.approx(np.sum(_VPN), rel=1e-12)


# --- non-vacuity: the remap actually MOVES mass in the right direction ----------

def test_growth_shifts_mass_to_thicker_categories():
    """Directional SMOKE test (rejects the identity and a fixed one-bin dump, but
    not every wrong split): uniform strong growth moves areal mass UP the
    thickness axis (area-weighted mean category index increases); melt moves it
    DOWN.  The quantitative per-bin split is pinned by the reference test."""
    idx = np.arange(_N)
    base_h = np.array([0.3, 1.0, 1.9, 3.0, 5.0])
    grow_h = np.array([0.59, 1.39, 2.39, 3.59, 9.0])
    melt_h = np.array([0.1, 0.5, 1.0, 1.9, 3.0])
    r_grow, r_melt = _run(base_h, grow_h), _run(base_h, melt_h)
    mean_idx = lambda r: np.sum(idx * r["a"]) / np.sum(r["a"])
    base_r = _run(base_h, base_h)
    assert mean_idx(r_grow) > mean_idx(base_r)   # growth: mass climbs
    assert mean_idx(r_melt) < mean_idx(base_r)   # melt: mass drops


# --- tier 4: published form + centering departure ------------------------------

def test_g1_coefficient_and_bounds_match_module():
    """The published first-moment slope coefficient (12) and the hardcoded CICE
    bounds used by the reference are pinned to the module's own values."""
    assert _LIPSCOMB_G1_COEFF == _O_G1_COEFF == 12.0
    assert np.array(category_bounds(_N)) == pytest.approx(_LO, rel=0, abs=1e-12)
    assert np.array(upper_bounds(_N)) == pytest.approx(_HI, rel=0, abs=1e-12)


@pytest.mark.parametrize("eta_frac", [-1.0, -0.5, 0.0, 0.5, 1.0])   # eta = eta_frac * H/6
def test_lipscomb_reconstruction_moments_and_positivity(eta_frac):
    """The kernel's BIN-CENTRE-anchored reconstruction g(h) = a/H + g1*(h-centre),
    g1 = 12*a*eta/H^3, preserves BOTH moments analytically (∫g dh = a, ∫h·g dh =
    a·h̄) AND stays NON-NEGATIVE across the whole displaced bin for |eta| <= H/6 —
    so the ±H/6 clip is the correct positivity bound.  (A mean-anchored G0 = a/H at
    h̄ would break the zeroth moment and drive g negative near eta = H/6; this pins
    the corrected form.)"""
    a, center, H = 0.25, 1.4, 0.8
    eta = eta_frac * H / 6.0
    hbar = center + eta
    g1 = _O_G1_COEFF * a * eta / H ** 3
    hL, hR = center - H / 2, center + H / 2

    # Center-anchored g(h) = a/H + g1*(h - center)  (== g0 + g1*h, g0 = a/H - g1*center).
    g = lambda h: a / H + g1 * (h - center)
    I0 = a / H * H + g1 * ((hR - center) ** 2 - (hL - center) ** 2) / 2
    I1 = (a / H - g1 * center) * (hR ** 2 - hL ** 2) / 2 + g1 * (hR ** 3 - hL ** 3) / 3
    assert I0 == pytest.approx(a, rel=1e-13)              # zeroth moment = area (exact)
    assert I1 / I0 == pytest.approx(hbar, rel=1e-13)      # first moment = mean (exact)
    # positivity on the closed bin: linear, so the min is at an endpoint
    assert g(hL) >= -1e-13 and g(hR) >= -1e-13
    if abs(abs(eta_frac) - 1.0) < 1e-12:                  # at the clip edge g touches 0
        assert min(g(hL), g(hR)) == pytest.approx(0.0, abs=1e-13)


# --- edge cases ----------------------------------------------------------------

def test_empty_ice_returns_zero():
    """All-open-water input (a_new = 0) -> zero area and volume everywhere."""
    r = _run(np.array([0.3, 1.0, 1.9, 3.0, 5.0]), np.array([0.45, 1.2, 2.1, 3.3, 5.4]),
             a=np.zeros(_N))
    assert np.all(r["a"] == 0.0)
    assert np.all(r["h"] * r["a"] == 0.0)


# --- AD-safety -----------------------------------------------------------------

def test_lipscomb_remap_grad_finite_x64_and_float32():
    """grad of a scalar of the remapped state wrt the post-thermo thickness is
    finite through the boundary interp, η-clip, overlap max/min, and the divisions
    — in x64 AND float32.  Finite-AD coverage through nonsmooth clip/max kinks,
    not a differentiability proof."""
    def _loss(hn, ho, a, T):
        r = lipscomb_2001_remap(ho, a, hn, a, _N, _DT, T_new=T)
        return jnp.sum(r["h"] ** 2 + r["a"] ** 2)

    def _check(dtype):
        ho = jnp.asarray(np.array([0.3, 1.0, 1.9, 3.0, 5.0])[None], dtype=dtype)
        a = jnp.asarray(_A[None], dtype=dtype)
        T = jnp.asarray(_T[None], dtype=dtype)
        hn = jnp.asarray(np.array([0.45, 1.2, 2.1, 3.3, 5.4])[None], dtype=dtype)
        g = jax.grad(_loss)(hn, ho, a, T)
        assert g.dtype == dtype                    # gradient genuinely at this precision
        assert bool(jnp.all(jnp.isfinite(g)))
        assert float(jnp.sum(jnp.abs(g))) > 0.0    # non-trivial dependence

    _check(jnp.float64)
    category_bounds.cache_clear()
    jax.config.update("jax_enable_x64", False)
    _check(jnp.float32)   # autouse fixture clears the cache + restores x64 afterwards
