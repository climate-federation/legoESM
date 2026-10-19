"""Oracle-faithfulness pins for the Bou-Zeid-Meneveau-Parlange (2005) LASD SGS.

Oracle: the jax-alfa Locally-Averaged Scale-Dependent Dynamic model
``DynamicSGS_LASDD_SM.LASDD`` (Sukanta Basu, github Sukantabasu/jax-alfa) — the
reference the module cites — implementing Bou-Zeid, Meneveau & Parlange (2005),
Phys. Fluids 17, 025105, with the Lilly (1992) least-squares error functional.
``legoesm...turbulence.lasd_core`` is a faithful JAX port; these pins lock its
closed forms to round-off (rel 1e-12) against an INDEPENDENT numpy reimplementation
typed from the jax-alfa oracle (``LASDD``/``Filtering_Level1/2``/``Utilities.Roots``/
``Utilities.Imfilter``), NOT read back from ``lasd_core`` — so they canary the
STRUCTURE, not a copy.

Faithful forms pinned:
  - spectral test filter: sharp cutoff at ``round(N/(2·FGR·TFR))`` (FGR=1) — fold the
    full-fft axis, halve the rfft axis (oracle ``Filtering_Level1/2``).
  - 3×3 periodic box average /9 (oracle ``Utilities.Imfilter``).
  - the max-real-root-in-(0,5) β rule, default 1.0 (oracle ``ComputeBeta1``; solved by
    exhaustive real-root isolation, not the oracle's Laguerre ``Roots``) — pinned against ``numpy.roots`` (a DIFFERENT algorithm:
    companion-matrix eigenvalues), so the agreement is non-circular.
  - the LASDD assembly: L_ij/Q_ij Germano stresses at 2Δ/4Δ, the a1..e2 plane-mean
    contractions with the oracle's exact ``2L²``/``TFR^n``/``8L⁴`` weights, the
    aa..ff quintic coefficients, M_ij (β scale-dependence), LM/MM, box-average, and
    the ``C_s²`` clip mask (MM~0 | C_s²<0 | C_s²>cs_max²).

Departures (legoESM adaptations, documented — the scheme is otherwise faithful):
  - ``cs_max`` is a CONFIGURABLE clip extension: the default cs_max=1 reproduces the
    oracle mask exactly, but cs_max != 1 changes the C_s² ceiling (a closure change).
  - C_s² uses a RAW divide after flooring the denominator (``LMx/where(|MMx|<1e-10,
    1e-10, MMx)``) — NOT the project ``safe_divide`` helper (forward-identical: the |MM|<1e-10
    cells are masked to 0).
  - the β root by exhaustive isolation with an implicit-function JVP (the oracle's
    8-guess Laguerre mis-solved the tiny Germano quintics of a real CBL; see the
    ``test_beta_*`` defect tests).
  - a ``(ny, nx, nz)`` layout (oracle ``(nx, ny, nz)``) — a transpose convention — and
    an optional MPI y-slab distributed path (serial path pinned here).

The β-solver is pinned to 1e-6 vs numpy.roots, so the full ASSEMBLY pin reuses the
separately-pinned ``max_real_root_beta`` as a GIVEN (like the shared thermo in
the convection pins) and independently reimplements only the L/Q/M/LM/MM/mask glue at
1e-12; the solver itself is pinned to 1e-6 vs numpy.roots in its own test, AND its β on the
assembly-produced polynomials aa..ff is cross-checked vs numpy.roots (β-realism).

Differentiability: β carries the exact implicit-function derivative of the selected root
(custom JVP, higher orders exact); an all-zero (strain-free) or no-root-in-(0,5) quintic
returns the default 1.0 with zero gradient.  A near-multiple root has a large-but-finite
derivative (the true one).  Pinned by the zero-
strain, mixed dead+active level, and repeated-root (x^5) gradient tests.
"""

from __future__ import annotations

import pytest
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.turbulence.lasd_core import (
    spectral_test_filter,
    imfilter_box3,
    max_real_root_beta,
    beta_from_germano_moments,
    comp_horner,
    lasd_cs2,
    _LASD_LILLY_COEFF,
    _TFR,
)

_TFR_ORACLE = 2.0        # test-filter ratio (Bou-Zeid 2005; oracle DerivedVars TFR)
_FGR = 1.0               # filter-to-grid ratio (implicit filtering)


def _np(x):
    return np.asarray(x, dtype=np.float64)


# ---- independent numpy oracle (typed from jax-alfa LASDD / Filtering / Utilities) ----

def _np_spectral_filter(field_yxz, cut_y, cut_x):
    """Independent sharp spectral cutoff (oracle Filtering_Level1/2, legoESM layout):
    rfft2 over (y, x); keep the folded full-fft y-axis modes |k_y|<cut_y and the rfft
    x-axis modes k_x<cut_x; zero the rest; irfft2."""
    ny, nx = field_yxz.shape[0], field_yxz.shape[1]
    fh = np.fft.rfft2(field_yxz, axes=(0, 1))
    iy = np.arange(ny)
    fold_y = np.minimum(iy, ny - iy)
    mask_y = (fold_y < cut_y)[:, None, None]
    mask_x = (np.arange(fh.shape[1]) < cut_x)[None, :, None]
    fh = np.where(mask_y & mask_x, fh, 0.0)
    return np.fft.irfft2(fh, axes=(0, 1), s=(ny, nx))


def _np_box3(field_yxz):
    """Independent 3×3 periodic box average /9 (oracle Utilities.Imfilter)."""
    fx = (np.roll(field_yxz, 1, axis=1) + field_yxz + np.roll(field_yxz, -1, axis=1)) / 3.0
    return (np.roll(fx, 1, axis=0) + fx + np.roll(fx, -1, axis=0)) / 3.0


def _cutoffs(ny, nx):
    cut1y = round(ny / (2 * _FGR * _TFR_ORACLE))
    cut1x = round(nx / (2 * _FGR * _TFR_ORACLE))
    cut2y = round(ny / (2 * _FGR * _TFR_ORACLE * _TFR_ORACLE))
    cut2x = round(nx / (2 * _FGR * _TFR_ORACLE * _TFR_ORACLE))
    return cut1y, cut1x, cut2y, cut2x


def _np_beta_from_numpy_roots(aa, bb, cc, dd, ee, ff):
    """Independent β per level = max real root in (0,5) of ff x^5+ee x^4+dd x^3+cc x^2+
    bb x+aa, default 1.0 — via numpy.roots (companion-matrix eigenvalues, NOT Laguerre)."""
    nz = aa.shape[0]
    out = np.empty(nz)
    for k in range(nz):
        coeffs = np.array([ff[k], ee[k], dd[k], cc[k], bb[k], aa[k]], dtype=np.float64)
        r = np.roots(coeffs)
        real = r[np.abs(r.imag) < 1e-6].real
        valid = real[(real > 0.0) & (real < 5.0)]
        out[k] = valid.max() if valid.size else 1.0
    return out


def _lasd_cs2_oracle(uc, vc, wc, S11, S22, S33, S12, S13, S23, Smag, delta,
                     cs_max=1.0, *, beta_solver=None, force_beta=None):
    """Independent numpy LASDD assembly (oracle DynamicSGS_LASDD_SM.LASDD), legoESM
    (ny,nx,nz) layout.  Reuses the separately-pinned β-solver as a GIVEN (default: the
    module's Laguerre, which is pinned vs numpy.roots elsewhere); ``force_beta`` overrides
    β for the scale-dependence canary."""
    uc, vc, wc = _np(uc), _np(vc), _np(wc)
    S11, S22, S33 = _np(S11), _np(S22), _np(S33)
    S12, S13, S23, Smag = _np(S12), _np(S13), _np(S23), _np(Smag)
    ny, nx, nz = uc.shape
    L2 = (_np(delta) ** 2).reshape(1, 1, nz)
    TFR = _TFR_ORACLE
    c1y, c1x, c2y, c2x = _cutoffs(ny, nx)
    F1 = lambda f: _np_spectral_filter(f, c1y, c1x)   # noqa: E731
    F2 = lambda f: _np_spectral_filter(f, c2y, c2x)   # noqa: E731
    pm = lambda f: np.mean(f, axis=(0, 1))            # noqa: E731

    u_h, v_h, w_h = F1(uc), F1(vc), F1(wc)
    u_d, v_d, w_d = F2(uc), F2(vc), F2(wc)
    L11, L22, L33 = F1(uc * uc) - u_h * u_h, F1(vc * vc) - v_h * v_h, F1(wc * wc) - w_h * w_h
    L12, L13, L23 = F1(uc * vc) - u_h * v_h, F1(uc * wc) - u_h * w_h, F1(vc * wc) - v_h * w_h
    Q11, Q22, Q33 = F2(uc * uc) - u_d * u_d, F2(vc * vc) - v_d * v_d, F2(wc * wc) - w_d * w_d
    Q12, Q13, Q23 = F2(uc * vc) - u_d * v_d, F2(uc * wc) - u_d * w_d, F2(vc * wc) - v_d * w_d

    S11h, S22h, S33h = F1(S11), F1(S22), F1(S33)
    S12h, S13h, S23h = F1(S12), F1(S13), F1(S23)
    S11d, S22d, S33d = F2(S11), F2(S22), F2(S33)
    S12d, S13d, S23d = F2(S12), F2(S13), F2(S23)
    S_h = np.sqrt(2.0 * (S11h**2 + S22h**2 + S33h**2 + 2.0 * (S12h**2 + S13h**2 + S23h**2)))
    S_d = np.sqrt(2.0 * (S11d**2 + S22d**2 + S33d**2 + 2.0 * (S12d**2 + S13d**2 + S23d**2)))
    SS11h, SS22h, SS33h = F1(Smag * S11), F1(Smag * S22), F1(Smag * S33)
    SS12h, SS13h, SS23h = F1(Smag * S12), F1(Smag * S13), F1(Smag * S23)
    SS11d, SS22d, SS33d = F2(Smag * S11), F2(Smag * S22), F2(Smag * S33)
    SS12d, SS13d, SS23d = F2(Smag * S12), F2(Smag * S13), F2(Smag * S23)

    a1 = pm(2.0 * L2 * (L11 * SS11h + L22 * SS22h + L33 * SS33h
                        + 2.0 * (L12 * SS12h + L13 * SS13h + L23 * SS23h)))
    a2 = pm(2.0 * L2 * (Q11 * SS11d + Q22 * SS22d + Q33 * SS33d
                        + 2.0 * (Q12 * SS12d + Q13 * SS13d + Q23 * SS23d)))
    b1 = pm(2.0 * L2 * (TFR**2) * S_h * (L11 * S11h + L22 * S22h + L33 * S33h
                                         + 2.0 * (L12 * S12h + L13 * S13h + L23 * S23h)))
    b2 = pm(2.0 * L2 * (TFR**4) * S_d * (Q11 * S11d + Q22 * S22d + Q33 * S33d
                                         + 2.0 * (Q12 * S12d + Q13 * S13d + Q23 * S23d)))
    c1 = pm((2.0 * L2)**2 * (SS11h**2 + SS22h**2 + SS33h**2
                             + 2.0 * (SS12h**2 + SS13h**2 + SS23h**2)))
    c2 = pm((2.0 * L2)**2 * (SS11d**2 + SS22d**2 + SS33d**2
                             + 2.0 * (SS12d**2 + SS13d**2 + SS23d**2)))
    d1 = pm((4.0 * L2**2) * (TFR**4) * (S_h**2) * (S11h**2 + S22h**2 + S33h**2
                                                   + 2.0 * (S12h**2 + S13h**2 + S23h**2)))
    d2 = pm((4.0 * L2**2) * (TFR**8) * (S_d**2) * (S11d**2 + S22d**2 + S33d**2
                                                   + 2.0 * (S12d**2 + S13d**2 + S23d**2)))
    e1 = pm((8.0 * L2**2) * (TFR**2) * S_h * (S11h * SS11h + S22h * SS22h + S33h * SS33h
                                              + 2.0 * (S12h * SS12h + S13h * SS13h + S23h * SS23h)))
    e2 = pm((8.0 * L2**2) * (TFR**4) * S_d * (S11d * SS11d + S22d * SS22d + S33d * SS33d
                                              + 2.0 * (S12d * SS12d + S13d * SS13d + S23d * SS23d)))

    aa = a1 * c2 - a2 * c1
    bb = a2 * e1 - b1 * c2
    cc = b2 * c1 - a1 * e2 - a2 * d1
    dd = b1 * e2 - b2 * e1
    ee = a1 * d2 + b2 * d1
    ff = -b1 * d2
    if force_beta is not None:
        beta = np.full(nz, float(force_beta))
    elif beta_solver is not None:
        beta = beta_solver(aa, bb, cc, dd, ee, ff)
    else:
        beta = _np(max_real_root_beta(
            jnp.stack([jnp.asarray(ff), jnp.asarray(ee), jnp.asarray(dd),
                       jnp.asarray(cc), jnp.asarray(bb), jnp.asarray(aa)], -1)))
    beta3d = beta.reshape(1, 1, nz)

    T1, T2 = 2.0 * L2, 2.0 * (TFR**2) * L2
    M11 = T1 * SS11h - T2 * beta3d * S_h * S11h
    M22 = T1 * SS22h - T2 * beta3d * S_h * S22h
    M33 = T1 * SS33h - T2 * beta3d * S_h * S33h
    M12 = T1 * SS12h - T2 * beta3d * S_h * S12h
    M13 = T1 * SS13h - T2 * beta3d * S_h * S13h
    M23 = T1 * SS23h - T2 * beta3d * S_h * S23h
    LM = L11 * M11 + L22 * M22 + L33 * M33 + 2.0 * (L12 * M12 + L13 * M13 + L23 * M23)
    MM = M11**2 + M22**2 + M33**2 + 2.0 * (M12**2 + M13**2 + M23**2)

    LMx, MMx = _np_box3(LM), _np_box3(MM)
    cs2 = LMx / np.where(np.abs(MMx) < 1.0e-10, 1.0e-10, MMx)
    invalid = (np.abs(MMx) < 1.0e-10) | (cs2 < 0.0) | (cs2 > cs_max**2)
    return dict(cs2=np.where(invalid, 0.0, cs2), aa=aa, bb=bb, cc=cc, dd=dd, ee=ee,
                ff=ff, beta=beta, MM=MMx)


# ---- synthetic horizontally-periodic LES plane (non-square: ny != nx) ----

def _make_field(ny=8, nx=12, nz=4, seed=0):
    y = np.linspace(0, 2 * np.pi, ny, endpoint=False)[:, None, None]
    x = np.linspace(0, 2 * np.pi, nx, endpoint=False)[None, :, None]
    z = (np.arange(nz) + 1)[None, None, :]
    uc = np.sin(y + 0.3 * z) * np.cos(2 * x) + 0.5 * np.cos(3 * y - x)
    vc = np.cos(2 * y) * np.sin(x + 0.2 * z) - 0.4 * np.sin(y + 2 * x)
    wc = 0.3 * np.sin(y - x + 0.1 * z) + 0.2 * np.cos(4 * x)
    S11 = 0.7 * np.cos(y + x) + 0.1 * z
    S22 = 0.6 * np.sin(2 * y - x)
    S33 = -(S11 + S22)                                   # trace-free strain
    S12 = 0.5 * np.sin(y + 2 * x + 0.1 * z)
    S13 = 0.4 * np.cos(3 * y - x)
    S23 = 0.3 * np.sin(y - 2 * x)
    Smag = np.sqrt(2.0 * (S11**2 + S22**2 + S33**2 + 2.0 * (S12**2 + S13**2 + S23**2)))
    delta = np.array([10.0, 12.0, 15.0, 20.0])[:nz]
    b = np.broadcast_to
    shp = (ny, nx, nz)
    return (b(uc, shp).copy(), b(vc, shp).copy(), b(wc, shp).copy(),
            b(S11, shp).copy(), b(S22, shp).copy(), b(S33, shp).copy(),
            b(S12, shp).copy(), b(S13, shp).copy(), b(S23, shp).copy(),
            b(Smag, shp).copy(), delta)


_F = _make_field()


def _cs2_call(field, cs_max=1.0):
    args = [jnp.asarray(a) for a in field[:-1]]
    return lasd_cs2(*args, jnp.asarray(field[-1]), cs_max=cs_max)


# ---------------------------- spectral test filter ---------------------------- #

def test_spectral_filter_matches_independent_fft():
    fld = _F[0]                                          # uc
    ny, nx, _ = fld.shape
    c1y, c1x, _, _ = _cutoffs(ny, nx)
    got = _np(spectral_test_filter(jnp.asarray(fld), c1y, c1x))
    ref = _np_spectral_filter(fld, c1y, c1x)
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)


def test_spectral_filter_passes_low_kills_high_mode_x_axis():
    # PHYSICAL cutoff property (non-circular): a pure Fourier mode BELOW the cutoff
    # passes ~unchanged; a mode ABOVE the cutoff is zeroed.  cut_x on the rfft x-axis.
    ny, nx, nz = 8, 16, 1
    xx = np.arange(nx)[None, :, None]
    cut = round(nx / (2 * _TFR_ORACLE))                 # = 4
    low = np.cos(2 * np.pi * (cut - 2) * xx / nx) * np.ones((ny, nx, nz))
    high = np.cos(2 * np.pi * (cut + 2) * xx / nx) * np.ones((ny, nx, nz))
    lo = _np(spectral_test_filter(jnp.asarray(low), ny, cut))
    hi = _np(spectral_test_filter(jnp.asarray(high), ny, cut))
    assert np.max(np.abs(lo - low)) < 1e-10             # below-cutoff mode preserved
    assert np.max(np.abs(hi)) < 1e-10                   # above-cutoff mode removed


def test_spectral_filter_folds_full_fft_y_axis():
    # The y-axis is the FULL fft axis (folded via |k_y| = min(iy, ny-iy)), codex R1 #4.
    # Non-square grid.  A real y-cosine at wavenumber k puts energy at BOTH k and ny-k;
    # both fold to |k|, so the fold (high-index / "negative" k_y) branch is exercised.
    ny, nx, nz = 12, 8, 1                                # non-square (ny != nx)
    yy = np.arange(ny)[:, None, None]
    cut_y = round(ny / (2 * _TFR_ORACLE))               # = 3 -> keep |k_y| in {0,1,2}
    cut_x = nx                                          # x fully open (isolate y)
    # k=2 (< cut 3): energy at k_y=2 AND k_y=10 (folds to 2) -> BOTH survive -> preserved
    low = np.cos(2 * np.pi * 2 * yy / ny) * np.ones((ny, nx, nz))
    # k=4 (>= cut 3): energy at k_y=4 and k_y=8 (folds to 4) -> BOTH removed -> zeroed
    high = np.cos(2 * np.pi * 4 * yy / ny) * np.ones((ny, nx, nz))
    lo = _np(spectral_test_filter(jnp.asarray(low), cut_y, cut_x))
    hi = _np(spectral_test_filter(jnp.asarray(high), cut_y, cut_x))
    assert np.max(np.abs(lo - low)) < 1e-10             # low |k_y| (incl. its fold) preserved
    assert np.max(np.abs(hi)) < 1e-10                   # high |k_y| (incl. its fold) removed


# ------------------------------- box3 average -------------------------------- #

def test_box3_matches_independent_roll():
    fld = _F[0]
    got = _np(imfilter_box3(jnp.asarray(fld)))
    ref = _np_box3(fld)
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)


def test_box3_is_ninth_weight_on_delta():
    # A unit spike spreads to its 3x3 periodic neighbourhood with weight 1/9 each.
    fld = np.zeros((5, 7, 1))
    fld[2, 3, 0] = 1.0
    out = _np(imfilter_box3(jnp.asarray(fld)))
    assert abs(out[2, 3, 0] - 1.0 / 9.0) < 1e-14        # centre 1/9
    assert abs(out[1, 2, 0] - 1.0 / 9.0) < 1e-14        # diagonal neighbour 1/9
    assert abs(out.sum() - 1.0) < 1e-13                 # mass conserved (9 * 1/9)
    assert abs(out[0, 0, 0]) < 1e-14                    # far cell untouched


# --------------------------------- β-solver ---------------------------------- #

def test_beta_matches_numpy_roots():
    # Non-circular: pin the module's β solver against numpy.roots (a DIFFERENT
    # algorithm — companion-matrix eigenvalues).  Quintics with a known real root in
    # (0,5); tol 1e-6.
    rng = np.random.default_rng(1)
    polys = []
    for _ in range(6):
        # roots: one target in (0.5,3), four others (some complex, some outside (0,5))
        target = rng.uniform(0.5, 3.0)
        others = [rng.uniform(5.5, 8.0), 6.0 + 1.0j, 6.0 - 1.0j, rng.uniform(-4.0, -0.5)]
        coeffs = np.poly(np.array([target] + others))    # leading coeff 1 == ff
        polys.append((coeffs, target))
    coeffs6 = jnp.asarray(np.stack([p[0] for p in polys]))
    beta = _np(max_real_root_beta(coeffs6))
    for k, (coeffs, target) in enumerate(polys):
        ref = _np_beta_from_numpy_roots(*[np.array([coeffs[5 - i]]) for i in range(6)])
        assert abs(beta[k] - ref[0]) < 1e-6              # matches numpy.roots
        assert abs(beta[k] - target) < 1e-6             # and the constructed root


@pytest.mark.parametrize("scale", [1e-17, 1e-30, 1e-49, 1e20])
def test_beta_is_invariant_to_coefficient_scale(scale):
    # Roots do not depend on the polynomial's overall scale; the Germano plane-mean
    # coefficients of a real CBL are O(1e-49..1e-17), where an unnormalised |p| < tol
    # stop test declares every guess converged at once (measured: residual 0.9).
    rng = np.random.default_rng(2)
    polys = []
    for _ in range(6):
        target = rng.uniform(0.5, 3.0)
        others = [rng.uniform(5.5, 8.0), 6.0 + 1.0j, 6.0 - 1.0j, rng.uniform(-4.0, -0.5)]
        polys.append((np.poly(np.array([target] + others)).real, target))
    coeffs6 = jnp.asarray(np.stack([scale * p[0] for p in polys]))
    beta = _np(max_real_root_beta(coeffs6))
    for k, (_, target) in enumerate(polys):
        assert abs(beta[k] - target) < 1e-6, (scale, k, beta[k], target)


def test_beta_finds_an_odd_multiple_root():
    # (x-2.5)^5: a 5-fold root is only resolvable to ~eps^(1/5) ~ 7e-4 by any method
    coeffs = np.poly(np.full(5, 2.5))
    beta = float(_np(max_real_root_beta(jnp.asarray(coeffs[None, :])))[0])
    assert abs(beta - 2.5) < 5e-3, beta


@pytest.mark.parametrize("roots", [[0.0, 6.0, 7.0, 8.0, 9.0], [0.0] * 5,
                                   [5.0, 6.0, 7.0, -1.0, -2.0]])
def test_beta_excludes_roots_on_the_window_edges(roots):
    # roots exactly at 0 or 5 are outside the OPEN (0, 5): default 1.0
    coeffs = np.poly(np.array(roots))
    beta = float(_np(max_real_root_beta(jnp.asarray(coeffs[None, :])))[0])
    assert beta == 1.0, beta


@pytest.mark.parametrize("root,dtype", [(5.0 - 2.0 ** -47, jnp.float64),
                                        (5.0 - 2.0 ** -17, jnp.float32)])
def test_beta_keeps_an_interior_root_next_to_the_edge(root, dtype):
    # a root one-few ulps inside 5 is a valid β (codex round 6: edge snapping lost it)
    c = jnp.asarray([[0.0, 0.0, 0.0, 0.0, 1.0, -root]], dtype)
    np.testing.assert_allclose(float(_np(max_real_root_beta(c))[0]), root, rtol=0, atol=0)


@pytest.mark.parametrize("dtype,big", [(jnp.float64, 2.0 ** 1022), (jnp.float32, 2.0 ** 126)])
def test_beta_normalisation_does_not_overflow(dtype, big):
    # finite coefficients near the float limit (codex round 7: a built 2^e overflowed)
    c = jnp.asarray([[0.0, 0.0, 0.0, 0.0, big, -2.0 * big]], dtype)
    assert float(_np(max_real_root_beta(c))[0]) == 2.0
    g = jax.grad(lambda c: jnp.sum(max_real_root_beta(c)))(c)
    assert np.all(np.isfinite(np.asarray(g, np.float64)))


def test_compensated_horner_products_are_error_free_under_jit():
    # Dekker's TwoProduct is exact only if the compiler does not contract a*b+c into an
    # FMA or reassociate: check the evaluated value against exact rational arithmetic
    from fractions import Fraction
    rng = np.random.default_rng(9)
    c = rng.standard_normal(6)
    x = rng.uniform(0.0, 5.0, 64)
    got = np.asarray(jax.jit(comp_horner)(jnp.asarray(c), jnp.asarray(x)))
    for xi, gi in zip(x, got):
        exact = Fraction(0)
        for ci in c:
            exact = exact * Fraction(xi) + Fraction(ci)
        # compensated Horner: error <= eps|p| + O(eps^2) * sum|c||x|^i
        bound = 2.3e-16 * abs(float(exact)) + 1e-28 * float(np.polyval(np.abs(c), xi))
        assert abs(Fraction(float(gi)) - exact) <= Fraction(bound), (xi, gi, float(exact))


def test_beta_finds_the_interior_root_next_to_an_exact_edge_root():
    # codex round 7/8: exact p(5) = 0 lost to plain-Horner cancellation (A = 2^51) made the
    # edge root look interior (4.99999999984); the true largest interior root is
    # 4.999999150814952869...
    A, v = 2.0 ** 51, 2.0 ** 40
    c = np.array([A, -A + 3, -20 * A, -225 * A, 1125 * A + v, -5 * (375 + v)])
    assert float(comp_horner(jnp.asarray(c), jnp.asarray(5.0))) == 0.0
    beta = float(_np(max_real_root_beta(jnp.asarray(c[None, :])))[0])
    np.testing.assert_allclose(beta, 4.999999150814952869, rtol=1e-15)


def test_beta_with_a_negative_zero_breakpoint():
    # x^5 - 32: the recursion produces a -0.0 breakpoint; its signed bit pattern broke the
    # bisection ordering (returned the default 1.0 instead of 2)
    c = jnp.asarray([[1.0, 0.0, 0.0, 0.0, 0.0, -32.0]])
    np.testing.assert_allclose(float(_np(max_real_root_beta(c))[0]), 2.0, rtol=1e-15)


def test_beta_gradient_is_accurate_under_cancellation():
    # (x-4)^5 - 2^-42: p'(beta) cancels in plain Horner (relative error 1.6e-4)
    from fractions import Fraction
    c = np.poly(np.full(5, 4.0))
    c[-1] -= 2.0 ** -42
    f = lambda c: jnp.sum(max_real_root_beta(c[None, :]))  # noqa: E731
    beta = float(f(jnp.asarray(c)))
    g = np.asarray(jax.grad(f)(jnp.asarray(c)))
    b = Fraction(beta)
    dp = sum(Fraction(float(ci)) * (5 - i) * b ** (4 - i) for i, ci in enumerate(c[:-1]))
    np.testing.assert_allclose(g[-1], float(-1 / dp), rtol=1e-12)


@pytest.mark.parametrize("root,dtype,rtol", [(1e-33, jnp.float32, 1e-6),
                                             (1e-290, jnp.float64, 1e-14)])
def test_beta_keeps_relative_accuracy_for_tiny_roots(root, dtype, rtol):
    # codex round 10: |p| near a tiny root was subnormal, flushed to 0 (23 % error);
    # full relative precision is the contract down to ~2^-110 fp32 / 2^-990 fp64
    c = jnp.asarray([[0.0, 0.0, 0.0, 0.0, 1.0, -root]], dtype)
    np.testing.assert_allclose(float(_np(max_real_root_beta(c))[0]), root, rtol=rtol)


@pytest.mark.parametrize("shape", [(1, 2), (1, 1), (6,), (2, 3, 6)])
def test_beta_refuses_a_non_quintic_shape(shape):
    with pytest.raises(TypeError):
        max_real_root_beta(jnp.ones(shape))


def test_beta_second_derivative_is_exact():
    # p = x^2 - t (padded quintic): beta = sqrt(t), d2beta/dt2 = -t^-1.5 / 4 = -1/32 at 4
    def beta(t):
        c = jnp.stack([0.0, 0.0, 0.0, 1.0, 0.0, -t])
        return max_real_root_beta(c[None, :])[0]
    t = jnp.asarray(4.0)
    np.testing.assert_allclose(float(beta(t)), 2.0, rtol=1e-12)
    np.testing.assert_allclose(float(jax.grad(beta)(t)), 0.25, rtol=1e-9)
    np.testing.assert_allclose(float(jax.grad(jax.grad(beta))(t)), -1.0 / 32.0, rtol=1e-9)


def test_beta_finds_the_largest_of_clustered_roots():
    # roots 4.6..4.9 within 0.1: the 8-guess Laguerre returned ~4.6
    coeffs = np.poly(np.array([4.6, 4.7, 4.8, 4.9, 6.0]))
    beta = float(_np(max_real_root_beta(jnp.asarray(coeffs[None, :])))[0])
    assert abs(beta - 4.9) < 1e-8, beta


@pytest.mark.parametrize("coeffs", [
    np.poly(np.array([1.0, 2.0 + 5e-7j, 2.0 - 5e-7j, 6.0, 7.0])).real,
    # (x-1)(x-6)(x-7)((x-4)^2 + 2^-42), exact in fp64 (codex round 4)
    np.polymul(np.poly(np.array([1.0, 6.0, 7.0])), [1.0, -8.0, 16.0 + 2.0 ** -42])])
def test_beta_rejects_a_near_real_complex_pair(coeffs):
    # a complex pair is not a real root; the |Im| < 1e-6 filter accepted 2 ± 5e-7 i
    beta = float(_np(max_real_root_beta(jnp.asarray(coeffs[None, :])))[0])
    assert abs(beta - 1.0) < 1e-9, beta


def test_beta_keeps_relative_accuracy_near_zero():
    # root 1e-30 (padded linear): 64 halvings alone gave 1.4e-19
    c = jnp.asarray([[0.0, 0.0, 0.0, 0.0, 1.0, -1e-30]])
    np.testing.assert_allclose(float(_np(max_real_root_beta(c))[0]), 1e-30, rtol=1e-9)
    g = jax.grad(lambda c: jnp.sum(max_real_root_beta(c)))(c)
    np.testing.assert_allclose(float(g[0, 4]), -1e-30, rtol=1e-9)

    def beta(t):   # x^2 - t, root sqrt(t) = 1e-30, d beta/dt = 1/(2 sqrt t) = 5e29
        return max_real_root_beta(jnp.stack([0.0, 0.0, 0.0, 1.0, 0.0, -t])[None, :])[0]
    t = jnp.asarray(1e-60)
    np.testing.assert_allclose(float(beta(t)), 1e-30, rtol=1e-9)
    np.testing.assert_allclose(float(jax.grad(beta)(t)), 5e29, rtol=1e-9)


def test_beta_handles_a_degree_drop():
    # leading coefficient 0: a quartic with roots 0.5, 2, 3.5 and 7 -> 3.5
    c = np.concatenate([[0.0], np.poly(np.array([0.5, 2.0, 3.5, 7.0]))])
    beta = float(_np(max_real_root_beta(jnp.asarray(c[None, :])))[0])
    np.testing.assert_allclose(beta, 3.5, rtol=1e-12)


@pytest.mark.parametrize("roots", [[1.0, 1j, -1j, 2j, -2j],
                                   [1.7, 6.0, 6.5 + 1j, 6.5 - 1j, -2.0],
                                   [0.8, 1.3, 2.9, 6.0, -1.0]])
def test_beta_gradient_is_the_implicit_derivative(roots):
    # d beta / d c_i = -beta^(5-i) / p'(beta) at the selected root; the Laguerre version
    # averaged tied guesses (x^5 - 1: -0.175 instead of -0.2)
    coeffs = np.poly(np.array(roots)).real
    f = lambda c: jnp.sum(max_real_root_beta(c[None, :]))  # noqa: E731
    beta = float(f(jnp.asarray(coeffs)))
    g = np.asarray(jax.grad(f)(jnp.asarray(coeffs)))
    ref = -beta ** np.arange(5, -1, -1) / np.polyval(np.polyder(coeffs), beta)
    np.testing.assert_allclose(g, ref, rtol=1e-9)


def test_beta_gradient_is_finite_and_scale_covariant():
    # beta(s c) == beta(c), so d beta/d(s c) == (d beta/d c) / s: finite at the tiny
    # scales of real Germano coefficients, in fp64 and fp32.
    coeffs = np.poly(np.array([1.7, 6.0, 6.5 + 1j, 6.5 - 1j, -2.0])).real

    def grad_at(s, dtype):
        f = lambda c: jnp.sum(max_real_root_beta(c[None, :]))  # noqa: E731
        return np.asarray(jax.grad(f)(jnp.asarray(s * coeffs, dtype)), np.float64)

    g1 = grad_at(1.0, jnp.float64)
    assert np.all(np.isfinite(g1)) and np.any(g1 != 0.0)
    for s in (1e-17, 1e-30):
        gs = grad_at(s, jnp.float64)
        assert np.all(np.isfinite(gs)), (s, gs)
        np.testing.assert_allclose(gs * s, g1, rtol=1e-6, atol=1e-12 * np.abs(g1).max())
    g32 = grad_at(1e-30, jnp.float32)
    assert np.all(np.isfinite(g32)), g32


def test_beta_from_tiny_germano_moments_in_fp32():
    # real-CBL moments are ~1e-25: their products underflow fp32 unless each test-filter
    # level is normalised first (roots unchanged)
    rng = np.random.default_rng(5)
    m1 = rng.standard_normal((5, 8))
    m2 = rng.standard_normal((5, 8))
    ref = _np(beta_from_germano_moments(list(jnp.asarray(m1)), list(jnp.asarray(m2))))
    assert np.sum(np.abs(ref - 1.0) > 1e-3) >= 3              # non-vacuous: real roots
    tiny = 1e-25
    got = _np(beta_from_germano_moments(list(jnp.asarray(tiny * m1, jnp.float32)),
                                        list(jnp.asarray(tiny * m2, jnp.float32))))
    np.testing.assert_allclose(got, ref, atol=1e-4)


def test_beta_from_germano_moments_gradient_matches_finite_difference():
    rng = np.random.default_rng(5)
    m1, m2 = rng.standard_normal((5, 8)), rng.standard_normal((5, 8))
    d1 = rng.standard_normal((5, 8))

    def f(e):
        return jnp.sum(beta_from_germano_moments(list(jnp.asarray(m1) + e * jnp.asarray(d1)),
                                                 list(jnp.asarray(m2))))
    g = float(jax.grad(f)(jnp.asarray(0.0)))
    h = 1e-6
    fd = (float(f(jnp.asarray(h))) - float(f(jnp.asarray(-h)))) / (2 * h)
    assert np.isfinite(g) and abs(g) > 1e-6
    np.testing.assert_allclose(g, fd, rtol=1e-5)


def test_beta_defaults_to_one_when_no_root_in_range():
    # Quintic whose only real roots are OUTSIDE (0,5) -> β defaults to 1.0.
    coeffs = np.poly(np.array([6.0, 7.0, -1.0, 8.0 + 1j, 8.0 - 1j]))  # none in (0,5)
    beta = float(_np(max_real_root_beta(jnp.asarray(coeffs[None, :])))[0])
    assert beta == 1.0


def test_beta_picks_max_real_root_in_range():
    # Two real roots in (0,5): 1.2 and 3.7 -> the solver returns the MAX (3.7).
    coeffs = np.poly(np.array([1.2, 3.7, 6.5, 7.0 + 1j, 7.0 - 1j]))
    beta = float(_np(max_real_root_beta(jnp.asarray(coeffs[None, :])))[0])
    assert abs(beta - 3.7) < 1e-6


# ----------------------------- lasd_cs2 assembly ----------------------------- #

def test_lasd_cs2_assembly_matches_oracle():
    got = _np(_cs2_call(_F))
    ref = _lasd_cs2_oracle(*_F)["cs2"]
    # non-vacuous: some cells are active (nonzero C_s^2), not all masked
    assert np.count_nonzero(ref) > 0
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)


def test_beta_solver_on_ASSEMBLY_polynomials_matches_numpy_roots():
    # β-REALISM (codex R1 #3): the assembly test reuses the module's Laguerre as a given,
    # so it cannot catch a β-solver error on the polynomials the LASD assembly ACTUALLY
    # produces.  Close the loop: take the assembly-generated aa..ff (from the real field,
    # NOT friendly monic quintics), solve β with the module's Laguerre, and cross-check
    # against numpy.roots (companion-matrix eigenvalues — a different algorithm).
    d = _lasd_cs2_oracle(*_F)
    coeffs6 = jnp.stack([jnp.asarray(d["ff"]), jnp.asarray(d["ee"]), jnp.asarray(d["dd"]),
                         jnp.asarray(d["cc"]), jnp.asarray(d["bb"]), jnp.asarray(d["aa"])], -1)
    beta_module = _np(max_real_root_beta(coeffs6))
    beta_np = _np_beta_from_numpy_roots(d["aa"], d["bb"], d["cc"], d["dd"], d["ee"], d["ff"])
    # non-vacuous: the real field's polynomials have genuine roots in (0,5) (β != default 1)
    assert np.sum(np.abs(beta_module - 1.0) > 1e-3) >= 1
    np.testing.assert_allclose(beta_module, beta_np, atol=1e-6)   # Laguerre tol vs numpy.roots


def test_cs2_masks_zero_strain_to_zero():
    # Uniform velocity + zero strain => M_ij = 0 => MM ~ 0 => C_s^2 masked to 0.
    ny, nx, nz = 8, 12, 3
    zeros = np.zeros((ny, nx, nz))
    uc = np.full((ny, nx, nz), 2.0)
    delta = np.array([10.0, 12.0, 15.0])
    out = _np(lasd_cs2(jnp.asarray(uc), jnp.asarray(uc), jnp.asarray(uc),
                       *[jnp.asarray(zeros)] * 6, jnp.asarray(zeros), jnp.asarray(delta)))
    assert np.all(out == 0.0)


def test_cs2_clips_above_cs_max():
    # Lowering cs_max masks more cells to 0 (the cs2 > cs_max^2 branch is live).
    base = _np(_cs2_call(_F, cs_max=1.0))
    cap = 0.005                                          # cs_max^2 = 2.5e-5
    tight = _np(_cs2_call(_F, cs_max=cap))
    assert np.any(base > cap**2)                         # cells exceed the tight cap
    assert np.all(tight[base > cap**2] == 0.0)           # ...and are clipped to 0
    # cells already below the tight cap are unchanged
    keep = (base > 0) & (base <= cap**2)
    assert np.any(keep)                                  # non-vacuous: some survive
    np.testing.assert_allclose(tight[keep], base[keep], rtol=1e-12)


def test_beta_scale_dependence_is_applied():
    # β enters M_ij (T2*β*S_h*S_ij).  Forcing β=1 (scale-INDEPENDENT dynamic model)
    # changes the C_s^2 field vs the solved β — proving the scale-dependent β is live.
    got = _np(_cs2_call(_F))
    solved = _lasd_cs2_oracle(*_F)
    forced = _lasd_cs2_oracle(*_F, force_beta=1.0)["cs2"]
    np.testing.assert_allclose(got, solved["cs2"], rtol=1e-12, atol=1e-14)
    assert np.any(np.abs(solved["beta"] - 1.0) > 1e-3)   # solved β genuinely != 1
    assert np.max(np.abs(solved["cs2"] - forced)) > 1e-9  # ...and it changes C_s^2


def test_lilly_and_tfr_constants():
    # Bou-Zeid/Lilly constants: the Lilly (1992) error-functional coefficient 8
    # (oracle 8·L⁴ weight on e1/e2) and the test-filter ratio 2 (oracle TFR).
    assert _LASD_LILLY_COEFF == 8.0
    assert _TFR == 2.0


def test_differentiable_wrt_velocity_active_field():
    # AD-safe on an ACTIVE field through the Laguerre freeze-on-convergence scan, the
    # floored-denominator divide, and the mask/clip: finite grad of sum(C_s^2) wrt a
    # velocity scale.
    def loss(scale):
        args = [jnp.asarray(_F[0]) * scale] + [jnp.asarray(a) for a in _F[1:-1]]
        return jnp.sum(lasd_cs2(*args, jnp.asarray(_F[-1])))

    g = float(jax.grad(loss)(jnp.asarray(1.0)))
    assert np.isfinite(g)


def test_differentiable_through_beta_default_no_valid_root():
    # AD-safe through the REALISTIC degenerate path (codex R1 #6): a level whose β quintic
    # has NONZERO coefficients but NO valid root in (0,5) hits the ``jnp.nanmax`` ->
    # ``where(isnan, 1.0, mx)`` β-default branch.  This is the degenerate path a real LES
    # can actually reach (any level with no in-range root), and it MUST be AD-safe.
    coeffs = np.poly(np.array([6.0, 7.0, -1.0, 8.0 + 1j, 8.0 - 1j])).real  # none in (0,5)

    def loss(scale):
        return jnp.sum(max_real_root_beta(jnp.asarray(coeffs)[None, :] * scale))

    assert float(loss(jnp.asarray(1.0))) == 1.0                 # β defaults to 1.0
    g = float(jax.grad(loss)(jnp.asarray(1.0)))
    assert np.isfinite(g)                                       # ...and the gradient is finite


def test_fully_degenerate_zero_strain_grad_is_finite():
    # The all-zero-quintic AD trap (codex R1 #6) is GUARDED: an exactly-zero-strain plane
    # makes every β coefficient 0, on which the raw Laguerre √0/0·∞ step poisons the reverse
    # gradient with NaN.  ``per_level`` detects ``max|coeff|==0`` and feeds the solver a
    # benign quintic, selecting the default 1.0 forward — so the FORWARD is a clean masked 0
    # AND the gradient is finite.  (Was a documented NaN limitation before the guard.)
    ny, nx, nz = 6, 8, 2
    delta = np.array([10.0, 12.0])

    def loss(scale):
        uc = jnp.asarray(np.full((ny, nx, nz), 2.0)) * scale    # uniform -> exactly zero strain
        z = jnp.zeros((ny, nx, nz))
        return jnp.sum(lasd_cs2(uc, uc, uc, z, z, z, z, z, z, z, jnp.asarray(delta)))

    assert float(loss(jnp.asarray(1.0))) == 0.0                 # forward: clean masked 0
    assert np.isfinite(float(jax.grad(loss)(jnp.asarray(1.0))))  # grad: finite (guarded)


def test_mixed_dead_and_active_levels_grad_is_finite():
    # codex R2: β is solved PER LEVEL, so a SINGLE exactly-zero-strain level (a rigid lid /
    # laminar layer) among otherwise-active levels would nan the whole gradient without the
    # guard.  A 3-level field with level 1 = exactly zero strain must still yield a finite
    # gradient AND leave the active levels' C_s^2 untouched (guard is forward-identical off
    # the degenerate level).
    ny, nx, nz = 6, 8, 3
    rng = np.random.default_rng(3)
    S = [rng.standard_normal((ny, nx, nz)) * 0.5 for _ in range(6)]
    for s in S:
        s[:, :, 1] = 0.0                                        # level 1: dead (zero strain)
    Smag = np.sqrt(2.0 * (S[0]**2 + S[1]**2 + S[2]**2 + 2.0 * (S[3]**2 + S[4]**2 + S[5]**2)))
    uc = rng.standard_normal((ny, nx, nz))
    vc = rng.standard_normal((ny, nx, nz))
    wc = rng.standard_normal((ny, nx, nz))
    delta = np.array([10.0, 12.0, 15.0])

    def loss(scale):
        return jnp.sum(lasd_cs2(jnp.asarray(uc) * scale, jnp.asarray(vc), jnp.asarray(wc),
                                *[jnp.asarray(x) for x in S], jnp.asarray(Smag),
                                jnp.asarray(delta)))

    out = _np(lasd_cs2(jnp.asarray(uc), jnp.asarray(vc), jnp.asarray(wc),
                       *[jnp.asarray(x) for x in S], jnp.asarray(Smag), jnp.asarray(delta)))
    ref = _lasd_cs2_oracle(uc, vc, wc, *S, Smag, delta)["cs2"]  # independent oracle
    assert np.all(out[:, :, 1] == 0.0)                          # dead level masked to 0
    assert np.count_nonzero(out[:, :, [0, 2]]) > 0              # active levels non-vacuous
    # forward-IDENTICAL to the oracle on EVERY level (codex R2 #2: active levels untouched
    # by the guard, not merely nonzero) — proves the per-level guard perturbs nothing
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-14)
    assert np.isfinite(float(jax.grad(loss)(jnp.asarray(1.0))))  # finite grad despite dead level


def test_repeated_root_disc_zero_grad_is_finite():
    # codex R3: sqrt(disc) has a singular VJP at disc==0, which a REPEATED-ROOT polynomial
    # hits exactly — e.g. x^5 gives disc==0 at every Laguerre guess (root 0, multiplicity 5).
    # The double-``where`` disc guard keeps the gradient finite while the FORWARD (default
    # β=1.0, since the repeated root 0 is not in (0,5)) is unchanged.
    x5 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])              # x^5: max|coeff| != 0 (not all-zero)

    def loss(scale):
        return jnp.sum(max_real_root_beta(jnp.asarray(x5)[None, :] * scale))

    assert float(loss(jnp.asarray(1.0))) == 1.0                # β defaults (root 0 not in (0,5))
    assert np.isfinite(float(jax.grad(loss)(jnp.asarray(1.0))))  # finite grad (disc guard)
