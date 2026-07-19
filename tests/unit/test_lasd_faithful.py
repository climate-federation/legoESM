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
  - Laguerre max-real-root-in-(0,5) β-solver, default 1.0 (oracle ``Roots`` +
    ``ComputeBeta1``) — pinned against ``numpy.roots`` (a DIFFERENT algorithm:
    companion-matrix eigenvalues), so the agreement is non-circular.
  - the LASDD assembly: L_ij/Q_ij Germano stresses at 2Δ/4Δ, the a1..e2 plane-mean
    contractions with the oracle's exact ``2L²``/``TFR^n``/``8L⁴`` weights, the
    aa..ff quintic coefficients, M_ij (β scale-dependence), LM/MM, box-average, and
    the ``C_s²`` clip mask (MM~0 | C_s²<0 | C_s²>cs_max²).

Departures (legoESM adaptations, documented — the scheme is otherwise faithful):
  - ``cs_max`` is a CONFIGURABLE clip extension: the default cs_max=1 reproduces the
    oracle mask exactly, but cs_max != 1 changes the C_s² ceiling (a closure change).
  - C_s² uses a RAW divide after flooring the denominator (``LMx/where(|MMx|<1e-10,
    1e-10, MMx)``) — NOT the project ``safe_divide`` helper — and a fixed-trip
    ``lax.scan`` freeze-on-convergence Laguerre (oracle uses raw divide + ``while_loop``).
    Both return the oracle's forward result (the |MM|<1e-10 cells are masked to 0; the
    scan freezes the converged root to the Laguerre tol) but the scan is NOT bit-identical
    to the while_loop; both are for reverse-mode differentiability.
  - a ``(ny, nx, nz)`` layout (oracle ``(nx, ny, nz)``) — a transpose convention — and
    an optional MPI y-slab distributed path (serial path pinned here).

The β-solver's Laguerre tolerance is 1e-6, so the full ASSEMBLY pin reuses the
separately-pinned ``laguerre_max_real_root_beta`` as a GIVEN (like the shared thermo in
the convection pins) and independently reimplements only the L/Q/M/LM/MM/mask glue at
1e-12; the solver itself is pinned to 1e-6 vs numpy.roots in its own test, AND its β on the
assembly-produced polynomials aa..ff is cross-checked vs numpy.roots (β-realism).

Differentiability: AD-safe (finite reverse-mode gradient) almost everywhere.  Two Laguerre
singularities a repeated/degenerate β-quintic can hit are GUARDED forward-identically: (i)
the all-zero quintic (a strain-free level → every β coefficient 0) — ``per_level`` feeds the
solver a benign quintic and selects the default 1.0; (ii) sqrt(disc) at disc==0 (a repeated
root such as x^5) — a double-``where`` keeps the VJP finite.  A no-root-in-(0,5) quintic (the
default-to-1.0 branch) is also finite-gradient.  NOT smoothed (kept faithful to the oracle's
raw sqrt): a tiny-nonzero disc from a NEAR-repeated-root gives a large-but-finite VJP — a
measure-zero-adjacent locus a physical strain field does not reach.  Pinned by the zero-
strain, mixed dead+active level, and repeated-root (x^5) gradient tests.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.turbulence.lasd_core import (
    spectral_test_filter,
    imfilter_box3,
    laguerre_max_real_root_beta,
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
        beta = _np(laguerre_max_real_root_beta(
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


# ---------------------------- Laguerre β-solver ------------------------------ #

def test_laguerre_matches_numpy_roots():
    # Non-circular: pin the module's Laguerre solver against numpy.roots (a DIFFERENT
    # algorithm — companion-matrix eigenvalues).  Quintics with a known real root in
    # (0,5); Laguerre tol 1e-6.
    rng = np.random.default_rng(1)
    polys = []
    for _ in range(6):
        # roots: one target in (0.5,3), four others (some complex, some outside (0,5))
        target = rng.uniform(0.5, 3.0)
        others = [rng.uniform(5.5, 8.0), 6.0 + 1.0j, 6.0 - 1.0j, rng.uniform(-4.0, -0.5)]
        coeffs = np.poly(np.array([target] + others))    # leading coeff 1 == ff
        polys.append((coeffs, target))
    coeffs6 = jnp.asarray(np.stack([p[0] for p in polys]))
    beta = _np(laguerre_max_real_root_beta(coeffs6))
    for k, (coeffs, target) in enumerate(polys):
        ref = _np_beta_from_numpy_roots(*[np.array([coeffs[5 - i]]) for i in range(6)])
        assert abs(beta[k] - ref[0]) < 1e-6              # matches numpy.roots
        assert abs(beta[k] - target) < 1e-6             # and the constructed root


def test_laguerre_defaults_to_one_when_no_root_in_range():
    # Quintic whose only real roots are OUTSIDE (0,5) -> β defaults to 1.0.
    coeffs = np.poly(np.array([6.0, 7.0, -1.0, 8.0 + 1j, 8.0 - 1j]))  # none in (0,5)
    beta = float(_np(laguerre_max_real_root_beta(jnp.asarray(coeffs[None, :])))[0])
    assert beta == 1.0


def test_laguerre_picks_max_real_root_in_range():
    # Two real roots in (0,5): 1.2 and 3.7 -> the solver returns the MAX (3.7).
    coeffs = np.poly(np.array([1.2, 3.7, 6.5, 7.0 + 1j, 7.0 - 1j]))
    beta = float(_np(laguerre_max_real_root_beta(jnp.asarray(coeffs[None, :])))[0])
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
    beta_module = _np(laguerre_max_real_root_beta(coeffs6))
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
        return jnp.sum(laguerre_max_real_root_beta(jnp.asarray(coeffs)[None, :] * scale))

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
        return jnp.sum(laguerre_max_real_root_beta(jnp.asarray(x5)[None, :] * scale))

    assert float(loss(jnp.asarray(1.0))) == 1.0                # β defaults (root 0 not in (0,5))
    assert np.isfinite(float(jax.grad(loss)(jnp.asarray(1.0))))  # finite grad (disc guard)
