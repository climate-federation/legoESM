"""Bou-Zeid–Meneveau–Parlange (2005) scale-dependent dynamic SGS — shared core.

Grid-agnostic implementation of the LASD Smagorinsky coefficient ``C_s²(x,y,z)``
faithful to the jax-alfa oracle ``DynamicSGS_LASDD_SM.LASDD``. It operates on
cell-centre velocities + the resolved strain tensor on a horizontally-periodic
plane ``(ny, nx, nz)`` and is called by BOTH plane LES cores:

* the compressible plane dycore
  (``compressible_euler_plane._compute_scale_dependent_dynamic_smag_cs_plane``),
* the pseudo-spectral incompressible core (``spectral_les_plane.eddy_viscosity``).

The dynamic procedure adds a second test filter at ``4Δ`` and solves the Germano
identity at both ratios for the scale-dependence parameter
``β = C_s²(2Δ)/C_s²(Δ)`` per level (quintic root), then forms a LOCALLY-averaged
(3×3) ``C_s²`` field. Test filters are sharp spectral cutoffs (oracle
``Filtering_Level1/2``); the polynomial coefficients are plane-averaged.

Faithfulness
------------
Every closed form here is a faithful port of the jax-alfa oracle
(``DynamicSGS_LASDD_SM.LASDD``, ``Filtering_Level1/2``, ``Utilities.Roots`` +
``Imfilter`` + ``PlanarMean``), pinned to round-off (rel 1e-12) against an INDEPENDENT
numpy reimplementation in ``tests/unit/test_lasd_faithful.py``: the sharp spectral test
filter (cutoff ``round(N/(2·FGR·TFR))``), the 3×3 periodic box average /9, the Laguerre
max-real-root-in-(0,5) β-solver (default 1.0, cross-checked against ``numpy.roots`` — a
different algorithm), and the full a1..e2 / aa..ff / M_ij / LM / MM / ``C_s²``-clip
assembly (with the Lilly-1992 error-functional coefficient ``_LASD_LILLY_COEFF=8`` and
``_TFR=2``).  Documented departures from the oracle:
- ``cs_max`` is a CONFIGURABLE clip extension: the oracle hardcodes 1.0, and the default
  ``cs_max=1`` reproduces the oracle mask exactly, but ``cs_max != 1`` DOES change the
  modeled closure (a different ceiling on ``C_s²``) — it is an extension, not a pure
  reformatting.
- ``C_s²`` uses a RAW divide after FLOORING the denominator
  (``LMx / where(|MMx|<1e-10, 1e-10, MMx)``), not the project ``safe_divide`` helper;
  it is forward-identical to the oracle (the ``|MMx|<1e-10`` cells are masked to 0 either
  way) and keeps the reverse-mode cotangent finite by never dividing by a tiny denominator.
- the β-solver is a fixed-trip ``lax.scan`` that FREEZES the converged root via ``where``
  (the oracle uses a ``while_loop`` that stops early).  It returns the SAME converged root
  as the oracle to the Laguerre tolerance, but is NOT bit-identical to the ``while_loop``
  (it keeps evaluating — and discarding — later Laguerre steps after convergence) and can
  differ at a convergence boundary; the scan form is what makes it reverse-mode
  differentiable.
- the ``(ny, nx, nz)`` layout (oracle ``(nx, ny, nz)`` — a transpose) and the optional
  MPI y-slab distributed path (serial path is the pinned one).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

# Machine-checked scheme contract (see tests/test_physics_contracts.py). This
# core computes the scale-dependent dynamic Smagorinsky coefficient C_s^2; the
# public physical-quantity entry is ``lasd_cs2``.
__physics_contract__ = {
    "summary": (
        "Bou-Zeid-Meneveau-Parlange (2005) Lagrangian scale-dependent dynamic "
        "SGS: solves the Germano identity at 2-delta and 4-delta test filters "
        "for the scale-dependence parameter beta and returns the locally "
        "averaged Smagorinsky coefficient C_s^2(x,y,z) for a plane LES."
    ),
    "inputs": {
        "uc": "m/s", "vc": "m/s", "wc": "m/s",
        "S11": "1/s", "S22": "1/s", "S33": "1/s",
        "S12": "1/s", "S13": "1/s", "S23": "1/s",
        "Smag": "1/s (resolved strain-rate magnitude |S|)",
        "delta": "m (grid filter width per level)",
        "cs_max": "1 (upper clip on C_s)",
    },
    "outputs": {
        "cs2": "1 (dimensionless Smagorinsky coefficient C_s^2 field)",
    },
    "sign_convention": (
        "C_s^2 >= 0: masked to 0 where the Germano denominator MM ~ 0, where "
        "C_s^2 < 0, or where C_s^2 > cs_max^2 (backscatter/invalid clipped). "
        "The caller forms the down-gradient eddy viscosity nu_t = "
        "C_s^2 * delta^2 * |S| >= 0. Returns a coefficient, not a tendency."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Bou-Zeid, Meneveau & Parlange (2005), Phys. Fluids 17, 025105; "
        "Lilly (1992) dynamic-SGS error functional; jax-alfa LASDD oracle"
    ),
    "idealized_test": (
        "jax-alfa DynamicSGS_LASDD_SM.LASDD parity on a horizontally-periodic "
        "LES plane; C_s^2 in [0, cs_max^2]; zero where MM ~ 0; beta defaults "
        "to 1 when the quintic has no real root in (0, 5)."
    ),
}

_TFR = 2.0   # test-filter ratio (FGR=1): level-1 = 2Δ, level-2 = 4Δ


# --------------------------------------------------------------------------- #
# Lilly (1992) dynamic-SGS error-functional coefficient.
_LASD_LILLY_COEFF = 8.0

def spectral_test_filter(field_yxz: jax.Array, cut_y: int, cut_x: int,
                         layout=None):
    """Sharp spectral cutoff TEST filter over the periodic ``(y, x)`` plane
    (oracle ``Filtering_Level1/2``): zero every mode with ``|k_y| ≥ cut_y`` or
    ``k_x ≥ cut_x``. Vertical left unfiltered (ABL LES is homogeneous in (x,y)).

    ``layout`` (a ``SpectralLESLayout``) selects the y-slab distributed FFT; the
    kx cutoff is applied on this rank's kx-column slab via the columns' GLOBAL
    indices. ``None`` ⇒ serial ``jnp.fft.rfft2``."""
    if layout is None:
        ny, nx = field_yxz.shape[0], field_yxz.shape[1]
        fh = jnp.fft.rfft2(field_yxz, axes=(0, 1))
        iy = jnp.arange(ny)
        fold_y = jnp.minimum(iy, ny - iy)
        mask_y = (fold_y < cut_y)[:, None, None]
        mask_x = (jnp.arange(fh.shape[1]) < cut_x)[None, :, None]
        fh = jnp.where(mask_y & mask_x, fh, 0.0)
        return jnp.fft.irfft2(fh, axes=(0, 1), s=(ny, nx))
    from legoesm.parallel.distributed_fft import (
        distributed_rfft2, distributed_irfft2, local_kx_slice)
    ny, nx = layout.ny_global, layout.nx
    fh = distributed_rfft2(field_yxz, ny_global=ny, nx=nx,
                           n_ranks=layout.n_ranks, comm=layout.comm)
    iy = jnp.arange(ny)
    fold_y = jnp.minimum(iy, ny - iy)
    mask_y = (fold_y < cut_y)[:, None, None]
    lo, hi = local_kx_slice(nx, layout.n_ranks, layout.rank)
    mask_x = (jnp.arange(lo, hi) < cut_x)[None, :, None]   # GLOBAL kx index
    fh = jnp.where(mask_y & mask_x, fh, 0.0)
    return distributed_irfft2(fh, ny_global=ny, nx=nx,
                              n_ranks=layout.n_ranks, comm=layout.comm)


def _y_ring_rows(fx, layout):
    """Return ``(last_row_of_prev_rank, first_row_of_next_rank)`` for the periodic
    y-ring of the slab decomposition — the two neighbour rows ``imfilter_box3``
    needs for its ∂y 3-point average. Two directional sendrecv shifts with
    distinct tags (unambiguous even at n_ranks=2, where prev==next)."""
    import mpi4jax  # noqa: F401
    from legoesm.parallel.halo_exchange import get_sendrecv_vjp
    sr = get_sendrecv_vjp(mpi4jax)
    n, r, comm = layout.n_ranks, layout.rank, layout.comm
    prev, nxt = (r - 1) % n, (r + 1) % n
    top, bot = fx[0:1], fx[-1:]
    tmpl = jnp.zeros_like(bot)
    # send my last row to next, receive prev's last row from prev
    last_of_prev = sr(bot, tmpl, prev, nxt, 700, 700, comm)  # coeff-ok: MPI halo message tags
    # send my first row to prev, receive next's first row from next
    first_of_next = sr(top, jnp.zeros_like(top), nxt, prev, 701, 701, comm)  # coeff-ok: MPI halo message tags
    return last_of_prev, first_of_next


def imfilter_box3(field_yxz: jax.Array, layout=None) -> jax.Array:
    """Periodic 3×3 horizontal box average (oracle ``Utilities.Imfilter``) — the
    LASD LOCAL averaging of the Germano ratio before forming ``C_s²``. Under the
    y-slab ``layout`` the ∂y rolls cross ranks, so the two boundary rows are
    exchanged with the y-neighbours (∂x is local — x is undecomposed)."""
    fx = (jnp.roll(field_yxz, 1, axis=1) + field_yxz
          + jnp.roll(field_yxz, -1, axis=1)) / 3.0
    if layout is None:
        return (jnp.roll(fx, 1, axis=0) + fx + jnp.roll(fx, -1, axis=0)) / 3.0
    last_of_prev, first_of_next = _y_ring_rows(fx, layout)
    fy_up = jnp.concatenate([last_of_prev, fx[:-1]], axis=0)   # roll(+1, axis=0)
    fy_dn = jnp.concatenate([fx[1:], first_of_next], axis=0)   # roll(-1, axis=0)
    return (fy_up + fx + fy_dn) / 3.0


def laguerre_max_real_root_beta(coeffs6: jax.Array) -> jax.Array:
    """Largest real root in ``(0, 5)`` of a quintic, default ``1.0``.

    Faithful port of the jax-alfa β-solver (``Utilities.Roots`` Laguerre +
    ``ComputeBeta1``). Fixed-trip ``lax.scan`` (freeze-on-convergence) ⇒
    reverse-mode differentiable. ``coeffs6`` is ``(..., 6)`` DESCENDING degree.

    AD note: differentiable almost everywhere (reverse-mode finite), with the two Laguerre
    singularities that a repeated/degenerate polynomial can hit GUARDED, forward-identically:
    (i) the ALL-ZERO polynomial (a strain-free level → every coefficient 0) is detected in
    ``per_level`` and the solver is fed a benign quintic (default ``1.0`` selected forward);
    (ii) ``sqrt(disc)`` at ``disc == 0`` (a repeated root, e.g. ``x^5`` → disc == 0 at every
    guess) uses a double-``where`` so the VJP never differentiates sqrt at 0.  A quintic with
    no root in ``(0, 5)`` (the ``nanmax`` → default-``1.0`` branch) is also finite-gradient.
    NOT smoothed: a tiny-but-nonzero ``disc`` from a NEAR-repeated-root polynomial gives a
    large-but-finite VJP (kept faithful to the oracle's raw sqrt) — a measure-zero-adjacent
    locus not reached by a physical strain field.  See ``test_lasd_faithful.py``."""
    cdtype = jnp.complex128 if jax.config.jax_enable_x64 else jnp.complex64
    guesses = jnp.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.5], cdtype)  # coeff-ok: dynamic-coeff root-search initial guesses
    n_deg, tol, max_iter = 5, 1e-6, 20  # coeff-ok: polynomial degree / tol / max iterations
    # A benign, well-conditioned quintic (x^5 - 1, real root at 1.0) fed to the solver in
    # place of an all-zero (strain-free-level) polynomial, so the Laguerre sqrt(0)/0-inf
    # trap never produces a NaN cotangent.  The default 1.0 is selected forward regardless.
    benign_quintic = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, -1.0])  # coeff-ok: AD-guard poly

    def one_root(coeffs, x0):
        coeffs = coeffs.astype(cdtype)
        eps = 1e-16 + 0j

        def step(carry, _):
            x, conv = carry
            f = jnp.polyval(coeffs, x)
            df = jnp.polyval(jnp.polyder(coeffs), x)
            d2f = jnp.polyval(jnp.polyder(jnp.polyder(coeffs)), x)
            fs = jnp.where(jnp.abs(f) < jnp.abs(eps), eps, f)
            G = df / fs
            H = G ** 2 - d2f / fs
            disc = (n_deg - 1) * (n_deg * H - G ** 2)
            # AD guard: sqrt(disc) has a SINGULAR reverse derivative at disc == 0, which a
            # repeated-root polynomial (e.g. x^5 -> disc == 0 at every guess) hits exactly.
            # Double-``where``: forward sqrt(0) = 0 is unchanged, but the VJP never
            # differentiates sqrt AT 0 (the inner where feeds 1 there).
            disc_zero = disc == 0
            sq = jnp.where(disc_zero, jnp.zeros_like(disc),
                           jnp.sqrt(jnp.where(disc_zero, jnp.ones_like(disc), disc)))
            d1, d2 = G + sq, G - sq
            denom = jnp.where(jnp.abs(d1) > jnp.abs(d2), d1, d2)
            denom = jnp.where(jnp.abs(denom) < jnp.abs(eps), eps, denom)
            xn = x - n_deg / denom
            now = (jnp.abs(xn - x) < tol * (1 + jnp.abs(x))) | (jnp.abs(f) < tol)
            return (jnp.where(conv, x, xn), conv | now), None

        (xf, conv), _ = jax.lax.scan(
            step, (x0, jnp.array(False)), None, length=max_iter)
        return jnp.where(conv, xf, jnp.nan + 0j)

    def per_level(coeffs):
        # AD guard: a strain-free level yields an ALL-ZERO quintic (0·strain = 0 exactly),
        # on which the Laguerre step is a √0 / 0·∞ trap that poisons the reverse gradient
        # with NaN (the forward still defaults to 1.0).  Detect it (max|coeff| == 0, exact
        # for a 0·strain level) and feed the solver the BENIGN quintic instead, then select
        # the default 1.0 forward.  Forward-IDENTICAL for every non-degenerate polynomial
        # (``where`` picks ``coeffs``); NaN-free gradient on the degenerate one.
        degenerate = jnp.max(jnp.abs(coeffs)) == 0.0
        safe = jnp.where(degenerate, benign_quintic, coeffs)
        roots = jax.vmap(lambda g: one_root(safe, g))(guesses)
        valid = jnp.where(
            (jnp.abs(jnp.imag(roots)) < 1e-6)
            & (jnp.real(roots) > 0.0) & (jnp.real(roots) < 5.0),  # coeff-ok: physical-root upper bound
            jnp.real(roots), jnp.nan)
        mx = jnp.nanmax(valid)
        return jnp.where(degenerate, 1.0, jnp.where(jnp.isnan(mx), 1.0, mx))

    return jax.vmap(per_level)(coeffs6)


# --------------------------------------------------------------------------- #
def lasd_cs2(uc, vc, wc, S11, S22, S33, S12, S13, S23, Smag,
             delta, cs_max: float = 1.0, layout=None) -> jax.Array:
    """Scale-dependent dynamic Smagorinsky ``C_s²(x,y,z)`` (Bou-Zeid 2005).

    Parameters
    ----------
    uc, vc, wc : ``(ny, nx, nz)`` cell-centre velocities.
    S11..S23, Smag : ``(ny, nx, nz)`` resolved strain components + ``|S|``.
    delta : ``(nz,)`` grid filter width ``Δ`` per level.
    cs_max : upper clip on ``C_s²`` (oracle mask uses 1.0).

    Returns ``C_s²`` field ``(ny, nx, nz)`` (locally averaged, masked to a valid
    LES range). The caller forms ``ν_t = C_s² · Δ² · |S|``.
    """
    ny, nx, nz = uc.shape
    # Test-filter cutoffs must come from the GLOBAL horizontal extent. Under the
    # y-slab MPI layout uc is a y-slab, so uc.shape[0] is ny_LOCAL — using it
    # would shrink cut_y per rank (e.g. NY=12,np=4 → cut2y=0 zeros the whole
    # level-2 y filter). x is undecomposed (nx already global), but read both
    # from the layout for clarity. (Bug caught by codex review 2026-06-09.)
    ny_g = layout.ny_global if layout is not None else ny
    nx_g = layout.nx if layout is not None else nx
    L2 = (jnp.asarray(delta) ** 2).reshape(1, 1, nz)               # Δ² (1,1,nz)
    cut1y, cut1x = round(ny_g / (2 * _TFR)), round(nx_g / (2 * _TFR))            # 2Δ
    cut2y, cut2x = round(ny_g / (2 * _TFR * _TFR)), round(nx_g / (2 * _TFR * _TFR))  # 4Δ
    F1 = lambda f: spectral_test_filter(f, cut1y, cut1x, layout)  # noqa: E731
    F2 = lambda f: spectral_test_filter(f, cut2y, cut2x, layout)  # noqa: E731
    if layout is None:
        pm = lambda f: jnp.mean(f, axis=(0, 1))                   # noqa: E731
    else:
        from legoesm.parallel.reductions import global_sum_mpi
        _pden = layout.ny_global * layout.nx
        # Reduce over the layout's communicator (matches the distributed FFT), not
        # COMM_WORLD — consistent on sub-communicators (codex 2026-06-09).
        pm = lambda f: (global_sum_mpi(jnp.sum(f, axis=(0, 1)),  # noqa: E731
                                       comm=layout.comm) / _pden)

    u_h, v_h, w_h = F1(uc), F1(vc), F1(wc)
    u_d, v_d, w_d = F2(uc), F2(vc), F2(wc)
    L11, L22, L33 = F1(uc*uc)-u_h*u_h, F1(vc*vc)-v_h*v_h, F1(wc*wc)-w_h*w_h
    L12, L13, L23 = F1(uc*vc)-u_h*v_h, F1(uc*wc)-u_h*w_h, F1(vc*wc)-v_h*w_h
    Q11, Q22, Q33 = F2(uc*uc)-u_d*u_d, F2(vc*vc)-v_d*v_d, F2(wc*wc)-w_d*w_d
    Q12, Q13, Q23 = F2(uc*vc)-u_d*v_d, F2(uc*wc)-u_d*w_d, F2(vc*wc)-v_d*w_d

    S11h, S22h, S33h = F1(S11), F1(S22), F1(S33)
    S12h, S13h, S23h = F1(S12), F1(S13), F1(S23)
    S11d, S22d, S33d = F2(S11), F2(S22), F2(S33)
    S12d, S13d, S23d = F2(S12), F2(S13), F2(S23)
    S_h = jnp.sqrt(2.0*(S11h**2+S22h**2+S33h**2+2.0*(S12h**2+S13h**2+S23h**2)))
    S_d = jnp.sqrt(2.0*(S11d**2+S22d**2+S33d**2+2.0*(S12d**2+S13d**2+S23d**2)))
    SS11h, SS22h, SS33h = F1(Smag*S11), F1(Smag*S22), F1(Smag*S33)
    SS12h, SS13h, SS23h = F1(Smag*S12), F1(Smag*S13), F1(Smag*S23)
    SS11d, SS22d, SS33d = F2(Smag*S11), F2(Smag*S22), F2(Smag*S33)
    SS12d, SS13d, SS23d = F2(Smag*S12), F2(Smag*S13), F2(Smag*S23)

    a1 = pm(2.0*L2*(L11*SS11h+L22*SS22h+L33*SS33h
                    + 2.0*(L12*SS12h+L13*SS13h+L23*SS23h)))
    a2 = pm(2.0*L2*(Q11*SS11d+Q22*SS22d+Q33*SS33d
                    + 2.0*(Q12*SS12d+Q13*SS13d+Q23*SS23d)))
    b1 = pm(2.0*L2*(_TFR**2)*S_h*(L11*S11h+L22*S22h+L33*S33h
                                  + 2.0*(L12*S12h+L13*S13h+L23*S23h)))
    b2 = pm(2.0*L2*(_TFR**4)*S_d*(Q11*S11d+Q22*S22d+Q33*S33d
                                  + 2.0*(Q12*S12d+Q13*S13d+Q23*S23d)))
    c1 = pm((2.0*L2)**2*(SS11h**2+SS22h**2+SS33h**2
                         + 2.0*(SS12h**2+SS13h**2+SS23h**2)))
    c2 = pm((2.0*L2)**2*(SS11d**2+SS22d**2+SS33d**2
                         + 2.0*(SS12d**2+SS13d**2+SS23d**2)))
    d1 = pm((4.0*L2**2)*(_TFR**4)*(S_h**2)*(S11h**2+S22h**2+S33h**2
                                            + 2.0*(S12h**2+S13h**2+S23h**2)))
    d2 = pm((4.0*L2**2)*(_TFR**8)*(S_d**2)*(S11d**2+S22d**2+S33d**2
                                            + 2.0*(S12d**2+S13d**2+S23d**2)))
    e1 = pm((_LASD_LILLY_COEFF*L2**2)*(_TFR**2)*S_h*(S11h*SS11h+S22h*SS22h+S33h*SS33h
                                       + 2.0*(S12h*SS12h+S13h*SS13h+S23h*SS23h)))
    e2 = pm((_LASD_LILLY_COEFF*L2**2)*(_TFR**4)*S_d*(S11d*SS11d+S22d*SS22d+S33d*SS33d
                                       + 2.0*(S12d*SS12d+S13d*SS13d+S23d*SS23d)))

    aa = a1*c2 - a2*c1
    bb = a2*e1 - b1*c2
    cc = b2*c1 - a1*e2 - a2*d1
    dd = b1*e2 - b2*e1
    ee = a1*d2 + b2*d1
    ff = -b1*d2
    beta = laguerre_max_real_root_beta(jnp.stack([ff, ee, dd, cc, bb, aa], -1))
    beta3d = beta.reshape(1, 1, nz)

    T1, T2 = 2.0*L2, 2.0*(_TFR**2)*L2
    M11 = T1*SS11h - T2*beta3d*S_h*S11h
    M22 = T1*SS22h - T2*beta3d*S_h*S22h
    M33 = T1*SS33h - T2*beta3d*S_h*S33h
    M12 = T1*SS12h - T2*beta3d*S_h*S12h
    M13 = T1*SS13h - T2*beta3d*S_h*S13h
    M23 = T1*SS23h - T2*beta3d*S_h*S23h
    LM = (L11*M11+L22*M22+L33*M33+2.0*(L12*M12+L13*M13+L23*M23))
    MM = (M11**2+M22**2+M33**2+2.0*(M12**2+M13**2+M23**2))

    LMx, MMx = imfilter_box3(LM, layout), imfilter_box3(MM, layout)
    cs2 = LMx / jnp.where(jnp.abs(MMx) < 1.0e-10, 1.0e-10, MMx)
    invalid = (jnp.abs(MMx) < 1.0e-10) | (cs2 < 0.0) | (cs2 > cs_max ** 2)
    return jnp.where(invalid, 0.0, cs2)                           # C_s² (ny,nx,nz)
