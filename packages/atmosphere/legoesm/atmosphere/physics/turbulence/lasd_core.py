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
filter (cutoff ``round(N/(2·FGR·TFR))``), the 3×3 periodic box average /9, the
max-real-root-in-(0,5) β rule (default 1.0, cross-checked against ``numpy.roots`` — a
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
- the β root is found by exhaustive isolation of the SIGN-CHANGING real roots
  (derivative-root intervals + bisection; a tangent / even-multiplicity root is in fp64
  indistinguishable from a near-real complex pair and is reported only if round-off makes
  ``p`` change sign), NOT the oracle's 8-guess Laguerre,
  which on the coefficients of a real CBL (O(1e-49..1e-17)) stopped after one step on
  every level, could miss the largest of clustered roots, accepted near-real complex
  roots and averaged gradients over guesses; the Germano moments are normalised per
  test-filter level before the quintic is formed (fp32 underflow).
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


_BISECT_ITERS = 64   # bit-pattern halvings: any non-negative fp64 bracket -> adjacent floats
#                      (needs >= 63 in fp64; recheck if the window or dtype changes)
_BETA_LO, _BETA_HI = 0.0, 5.0   # physical β window (oracle ComputeBeta1)  # coeff-ok: β window


def _two_sum(a, b):
    """Knuth's error-free sum: a + b == s + e exactly."""
    s = a + b
    z = s - a
    return s, (a - (s - z)) + (b - z)


def _two_prod(a, b):
    """Dekker's error-free product (Veltkamp split): a * b == p + e exactly."""
    if a.dtype not in (jnp.float32, jnp.float64):
        raise TypeError(f"error-free product needs float32/float64, got {a.dtype}")
    f = 134217729.0 if a.dtype == jnp.float64 else 4097.0   # coeff-ok: 2^27+1 / 2^12+1 splitters
    def split(x):
        t = f * x
        h = t - (t - x)
        return h, x - h
    p = a * b
    ah, al = split(a)
    bh, bl = split(b)
    return p, al * bl - (((p - ah * bh) - al * bh) - ah * bl)


def comp_horner(c, x):
    """Compensated Horner (Graillat, Langlois & Louvet 2005): ``polyval(c, x)`` as if
    in twice the working precision, so the SIGN of p is right much closer to a root
    (e.g. an exact zero at a window edge evaluates to exactly 0). ``c`` DESCENDING."""
    s = jnp.broadcast_to(c[0], jnp.shape(x)).astype(c.dtype)
    err = jnp.zeros_like(s)
    for i in range(1, c.shape[-1]):
        p, pe = _two_prod(s, x)
        s, se = _two_sum(p, c[i])
        err = err * x + (pe + se)
    return s + err


def _real_roots_in(c, lo, hi):
    """Every SIGN-CHANGING real root of ``polyval(c, x)`` in the CLOSED ``[lo, hi]``.

    Exhaustive: the real roots of ``p'`` (recursively, down to the linear case) split
    ``[lo, hi]`` into intervals on which ``p`` is monotone, so each holds at most one
    sign-changing root, found by bisection (signs from :func:`comp_horner`) on the float
    BIT PATTERN (monotone for
    non-negative floats, so ``lo >= 0``): 64 halvings reach adjacent floats, i.e. full
    relative precision for roots above ~2^-110 (fp32) / 2^-990 (fp64) — where the low
    half of x in the Veltkamp split (x·2^-12 / x·2^-27) is still normal; closer to 0 the
    compensated product's low parts go subnormal and only absolute precision is kept
    (irrelevant for β, an O(1) ratio of Smagorinsky coefficients). A zero (as evaluated) at an interval end is
    returned exactly on it.
    Only sign changes certify a root: an even-multiplicity (tangent) root is in fp64
    indistinguishable from a complex pair within ~sqrt(eps) of the axis, and neither is
    reported unless round-off makes ``p`` change sign. ``c`` is ``(n+1,)`` DESCENDING;
    returns ``(n,)`` candidates, NaN where an interval holds no root."""
    if lo < 0:
        raise ValueError(f"bit-pattern bisection needs lo >= 0, got {lo}")
    n = c.shape[-1] - 1
    if n == 1:
        ok = c[0] != 0
        r = jnp.where(ok, -c[1] / jnp.where(ok, c[0], 1.0), jnp.nan)
        return jnp.where((r >= lo) & (r <= hi), r, jnp.nan)[None]
    crit = _real_roots_in(jnp.polyder(c), lo, hi)
    pts = jnp.sort(jnp.where(jnp.isnan(crit), hi, crit))
    a = jnp.concatenate([jnp.full((1,), lo, c.dtype), pts])
    b = jnp.concatenate([pts, jnp.full((1,), hi, c.dtype)])

    # signs from c * 2^S (exact): near a tiny root |p| would otherwise be subnormal and
    # flush to zero; |c| <= 120 (normalised, 4th derivative), x <= 5 and the 2^12+1 split keep
    # every intermediate of comp_horner finite (fp32: 2^88*120*5^5*4097 < 3.4e38)
    cs = c * (2.0 ** (88 if c.dtype == jnp.float32 else 900))   # coeff-ok: exact power-of-two sign scaling, within fp32/fp64 range
    pa, pb = comp_horner(cs, a), comp_horner(cs, b)
    bracket = (b > a) & (jnp.sign(pa) * jnp.sign(pb) <= 0) & ~((pa == 0) & (pb == 0))
    s_lo = jnp.sign(pa)
    itype = jnp.int64 if c.dtype == jnp.float64 else jnp.int32
    # -0.0 -> +0.0 first: the signed bit pattern of -0.0 is not ordered with the positives
    bits = lambda x: jax.lax.bitcast_convert_type(jnp.where(x == 0, 0.0, x).astype(c.dtype),  # noqa: E731
                                                  itype)
    flt = lambda i: jax.lax.bitcast_convert_type(i, c.dtype)         # noqa: E731

    def halve(_, ab):
        i0, i1 = ab
        im = i0 + (i1 - i0) // 2
        right = jnp.sign(comp_horner(cs, flt(im))) == s_lo
        return jnp.where(right, im, i0), jnp.where(right, i1, im)

    i0, i1 = jax.lax.fori_loop(0, _BISECT_ITERS, halve, (bits(a), bits(b)))
    x0, x1 = flt(i0), flt(i1)
    p0, p1 = jnp.abs(comp_horner(cs, x0)), jnp.abs(comp_horner(cs, x1))
    root = jnp.where(pa == 0, a, jnp.where(pb == 0, b, jnp.where(p1 < p0, x1, x0)))
    return jnp.where(bracket, root, jnp.nan)


def _scale_pow2(x, ref):
    """``x / 2^e`` with ``2^e`` the power of two at or above |ref| (``ref == 0``: ``x``).
    EXACT (an exponent shift, applied as two power-of-two factors 2^-(e//2), 2^-(e-e//2)
    so neither overflows near the float limits), so exact zeros of a polynomial survive (a
    root ON a window edge). No gradient flows through ``e``."""
    _, e = jnp.frexp(jax.lax.stop_gradient(ref))
    e = jnp.where(ref == 0, 0, e)
    h = e // 2
    return x * _exact_pow2(-h, x.dtype) * _exact_pow2(-(e - h), x.dtype)


def _exact_pow2(k, dtype):
    """2^k for integer k in the normal range, built from its bit pattern (exact)."""
    if dtype == jnp.float64:
        return jax.lax.bitcast_convert_type((k.astype(jnp.int64) + 1023) << 52, dtype)  # coeff-ok: IEEE-754 binary64 exponent bias / mantissa bits
    return jax.lax.bitcast_convert_type((k.astype(jnp.int32) + 127) << 23, dtype)  # coeff-ok: IEEE-754 binary32 exponent bias / mantissa bits


def _derivative_at(c, x):
    """``p'(x)`` with the coefficients ``k·c_k`` formed error-free (TwoProduct) and
    evaluated compensated — accurate where p' itself suffers cancellation."""
    n = c.shape[-1] - 1
    k = jnp.arange(n, 0, -1).astype(c.dtype)
    hi, lo = _two_prod(k, c[:-1])
    return comp_horner(hi, x) + jnp.polyval(lo, x)


def _beta_search(c):
    """(β, found): largest root of the (already normalised) quintic in the OPEN (0, 5),
    else 1.0. Near an edge the result is limited by the conditioning of ``p`` there: a
    root closer to 0 or 5 than Horner round-off / |p'| may fall on either side."""
    roots = _real_roots_in(c, _BETA_LO, _BETA_HI)
    roots = jnp.where((roots > _BETA_LO) & (roots < _BETA_HI), roots, jnp.nan)
    found = ~jnp.all(jnp.isnan(roots))
    b = jnp.max(jnp.where(jnp.isnan(roots), -jnp.inf, roots))
    return jnp.where(found, b, 1.0), found


@jax.custom_jvp
def _beta_root(c):
    return _beta_search(c)[0]


@_beta_root.defjvp
def _beta_root_jvp(primals, tangents):
    # implicit function theorem at the selected root: dβ = -(Σ dc_i β^(5-i)) / p'(β);
    # β from _beta_root itself so higher derivatives are exact too
    (c,), (dc,) = primals, tangents
    dc = jnp.zeros_like(c) if dc is None else dc
    b = _beta_root(c)
    found = jax.lax.stop_gradient(_beta_search(c)[1])
    dp = _derivative_at(c, b)
    ok = found & (dp != 0)
    return b, jnp.where(ok, -jnp.polyval(dc, b) / jnp.where(ok, dp, 1.0), 0.0)


def max_real_root_beta(coeffs6: jax.Array) -> jax.Array:
    """Largest real root in ``(0, 5)`` of a quintic per row, default ``1.0`` (the oracle's
    ``ComputeBeta1`` rule). ``coeffs6`` is ``(n, 6)`` DESCENDING degree.

    The polynomial is scaled by the power of two above its max |coefficient| (exact;
    roots are scale invariant; the Germano coefficients are O(1e-49..1e-17) in a real
    CBL) and its real roots are
    isolated exhaustively (:func:`_real_roots_in`). The derivative is the exact implicit
    one at the selected root (custom JVP), not an average over search candidates. A row
    with no root in (0, 5), or all-zero coefficients, returns 1.0 with zero gradient.
    Accuracy at a k-fold root is limited to ~eps^(1/k) by evaluation cancellation."""

    if (coeffs6.dtype not in (jnp.float32, jnp.float64) or coeffs6.ndim != 2
            or coeffs6.shape[1] != 6):
        raise TypeError(f"β quintic coefficients must be a real float32/float64 (n, 6) "
                        f"array, got {coeffs6.dtype} {coeffs6.shape}")

    def per_level(coeffs):
        scale = jnp.max(jnp.abs(coeffs))
        degenerate = scale == 0.0
        c = jnp.where(degenerate, jnp.zeros_like(coeffs).at[0].set(1.0),   # x^5: β = 1
                      _scale_pow2(coeffs, scale))
        return jnp.where(degenerate, 1.0, _beta_root(c))

    return jax.vmap(per_level)(coeffs6)


def beta_from_germano_moments(m1, m2):
    """β per level from the level-1 (2Δ) moments ``m1 = (a1, b1, c1, d1, e1)`` and the
    level-2 (4Δ) moments ``m2`` (each ``(nz,)``). Every quintic coefficient is a sum of
    (level-1 × level-2) products, so each level's moments are first normalised by their
    own max (exact: the roots do not change) — keeping the products representable in
    fp32, where unscaled O(1e-25) moments underflow to an all-zero quintic."""
    def unit(m):
        s = jnp.max(jnp.abs(jnp.stack(m)), axis=0)
        return [_scale_pow2(x, s) for x in m]
    a1, b1, c1, d1, e1 = unit(m1)
    a2, b2, c2, d2, e2 = unit(m2)
    aa = a1*c2 - a2*c1
    bb = a2*e1 - b1*c2
    cc = b2*c1 - a1*e2 - a2*d1
    dd = b1*e2 - b2*e1
    ee = a1*d2 + b2*d1
    ff = -b1*d2
    return max_real_root_beta(jnp.stack([ff, ee, dd, cc, bb, aa], -1))


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

    beta = beta_from_germano_moments((a1, b1, c1, d1, e1), (a2, b2, c2, d2, e2))
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
