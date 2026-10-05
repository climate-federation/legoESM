"""Thomas algorithm for tridiagonal systems, JIT-friendly.

Solves the tridiagonal system:

    a[k] * x[k-1] + b[k] * x[k] + c[k] * x[k+1] = d[k]

for k = 0, ..., n-1, where a[0] = 0 and c[n-1] = 0.

Uses jax.lax.scan for the forward/backward sweeps, enabling
efficient JIT compilation and automatic differentiation.

Used by the semi-implicit acoustic substeps in the compressible
Euler equations to solve for vertical velocity w implicitly.

References
----------
- Thomas, L. H. (1949). Elliptic problems in linear difference equations
  over a network. Watson Sci. Comput. Lab. Report, Columbia University.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

_TINY = float(jnp.finfo(jnp.float32).tiny)  # Smallest normal float32 (~1.18e-38)


@partial(jax.custom_vjp, nondiff_argnums=(4,))
def thomas_solve(
    a: jax.Array,
    b: jax.Array,
    c: jax.Array,
    d: jax.Array,
    operation_order: str = "normalised",
) -> jax.Array:
    """Solve a tridiagonal system via the Thomas algorithm.

    Parameters
    ----------
    a : jax.Array, shape (..., n)
        Sub-diagonal coefficients. ``a[..., 0]`` is unused (set to 0).
    b : jax.Array, shape (..., n)
        Main diagonal coefficients (must be nonzero).
    c : jax.Array, shape (..., n)
        Super-diagonal coefficients. ``c[..., -1]`` is unused (set to 0).
    d : jax.Array, shape (..., n)
        Right-hand side vector.

    Returns
    -------
    x : jax.Array, shape (..., n)
        Solution vector.

    Notes
    -----
    The leading dimensions ``...`` are batched over (column-parallel).
    The last dimension is the tridiagonal system size.

    Differentiability: a ``custom_vjp`` supplies the *analytic* adjoint of the
    linear solve (differentiate the solution, not the algorithm).  The reverse
    mode of the raw Thomas recursion divides by ``denom**2`` per pivot, so a
    pivot clamped to ``_TINY`` (a near-singular system — e.g. an implicit
    soil-thermal matrix whose heat capacity collapses) makes the *forward*
    finite but the *backward* overflow to NaN.  The adjoint here instead solves
    the transposed tridiagonal system ``Aᵀ λ = x̄`` with the SAME stable forward
    Thomas sweep, then forms the band/RHS cotangents from ``λ`` and ``x`` — no
    ``1/denom**2`` term ever appears, and the forward values are bit-identical.

    ``operation_order``:
      * ``"normalised"`` (default) divides by ``b + _TINY`` / ``denom +
        _TINY``, i.e. every pivot carries the clamp described above.
      * ``"nemo_unnormalised"`` transcribes NEMO's own forward elimination and
        has NO pivot clamp: it divides by the running diagonal exactly as the
        Fortran does, because adding a clamp would change the arithmetic this
        arm exists to reproduce. That is safe here because its only caller,
        the sea-ice vertical heat solve (``ice/bitz_lipscomb.py``), builds a
        diagonally dominant matrix, whose pivots stay bounded away from zero.
        A new caller must establish the same property before selecting it.

    Limitations (by design, not bugs):
      * The adjoint is the VJP of the IDEAL solve ``A⁻¹d``.  Where the forward
        clamps a (near-singular) pivot to ``_TINY`` it is a surrogate, not the
        exact derivative of the clamped map — that is the whole point (the exact
        derivative is the NaN we are avoiding), and on a well-conditioned system
        it equals the raw element-wise autodiff to machine precision.
      * Reverse-only: ``jax.jvp``/``jacfwd`` through ``thomas_solve`` raise
        (a ``custom_vjp`` defines no JVP).  Pinned by
        ``test_thomas_solve_custom_vjp.py``.
      * FORWARD-OVER-REVERSE (``jax.jvp`` of ``jax.grad``) DOES work, because
        this rule's ``λ`` solve calls the undecorated impl rather than
        re-entering the wrapper (#1736).  An earlier version of this docstring
        claimed "no production/test path forward-diffs this solver"; that was
        false -- ``da/curvature.py``'s ``hvp``, ``gauss_newton_hvp`` and
        ``dense_hessian`` are all forward-over-reverse, and all three raised
        here.  It also claimed the LAPACK ``thomas_solve_batched`` "keeps both
        modes"; its CPU default is ``vmap(thomas_solve)``, so that is only true
        under ``LEGOESM_TRIDIAG=lapack|pcr``.
      * The price of the line above: higher-order reverse mode no longer
        recurses through this rule, so a SECOND derivative taken where a pivot
        is clamped differentiates the raw ``1/denom**2`` and can overflow in
        float32.  float64 is unaffected, and the measured pivot margin on the
        production soil column is ~7e33 above the clamp.  Pinned by test.
    """
    return _thomas_solve_impl(a, b, c, d, operation_order)


def _thomas_solve_impl(
    a: jax.Array,
    b: jax.Array,
    c: jax.Array,
    d: jax.Array,
    operation_order: str = "normalised",
) -> jax.Array:
    """Raw Thomas forward sweep (the primal computation; see ``thomas_solve``)."""
    n = b.shape[-1]

    # Promote a/b/c/d to a common working dtype before solving. Under x64 the
    # implicit-diffusion coefficients (a,b,c built from f64 K_v/dz/dt) are f64
    # while the RHS d (the f32-storage tracer field) is f32; the per-row updates
    # then promote to f64 and scatter into the f32 c_star/d_star/x buffers -> an
    # implicit f64->f32 downcast FutureWarning (a future JAX error). Solve in the
    # higher precision, then return the solution in the RHS dtype so callers'
    # state stores stay dtype-stable. (The prior mixed-precision path downcast the
    # d-derived terms to f32 anyway, so this is strictly more accurate.)
    out_dtype = d.dtype
    work_dtype = jnp.result_type(a, b, c, d)
    a = jnp.asarray(a, work_dtype)
    b = jnp.asarray(b, work_dtype)
    c = jnp.asarray(c, work_dtype)
    d = jnp.asarray(d, work_dtype)

    if operation_order == "nemo_unnormalised":
        diagonal = b
        rhs = d

        def unnormalised_forward(k, carry):
            diagonal_k, rhs_k = carry
            previous = diagonal_k[..., k - 1]
            diagonal_k = diagonal_k.at[..., k].set(
                diagonal_k[..., k]
                - (a[..., k] * c[..., k - 1]) / previous
            )
            rhs_k = rhs_k.at[..., k].set(
                rhs_k[..., k]
                - (a[..., k] * rhs_k[..., k - 1]) / previous
            )
            return diagonal_k, rhs_k

        diagonal, rhs = jax.lax.fori_loop(
            1, n, unnormalised_forward, (diagonal, rhs)
        )
        solution = jnp.zeros_like(rhs)
        solution = solution.at[..., -1].set(
            rhs[..., -1] / diagonal[..., -1]
        )

        def unnormalised_backward(reverse_k, current):
            k = n - 2 - reverse_k
            return current.at[..., k].set(
                (rhs[..., k] - c[..., k] * current[..., k + 1])
                / diagonal[..., k]
            )

        solution = jax.lax.fori_loop(
            0, n - 1, unnormalised_backward, solution
        )
        return jax.lax.convert_element_type(solution, out_dtype)

    if operation_order != "normalised":
        raise ValueError(f"unknown Thomas operation order {operation_order!r}")

    # Sweep over the system axis moved to the FRONT, so every level is one
    # contiguous (batch,) slice.  Updating ``[..., k]`` of a (batch, n) array
    # in a fori_loop touched a strided column per level and was 4-7x slower
    # on CPU (2026-09-25, 32 Milan cores, (35972, 40) f64: 63 -> 16 ms); the
    # per-element arithmetic is unchanged (bit-identical to the loop on the
    # CPU XLA build measured; a compiler that contracts to FMA differently
    # may move the last bit).  Operands are broadcast first so the output is
    # the full broadcast shape by construction.
    a, b, c, d = (jnp.moveaxis(v, -1, 0)
                  for v in jnp.broadcast_arrays(a, b, c, d))
    c0_star = c[0] / (b[0] + _TINY)
    d0_star = d[0] / (b[0] + _TINY)
    if n == 1:
        # no sweep to run (and lax.scan refuses zero-length scans without jit)
        return jax.lax.convert_element_type(d0_star[..., None], out_dtype)

    def forward_body(carry, xs):
        c_prev, d_prev = carry
        ak, bk, ck, dk = xs
        denom = bk - ak * c_prev
        denom = jnp.where(jnp.abs(denom) < _TINY, _TINY, denom)
        c_star_k = ck / denom
        d_star_k = (dk - ak * d_prev) / denom
        return (c_star_k, d_star_k), (c_star_k, d_star_k)

    _, (c_star, d_star) = jax.lax.scan(
        forward_body, (c0_star, d0_star), (a[1:], b[1:], c[1:], d[1:]))
    c_star = jnp.concatenate([c0_star[None], c_star])
    d_star = jnp.concatenate([d0_star[None], d_star])

    # Backward substitution: x[n-1] = d_star[n-1], x[k] = d*[k] - c*[k] x[k+1]
    def backward_body(x_next, xs):
        c_k, d_k = xs
        x_k = d_k - c_k * x_next
        return x_k, x_k

    _, x = jax.lax.scan(backward_body, d_star[-1], (c_star[:-1], d_star[:-1]),
                        reverse=True)
    x = jnp.moveaxis(jnp.concatenate([x, d_star[-1:]]), 0, -1)

    return jax.lax.convert_element_type(x, out_dtype)


def _thomas_solve_fwd(a, b, c, d, operation_order):
    x = _thomas_solve_impl(a, b, c, d, operation_order)
    return x, (a, b, c, x, d.shape)


def _thomas_solve_bwd(operation_order, res, x_bar):
    """Adjoint of ``A x = d`` (A tridiagonal): d̄ = A⁻ᵀ x̄ =: λ, and the band
    cotangents from ``x = A⁻¹ d`` ⇒ Ā = -λ xᵀ restricted to the three bands:
    ā[k] = -λ[k] x[k-1], b̄[k] = -λ[k] x[k], c̄[k] = -λ[k] x[k+1]."""
    a, b, c, x, res_d_shape = res
    work = jnp.result_type(a, b, c, x, x_bar)
    aw = jnp.asarray(a, work); bw = jnp.asarray(b, work); cw = jnp.asarray(c, work)
    xw = jnp.asarray(x, work); xbar = jnp.asarray(x_bar, work)
    # The primal broadcasts its operands before sweeping; the band shifts below
    # must act on the FULL system axis, so a singleton band (a constant sub- or
    # super-diagonal) is expanded first and its cotangent reduced back at the end.
    aw, bw, cw, xw, xbar = jnp.broadcast_arrays(aw, bw, cw, xw, xbar)

    # Transposed system Aᵀ λ = x̄.  Aᵀ has sub-diag aT[k]=c[k-1], super-diag
    # cT[k]=a[k+1], same main diag b.  Solve with the SAME stable forward sweep,
    # calling the UNDECORATED impl rather than re-entering the wrapper (#1736).
    #
    # WHY NOT THE WRAPPER: re-entering it makes this rule's own body contain a
    # custom_vjp call, and forward-differentiating a custom_vjp is undefined in
    # JAX.  That made ``jax.jvp(jax.grad(f))`` -- i.e. every forward-over-reverse
    # second-order operator in da/curvature.py -- raise on any objective
    # containing this solve.  Measured: the refusal is raised HERE, at this
    # line, not at the primal call; routing this one solve to the impl makes
    # the land-column HVP finite and match a central finite difference to
    # rel 3.6e-09 against the test's 5e-3 bar, with the first-order gradient
    # bit-unchanged.
    #
    # WHAT IT COSTS: higher-order reverse mode no longer recurses through this
    # rule, so a SECOND derivative taken where a pivot is clamped differentiates
    # the raw recursion's 1/denom**2 and can overflow in float32 (float64 is
    # fine).  That trade is pinned by a test rather than left to prose; see
    # tests/unit/test_thomas_solve_custom_vjp.py.  The first-order adjoint
    # below is unaffected -- it never forms 1/denom**2.
    zc = jnp.zeros_like(cw[..., :1])
    aT = jnp.concatenate([zc, cw[..., :-1]], axis=-1)
    cT = jnp.concatenate([aw[..., 1:], jnp.zeros_like(aw[..., :1])], axis=-1)
    lam = _thomas_solve_impl(aT, bw, cT, xbar, operation_order)

    x_km1 = jnp.concatenate([jnp.zeros_like(xw[..., :1]), xw[..., :-1]], axis=-1)
    x_kp1 = jnp.concatenate([xw[..., 1:], jnp.zeros_like(xw[..., :1])], axis=-1)
    a_bar = -lam * x_km1
    b_bar = -lam * xw
    c_bar = -lam * x_kp1
    a_bar = a_bar.at[..., 0].set(0.0)    # a[..., 0] is unused (structurally 0)
    c_bar = c_bar.at[..., -1].set(0.0)   # c[..., -1] is unused (structurally 0)
    d_bar = lam

    # The primal broadcasts a/b/c/d to a common shape; a cotangent must come
    # back in each PRIMAL's shape, summed over the axes broadcasting added.
    return (jnp.asarray(_sum_to_shape(a_bar, a.shape), a.dtype),
            jnp.asarray(_sum_to_shape(b_bar, b.shape), b.dtype),
            jnp.asarray(_sum_to_shape(c_bar, c.shape), c.dtype),
            jnp.asarray(_sum_to_shape(d_bar, res_d_shape), x.dtype))


def _sum_to_shape(g: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    """Reduce a full-broadcast cotangent to the primal's ``shape`` (the
    transpose of ``jnp.broadcast_to``)."""
    if g.shape == tuple(shape):
        return g
    lead = g.ndim - len(shape)
    g = jnp.sum(g, axis=tuple(range(lead)))
    keep = tuple(i for i, (n, m) in enumerate(zip(g.shape, shape)) if n != m)
    return jnp.sum(g, axis=keep, keepdims=True) if keep else g


thomas_solve.defvjp(_thomas_solve_fwd, _thomas_solve_bwd)


def pcr_solve_batched(
    a: jax.Array,
    b: jax.Array,
    c: jax.Array,
    d: jax.Array,
) -> jax.Array:
    """Parallel cyclic reduction for batched tridiagonal systems.

    Pure-JAX. Unlike :func:`thomas_solve_batched` which calls cuSPARSE
    via a custom_call (fusion barrier), PCR is a sequence of elementwise
    ops that XLA can fuse into surrounding compute. For the small
    nlev (~30) used in the SI acoustic substep this is comparable in
    raw speed to cuSPARSE but enables single-kernel substep bodies.

    Algorithm: at level k (stride = 2^k), each row eliminates its
    sub/super-diagonal contributions from the row stride positions
    away. After log2(n_pad) levels, every row is decoupled and
    ``x = d / b`` solves the system trivially.

    Shape ``(..., n)`` for a/b/c/d (n is the tridiag system size, last
    axis); leading axes are batched over. n need not be a power of 2 —
    the system is padded to the next power of 2 with identity rows
    (``b=1, a=c=d=0``) which decouple from the original system.

    Stability assumption
    --------------------
    PCR is pivot-free; roundoff growth depends on diagonal dominance.
    The SI acoustic system has ``b ~ 1 + alpha`` (alpha > 0), so b is
    bounded away from zero — PCR is stable in this regime. Empirically
    (iter-73 sweep at dt=2s, dx=2km, plane CRM):
    - n_substeps=2 (acoustic CFL ≈ 0.17): **NaN** within 30 steps
    - n_substeps=3 (CFL ≈ 0.113): stable
    The practical stability bound under default off-centering
    (``beta=0``) is between CFL=0.113 and 0.17 — narrower than the
    canonical 0.7 acoustic CFL because of off-centering + buoyancy +
    advection coupling. For weakly-diagonal-dominant systems or to
    A/B-validate, fall back via ``LEGOESM_TRIDIAG=legacy`` or
    ``=cusparse``.

    AD memory cost
    --------------
    Each PCR level materializes ~12 intermediate arrays (a, b, c, d
    plus a_up, b_up, c_up, d_up, alpha, beta, new a/b/c/d). For
    n_sys=29 → 5 levels → ~60 intermediates per substep call.
    Reverse-mode AD captures all forward ops, so peak AD memory
    grows with substep count × levels × state size. Compared to
    cuSPARSE (which has a single custom_call with internal-only
    state), PCR's AD footprint is larger by roughly 5-10× per
    substep. Training with ``jax.grad`` on large grids may require
    activation checkpointing.

    Division-in-graph caveat
    ------------------------
    ``jnp.where`` does NOT short-circuit. ``alpha = jnp.where(has_above,
    -a / b_up, 0.0)`` evaluates the division at all rows including those
    masked out. ``b_up`` is set to 1.0 at masked rows via constant-pad,
    so the division is safe. If a future caller introduces a real row
    with b==0, the graph will propagate NaN even where masked.

    Compile-time
    ------------
    The Python for-loop unrolls log2(n_pad) levels at trace time. Each
    level adds ~12 elementwise ops to the traced graph. For n=29
    (5 levels) × 6 substeps × 3 RK3 stages × outer jit = ~5s compile
    in practice. Acceptable for JIT-once workloads.
    """
    if a.shape != b.shape or a.shape != c.shape or a.shape != d.shape:
        raise ValueError(
            f"pcr_solve_batched expects matching shapes; got a={a.shape}, "
            f"b={b.shape}, c={c.shape}, d={d.shape}"
        )
    if a.ndim < 1:
        raise ValueError(
            f"pcr_solve_batched requires at least 1 axis (the tridiag "
            f"system axis as the last dim); got shape {a.shape}"
        )
    if a.shape[-1] < 2:
        raise ValueError(
            f"pcr_solve_batched requires n>=2 on trailing axis; got {a.shape[-1]}"
        )

    import math
    n = a.shape[-1]
    pad_axes = ((0, 0),) * (a.ndim - 1)
    n_pad = 1 << max(1, (n - 1).bit_length())
    pad_n = n_pad - n

    if pad_n > 0:
        a = jnp.pad(a, (*pad_axes, (0, pad_n)))
        b = jnp.pad(b, (*pad_axes, (0, pad_n)), constant_values=1.0)
        c = jnp.pad(c, (*pad_axes, (0, pad_n)))
        d = jnp.pad(d, (*pad_axes, (0, pad_n)))

    idx = jnp.arange(n_pad)
    n_levels = int(math.log2(n_pad))

    for k in range(n_levels):
        stride = 1 << k
        has_above = idx >= stride
        has_below = idx < n_pad - stride

        a_up = jnp.pad(a[..., :-stride], (*pad_axes, (stride, 0)))
        b_up = jnp.pad(
            b[..., :-stride], (*pad_axes, (stride, 0)), constant_values=1.0,
        )
        c_up = jnp.pad(c[..., :-stride], (*pad_axes, (stride, 0)))
        d_up = jnp.pad(d[..., :-stride], (*pad_axes, (stride, 0)))

        a_dn = jnp.pad(a[..., stride:], (*pad_axes, (0, stride)))
        b_dn = jnp.pad(
            b[..., stride:], (*pad_axes, (0, stride)), constant_values=1.0,
        )
        c_dn = jnp.pad(c[..., stride:], (*pad_axes, (0, stride)))
        d_dn = jnp.pad(d[..., stride:], (*pad_axes, (0, stride)))

        alpha = jnp.where(has_above, -a / b_up, 0.0)
        beta = jnp.where(has_below, -c / b_dn, 0.0)

        a = alpha * a_up
        c = beta * c_dn
        b_new = b + alpha * c_up + beta * a_dn
        d = d + alpha * d_up + beta * d_dn
        b = b_new

    x = d / b
    if pad_n > 0:
        x = x[..., :n]
    return x


def thomas_solve_shared(
    a: jax.Array,
    b: jax.Array,
    c: jax.Array,
    ds: "tuple[jax.Array, ...]",
) -> "tuple[jax.Array, ...]":
    """Thomas solve with ONE factorization and SEVERAL right-hand sides.

    All ``ds`` share the same tridiagonal matrix ``(a, b, c)`` — the
    forward-elimination factors (``denom``, ``c_star``) are computed
    ONCE and reused for every RHS, instead of once per
    :func:`thomas_solve` call.  The arithmetic per RHS is kept
    operation-for-operation identical to :func:`thomas_solve` (same
    ``_TINY`` clamps, same loop order, same dtype promotion when all
    RHS share a dtype), so each returned solution is BIT-IDENTICAL to
    the corresponding single-RHS call.  Use case: the ocean implicit
    vertical mixing solves T and S against the same diffusivity matrix
    (vmix is memory-bandwidth-bound — building/sweeping the
    coefficients once is the saving; jobs 8458934/8459136).

    Falls back to per-RHS :func:`thomas_solve` when the RHS dtypes
    differ (a shared work dtype would change the promotion of the
    narrower field and break bit-faithfulness).

    Parameters
    ----------
    a, b, c : jax.Array, shape (..., n)
        Shared tridiagonal coefficients (conventions of
        :func:`thomas_solve`).
    ds : tuple of jax.Array, each shape (..., n)
        Right-hand sides.

    Returns
    -------
    tuple of jax.Array — solutions, in input order, each in its RHS
    dtype.
    """
    ds = tuple(ds)
    if not ds:
        return ()
    if len(ds) == 1 or len({d.dtype for d in ds}) != 1:
        return tuple(thomas_solve(a, b, c, d) for d in ds)

    n = b.shape[-1]
    out_dtype = ds[0].dtype
    work_dtype = jnp.result_type(a, b, c, ds[0])
    a = jnp.asarray(a, work_dtype)
    b = jnp.asarray(b, work_dtype)
    c = jnp.asarray(c, work_dtype)
    ds = tuple(jnp.asarray(d, work_dtype) for d in ds)

    # Same scan form as _thomas_solve_impl (see the note there on why not
    # fori_loop + .at[k].set).
    c0_star = c[..., 0] / (b[..., 0] + _TINY)
    d0_stars = tuple(d[..., 0] / (b[..., 0] + _TINY) for d in ds)
    if n == 1:
        return tuple(
            jax.lax.convert_element_type(d0[..., None], out_dtype)
            for d0 in d0_stars
        )
    am, bm, cm = (jnp.moveaxis(v, -1, 0) for v in (a, b, c))
    dms = tuple(jnp.moveaxis(d, -1, 0) for d in ds)

    def forward_body(carry, k_inputs):
        c_prev, d_prevs = carry
        ak, bk, ck, dks = k_inputs
        denom = bk - ak * c_prev
        denom = jnp.where(jnp.abs(denom) < _TINY, _TINY, denom)
        c_star_k = ck / denom
        d_star_ks = tuple(
            (dk - ak * d_prev) / denom for dk, d_prev in zip(dks, d_prevs)
        )
        return (c_star_k, d_star_ks), (c_star_k, d_star_ks)

    _, (c_rest, d_rests) = jax.lax.scan(
        forward_body, (c0_star, d0_stars),
        (am[1:], bm[1:], cm[1:], tuple(dm[1:] for dm in dms)),
    )
    c_star = jnp.concatenate([c0_star[None], c_rest], axis=0)
    d_stars = tuple(
        jnp.concatenate([d0[None], dr], axis=0)
        for d0, dr in zip(d0_stars, d_rests)
    )

    def backward_body(x_nexts, k_inputs):
        c_star_k, d_star_ks = k_inputs
        x_ks = tuple(
            d_star_k - c_star_k * x_next
            for d_star_k, x_next in zip(d_star_ks, x_nexts)
        )
        return x_ks, x_ks

    x_lasts = tuple(ds_[-1] for ds_ in d_stars)
    _, x_rests = jax.lax.scan(
        backward_body, x_lasts,
        (c_star[:-1], tuple(ds_[:-1] for ds_ in d_stars)), reverse=True,
    )
    xs = tuple(
        jnp.moveaxis(jnp.concatenate([xr, xl[None]], axis=0), 0, -1)
        for xr, xl in zip(x_rests, x_lasts)
    )
    return tuple(
        jax.lax.convert_element_type(x, out_dtype) for x in xs
    )


def thomas_solve_batched(
    a: jax.Array,
    b: jax.Array,
    c: jax.Array,
    d: jax.Array,
) -> jax.Array:
    """Batched Thomas solve over columns.

    On GPU (CUDA): wraps cuSPARSE's batched tridiagonal solver via
    ``jax.lax.linalg.tridiagonal_solve``. On other backends (CPU/Metal/TPU)
    where the GPU primitive may be missing/slow: falls back to the legacy
    fori_loop Thomas.

    Verified correctness (iter-39/40 PR #320):
    - fp64 max residual 9.99e-16 (machine precision)
    - fp32 max residual ~5e-7 (machine precision)
    - AD via `jax.grad` works; gradients finite with mean ~1.0 for
      sum-of-output test
    - 6 acoustic-substep tests PASS

    Vertical axis convention: ``a, b, c, d`` MUST have the tridiagonal
    system as the LAST axis. Leading axes are batched over.

    Parameters
    ----------
    a : jax.Array, shape (..., n_sys)
        Sub-diagonal coefficients. ``a[..., 0]`` is unused (set to 0).
    b : jax.Array, shape (..., n_sys)
        Main diagonal coefficients (must be nonzero).
    c : jax.Array, shape (..., n_sys)
        Super-diagonal coefficients. ``c[..., -1]`` is unused (set to 0).
    d : jax.Array, shape (..., n_sys)
        Right-hand side.

    Returns
    -------
    x : jax.Array, shape (..., n_sys)
    """
    # Stronger argument validation per codex iter-57 review:
    if a.shape != b.shape or a.shape != c.shape or a.shape != d.shape:
        raise ValueError(
            f"thomas_solve_batched expects a/b/c/d to share shape; "
            f"got a={a.shape}, b={b.shape}, c={c.shape}, d={d.shape}"
        )
    if a.dtype != b.dtype or a.dtype != c.dtype or a.dtype != d.dtype:
        raise ValueError(
            f"thomas_solve_batched expects a/b/c/d to share dtype; "
            f"got a={a.dtype}, b={b.dtype}, c={c.dtype}, d={d.dtype}"
        )
    if a.ndim < 1:
        raise ValueError(
            f"thomas_solve_batched requires at least 1 axis (the tridiag "
            f"system axis as the last dim); got shape {a.shape}"
        )
    if a.shape[-1] < 2:
        raise ValueError(
            f"thomas_solve_batched requires n_sys >= 2 on the trailing "
            f"axis; got n_sys={a.shape[-1]}"
        )

    # Backend selection priority (post iter-72):
    #   1. env LEGOESM_TRIDIAG=pcr      -> pure-JAX PCR (default on GPU)
    #   2. env LEGOESM_TRIDIAG=cusparse -> jax.lax.linalg.tridiagonal_solve (custom_call)
    #   3. env LEGOESM_TRIDIAG=lapack   -> jax.lax.linalg.tridiagonal_solve on ANY
    #                                       backend (CPU LAPACK ``gtsv`` / GPU
    #                                       cuSPARSE); the CPU win over the
    #                                       fori_loop legacy (~2000x). Opt-in
    #                                       (scaling review 2026-06-13 lever #8).
    #   4. env LEGOESM_TRIDIAG=legacy   -> fori_loop Thomas (debug)
    #   5. CUDA backend                  -> PCR (new default; +51% throughput
    #                                       over cuSPARSE at peak via XLA fusion)
    #   6. Else                          -> legacy fori_loop
    #
    # The FFI primitive is AD-safe: jax registers a JVP + transpose +
    # batching rule for ``tridiagonal_solve_p`` (reverse-mode differentiable,
    # vmap-able).  Kept OPT-IN (not the CPU default) because the CPU LAPACK
    # ``gtsv_ffi`` lowering requires a recent jaxlib (>= 0.4.35); the
    # fori_loop legacy stays the universal, version-independent default so a
    # contributor on an older jax is never broken by a missing lowering.
    #
    # NOTE: the env var is read at JIT trace time and baked into the
    # compiled graph; changing the env var after JIT compile has no
    # effect on a cached compilation. Set it BEFORE importing legoesm
    # in your driver script.
    import os
    forced = os.environ.get("LEGOESM_TRIDIAG", "").lower()
    if forced == "pcr":
        return pcr_solve_batched(a, b, c, d)
    if forced == "lapack":
        return _ffi_tridiagonal_solve(a, b, c, d)
    if forced == "cusparse":
        try:
            from jax.lax.linalg import tridiagonal_solve  # noqa: F401
            on_gpu = jax.default_backend() in ("gpu", "cuda")
        except (ImportError, AttributeError):
            on_gpu = False
        if on_gpu:
            return _ffi_tridiagonal_solve(a, b, c, d)
        return _thomas_solve_batched_legacy(a, b, c, d)
    if forced == "legacy":
        return _thomas_solve_batched_legacy(a, b, c, d)

    # No env override: prefer PCR on GPU (best perf), legacy on CPU/Metal/TPU
    # (PCR has more elementwise ops; on CPU the simpler Thomas is competitive
    # and PCR's pad-heavy structure may not lower as cleanly).
    try:
        on_gpu = jax.default_backend() in ("gpu", "cuda")
    except AttributeError:
        on_gpu = False
    if on_gpu:
        return pcr_solve_batched(a, b, c, d)
    return _thomas_solve_batched_legacy(a, b, c, d)


def _ffi_tridiagonal_solve(
    a: jax.Array,
    b: jax.Array,
    c: jax.Array,
    d: jax.Array,
) -> jax.Array:
    """Batched tridiagonal solve via ``jax.lax.linalg.tridiagonal_solve``.

    Backend-agnostic FFI primitive: lowers to LAPACK ``gtsv`` on CPU and to
    cuSPARSE ``gtsv2`` on CUDA — both single, natively-batched invocations.
    Same numerical behavior as :func:`pcr_solve_batched` and
    :func:`_thomas_solve_batched_legacy` to machine epsilon (LAPACK ``gtsv``
    uses partial pivoting, so it is at least as stable as the plain Thomas
    sweep for well-conditioned vertical operators).

    AD-safe: ``tridiagonal_solve_p`` carries a JVP + transpose rule, so this
    is reverse-mode differentiable (``jax.grad``) and vmap-able.

    Diagonal convention (matches :func:`thomas_solve_batched`): ``a`` is the
    sub-diagonal (``a[...,0]`` unused), ``b`` the main diagonal, ``c`` the
    super-diagonal (``c[...,-1]`` unused), ``d`` the RHS — passed as
    ``tridiagonal_solve(dl=a, d=b, du=c, b=d)``.
    """
    from jax.lax.linalg import tridiagonal_solve
    import math

    orig_shape = a.shape
    spatial_shape = orig_shape[:-1]
    n_sys = orig_shape[-1]
    # ``math.prod(())`` is 1 (the scalar-leading 1D case reshapes to
    # (1, n_sys)); a zero-size leading batch must stay 0, NOT be clamped to
    # 1 — ``max(1, ...)`` tried to reshape a size-0 array into (1, n_sys)
    # and crashed, unlike the legacy path (codex 2026-06-13 MED).
    n_cols = math.prod(spatial_shape)

    a_flat = a.reshape(n_cols, n_sys)
    b_flat = b.reshape(n_cols, n_sys)
    c_flat = c.reshape(n_cols, n_sys)
    d_flat = d.reshape(n_cols, n_sys)

    x_flat = tridiagonal_solve(
        a_flat, b_flat, c_flat, d_flat[..., None],
    )[..., 0]
    return x_flat.reshape(orig_shape)


def _thomas_solve_batched_legacy(
    a: jax.Array,
    b: jax.Array,
    c: jax.Array,
    d: jax.Array,
) -> jax.Array:
    """Legacy fori_loop-based batched Thomas. Kept for CPU fallback /
    regression testing. ~2000× slower than the cuSPARSE-backed default."""
    orig_shape = a.shape
    spatial_shape = orig_shape[:-1]
    n_sys = orig_shape[-1]
    n_cols = 1
    for s in spatial_shape:
        n_cols *= s

    a_flat = a.reshape(n_cols, n_sys)
    b_flat = b.reshape(n_cols, n_sys)
    c_flat = c.reshape(n_cols, n_sys)
    d_flat = d.reshape(n_cols, n_sys)
    x_flat = jax.vmap(thomas_solve)(a_flat, b_flat, c_flat, d_flat)
    return x_flat.reshape(orig_shape)


def cyclic_thomas_batched(
    a: jax.Array,
    b: jax.Array,
    c: jax.Array,
    d: jax.Array,
) -> jax.Array:
    """Batched PERIODIC (cyclic) tridiagonal solve via Sherman–Morrison.

    Solves ``C x = d`` where ``C`` is tridiagonal with periodic wrap
    terms: row ``0`` couples ``x[n-1]`` with coefficient ``a[..., 0]``
    (the wrapped sub-diagonal) and row ``n-1`` couples ``x[0]`` with
    coefficient ``c[..., -1]`` (the wrapped super-diagonal) — the
    natural convention for a zonally periodic stencil where ``a[i]``
    multiplies ``x[i-1 mod n]`` and ``c[i]`` multiplies ``x[i+1 mod n]``.

    Standard Sherman–Morrison rank-1 reduction (Numerical Recipes §2.7):
    two :func:`thomas_solve_batched` solves of the same modified
    tridiagonal system plus a rank-1 correction.  Pure ``jnp`` ops —
    JIT/scan/vmap/AD-safe; cost = 2 Thomas solves per call.

    Parameters mirror :func:`thomas_solve_batched` (system on the LAST
    axis, leading axes batched), except ``a[..., 0]`` and ``c[..., -1]``
    are USED (the periodic wrap coefficients).  Requires ``n_sys >= 3``.
    Diagonal dominance of the underlying operator keeps the modified
    system (``b[...,0] - gamma``, ``b[...,-1] - a0*c_last/gamma`` with
    ``gamma = -b[...,0]``) safely factorizable for the Helmholtz-class
    matrices this serves (diag > 0, off-diag <= 0).
    """
    if a.shape != b.shape or a.shape != c.shape or a.shape != d.shape:
        raise ValueError(
            f"cyclic_thomas_batched expects a/b/c/d to share shape; got "
            f"a={a.shape}, b={b.shape}, c={c.shape}, d={d.shape}"
        )
    if a.shape[-1] < 3:
        raise ValueError(
            f"cyclic_thomas_batched requires n_sys >= 3 on the trailing "
            f"axis; got n_sys={a.shape[-1]}"
        )
    alpha = c[..., -1]            # row n-1 -> col 0 (wrapped super-diag)
    beta = a[..., 0]              # row 0 -> col n-1 (wrapped sub-diag)
    gamma = -b[..., 0]            # NR convention (avoids zero pivot)

    b_mod = b.at[..., 0].add(-gamma)
    b_mod = b_mod.at[..., -1].add(-alpha * beta / gamma)
    # Zero the (unused-by-Thomas but validated) wrap entries.
    a_mod = a.at[..., 0].set(0.0)
    c_mod = c.at[..., -1].set(0.0)

    y = thomas_solve_batched(a_mod, b_mod, c_mod, d)
    u = jnp.zeros_like(d)
    u = u.at[..., 0].set(gamma)
    u = u.at[..., -1].set(alpha)
    z = thomas_solve_batched(a_mod, b_mod, c_mod, u)

    vy = y[..., 0] + beta * y[..., -1] / gamma
    vz = z[..., 0] + beta * z[..., -1] / gamma
    factor = vy / (1.0 + vz)
    return y - z * factor[..., None]
