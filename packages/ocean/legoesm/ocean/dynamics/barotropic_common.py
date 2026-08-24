"""Grid-agnostic helpers shared by barotropic_*.py substep solvers.

Factors small repeated patterns out of the four grid-specific
barotropic solvers (cubed-sphere A-grid, cubed-sphere C-grid, lat-lon
C-grid, MPAS).  Closes #214 (Phase 2).

The helpers are intentionally thin: each replaces a few lines that were
character-for-character identical across solvers, so future updates to
the BEBT scheme, the cosine time filter, or MAXVEL clipping touch one
file instead of three.

This module performs no halo exchange and does not depend on any
grid-specific operator package.  Callers are responsible for staging
field shapes correctly before invoking these helpers.

The one exception is :func:`solve_helmholtz_implicit`, which DOES carry
the distributed-solver logic shared by the implicit Crank-Nicolson
free-surface solvers (lat-lon C-grid and MPAS Voronoi).  The
``A_op``/``M_inv`` operators it receives are opaque callables — the halo
exchange and area metrics live inside them, supplied by the grid-specific
caller — so this module still touches no operator package directly.
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp


def compute_power_law_filter_weights(
    n_substeps: int,
    dtype: jnp.dtype,
    *,
    p: int = 2,
    q: int = 4,
    r: float = 0.18927,
):
    """Shchepetkin & McWilliams (2005) power-law barotropic averaging filter.

    This is the averaging kernel used by ROMS, MOM6 and Oceananigans'
    ``SplitExplicitFreeSurface`` (``averaging_shape_function``).  Unlike the
    cosine (Hanning) filter — which is only first-order accurate and is
    *documented* to excite the 2Δx barotropic checkerboard (the C-grid Coriolis
    rotational null mode; see docs/issues/barotropic_mode_noise.md §B) — the
    power-law filter spans an EXTENDED window τ∈(0, 2] (i.e. the barotropic
    substeps run for ~1.5× the baroclinic step) with the shape

        w(τ) = (τ/τ₀)^p · (1 − (τ/τ₀)^q) − r·(τ/τ₀),
        τ₀ = (p+2)(p+q+2) / [(p+1)(p+q+1)],

    whose centroid sits at the baroclinic step (τ=1) and which strongly damps
    the grid mode.  The window is trimmed at the last positive weight ``M★``
    and the averaging weights are normalised to sum to 1.  ``transport_weights``
    (the SM2005 secondary weights) keep the time-averaged barotropic transport
    consistent with the SSH evolution for volume conservation.

    Returns
    -------
    (w_avg, w_total, w_transport, n_loop) : the per-substep averaging weights
        (length ``n_loop`` = M★), their sum (== 1), the transport weights, and
        the (static) number of substeps to run.  ``n_loop`` > ``n_substeps``
        because the window extends past the baroclinic step.
    """
    import numpy as _np
    tau0 = (p + 2) * (p + q + 2) / ((p + 1) * (p + q + 1))
    # τ resolved at the same Δτ = dt/n_substeps as the cosine path, spanning
    # (0, 2]; this is 2·n_substeps candidate substeps before trimming.
    tau = _np.arange(1, 2 * n_substeps + 1, dtype=_np.float64) / n_substeps
    w = (tau / tau0) ** p * (1.0 - (tau / tau0) ** q) - r * (tau / tau0)
    m_star = int(_np.max(_np.where(w > 0.0)[0]) + 1)   # last positive weight
    w = w[:m_star]
    w = w / w.sum()
    # SM2005 transport weights: transport_w[i] = sum(w[i:]) / n_substeps so the
    # cumulative averaged transport closes the depth-integrated continuity.
    w_transport = _np.array(
        [w[i:].sum() for i in range(m_star)], dtype=_np.float64
    ) / n_substeps
    return (
        jnp.asarray(w, dtype=dtype),
        jnp.asarray(w.sum(), dtype=dtype),
        jnp.asarray(w_transport, dtype=dtype),
        m_star,
    )


def compute_nemo_boxcar_centred_weights(
    n_substeps: int,
    dtype: jnp.dtype,
    substep_scale: int = 1,
):
    """NEMO dynspg_ts centred boxcar averaging (ln_bt_fw=F, nn_bt_flt=2).

    NEMO's centred split-explicit runs the barotropic past the
    baroclinic step and averages η/U over a boxcar of width ``2·nn_e``
    CENTRED on the new-time point (ts_wgt CASE(2): ``zwgt1(jn)=1`` where
    ``|jn − jic|/nn_e < 1``; the centre ``jic = 2·nn_e`` lands on the
    new-time point of a 2Δt integration, i.e. ``jn = n_substeps``).

    ``substep_scale`` (default 1) is the ``_barotropic_substep_scale`` the
    MLF caller applies: it feeds the SCALED substep count ``n_substeps =
    nn_e · substep_scale`` (DINO: 23·2 = 46) so the substep length stays
    CFL-safe over the 2Δt leap-frog window.  The boxcar HALF-width must
    remain the UNSCALED ``nn_e = n_substeps / substep_scale`` (NEMO's
    ``nn_e``), NOT the scaled count — otherwise the window doubles
    (±2·nn_e) and ``n_loop`` becomes ``2·n_substeps − 1`` (91 for DINO)
    instead of NEMO's ``icycle = 2·nn_e = 68``.  With ``substep_scale=1``
    (every forward-Euler caller) ``half_width == n_substeps`` and the
    weights are byte-identical to the pre-fix path.  The window therefore
    spans τ ∈ (0, 2) baroclinic steps of length ``nn_e·dt_s``: the loop
    runs to the last in-window substep ``jn = n_substeps + nn_e − 1``
    (``n_loop = 68`` for DINO; ``2·n − 1`` when ``substep_scale=1``).
    This is the DINO namelist value (namdyn_spg nn_bt_flt=2); the older
    nn_bt_flt=1 (width nn_e, ``<0.5``) is not used by any shipped card.
    The secondary (transport) weights are the SM2005/ts_wgt tail sums
    ``w_transport[j] = Σ_{i≥j} w_i / n_substeps`` — the unique choice
    that keeps ``div(Hu_avg) == (η_old − η_avg)/dt`` (uniform-tracer
    preservation), exactly as the other filters in this module.

    NOTE the MLF-frame remainder (ladder step 4): NEMO starts the
    barotropic from the BEFORE state (t−Δt) so its window is centred at
    t+Δt of a 2Δt integration.  On the forward core the start is t; the
    window centring and width above are the faithful forward-frame
    reduction, and the before-state start arrives with the MLF
    integrator.

    Returns
    -------
    (w_avg, w_total, w_transport, n_loop)
        Per-substep averaging weights (length ``n_loop``, leading
        entries zero until the window opens), their sum (== 1), the
        transport weights, and the number of substeps to run.
    """
    import numpy as _np
    if n_substeps < 2:
        raise ValueError(
            f"nemo_boxcar_centred needs n_substeps >= 2, got {n_substeps!r}")
    if substep_scale < 1 or n_substeps % substep_scale != 0:
        raise ValueError(
            f"substep_scale={substep_scale!r} must be >=1 and divide "
            f"n_substeps={n_substeps!r} (n_substeps = nn_e * substep_scale).")
    half_width = n_substeps // substep_scale               # NEMO nn_e
    jn = _np.arange(1, 3 * n_substeps + 1, dtype=_np.float64)
    # nn_bt_flt=2: boxcar HALF-width == nn_e (full width 2*nn_e), the DINO
    # namelist value (ts_wgt CASE(2): |jn-jic|/nn_e < 1).  The centre
    # jic == n_substeps (== 2*nn_e under the MLF scale); the window spans
    # jn in (n_substeps - nn_e, n_substeps + nn_e), i.e. tau in (0, 2)
    # baroclinic steps of length nn_e*dt_s centred at t+dt.
    w = (_np.abs(jn - n_substeps) / half_width < 1.0).astype(_np.float64)
    m_star = int(_np.max(_np.where(w > 0.0)[0]) + 1)   # last in-window substep
    w = w[:m_star]
    w = w / w.sum()
    w_transport = _np.array(
        [w[i:].sum() for i in range(m_star)], dtype=_np.float64
    ) / n_substeps
    return (
        jnp.asarray(w, dtype=dtype),
        jnp.asarray(w.sum(), dtype=dtype),
        jnp.asarray(w_transport, dtype=dtype),
        m_star,
    )


def nemo_auto_substeps(
    dt: float,
    H_max_wet: float,
    inv_e1_sq_plus_inv_e2_sq_max: float,
    g: float,
    cmax: float = 0.8,
) -> int:
    """NEMO ln_bt_auto substep count (dynspg_ts.F90:1223-1240).

    ``zcu = sqrt(g·H·(1/e1² + 1/e2²))`` per wet T-cell; ``nn_e =
    CEILING(dt / cmax · max(zcu))``.  The caller supplies the maximum of
    ``g``-free metric factor and the deepest wet column so the helper
    stays grid-agnostic:  pass ``max over wet cells of
    (1/e1² + 1/e2²)`` evaluated AT the same cells used for ``H``
    (conservative: pass the global maxima of each — an upper bound that
    can only increase the substep count).
    """
    import math as _math
    zcmax = _math.sqrt(g * max(H_max_wet, 0.0)
                       * inv_e1_sq_plus_inv_e2_sq_max)
    n = int(_math.ceil(dt / cmax * zcmax))
    if n < 2:
        raise ValueError(
            f"nemo_auto_substeps computed n={n!r} (dt={dt}, cmax={cmax}) — "
            "check the metric/H inputs.")
    return n


def compute_filter_weights(
    n_substeps: int,
    dtype: jnp.dtype,
    *,
    use_cosine: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    """Return per-substep accumulator weights for time-averaging.

    The cosine bell (Hanning window) suppresses the side lobes of the
    plain box filter that alias barotropic modes into the baroclinic
    coupling.  Both the lat-lon C-grid and MPAS solvers compute the
    weights with the same formula:

        w_i = 1 + cos(π · (i+1 - n) / n)          (cosine)
        w_i = 1                                   (box)

    over ``i+1 = 1 .. 2n-1``, i.e. a window CENTRED ON ``t + dt`` rather
    than on the middle of the step.  A window centred on ``t + dt/2``
    returns a mid-step free surface as the end-of-step state, which
    propagates gravity waves at roughly HALF their correct speed (measured
    2026-08-12: T_model/T_exact 1.86 cosine, 1.92 box, and 1.00 for both
    ``nemo_ab3am4`` and ``implicit_cn``, neither of which averages).  Costs
    ``2n-1`` substeps instead of ``n``; NEMO's centred boxcar pays the same
    factor.

    Transport (``Hu``) weights — continuity-consistent (SM2005)
    ----------------------------------------------------------------
    The time-averaged SSH is ``eta_avg = (1/w_total)·Σ_i w_i·eta_{i+1}``
    with the substep continuity ``eta_{i+1} = eta_i − dt_s·div(flux_i)``.
    Telescoping gives

        eta_old − eta_avg = (dt_s/w_total)·Σ_j div(flux_j)·tail_j,
        tail_j = Σ_{i≥j} w_i.

    The flux-form tracer step requires the depth-integrated transport
    ``Hu_avg`` to satisfy the discrete continuity invariant
    ``div(Hu_avg) == (eta_old − eta_avg)/dt`` (``dt = n·dt_s``) so a
    uniform tracer is preserved.  EXACT ONLY FOR THE SOURCE-FREE
    RECURRENCE: with a free-surface source ``S_j`` (slow eta forcing,
    freshwater, eta diffusion applied after the accumulation) the identity
    picks up the TRANSPORT-WEIGHTED source average,
    ``div(Hu_avg) == (eta_old − eta_avg)/dt + Σ_j w_transport_j·S_j``.
    That reduces to ``+ S`` only for a source held constant across the
    substeps, which eta diffusion — being state-dependent, hence different
    every substep — is NOT (codex 2026-08-12).  Matching the two expressions gives the
    ONLY consistent per-substep transport weight

        w_transport[j] = tail_j / (n_substeps · w_total).

    This is exactly the Shchepetkin & McWilliams (2005) secondary
    (transport) weight used by the ``power_law`` path; with a uniform
    (box) ``w_i=1`` it is ``(n_loop−j)/(n·n_loop)`` with ``n_loop = 2n−1`` — NOT the flat ``1/n`` that the
    earlier code used, which broke continuity for BOTH box and cosine
    (the cosine inconsistency was the worse of the two, ~99 % residual;
    box ~95 %).  Returning it here makes every non-``power_law`` filter
    continuity-consistent through a single owner.

    Parameters
    ----------
    n_substeps : int
        Number of barotropic substeps.
    dtype : jnp.dtype
        Working precision for the weight array.
    use_cosine : bool
        ``config.barotropic_time_filter == "cosine"``.

    Returns
    -------
    w_filter : jax.Array, shape (2*n_substeps - 1,)
        Per-substep averaging weight (eta / velocity) passed as ``xs``
        to ``lax.scan`` (or indexed inside ``fori_loop``).
    w_total : jax.Array, scalar
        ``sum(w_filter)`` — normalises the eta / velocity accumulators.
    w_transport : jax.Array, shape (2*n_substeps - 1,)
        Per-substep TRANSPORT weight (continuity-consistent; the
        accumulator ``Σ_i w_transport_i·flux_i`` IS the time-averaged
        transport ``Hu_avg`` directly — no further ``/n_substeps``
        normalisation).  On the centred window it sums to 1 ANALYTICALLY
        for both filters — ``Σ_j tail_j == Σ_j j·w_j == n·w_total`` by the
        window's symmetry about ``j = n`` — up to floating-point rounding
        (measured 1.0000000000000002 at n=10, cosine, fp64: the reduction
        orders differ).  On the old half window the box sum was
        ``(n+1)/(2n)``.
    n_loop : int
        Number of substeps to run, ``2*n_substeps - 1``.
    """
    # Substep ``i`` (0-based) produces the state at time ``t + (i+1)*dt_s``,
    # so a window centred on ``t + dt`` is centred on ``i+1 == n_substeps``.
    # Running the loop over ``i+1 in [1, 2n-1]`` makes the weights exactly
    # symmetric about that centre, hence n_loop = 2*n_substeps - 1.
    n_loop = 2 * n_substeps - 1
    tau = jnp.arange(1, n_loop + 1, dtype=dtype) - n_substeps   # 0 at the centre
    if use_cosine:
        # Hanning bell of full width 2n centred at tau == 0.  Positive
        # everywhere on this range (the smallest weight, at the two ends,
        # is 1 - cos(pi/n) > 0), so the ``n_substeps < 2`` degeneracy the
        # old half-window formula had -- 1 + cos(-pi) == 0, a division by
        # zero downstream -- cannot arise and needs no special case.
        w_filter = 1.0 + jnp.cos(jnp.pi * tau / n_substeps)
    else:
        w_filter = jnp.ones(n_loop, dtype=dtype)
    w_total = jnp.sum(w_filter)
    # tail_j = sum_{i>=j} w_filter[i]  (reverse cumulative sum).
    tail = jnp.cumsum(w_filter[::-1])[::-1]
    # The denominator stays the PHYSICAL n_substeps: ``dt = n_substeps*dt_s``
    # is the baroclinic step the transport must close continuity over, and it
    # is unaffected by how far past t+dt the averaging window reaches.
    w_transport = tail / (jnp.asarray(n_substeps, dtype=dtype) * w_total)
    return w_filter, w_total, w_transport, n_loop


AFTER_RECONCILE_SCHEMES = ("off", "nemo_mlf_baro_corr")


def validate_after_reconcile(scheme: str) -> str:
    """Dispatch gate for ``BarotropicConfig.barotropic_after_reconcile``.

    Called on the STATIC config value at the top of each outer step, before any
    array work, so a typo stops the run instead of silently selecting ``"off"``
    (CLAUDE.md dispatch hardening: a bare ``else: <default>`` here would run
    different physics on a misspelling).
    """
    if scheme not in AFTER_RECONCILE_SCHEMES:
        raise ValueError(
            f"unknown barotropic_after_reconcile scheme {scheme!r}: must be "
            f"one of {AFTER_RECONCILE_SCHEMES}.")
    return scheme


def after_level_column_mean_reconcile(
    field: jnp.ndarray,
    h_face_ref: jnp.ndarray,
    target_mean: jnp.ndarray,
    face_mask3: jnp.ndarray,
    min_water_col: float,
) -> jnp.ndarray:
    """NEMO ``mlf_baro_corr``'s committed reconciliation, as one kernel.

    Transcribed from ``cfgs/DINO/MY_SRC/stpmlf.F90:754-765`` (the build that
    ran; ``src/OCE`` differs and its line numbers do not apply)::

        zue(ji,jj) = SUM_k e3u(ji,jj,jk,Kaa) * puu(ji,jj,jk,Kaa) * umask(ji,jj,jk)
        puu(ji,jj,jk,Kaa) = ( puu(ji,jj,jk,Kaa)
           &                - zue(ji,jj) * r1_hu(ji,jj,Kaa)
           &                + uu_b(ji,jj,Kaa) ) * umask(ji,jj,jk)

    i.e. replace the column's own thickness-weighted depth mean by
    ``target_mean``.

    WHICH THICKNESS, and why it is the REFERENCE one (this is not obvious from
    the Fortran, and reading it as written gets it wrong).  Those two lines
    LOOK like they weight at the after time level, ``Kaa``.  They do not.  DINO
    builds with ``key_qco`` (``cpp_DINO.fcm``), whose substitutions
    (``WORK/domzgr_substitute.h90:127,137,46,51``) are::

        e3u(i,j,k,t)  ->  e3u_0(i,j,k) * (1 + r3u(i,j,t)*umask(i,j,k))
        r1_hu(i,j,t)  ->  r1_hu_0(i,j) / (1 + r3u(i,j,t))

    On a wet cell ``umask = 1``, so the ``(1 + r3u(Kaa))`` factor is CONSTANT
    over ``k`` within a column and appears once in the sum and once, inverted,
    in the divisor.  It CANCELS EXACTLY::

        zue * r1_hu(Kaa) = SUM_k( e3u_0 * u * umask ) / hu_0

    So NEMO's reconciliation is INDEPENDENT OF THE TIME LEVEL and weights by
    the fixed REFERENCE ladder ``e3u_0 / hu_0``.  This kernel therefore takes
    the reference face thickness, not a live one.  RETRACTED 2026-08-21: an
    earlier revision took the live AFTER-level thickness and its docstring
    called that "the after-level thickness NEMO divides by".  That was a true
    reading of the Fortran text and a false reading of its arithmetic.

    HOW THE CALLER MUST BUILD ``h_face_ref``, and one thing NOT to assume.
    NEMO builds ``e3u_0`` as an ARITHMETIC mean of the adjacent ``e3t_0``
    (``zgr_lib.F90``), not as a minimum -- ``min`` is the MOM6/MITgcm ``hFacW``
    convention.  On a ladder that is horizontally uniform the two coincide, so
    a min-rule face depth is exact there and only there.  DINO is exactly that
    case (``namelist_cfg:70-72`` ``ln_zco_nam=.true.``, ``ln_zps_nam=.false.``,
    i.e. a pure z-coordinate with NO partial steps; legoESM's bridge builds a
    matching full-step coordinate), so on that card this kernel reproduces
    ``SUM_k e3u_0*u*umask / hu_0`` exactly.  A card WITH partial steps would
    need its face thickness built NEMO's way -- averaged on the unmasked
    reference ladder and then masked -- before this kernel is faithful there.

    Parameters
    ----------
    field
        3-D face velocity at the after level, ``(..., nlev)``.
    h_face_ref
        REFERENCE face-cell thicknesses (NEMO ``e3u_0``), i.e. the ladder with
        no free-surface scaling applied.  The caller names the ladder; this
        function cannot check it.
    target_mean
        Depth-uniform mean to install, broadcastable against ``field`` with a
        trailing singleton level axis (NEMO ``uu_b(:,:,Kaa)``).
    face_mask3
        3-D wet-face mask.  NEMO's ``umask`` factor, applied to the velocity
        INSIDE the sum (as NEMO does), to the thicknesses, and to the result.
    min_water_col
        Divide guard on the summed column depth.  This is the LAND guard (a wet
        column always exceeds it), not a physics clip: NEMO's own divisor adds
        ``1 - ssumask`` for exactly this reason.

    Returns
    -------
    jnp.ndarray
        ``field`` with its reference-thickness column mean replaced.  Applying
        it twice agrees with applying it once to roundoff, and it reduces to
        the identity, to roundoff, when ``target_mean`` already equals the
        column's own mean.

    Sign/geometry convention: ``h_face_ref > 0``, thicknesses sum downward, and
    no term changes sign with the z-axis direction -- this is a weighted-mean
    replacement, not a flux.
    """
    wet = face_mask3 > 0
    h = jnp.where(wet, h_face_ref, 0.0)
    # mask the VELOCITY inside the sum too, as NEMO's ``* umask(ji,jj,jk)``
    # does: ``0 * NaN`` is NaN, so an unmasked land value would poison the
    # whole column mean rather than being ignored.
    f = jnp.where(wet, field, 0.0)
    depth = jnp.maximum(jnp.sum(h, axis=-1, keepdims=True), min_water_col)
    own_mean = jnp.sum(h * f, axis=-1, keepdims=True) / depth
    return (field - own_mean + target_mean) * face_mask3


def bebt_blend(
    eta_new: jnp.ndarray,
    eta_old: jnp.ndarray,
    bebt: float | jnp.ndarray,
) -> jnp.ndarray:
    """Backward-Euler/Backward-time blend of new and old eta for the PGF.

    ``bebt = 0`` recovers the standard forward-backward scheme; the
    MOM6 default ``bebt = 0.2`` introduces semi-implicit damping of
    the fastest barotropic gravity waves (#205).
    """
    return (1.0 - bebt) * eta_new + bebt * eta_old


def maxvel_clip(field: jnp.ndarray, maxvel: float | jnp.ndarray) -> jnp.ndarray:
    """Symmetric clip of barotropic velocity components.

    Used to suppress runaway velocities at single grid points that
    would otherwise crash the solver before the substep finishes.
    """
    return jnp.clip(field, -maxvel, maxvel)


def coriolis_at_faces(grid, dtype) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Coriolis parameter at C-grid u-/v-faces ``(f_u, f_v)``.

    Single source of truth for the semi-implicit Coriolis face values that
    the lat-lon C-grid barotropic solvers (explicit + implicit) and the
    full PE step each reconstructed with a byte-identical inline block
    (#517).  Behaviour, in preference order:

    1. **Stored metrics (the shipping path).** If the geometry carries
       pre-computed ``grid.f_u`` (shape ``(n_lat, n_lon+1)``) and
       ``grid.f_v`` (``(n_lat+1, n_lon)``) — every ``LatLonCGridGeometry``,
       including tripolar — return those cast to ``dtype``.  Bit-identical
       to the previous inline ``hasattr(grid, "f_u")`` branch.
    2. **Reconstruct from cell-centre ``grid.f``** (lean ``LatLonGrid``,
       which lacks face metrics): average adjacent cells onto the faces.
       Bit-identical to the previous inline ``else`` branch.

    Fold safety (the latent bug this dedup closes): the reconstruction in
    (2) is NOT tripolar-fold-aware — it averages cell-centre ``f`` without
    the fold's ``vector_sign_v`` flip on the north v-row, so on a folded
    grid it would yield wrong vorticity at the seam.  Real folded grids
    always take path (1) (they store ``f_u/f_v``).  Should a folded grid
    ever reach (2) without stored face metrics, RAISE rather than silently
    mis-reconstruct (dispatch-hardening: a latent silent-wrong-answer
    becomes a loud error; no shipping path changes).

    Operator-package-free: the fold check reads ``grid.fold.is_active``
    directly (mirrors ``operators_latlon_cgrid.is_tripolar``) so this
    module keeps its no-operator-import contract.
    """
    if hasattr(grid, "f_u") and hasattr(grid, "f_v"):
        return grid.f_u.astype(dtype), grid.f_v.astype(dtype)

    fold = getattr(grid, "fold", None)
    if fold is not None and bool(getattr(fold, "is_active", False)):
        raise ValueError(
            "coriolis_at_faces: tripolar/folded grid is missing stored "
            "f_u/f_v. The reconstruct-from-grid.f fallback is not "
            "fold-aware (no vector_sign_v flip on the north v-row) and "
            "would produce wrong vorticity at the fold seam. Populate "
            "grid.f_u/grid.f_v (LatLonCGridGeometry does this) instead of "
            "passing a bare fold-less grid."
        )

    f_cell = grid.f.astype(dtype)
    f_u = 0.5 * (jnp.roll(f_cell, 1, axis=1) + f_cell)
    f_u = jnp.concatenate([f_u, f_u[:, 0:1]], axis=1)
    f_v_interior = 0.5 * (f_cell[:-1] + f_cell[1:])
    f_v = jnp.concatenate([f_cell[0:1], f_v_interior, f_cell[-1:]], axis=0)
    return f_u, f_v


# ---------------------------------------------------------------------
# Distributed implicit Helmholtz solve (shared lat-lon C-grid + MPAS)
# ---------------------------------------------------------------------
#
# The implicit Crank-Nicolson free-surface solvers form the SAME elliptic
# problem on every grid:
#
#     A eta = rhs,   A = I - coeff * div(H * grad)
#
# ``A`` is symmetric positive-definite in the area-weighted cell inner
# product (the FV gradient and divergence are discrete adjoints), so a
# Jacobi-preconditioned conjugate-gradient solve converges quickly.
#
# Single rank uses ``jax.scipy.sparse.linalg.cg`` (residual-terminated
# while_loop, the historical production path — numerics preserved).
#
# Under MPI the stock CG deadlocks: its dot products are rank-local and
# its termination is residual-dependent, so different ranks run different
# iteration counts and desynchronise the halo-exchange / allreduce
# collective schedule.  The distributed path here replaces it with a
# HAND-ROLLED, fixed-iteration PCG (a static-length ``lax.fori_loop`` of
# exactly ``max_iter`` iterations).  Standard preconditioned CG needs two
# sequentially-dependent inner products per iteration — ``p·Ap`` (for α)
# and ``r·z`` (for β, which needs the updated ``r`` that depends on α) —
# so the loop issues TWO batched ``allreduce(SUM)`` per iteration.  The
# residual-monitor ``r·r`` is folded into the SECOND reduction so the
# diagnostic costs no extra message.  Two reductions/iter is still a
# FIXED count independent of global resolution (the weak-scaling
# property) — vastly fewer than ``explicit_substep``'s O(n_substeps ∝
# resolution) reductions.  A one-reduction pipelined CG
# (Chronopoulos-Gear) is possible but changes the rounding and would
# break the tight single-rank-vs-stock-CG equivalence pin, so it is not
# used.  Every rank runs the identical collective schedule => no
# deadlock, JIT-static trace.
#
# Differentiability — why NOT ``jax.lax.custom_linear_solve``:
# the original design wrapped the solve in ``custom_linear_solve`` (for
# the implicit-function adjoint, avoiding storing the M primal
# iterations).  That is INCOMPATIBLE with the MPI halo: under MPI
# ``A_op`` contains the halo ``sendrecv`` via ``_sendrecv_vjp`` (a
# ``jax.custom_vjp``), and ``custom_linear_solve`` forms the
# operator-parameter cotangent by LINEAR-TRANSPOSING ``matvec`` — JAX
# cannot transpose a ``custom_vjp_call`` ("Transpose rule for
# 'custom_vjp_call' not implemented"), so it fails at trace time even
# in a forward-only call (verified: np=2/4/8 crash, np=1 fine).
#
# Instead the distributed path is UNROLLED and differentiated directly:
# the fixed-M ``fori_loop`` (-> ``scan``, reverse-mode differentiable)
# over ``A_op`` (halo ``_sendrecv_vjp`` — full VJP) and
# ``_global_dot_batch`` (``allreduce(SUM)`` — full VJP) uses ONLY the
# two collectives the repo's MPI-AD doctrine blesses (CLAUDE.md), so
# ``jax.grad`` works under MPI.  Its gradient is the derivative of "M
# PCG iterations", which equals the implicit-solve gradient to within
# the converged residual (M chosen so residual <= tol).  The SINGLE-rank
# path keeps stock ``jax.scipy.sparse.linalg.cg``, whose OWN internal
# ``custom_linear_solve`` gives the exact implicit-function adjoint (no
# MPI halo there, so no transpose problem).  Net: single-rank = exact
# implicit adjoint; multi-rank = unrolled fixed-M adjoint (correct to
# convergence), both on a static collective schedule.


class HelmholtzSolveDiagnostics(NamedTuple):
    """Diagnostics returned alongside the implicit Helmholtz solution.

    ``rel_residual`` is the GLOBAL relative residual
    ``sqrt(global(r·r) / global(rhs·rhs))`` after the solve.  For the
    fixed-iteration distributed PCG it is the only convergence signal;
    callers log it (or raise outside JIT) but MUST NOT branch the
    compiled step on it — that would reintroduce data-dependent control
    flow and desynchronise collectives.  For the single-rank stock-CG
    path it is computed once at the end for a uniform return signature.

    ``converged`` mirrors stock CG's ``info == 0`` convention loosely:
    ``True`` when ``rel_residual <= residual_tol``.  It is a host-side
    diagnostic only.
    """

    rel_residual: jnp.ndarray
    converged: jnp.ndarray


def _global_dot_batch(
    pairs: list[tuple[jnp.ndarray, jnp.ndarray]],
) -> list[jnp.ndarray]:
    """Batched global inner products ``sum(a*b)`` over a flat field.

    Each ``(a, b)`` pair is reduced to a scalar ``sum(a*b)``.  The cross-
    rank reduction fires ONLY when actually multi-process
    (``is_multi_process()`` — same runtime gate as
    ``eta_floor.clamp_and_redistribute``): under a single process the
    local sum already IS the global sum, so the fixed-iteration PCG
    *algorithm* is exercised identically with or without MPI (this is
    what makes the single-rank-vs-stock-CG equivalence test runnable in
    one process).  When multi-process, all scalars are folded into ONE
    ``batch_allreduce_mpi(op="sum")`` message (the dominant barotropic-
    loop cost is MPI latency, so a single message per iteration matters).
    ``allreduce(SUM)`` is the only AD-safe collective (CLAUDE.md) — its
    reverse-mode VJP is itself an ``allreduce(SUM)`` — so unrolling the
    PCG and differentiating straight through these reductions is AD-safe.
    """
    local = [jnp.sum(a * b) for (a, b) in pairs]
    # SPMD (single-controller shard_map, route-B multi-GPU — no mpi4jax)
    # path: each ``local`` sum is a PARTIAL sum over this device's shard
    # (one latitude band) and must be summed across the mesh shard axis
    # with ``jax.lax.psum``.  Checked FIRST because ``is_multi_process()``
    # is FALSE under one process — otherwise the partial sum would be
    # silently returned as the "global" dot and every band would converge
    # to its own sub-system (the MPAS analogue of this bug was job
    # 8460616).  Backend is ``"spmd"`` ONLY when armed by
    # ``activate_latlon_spmd_halo`` (cube SPMD does not call this), so this
    # branch is inert for the serial and MPI paths.  ``psum`` is
    # self-transposing => AD-safe, same as ``allreduce(SUM)``.
    from legoesm.grids.halo import get_halo_backend, get_spmd_mesh
    if get_halo_backend() == "spmd":
        mesh = get_spmd_mesh()
        if mesh is None:
            # backend armed "spmd" but no mesh set: an invalid state
            # reachable only via the public set_halo_backend("spmd")
            # without a matching set_spmd_mesh.  FAIL FAST rather than
            # silently return unreduced partial sums inside a sharded
            # solve (codex LOW) — the supported activators
            # (activate_latlon_spmd_halo / the cube equivalent) always set
            # the mesh together with the backend.
            raise RuntimeError(
                "_global_dot_batch: halo backend is 'spmd' but no SPMD mesh "
                "is set; arm it via activate_latlon_spmd_halo(mesh).")
        # Route to psum ONLY for the lat-band ocean SPMD mesh, keyed on the
        # ``"lat"`` axis BY NAME (activate_latlon_spmd_halo guarantees it).
        # The cube atm SPMD backend ALSO sets backend=="spmd" but with a
        # ``("face", ...)`` mesh; in a coupled run that mesh could be armed
        # while this ocean barotropic PCG runs, and psum'ing over a
        # non-lat (or replicated) axis would multiply the dots by the
        # device count or crash (codex HIGH).  When the armed SPMD mesh is
        # not the lat-band one, fall through to the MPI/local logic below
        # (ocean fields are never cube-sharded, so the local/allreduce sum
        # is the correct reduction there).
        if "lat" in tuple(mesh.axis_names):
            from legoesm.parallel.reductions import batch_psum_spmd
            return batch_psum_spmd(local, "lat")
    # Function-scope import: ``reductions`` pulls in mpi4jax lazily and
    # ``core.operators`` (cross-package), so keep it out of module top.
    from legoesm.parallel.reductions import (
        batch_allreduce_mpi,
        is_multi_process,
    )
    if not is_multi_process():
        return local
    return batch_allreduce_mpi(local, op="sum")


def _fixed_iteration_pcg(
    A_op: Callable[[jnp.ndarray], jnp.ndarray],
    b: jnp.ndarray,
    M_inv: Callable[[jnp.ndarray], jnp.ndarray],
    x0: jnp.ndarray,
    *,
    max_iter: int,
    dot_weight: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Jacobi-preconditioned CG run for EXACTLY ``max_iter`` iterations.

    Standard preconditioned conjugate gradient (Shewchuk 1994, Alg. B3)
    with a STATIC iteration count: no residual-dependent ``while_loop``,
    so all MPI ranks execute the identical collective schedule.

    Reductions per iteration (FIXED, resolution-independent):
      * ``p·Ap`` (for α) — one batched reduction; then
      * ``r·z`` AND ``r·r`` (for β and the residual monitor) — folded
        into ONE batched reduction, since both need the post-α ``r``.
    => two batched ``allreduce(SUM)`` per iteration (the residual costs
    no extra message).  Under a single process :func:`_global_dot_batch`
    is a no-op reduction, so the *algorithm* is identical with/without
    MPI (this is what makes the single-rank equivalence test runnable in
    one process).

    Grid-agnostic: ``A_op`` and ``M_inv`` are opaque callables operating
    on whatever array shape the caller uses ((n_lat, n_lon) for lat-lon,
    (nCells,) for MPAS).  ``A_op`` owns the halo exchange; ``M_inv`` is
    communication-free (diagonal Jacobi).

    Safety floors (``1e-30``) guard the CG scalar divisions on the very
    first land-only or zero-rhs columns; they are math safeguards, not
    tunable parameters (CLAUDE.md "safety floors exempt").

    Returns
    -------
    (x, rr) : the solution after ``max_iter`` iterations and the final
        GLOBAL ``r·r`` (squared residual norm).  ``r·r`` is folded into
        the per-iteration ``r·z`` reduction (no extra message), so it is
        a free self-monitoring signal — ``solve_helmholtz_implicit``
        consumes this ``rr`` directly for the relative-residual
        diagnostic (no second ``A_op``).
    """
    # ``dot_weight`` (optional): weighted/owned-masked dots for every CG
    # scalar — REQUIRED on partitioned unstructured meshes whose local
    # arrays carry HALO entries (an unweighted local sum double-counts
    # them in the allreduce; pass owned_mask·area).  ``None`` keeps the
    # historical Euclidean dots bit-exactly (the lat-lon band path,
    # whose rows partition without overlap).
    if dot_weight is None:
        def _dotw(a):
            return a
    else:
        def _dotw(a):
            return a * dot_weight

    r0 = b - A_op(x0)
    z0 = M_inv(r0)
    # Initial r·z and r·r (one batched reduction).
    rz0, rr0 = _global_dot_batch([(_dotw(r0), z0), (_dotw(r0), r0)])

    # FREEZE threshold for the CG scalar divisions, RELATIVE to the
    # initial r·z (all of rz/pAp live in the same units as rz0).  Once
    # the residual hits the working-precision floor the unrolled CG keeps
    # running its remaining static iterations on a ~converged state;
    # there ``rz``/``pAp`` are at round-off noise and ``rz_new/rz`` is a
    # noisy ``0/0`` whose REVERSE-MODE value blows up to NaN over many
    # iterations.  ``rz`` is a residual-SQUARED quantity, so its noise
    # floor is ~``rz0 * eps^2`` (the residual relative floor is ~eps).
    # Freezing at ``rz0 * eps^2`` therefore engages exactly at the
    # round-off plateau (f32: ~rz0*1e-14, matching the observed noise;
    # f64: ~rz0*5e-32, far below the 1e-10 convergence target so f64
    # accuracy is unaffected) and zeroes the already-converged update
    # with a finite gradient.  (A fixed ``1e-30`` floor never engages in
    # f32; a ``rz0*eps`` floor freezes f64 prematurely.)
    #
    # ACCEPTED NONDIFFERENTIABLE POINT: if ``x0`` ALREADY solves
    # ``A x0 = b`` exactly (``r0 = 0`` => ``rz0 = 0`` => ``den_floor =
    # tiny``), every ``_safe_div`` freezes and the loop returns ``x0`` with
    # ZERO gradient sensitivity through the solve — whereas the true
    # implicit derivative at ``rhs = A x0`` is nonzero.  For the lat-lon
    # caller ``rhs - A·eta_old = dt·(F_slow_eta·mask - div_HU_pred)``, so
    # ``r0 = 0`` is reachable at an EXACT rest state with no slow forcing
    # and zero predicted transport divergence (or exact cancellation) —
    # rare but not impossible.  At such a point the solve is already at
    # its answer and the zeroed gradient is a measure-zero degeneracy, not
    # a bias on a generic trajectory; a forward run is unaffected (the
    # output ``x0`` is correct).  Not worth a special-cased zero-residual
    # adjoint; documented so a caller differentiating through a perfectly
    # quiescent step knows the gradient there is degenerate.
    finfo = jnp.finfo(b.dtype)
    rz0_mag = jnp.abs(rz0)
    den_floor = jnp.maximum(rz0_mag * finfo.eps * finfo.eps, finfo.tiny)

    def _safe_div(num, den):
        # GRADIENT-safe guarded division: num/den when |den|>den_floor,
        # else 0.  The "double where" — dividing by
        # ``where(cond, den, 1)`` (NOT the floor) — keeps BOTH the forward
        # AND the reverse-mode value finite (dividing by the floor would
        # make the dead branch's gradient ``-num/floor^2`` overflow).
        cond = jnp.abs(den) > den_floor
        safe_den = jnp.where(cond, den, jnp.ones_like(den))
        return jnp.where(cond, num / safe_den, jnp.zeros_like(num))

    class _CGState(NamedTuple):
        x: jnp.ndarray
        r: jnp.ndarray
        p: jnp.ndarray
        rz: jnp.ndarray
        rr: jnp.ndarray

    def body(_i: int, st: _CGState) -> _CGState:
        Ap = A_op(st.p)
        # Reduction 1/iter: p·Ap (needed for α).
        (pAp,) = _global_dot_batch([(_dotw(st.p), Ap)])
        alpha = _safe_div(st.rz, pAp)
        x_new = st.x + alpha * st.p
        r_new = st.r - alpha * Ap
        z_new = M_inv(r_new)
        # Reduction 2/iter: r·z (for β) AND r·r (residual monitor),
        # batched into one message.
        rz_new, rr_new = _global_dot_batch(
            [(_dotw(r_new), z_new), (_dotw(r_new), r_new)],
        )
        beta = _safe_div(rz_new, st.rz)
        p_new = z_new + beta * st.p
        return _CGState(x=x_new, r=r_new, p=p_new, rz=rz_new, rr=rr_new)

    init = _CGState(x=x0, r=r0, p=z0, rz=rz0, rr=rr0)
    final = jax.lax.fori_loop(0, int(max_iter), body, init)
    return final.x, final.rr


def _fixed_iteration_pcg_single_reduce(
    A_op: Callable[[jnp.ndarray], jnp.ndarray],
    b: jnp.ndarray,
    M_inv: Callable[[jnp.ndarray], jnp.ndarray],
    x0: jnp.ndarray,
    *,
    max_iter: int,
    dot_weight: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Single-reduction fixed-M PCG (Chronopoulos & Gear 1989 recurrences).

    Mathematically equivalent (in exact arithmetic) to
    :func:`_fixed_iteration_pcg`, but restructured so each iteration
    issues ONE batched ``allreduce`` (3 scalars: ``r·z``, ``w·z``,
    ``r·r``) instead of two sequentially-dependent reductions — the
    multi-node weak-scaling lever (probe 8460255: np32 weak growth
    ×7.9 is allreduce-latency-driven; this halves the per-step
    reduction count from ``2M+1`` to ``M+1``).  Costs one extra carried
    vector (``s = A p``) and one extra axpy per iteration.

    Recurrences (preconditioned CG-CG form; ``z = M⁻¹ r``, ``w = A z``):

        β_j = ρ_j / ρ_{j-1}              ρ_j = r_j·z_j
        t_j = μ_j − β_j² · t_{j-1}       μ_j = w_j·z_j   (t_j ≡ p_j·A p_j)
        α_j = ρ_j / t_j

    (β SQUARED: ``p_j·Ap_j = z_j·w_j + 2β (z_j·A p_{j-1}) + β² t_{j-1}``
    and ``z_j·A p_{j-1} = −ρ_j/α_{j-1} = −β t_{j-1} · ρ_j/ρ_{j-1}·...``
    collapses the cross term to ``−2β² t_{j-1}`` — a β¹ recurrence
    diverges, caught by the equivalence gate in job 8460547.)

    INNER PRODUCT (codex 2026-06-11 CRITICAL): every CG scalar here is
    the AREA-WEIGHTED dot ``⟨a,b⟩_W = Σ a·W·b`` with ``W =
    dot_weight`` (masked cell area).  The recurrence's cross-term
    identity ``⟨Az, p⟩ = ⟨z, Ap⟩`` requires A self-adjoint in the dot
    being used — the FV Helmholtz carries a ``1/area`` divergence
    factor and is self-adjoint ONLY in the W-inner product (``Aᵀ = W A
    W⁻¹``, pinned by the unit suite); the diagonal Jacobi ``M⁻¹`` is
    self-adjoint in W too.  Euclidean dots (the standard body's
    choice) are safe THERE because standard PCG measures ``p·Ap``
    directly each iteration instead of reconstructing it; here a
    Euclidean reconstruction silently biases α on any varying-area
    grid.  ``dot_weight`` is therefore REQUIRED — the dispatch refuses
    ``single_reduce`` without it rather than falling back to a wrong
    inner product.
        p_j = z_j + β_j p_{j-1}
        s_j = w_j + β_j s_{j-1}          (≡ A p_j — no second matvec)
        x_{j+1} = x_j + α_j p_j
        r_{j+1} = r_j − α_j s_j

    Carrying ``t = p·Ap`` directly avoids dividing by a possibly-frozen
    ``α`` and inherits the SAME ``_safe_div`` round-off-plateau freeze
    (and its documented zero-residual gradient degeneracy) as the
    standard body.  In floating point the iterates differ from standard
    PCG at round-off order; the solver-tolerance gates (not bit-exact
    ones) apply — see ``PARITY_TOLS_PCG_MPI``'s rationale.

    Returns ``(x, rr)`` with ``rr`` = final global ``r·r``, same
    contract as :func:`_fixed_iteration_pcg`.
    """
    W = dot_weight
    r0 = b - A_op(x0)
    z0 = M_inv(r0)
    w0 = A_op(z0)
    # ONE batched init reduction: ρ0, μ0 in the W-inner product, plus
    # the residual monitor — ALSO W-weighted: on halo-carrying
    # partitioned meshes (MPAS) an unweighted local r·r double-counts
    # halo entries in the allreduce; on lat-lon this makes the
    # diagnostic the area-weighted norm (documented, conservative).
    rho0, mu0, rr0 = _global_dot_batch(
        [(r0 * W, z0), (w0 * W, z0), (r0 * W, r0)],
    )

    finfo = jnp.finfo(b.dtype)
    rho0_mag = jnp.abs(rho0)
    den_floor = jnp.maximum(rho0_mag * finfo.eps * finfo.eps, finfo.tiny)

    def _safe_div(num, den):
        cond = jnp.abs(den) > den_floor
        safe_den = jnp.where(cond, den, jnp.ones_like(den))
        return jnp.where(cond, num / safe_den, jnp.zeros_like(num))

    class _CGSRState(NamedTuple):
        x: jnp.ndarray
        r: jnp.ndarray
        p: jnp.ndarray
        s: jnp.ndarray      # A p, maintained by recurrence
        rho: jnp.ndarray    # r·z
        t: jnp.ndarray      # p·A p, maintained by recurrence
        alpha: jnp.ndarray
        rr: jnp.ndarray

    def body(_i: int, st: _CGSRState) -> _CGSRState:
        # Apply the PREVIOUS iteration's α (deferred so the new scalars
        # of this iteration come from one reduction below).
        x_new = st.x + st.alpha * st.p
        r_new = st.r - st.alpha * st.s
        z_new = M_inv(r_new)
        w_new = A_op(z_new)
        # The single batched reduction of the iteration (W-dots for
        # the CG scalars AND the r·r monitor — see the init comment).
        rho_new, mu_new, rr_new = _global_dot_batch(
            [(r_new * W, z_new), (w_new * W, z_new), (r_new * W, r_new)],
        )
        beta = _safe_div(rho_new, st.rho)
        t_new = mu_new - beta * beta * st.t
        alpha_new = _safe_div(rho_new, t_new)
        p_new = z_new + beta * st.p
        s_new = w_new + beta * st.s
        return _CGSRState(
            x=x_new, r=r_new, p=p_new, s=s_new,
            rho=rho_new, t=t_new, alpha=alpha_new, rr=rr_new,
        )

    t0 = mu0
    alpha0 = _safe_div(rho0, t0)
    init = _CGSRState(
        x=x0, r=r0, p=z0, s=w0, rho=rho0, t=t0, alpha=alpha0, rr=rr0,
    )
    # Each body call applies one α-update then prepares the next α —
    # ``max_iter`` calls ⇒ exactly ``max_iter`` x/r updates and
    # ``max_iter + 1`` reductions total (incl. init), vs ``2·max_iter
    # + 1`` for the standard body.  The final iteration's prepared
    # (p, s, α) are discarded — its reduction still ran, keeping the
    # collective schedule static.
    final = jax.lax.fori_loop(0, int(max_iter), body, init)
    return final.x, final.rr


def global_rel_residual(
    A_op: Callable[[jnp.ndarray], jnp.ndarray],
    x: jnp.ndarray,
    b: jnp.ndarray,
) -> jnp.ndarray:
    """Global relative residual ``sqrt(global(r·r)/global(b·b))``.

    Returned as a DIAGNOSTIC (never loop control).  The two squared
    norms are reduced in a single batched ``allreduce`` under MPI.
    Wrapped in ``stop_gradient``: the diagnostic must never pull the
    (non-custom-VJP) batched allreduce into reverse mode if a caller
    differentiates a scalar that happens to include it.
    """
    r = b - A_op(x)
    rr, bb = _global_dot_batch([(r, r), (b, b)])
    eps = jnp.asarray(1.0e-30, dtype=b.dtype)
    return jax.lax.stop_gradient(jnp.sqrt(rr / jnp.maximum(bb, eps)))


# Relative-residual floor for the PCG/CG, expressed in machine epsilons of the
# WORKING dtype.  A fixed-iteration (or stock) PCG cannot drive the relative
# residual sqrt(r·r/b·b) below the rounding-noise floor ~ sqrt(N)·eps; for the
# diagonally-dominant free-surface Helmholtz this bottoms out a couple of
# orders above eps.  1e3·eps is a safe practical floor (f64: ~2.2e-13, well
# below the 1e-10 default, so f64 is unchanged; f32: ~1.2e-4, which the solver
# CAN reach — a hardcoded 1e-10 is ~3 orders below f32 eps ≈ 1.19e-7 and would
# be permanently unreachable, leaving ``converged`` always False and the
# single-rank stock-CG ``while_loop`` grinding to ``maxiter`` every step).
_PCG_REL_TOL_EPS_FLOOR = 1.0e3


def precision_aware_rel_tol(
    requested_tol: float | jnp.ndarray, dtype: jnp.dtype,
) -> jnp.ndarray:
    """Floor a relative-residual tolerance to what *dtype* can actually reach.

    Returns a scalar of *dtype*.

    * **float64 (and any wider) → pure pass-through.**  The full-precision
      reference path is byte-identical: ANY requested f64 tolerance (the
      1e-10 default, or a tighter custom 1e-13, …) is returned unchanged.
    * **float32 (and narrower) → floored** to
      ``max(requested_tol, _PCG_REL_TOL_EPS_FLOOR · eps(dtype))`` (~1.2e-4 in
      f32).  This raises an unreachable f64-tuned tolerance (e.g. 1e-10,
      ~1000× below f32 machine epsilon ≈ 1.19e-7) up to a value the iterative
      solver can satisfy — so the ``converged`` diagnostic stays meaningful
      and a residual-gated stock-CG ``while_loop`` terminates instead of
      running to its iteration cap.  A tolerance already above the floor
      (a deliberately loose request) passes through unchanged.

    Used by the implicit free-surface solvers (lat-lon C-grid + MPAS) for both
    the ``converged``-flag acceptance tolerance and the stock-CG ``tol``; the
    distributed fixed-iteration PCG runs a static iteration count regardless,
    so this only affects the diagnostic there.

    JAX-safe: ``requested_tol`` may be a Python float OR a traced scalar (no
    host sync, no ``float()`` on a tracer); the floor is a static value of
    *dtype* (``jnp.finfo`` reads the static dtype, not a tracer).
    """
    dt = jnp.dtype(dtype)
    requested = jnp.asarray(requested_tol, dtype=dt)
    # Pass f64 (and any dtype at least as wide) straight through so the
    # reference path is byte-identical for ANY f64 tolerance, not just the
    # default — the eps floor only matters for the narrow (fp32) modes.
    if jnp.finfo(dt).eps <= jnp.finfo(jnp.float64).eps:
        return requested
    floor = jnp.asarray(_PCG_REL_TOL_EPS_FLOOR * jnp.finfo(dt).eps, dtype=dt)
    return jnp.maximum(requested, floor)


def solve_helmholtz_implicit(
    A_op: Callable[[jnp.ndarray], jnp.ndarray],
    rhs: jnp.ndarray,
    M_inv: Callable[[jnp.ndarray], jnp.ndarray],
    x0: jnp.ndarray,
    *,
    distributed: bool,
    fixed_iters: int,
    residual_tol: float | jnp.ndarray,
    stock_cg_tol: float | jnp.ndarray,
    stock_cg_maxiter: int,
    pcg_variant: str = "standard",
    dot_weight: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, HelmholtzSolveDiagnostics]:
    """Solve ``A eta = rhs`` for the implicit free-surface step.

    Shared entry point for the implicit Crank-Nicolson barotropic solve
    (lat-lon C-grid; MPAS uses stock CG directly — see
    ``barotropic_implicit_mpas.py``).  No duplicated solver numerics
    (``tests/ocean/unit/test_no_scheme_duplication.py``).

    Dispatch (static Python branch on ``distributed`` — NOT a traced
    ``jnp.where``, so only one solver is traced/compiled):

    * ``distributed=False`` → ``jax.scipy.sparse.linalg.cg`` exactly as
      the historical single-rank production path (numerics preserved to
      within stock-CG tolerance; its OWN internal ``custom_linear_solve``
      gives the exact implicit-function adjoint — no MPI halo there).
    * ``distributed=True``  → UNROLLED fixed-iteration distributed PCG
      (static ``fori_loop`` of exactly ``fixed_iters`` iterations).
      Reverse-mode AD differentiates straight through the loop; every op
      inside (``A_op`` halo ``_sendrecv_vjp``, ``_global_dot_batch``
      ``allreduce(SUM)``) has a full VJP, so ``jax.grad`` works under
      MPI.  ``jax.lax.custom_linear_solve`` is NOT used: it would
      linear-transpose ``A_op``, and the MPI halo ``_sendrecv_vjp``
      (a ``custom_vjp``) has no transpose rule — see the module note.

    Parameters
    ----------
    A_op, M_inv : callables
        Helmholtz operator and Jacobi preconditioner.  ``A_op`` owns the
        halo exchange (MPI-correct C-grid / TRiSK operators); ``M_inv``
        is diagonal (no communication).
    rhs, x0 : jax.Array
        Right-hand side and initial guess, same grid shape.  The
        distributed PCG warm-starts from ``x0`` (same as stock CG).
    distributed : bool
        ``is_distributed()`` from the caller (static).
    fixed_iters : int
        ``M`` — distributed PCG runs EXACTLY this many iterations.
    residual_tol : float
        Acceptance tolerance for the returned global relative residual
        (diagnostic only; never loop control).
    stock_cg_tol, stock_cg_maxiter :
        Single-rank stock-CG tolerance / iteration cap.

    Returns
    -------
    (eta_new, diagnostics)
        ``diagnostics`` is a :class:`HelmholtzSolveDiagnostics`; the
        caller logs / raises on ``rel_residual`` OUTSIDE the JIT.
    """
    if not distributed:
        eta_new, _info = jax.scipy.sparse.linalg.cg(
            A_op, rhs, x0=x0, tol=stock_cg_tol,
            maxiter=int(stock_cg_maxiter), M=M_inv,
        )
        rel = global_rel_residual(A_op, eta_new, rhs)
        return eta_new, HelmholtzSolveDiagnostics(
            rel_residual=rel, converged=rel <= residual_tol,
        )

    # Distributed: unrolled fixed-M PCG (differentiable straight through;
    # static collective schedule).  The PCG returns its final global r·r
    # (folded free into the per-iteration r·z reduction), so the relative
    # residual diagnostic costs no extra A_op.
    # ``pcg_variant`` dispatch (validated here on the static Python
    # string — dispatch discipline: unknown ⇒ ValueError, never a
    # silent default):
    #   "standard"      — 2 reductions/iter (Shewchuk B3).
    #   "single_reduce" — 1 batched reduction/iter (Chronopoulos-Gear);
    #                     the multi-node weak-scaling lever.
    if pcg_variant == "standard":
        # ``dot_weight`` optional here (None = historical Euclidean dots,
        # bit-exact for the lat-lon band path); REQUIRED semantics on
        # halo-carrying partitioned meshes — see _fixed_iteration_pcg.
        eta_new, rr = _fixed_iteration_pcg(
            A_op, rhs, M_inv, x0, max_iter=fixed_iters,
            dot_weight=dot_weight,
        )
    elif pcg_variant == "single_reduce":
        if dot_weight is None:
            # LOUD refusal — a Euclidean fallback would silently bias
            # the reconstructed p·Ap on any varying-area grid (the
            # recurrence needs the W-self-adjoint inner product).
            raise ValueError(
                "solve_helmholtz_implicit: pcg_variant='single_reduce' "
                "requires dot_weight (the masked cell area making the "
                "Helmholtz self-adjoint in the weighted inner product)."
            )
        eta_new, rr = _fixed_iteration_pcg_single_reduce(
            A_op, rhs, M_inv, x0, max_iter=fixed_iters,
            dot_weight=dot_weight,
        )
    else:
        raise ValueError(
            "solve_helmholtz_implicit: unknown pcg_variant "
            f"{pcg_variant!r}; expected 'standard' or 'single_reduce'."
        )
    # rhs norm for the relative-residual diagnostic — same weighting as
    # the solver's rr (owned-masked on partitioned meshes; halo entries
    # would double-count in the allreduce otherwise).
    _rhs_w = rhs if dot_weight is None else rhs * dot_weight
    (bb,) = _global_dot_batch([(_rhs_w, rhs)])
    eps = jnp.asarray(1.0e-30, dtype=rhs.dtype)
    rel = jax.lax.stop_gradient(jnp.sqrt(rr / jnp.maximum(bb, eps)))
    return eta_new, HelmholtzSolveDiagnostics(
        rel_residual=rel, converged=rel <= residual_tol,
    )
