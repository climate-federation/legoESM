ADVERSARIAL REVIEW. Answer in prose, <= 80 lines. Do NOT run shell commands
unless a single grep is essential. Do NOT reprint any code.

Context: legoESM, MPAS hydrostatic dycore, sigma coordinate, nlev=30, dt=75 s,
100-year AMIP. A new "tropopause-refined" sigma grid replaces uniform-in-sigma
level placement at FIXED level count. Grid at refine=3 (p_s=985 hPa), layer
thicknesses top->bottom in hPa:
39.8 24.6 17.3 14.9 14.2 14.3 14.9 16.1 17.9 20.2 23.2 26.8 30.7 34.4 37.4
39.4 40.7 41.4 41.8 42.1 42.2 42.3 42.3 42.3 42.3 42.3 42.3 42.3 42.3 42.3
Uniform baseline: 32.5 hPa in every layer. Full-level pressures at refine=3:
29.7 61.9 82.8 98.9 113.5 127.7 142.3 157.8 174.8 193.9 215.6 240.6 ... 963.8

Here is the new grid-construction function:
def tropopause_refined_sigma_half(
    n_levels: int,
    sigma_top: float = 0.01,
    sigma_refine: float = 0.12,
    refine: float = 3.0,
    width: float = 0.45,
) -> np.ndarray:
    """Half-level sigma with layers REDISTRIBUTED toward the tropopause.

    The uniform-in-sigma default (:func:`create_sigma_coordinate`) spaces
    every layer by the same ``dp = dsigma * p_s`` — about 33 hPa at 30
    levels — so the tropical tropopause layer, whose structure is a
    10-20 hPa affair, is spanned by ~3 levels and the model forms no cold
    point (its coldest tropical level lands at the ~26 hPa top instead of
    ~100 hPa; measured 2026-07-25).  Adding levels does NOT fix this: at
    40 levels the TTL still gets 4 levels, while the thinner top layers
    destabilised the model (blow-up at day 46 with dt=60).

    So redistribute at FIXED count instead.  Levels are placed by the
    standard equidistribution principle: they are the quantiles of a
    density ``d(sigma)`` in SIGMA space, so a flat density reproduces the
    uniform default exactly and the refinement only steals layers from the
    (over-resolved) mid-troposphere::

        d(sigma) = 1 + (refine - 1) * exp(-0.5 * ((ln sigma - ln sigma_refine) / width)^2)

    The bump is Gaussian in LOG sigma because atmospheric structure scales
    with log-pressure; ``width`` is therefore in log-sigma units (0.45 ~ a
    factor e^0.45 = 1.6 in pressure either side of the centre).  Working in
    sigma (not log-sigma) for the equidistribution is deliberate: a pure
    log-sigma grid would put 166 hPa between the lowest levels at 30
    levels and destroy the boundary layer.

    Parameters
    ----------
    n_levels : int
        Number of layers (returns ``n_levels + 1`` half levels).
    sigma_top : float
        Sigma of the model top (same meaning as in
        :func:`create_sigma_coordinate`).
    sigma_refine : float
        Centre of the refinement, in sigma (0.12 ~ 120 hPa at p_s = 1000
        hPa — the tropical cold point).
    refine : float
        Peak density ratio.  ``refine = 1`` reproduces the uniform grid
        EXACTLY (the identity case, pinned by a test).
    width : float
        Gaussian half-width of the bump in log-sigma units.

    Returns
    -------
    numpy.ndarray, shape ``(n_levels + 1,)``
        Monotone increasing half-level sigma from ``sigma_top`` to 1.
    """
    if not (refine >= 1.0 and width > 0.0):
        raise ValueError(
            f"tropopause_refined_sigma_half: need refine >= 1 and width > 0; "
            f"got refine={refine!r}, width={width!r}")
    if not (0.0 < sigma_top < sigma_refine < 1.0):
        raise ValueError(
            f"tropopause_refined_sigma_half: need 0 < sigma_top < "
            f"sigma_refine < 1; got sigma_top={sigma_top!r}, "
            f"sigma_refine={sigma_refine!r}")
    # Fine auxiliary grid; the quantile inversion below is a 1-D interp, so
    # resolution here only sets the placement accuracy (not a runtime cost:
    # this runs once at setup, on the host, in float64).
    s = np.linspace(sigma_top, 1.0, 20001, dtype=np.float64)
    _ln = np.log(s)
    dens = 1.0 + (refine - 1.0) * np.exp(
        -0.5 * ((_ln - np.log(sigma_refine)) / width) ** 2)
    # DELIBERATELY single-bump.  A matching surface bump was tried and
    # rejected (measured 2026-07-25, 30 levels, refine=3): it does protect
    # the lowest layer (42 -> 23 hPa) but it is very WIDE in sigma (it spans
    # sigma ~ 0.6-1.0), so it starves the tropopause back to 5 levels from 8
    # and thickens the TOP layer to 57 hPa.  At 30 levels the grid cannot
    # refine both ends; the tropopause is the identified defect, so it wins.
    # ACCEPTED COST: the lowest layer coarsens ~30% (33 -> 42 hPa at
    # refine=3).  Side benefit: the top layer THICKENS (33 -> 40 hPa), the
    # opposite of the thin-top configuration that blew L40 up at day 46.
    # Equidistribution: place levels at equal increments of the cumulative
    # density, so spacing ~ 1/d — fine where d is large.
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1])
                                           * np.diff(s))])
    cum /= cum[-1]
    targets = np.linspace(0.0, 1.0, n_levels + 1)
    half = np.interp(targets, cum, s)
    # Pin the ends exactly (interp round-off would otherwise move the lid
    # and the surface by ~1e-16, which the hydrostatic integration and the
    # p_s = sigma=1 identity both assume).
    half[0], half[-1] = sigma_top, 1.0
    return half


Here is the vertical biharmonic filter it interacts with
(primitive_eq_mpas.py:148, called on T at line 543 and on EVERY tracer at 586,
with nu_vert4_T = 2.0e-6 1/s):
def vertical_del4_T_tendency(T_3d: jax.Array, nu_vert4_T: float) -> jax.Array:
    """Scale-selective vertical biharmonic damping of the grid-scale T mode.

    Returns the tendency ``-nu · ∂⁴T/∂σ⁴`` (a discrete fourth-difference on the
    level INDEX — so ``nu_vert4_T`` is a filter RATE [1/s], NOT a physical
    hyperdiffusivity like the horizontal ``nu_del4`` [m⁴/s]).  It damps the 2Δσ
    (Nyquist) vertical mode while leaving resolved vertical structure
    essentially untouched — the vertical analogue of the dycore's horizontal
    ``nu_del4`` biharmonic hyperdiffusion.

    Implemented as del2∘del2 (Laplacian of the Laplacian).  Boundary treatment:
    the INNER Laplacian is ``reflect``-padded (so a 2Δσ mode keeps its full
    ``-4`` Laplacian at the top/bottom levels — where the #930 checkerboard is
    worst, at the low-pressure top), while the OUTER Laplacian is ``edge``
    (zero-gradient) padded (a no-flux boundary → the column-integrated tendency
    is ZERO to machine precision for ANY profile, so the filter dissipates
    grid-scale variance WITHOUT spurious column heating/cooling).  Discrete 2Δσ
    ``(-1)^k`` response: ``-16·nu`` in the interior, ``-8·nu`` at the top/bottom
    (½ the interior rate — a boundary no-flux constraint of any conservative
    biharmonic; still strong).  An 8Δσ resolved wave sees ``≈-0.34·nu``
    (≈47× weaker than 2Δσ), so the filter is grid-scale-selective.

    Parameters
    ----------
    T_3d : jax.Array
        Temperature, shape ``(..., nlev)``.
    nu_vert4_T : float
        Biharmonic filter rate [1/s].  ``0.0`` ⇒ exact zero tendency.

    Returns
    -------
    jax.Array
        Vertical-hyperdiffusion tendency of T, same shape as ``T_3d``.
    """
    pad_axes = ((0, 0),) * (T_3d.ndim - 1)
    # Inner Laplacian: reflect BC keeps the FULL 2Δσ response at the boundary
    # levels (a plain edge/no-flux inner BC halves it again and leaves a slowly
    # decaying top boundary mode).
    Tp = jnp.pad(T_3d, (*pad_axes, (1, 1)), mode="reflect")
    lap = Tp[..., :-2] - 2.0 * Tp[..., 1:-1] + Tp[..., 2:]      # ∂²/∂σ²
    # Outer Laplacian: edge (zero-gradient / no-flux) BC ⇒ Σ_k tendency = 0
    # exactly (flux form), so the filter conserves column-integrated T.
    lap_p = jnp.pad(lap, (*pad_axes, (1, 1)), mode="edge")
    bih = lap_p[..., :-2] - 2.0 * lap_p[..., 1:-1] + lap_p[..., 2:]  # ∂⁴/∂σ⁴ (>0 at 2Δσ)
    return -nu_vert4_T * bih


Answer these SEVEN questions, each in one short paragraph, each labelled
CONFIRMED / REFUTED / UNCERTAIN:

Q1. The filter's docstring claims "the column-integrated tendency is ZERO to
machine precision for ANY profile", and the tracer call site claims it
"conserves column moisture to machine precision". I measure sum_k(tend_k) = 0
exactly but sum_k(tend_k * dsigma_k) != 0 on the refined grid: -7.86 W/m2 of
column enthalpy and -0.135 mm/day of column moisture for a +-5 K / +-1 g/kg
2-delta checkerboard (uniform grid: exactly 0; hybrid L40 stretched: -0.99
W/m2). Is my diagnosis right, and which fix is correct --
 (a) tend -= (tend*dsigma).sum()/dsigma.sum()   [mass-weighted mean removal;
     measured to leave the uniform grid unchanged to 1.4e-20 K/s], or
 (b) rewrite as a mass-flux divergence -(F_{k+1/2}-F_{k-1/2})/dsigma_k
     [exactly conservative but changes the effective damping RATE by
     1/dsigma, so NOT backward-compatible with existing uniform runs]?
Say which you would ship and why. Is (a) physically objectionable?

Q2. The refined grid's max adjacent-layer thickness ratio is 1.618 at
refine=3 and 2.406 at refine=6. Every grid this model has ever run is <=1.066.
The config validator allows refine up to 6. What is the correct upper bound
and on what NUMERICAL criterion (name the discretization whose accuracy or
stability degrades, and at what ratio)?

Q3. Does ANY term in a hydrostatic sigma primitive-equation model or its
physics have an explicit stability limit scaling like 1/dsigma or 1/dp, that
would bind at dsigma_min = 0.0144 (14.2 hPa), dt = 75 s? I computed the
first-order-upwind vertical advective CFL as <= 0.26 for omega up to 5 Pa/s,
and the index-space biharmonic limit as 16*nu*dt = 2.4e-3. What did I miss?

Q4. Equidistribution is performed in SIGMA but the density bump is Gaussian in
LOG SIGMA. Is that mathematically coherent? Does the docstring's claim
("spacing ~ 1/d") hold, and does the resulting distribution match what is
claimed (refinement "steals layers from the mid-troposphere")?

Q5. The construction is a 20001-point trapezoid cumulative + np.interp
inversion, float64, run once on the host. I measure max placement error
1.1e-8 in sigma vs a 4e6-point reference, and zero non-monotone cases over a
large parameter sweep including after the float32 storage cast. Is there any
parameter combination in the allowed box (refine in [1,6], width>0,
0<sigma_top<sigma_refine<1, n_levels>=2) where this construction produces a
non-monotone, degenerate, or badly-placed grid? Consider n_levels=2 and 3.

Q6. sigma_refine=0.12 fixes the refinement in SIGMA, so it tracks the surface
pressure: 121 hPa where p_s=1010, but 78 hPa over Tibet where p_s=650. The
target is the tropical cold point at ~95-100 hPa. Is sigma-following
refinement the right choice here, or does this argue for the hybrid
coordinate? Is 0.12 the right centre value?

Q7. What is the single most likely way this change makes a 100-year AMIP run
WORSE than the uniform grid it replaces? Be specific and physical.
