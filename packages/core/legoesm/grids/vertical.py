"""Vertical coordinates for legoESM.

Three vertical coordinate systems are provided:

1. **SigmaCoordinate** (pressure-based, for hydrostatic primitive equations):
   σ = p / p_s, terrain-following in pressure space.

2. **HybridSigmaPressureCoordinate** (hybrid σ-p, for hydrostatic PE):
   p(k) = A(k)·p_ref + B(k)·p_s.  Near the top, levels are pure
   pressure surfaces (B→0); near the surface, pure sigma (A→0, B→1).
   Eliminates pressure-gradient force errors over steep topography.

3. **HeightCoordinate** (height-based, for non-hydrostatic compressible Euler):
   z* = H·(z - z_s)/(H - z_s), terrain-following in physical height space.
   Includes 1D reference state profiles (rho_0, theta_0, pi_0) for
   reference-state subtraction.

4. **TerrainMetric**: 3D coordinate metric terms that depend on surface
   elevation z_s(x, y). Used with HeightCoordinate for terrain-following
   transformations.

All functions are pure (no side effects) and compatible with jax.jit,
jax.grad, jax.vmap, and jax.lax.scan.

References
----------
- Simmons & Burridge (1981): An Energy and Angular-Momentum Conserving
  Vertical Finite-Difference Scheme and Hybrid Vertical Coordinates.
- Klemp et al. (2007): A Terrain-Following Coordinate with Smoothed
  Coordinate Surfaces.
"""

from __future__ import annotations

import logging
import math
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)

_TINY = float(jnp.finfo(jnp.float32).tiny)  # Smallest normal float32 (~1.18e-38)

from legoesm import constants


class SigmaCoordinate(NamedTuple):
    """Sigma vertical coordinate definition.

    Levels are indexed top-to-bottom: k=0 is the model top (σ≈0),
    k=n_levels-1 is the lowest level (σ≈1).

    Fields
    ------
    n_levels : int
        Number of full levels.
    sigma_full : jax.Array
        Sigma values at full (mid-) levels, shape (n_levels,).
        These are cell centers where prognostic variables live.
    sigma_half : jax.Array
        Sigma values at half (interface) levels, shape (n_levels+1,).
        sigma_half[0] = 0 (model top), sigma_half[-1] = 1 (surface).
    dsigma : jax.Array
        Layer thickness in sigma, shape (n_levels,).
        dsigma[k] = sigma_half[k+1] - sigma_half[k].
    ln_ratio : jax.Array
        Log ratio ln(σ_{k+1/2} / σ_{k-1/2}), shape (n_levels,).
        Used in geopotential computation.
    alpha : jax.Array
        Simmons-Burridge coefficient ln(σ_{k+1/2} / σ_full_k), shape (n_levels,).
        Used in geopotential computation.
    fractional_sigma : jax.Array
        (σ_{k+1/2} - σ_top) / (1 - σ_top) for k=1..n_levels, shape (n_levels,).
        Used in sigma-dot computation.
    dsigma_full : jax.Array
        Differences between full levels: diff(sigma_full), shape (n_levels-1,).
        Used in vertical advection.
    """
    n_levels: int
    sigma_full: jax.Array
    sigma_half: jax.Array
    dsigma: jax.Array
    ln_ratio: jax.Array
    alpha: jax.Array
    fractional_sigma: jax.Array
    dsigma_full: jax.Array

    # ------------------------------------------------------------------
    # VerticalCoordProtocol methods
    # ------------------------------------------------------------------

    def pressure_at_full(self, p_s):
        return pressure_from_sigma(self.sigma_full, p_s)

    def pressure_at_half(self, p_s):
        return pressure_from_sigma(self.sigma_half, p_s)

    def layer_thickness_dp(self, p_s):
        return self.dsigma * p_s[..., None]


#: Cost per cell per level, picoseconds, MEASURED on one A100 in float32 with
#: the icosahedral dynamical core at 163,842 cells, 60 timed steps per arm,
#: two replicates each, spreads at or under 1% apart from one 4.8% outlier
#: (2026-08-19, jobs 27070658 / 27073083 / 27073084 / 27074125 / 27074126 /
#: 27074455 / 27076677).  Level counts NOT listed here were not measured.
_LEVEL_COST_PS = {
    13: 3520, 16: 1547, 18: 2318, 20: 1524, 21: 2578, 22: 4910, 24: 4147,
    25: 4224, 26: 4755, 27: 3861, 28: 4058, 30: 4591, 31: 3741, 32: 1629,
    34: 2949, 36: 1706, 40: 1664, 52: 1621,
}
#: Anything at or below this is "as cheap as the cheapest counts measured".
_LEVEL_COST_FAST_PS = 1800
#: The counts that came in at or under that bar, cheapest first by count.
_LEVELS_MEASURED_FAST = tuple(
    n for n in sorted(_LEVEL_COST_PS) if _LEVEL_COST_PS[n] <= _LEVEL_COST_FAST_PS
)


def warn_if_unaligned_levels(n_levels: int, dtype=None, *, where: str) -> None:
    """Warn when the level count was MEASURED expensive ON THE UNSTRUCTURED CORE.

    SCOPE, and it is narrow. The lat-lon core was measured over the same
    level counts on one GPU and is FLAT: 16 through 40 levels span 851 to
    954 picoseconds per column per level, a 1.12x spread end to end, with 26
    levels at 1.04x the cheapest and every arm reproducing to 0.0% (job
    27078027). So this is NOT a compiler or hardware property that every
    model pays -- it belongs to the unstructured core's kernels, and this
    function is called from there rather than from the vertical-coordinate
    factories, which serve both.

    On this model, in float32, on one A100, the cost per cell per level
    varies by a factor of three between level counts, and NOT in any pattern
    that a rule can express.  Measured, picoseconds per cell per level:

        16  1547     18  2318     20  1524     21  2578     22  4910
        24  4147     25  4224     26  4755     27  3861     28  4058
        30  4591     31  3741     32  1629     34  2949     36  1706
        40  1664     52  1621

    Sixteen, twenty, thirty-two, thirty-six, forty and fifty-two are cheap.
    Everything from twenty-two to thirty-one is about three times more
    expensive per level, and so are eighteen, twenty-one and thirty-four.
    The step at thirty levels is 22.6 ms and at thirty-two it is 8.5 ms --
    more work, a third of the time.  Production runs twenty-six.

    TWO EXPLANATIONS WERE TESTED AND BOTH FAILED.  Byte alignment of a
    cell's column: twenty-four and twenty-eight levels give 96- and
    112-byte strides, both multiples of the 16-byte vector width, and both
    are slow.  Cache capacity: fifty-two levels holds the largest live set
    of any count measured and is the cheapest.  The sharp irregular
    transitions -- thirty slow, thirty-two fast, thirty-four slow,
    thirty-six fast -- look like the compiler choosing different code per
    shape, but that is not established either, so this warning names no
    cause.

    What that costs the reader: the table is specific to this model, this
    GPU, this compiler version and float32, and may move under any of them.
    A level count absent from the table is UNMEASURED, not fast.

    ``dtype`` is the dtype the STATE arrays are stored in, resolved from the
    active precision policy when omitted; the measurement is float32 and
    other widths are left alone rather than extrapolated.

    ``where`` names the caller so the message points at the model being
    built.
    """
    import warnings

    if dtype is None:
        try:
            from legoesm.core.precision import get_policy
            dtype = get_policy().storage
        except Exception:
            dtype = jnp.float32
    try:
        itemsize = int(np.dtype(dtype).itemsize)
    except TypeError:
        # An unusual dtype object is not a reason to fail coordinate
        # construction; skip the advisory rather than raise from it.
        return
    n_levels = int(n_levels)
    if n_levels <= 0 or itemsize != 4:
        return
    cost = _LEVEL_COST_PS.get(n_levels)
    if cost is None or cost <= _LEVEL_COST_FAST_PS:
        return
    best = min(_LEVEL_COST_PS[n] for n in _LEVELS_MEASURED_FAST)
    warnings.warn(
        f"{where}: {n_levels} vertical levels was MEASURED expensive on the "
        f"unstructured core — {cost} picoseconds per cell per level against "
        f"{best} for "
        f"the cheapest counts measured, a factor of {cost / best:.1f}. At "
        f"163,842 cells the step is 22.6 ms at 30 levels and 8.5 ms at 32: "
        f"more work, a third of the time. Level counts measured cheap: "
        f"{', '.join(str(n) for n in _LEVELS_MEASURED_FAST)}. The cause is "
        f"NOT established — byte alignment of the column stride and cache "
        f"capacity were both tested and both refuted — so this is a table of "
        f"measurements, not a rule, and counts absent from it are unmeasured "
        f"rather than cheap. If {n_levels} is a physics requirement, keep it "
        f"and expect the cost. The lat-lon core does NOT show this — it is "
        f"flat to 1.12x across the same level counts — so this is specific "
        f"to the unstructured kernels. Re-measure on different hardware or "
        f"a different compiler before trusting any of it.",
        stacklevel=3,
    )


SIGMA_LAYOUTS = ("standard", "l30_trop_logstrat")
# The L30 grid's fourth interface (0.01 + 3 * 0.033) is where the log-spaced
# stratosphere joins the retained troposphere; a lid above 0.05 would leave
# the new layers too thin to resolve in float32 (interfaces collapse near the
# join), so the layout accepts sigma_top in (0, 0.05].
L30_LOGSTRAT_JOIN = 0.109
L30_LOGSTRAT_SIGMA_TOP_MAX = 0.05


def l30_trop_logstrat_sigma_half(n_levels: int, sigma_top: float, dtype) -> jnp.ndarray:
    """Half-level sigmas for the L30-troposphere / log-stratosphere layout.

    Keeps the production L30 sigma grid's bottom 27 layers EXACTLY
    (``jnp.linspace(0.01, 1.0, 31, dtype)[3:]``, join at sigma 0.109) and
    replaces its top three layers by ``n_levels - 27`` layers uniform in
    ln(sigma) from ``sigma_top`` to the join (at p_s 1000 hPa, n_levels 36,
    sigma_top 0.002: 2.0-3.1, 3.1-4.9, 4.9-7.6, 7.6-11.8, 11.8-18.4,
    18.4-28.7, 28.7-44.8, 44.8-69.9, 69.9-109 hPa).  Raises ValueError unless
    ``n_levels >= 28`` and ``0 < sigma_top < 0.109``.
    """
    if n_levels < 28:
        raise ValueError(
            f"n_levels must be >= 28 for the l30_trop_logstrat layout, got {n_levels}")
    # Validation on Python floats (no traced values): the join is the fourth
    # L30 interface, 0.01 + 3*0.033; the top must leave every new layer a
    # thickness resolvable in the working dtype, hence the margin.
    join_f = L30_LOGSTRAT_JOIN
    if not 0.0 < sigma_top <= L30_LOGSTRAT_SIGMA_TOP_MAX:
        raise ValueError(
            "sigma_top must satisfy 0 < sigma_top <= "
            f"{L30_LOGSTRAT_SIGMA_TOP_MAX} (below the L30 join {join_f:.4g}), got {sigma_top}")
    old_half = jnp.linspace(0.01, 1.0, 31, dtype=dtype)
    join = old_half[3]
    n = n_levels - 27
    upper = sigma_top * (join / sigma_top) ** (jnp.arange(n + 1, dtype=dtype) / n)
    upper = upper.at[0].set(sigma_top).at[-1].set(join)
    return jnp.concatenate((upper[:-1], old_half[3:]))


def create_sigma_coordinate(
    n_levels: int,
    sigma_top: float = 0.01,
    dtype=None,
    tropopause_refine: float = 1.0,
    sigma_refine: float = 0.12,
    refine_width: float = 0.45,
    *,
    layout: str = "standard",
) -> SigmaCoordinate:
    """Create a sigma coordinate (uniform by default).

    Parameters
    ----------
    n_levels : int
        Number of vertical levels.
    sigma_top : float
        Sigma value at the model top. Default 0.01 corresponds to
        p_top ≈ 10 hPa when p_s = 1000 hPa. Must be > 0 to avoid
        log(0) in the hydrostatic geopotential integration.
    dtype : jnp.dtype or None
        Dtype for coordinate arrays. If None, uses the precision
        policy's compute dtype (defaults to float32 when no policy
        is active). Explicit dtype overrides the policy.
    tropopause_refine : float
        Peak density ratio for the tropopause refinement (see
        :func:`tropopause_refined_sigma_half`).  ``1.0`` (default) is the
        UNIFORM grid, bit-identical to the pre-refinement behaviour.
        Values > 1 redistribute layers toward ``sigma_refine`` at the SAME
        level count — the fix for the unresolved tropical cold point.
    sigma_refine, refine_width : float
        Centre (in sigma) and log-sigma half-width of the refinement.
    layout : str
        One of ``SIGMA_LAYOUTS``: ``"standard"`` (default) keeps the uniform
        or tropopause-refined placement; ``"l30_trop_logstrat"`` pins the L30
        grid below sigma 0.109 and adds log-spaced layers up to ``sigma_top``.

    Returns
    -------
    SigmaCoordinate
        The vertical coordinate definition.
    """
    # Resolve dtype from precision policy if not explicitly provided.
    # Default to float32 for backward compatibility (PE and tracer
    # transport models run in float32, and mixing float64 sigma arrays
    # with float32 state arrays triggers scatter-cast warnings).
    if dtype is None:
        try:
            from legoesm.core.precision import get_policy
            dtype = get_policy().compute
        except Exception:
            dtype = jnp.float32
    if layout not in SIGMA_LAYOUTS:
        raise ValueError(
            f"layout must be one of SIGMA_LAYOUTS={SIGMA_LAYOUTS}, got {layout!r}")
    if layout == "l30_trop_logstrat":
        if tropopause_refine != 1.0:
            raise ValueError(
                "layout='l30_trop_logstrat' requires tropopause_refine == 1.0, "
                f"got tropopause_refine={tropopause_refine}")
        sigma_half = l30_trop_logstrat_sigma_half(n_levels, sigma_top, dtype)
    elif tropopause_refine == 1.0:
        # Uniform (default) — kept as the literal linspace so the untouched
        # path stays bit-identical to the pre-refinement code.
        sigma_half = jnp.linspace(sigma_top, 1.0, n_levels + 1, dtype=dtype)
    else:
        sigma_half = jnp.asarray(
            tropopause_refined_sigma_half(
                n_levels, sigma_top=sigma_top, sigma_refine=sigma_refine,
                refine=tropopause_refine, width=refine_width),
            dtype=dtype)
    sigma_full = 0.5 * (sigma_half[:-1] + sigma_half[1:])
    dsigma = sigma_half[1:] - sigma_half[:-1]

    # Precompute constants for geopotential (avoids log calls in hot loop)
    sigma_half_safe = jnp.clip(sigma_half, _TINY, None)
    ln_ratio = jnp.log(sigma_half_safe[1:] / sigma_half_safe[:-1])  # (n_levels,)
    # Exact Simmons-Burridge (1981) alpha coefficient:
    #   α_k = 1 - (σ_{k-1/2} / Δσ_k) * ln(σ_{k+1/2} / σ_{k-1/2})
    alpha = 1.0 - (sigma_half_safe[:-1] / dsigma) * ln_ratio  # (n_levels,)

    # Precompute fractional sigma for sigma-dot computation
    sigma_range = 1.0 - sigma_top
    fractional_sigma = (sigma_half[1:] - sigma_top) / sigma_range  # (n_levels,)

    # Precompute full-level differences for vertical advection
    dsigma_full = jnp.diff(sigma_full)  # (n_levels-1,)

    return SigmaCoordinate(
        n_levels=n_levels,
        sigma_full=sigma_full,
        sigma_half=sigma_half,
        dsigma=dsigma,
        ln_ratio=ln_ratio,
        alpha=alpha,
        fractional_sigma=fractional_sigma,
        dsigma_full=dsigma_full,
    )


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
    40 levels the TTL still gets 4 levels, and that L40 run (uniform σ,
    24.4 hPa in EVERY layer) blew up at day 46 with dt=60.  The L40 failure
    mode is NOT attributed here — it appeared across levels 0-11 (10-302 hPa)
    and no mechanism has been instrumented; "thin layers destabilise" is an
    untested hypothesis, so it is not used as a design argument below.

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
    # Fine auxiliary QUADRATURE grid; the quantile inversion below is a 1-D
    # interp, so resolution here only sets the placement accuracy (not a
    # runtime cost: this runs once at setup, on the host, in float64).
    # LOG-spaced, matching the density's own log-σ structure: a σ-uniform mesh
    # has constant Δσ ≈ 5e-5 and therefore CANNOT resolve a narrow bump placed
    # near the lid (``width`` 0.02 at ``sigma_refine`` ≈ ``sigma_top`` = 1e-5
    # spans Δσ ~ 2e-7, i.e. zero mesh points), giving a worst-case placement
    # error of 3.1e-5 in σ over the allowed parameter box.  The log mesh
    # resolves every allowed bump uniformly: worst case 2.0e-7 (155x better),
    # and it moves the DEFAULT grid by only 7.8e-9 in σ (8e-6 hPa) — toward
    # the exact answer, not away.  ``refine == 1`` stays exact either way
    # (a constant integrand is exact under the trapezoid on any mesh).
    s = np.geomspace(sigma_top, 1.0, 20001, dtype=np.float64)
    _ln = np.log(s)
    dens = 1.0 + (refine - 1.0) * np.exp(
        -0.5 * ((_ln - np.log(sigma_refine)) / width) ** 2)
    # DELIBERATELY single-bump.  A matching surface bump was tried and
    # rejected (measured 2026-07-25, 30 levels, refine=3): it does protect
    # the lowest layer (42 -> 23 hPa) but it is very WIDE in sigma (it spans
    # sigma ~ 0.6-1.0), so it starves the tropopause back to 5 levels from 8
    # and thickens the TOP layer to 57 hPa.  At 30 levels the grid cannot
    # refine both ends; the tropopause is the identified defect, so it wins.
    # ACCEPTED COSTS, both measured at nlev=30 / refine=3, both real:
    #   (a) the lowest layer coarsens ~30% (33 -> 42 hPa);
    #   (b) the thinnest layer is 14.2 hPa at ~113 hPa — 42% THINNER than
    #       anything the L40 run that blew up ever had, and inside the same
    #       10-302 hPa band where that failure appeared.  The top layer k=0
    #       does thicken (33 -> 40 hPa), but that covers ONE level of a
    #       twelve-level failure, so it is NOT a safety argument.
    #   (c) the max adjacent-layer thickness ratio rises 1.00 -> 1.62,
    #       vs <= 1.07 on every grid this model has run successfully.
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


def pressure_from_sigma(
    sigma: jax.Array,
    p_s: jax.Array,
) -> jax.Array:
    """Compute pressure from sigma and surface pressure.

    p = σ · p_s

    Parameters
    ----------
    sigma : jax.Array
        Sigma values, shape (nlev,) or (nlev+1,).
    p_s : jax.Array
        Surface pressure, shape (6, n, n).

    Returns
    -------
    jax.Array
        Pressure, shape (6, n, n, nlev) or (6, n, n, nlev+1).
    """
    # Broadcast: p_s[..., None] is (...,1), sigma is (nlev,) -> (...,nlev)
    return p_s[..., None] * sigma


def compute_geopotential(
    T: jax.Array,
    p_s: jax.Array,
    sigma_coord: SigmaCoordinate,
    phis: jax.Array,
) -> jax.Array:
    """Compute geopotential at full levels via Simmons-Burridge integration.

    Uses the standard Simmons & Burridge (1981) hydrostatic integration
    with the proper α_k correction for placing the geopotential at
    full (mid-layer) levels rather than interfaces. This gives much
    better accuracy than simple interface averaging, especially for the
    top layers where σ spacing is large in ln(p).

    The geopotential at full level k is:

        Φ_k = Φ_s + R_d Σ_{k'=k+1}^{nlev-1} T_{k'} ln(σ_{k'+1/2}/σ_{k'-1/2})
              + α_k R_d T_k

    where the Simmons-Burridge coefficient:

        α_k = 1 - (σ_{k-1/2} / Δσ_k) ln(σ_{k+1/2} / σ_{k-1/2})

    This coefficient accounts for the sub-layer vertical distribution
    of pressure. For thin layers (Δσ/σ → 0), α_k → 0.5·ln(σ_{k+1/2}/σ_{k-1/2}),
    recovering the interface-average. For thick layers (like the top level),
    α_k differs significantly from the simple average and gives much
    better accuracy.

    Parameters
    ----------
    T : jax.Array
        Temperature, shape (6, n, n, nlev).
    p_s : jax.Array
        Surface pressure, shape (6, n, n).
    sigma_coord : SigmaCoordinate
        Vertical coordinate.
    phis : jax.Array
        Surface geopotential (g·z_s), shape (6, n, n).

    Returns
    -------
    jax.Array
        Geopotential at full levels, shape (6, n, n, nlev).
    """
    R_d = constants.R_d

    # Use precomputed log ratios and alpha coefficients
    ln_ratio = sigma_coord.ln_ratio  # (nlev,)
    alpha = sigma_coord.alpha  # (nlev,)

    # Geopotential thickness of each full layer (interface to interface):
    # ΔΦ_k = R_d · T_k · ln(σ_{k+1/2} / σ_{k-1/2})
    dPhi = R_d * T * ln_ratio  # (...,nlev)

    # Geopotential at the BOTTOM interface of each layer:
    # Φ_bottom_interface[k] = phis + Σ_{k'=k}^{nlev-1} ΔΦ_{k'}
    # (i.e., the interface ABOVE layer k, which is the bottom of layer k-1)
    # Build from bottom: cumulative sum from k to nlev-1
    dPhi_reversed = dPhi[..., ::-1]
    cumsum_reversed = jnp.cumsum(dPhi_reversed, axis=-1)
    cumsum = cumsum_reversed[..., ::-1]  # cumsum[k] = sum from k to nlev-1

    # Interface geopotentials: Φ_interface[k] = phis + Σ_{k'=k}^{nlev-1} ΔΦ_{k'}
    Phi_above = phis[..., None] + cumsum  # (6,n,n,nlev) — top of each layer

    # Full-level geopotential using Simmons-Burridge:
    # Φ_k = Φ_interface[k+1] + α_k · R_d · T_k
    # where Φ_interface[k+1] is the BOTTOM interface of layer k.
    #
    # Φ_interface[k+1] for k=0..nlev-2 is Phi_above[k+1]
    # Φ_interface[nlev] (surface) = phis
    Phi_below = jnp.concatenate(
        [Phi_above[..., 1:], phis[..., None]], axis=-1
    )  # (6,n,n,nlev) — bottom interface of each layer

    Phi_full = Phi_below + alpha * R_d * T  # (...,nlev)

    return Phi_full


def compute_sigma_dot(
    div_3d: jax.Array,
    sigma_coord: SigmaCoordinate,
) -> jax.Array:
    """Diagnose sigma-dot (vertical velocity in sigma coordinates).

    From the continuity equation in sigma coordinates with σ_top > 0:

        ∂(ln p_s)/∂t + div(v) + ∂σ̇/∂σ = 0

    where ∂(ln p_s)/∂t = -D_total / (1 - σ_top).

    Integrating from the top boundary (σ = σ_top, where σ̇ = 0):

        σ̇_{k+1/2} = (σ_{k+1/2} - σ_top) / (1 - σ_top) · D_total
                     - Σ_{k'=0}^{k} div(v_k') · Δσ_k'

    where D_total = Σ div(v_k) · Δσ_k.

    The factor (σ_{k+1/2} - σ_top) / (1 - σ_top) ensures exact
    discrete continuity closure: the residual
    ∂(ln p_s)/∂t + div(v_k) + (σ̇_{k+1/2} - σ̇_{k-1/2})/Δσ_k = 0
    holds to machine precision at every level.

    Boundary conditions: σ̇ = 0 at σ = σ_top (top) and σ = 1 (surface).

    Parameters
    ----------
    div_3d : jax.Array
        Horizontal divergence at each level, shape (6, n, n, nlev).
    sigma_coord : SigmaCoordinate
        Vertical coordinate.

    Returns
    -------
    jax.Array
        Sigma-dot at half (interface) levels, shape (6, n, n, nlev+1).
        Index 0 = model top (σ̇=0), index nlev = surface (σ̇=0).
    """
    dsigma = sigma_coord.dsigma  # (nlev,)

    # Weighted divergence: D_k * Δσ_k
    div_dsigma = div_3d * dsigma  # (...,nlev)

    # Cumulative sum from top: Σ_{k'=0}^{k} D_k' * Δσ_k'
    # ``cumsum_div[..., -1]`` is exactly ``Σ D_k * Δσ_k`` — extract it
    # rather than calling ``jnp.sum`` independently.  Under any sharding
    # of the level (axis -1) this drops the per-call cross-level
    # collective from 2 to 1.  Mirrors iter-50/51 in the spectral PE.
    cumsum_div = jnp.cumsum(div_dsigma, axis=-1)  # (...,nlev)
    D_total = cumsum_div[..., -1:]  # (...,1)

    # σ̇ at interfaces 1..nlev:
    # σ̇_{k+1/2} = (σ_{k+1/2} - σ_top) / (1 - σ_top) · D_total - cumsum_div[k]
    # Use precomputed fractional_sigma
    fractional_sigma = sigma_coord.fractional_sigma  # (nlev,)

    sigma_dot_inner = (
        fractional_sigma * D_total - cumsum_div
    )  # (...,nlev)

    # Prepend top (σ̇=0) and force bottom boundary (zero by construction).
    # Drop the (∼0) last element + pad with zeros on both ends in one
    # ``jnp.pad`` — replaces alloc-zeros + concatenate + scatter (3 HLO
    # ops) with slice + Pad (2 HLO ops).
    pad_axes = ((0, 0),) * (sigma_dot_inner.ndim - 1)
    sigma_dot = jnp.pad(sigma_dot_inner[..., :-1], (*pad_axes, (1, 1)))

    return sigma_dot


def compute_sigma_dot_and_total(
    div_3d: jax.Array,
    sigma_coord: SigmaCoordinate,
) -> tuple[jax.Array, jax.Array]:
    """Diagnose sigma-dot AND return the column-integrated divergence.

    Identical to :func:`compute_sigma_dot` for the σ̇ output, but also
    returns the column-integrated divergence ``D_total = Σ div_k · Δσ_k``
    (shape ``(..., 1)``).  Callers that need both σ̇ and ``D_total``
    (e.g. the cubed-sphere FV3 PE non-hybrid path, which uses the
    column sum for ``dp_s/dt = -p_s · D_total / (1 - σ_top)``) can use
    this single call instead of running both ``jnp.sum`` and
    ``compute_sigma_dot`` — saving one cross-level collective per call.

    See :func:`compute_sigma_dot` for full documentation.

    Returns
    -------
    sigma_dot : jax.Array, shape (..., nlev+1)
    D_total : jax.Array, shape (..., 1)
    """
    dsigma = sigma_coord.dsigma  # (nlev,)
    div_dsigma = div_3d * dsigma  # (..., nlev)
    cumsum_div = jnp.cumsum(div_dsigma, axis=-1)  # (..., nlev)
    D_total = cumsum_div[..., -1:]  # (..., 1)

    fractional_sigma = sigma_coord.fractional_sigma  # (nlev,)
    sigma_dot_inner = fractional_sigma * D_total - cumsum_div

    pad_axes = ((0, 0),) * (sigma_dot_inner.ndim - 1)
    sigma_dot = jnp.pad(sigma_dot_inner[..., :-1], (*pad_axes, (1, 1)))

    return sigma_dot, D_total


def _vertical_advection_upwind_sigma(
    field: jax.Array,
    sigma_dot: jax.Array,
    sigma_coord: SigmaCoordinate,
) -> jax.Array:
    """First-order upwind (donor-cell) ``-σ̇·∂f/∂σ`` — the DEFAULT path.

    Factored verbatim out of :func:`vertical_advection` so the van-Leer variant
    can reuse it at the two boundary levels without a second copy of the
    stencil.  Byte-identical to the pre-factoring code (pinned by
    ``test_default_is_bit_identical_to_legacy_upwind``).
    """
    # Interpolate σ̇ from half-levels to full levels
    sigma_dot_full = 0.5 * (sigma_dot[..., :-1] + sigma_dot[..., 1:])  # (6,n,n,nlev)

    # Backward / forward differences both consume ``df_bwd / dsigma_bwd``
    # — compute the inner gradient once and pad along the trailing
    # axis instead of building two padded arrays via fresh
    # ``jnp.zeros`` + concatenate.  Single Pad HLO op each.
    dsigma_bwd = sigma_coord.dsigma_full  # (nlev-1,) precomputed diff(sigma_full)
    df_bwd = jnp.diff(field, axis=-1)  # (...,nlev-1) f_{k+1} - f_k
    df_over = df_bwd / dsigma_bwd
    pad_axes = ((0, 0),) * (df_over.ndim - 1)
    # At k=0 (top level): backward gradient = 0 (zero-gradient BC)
    grad_bwd = jnp.pad(df_over, (*pad_axes, (1, 0)))
    # At k=nlev-1 (bottom level): forward gradient = 0 (zero-gradient BC)
    grad_fwd = jnp.pad(df_over, (*pad_axes, (0, 1)))

    # Upwind selection: σ̇ > 0 (downward) → backward, σ̇ < 0 (upward) → forward
    grad = jnp.where(sigma_dot_full > 0, grad_bwd, grad_fwd)

    # Tendency: -σ̇ · ∂f/∂σ
    return -sigma_dot_full * grad


# "sb" is the conservative Simmons-Burridge flux form, HYBRID-lane only
# (:func:`vertical_advection_hybrid_sb`); "van_leer" is SIGMA-lane only.
# Each lane refuses the other's scheme rather than run it silently inert.
VERTICAL_ADVECTION_SCHEMES = ("upwind", "van_leer", "sb", "ppm")


def van_leer_face_values_sigma(
    field: jax.Array,
    sigma_coord: SigmaCoordinate,
) -> tuple[jax.Array, jax.Array]:
    """Slope-limited interface values for the sigma grid, ``(q_pos, q_neg)``.

    ``q_pos`` is the reconstruction from the cell ABOVE interface ``j`` (used
    when σ̇ > 0, descent); ``q_neg`` from the cell BELOW.  Both have shape
    ``(..., nlev+1)``.

    Public so the boundedness property can be tested on THIS reconstruction —
    including its stretched-grid weights and its clip — rather than on the
    uniform-grid :func:`legoesm.core.flux_limiters.van_leer_face_values` it
    reduces to (codex round 1, P2).  See
    :func:`_vertical_advection_van_leer_sigma` for the derivation, the
    boundary treatment and the monotonicity scope.
    """
    return _van_leer_face_values(field, sigma_coord.dsigma_full,
                                 sigma_coord.dsigma)


def _pad_last(x: jax.Array, before: int, after: int) -> jax.Array:
    return jnp.pad(x, ((0, 0),) * (x.ndim - 1) + ((before, after),), mode="edge")


def _van_leer_face_values(
    field: jax.Array,
    dc_full: jax.Array,
    d_layer: jax.Array,
    smoothing_width: float = 0.0,
) -> tuple[jax.Array, jax.Array]:
    """Metric-aware van-Leer face values on any monotone vertical coordinate.

    ``dc_full`` (..., nlev-1) are centre-to-centre spacings and ``d_layer``
    (..., nlev) layer thicknesses, in the same unit (sigma or Pa); both may be
    1-D (sigma) or carry the column axes (hybrid pressure).
    """
    from legoesm.core.flux_limiters import (
        grad_safe_ratio,
        ratio_grad_floor,
        van_leer_limiter,
    )

    nlev = field.shape[-1]
    if nlev < 4:
        raise ValueError(
            f"the van-Leer vertical reconstruction needs at least 4 vertical "
            f"levels for its 4-cell stencil; got nlev={nlev}."
        )
    # Linear-extrapolation ghosts: f_{-1} = 2f_0 - f_1 places the ghost one
    # CENTRE spacing beyond the edge, consistent with the edge-padded spacings
    # below, so a linear profile keeps r = 1 at face 1.
    f0, f1 = field[..., 0:1], field[..., 1:2]
    fm1, fm2 = field[..., -1:], field[..., -2:-1]
    fp = jnp.concatenate(
        [3.0 * f0 - 2.0 * f1, 2.0 * f0 - f1, field,
         2.0 * fm1 - fm2, 3.0 * fm1 - 2.0 * fm2], axis=-1)  # (..., nlev+4)
    # Interface j (0..nlev) separates cell j-1 (above) from cell j (below);
    # the 4-cell stencil is [j-2, j-1, j, j+1].
    f_jm2 = fp[..., 0:nlev + 1]
    f_jm1 = fp[..., 1:nlev + 2]
    f_j = fp[..., 2:nlev + 3]
    f_jp1 = fp[..., 3:nlev + 4]
    # Centre-to-centre spacings, edge-padded: dc_up/dc_loc/dc_dn at face j are
    # sigma_full[j-1]-sigma_full[j-2], [j]-[j-1], [j+1]-[j].
    dcp = _pad_last(dc_full, 2, 2)  # (..., nlev+3)
    dc_up = dcp[..., 0:nlev + 1]
    dc_loc = dcp[..., 1:nlev + 2]
    dc_dn = dcp[..., 2:nlev + 3]
    # MUSCL face weights: donor half-thickness / centre-to-centre distance.
    # Exactly 0.5 each on a uniform grid.
    dsp = _pad_last(d_layer, 1, 1)  # (..., nlev+2)
    d_above, d_below = dsp[..., :-1], dsp[..., 1:]          # (..., nlev+1)
    d_sum = d_above + d_below

    if smoothing_width > 0.0:
        from legoesm.core.smooth import smooth_abs, smooth_max, smooth_min

        # Work in mixing-ratio increments, avoiding singular slope ratios at
        # plateaus. The harmonic mean tends to the hard van Leer slope as
        # width tends to zero; equal slopes and constant fields remain exact.
        delta = f_j - f_jm1
        above = (f_jm1 - f_jm2) * (dc_loc / dc_up)
        below = (f_jp1 - f_j) * (dc_loc / dc_dn)
        abs_delta = smooth_abs(delta, smoothing_width**2)
        abs_above = smooth_abs(above, smoothing_width**2)
        abs_below = smooth_abs(below, smoothing_width**2)
        slope_pos = (above * abs_delta + delta * abs_above) / (abs_above + abs_delta)
        slope_neg = (below * abs_delta + delta * abs_below) / (abs_below + abs_delta)
        corrections = jnp.stack(((d_above / d_sum) * slope_pos,
                                 (d_below / d_sum) * slope_neg))
        # median(c, 0, delta) is clip(c, min(0,delta), max(0,delta)).
        # Smooth max/min of the SAME three values are opposite at c=delta=0:
        # their sum cancels exactly, preserving a constant field. This also
        # approaches the hard stretched-grid clip as width tends to zero.
        sharpness = 1.0 / smoothing_width
        upper = smooth_max(smooth_max(corrections, 0.0, sharpness), delta, sharpness)
        lower = smooth_min(smooth_min(corrections, 0.0, sharpness), delta, sharpness)
        bounded = corrections + delta - (upper + lower)
        return f_jm1 + bounded[0], f_j - bounded[1]

    eps = 1e-30
    t_grad = ratio_grad_floor(jnp.result_type(field))
    delta = f_j - f_jm1
    s_loc = delta / dc_loc
    ok = jnp.abs(s_loc) > t_grad
    den = jnp.where(jnp.abs(s_loc) > eps, s_loc, eps)
    r_pos = grad_safe_ratio((f_jm1 - f_jm2) / dc_up, den, ok)
    r_neg = grad_safe_ratio((f_jp1 - f_j) / dc_dn, den, ok)
    q_pos = f_jm1 + (d_above / d_sum) * van_leer_limiter(r_pos) * delta
    q_neg = f_j - (d_below / d_sum) * van_leer_limiter(r_neg) * delta
    # Monotone bound (non-binding on a uniform grid; see the kernel docstring).
    lo, hi = jnp.minimum(f_jm1, f_j), jnp.maximum(f_jm1, f_j)
    return jnp.clip(q_pos, lo, hi), jnp.clip(q_neg, lo, hi)


def _vertical_advection_van_leer_sigma(
    field: jax.Array,
    sigma_dot: jax.Array,
    sigma_coord: SigmaCoordinate,
) -> jax.Array:
    """Monotone (van-Leer TVD) sigma vertical advection ``-σ̇·∂f/∂σ``.

    Sigma-coordinate counterpart of ``compressible_euler.
    _theta_vert_advection_van_leer_kernel`` (height coordinate, iter-200) and of
    the plane CRM's ``vertical_advection_van_leer_plane``.

    RELATION TO THE SHARED LIMITER, stated precisely (codex round 2): this is a
    METRIC-AWARE sigma-specific reconstruction
    (:func:`van_leer_face_values_sigma`), NOT a call into
    :func:`legoesm.core.flux_limiters.van_leer_face_values`.  It uses that
    module's ``van_leer_limiter`` and ``grad_safe_ratio`` — the limiter shape
    and the AD-safe ratio guard, which are the parts that carry numerics — but
    forms its own SLOPE ratios and Δσ-weighted face positions, because the
    shared helper assumes uniform spacing and cannot express them (with raw
    difference ratios the scheme was measurably WORSE than upwind on a
    stretched grid).  On a uniform grid the two are equal, and
    ``test_uniform_grid_matches_the_shared_van_leer_face_values`` asserts
    BOTH returned arrays element-by-element against the shared helper (not a
    tendency inversion, which would only pin successive differences — codex
    round 3) — so the HD-1 smoothness-ratio SIGN convention is pinned to the
    shared implementation by test, not by call graph.

    ADVECTIVE form, like the first-order sibling — the flux form
    ``-∂(σ̇f)/∂σ`` would add the spurious ``-f·∂σ̇/∂σ``.  Written as the
    *difference of face-minus-cell* increments::

        -σ̇·∂f/∂σ|_k = -[ σ̇_{k+1}(q_{k+1} - f_k) - σ̇_k(q_k - f_k) ] / Δσ_k

    which is algebraically the flux divergence minus ``f_k·∂σ̇/∂σ`` but forms
    no large cancelling pair, and is exactly invariant to a constant offset in
    ``f`` at every level including the boundaries.

    ``q_j`` is the slope-limited face value at interface ``j`` (σ increases
    DOWNWARD with index, so σ̇ > 0 = descent ⇒ the donor cell is ``j-1``,
    above).  The two GHOST cells at each end are LINEAR EXTRAPOLATIONS, not an
    ``edge`` (zero-gradient) pad: an edge pad zeroes the upwind slope seen by
    faces 1 and ``nlev-1``, collapsing the limiter to donor cell there and
    re-introducing first-order diffusion two levels deep into the interior —
    measured on a linear profile (exact for this scheme) as a 50% tendency
    error at level 1, i.e. right at the 40-100 hPa levels this exists to fix.

    BOUNDARY LEVELS ``k=0`` and ``k=nlev-1`` KEEP THE FIRST-ORDER TENDENCY, and
    that is a stability requirement, not a shortcut.  σ̇ vanishes at the lid and
    the surface, so the boundary layer has only ONE interior face; for descent
    at the top (σ̇ > 0) that face is DOWNWIND of the cell, and a downwind
    difference is unconditionally unstable — the update becomes
    ``f_0 <- (1+νθ)f_0 - νθ f_1``, an extrapolation whose coefficient on the
    neighbour is negative.  Measured: 500 forward-Euler steps grew a random
    column from 300 K to 360 K before this fallback was added.  The first-order
    path's zero-gradient BC (gradient = 0 when the upstream cell is outside the
    domain) is the stable choice, so the two boundary levels are byte-identical
    to today under BOTH schemes.  The published diagnosis already excludes k=0
    as a boundary-stencil artifact, and k=0 here is 26 hPa — six levels above
    the 91.4 hPa maximum this option targets.

    MONOTONICITY — what is and is NOT guaranteed (codex rounds 1-3, P1).  Two
    separate statements; do not conflate them.

    (a) The INTERIOR RECONSTRUCTION is bounded UNCONDITIONALLY: for faces
    ``1..nlev-1`` ``q`` is clipped to the two adjacent REAL cell values, so a
    face value never leaves the local range, on any grid, for any input.
    Faces ``0`` and ``nlev`` are bounded against a GHOST and a real cell, but
    they never enter a tendency — both boundary levels take the first-order
    result (below).

    (b) The UPDATE is monotone only under a Courant condition, and the
    DERIVATION below holds for a UNIFORM grid.  One forward-Euler step of the
    advective form with ``q_j = f_{j-1} + θ_j (f_j - f_{j-1})``,
    ``θ_j in [0, 1]``, gives::

        f_k^{n+1} = f_k + C⁻(f_{k-1} - f_k),
        C⁻ = ν_k (1 - θ_k) + ν_{k+1} θ_{k+1} / r_{k+1},   ν = σ̇ dt / Δσ_k

    and van Leer's ``phi(r)/r <= 2`` with the uniform weight ``w = 1/2``
    bounds ``C⁻ <= ν_k + ν_{k+1}``, i.e. TVD while ``ν_k + ν_{k+1} <= 1``.

    SCOPE of (b), stated because codex round 3 caught the over-reach: on a
    STRETCHED grid the ``θ_{k+1}/r_{k+1}`` factor carries the neighbouring
    layer's width ratio rather than this face's weight, so the constant in the
    bound changes and ``ν_k + ν_{k+1} <= 1`` is NOT derived there; and with a
    VARYING σ̇ the advective (non-conservative) form is not automatically TVD
    at all — the anti-diffusive face term is uncompensated at the cell edge.
    On both of those there is NO guarantee: what exists is REGRESSION
    EVIDENCE from a small number of cases —
    ``test_tracer_blob_stays_positive_and_bounded_under_varying_sigma_dot``
    (two profiles, sign-changing σ̇, ~2.1x the production raw-face Courant) and
    ``test_stretched_grid_stays_exact_on_linear_and_monotone`` (one random
    column on ``refine=3``).  Examples, not a proof; a refined-grid production
    arm should re-measure.

    And it is genuinely NOT monotone above the bound: codex produced an
    overshoot to ``-0.017 / 0.991`` from a ``[0, 1]`` Gaussian in one Euler
    step at per-face Courant 0.75 (a top-hat does NOT trigger it; the first
    overshoot appears near 0.6).  First-order upwind, by contrast, is monotone
    for ``ν <= 1`` on any grid, so this scheme trades stability margin for
    order.

    MEASURED on the run this targets (``s9_courant.py``), in the RAW INTERFACE
    velocities the update actually multiplies — NOT the half->full average the
    first-order path uses, which cancels opposite-signed adjacent interfaces
    (codex round 2): the global max of ``ν_k + ν_{k+1}`` over all cells, levels
    and the 37 year-1 checkpoints is **0.0642** at dt = 75 s (per-checkpoint
    mean 0.0537, median 0.0543; largest single raw face 0.0322), i.e. **15.6x**
    inside the bound.  SCOPE: those are 10-day checkpoint SNAPSHOTS, so they
    bound the sampled phase, not every RK stage of every step — treat the
    margin as strong evidence, not a proof.  Do NOT read the "monotone" label
    as unconditional.

    Order and dissipation: for a smooth profile the van-Leer limiter tends to
    ``phi = 1`` and ``q`` becomes the linear interpolant, i.e. the CENTRED
    2nd-order operator — the ``K_σ = |σ̇|Δσ/2`` implicit diffusion of the
    first-order upwind gradient is removed, not merely reduced.  At an extremum
    ``phi -> 0`` and the scheme falls back to donor cell, which is what
    suppresses a dispersive overshoot at a sharp tropopause (worse than the
    diffusion it cures) — under the Courant condition of (b) above, not
    unconditionally.  Note the model advects θ, not T: θ is monotone in a stably
    stratified column, so the tropical cold point is NOT a θ extremum and the
    limiter does not clip there.

    NON-UNIFORM σ: the reconstruction uses true SLOPES (divided by the
    centre-to-centre spacing) and a face weight
    ``w = Δσ_donor / (Δσ_donor + Δσ_other)``, so the smooth limit is the exact
    linear interpolant on ANY spacing.  Both reduce to the uniform-grid
    ``0.5 * phi * Δf`` form when the spacing is constant, and
    ``test_uniform_grid_matches_the_shared_van_leer_face_values`` asserts
    equality with :func:`legoesm.core.flux_limiters.van_leer_face_values` there
    — so the HD-1 smoothness-ratio sign convention is still pinned to the one
    shared implementation.  Without this, a ``tropopause_refine > 1`` grid made
    the scheme WORSE than upwind on a linear profile (9.2e-07 vs 1.3e-17),
    because upwind's divided difference is exact on a linear field at any
    spacing while a raw-difference ratio gives ``phi != 1``.  The extra
    ``clip`` to the two adjacent cell values is what keeps the RETURNED face
    value in range when ``w > 1/2`` — note it bounds ``q``, NOT the product
    ``w * phi``, which is formed before the clip and can reach ``1.5`` at
    ``w = 0.75, phi = 2`` (codex round 4).  The clip is provably non-binding on
    a uniform grid, where ``w * phi <= 1`` already.
    """
    if field.shape[-1] < 4:
        # FAIL LOUD rather than run inert: the 4-cell stencil is undefined and
        # a silent fall-back to upwind is exactly the "selected but does
        # nothing" failure this repo forbids (codex round 1, P2).
        raise ValueError(
            f"scheme='van_leer' needs at least 4 vertical levels for its "
            f"4-cell stencil; got nlev={field.shape[-1]}. Use scheme='upwind'."
        )
    up = _vertical_advection_upwind_sigma(field, sigma_dot, sigma_coord)
    q_pos, q_neg = van_leer_face_values_sigma(field, sigma_coord)
    # σ̇ > 0 (descent) ⇒ donor is the cell ABOVE ⇒ the left-biased value.
    q_face = jnp.where(sigma_dot > 0, q_pos, q_neg)  # (..., nlev+1)
    inc_top = sigma_dot[..., :-1] * (q_face[..., :-1] - field)
    inc_bot = sigma_dot[..., 1:] * (q_face[..., 1:] - field)
    tend = -(inc_bot - inc_top) / sigma_coord.dsigma
    # Boundary levels keep the first-order (stable, zero-gradient-BC) tendency.
    return tend.at[..., 0].set(up[..., 0]).at[..., -1].set(up[..., -1])


def vertical_advection(
    field: jax.Array,
    sigma_dot: jax.Array,
    sigma_coord: SigmaCoordinate,
    scheme: str = "upwind",
) -> jax.Array:
    """Compute vertical advection using the advective form with upwind.

    Computes: -σ̇ · ∂f/∂σ

    This is the **advective form**, appropriate for non-mass-weighted
    variables (u, v, T). The flux form -∂(σ̇·f)/∂σ would add a spurious
    term -f·∂σ̇/∂σ that causes exponential instability.

    σ̇ is interpolated from half-levels to full levels, and the vertical
    gradient uses upwind differencing:
    - σ̇ > 0 (downward): ∂f/∂σ ≈ (f_k - f_{k-1}) / Δσ  (backward)
    - σ̇ < 0 (upward):   ∂f/∂σ ≈ (f_{k+1} - f_k) / Δσ  (forward)

    Parameters
    ----------
    field : jax.Array
        Field to advect, shape (6, n, n, nlev).
    sigma_dot : jax.Array
        Sigma-dot at interfaces, shape (6, n, n, nlev+1).
    sigma_coord : SigmaCoordinate
        Vertical coordinate (provides sigma_full and dsigma).
    scheme : str
        ``"upwind"`` (default) = the first-order donor-cell gradient
        described above, whose leading truncation error is a diffusion
        ``K_σ = |σ̇|·Δσ/2``.  ``"van_leer"`` = the 2nd-order TVD reconstruction
        (:func:`_vertical_advection_van_leer_sigma`), which removes that
        implicit diffusion where the profile is smooth; its face values are
        unconditionally bounded and its update is monotone under a Courant
        condition (see that docstring — NOT unconditionally).  Requires
        ``nlev >= 4`` and raises below it.  Static Python string: the branch is
        resolved at trace time, only one branch is traced, and the default path
        is untouched (bit-identical).  Both branches are differentiable ALMOST
        EVERYWHERE — the upwind ``where(σ̇>0, ...)``, the van-Leer ``abs`` kink
        at ``r=0`` and the reconstruction ``clip`` are each non-smooth on a
        measure-zero set, as in every limiter in this repo.

    Returns
    -------
    jax.Array
        Vertical advection tendency, shape (6, n, n, nlev).
    """
    if scheme == "van_leer":
        return _vertical_advection_van_leer_sigma(field, sigma_dot, sigma_coord)
    if scheme != "upwind":
        raise ValueError(
            f"unknown vertical advection scheme {scheme!r} for the sigma "
            "coordinate; expected 'upwind' or 'van_leer' ('sb' is the "
            "hybrid-coordinate flux form, vertical_advection_hybrid_sb)"
        )
    return _vertical_advection_upwind_sigma(field, sigma_dot, sigma_coord)


#: Vertical schemes of the prescribed-wind tracer transport models:
#: "upwind" = the in-tendency donor cell (:func:`vertical_advection`);
#: "fv3_remap" = the split Lagrangian remap (:func:`vertical_remap_sigma_fv3`).
TRACER_VERTICAL_SCHEMES = ("upwind", "fv3_remap")


def check_tracer_vertical_scheme(scheme: str) -> None:
    """Raise on an unknown prescribed-wind tracer vertical scheme."""
    if scheme not in TRACER_VERTICAL_SCHEMES:
        raise ValueError(f"unknown vertical_scheme {scheme!r}; expected one "
                         f"of {TRACER_VERTICAL_SCHEMES}")


#: FV3 tracer remap order of the certified duo deck (``FV3DuoConfig.kord_tr``).
FV3_REMAP_KORD_TR = 9


def vertical_remap_sigma_fv3(
    q: jax.Array,
    sigma_dot: jax.Array,
    sigma_coord: SigmaCoordinate,
    dt: float,
    *,
    kord: int = FV3_REMAP_KORD_TR,
) -> jax.Array:
    """Split Lagrangian vertical step of prescribed-wind tracer transport,
    using the FV3 tracer remap kernel.

    Not FV3's transport itself (FV3 moves layers with the air-mass fluxes of
    its own dynamics and remaps the resulting pressure interfaces); here the
    interface displacement is PRESCRIBED, ``dt * sigma_dot``, and the
    horizontal step before it is the caller's advective-form one.  The
    layer interfaces are moved for one step (Lagrangian surfaces) and the
    tracer is remapped back to the fixed sigma levels with the certified
    duo remap
    :func:`legoesm.core.fv3_mapz.map1_q2` (``fv_mapz.F90`` tracer branch:
    ``iv=0`` positive-definite ``scalar_profile``, ``kord`` 9 = the duo
    deck's ``kord_tr``, ``q_min=0``, no ``fillz`` as the deck's
    ``fill=.F.``).

    Source (Lagrangian) interfaces: ``sigma_half + dt * sigma_dot``; target
    (Eulerian) interfaces: ``sigma_half``.  The coordinate is sigma itself:
    with a uniform surface pressure ``p = p_s sigma`` and the remap is
    homogeneous of degree 0 in the coordinate (only thickness ratios enter
    ``scalar_profile`` / ``_rezone``), so a ``p_s`` scale factor cancels.

    Budget (RULE 0): the remap conserves the column integral of the tracer
    over the Lagrangian layers, ``sum_k q_new dsigma_k == sum_k q
    dsigma_lag_k`` to round-off, because both interface sets share the top
    and bottom (``sigma_dot`` is zero there).  It is the advective form of
    ``dq/dt = -sigma_dot dq/dsigma`` (a uniform tracer stays uniform), like
    the upwind operator it replaces.  KNOWN NON-CONSERVING PATH (reported,
    RULE 0): composed with the callers' advective-form horizontal step the
    GLOBAL tracer mass is not conserved (measured, DCMIP 1-2 lat-lon
    72x144 L30, one day: -0.023 %, vs -8.2 % for the upwind default);
    closing it needs flux-form horizontal transport on evolving layer
    thicknesses, a design change.

    Refusals (never clip): a nonzero ``sigma_dot`` at the model top or
    surface (mass would leave the column; read from the bits of the
    caller's values, so -0.0 passes and a subnormal does not), a
    subnormal ``sigma_dot`` or ``dt``, a nonzero wind whose
    ``dt * sigma_dot`` underflows, a displacement that rounds on the
    interface by more than 1e-3 of an adjacent layer, a non-finite interface or thickness, Lagrangian interfaces
    that cross or touch (``dsigma_lag <= 0``, the vertical Lagrangian CFL
    limit), a target layer of non-positive thickness, a non-scalar dt,
    and a target edge the rezone cannot bracket, raise at run time through
    ``equinox.error_if`` (``EQX_ON_ERROR=raise``, equinox's default; under
    ``EQX_ON_ERROR=nan`` the output is NaN-poisoned instead, never a
    plausible value).  Geometry, guards and kernel run in the promoted
    dtype of tracer, wind, dt and coordinate; only the output is cast back
    to the tracer dtype.

    Parameters
    ----------
    q : (..., nlev, n_tracers)
    sigma_dot : (..., nlev+1) at half levels, evaluated by the caller at the
        mid-time of the step.
    sigma_coord : SigmaCoordinate
    dt : step [s]
    kord : FV3 remap order (STATIC; ``kord <= 7`` is refused by fv3_mapz).

    Returns
    -------
    (..., nlev, n_tracers) remapped tracers.
    """
    import equinox as eqx
    from legoesm.core.fv3_mapz import map1_q2

    *lead, nlev, ntr = q.shape
    if sigma_dot.shape != (*lead, nlev + 1):
        raise ValueError(f"sigma_dot shape {sigma_dot.shape} != "
                         f"{(*lead, nlev + 1)}")
    if tuple(jnp.shape(sigma_coord.sigma_half)) != (nlev + 1,):
        raise ValueError(f"sigma_half shape {jnp.shape(sigma_coord.sigma_half)}"
                         f" != {(nlev + 1,)}")
    for _nm, _a in (("q", q), ("sigma_dot", sigma_dot)):
        if not jnp.issubdtype(jnp.asarray(_a).dtype, jnp.floating):
            raise ValueError(f"vertical_remap_sigma_fv3: {_nm} must be a "
                             f"real floating dtype, got {jnp.asarray(_a).dtype}")
    dtype = q.dtype
    im = math.prod(lead)
    sd_in = sigma_dot.reshape(im, nlev + 1)
    # NaN/Inf at a boundary is reported by the non-finite guard instead.
    # Nonzero is read from the BITS (sign bit masked: -0.0 is allowed),
    # so a subnormal boundary wind is refused even on backends that
    # flush subnormals to zero in comparisons.
    bnd = jnp.stack([sd_in[:, 0], sd_in[:, -1]])
    _uint = {2: jnp.uint16, 4: jnp.uint32, 8: jnp.uint64}[
        jnp.dtype(bnd.dtype).itemsize]
    _bits = jax.lax.bitcast_convert_type(bnd, _uint)
    _sign = jnp.array(1, _uint) << (8 * jnp.dtype(_uint).itemsize - 1)
    _mag = _bits & ~_sign
    boundary_flux = jnp.any((_mag != 0) & jnp.isfinite(bnd))
    # A SUBNORMAL wind anywhere is refused: its product with dt is not
    # reliable on flush-to-zero backends (bits nonzero, |value| < tiny).
    _all_bits = jax.lax.bitcast_convert_type(sd_in, _uint) & ~_sign
    subnormal = jnp.any((_all_bits != 0)
                        & (jnp.abs(sd_in) < jnp.finfo(sd_in.dtype).tiny))
    # The whole remap (geometry, guards and kernel) runs in the promoted
    # dtype cdt of tracer, wind, dt and coordinate; only the output is cast
    # back to the tracer dtype.  So no early cast of the wind or the
    # coordinate can hide a crossing or erase a displacement, and the
    # column budget closes on the geometry the kernel actually used.
    dt_a = jnp.asarray(dt)
    if dt_a.ndim != 0 or jnp.issubdtype(dt_a.dtype, jnp.complexfloating):
        raise ValueError(f"vertical_remap_sigma_fv3: dt must be a real "
                         f"scalar, got shape {dt_a.shape} {dt_a.dtype}")
    if isinstance(dt, (int, float, np.integer, np.floating)):
        # host scalar (checked with numpy, also when called under a trace):
        # the conversion itself (e.g. x64 disabled) must not turn a nonzero
        # dt into zero or a subnormal
        _hdt = jax.dtypes.canonicalize_dtype(np.asarray(dt).dtype)
        if np.issubdtype(_hdt, np.floating):
            _dtc = float(np.asarray(dt, _hdt))
            if float(dt) != 0.0 and abs(_dtc) < float(np.finfo(_hdt).tiny):
                raise ValueError(f"vertical_remap_sigma_fv3: dt={dt!r} "
                                 f"underflows in {_hdt}; refusing")
    sh_raw = jnp.asarray(sigma_coord.sigma_half)
    cdt = jnp.promote_types(
        jnp.promote_types(jnp.promote_types(dtype, sd_in.dtype), dt_a.dtype),
        sh_raw.dtype if jnp.issubdtype(sh_raw.dtype, jnp.floating)
        else dtype)
    if jnp.issubdtype(dt_a.dtype, jnp.floating):
        _dt_u = {2: jnp.uint16, 4: jnp.uint32, 8: jnp.uint64}[
            jnp.dtype(dt_a.dtype).itemsize]
        _dt_bits = jax.lax.bitcast_convert_type(dt_a, _dt_u) & ~(
            jnp.array(1, _dt_u) << (8 * jnp.dtype(_dt_u).itemsize - 1))
        dt_nonzero = _dt_bits != 0
        dt_subnormal = dt_nonzero & (jnp.abs(dt_a)
                                     < jnp.finfo(dt_a.dtype).tiny)
    else:
        dt_nonzero = dt_a != 0
        dt_subnormal = jnp.asarray(False)
    disp = dt_a.astype(cdt) * sd_in.astype(cdt)
    # a nonzero wind whose displacement underflows (flushed to zero or
    # subnormal) over a nonzero step would silently drop motion: refused;
    # dt's own zero/subnormal test reads its bits (flush-to-zero safe)
    underflow = dt_subnormal | (dt_nonzero & jnp.any(
        (_all_bits != 0) & (jnp.abs(disp) < jnp.finfo(cdt).tiny)))
    s_half = sh_raw.astype(cdt)
    pe2 = jnp.broadcast_to(s_half, (im, nlev + 1))
    pe1 = pe2 + disp
    # The displacement may round in pe2 + disp, but not by more than
    # 1e-3 of an adjacent target layer (a relative perturbation of that
    # layer's budget); otherwise the motion is not representable: refused.
    _dp2 = s_half[1:] - s_half[:-1]
    _thin = jnp.minimum(_dp2[:-1], _dp2[1:])            # interior interfaces
    rounded_away = jnp.any(    # gated on positive target layers below
        jnp.abs((pe1 - pe2)[:, 1:-1] - disp[:, 1:-1]) > 1.0e-3 * _thin)
    dp1 = pe1[:, 1:] - pe1[:, :-1]
    dp2 = s_half[1:] - s_half[:-1]
    nonfinite = (jnp.any(~jnp.isfinite(pe1)) | jnp.any(~jnp.isfinite(s_half))
                 | jnp.any(~jnp.isfinite(dp1)) | jnp.any(~jnp.isfinite(dp2)))
    # One guard per predicate, each with its own diagnostic (the finite
    # checks are masked by ``nonfinite`` so a NaN reports as non-finite).
    pe1 = eqx.error_if(
        pe1, boundary_flux,
        "vertical_remap_sigma_fv3: sigma_dot is nonzero at the top/surface "
        "(mass would leave the column); refusing")
    pe1 = eqx.error_if(
        pe1, subnormal,
        "vertical_remap_sigma_fv3: sigma_dot has a subnormal value "
        "(unreliable displacement); refusing")
    pe1 = eqx.error_if(
        pe1, underflow,
        "vertical_remap_sigma_fv3: a nonzero sigma_dot gives an underflowing "
        "(zero/subnormal) displacement dt*sigma_dot; refusing")
    pe1 = eqx.error_if(
        pe1, nonfinite,
        "vertical_remap_sigma_fv3: an interface or thickness is "
        "non-finite; refusing")
    pe1 = eqx.error_if(
        pe1, ~nonfinite & jnp.any(~(dp2 > 0)),
        "vertical_remap_sigma_fv3: a target (Eulerian) layer is not "
        "positive in the remap dtype; refusing")
    pe1 = eqx.error_if(
        pe1, ~nonfinite & jnp.all(dp2 > 0) & rounded_away,
        "vertical_remap_sigma_fv3: the displacement is not representable on "
        "the interface (rounding > 1e-3 of an adjacent layer); refusing")
    pe1 = eqx.error_if(
        pe1, ~nonfinite & jnp.any(~(dp1 > 0)),
        "vertical_remap_sigma_fv3: Lagrangian interfaces cross/touch "
        "(vertical CFL); refusing")
    # Fold tracers into the column axis: one rezone for all tracers.
    qc = jnp.moveaxis(q.astype(cdt).reshape(im, nlev, ntr), -1, 1).reshape(
        im * ntr, nlev)
    zero = jnp.zeros((im * ntr, 1), cdt)
    pe1c = jnp.concatenate([zero, jnp.repeat(pe1, ntr, axis=0)], axis=1)
    pe2c = jnp.concatenate([zero, jnp.repeat(pe2, ntr, axis=0)], axis=1)
    dp2c = jnp.concatenate([zero, pe2c[:, 2:] - pe2c[:, 1:-1]], axis=1)
    q2, ok = map1_q2(pe1c, jnp.concatenate([zero, qc], axis=1), pe2c, dp2c,
                     nlev, nlev, 0, kord, 0.0, return_ok=True)
    q2 = eqx.error_if(q2, ~ok, "vertical_remap_sigma_fv3: unbracketed "
                      "target edge in the FV3 rezone; refusing")
    q2 = q2[:, 1:].reshape(im, ntr, nlev).astype(dtype)
    return jnp.moveaxis(q2, 1, -1).reshape(*lead, nlev, ntr)


def vertical_advection_theta(
    T: jax.Array,
    sigma_dot: jax.Array,
    p_s: jax.Array,
    sigma_coord: SigmaCoordinate,
    scheme: str = "upwind",
) -> jax.Array:
    """Combined vertical advection + adiabatic σ̇ term for temperature.

    Instead of computing  -σ̇·∂T/∂σ  and  κ·T·σ̇/σ  separately (which
    involves catastrophic cancellation at upper levels where 1/σ → ∞),
    this function advects potential temperature θ and converts back:

        -σ̇·∂T/∂σ + κ·T·σ̇/σ  =  -(p/p₀)^κ · σ̇·∂θ/∂σ

    This eliminates the 1/σ amplification and is numerically stable at
    all levels.

    Parameters
    ----------
    T : jax.Array
        Temperature, shape (..., nlev).
    sigma_dot : jax.Array
        Sigma-dot at interfaces, shape (..., nlev+1).
    p_s : jax.Array
        Surface pressure, shape (...).
    sigma_coord : SigmaCoordinate
        Vertical coordinate.
    scheme : str
        Vertical advection scheme for the θ transport, forwarded verbatim to
        :func:`vertical_advection` (``"upwind"`` default = bit-identical).

    Returns
    -------
    jax.Array
        Combined tendency: -σ̇·∂T/∂σ + κ·T·σ̇/σ, shape (..., nlev).
    """
    kappa = constants.kappa
    sigma_full = sigma_coord.sigma_full  # (nlev,)
    P_0 = constants.p_ref

    # Pressure at full levels
    p_full = sigma_full * p_s[..., None]  # (..., nlev)

    # Potential temperature: θ = T · (p₀/p)^κ
    theta = T * (P_0 / jnp.maximum(p_full, 1.0)) ** kappa

    # Advect θ: -σ̇ · ∂θ/∂σ  (using the selected scheme)
    adv_theta = vertical_advection(theta, sigma_dot, sigma_coord, scheme=scheme)

    # Convert back: tendency_T = (p/p₀)^κ · adv_θ
    exner = (p_full / P_0) ** kappa  # (p/p₀)^κ
    return exner * adv_theta


def compute_pressure_velocity(
    sigma_dot: jax.Array,
    p_s: jax.Array,
    dp_s_dt: jax.Array,
    sigma_coord: SigmaCoordinate,
) -> jax.Array:
    """Compute pressure velocity omega (ω = dp/dt) at full levels.

    ω = σ · (dp_s/dt) + p_s · σ̇

    This is needed for the adiabatic heating term κ·T·ω/p.

    Parameters
    ----------
    sigma_dot : jax.Array
        Sigma-dot at interfaces, shape (6, n, n, nlev+1).
    p_s : jax.Array
        Surface pressure, shape (6, n, n).
    dp_s_dt : jax.Array
        Surface pressure tendency, shape (6, n, n).
    sigma_coord : SigmaCoordinate
        Vertical coordinate.

    Returns
    -------
    jax.Array
        Pressure velocity (omega) at full levels, shape (6, n, n, nlev).
    """
    sigma_full = sigma_coord.sigma_full  # (nlev,)

    # Average sigma_dot to full levels
    sigma_dot_full = 0.5 * (sigma_dot[..., :-1] + sigma_dot[..., 1:])  # (6,n,n,nlev)

    # ω = σ · dp_s/dt + p_s · σ̇
    omega = (
        sigma_full * dp_s_dt[..., None]
        + p_s[..., None] * sigma_dot_full
    )

    return omega


# ==============================================================================
# Hybrid sigma-pressure vertical coordinate
# ==============================================================================


class HybridSigmaPressureCoordinate(NamedTuple):
    """Hybrid sigma-pressure vertical coordinate.

    Pressure at level k: p(k) = A(k) * p_ref + B(k) * p_s

    Near the model top B → 0 (pure pressure levels, no terrain influence).
    Near the surface A → 0, B → 1 (terrain-following sigma coordinate).
    This eliminates the large pressure-gradient force errors that pure
    sigma coordinates produce over steep topography.

    Levels are indexed top-to-bottom: k=0 is the model top, k=nlev-1
    is the lowest level.

    Fields
    ------
    n_levels : int
        Number of full levels.
    A_half : jax.Array
        Interface A coefficients, shape (nlev+1,). A_half[0] = p_top/p_ref.
    B_half : jax.Array
        Interface B coefficients, shape (nlev+1,). B_half[0] = 0 (top).
    A_full : jax.Array
        Mid-level A coefficients, shape (nlev,).
    B_full : jax.Array
        Mid-level B coefficients, shape (nlev,).
    p_ref : float
        Reference surface pressure [Pa], typically 1e5.
    dA : jax.Array
        A_half[k+1] - A_half[k], shape (nlev,).
    dB : jax.Array
        B_half[k+1] - B_half[k], shape (nlev,).
    B_range : float
        B_half[-1] - B_half[0], typically 1.0.
    ln_ratio_ref : jax.Array
        Precomputed log(p_half[k+1]/p_half[k]) at p_s = p_ref, shape (nlev,).
    alpha_ref : jax.Array
        Simmons-Burridge alpha at p_s = p_ref, shape (nlev,).
    dsigma_eff : jax.Array
        Effective layer thickness dA + dB for semi-implicit, shape (nlev,).
    """
    n_levels: int
    A_half: jax.Array
    B_half: jax.Array
    A_full: jax.Array
    B_full: jax.Array
    p_ref: float
    dA: jax.Array
    dB: jax.Array
    B_range: float
    ln_ratio_ref: jax.Array
    alpha_ref: jax.Array
    dsigma_eff: jax.Array

    # ------------------------------------------------------------------
    # VerticalCoordProtocol methods
    # ------------------------------------------------------------------

    def pressure_at_full(self, p_s):
        return pressure_from_hybrid(self, p_s, full=True)

    def pressure_at_half(self, p_s):
        return pressure_from_hybrid(self, p_s, full=False)

    def layer_thickness_dp(self, p_s):
        return dp_from_hybrid(self, p_s)

    # ------------------------------------------------------------------
    # Sigma-compatibility views
    # ------------------------------------------------------------------
    @property
    def sigma_half(self):
        """Effective interface sigma coefficients A_half + B_half.

        This preserves compatibility with utilities that only require a
        monotonic sigma-like coordinate shape and do not assume pure-sigma
        pressure dependence.
        """
        return self.A_half + self.B_half

    @property
    def sigma_full(self):
        """Effective full-level sigma coefficients A_full + B_full."""
        return self.A_full + self.B_full

    @property
    def dsigma(self):
        """Effective layer thickness coefficients dA + dB."""
        return self.dsigma_eff

    @property
    def dsigma_full(self):
        """Difference between full-level sigma coefficients."""
        return jnp.diff(self.sigma_full)

    @property
    def ln_ratio(self):
        """Log ratio ln(p_half[k+1]/p_half[k]) at reference pressure.

        Provides SigmaCoordinate-compatible access for routines like
        ``compute_geopotential`` that expect a ``ln_ratio`` attribute.
        """
        return self.ln_ratio_ref

    @property
    def alpha(self):
        """Simmons-Burridge alpha coefficient at reference pressure.

        Provides SigmaCoordinate-compatible access for routines like
        ``compute_geopotential`` that expect an ``alpha`` attribute.
        """
        return self.alpha_ref

    @property
    def fractional_sigma(self):
        """Fractional sigma for sigma-dot computation.

        fractional_sigma[k] = (sigma_half[k+1] - sigma_top) / (1 - sigma_top)

        This is the same quantity precomputed in SigmaCoordinate and
        required by ``compute_sigma_dot``.
        """
        s_half = self.sigma_half  # A_half + B_half
        sigma_top = s_half[0]
        sigma_range = s_half[-1] - sigma_top
        sigma_range = jnp.where(sigma_range > 0, sigma_range, 1.0)
        return (s_half[1:] - sigma_top) / sigma_range


def create_hybrid_coordinate(
    n_levels: int,
    A_half: jax.Array,
    B_half: jax.Array,
    p_ref: float = constants.p_ref,
    dtype=None,
) -> HybridSigmaPressureCoordinate:
    """Create a hybrid sigma-pressure coordinate from A/B coefficients.

    Parameters
    ----------
    n_levels : int
        Number of full levels.
    A_half : array-like
        Interface A coefficients, shape (nlev+1,).
    B_half : array-like
        Interface B coefficients, shape (nlev+1,).
    p_ref : float
        Reference surface pressure [Pa].
    dtype : jnp.dtype or None
        Dtype for coordinate arrays. If None, uses the precision
        policy's compute dtype (defaults to float32 when no policy
        is active).

    Returns
    -------
    HybridSigmaPressureCoordinate
    """
    if dtype is None:
        try:
            from legoesm.core.precision import get_policy
            dtype = get_policy().compute
        except Exception:
            dtype = jnp.float32
    A_half = jnp.asarray(A_half, dtype=dtype)
    B_half = jnp.asarray(B_half, dtype=dtype)

    A_full = 0.5 * (A_half[:-1] + A_half[1:])
    B_full = 0.5 * (B_half[:-1] + B_half[1:])
    dA = A_half[1:] - A_half[:-1]
    dB = B_half[1:] - B_half[:-1]
    B_range = float(B_half[-1] - B_half[0])

    # Reference pressures at p_s = p_ref
    p_half_ref = (A_half + B_half) * p_ref
    p_full_ref = (A_full + B_full) * p_ref
    p_half_ref_safe = jnp.clip(p_half_ref, 1e-10, None)
    jnp.clip(p_full_ref, 1e-10, None)

    ln_ratio_ref = jnp.log(p_half_ref_safe[1:] / p_half_ref_safe[:-1])
    # Exact Simmons-Burridge (1981) alpha:
    dP_ref = p_half_ref_safe[1:] - p_half_ref_safe[:-1]
    alpha_ref = 1.0 - (p_half_ref_safe[:-1] / dP_ref) * ln_ratio_ref

    dsigma_eff = dA + dB

    return HybridSigmaPressureCoordinate(
        n_levels=n_levels,
        A_half=A_half,
        B_half=B_half,
        A_full=A_full,
        B_full=B_full,
        p_ref=p_ref,
        dA=dA,
        dB=dB,
        B_range=B_range,
        ln_ratio_ref=ln_ratio_ref,
        alpha_ref=alpha_ref,
        dsigma_eff=dsigma_eff,
    )


def hybrid_min_valid_surface_pressure(
    A_half, B_half, p_ref: float = constants.p_ref,
) -> float:
    """Lowest surface pressure [Pa] at which every layer still has ``dp > 0``.

    ``dp_k = dA_k p_ref + dB_k p_s`` is LINEAR in ``p_s``, so a layer with
    ``dA_k < 0`` (which the near-surface layers of a hybrid grid always have,
    since ``A`` must return to 0 at the ground) collapses and then INVERTS once
    ``p_s`` drops below ``-dA_k p_ref / dB_k``.  The binding layer is the one
    with the largest such ratio.

    Below the returned pressure the coordinate hands the dycore NEGATIVE layer
    mass -- not a diagnostic nuisance: ``dp_from_hybrid`` feeds
    ``primitive_eq_latlon_cgrid``, ``primitive_eq_cdgrid``, ``spectral_pe`` and
    ``primitive_eq_mpas``.

    Returns 0.0 when no layer can invert (e.g. ``dA >= 0`` everywhere).

    Raises
    ------
    ValueError
        If a layer is invalid at EVERY surface pressure rather than below some
        threshold: ``dB < 0`` (non-monotone B), or ``dA <= 0`` with ``dB == 0``
        (``dp = dA p_ref <= 0`` regardless of ``p_s``).  Returning a finite
        "safe" pressure for those would be a false all-clear.
    """
    import numpy as np

    dA = np.diff(np.asarray(A_half, dtype=np.float64))
    dB = np.diff(np.asarray(B_half, dtype=np.float64))

    if np.any(dB < 0.0):
        raise ValueError(
            f"hybrid B_half must be non-decreasing; got {int((dB < 0).sum())} "
            "layer(s) with dB < 0 (dp would depend on p_s with the wrong sign)"
        )
    degenerate = (dB == 0.0) & (dA <= 0.0)
    if np.any(degenerate):
        raise ValueError(
            f"{int(degenerate.sum())} hybrid layer(s) have dB == 0 and "
            "dA <= 0, so dp = dA*p_ref <= 0 at EVERY surface pressure -- the "
            "grid is invalid, not merely limited to high p_s"
        )

    bad = (dA < 0.0) & (dB > 0.0)
    if not bad.any():
        return 0.0
    return float(np.max(-dA[bad] * p_ref / dB[bad]))


def make_hybrid_levels(
    n_levels: int,
    p_top_Pa: float = 200.0,
    p_ref: float = constants.p_ref,
    transition_exponent: int = 3,
    stretching: float = 0.0,
    p_s_min_Pa: float | None = None,
) -> HybridSigmaPressureCoordinate:
    """Generate hybrid coordinate with smooth sigma-to-pressure transition.

    Uses the parameterization:
        eta = stretched(linspace(0, 1, nlev+1))
        B = eta^exponent
        A = eta - B + (p_top/p_ref) * (1 - eta)

    At reference p_s = p_ref and zero stretching, levels are uniformly
    spaced in pressure. The exponent controls the sigma-to-pressure
    transition height. The stretching parameter concentrates levels near
    the surface for boundary layer resolution.

    Parameters
    ----------
    n_levels : int
        Number of full levels.
    p_top_Pa : float
        Model top pressure [Pa]. Default 200 Pa (2 hPa).
    p_ref : float
        Reference surface pressure [Pa].
    transition_exponent : int
        Power for B(eta) = eta^exponent. 1 = pure sigma, 3 = typical.
    stretching : float
        Sinh-based stretching parameter (>= 0). 0 = uniform spacing,
        2-3 = enhanced boundary layer resolution. The stretching maps
        eta -> sinh(s*eta)/sinh(s), concentrating levels near eta=1
        (the surface).
    p_s_min_Pa : float, optional
        Lowest surface pressure this grid must remain valid at.  When given,
        a coordinate that would produce NEGATIVE layer mass at that pressure
        is a hard error instead of silent garbage.  Pass the minimum ``p_s``
        the orography actually produces.

    Raises
    ------
    ValueError
        If ``p_s_min_Pa`` is given and the generated levels invert above it.

    Notes
    -----
    With the default ``B = eta**3`` the near-surface ``dB/deta -> 3``, so
    ``dp > 0`` needs ``p_s > (2 p_ref + p_top)/3 ~= 667 hPa``.  Real orography
    goes well below that: a 2.5-degree AMIP run reaches ``p_s = 543 hPa`` over
    the Tibetan Plateau, with 0.91% of global area under the threshold.  A
    warning naming the threshold is emitted whenever it exceeds 600 hPa, which
    the default configuration does -- see
    :func:`hybrid_min_valid_surface_pressure`.

    Returns
    -------
    HybridSigmaPressureCoordinate
    """
    import numpy as np

    eta = np.linspace(0.0, 1.0, n_levels + 1)
    if stretching > 0:
        # Concentrate levels near eta=1 (surface) for BL resolution
        eta = 1.0 - np.sinh(stretching * (1.0 - eta)) / np.sinh(stretching)
    B_half = eta ** transition_exponent
    A_half = eta - B_half + (p_top_Pa / p_ref) * (1.0 - eta)

    # Validity gate: below this surface pressure the near-surface layers carry
    # NEGATIVE mass, and dp_from_hybrid feeds the dycore, not just diagnostics.
    p_s_min_valid = hybrid_min_valid_surface_pressure(A_half, B_half, p_ref)
    if p_s_min_Pa is not None and p_s_min_valid >= p_s_min_Pa:
        raise ValueError(
            f"hybrid levels invert (dp <= 0) below p_s = "
            f"{p_s_min_valid / 100.0:.1f} hPa, but p_s_min_Pa requires validity "
            f"down to {p_s_min_Pa / 100.0:.1f} hPa. Lower transition_exponent "
            f"(currently {transition_exponent}) or raise p_s_min_Pa."
        )
    if p_s_min_valid > 6.0e4:
        logger.warning(
            "hybrid levels (n=%d, exponent=%d, stretching=%.1f) carry NEGATIVE "
            "layer mass for p_s < %.1f hPa; real orography reaches ~543 hPa "
            "over Tibet (~0.9%% of global area). Pass p_s_min_Pa to make this "
            "a hard error, or use vertical_coord='sigma'.",
            n_levels, transition_exponent, stretching, p_s_min_valid / 100.0,
        )

    return create_hybrid_coordinate(n_levels, A_half, B_half, p_ref)


# --- CAM6 L32 hybrid interface coefficients (CESM2.1 cam_vcoords_L32_c180105.nc) ---
# p_half[k] = A_half[k] * P0 + B_half[k] * p_s with P0 = 1e5 Pa, top at 2.255 hPa
# (CAM6 / CESM2 default 32-level grid, Danabasoglu et al. 2020).  Layer mass
# is positive for every p_s >= 500 hPa (min dp = 277.6 Pa, independent of
# p_s in the pure-pressure top), unlike make_hybrid_levels' analytic A(eta),
# which inverts below ~656 hPa.  Values are the file's float64 contents.
CAM6_L32_HYAI = (
    0.00225523952394724, 0.00503169186413288, 0.0101579474285245,
    0.0185553170740604, 0.0297346755951211, 0.0392730012536049,
    0.0471144989132881, 0.0562404990196228, 0.0668004974722862,
    0.0807014182209969, 0.0949410423636436, 0.11169321089983,
    0.131401270627975, 0.154586806893349, 0.181863352656364,
    0.17459799349308, 0.166050657629967, 0.155995160341263,
    0.14416541159153, 0.130248308181763, 0.113875567913055,
    0.0946138575673103, 0.0753444507718086, 0.0576589405536652,
    0.0427346378564835, 0.0316426791250706, 0.0252212174236774,
    0.0191967375576496, 0.0136180268600583, 0.00853108894079924,
    0.00397881818935275, 0.0, 0.0,
)
CAM6_L32_HYBI = (
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0393548272550106, 0.0856537595391273, 0.140122056007385,
    0.204201176762581, 0.279586911201477, 0.368274360895157,
    0.47261056303978, 0.576988518238068, 0.672786951065063,
    0.753628432750702, 0.813710987567902, 0.848494648933411,
    0.881127893924713, 0.911346435546875, 0.938901245594025,
    0.963559806346893, 0.985112190246582, 1.0,
)
CAM6_L32_P0 = 1.0e5


def make_cam6_l32_levels(p_ref: float = CAM6_L32_P0) -> HybridSigmaPressureCoordinate:
    """CAM6's 32-level hybrid grid from its published interface coefficients.

    ``p_ref`` must equal the table's own P0 (1e5 Pa): the A coefficients
    are defined against it, so another reference pressure would silently
    move every interface.
    """
    if p_ref != CAM6_L32_P0:
        raise ValueError(
            f"CAM6 L32 coefficients are defined against P0 = {CAM6_L32_P0} Pa; "
            f"got p_ref={p_ref}")
    A_half = jnp.asarray(CAM6_L32_HYAI, dtype=jnp.float64)
    B_half = jnp.asarray(CAM6_L32_HYBI, dtype=jnp.float64)
    return create_hybrid_coordinate(len(CAM6_L32_HYAI) - 1, A_half, B_half, p_ref)


def assert_hybrid_valid_for_surface_pressure(
    coord, p_s_min_Pa: float, *, context: str = "",
) -> None:
    """Refuse a hybrid coordinate that would carry NEGATIVE layer mass.

    ``B(eta) = eta**transition_exponent`` makes ``dB/deta -> exponent`` at the
    surface, so a near-surface layer has positive mass only while ``p_s`` stays
    above a threshold the coordinate alone fixes.  Below it ``dp_from_hybrid``
    returns negative thicknesses, and that feeds the dycore -- not a
    diagnostic.

    Why this is an error and not a warning (#1029).  The warning has existed,
    nothing passed ``p_s_min_Pa``, and the default L40 coordinate forbids
    surface pressures under 663.9 hPa, i.e. elevations above about 3450 m.
    Measured against 1-degree ETOPO that is **0.92% of the planet by area** --
    the Tibetan Plateau, the Andean altiplano, the Greenland and Antarctic
    domes -- and the figure is the same at 30, 32 and 40 levels.  On the
    idealized ``held_suarez_topo`` reproducer, TWO cells in that regime killed
    a 200-day run inside 200 steps.  A condition that lethal, silently active
    over a percent of the globe, is not something to keep warning about.

    Remedies, measured on the same probe
    (``scripts/validate/hybrid_negative_layer_mass_exposure.py``):

    * ``transition_exponent=2`` admits p_s down to ~498 hPa (~5870 m) and is
      valid over 100% of ETOPO;
    * ``vertical_coord='sigma'`` has no such threshold at all;
    * a coarser ``nlev`` does NOT help -- the threshold barely moves with it.

    Parameters
    ----------
    coord : HybridSigmaPressureCoordinate
        The coordinate to validate.  Objects without ``A_half``/``B_half``
        (sigma, the CAM table) are not hybrid in this sense and pass.
    p_s_min_Pa : float
        The lowest surface pressure this run will actually produce.
    context : str, optional
        Prepended to the message, e.g. the grid and level count, so the error
        names the run rather than only the coordinate.

    Raises
    ------
    ValueError
        If the coordinate inverts at or above ``p_s_min_Pa``.
    """
    import numpy as np

    A_half = getattr(coord, "A_half", None)
    B_half = getattr(coord, "B_half", None)
    if A_half is None or B_half is None:
        return
    if not np.isfinite(p_s_min_Pa) or p_s_min_Pa <= 0.0:
        raise ValueError(
            f"p_s_min_Pa must be a positive, finite pressure in Pa; "
            f"got {p_s_min_Pa!r}. A non-finite minimum usually means the "
            f"surface geopotential has not been built yet."
        )

    thr = float(hybrid_min_valid_surface_pressure(
        np.asarray(A_half), np.asarray(B_half), constants.p_ref))
    if thr <= 0.0 or p_s_min_Pa > thr:
        return

    # Make the threshold legible: "663.9 hPa" does not obviously read as
    # "forbids the Tibetan Plateau".
    z_thr = float(constants.R_d * 288.0 / constants.g
                  * np.log(constants.p_ref / thr))
    z_run = float(constants.R_d * 288.0 / constants.g
                  * np.log(constants.p_ref / p_s_min_Pa))
    where = f"{context}: " if context else ""
    raise ValueError(
        f"{where}this hybrid coordinate carries NEGATIVE layer mass below "
        f"p_s = {thr / 100:.1f} hPa (about {z_thr:.0f} m of orography), and "
        f"this run reaches p_s = {p_s_min_Pa / 100:.1f} hPa (about "
        f"{z_run:.0f} m). The near-surface layers invert there and the "
        f"negative thicknesses go into the dycore, not just a diagnostic "
        f"(#1029: two such cells killed a 200-day idealized run in 200 "
        f"steps). Fix by setting grid.transition_exponent=2 (valid to "
        f"~498 hPa / ~5870 m, i.e. all of ETOPO), or "
        f"grid.vertical_coord='sigma' (no threshold). Raising nlev does not "
        f"help; the threshold barely moves with it."
    )


def standard_hybrid_levels(
    n_levels: int = 40,
    p_ref: float = constants.p_ref,
    transition_exponent: int | None = None,
) -> HybridSigmaPressureCoordinate:
    """Create standard hybrid levels with good defaults for any resolution.

    Provides well-tested level designs for common configurations:

    - **L20**: p_top=1000 Pa (10 hPa), moderate stretching.
      5 BL levels, 10 troposphere, 5 stratosphere.
    - **L40** (default): p_top=200 Pa (2 hPa), enhanced BL resolution.
      ~8 BL levels, ~16 troposphere, ~8 UTLS, ~8 stratosphere.
    - **L60**: p_top=10 Pa (0.1 hPa), high BL + stratospheric resolution.
      ~10 BL levels, ~20 troposphere, ~15 UTLS, ~15 stratosphere.
    - **Other**: automatically selects p_top and stretching based on nlev.

    Parameters
    ----------
    n_levels : int
        Number of full levels. Default 40.
    p_ref : float
        Reference surface pressure [Pa].
    transition_exponent : int, optional
        Overrides the tier's ``B = eta**exponent`` power while keeping its
        ``p_top`` and stretching.  ``A + B`` does not depend on it, so the
        reference-state level pressures are unchanged; only how strongly the
        levels follow terrain moves.  ``None`` keeps the tier's value.

    Returns
    -------
    HybridSigmaPressureCoordinate
    """
    if n_levels <= 25:
        # Moderate resolution: 10 hPa top, mild stretching
        p_top = 1000.0
        stretching = 1.5
        tier_exponent = 2
    elif n_levels <= 45:
        # Standard resolution: 2 hPa top, good BL resolution
        p_top = 200.0
        stretching = 2.5
        tier_exponent = 3
    else:
        # High resolution: 0.1 hPa top, strong BL + strat resolution
        p_top = 10.0
        stretching = 2.5
        tier_exponent = 3

    return make_hybrid_levels(
        n_levels, p_top_Pa=p_top, p_ref=p_ref,
        transition_exponent=(tier_exponent if transition_exponent is None
                             else transition_exponent),
        stretching=stretching,
    )


def hybrid_from_sigma(
    sigma_coord: SigmaCoordinate,
    p_ref: float = constants.p_ref,
) -> HybridSigmaPressureCoordinate:
    """Convert a SigmaCoordinate to hybrid form (A=0, B=sigma).

    Useful for validation: a pure-sigma hybrid should give identical
    results to the original SigmaCoordinate.

    Parameters
    ----------
    sigma_coord : SigmaCoordinate
        Existing sigma coordinate.
    p_ref : float
        Reference surface pressure [Pa].

    Returns
    -------
    HybridSigmaPressureCoordinate
    """
    import numpy as np

    A_half = np.zeros(sigma_coord.n_levels + 1)
    B_half = np.array(sigma_coord.sigma_half)

    return create_hybrid_coordinate(sigma_coord.n_levels, A_half, B_half, p_ref)


def pressure_from_hybrid(
    coord: HybridSigmaPressureCoordinate,
    p_s: jax.Array,
    full: bool = True,
) -> jax.Array:
    """Compute pressure at full or half levels from hybrid coordinate.

    p = A * p_ref + B * p_s

    Parameters
    ----------
    coord : HybridSigmaPressureCoordinate
    p_s : jax.Array
        Surface pressure, shape (...,).
    full : bool
        If True, return full-level pressures; if False, half-level.

    Returns
    -------
    jax.Array
        Pressure, shape (..., nlev) or (..., nlev+1).
    """
    A = coord.A_full if full else coord.A_half
    B = coord.B_full if full else coord.B_half
    return A * coord.p_ref + B * p_s[..., None]


def dp_from_hybrid(
    coord: HybridSigmaPressureCoordinate,
    p_s: jax.Array,
) -> jax.Array:
    """Compute layer pressure thickness from hybrid coordinate.

    dp_k = dA_k * p_ref + dB_k * p_s

    Parameters
    ----------
    coord : HybridSigmaPressureCoordinate
    p_s : jax.Array
        Surface pressure, shape (...,).

    Returns
    -------
    jax.Array
        Layer pressure thickness, shape (..., nlev).
    """
    return coord.dA * coord.p_ref + coord.dB * p_s[..., None]


def mount_waves(
    km: int, pint: float = 300.0e2,
) -> tuple[jax.Array, jax.Array, jax.Array, int, jax.Array]:
    """FV3_3D iter 643: HIWPP mountain-wave hybrid-coord init.

    Faithful JAX port of FV3 ``mount_waves``
    (tools/fv_eta.F90:2346-2479, ``NO_UKMO_HB`` branch).  Builds
    hybrid (ak, bk) coords for the HIWPP mountain-wave test:

    Algorithm:
        Bottom 20 layers:  dz = 500 m (250 m if km > 60)
        Middle layers:     dz unchanged (s_fac = 1.0)
        Top 2 layers:      ze[2] = ze[3] + √2·dz; ze[1] = ze[2] + 2·dz
        dlnp[k] = g·dz[k] / (R_d·T0)   (isothermal hydrostatic)
        pe[k] = p00·exp(Σ above)
        ptop = pe[0]
        ks = max k where pint < pe[k]
        Pure pressure: k ≤ ks+1 → ak[k] = pe[k], bk[k] = 0
        Hybrid sigma:  k > ks+1 → bk[k] = (pe[k] - pint)/(pe[km] - pint)
                                  ak[k] = pe[k] - bk[k]·pe[km]
        bk[km] = 1; ak[km] = 0

    Parameters
    ----------
    km : int
        Number of layers (FV3 expects km > 22).
    pint : float, default 30000 Pa (300 hPa)
        Pure-pressure / sigma transition pressure.

    Returns
    -------
    ak, bk : jax.Array, shape (km+1,)
        Hybrid coordinates.
    ptop : jax.Array (scalar)
        Top-of-model pressure (Pa).
    ks : int
        Number of pure-pressure layers.
    pint_out : jax.Array (scalar)
        Adjusted transition pressure pe[ks+1].
    """
    if km < 23:
        raise ValueError(f"mount_waves requires km >= 23, got {km}")
    g = constants.g
    rdgas = constants.R_d
    p00 = constants.p_ref
    t0 = 300.0

    dz0 = 500.0 if km <= 60 else 250.0
    s_fac = 1.0

    # Build ze bottom-up.  FV3 1-indexed: ze[km+1]=0; bottom 20 (k=km..km-19)
    # uniform dz0; middle k=km-20..3 stretching by s_fac; top k=2,1 special.
    # 0-indexed equivalent (ze shape (km+1,), ze[km] = 0):
    ze = jnp.zeros((km + 1,))
    # Bottom 20: ze[km-1..km-20] (0-indexed) ascending uniformly by dz0
    bot_dz = dz0
    # ze[k] = ze[k+1] + dz0 for k = km-1, km-2, ..., km-20 (0-indexed)
    # In FV3: k = km, km-1, ..., km-19 (1-indexed) → 0-indexed k = km-1..km-20
    for i in range(20):
        ze = ze.at[km - 1 - i].set(ze[km - i] + bot_dz)
    # Middle: FV3 k = km-20 down to 3 (1-indexed); 0-indexed k = km-21..2
    # FV3 line 2389-2391: dz0 = s_fac * dz0; ze[k] = ze[k+1] + dz0
    # So dz0 grows by s_fac each iteration.  With s_fac=1.0, dz0 stays constant.
    cur_dz = bot_dz
    for k_python in range(km - 21, 1, -1):    # 0-indexed; stops at k_python=2 (FV3 k=3)
        cur_dz = s_fac * cur_dz
        ze = ze.at[k_python].set(ze[k_python + 1] + cur_dz)
    # Top: ze[1] (1-indexed 2) = ze[2] + √2·dz0; ze[0] (1-indexed 1) = ze[1] + 2·dz0
    ze = ze.at[1].set(ze[2] + jnp.sqrt(2.0) * cur_dz)
    ze = ze.at[0].set(ze[1] + 2.0 * cur_dz)

    # Compute dz and dlnp
    dz = ze[:-1] - ze[1:]                    # 0-indexed (km,); top→bottom
    dlnp = g * dz / (rdgas * t0)
    # pe1 from p00 (surface): peln[km] = log(p00); peln[k] = peln[k+1] - dlnp[k]
    peln = jnp.zeros((km + 1,))
    peln = peln.at[km].set(jnp.log(p00))
    # Build top-down via cumulative subtraction
    # peln[k] = log(p00) - Σ_{j=k}^{km-1} dlnp[j]
    cumsum_rev = jnp.cumsum(dlnp[::-1])[::-1]   # = Σ dlnp from k..km-1
    peln = peln.at[:km].set(jnp.log(p00) - cumsum_rev)
    pe1 = jnp.exp(peln)

    ptop = pe1[0]

    # Find ks (FV3 1-indexed: ks = k - 1 where pint < pe[k], first match)
    # 0-indexed: ks = j where pint < pe1[j+1] (j+1 is FV3 k); want smallest j.
    # FV3 loop: k=2..km; ks = 0 if no match.  In 0-indexed: scan pe1[1..km].
    pint_arr = jnp.asarray(pint)
    # Find first k_fortran >= 2 with pint < pe1[k_fortran].  0-indexed k_python = k_fortran - 1 >= 1.
    cond = pint_arr < pe1[1:]                  # pe1[1..km] in 0-indexed
    # If no True, ks = 0 (FV3 default)
    first_true = jnp.argmax(cond)
    any_true = jnp.any(cond)
    jnp.where(any_true, first_true, 0)  # 0-indexed: this is k_python - 1
    # FV3: ks = k_fortran - 1 = (k_python + 1) - 1 = k_python
    # k_python where condition is met = first_true + 1 (because we sliced from index 1)
    # Actually we used pe1[1:] so first_true index 0 corresponds to k_python=1 (FV3 k=2);
    # FV3 ks = k - 1 = k_python = first_true + 1.  Hmm.  Let me recompute.
    # If cond[i] = (pint < pe1[i+1]) for i=0..km-1, and FV3 detects at k_fortran = i+2
    # (where pe1[i+1] in 0-indexed corresponds to pe1(k_fortran) with k_fortran = i+2 in 1-indexed).
    # Then FV3 ks = k_fortran - 1 = i + 1.  So ks = first_true + 1 (when any_true).
    ks = int(jnp.where(any_true, first_true + 1, 0))
    pint_out = pe1[ks + 1] if ks + 1 <= km else pe1[km]

    # Build ak, bk (NO_UKMO_HB branch)
    ak = jnp.zeros((km + 1,))
    bk = jnp.zeros((km + 1,))
    # Pure-pressure: k ∈ [0, ks] (0-indexed; FV3 k=1..ks+1)
    ak = ak.at[:ks + 1].set(pe1[:ks + 1])
    # bk already zero
    # Hybrid: k ∈ [ks+1, km-1]; ak[km] = 0, bk[km] = 1
    if ks + 1 < km + 1:
        pe_k = pe1[ks + 1:km + 1]
        pint_safe = jnp.where(pe1[km] != pint_out, pe1[km] - pint_out, 1.0)
        bk_int = (pe_k - pint_out) / pint_safe
        bk = bk.at[ks + 1:km + 1].set(bk_int)
        ak_int = pe_k - bk_int * pe1[km]
        ak = ak.at[ks + 1:km + 1].set(ak_int)
        # Force bottom: bk[km] = 1, ak[km] = 0
        ak = ak.at[km].set(0.0)
        bk = bk.at[km].set(1.0)
    return ak, bk, ptop, ks, pint_out


def gw_1d(
    km: int, p0: float, ztop: float,
    isothermal: bool = False, t0: float = 300.0,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """FV3_3D iter 642: FV3 gravity-wave 1D vertical coord init.

    Faithful JAX port of FV3 ``gw_1d`` (tools/fv_eta.F90:2286-2344).
    Sets up a uniform-dz vertical coord with isothermal or
    constant-N² atmosphere; returns the resulting hybrid (ak, bk)
    coefficients, top-of-model pressure, and reference potential
    temperature profile.

    Algorithm:
        dz[k] = ztop / km                       # uniform
        ze[km] = 0; ze[k] = ze[k+1] + dz[k]    # bottom-up
        # If isothermal: N² = g²/(cp·T0); else N² = 0.0001 s⁻²
        s0 = g²/(cp·N²)
        pe[k] = p0·((1 - s0/T0) + s0/T0·exp(-N²·ze[k]/g))^(1/κ)
        ptop = pe[0]
        ak[0] = pe[0]; bk[0] = 0
        bk[k] = (pe[k] - pe[0]) / (pe[km] - pe[0])    for k ∈ [1, km-1]
        ak[k] = pe[0] · (1 - bk[k])
        ak[km] = 0; bk[km] = 1
        pk[k] = pe[k]^κ
        pt[k] = g·dz[k] / (cp · (pk[k+1] - pk[k]))

    Parameters
    ----------
    km : int
        Number of layers.
    p0 : float
        Reference surface pressure (Pa).
    ztop : float
        Top of model height (m).
    isothermal : bool, default False
        If True, use N² = g²/(cp·T0); else N² = 0.0001.
    t0 : float, default 300.0
        Reference temperature (K).

    Returns
    -------
    ak : jax.Array, shape (km+1,)
    bk : jax.Array, shape (km+1,)
    ptop : jax.Array (scalar)
    pt1 : jax.Array, shape (km,)
        Volume-mean potential temperature.
    """
    g = constants.g
    cp = constants.c_pd
    kappa = constants.kappa

    if isothermal:
        n2 = g * g / (cp * t0)
    else:
        n2 = 0.0001

    s0 = g * g / (cp * n2)

    # Uniform dz; ze built bottom-up: ze[km] = 0, ze[k] = ze[k+1] + dz
    dz_uniform = ztop / km
    dz1 = jnp.full((km,), dz_uniform)
    # ze[k] = (km - k) · dz_uniform (0-indexed, ze[0] = ztop, ze[km] = 0)
    ze = (km - jnp.arange(km + 1)) * dz_uniform

    # pe[k] = p0·((1 - s0/T0) + s0/T0·exp(-N²·ze[k]/g))^(1/κ)
    base = (1.0 - s0 / t0) + (s0 / t0) * jnp.exp(-n2 * ze / g)
    pe1 = p0 * base ** (1.0 / kappa)

    ptop = pe1[0]

    # ak, bk build
    ak = jnp.zeros((km + 1,))
    bk = jnp.zeros((km + 1,))
    ak = ak.at[0].set(pe1[0])
    # Interior k ∈ [1, km-1] (0-indexed)
    bk_int = (pe1[1:km] - pe1[0]) / (pe1[km] - pe1[0])
    bk = bk.at[1:km].set(bk_int)
    ak_int = pe1[0] * (1.0 - bk_int)
    ak = ak.at[1:km].set(ak_int)
    # Bottom k = km
    ak = ak.at[km].set(0.0)
    bk = bk.at[km].set(1.0)

    # pk and pt1
    pk1 = pe1 ** kappa
    pt1 = g * dz1 / (cp * (pk1[1:] - pk1[:-1]))
    return ak, bk, ptop, pt1


def compute_dz_var(
    km: int, ztop: float, s_rate: float = 1.0,
) -> jax.Array:
    """FV3_3D iter 641: variable dz with stretch rescaling.

    Faithful JAX port of FV3 ``compute_dz_var``
    (tools/fv_eta.F90:1930-1998).  Similar to iter-639
    ``hybrid_z_dz`` but with::

        - s_fac[km] = 0.125 (vs 0.12 in hybrid_z_dz)
        - middle layers: s_fac[k] = s_rate · s_fac[k+1] (no min-cap)
        - rescale dz so ze[0] = ztop exactly (FV3 lines 1981-1983)
        - sm1_edge with ntimes=2

    Default ``s_rate = 1.0`` gives uniform middle layers.  Top 8
    layers use the same FV3 multipliers as iter-639 (1.05 → 1.6).

    Requires ``km >= 18``.

    Parameters
    ----------
    km : int
        Number of layers (must be ≥ 18).
    ztop : float
        Top of model (m).
    s_rate : float, default 1.0
        Middle-layer stretch rate.

    Returns
    -------
    dz : jax.Array, shape ``(km,)``
        Layer thicknesses (top→bottom indexing).
    """
    if km < 18:
        raise ValueError(f"compute_dz_var requires km >= 18, got {km}")

    # Build s_fac (FV3 1-indexed → 0-indexed)
    s_fac = jnp.zeros((km,))
    bottom = jnp.asarray(
        [0.125, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
    )
    for i, v in enumerate(bottom):
        s_fac = s_fac.at[km - 1 - i].set(float(v))
    # Middle (no cap)
    for k in range(km - 11, 7, -1):
        s_fac = s_fac.at[k].set(s_rate * s_fac[k + 1])
    # Top 8 (same as hybrid_z_dz)
    top_mults = [1.05, 1.10, 1.15, 1.20, 1.30, 1.40, 1.50, 1.60]
    for i, mult in enumerate(top_mults):
        idx = 7 - i
        s_fac = s_fac.at[idx].set(mult * s_fac[idx + 1])

    sum1 = jnp.sum(s_fac)
    dz0 = ztop / sum1
    dz = s_fac * dz0

    # FV3 lines 1976-1980: ze[0]=ztop, ze[km]=0; rebuild bottom-up
    # ze[k] = ze[k+1] + dz[k] for k = km-1..1 (0-indexed); ze[0] = ztop
    cumsum_rev = jnp.cumsum(dz[::-1])[::-1]
    jnp.concatenate([cumsum_rev, jnp.asarray([0.0])])
    # FV3 line 1976 sets ze(1) = ztop AFTER building ze from dz; this
    # may not match the dz sum exactly.  Then FV3 rescales dz:
    #   dz(k) = dz(k) * (ztop/ze(1))   FV3 line 1983
    # where ze(1) here is the *unmodified* sum (= cumsum_rev[0]).
    actual_top = cumsum_rev[0]
    dz_rescaled = dz * (ztop / actual_top)

    # Rebuild ze from rescaled dz, set ze[0] = ztop
    cumsum_rev2 = jnp.cumsum(dz_rescaled[::-1])[::-1]
    ze = jnp.concatenate([cumsum_rev2, jnp.asarray([0.0])])
    ze = ze.at[0].set(ztop)

    # Apply sm1_edge with ntimes=2 (iter 636)
    ze = sm1_edge_fv3(ze, ntimes=2)

    # Recompute dz from ze
    dz_final = ze[:-1] - ze[1:]
    return dz_final


def hybrid_z_dz(
    km: int, ztop: float, s_rate: float = 1.06,
) -> jax.Array:
    """FV3_3D iter 639: FV3 hybrid-z layer thicknesses with s_rate stretch.

    Faithful JAX port of FV3 ``hybrid_z_dz``
    (tools/fv_eta.F90:1794-1855).  Builds an FV3-style stretched
    vertical profile using a per-layer stretch factor table::

        s_fac[km..km-9] = 0.12, 0.20, 0.30, ..., 1.0
        s_fac[k]        = min(4, s_rate · s_fac[k+1])   for k ∈ [9, km-10]
        s_fac[1..8]     = 1.6, 1.5, 1.4, 1.3, 1.2, 1.15, 1.1, 1.05
                          (top, applied to s_fac[k+1])

        dz0 = ztop / Σ s_fac
        dz[k] = s_fac[k] · dz0

    Then iter-636 ``sm1_edge_fv3`` is applied with ntimes=2.

    Requires ``km >= 18`` (so bottom 10 + top 8 don't overlap).

    Parameters
    ----------
    km : int
        Number of layers (must be ≥ 18).
    ztop : float
        Top of model (m).
    s_rate : float, default 1.06
        Stretch rate for middle layers, FV3 documented range
        [1.0, 1.1].

    Returns
    -------
    dz : jax.Array, shape ``(km,)``
        Layer thicknesses (top→bottom, FV3 final convention).
    """
    if km < 18:
        raise ValueError(f"hybrid_z_dz requires km >= 18, got {km}")

    # Build s_fac 0-indexed (FV3 1-indexed s_fac[k] → s_fac[k-1])
    s_fac = jnp.zeros((km,))
    # Bottom 10 (FV3 k=km..km-9 → 0-indexed [km-1, km-10])
    bottom = jnp.asarray(
        [0.12, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
    )
    # s_fac[km-1] = 0.12, s_fac[km-2] = 0.20, ..., s_fac[km-10] = 1.0
    for i, v in enumerate(bottom):
        s_fac = s_fac.at[km - 1 - i].set(float(v))
    # Middle (FV3 k=km-10..9 → 0-indexed [km-11, 8]): recurrence
    # s_fac[k] = min(4, s_rate · s_fac[k+1])
    # Sequential — use Python loop (km is static)
    for k in range(km - 11, 7, -1):  # 0-indexed: k_python = k_fortran - 1
        s_fac = s_fac.at[k].set(
            jnp.minimum(4.0, s_rate * s_fac[k + 1])
        )
    # Top 8 (FV3 k=8..1 → 0-indexed [7, 0]): specific multipliers
    top_mults = [1.05, 1.10, 1.15, 1.20, 1.30, 1.40, 1.50, 1.60]
    # FV3: s_fac(8) = 1.05·s_fac(9); s_fac(7) = 1.10·s_fac(8); ...
    # 0-indexed: s_fac[7] = 1.05·s_fac[8]; s_fac[6] = 1.10·s_fac[7]; ...
    for i, mult in enumerate(top_mults):
        idx = 7 - i  # 7, 6, 5, ..., 0
        s_fac = s_fac.at[idx].set(mult * s_fac[idx + 1])

    sum1 = jnp.sum(s_fac)
    dz0 = ztop / sum1
    dz = s_fac * dz0

    # Build ze top-down: FV3 ze[km+1]=0, then ze[k]=ze[k+1]+dz[k]
    # 0-indexed: ze[km]=0, ze[k]=ze[k+1]+dz[k] for k = km-1..0
    cumsum_rev = jnp.cumsum(dz[::-1])[::-1]
    # cumsum_rev[k] = dz[k]+dz[k+1]+...+dz[km-1]
    ze = jnp.concatenate([cumsum_rev, jnp.asarray([0.0])])
    # FV3 also sets ze(1) = ztop (FV3 line 1846 overrides top edge)
    ze = ze.at[0].set(ztop)

    # Apply iter-636 sm1_edge_fv3 with ntimes=2
    ze = sm1_edge_fv3(ze, ntimes=2)

    # Recompute dz from ze (FV3 line 1851-1853: dz(k) = ze(k) - ze(k+1))
    dz_final = ze[:-1] - ze[1:]
    return dz_final


def sm1_edge_fv3(
    ze: jax.Array, ntimes: int,
) -> jax.Array:
    """FV3_3D iter 636: 1D del-2 edge smoother on layer interfaces.

    Faithful JAX port of FV3 ``sm1_edge`` (tools/fv_eta.F90:
    2249-2284).  Smooths a column of layer-interface heights
    ``ze`` (shape ``(km+1,)``) via iterated del-2 flux on the
    layer thicknesses.

    Algorithm:
        dz[k] = ze[k+1] - ze[k]                    # thickness
        For n in 1..ntimes:
            k1 = 2 + (ntimes - n)                  # iteration shrinks top
            flux[k1] = flux[km] = 0                # boundary
            flux[k] = 0.25 · (dz[k] - dz[k-1])    # interior
            dz[k] += flux[k+1] - flux[k]
        ze rebuilt from dz, bottom-up.

    Used in FV3 hybrid-z setup to smooth oscillations at the
    top of the vertical-coordinate generation.

    Parameters
    ----------
    ze : jax.Array, shape ``(km+1,)``
        Layer interface heights (top-down indexing).
    ntimes : int
        Number of smoothing passes.

    Returns
    -------
    jax.Array, shape ``(km+1,)``
        Smoothed interface heights.  Bottom is preserved exactly
        (ze[km] unchanged); top adjusted by sum of flux changes.
    """
    df = 0.25
    km = ze.shape[0] - 1
    # Initial thicknesses dz[k] = ze[k+1] - ze[k]
    dz = ze[1:] - ze[:-1]               # (km,)
    k2 = km - 1                          # last interior thickness index

    # Iterate ntimes; each pass uses k1 = 2 + (ntimes - n) - 1 in 0-indexed
    # FV3 1-indexed: k1=2+(ntimes-n), flux range [k1+1, k2].
    # 0-indexed: k1=1+(ntimes-n), flux range [k1+1, k2] inclusive (but we
    # work on 0-indexed dz so adjust carefully).
    for n in range(1, ntimes + 1):
        k1_0idx = (ntimes - n) + 1     # 0-indexed start of smoothed region
        # flux[k] defined for k = k1+1 .. k2 (inclusive); FV3 indexing on
        # interfaces (km+1).  0-indexed: flux[k+1] - flux[k] for the dz
        # update at k1..k2 (1-indexed) which is k1_0idx..k2 (0-indexed).
        # Use 0-indexed flux of length km+1 (one per interface).
        flux = jnp.zeros((km + 1,))
        # Interior fluxes: flux[k] = df·(dz[k] - dz[k-1])  for k in [k1+1, k2]
        # In 0-indexed: flux at interface i corresponds to thickness
        # difference between dz[i] and dz[i-1].  Use indices i ∈
        # [k1_0idx+1, k2] inclusive.  Numpy-style mask.
        i_idx = jnp.arange(km + 1)
        in_range = (i_idx >= (k1_0idx + 1)) & (i_idx <= k2)
        # dz[k] - dz[k-1]: 0-indexed thicknesses at positions k, k-1
        # Build per-interface (k+1) lookup safely:
        # diff[k] = dz[k] - dz[k-1] for k in [1, km-1]; 0 elsewhere.
        # Pad dz to allow diff at boundaries safely.
        diff = jnp.zeros((km + 1,))
        diff = diff.at[1:km].set(dz[1:] - dz[:-1])
        flux_raw = df * diff
        flux = jnp.where(in_range, flux_raw, 0.0)
        # dz[k] update: dz[k] += flux[k+1] - flux[k] for k in [k1_0idx, k2]
        upd_range = (jnp.arange(km) >= k1_0idx) & (jnp.arange(km) <= k2)
        dz_upd = flux[1:km + 1] - flux[:km]
        dz = jnp.where(upd_range, dz + dz_upd, dz)

    # Rebuild ze from dz, bottom-up (ze[km] preserved; FV3 loop k=km..1)
    ze_new = jnp.zeros((km + 1,))
    ze_new = ze_new.at[km].set(ze[km])
    # ze[k] = ze[k+1] - dz[k]; loop top-down using cumulative sum
    # ze[km-1] = ze[km] - dz[km-1]; ze[km-2] = ze[km-1] - dz[km-2]; ...
    # = ze[km] - Σ_{j>=k} dz[j].  Use reverse cumsum.
    cumsum_rev = jnp.cumsum(dz[::-1])[::-1]   # cumsum from bottom up
    # cumsum_rev[k] = dz[k] + dz[k+1] + ... + dz[km-1]
    ze_new = ze_new.at[:km].set(ze[km] - cumsum_rev)
    return ze_new


def sb81_halflevel_construction(
    coord: HybridSigmaPressureCoordinate,
    p_s: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Simmons-Burridge (1981) half-level construction for hybrid coordinates.

    Single source for the (p_half_safe, ln_ratio, alpha) triple used by BOTH
    the hydrostatic geopotential integration and the momentum
    pressure-gradient correction.  Sharing the bit-identical ``alpha`` is a
    correctness requirement, not hygiene: the discrete rest-over-terrain
    cancellation of ``-grad(Phi) - R_d T grad(ln p)`` (#1029) holds only when
    the two terms difference the SAME floating-point fields.

    Parameters
    ----------
    coord : HybridSigmaPressureCoordinate
    p_s : jax.Array
        Surface pressure, shape (...,).

    Returns
    -------
    (p_half_safe, ln_ratio, alpha)
        Interface pressures clipped away from zero (..., nlev+1), layer log
        ratios ``ln(p_{k+1/2}/p_{k-1/2})`` (..., nlev), and the exact SB81
        ``alpha_k = 1 - (p_{k-1/2}/dp_k) ln_ratio_k`` (..., nlev).
    """
    p_half = pressure_from_hybrid(coord, p_s, full=False)  # (..., nlev+1)
    p_half_safe = jnp.clip(p_half, 1e-10, None)
    ln_ratio = jnp.log(p_half_safe[..., 1:] / p_half_safe[..., :-1])  # (..., nlev)
    dp = p_half_safe[..., 1:] - p_half_safe[..., :-1]
    alpha = 1.0 - (p_half_safe[..., :-1] / dp) * ln_ratio  # (..., nlev)
    return p_half_safe, ln_ratio, alpha


def as83_layer_exner(coord, p_s):
    """Arakawa-Suarez (1983) layer Exner function on a hybrid coordinate.

    ``P_{k+1/2} = (p_{k+1/2}/p_ref)^kappa`` at interfaces and
    ``pi_k = Delta_k(p P) / ((1 + kappa) Delta_k p)`` per layer, with
    ``d pi_k / d p_s = (Delta_k(P B) - pi_k Delta_k B) / Delta_k p``
    (from ``d(pP)/dp = (1 + kappa) P``).  Returns ``(P_half, pi, dpi_dps)``.
    """
    kappa = constants.kappa
    p_half = pressure_from_hybrid(coord, p_s, full=False)          # (..., nlev+1)
    P = (p_half / coord.p_ref) ** kappa
    dp = jnp.diff(p_half, axis=-1)
    pi = jnp.diff(p_half * P, axis=-1) / ((1.0 + kappa) * dp)
    B = coord.B_half.astype(p_half.dtype)
    dpi_dps = (jnp.diff(P * B, axis=-1) - pi * jnp.diff(B)) / dp
    return P, pi, dpi_dps


def as83_interface_theta(theta, P_half, pi):
    """Interface potential temperature ``theta_{k+1/2}`` (AS83 weights
    ``a_k = (P_{k+1/2} - pi_k) / (pi_{k+1} - pi_k)``), shape ``(..., nlev-1)``.
    With these weights the hydrostatic relation below makes the column total
    energy conserved exactly; uniform theta stays uniform."""
    a = (P_half[..., 1:-1] - pi[..., :-1]) / (pi[..., 1:] - pi[..., :-1])
    return a * theta[..., :-1] + (1.0 - a) * theta[..., 1:]


def as83_logmean_face_weight(pi_1, pi_2):
    """Weight ``w`` of ``theta_1`` in ``theta_f = w theta_1 + (1 - w) theta_2``
    such that ``c_p theta_f (pi_2 - pi_1) = c_p T0 ln(pi_2 / pi_1)`` for any
    isothermal pair ``theta_i = T0 / pi_i`` (log-mean face theta):
    ``w = [(1 + x) log1p(x) - x] / x^2``, ``x = pi_2/pi_1 - 1``, -> 1/2 at x = 0.
    Below ``|x| = 1e-4`` its Taylor series (agreeing to round-off) avoids 0/0."""
    x = pi_2 / pi_1 - 1.0
    small = jnp.abs(x) < 1e-4
    xs = jnp.where(small, 1.0, x)
    closed = ((1.0 + xs) * jnp.log1p(xs) - xs) / (xs * xs)
    series = 0.5 - x / 6.0 + x * x / 12.0 - x ** 3 / 20.0
    return jnp.where(small, series, closed)


def as83_geopotential(theta, theta_half, P_half, pi, phis):
    """Full-level geopotential, ``Phi_K = Phi_s + c_p theta_K (P_{K+1/2} - pi_K)``
    and ``Phi_k - Phi_{k+1} = c_p theta_{k+1/2} (pi_{k+1} - pi_k)``."""
    c_p = constants.c_pd
    phi_bottom = phis + c_p * theta[..., -1] * (P_half[..., -1] - pi[..., -1])
    inc = c_p * theta_half * (pi[..., 1:] - pi[..., :-1])           # (..., nlev-1)
    above = jnp.cumsum(inc[..., ::-1], axis=-1)[..., ::-1]           # sum_{j>=k} inc_j
    return phi_bottom[..., None] + jnp.concatenate(
        [above, jnp.zeros_like(phi_bottom)[..., None]], axis=-1)


def as83_vertical_theta_flux(theta, theta_half, mass_flux):
    """``dp_k * dtheta_k/dt`` from vertical transport:
    ``-[F_{k+1/2}(theta_{k+1/2} - theta_k) + F_{k-1/2}(theta_k - theta_{k-1/2})]``,
    ``F`` the downward interface mass flux (zero at top and bottom)."""
    F = mass_flux[..., 1:-1]                                         # interior interfaces
    pad = ((0, 0),) * (F.ndim - 1)
    lower = jnp.pad(F * (theta_half - theta[..., :-1]), (*pad, (0, 1)))
    upper = jnp.pad(F * (theta[..., 1:] - theta_half), (*pad, (1, 0)))
    return -(lower + upper)


def sb81_full_level_ln_p(
    coord: HybridSigmaPressureCoordinate,
    p_s: jax.Array,
) -> jax.Array:
    """SB81 full-level log-pressure ``ln p_k = ln p_{k+1/2} - alpha_k``.

    This is the discrete field whose horizontal gradient forms the
    energy-consistent pair with ``-grad(Phi)`` from
    :func:`compute_geopotential_hybrid`: because
    ``sum_{j>k} ln_ratio_j = ln p_s - ln p_{k+1/2}`` telescopes, at uniform
    temperature ``-grad(Phi_k) - R_d T grad(ln p_k)`` reduces to
    ``-grad(phi_s + R_d T ln p_s)``, which vanishes identically for a
    hydrostatically balanced rest state over terrain (#1029).  On a pure-sigma
    or ``A=0`` coordinate it reduces to ``ln p_s`` plus a spatially
    constant per-level offset, so its gradient equals ``grad(ln p_s)`` — the
    sigma-path correction — up to a ~1e-12 top-layer artifact of the
    ``p_half`` zero-clip when the top interface pressure is exactly 0
    (``alpha_0`` picks up a weak ``p_s`` dependence through the clipped
    ``ln`` ratio; physically nil, pinned by the A=0 unit test).

    Returns
    -------
    jax.Array
        Full-level log-pressure, shape (..., nlev).
    """
    p_half_safe, _, alpha = sb81_halflevel_construction(coord, p_s)
    return jnp.log(p_half_safe[..., 1:]) - alpha


def compute_geopotential_hybrid(
    T: jax.Array,
    p_s: jax.Array,
    coord: HybridSigmaPressureCoordinate,
    phis: jax.Array,
) -> jax.Array:
    """Compute geopotential using Simmons-Burridge for hybrid coordinates.

    Same algorithm as compute_geopotential but with pressure-dependent
    log ratios and alpha coefficients computed from the hybrid pressure.

    Parameters
    ----------
    T : jax.Array
        Temperature, shape (..., nlev).
    p_s : jax.Array
        Surface pressure, shape (...,).
    coord : HybridSigmaPressureCoordinate
    phis : jax.Array
        Surface geopotential, shape (...,).

    Returns
    -------
    jax.Array
        Geopotential at full levels, shape (..., nlev).
    """
    R_d = constants.R_d

    # Half-level construction shared with the hybrid PGF correction (#1029)
    _, ln_ratio, alpha = sb81_halflevel_construction(coord, p_s)

    # Geopotential thickness of each full layer
    dPhi = R_d * T * ln_ratio  # (..., nlev)

    # Cumulative sum from bottom: Phi_above[k] = sum from k to nlev-1
    dPhi_reversed = dPhi[..., ::-1]
    cumsum_reversed = jnp.cumsum(dPhi_reversed, axis=-1)
    cumsum = cumsum_reversed[..., ::-1]

    Phi_above = phis[..., None] + cumsum

    # Bottom interface of each layer
    Phi_below = jnp.concatenate(
        [Phi_above[..., 1:], phis[..., None]], axis=-1
    )

    Phi_full = Phi_below + alpha * R_d * T

    return Phi_full


def compute_mass_flux_from_cumsum(
    cumsum_mass_div: jax.Array,
    D_total_p: jax.Array,
    coord: HybridSigmaPressureCoordinate,
) -> jax.Array:
    """Half-level vertical mass flux from a precomputed cumulative mass-weighted divergence.

    Single-sources the hybrid continuity integration + boundary closure::

        F_{k+1/2} = (B_{k+1/2} - B_top) / B_range * D_total_p - cumsum_mass_div[k]

    with ``F = 0`` at the top and (by construction, ``frac_B[-1]=1`` and
    ``cumsum_mass_div[-1]=D_total_p``) at the surface.  Factored out of
    :func:`compute_mass_flux_hybrid` so callers that ALREADY hold
    ``cumsum(div_dp, axis=-1)`` — the C-grid dycore reuses it for ``dp_s_dt``
    (iter-54) — share this ONE drop-last + ``(1,1)`` pad closure instead of
    re-deriving the error-prone boundary handling.

    The ``div_dp`` the cumsum is taken over is the CALLER's choice — the advective
    ``div(v)*dp`` (this module's :func:`compute_mass_flux_hybrid`) or the exact
    flux-form ``div(dp*v)`` (the C-grid dycore; the MPAS dycore has it in hand as
    ``div_dp_3d_pre``) — so the flux convention stays a caller decision while the
    closure is single-sourced (CLAUDE.md: no duplicate dycore numerics).

    Parameters
    ----------
    cumsum_mass_div : jax.Array
        ``cumsum(div_dp, axis=-1)`` — cumulative mass-weighted divergence, (..., nlev).
    D_total_p : jax.Array
        Column total ``cumsum_mass_div[..., -1:]``, (..., 1) [Pa/s].
    coord : HybridSigmaPressureCoordinate

    Returns
    -------
    jax.Array
        Mass flux at half-levels, (..., nlev+1) [Pa/s]. ``F = 0`` at top + surface.
    """
    B_top = coord.B_half[0]
    frac_B = (coord.B_half[1:] - B_top) / coord.B_range  # (nlev,)
    mass_flux_inner = frac_B * D_total_p - cumsum_mass_div  # (..., nlev)
    # Drop the (∼0) surface element + pad both ends with F=0 in one Pad HLO op.
    pad_axes = ((0, 0),) * (mass_flux_inner.ndim - 1) + ((1, 1),)
    return jnp.pad(mass_flux_inner[..., :-1], pad_axes)


def compute_sigma_dot_from_cumsum(
    cumsum_mass_div: jax.Array,
    D_total_p: jax.Array,
    p_s: jax.Array,
    sigma_coord: SigmaCoordinate,
) -> jax.Array:
    """σ̇ at half-levels from a precomputed cumulative FLUX-FORM mass divergence.

    σ-coordinate analogue of :func:`compute_mass_flux_from_cumsum` —
    single-sources the flux-form continuity integration + boundary closure::

        σ̇_{k+1/2} = [frac_k · D_total_p − cumsum_k(div(dp·v))] / p_s

    where ``div(dp·v)`` is the exact flux-form layer-mass divergence
    (``dp_k = p_s · Δσ_k``) — NOT the advective ``div(v)·Δσ_k`` that
    :func:`compute_sigma_dot` integrates.  The two differ wherever
    ``∇p_s ≠ 0``; only the flux form telescopes to a globally
    mass-conserving ``dp_s/dt``.  Shared by the lat-lon C-grid, cubed-sphere
    and MPAS PE dycores (CLAUDE.md: no duplicate dycore numerics).

    Boundary closure: σ̇ = 0 at top and surface (drop the ∼0 last element,
    pad both ends — ``fractional_sigma[-1] = 1`` makes the surface element
    exact cancellation ``D_total_p − D_total_p``).

    Parameters
    ----------
    cumsum_mass_div : jax.Array
        ``cumsum(div(dp·v), axis=-1)``, shape (..., nlev) [Pa/s].
    D_total_p : jax.Array
        Column total ``cumsum_mass_div[..., -1:]``, shape (..., 1) [Pa/s].
    p_s : jax.Array
        Surface pressure, shape (...,) [Pa].
    sigma_coord : SigmaCoordinate

    Returns
    -------
    jax.Array
        σ̇ at half-levels, shape (..., nlev+1) [1/s]; 0 at top + surface.
    """
    frac = sigma_coord.fractional_sigma  # (nlev,)
    # 1e-10 Pa: division-safety floor only (p_s is clipped far above this).
    sigma_dot_inner = (frac * D_total_p - cumsum_mass_div) / (
        p_s[..., jnp.newaxis] + 1e-10
    )  # (..., nlev)
    pad_axes = ((0, 0),) * (sigma_dot_inner.ndim - 1) + ((1, 1),)
    return jnp.pad(sigma_dot_inner[..., :-1], pad_axes)


def compute_mass_flux_hybrid(
    div_3d: jax.Array,
    p_s: jax.Array,
    coord: HybridSigmaPressureCoordinate,
) -> tuple[jax.Array, jax.Array]:
    """Compute vertical mass flux at half-levels for hybrid coordinates.

    Returns ``(mass_flux, D_total_p)``:

        F_{k+1/2} = B_{k+1/2} * D_total_p - cumsum(D_k * dp_k)[k]
        D_total_p = sum(D_k * dp_k)

    where dp_k is the layer pressure thickness.  ``D_total_p`` is
    returned so the caller (e.g. ``spectral_pe_tendencies`` step 8)
    can reuse it for the surface-pressure tendency without recomputing
    the column sum — saves one cross-level collective per RK3 stage
    under spectral level-sharding.

    Boundary conditions: F = 0 at top and surface.

    Parameters
    ----------
    div_3d : jax.Array
        Horizontal divergence at each level, shape (..., nlev).
    p_s : jax.Array
        Surface pressure, shape (...,).
    coord : HybridSigmaPressureCoordinate

    Returns
    -------
    mass_flux : jax.Array
        Mass flux at half-levels, shape (..., nlev+1). Units: Pa/s.
    D_total_p : jax.Array
        Column-integrated mass-weighted divergence, shape (..., 1).
    """
    dp = dp_from_hybrid(coord, p_s)  # (..., nlev)

    # Mass-weighted divergence
    div_dp = div_3d * dp  # (..., nlev)

    # Cumulative sum from top — its last entry is ``D_total_p``, so we
    # extract that rather than calling ``jnp.sum`` independently.  Under
    # level-sharding this drops the per-stage cross-level collective from
    # 2 (sum + cumsum) to 1 (cumsum reuses its own last index).
    cumsum_div = jnp.cumsum(div_dp, axis=-1)  # (..., nlev)
    D_total_p = cumsum_div[..., -1:]  # (..., 1)

    # Mass flux at interfaces via the shared integration + boundary closure
    # (single-sourced so the C-grid / MPAS dycores reuse the SAME drop-last +
    # (1,1) pad on their own — flux-form — ``div_dp``).
    mass_flux = compute_mass_flux_from_cumsum(cumsum_div, D_total_p, coord)

    return mass_flux, D_total_p


def vertical_advection_hybrid(
    field: jax.Array,
    mass_flux: jax.Array,
    p_s: jax.Array,
    coord: HybridSigmaPressureCoordinate,
) -> jax.Array:
    """Compute vertical advection in hybrid coordinates.

    Computes: -F_full * df/dp

    where F is the mass flux interpolated from half-levels to full levels
    and df/dp uses upwind differencing in pressure space.

    Parameters
    ----------
    field : jax.Array
        Field to advect, shape (..., nlev).
    mass_flux : jax.Array
        Mass flux at half-levels from compute_mass_flux_hybrid,
        shape (..., nlev+1).
    p_s : jax.Array
        Surface pressure, shape (...,).
    coord : HybridSigmaPressureCoordinate

    Returns
    -------
    jax.Array
        Vertical advection tendency, shape (..., nlev).
    """
    # Interpolate mass flux to full levels
    F_full = 0.5 * (mass_flux[..., :-1] + mass_flux[..., 1:])  # (..., nlev)

    # Pressure at full levels
    p_full = pressure_from_hybrid(coord, p_s, full=True)  # (..., nlev)
    dp_full = jnp.diff(p_full, axis=-1)  # (..., nlev-1)
    dp_full_safe = jnp.clip(jnp.abs(dp_full), 1e-10, None)

    # Vertical gradient df/dp with upwind differencing.  Pad along the
    # trailing axis instead of allocating fresh ``zeros`` and
    # concatenating — single Pad HLO op each.
    df = jnp.diff(field, axis=-1)  # (..., nlev-1)
    df_over = df / dp_full_safe
    pad_axes = ((0, 0),) * (df_over.ndim - 1)
    grad_bwd = jnp.pad(df_over, (*pad_axes, (1, 0)))
    grad_fwd = jnp.pad(df_over, (*pad_axes, (0, 1)))

    # Upwind: F > 0 (downward mass flux) → backward difference
    grad = jnp.where(F_full > 0, grad_bwd, grad_fwd)

    return -F_full * grad


def vertical_advection_hybrid_sb(
    field: jax.Array,
    mass_flux: jax.Array,
    p_s: jax.Array,
    coord: HybridSigmaPressureCoordinate,
) -> jax.Array:
    """Simmons-Burridge centered vertical advection, hybrid coordinates.

        -eta_dot dp/deta * df/dp |_k
            = -(1/(2*dp_k)) * [ mdot_{k+1/2} * (f_{k+1} - f_k)
                              + mdot_{k-1/2} * (f_k   - f_{k-1}) ]

    with ``mdot`` the interface mass flux from
    :func:`compute_mass_flux_hybrid` (zero at top and surface) and ``dp_k``
    the LAYER thickness (half-level differences) — the conservation weight.
    Identical structure to the pure-sigma ``_vertical_advection_sigma_sb``:
    ``adv_k - f_k * (mdot_{k+1/2} - mdot_{k-1/2})/dp_k`` is a flux-form
    divergence whose dp-weighted column sum telescopes to the (zero)
    boundary fluxes, so ``sum(dp * f)`` is conserved in pairing with
    continuity — the property the upwind advective form
    (:func:`vertical_advection_hybrid`) lacks (measured -0.32..-0.47 K/day
    mass-weighted T sink on the sigma path at T63L8).
    """
    dp = dp_from_hybrid(coord, p_s)                # (..., nlev), layer thickness
    df = jnp.diff(field, axis=-1)                  # (..., nlev-1)
    md_int = mass_flux[..., 1:-1]                  # interior interfaces
    contrib = md_int * df
    pad_axes = ((0, 0),) * (contrib.ndim - 1)
    upper = jnp.pad(contrib, (*pad_axes, (1, 0)))  # mdot_{k-1/2}(f_k - f_{k-1})
    lower = jnp.pad(contrib, (*pad_axes, (0, 1)))  # mdot_{k+1/2}(f_{k+1} - f_k)
    return -(upper + lower) / (2.0 * jnp.clip(dp, 1e-10, None))


def vertical_advection_hybrid_van_leer(
    field: jax.Array,
    mass_flux: jax.Array,
    p_s: jax.Array,
    coord: HybridSigmaPressureCoordinate,
    smoothing_width: float = 1e-10,
    flux_smoothing_width: float = 1e-6,
) -> jax.Array:
    """Conservative, slope-limited (van Leer) tracer vertical advection, hybrid.

    Face-minus-cell increment form of :func:`_vertical_advection_van_leer_sigma`
    with the interface mass flux ``mdot`` [Pa/s, > 0 downward, 0 at top and
    surface] and the column's own layer thickness ``dp_k``::

        -[mdot_{k+1/2}(q_{k+1/2} - f_k) - mdot_{k-1/2}(q_{k-1/2} - f_k)] / dp_k

    This is the flux divergence minus ``f_k (mdot_{k+1/2}-mdot_{k-1/2})/dp_k``,
    so paired with the continuity that built ``mdot`` the column sum of
    ``dp*f`` telescopes to the zero boundary fluxes -- the property
    :func:`vertical_advection_hybrid` (advective upwind on half->full averaged
    fluxes) lacks.  A constant field has zero tendency at every level.  Scope:
    this closes the VERTICAL term only; the MPAS horizontal tracer operator is
    advective and not paired with the layer-thickness continuity.

    At zero width, face values use metric-aware van Leer reconstruction in
    pressure, clipped to the two adjacent cells. At positive width, they use
    the smooth reconstruction described below. The faces next to the top and
    bottom layers use the donor cell instead: a reconstructed value there is downwind
    of the boundary layer for inflow into the domain interior (see the sigma
    kernel's docstring for the instability), and a face shared by two cells
    must carry ONE value for the telescoping to hold.

    KNOWN DIFFERENCE (T18, W4): the default ``smoothing_width=1e-10``
    regularizes mixing-ratio increments [kg/kg] with a harmonic slope and
    smooth median face clamp; ``flux_smoothing_width=1e-6`` regularizes the
    upwind mass-flux split [Pa/s]. The previous single 1e-6 width was too
    large for condensate increments (T17: 1464 negative faces, -1.64e-7 kg/kg).
    Separate widths retain C-infinity dependence on tracer and mass flux at
    their positive defaults, including plateaus and zero flux. Each width is
    static and may independently be zero; BOTH zero restore the previous hard
    reconstruction and flux selection bitwise. A positive width whose square
    or reciprocal is unrepresentable in its input dtype raises.
    Exact constant fields and paired column conservation are retained. Finite
    widths do NOT guarantee TVD, positivity or bounded reconstruction; negative
    water reaches the existing conservative borrow on both production lanes
    (MPAS van Leer, lat-lon SB). This change does not modify that borrow.
    With the new defaults the seeded CAM L32 per-column maximum absolute
    tendency gaps [kg/kg/s] are 2.8815e-13..6.6878e-13 (seed-0 Gaussian),
    2.2949e-13..6.2433e-13 (seed-1 Gaussian), 9.0352e-14..1.0655e-13
    (top-hat), and 3.6982e-10..1.1407e-9 (seed-3 random). An offline
    standard-atmosphere level-2 MPAS mesh with CAM L32, production humidity
    initialization and seeded 1 m/s wind perturbations gives a full vapour
    tendency gap of 3.6705e-13 kg/kg/s maximum absolute and 6.7668e-5
    area-weighted relative L2. These are fixture measurements, not climate
    statistics. On the same MPAS fixture with the T17 condensate decks and
    dt=112.5 s, the new defaults give 1296 negative effective faces (minimum
    -1.8498e-11 kg/kg), 1300 negative forward-Euler updated cells (minimum
    -1.4606e-13 kg/kg), and max face error / regional hard-face range 0.24280.
    Whole-species vertical tendency relative L2 gaps for q_v/q_c/q_i/q_r/q_s
    and q_v*1e-6 are respectively 8.0275e-5, 9.1954e-5, 1.5663e-4,
    1.8808e-4, 1.2091e-3, 7.6004e-3. Full per-column gaps are stored in
    ``tests/unit/test_vertical_advection_hybrid_van_leer.py``.
    """
    # Each static regularizer is checked in the dtype of the quantity it smooths.
    # Either width may independently recover its hard branch.
    for name, width, dtype in (
        ("smoothing_width", smoothing_width, field.dtype),
        ("flux_smoothing_width", flux_smoothing_width, mass_flux.dtype),
    ):
        if not math.isfinite(width) or width < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
        if width > 0.0:
            limits = np.finfo(np.dtype(dtype))
            if not (float(limits.tiny) <= width * width <= float(limits.max)
                    and 1.0 / width <= float(limits.max)):
                raise ValueError(f"{name} square and reciprocal must be representable")
    dp = dp_from_hybrid(coord, p_s)                                  # (..., nlev)
    dc = jnp.diff(pressure_from_hybrid(coord, p_s, full=True), axis=-1)
    q_pos, q_neg = _van_leer_face_values(field, dc, dp, smoothing_width)
    donor_pos = jnp.concatenate([field[..., :1], field], axis=-1)    # cell above
    donor_neg = jnp.concatenate([field, field[..., -1:]], axis=-1)   # cell below
    nlev = field.shape[-1]
    edge = jnp.zeros(nlev + 1, dtype=bool).at[jnp.array([1, nlev - 1])].set(True)
    q_pos = jnp.where(edge, donor_pos, q_pos)
    q_neg = jnp.where(edge, donor_neg, q_neg)
    if flux_smoothing_width > 0.0:
        from legoesm.core.smooth import smooth_abs

        # Increment form keeps constants exactly zero even under smoothing.
        # One shared face flux still telescopes in the column product rule.
        flux_abs = smooth_abs(mass_flux, flux_smoothing_width**2) - flux_smoothing_width
        pos_top = q_pos[..., :-1] - field
        neg_top = q_neg[..., :-1] - field
        pos_bot = q_pos[..., 1:] - field
        neg_bot = q_neg[..., 1:] - field
        inc_top = (0.5 * mass_flux[..., :-1] * (pos_top + neg_top)
                   + 0.5 * flux_abs[..., :-1] * (pos_top - neg_top))
        inc_bot = (0.5 * mass_flux[..., 1:] * (pos_bot + neg_bot)
                   + 0.5 * flux_abs[..., 1:] * (pos_bot - neg_bot))
        return -(inc_bot - inc_top) / jnp.clip(dp, 1e-10, None)
    q_face = jnp.where(mass_flux > 0, q_pos, q_neg)
    inc_top = mass_flux[..., :-1] * (q_face[..., :-1] - field)
    inc_bot = mass_flux[..., 1:] * (q_face[..., 1:] - field)
    return -(inc_bot - inc_top) / jnp.clip(dp, 1e-10, None)


def _face_minus_cell_tendency(field, mass_flux, q_face, dp):
    """``-[F_{k+1/2}(q_{k+1/2} - f_k) - F_{k-1/2}(q_{k-1/2} - f_k)] / dp_k``.

    The update shared by the van Leer and PPM tracer operators: paired with
    the continuity that built ``F`` (zero at the top and the surface), the
    column sum of ``dp * f`` telescopes, and a constant field has zero
    tendency whatever the face values."""
    inc_top = mass_flux[..., :-1] * (q_face[..., :-1] - field)
    inc_bot = mass_flux[..., 1:] * (q_face[..., 1:] - field)
    return -(inc_bot - inc_top) / jnp.clip(dp, 1e-10, None)



PPM_KORDS = (4, 7)


def _ppm_shift(x, n):
    """``x(k+n)`` at every level k (last axis), edge-replicated beyond the
    column so every arm stays finite; callers mask the levels where the
    Fortran loop does not run."""
    if n == 0:
        return x
    pad = ((0, 0),) * (x.ndim - 1)
    if n > 0:
        return jnp.pad(x[..., n:], (*pad, (0, n)), mode="edge")
    return jnp.pad(x[..., :n], (*pad, (-n, 0)), mode="edge")


def _ppm_levels(km, lo, hi):
    """Static mask of Fortran levels lo..hi (1-based, inclusive)."""
    k = np.arange(1, km + 1)
    return jnp.asarray((k >= lo) & (k <= hi))


class _PPMSwitch:
    """The limiter's branch decisions, hard (CAM exactly) or smooth.

    Every decision is taken on the BULK field (index 0 of the stacked
    arrays) and applied to the whole stack, so a twin (index 1) follows the
    bulk's branches and its transport is linear in the twin (proposal
    section 3).  ``sharpness = inf`` gives Fortran's comparisons exactly,
    strict or not as in the source.  A finite sharpness replaces each
    comparison ``lhs < rhs`` by the weight ``sigmoid(sharpness * z)`` with
    ``z = (rhs - lhs) / (|lhs| + |rhs|)``, a relative difference in
    [-1, 1] (``legoesm.core.smooth.sigmoid_switch``), and an equality test
    ``x == 0`` by ``exp(-(sharpness * x / |x|)^2)``; the branches are then
    blended with that weight.  The relative form keeps the limiter
    positively homogeneous (scaling the field scales its transport), so a
    tracer with a uniform ratio to its bulk keeps that ratio under its own
    limiter too.  A comparison against zero gives z = +-1 and stays sharp
    at any sharpness above about 40 in float64.
    """

    def __init__(self, sharpness):
        self.hard = sharpness == math.inf
        self.sharpness = sharpness

    def lt(self, lhs, rhs, strict=True):
        if self.hard:
            return lhs < rhs if strict else lhs <= rhs
        from legoesm.core.smooth import sigmoid_switch
        lhs = jnp.broadcast_to(lhs, jnp.shape(rhs)) if jnp.ndim(lhs) == 0 else lhs
        d = rhs - lhs
        den = jnp.abs(lhs) + jnp.abs(rhs)
        z = jnp.where(den > 0, d / jnp.where(den > 0, den, 1.0), 0.0)
        return sigmoid_switch(z, self.sharpness)

    def eq0(self, x):
        if self.hard:
            return x == 0
        den = jnp.abs(x)
        z = jnp.where(den > 0, x / jnp.where(den > 0, den, 1.0), 0.0)
        return jnp.exp(-(self.sharpness * z) ** 2)

    @staticmethod
    def both(w1, w2):
        return (w1 & w2) if w1.dtype == jnp.bool_ else w1 * w2

    @staticmethod
    def pick(w, a, b):
        """``a`` where the decision ``w`` holds, else ``b`` (stack-aware)."""
        if w.dtype == jnp.bool_:
            return jnp.where(w, a, b)
        return b + w * (a - b)

    # value operations on stacked arrays (bulk at index 0)
    def max2(self, x, y):
        return self.pick(self.lt(y[0], x[0], strict=False), x, y)

    def min2(self, x, y):
        return self.pick(self.lt(x[0], y[0], strict=False), x, y)

    def abs(self, x):
        return self.pick(self.lt(jnp.zeros_like(x[0]), x[0], strict=False), x, -x)

    def sign(self, a, b):
        """Fortran ``sign(a, b)``: ``|a|`` where ``b >= 0``, else ``-|a|``."""
        aa = self.abs(a)
        return self.pick(self.lt(jnp.zeros_like(b[0]), b[0], strict=False), aa, -aa)


def _kmppm(sw, dm, a, al, ar, a6, lmt):
    """mapz_module.F90 kmppm (:1092-1198) on stacked cells; returns
    ``(al, ar, a6)``.  ``lmt`` 0: standard PPM constraint; 1: Lin's full
    monotonicity constraint; 2: positive definite constraint."""
    if lmt == 0:
        da1 = ar - al
        da2 = da1 * da1
        a6da = a6 * da1
        lo = sw.lt(a6da[0], -da2[0])            # a6da < -da2
        hi = sw.lt(da2[0], a6da[0])             # a6da >  da2
        a6_lo = 3.0 * (al - a)
        a6_hi = 3.0 * (ar - a)
        al_n = sw.pick(lo, al, sw.pick(hi, ar - a6_hi, al))
        ar_n = sw.pick(lo, al - a6_lo, ar)
        a6_n = sw.pick(lo, a6_lo, sw.pick(hi, a6_hi, a6))
        z = sw.eq0(dm[0])                       # dm == 0: flat cell
        return (sw.pick(z, a, al_n), sw.pick(z, a, ar_n),
                sw.pick(z, jnp.zeros_like(a6), a6_n))
    if lmt == 1:
        qmp = 2.0 * dm
        al = a - sw.sign(sw.min2(sw.abs(qmp), sw.abs(al - a)), qmp)
        ar = a + sw.sign(sw.min2(sw.abs(qmp), sw.abs(ar - a)), qmp)
        return al, ar, 3.0 * (2.0 * a - (al + ar))
    if lmt == 2:
        dlr = ar - al
        out = sw.lt(sw.abs(dlr)[0], -a6[0])     # |ar - al| < -a6
        # a6 < 0 wherever ``out`` holds; elsewhere fmin's value is discarded,
        # so the divisor there only has to be finite and non-zero.
        a6s = jnp.where(a6[0] < 0, a6[0], -1.0)
        fmin = a[0] + 0.25 * dlr[0] ** 2 / a6s + a6[0] / 12.0
        act = sw.both(out, sw.lt(fmin, jnp.zeros_like(fmin)))
        flat = sw.both(sw.lt(a[0], ar[0]), sw.lt(a[0], al[0]))
        rgt = sw.lt(al[0], ar[0])                # ar > al
        a6_l = 3.0 * (al - a)
        a6_r = 3.0 * (ar - a)
        al_n = sw.pick(flat, a, sw.pick(rgt, al, ar - a6_r))
        ar_n = sw.pick(flat, a, sw.pick(rgt, al - a6_l, ar))
        a6_n = sw.pick(flat, jnp.zeros_like(a6), sw.pick(rgt, a6_l, a6_r))
        return sw.pick(act, al_n, al), sw.pick(act, ar_n, ar), sw.pick(act, a6_n, a6)
    raise ValueError(f"kmppm: lmt={lmt!r} is not ported (0, 1 or 2)")


def ppm2m_profile(a, dp, *, iv, kord, sharpness=math.inf):
    """CAM-FV ``ppm2m`` parabolas, ``(AL, AR, A6)`` per cell.

    Port of mapz_module.F90 ppm2m (:672-950, cam_cesm2_1_rel_60) with kmppm
    and steepz.  The parabola in cell k is
    ``f(s) = AL + s [(AR - AL) + A6 (1 - s)]``, ``s`` from 0 at the top edge
    to 1 at the bottom edge, with mean ``a``.

    Parameters
    ----------
    a : (S, ..., km) cell means, levels top first.  ``S`` stacks a bulk
        field (index 0) and, in twin mode, a twin (index 1) that takes every
        branch decision from the bulk.
    dp : (..., km) layer thickness [Pa], the ``delp`` of ppm2m.
    iv : 0 (positive definite scalar) or 1 (other fields).  Winds (iv = -1)
        are not ported.
    kord : 4 (Lin's full monotonicity constraint, CAM-FV's T_v) or 7
        (Huynh's constraint, quasi monotone, plus the positive definite
        constraint for iv = 0: CAM-FV's constituents).
    sharpness : ``inf`` = the Fortran branches exactly; finite = the smooth
        decisions of :class:`_PPMSwitch` (RULE 5).

    Notes
    -----
    Every Fortran k loop here reads only arrays written by earlier loops, so
    each is vectorised over k with static level masks.  The tie-breaking of
    Fortran ``max``/``min`` never changes a value.  The two divisions whose
    value can be discarded (by ``df2`` in steepz and by ``a6`` in kmppm
    lmt 2) get a finite stand-in divisor where the quotient is not used.
    """
    if iv not in (0, 1):
        raise ValueError(f"ppm2m_profile: iv={iv!r} not ported (0 or 1)")
    if kord not in PPM_KORDS:
        raise ValueError(f"ppm2m_profile: kord={kord!r} not ported; one of {PPM_KORDS}")
    if not (sharpness > 0):
        raise ValueError(f"ppm2m_profile: sharpness must be > 0 (inf = hard), got {sharpness!r}")
    km = a.shape[-1]
    if km < 4:
        raise ValueError(f"ppm2m_profile needs at least 4 levels; got {km}")
    sw = _PPMSwitch(sharpness)
    S = _ppm_shift
    lev = lambda lo, hi: _ppm_levels(km, lo, hi)  # noqa: E731
    zero = jnp.zeros_like(a)
    am1, ap1 = S(a, -1), S(a, 1)
    dpm1, dpp1 = S(dp, -1), S(dp, 1)
    delq = ap1 - a                     # Fortran delq(k) = a(k+1) - a(k)
    delqm1 = S(delq, -1)               # delq(k-1)
    d4 = dpm1 + dp                     # d4(k) = delp(k-1) + delp(k)
    d4p1, d4m1 = S(d4, 1), S(d4, -1)

    # k = 2..km-1: limited slope dc and first-guess mismatch df2
    c1 = (dpm1 + 0.5 * dp) / d4p1
    c2 = (dpp1 + 0.5 * dp) / d4
    tmp = dp * (c1 * delq + c2 * delqm1) / (d4 + dpp1)
    qmax = sw.max2(sw.max2(am1, a), ap1) - a
    qmin = a - sw.min2(sw.min2(am1, a), ap1)
    dc = sw.sign(sw.min2(sw.min2(sw.abs(tmp), qmax), qmin), tmp)
    m_int = lev(2, km - 1)
    dc = jnp.where(m_int, dc, zero)
    df2 = jnp.where(m_int, tmp, zero)

    # k = 3..km-1: fourth-order edge value below cell k-1
    cc1 = delqm1 * dpm1 / d4
    a1 = d4m1 / (d4 + dpm1)
    a2 = d4p1 / (d4 + dp)
    dcm1 = S(dc, -1)
    al = am1 + cc1 + 2.0 / (d4m1 + d4p1) * (
        dp * (cc1 * (a1 - a2) + a2 * dcm1) - dpm1 * a1 * dc)
    al = jnp.where(lev(3, km - 1), al, zero)

    # steepz: decisions and the steepening weight alfa from the bulk
    rat = delqm1[0] / d4                                   # k = 2..km
    f = (S(rat, 1) - rat) / (dpm1 + dp + dpp1)             # k = 2..km-1
    fp, fm = S(f, 1), S(f, -1)
    dg2 = (fp - fm) * ((dpp1 - dpm1) ** 2 + d4 * d4p1)
    df2b = df2[0]
    ratio = (-0.1875 * dg2 / jnp.where(df2b != 0, df2b, 1.0))[None]
    alfa_v = sw.max2(jnp.zeros_like(ratio), sw.min2(ratio, jnp.full_like(ratio, 0.5)))[0]
    nonzero = (df2b != 0) if sw.hard else 1.0 - sw.eq0(df2b)
    cond = sw.both(sw.lt(fp * fm, jnp.zeros_like(fm)), nonzero)
    alfa = jnp.where(lev(3, km - 2), sw.pick(cond, alfa_v, jnp.zeros_like(alfa_v)), 0.0)
    alm1 = S(alfa, -1)
    al_st = (1.0 - alm1 - alfa) * al + alm1 * (a - dc) + alfa * (am1 + dcm1)
    al = jnp.where(lev(4, km - 2), al_st, al)

    # top: area-preserving cubic with zero second derivative at the top
    d1, d2 = dp[..., 0], dp[..., 1]
    a_1, a_2 = a[..., 0], a[..., 1]
    qm = (d2 * a_1 + d1 * a_2) / (d1 + d2)
    dq = 2.0 * (a_2 - a_1) / (d1 + d2)
    ct1 = 4.0 * (al[..., 2] - qm - d2 * dq) / (d2 * (2.0 * d2 * d2 + d1 * (d2 + 3.0 * d1)))
    ct3 = dq - 0.5 * ct1 * (d2 * (5.0 * d1 + d2) - 3.0 * d1 ** 2)
    al2 = qm - 0.25 * ct1 * d1 * d2 * (d2 + 3.0 * d1)
    al1 = d1 * (2.0 * ct1 * d1 ** 2 - ct3) + al2
    dc1 = 0.5 * (a_1 - al1)
    al2 = sw.max2(sw.min2(a_1, a_2), al2)
    al2 = sw.min2(sw.max2(a_1, a_2), al2)
    if iv == 0:
        al1 = sw.max2(jnp.zeros_like(al1), al1)
        al2 = sw.max2(jnp.zeros_like(al2), al2)

    # bottom: area-preserving cubic with zero second derivative at the surface
    d1, d2 = dp[..., -1], dp[..., -2]
    a_k, a_km1 = a[..., -1], a[..., -2]
    qm = (d2 * a_k + d1 * a_km1) / (d1 + d2)
    dq = 2.0 * (a_km1 - a_k) / (d1 + d2)
    cb1 = (al[..., -2] - qm - d2 * dq) / (d2 * (2.0 * d2 * d2 + d1 * (d2 + 3.0 * d1)))
    cb3 = dq - 2.0 * cb1 * (d2 * (5.0 * d1 + d2) - 3.0 * d1 ** 2)
    alk = qm - cb1 * d1 * d2 * (d2 + 3.0 * d1)
    ark = d1 * (8.0 * cb1 * d1 ** 2 - cb3) + alk
    dck = 0.5 * (ark - a_k)
    alk = sw.max2(sw.min2(a_k, a_km1), alk)
    alk = sw.min2(sw.max2(a_k, a_km1), alk)
    if iv == 0:
        ark = sw.max2(jnp.zeros_like(ark), ark)

    al = al.at[..., 0].set(al1).at[..., 1].set(al2).at[..., -1].set(alk)
    dc = dc.at[..., 0].set(dc1).at[..., -1].set(dck)
    ar = jnp.concatenate([al[..., 1:], ark[..., None]], axis=-1)   # ar(k) = al(k+1)
    a6 = 3.0 * (2.0 * a - (al + ar))

    def _apply(mask, new):
        return tuple(jnp.where(mask, n, o) for n, o in zip(new, (al, ar, a6)))

    # top two layers: standard PPM constraint
    al, ar, a6 = _apply(lev(1, 2), _kmppm(sw, dc, a, al, ar, a6, 0))
    m_mid = lev(3, km - 2)
    if kord >= 7:
        h2 = S(dc, 1) - S(dc, -1)                 # k = 2..km-1
        fac = 1.5 if kord == 7 else 0.125
        pmp = 2.0 * dc
        lac = a + fac * S(h2, -1) + dc
        qmp = a + pmp
        lo = sw.min2(sw.min2(a, qmp), lac)
        hi = sw.max2(sw.max2(a, qmp), lac)
        ar_n = sw.min2(sw.max2(ar, lo), hi)
        qmp = a - pmp
        lac = a + fac * S(h2, 1) - dc
        lo = sw.min2(sw.min2(a, qmp), lac)
        hi = sw.max2(sw.max2(a, qmp), lac)
        al_n = sw.min2(sw.max2(al, lo), hi)
        a6_n = 3.0 * (2.0 * a - (al_n + ar_n))
        new = (al_n, ar_n, a6_n)
        if iv == 0 and kord == 7:
            new = _kmppm(sw, dc, a, *new, 2)
        al, ar, a6 = _apply(m_mid, new)
    else:                                         # kord 4: lmt = 1
        al, ar, a6 = _apply(m_mid, _kmppm(sw, dc, a, al, ar, a6, 1))
    # bottom two layers: A6 recomputed, standard PPM constraint
    a6_b = 3.0 * (2.0 * a - (al + ar))
    al, ar, a6 = _apply(lev(km - 1, km), _kmppm(sw, dc, a, al, ar, a6_b, 0))
    return al, ar, a6


def ppm_swept_face_values(al, ar, a6, mass_flux, dp, dt, *, upwind=None):
    """Mean of the donor cell's parabola over the mass each interface sweeps
    in ``dt``, at every interface (..., km+1); the two boundary faces carry
    zero flux and return 0.

    ``F > 0`` (downward): donor = the cell above, swept ``s`` in
    ``[1 - c, 1]`` with ``c = F dt / dp_above``; ``F < 0``: donor = the cell
    below, ``s`` in ``[0, c]`` with ``c = -F dt / dp_below``.  The mean of
    ``f`` over ``[s0, s1]`` is map1_ppm's
    ``AL + (AR - AL + A6)(s0 + s1)/2 - A6 (s1^2 + s1 s0 + s0^2)/3``
    (mapz_module.F90:325-327), which has no division by ``c``: at ``c = 0``
    it is the donor's edge value (the method-of-lines limit).  Valid for
    ``c <= 1``; a larger ``c`` would integrate the parabola beyond its cell
    (the proposal measured about 0.01 at L32, dt 300 s).  ``upwind``
    (..., km+1), when given, decides the donor by its sign instead of
    ``mass_flux`` (same sign, but defined where ``mass_flux * dt`` is 0)."""
    Fi = mass_flux[..., 1:-1]
    r3 = 1.0 / 3.0
    s0 = 1.0 - Fi * dt / dp[..., :-1]
    al_u, ar_u, a6_u = al[..., :-1], ar[..., :-1], a6[..., :-1]
    down = al_u + 0.5 * (ar_u - al_u + a6_u) * (s0 + 1.0) - a6_u * r3 * (1.0 + s0 + s0 * s0)
    c = -Fi * dt / dp[..., 1:]
    al_d, ar_d, a6_d = al[..., 1:], ar[..., 1:], a6[..., 1:]
    up = al_d + 0.5 * (ar_d - al_d + a6_d) * c - a6_d * r3 * c * c
    sel = Fi if upwind is None else upwind[..., 1:-1]
    face = jnp.where(sel > 0, down, up)
    pad = ((0, 0),) * (face.ndim - 1)
    return jnp.pad(face, (*pad, (1, 1)))


def vertical_advection_hybrid_ppm(
    field: jax.Array,
    mass_flux: jax.Array,
    p_s: jax.Array,
    coord: HybridSigmaPressureCoordinate,
    dt: float,
    *,
    kord: int,
    iv: int,
    sharpness: float,
) -> jax.Array:
    """Eulerian flux-form PPM vertical transport on the hybrid lane.

    The face value of :func:`vertical_advection_hybrid_van_leer`'s
    face-minus-cell form is replaced by the donor cell's limited CAM-FV
    parabola (:func:`ppm2m_profile`) integrated over the mass the face sweeps
    in ``dt`` (:func:`ppm_swept_face_values`).  Continuity, the mass flux and
    the update are unchanged, so the column sum of ``dp * field`` telescopes
    and a constant field keeps zero tendency.  Proposal option (b1):
    results/legoesm/scm/ppm_remap_proposal.md.


    KNOWN DIFFERENCES (RULE 5): at ``sharpness = inf`` the parabolas are
    CAM-FV's (tests: tests/grids/test_vertical_ppm.py, fixture
    tests/grids/oracle_fixtures/cam_fv_ppm2m.npz).  At a finite sharpness the
    limiter and the twin fallback blend their branches where the compared
    quantities differ by less than about 4/sharpness relative; the gaps
    measured on the fixture columns are stored in that test.  The smoothing
    can leave small negative face values where kord 7's positive definite
    constraint would have clipped exactly.  The upwind choice of donor by
    the sign of the mass flux stays a hard switch, as in the van Leer
    operator: the flux ``F (q_face - q)`` is continuous across it.
    The limiter's comparisons against ZERO (``abs``, ``sign``, the flat-cell
    ``dm == 0`` test) see a relative difference of exactly +-1 and so stay
    hard at any finite sharpness; ``abs`` is continuous and ``sign`` multiplies
    a magnitude that vanishes with ``dm``, so AD is exact almost everywhere,
    but these are not smoothed (follow-up: an absolute-width form).  Valid for
    a vertical Courant number ``c <= 1`` (measured about 0.01 at L32, dt
    300 s); larger ``c`` is not guarded.
    """
    return ppm_flux_tendency(field, mass_flux, dp_from_hybrid(coord, p_s), dt,
                             kord=kord, iv=iv, sharpness=sharpness)




def ppm_flux_tendency(field, mass_flux, dp, dt, *, kord, iv, sharpness):
    """:func:`vertical_advection_hybrid_ppm` on given layer thicknesses ``dp``
    (..., km) [Pa]; see there."""
    al, ar, a6 = ppm2m_profile(field[None], dp, iv=iv, kord=kord, sharpness=sharpness)
    face = ppm_swept_face_values(al, ar, a6, mass_flux, dp, dt)
    return _face_minus_cell_tendency(field, mass_flux, face[0], dp)


def cam_fv_pkz(p_half):
    """CAM-FV's layer mean of ``p^kappa``,
    ``(pe_{k+1}^kappa - pe_k^kappa) / (kappa (ln pe_{k+1} - ln pe_k))``
    (the pkz that te_map multiplies pt by), on edges (..., km+1) [Pa]."""
    kappa = constants.kappa
    pk = p_half ** kappa
    return (pk[..., 1:] - pk[..., :-1]) / (
        kappa * (jnp.log(p_half[..., 1:]) - jnp.log(p_half[..., :-1])))


def ppm_tv_lnp_start(T, q_v, p_half, mass_flux, step):
    """CAM-FV te_map's state before its temperature remap, for a step of
    ``step`` seconds: ``(tv_star, x, w)``.

    * ``tv_star``: the virtual temperature after the Lagrangian step, pt
      times pkz of the Lagrangian edges ``pe + F step`` (pt conserved,
      te_map.F90 "Converting pt to t"), i.e.
      ``T (1 + zvir q_v) pkz(pe + F step) / pkz(pe)``;
    * ``x = ln pe``, the coordinate CAM-FV maps temperature over;
    * ``w = ln pe - ln(pe - F step)`` at each interface, the ln p interval
      the face sweeps (F > 0 downward; 0 at the top and the surface).
    """
    from legoesm.thermo import virtual_temperature
    tv = virtual_temperature(T, q_v)
    tv_star = tv * cam_fv_pkz(p_half + mass_flux * step) / cam_fv_pkz(p_half)
    x = jnp.log(p_half)
    w = x - jnp.log(p_half - mass_flux * step)
    return tv_star, x, w


def ppm_tv_lnp_tendency(T, q_v, dq_v, mass_flux, p_half, dt, *, sharpness):
    """Temperature tendency of CAM-FV's vertical step on the Eulerian grid:
    the F part of the adiabatic term plus the vertical transport, as te_map
    does it (te_map.F90:476-569, cam_cesm2_1_rel_60).

    ``tv_star`` (:func:`ppm_tv_lnp_start`) is reconstructed with kord 4,
    iv 1 (``ppm2m``) over ``ln pe``, and each face carries the mean of its
    donor's parabola over the ln p interval it sweeps in ``dt``, in the
    face-minus-cell form with ln p thicknesses as the weights (the quantity
    map1_ppm on log pe conserves is ``sum T_v d ln p``).  The new
    temperature divides the new virtual temperature by ``1 + zvir q_v``
    with the vapour after its own vertical step, ``q_v + dt dq_v``, as
    te_map divides by the remapped vapour.  The tendency is
    ``(T_new - T) / dt``; for a Python ``dt == 0`` it is the limit, the
    derivative in the step length at zero.

    ``dq_v`` is the vapour's vertical (PPM) tendency; pass zeros for a dry
    state (``q_v`` zero too).  ``p_half`` (..., km+1) must be positive
    (CAM-FV's ptop > 0).
    """
    def new_T(step):
        tv_star, x, w = ppm_tv_lnp_start(T, q_v, p_half, mass_flux, step)
        dx = jnp.diff(x, axis=-1)
        al, ar, a6 = ppm2m_profile(tv_star[None], dx, iv=1, kord=4, sharpness=sharpness)
        # donor by the sign of F: at step 0 the swept interval w is exactly
        # zero and would not say which side the donor is on
        face = ppm_swept_face_values(al, ar, a6, w, dx, 1.0, upwind=mass_flux)[0]
        incr = _face_minus_cell_tendency(tv_star, w, face, dx)
        from legoesm.thermo import virtual_temperature
        return (tv_star + incr) / virtual_temperature(jnp.ones_like(q_v), q_v + step * dq_v)

    if isinstance(dt, (int, float)) and dt == 0:
        zero = jnp.zeros((), dtype=jnp.result_type(T))
        return jax.jvp(new_T, (zero,), (jnp.ones_like(zero),))[1]
    return (new_T(dt) - T) / dt


def vertical_advection_tv_hybrid_ppm(T, q_v, dq_v, mass_flux, p_s, coord, dt, *, sharpness):
    """:func:`ppm_tv_lnp_tendency` on the hybrid lane's interface pressures.

    Replaces :func:`vertical_advection_theta_hybrid` under the ``ppm`` scheme:
    it returns the same quantity, ``-F dT/dp + kappa T F / p`` (the F part of
    the vertical advection plus the adiabatic term), in CAM-FV's discrete
    form.  Needs a model top above zero pressure, as CAM-FV's ln p map does.
    """
    # Host check on a concrete coordinate; skipped when it is traced.
    concrete = not any(isinstance(x, jax.core.Tracer)
                       for x in (coord.A_half, coord.B_half, coord.p_ref))
    a0 = np.asarray(coord.A_half)[0] * float(coord.p_ref) if concrete else 1.0
    b0 = np.asarray(coord.B_half)[0] if concrete else 0.0
    if not (a0 > 0.0 and b0 == 0.0):
        raise ValueError(
            "the ppm temperature path maps T_v over ln p (CAM-FV te_map) and "
            "needs a model top above zero pressure (pure-pressure top "
            f"A_half[0] * p_ref > 0, B_half[0] = 0); got {a0!r} Pa, B {b0!r}")
    p_half = pressure_from_hybrid(coord, p_s, full=False)
    return ppm_tv_lnp_tendency(T, q_v, dq_v, mass_flux, p_half, dt, sharpness=sharpness)




def vertical_advection_theta_hybrid(
    T: jax.Array,
    mass_flux: jax.Array,
    p_s: jax.Array,
    coord: HybridSigmaPressureCoordinate,
    *,
    conservative: bool = False,
) -> jax.Array:
    """Combined vertical advection + adiabatic mass-flux term for T (hybrid).

    Hybrid-coordinate mirror of :func:`vertical_advection_theta`.  Instead of
    computing  -F·∂T/∂p  and  κ·T·F/p  separately (which involves catastrophic
    cancellation at upper levels where 1/p → ∞), this advects potential
    temperature θ and converts back:

        -F·∂T/∂p + κ·T·F/p  =  -(p/p₀)^κ · F·∂θ/∂p

    with θ = T·(p₀/p)^κ.  This is the *same continuous operator* — no sign
    flip — but it cancels the two large, near-equal terms **before**
    discretization, so it eliminates the 1/p amplification of the mismatched-
    stencil 2Δz residual at the stretched top levels (#930).  The σ-convention
    (index 0 = model top, F > 0 downward) is inherited verbatim from the reused
    :func:`vertical_advection_hybrid`.

    Parameters
    ----------
    T : jax.Array
        Temperature, shape (..., nlev).
    mass_flux : jax.Array
        Mass flux at half-levels from :func:`compute_mass_flux_hybrid`,
        shape (..., nlev+1).
    p_s : jax.Array
        Surface pressure, shape (...,).
    coord : HybridSigmaPressureCoordinate

    Returns
    -------
    jax.Array
        Combined tendency: -F·∂T/∂p + κ·T·F/p, shape (..., nlev).
    """
    kappa = constants.kappa
    P_0 = constants.p_ref

    # Pressure at full levels; single exner used both directions so the
    # θ round-trip is exact even where p_full < 1 Pa (matches the sigma
    # sibling's ``jnp.maximum(p_full, 1.0)`` floor).
    p_full = pressure_from_hybrid(coord, p_s, full=True)  # (..., nlev)
    exner = (jnp.maximum(p_full, 1.0) / P_0) ** kappa  # (p/p₀)^κ

    # Potential temperature θ = T / exner = T·(p₀/p)^κ
    theta = T / exner

    # Advect θ, then convert back: -exner·F·∂θ/∂p.  ``conservative=True``
    # swaps the upwind advective operator for the Simmons-Burridge flux form,
    # which satisfies the discrete product rule (column residual 6.7e-16
    # against the advective form's 3.8e-2 of the interfacial exchange on the
    # CAM L32 table); the exner round-trip is identical either way.
    _op = vertical_advection_hybrid_sb if conservative else vertical_advection_hybrid
    return exner * _op(theta, mass_flux, p_s, coord)


def sb81_omega_over_p_dyn(
    cumsum_mass_div: jax.Array,
    coord: HybridSigmaPressureCoordinate,
    p_s: jax.Array,
    dp_s_dt: jax.Array | None = None,
) -> jax.Array:
    """SB81 α-weighted DYNAMIC part of the energy conversion ``(ω/p)_k``.

    Simmons & Burridge (1981) / IFS discretization of the non-advective part
    of the thermodynamic conversion term (#1029 ω-side)::

        (ω/p)_k^dyn = -(1/Δp_k) [ L_k · (Σ_{j<k} C_j - B_top·dp_s/dt)
                                   + α_k · C_k ]

    with ``C_j = ∇·(v_j Δp_j)`` the flux-form layer mass divergence,
    ``L_k = ln(p_{k+1/2}/p_{k-1/2})`` and ``α_k`` the exact SB81 alpha from
    :func:`sb81_halflevel_construction` — the SAME half-level construction
    the geopotential integration and the momentum/thermo ``ln p`` gradients
    use (the arithmetic ``ω_full / p_full`` form this replaces was a third,
    independent discretization of the same continuous operator).  ``Δp_k``
    is differenced INTERNALLY from the same clipped half-level pressures as
    ``L_k``/``α_k`` — passing an externally-built ``dA + dB·p_s`` thickness
    would differ by rounding (and by the clip in a zero-p-top layer),
    breaking the discrete identities below.

    The caller adds the advective part ``v_k · ∇(ln p_k^SB)`` separately
    (the shared SB81 full-level field of :func:`sb81_full_level_ln_p`);
    together they discretize the full ``ω/p``.  The ``∂p/∂t`` and
    ``η̇ ∂p/∂η`` contributions are CONTAINED in the cumulative-divergence
    expression (continuity + the ``F = 0`` top closure fold them in) —
    EXCEPT the top-boundary term when the coordinate's top interface itself
    moves in pressure (``B_top != 0``, e.g. a sigma-like coordinate with
    ``p_top = sigma_top·p_s``): there ``(∂p/∂t + η̇ ∂p/∂η)(p̂) =
    B_top·dp_s/dt - cumsum(p̂)`` and the constant layer-averages against
    ``dp/p`` to ``+ B_top·dp_s/dt·L_k/Δp_k``.  Pass ``dp_s_dt`` (the RAW
    continuity diagnosis ``-D_total/B_range``, NOT a globally corrected
    variant — a zero-mean fixer applied to the prognostic ``dp_s/dt``
    breaks the continuity identity this derivation rests on) to include
    it; the term is multiplied by ``coord.B_half[0]`` traced (no Python
    branch), so ``B_top = 0`` coordinates const-fold it away and the
    function stays jit/AD-safe for traced coordinates.

    Discrete column identity (unit-tested to fp64 roundoff, not bit
    exactness — separate ``log``/multiply/reduce roundings)::

        Σ_k Δp_k (ω/p)_k^dyn = -Σ_j C_j (ln p_s - ln p_j^SB)
                               + B_top·dp_s/dt · ln(p_s / p_top_safe)

    i.e. the column-integrated conversion telescopes onto the SAME discrete
    ``ln p^SB`` field whose gradient does the momentum PGF work.  This is a
    VERTICAL-discretization consistency statement only: on the C-grid the
    horizontal pairing (face-flux ``C`` vs the centre-averaged
    ``v·∇ln p^SB`` product) is not exact summation-by-parts, so no exact
    global energy-conservation claim follows (#1029 tracks the residual
    via the forced ``held_suarez_topo`` A/B, not an algebraic proof).

    Top-layer convention at an exactly-zero-pressure top: the clipped
    construction gives ``α_0 → 1`` (documented in
    :func:`sb81_full_level_ln_p`), NOT the IFS ``α_1 = ln 2`` special case
    — chosen so the conversion, the geopotential and the ``ln p^SB``
    gradients keep ONE α field; adopting the IFS convention would have to
    change all three together.

    Parameters
    ----------
    cumsum_mass_div : jax.Array
        ``cumsum(div(dp·v), axis=-1)`` — INCLUSIVE cumulative flux-form mass
        divergence, shape (..., nlev) [Pa/s].
    coord : HybridSigmaPressureCoordinate
    p_s : jax.Array
        Surface pressure, shape (...,) [Pa].
    dp_s_dt : jax.Array or None
        RAW surface-pressure tendency ``-D_total/B_range``, shape (...,)
        [Pa/s].  Only consumed through ``B_top`` (moving-top coordinates);
        ``None`` omits the term.

    Returns
    -------
    jax.Array
        ``(ω/p)_k^dyn``, shape (..., nlev) [1/s].
    """
    p_half_safe, ln_ratio, alpha = sb81_halflevel_construction(coord, p_s)
    # Internal Δp from the SAME clipped half-level pressures as L/α.
    dp = p_half_safe[..., 1:] - p_half_safe[..., :-1]
    # C_k from the inclusive cumsum (C_0 = cumsum_0): one Pad HLO, no concat.
    pad_axes = ((0, 0),) * (cumsum_mass_div.ndim - 1) + ((1, 0),)
    cumsum_excl = jnp.pad(cumsum_mass_div[..., :-1], pad_axes)  # Σ_{j<k} C_j
    C = cumsum_mass_div - cumsum_excl                           # C_k
    if dp_s_dt is not None:
        # Moving-top term: traced multiply by B_half[0] (jit/AD-safe; a
        # static B_top = 0 const-folds to the fixed-top expression).
        cumsum_excl = cumsum_excl - coord.B_half[0] * dp_s_dt[..., jnp.newaxis]
    num = ln_ratio * cumsum_excl + alpha * C
    # 1e-10 Pa: division-safety floor only (real layer thicknesses are far
    # above it; matches the module's other dp floors).
    return -num / jnp.maximum(dp, 1e-10)


def compute_omega_hybrid(
    mass_flux: jax.Array,
    p_s: jax.Array,
    dp_s_dt: jax.Array,
    coord: HybridSigmaPressureCoordinate,
) -> jax.Array:
    """Compute pressure velocity omega for hybrid coordinates.

    omega_k = B_full_k * dp_s/dt + F_k_full

    where F_k_full is the mass flux interpolated to full levels.

    Parameters
    ----------
    mass_flux : jax.Array
        Mass flux at half-levels, shape (..., nlev+1).
    p_s : jax.Array
        Surface pressure, shape (...,).
    dp_s_dt : jax.Array
        Surface pressure tendency, shape (...,).
    coord : HybridSigmaPressureCoordinate

    Returns
    -------
    jax.Array
        Pressure velocity at full levels, shape (..., nlev).
    """
    F_full = 0.5 * (mass_flux[..., :-1] + mass_flux[..., 1:])  # (..., nlev)

    omega = coord.B_full * dp_s_dt[..., None] + F_full

    return omega


# ==============================================================================
# Height-based vertical coordinate (non-hydrostatic)
# ==============================================================================

class HeightCoordinate(NamedTuple):
    """Height-based terrain-following vertical coordinate (z-star).

    The z-star coordinate is defined as:

        z* = H · (z - z_s) / (H - z_s)

    where H is the model top height and z_s is the surface elevation.
    z* ranges from 0 at the surface to H at the model top.

    Levels are indexed top-to-bottom: k=0 is the model top (z*=H),
    k=n_levels-1 is the lowest level (z* near 0).

    The reference state (rho_0, theta_0, pi_0) is a 1D hydrostatically
    balanced profile used for reference-state subtraction in the
    compressible Euler equations.

    Fields
    ------
    n_levels : int
        Number of full levels.
    H : float
        Model top height [m].
    z_full : jax.Array
        z* at full (mid-) levels [m], shape (nlev,). Top-to-bottom.
    z_half : jax.Array
        z* at half (interface) levels [m], shape (nlev+1,). Top-to-bottom.
        z_half[0] = H (model top), z_half[-1] = 0 (surface).
    dz : jax.Array
        Layer thickness dz* [m], shape (nlev,).
        dz[k] = z_half[k] - z_half[k+1] (positive since top-to-bottom).
    dz_half : jax.Array
        Distance between adjacent full levels [m], shape (nlev-1,).
        dz_half[k] = z_full[k] - z_full[k+1].
    rho_ref : jax.Array
        Reference density at full levels [kg/m^3], shape (nlev,).
    theta_ref : jax.Array
        Reference potential temperature at full levels [K], shape (nlev,).
    exner_ref : jax.Array
        Reference Exner function at full levels [-], shape (nlev,).
    exner_ref_half : jax.Array
        Reference Exner function at half levels [-], shape (nlev+1,).
    rho_ref_half : jax.Array
        Reference density at half levels [kg/m^3], shape (nlev+1,).
    u_geo0, v_geo0 : jax.Array | None
        Geostrophic reference wind profiles [m/s], shape (nlev,) or None.
        SAM ``coriolis.f90`` applies the Coriolis force to the DEPARTURE
        from the geostrophic wind — ``dudt += f·(v−vg0)``,
        ``dvdt −= f·(u−ug0)`` — so the large-scale balanced mean wind is
        not spuriously spun up by an inertial oscillation. ``None`` (the
        default) means zero reference wind (the RCE / no-mean-wind case),
        which reduces to applying Coriolis to the full wind. For GATE/LBA
        these carry the prescribed initial/geostrophic sounding wind.
    """
    n_levels: int
    H: float
    z_full: jax.Array
    z_half: jax.Array
    dz: jax.Array
    dz_half: jax.Array
    rho_ref: jax.Array
    theta_ref: jax.Array
    exner_ref: jax.Array
    exner_ref_half: jax.Array
    rho_ref_half: jax.Array
    u_geo0: jax.Array | None = None
    v_geo0: jax.Array | None = None


class TerrainMetric(NamedTuple):
    """Terrain-following coordinate metric terms.

    These are 3D arrays that depend on horizontal position through
    the surface elevation z_s(x, y).

    The Jacobian J = dz/dz* = (H - z_s) / H maps between z* and
    physical z. All vertical derivatives in z* must be divided by J
    to get derivatives in physical z.

    Fields
    ------
    z_s : jax.Array
        Surface elevation [m], shape (6, n, n).
    jacobian : jax.Array
        dz/dz* = (H - z_s) / H, shape (6, n, n). Always > 0.
    z_full_3d : jax.Array
        Physical z at full levels [m], shape (6, n, n, nlev).
    z_half_3d : jax.Array
        Physical z at half levels [m], shape (6, n, n, nlev+1).
    """
    z_s: jax.Array
    jacobian: jax.Array
    z_full_3d: jax.Array
    z_half_3d: jax.Array


def _default_theta_ref(z: jax.Array) -> jax.Array:
    """Default reference potential temperature: isothermal at 300 K.

    For an isothermal atmosphere T=T0, the potential temperature is:
        theta(z) = T0 * (p0/p(z))^kappa
    But for simplicity, we use a constant theta_0 = 300 K.
    """
    return jnp.full_like(z, 300.0)


def compute_reference_state(
    z: jax.Array,
    theta_ref_fn: Callable[[jax.Array], jax.Array],
    p_sfc: float | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute 1D reference state by integrating hydrostatic balance.

    Given theta_0(z), integrate the hydrostatic equation:

        d(pi_0)/dz = -g / (c_p · theta_0(z))

    where pi = (p / p_0)^kappa is the dimensionless Exner function.

    Then recover density from the equation of state:

        rho_0 = p / (R_d · theta_0 · pi_0)

    Two boundary-condition modes:

    1. **Top-down (legacy, default when p_sfc is None)**: integrate
       downward from a model-top BC ``p_top = p_0 * exp(-g * z_top /
       (R_d * 250))`` with hardcoded T_avg=250 K column-mean. This
       gives the wrong pi at high-pressure levels for tall domains:
       for Wing 2018 RCE300 with a 33 km model top, ``pi(z=550m) =
       1.027`` instead of the correct 0.987 (4% over-estimate;
       12 K too-hot diagnosed T at lowest model level). Kept as
       DEFAULT for backward-compat with hundreds of existing tests
       that assert specific reference-state values.

    2. **Bottom-up (iter-95, opt-in via p_sfc)**: integrate UPWARD
       from a known surface BC ``pi_sfc = (p_sfc/p_ref)^kappa``.
       Scientifically correct; matches Wing 2018 T at z=550m to
       0.07 K (vs 12 K error in legacy mode). Use this for any IC
       where surface pressure is known, especially RCE / RCEMIP1
       / aquaplanet / Held-Suarez setups.

    iter-95 motivation: the legacy BC error broke surface-flux
    coupling in the plane CRM (air ~9 K above prescribed SST=300 K
    driving the wrong sign of heat flux), preventing convection
    initiation even after 9+ sim-days at 12x12 AND 32x32 domains.
    See CRM_implementation.md iter-95 for the full diagnostic
    chain.

    Parameters
    ----------
    z : jax.Array
        Height values [m], shape (n_points,). Must be sorted
        top-to-bottom (decreasing).
    theta_ref_fn : callable
        Function theta_0(z) -> potential temperature [K].
    p_sfc : float, optional
        Surface pressure (at z=0) [Pa]. If provided, switches to
        the bottom-up integration mode (iter-95 correct BC). For
        Wing 2018 RCE cases, pass 101480.0 Pa. If None (default),
        uses the legacy top-down BC for backward-compat.

    Returns
    -------
    rho_0 : jax.Array
        Reference density [kg/m^3], shape (n_points,).
    theta_0 : jax.Array
        Reference potential temperature [K], shape (n_points,).
    exner_0 : jax.Array
        Reference Exner function [-], shape (n_points,).
    """
    g = constants.g
    c_p = constants.c_pd
    R_d = constants.R_d
    p_0 = constants.p_ref

    theta_0 = theta_ref_fn(z)
    z.shape[0]
    dz_vals = jnp.diff(z)  # (n-1,) — negative since z decreasing
    integrand = -g / (c_p * theta_0)  # (n,)
    integrand_avg = 0.5 * (integrand[:-1] + integrand[1:])  # (n-1,)

    if p_sfc is None:
        # --------------- LEGACY: top-down integration ----------
        z_top = z[0]
        T_avg = 250.0  # rough average temperature for scale height
        p_top = p_0 * jnp.exp(-g * z_top / (R_d * T_avg))
        pi_top = (p_top / p_0) ** (R_d / c_p)
        d_pi = integrand_avg * dz_vals  # POSITIVE (negative * negative)
        pi_increments = jnp.cumsum(d_pi)  # (n-1,)
        exner_0 = jnp.concatenate(
            [jnp.array([pi_top]), pi_top + pi_increments]
        )
    else:
        # --------------- iter-95: bottom-up integration --------
        # pi_sfc at z=0 from known p_sfc.
        pi_sfc = (p_sfc / p_0) ** (R_d / c_p)
        # Extrapolate from z=0 (where pi=pi_sfc) to z[-1] (lowest
        # model level, > 0) using local d(pi)/dz at theta(z=0).
        theta_sfc = theta_ref_fn(jnp.array([0.0]))[0]
        pi_lowest = pi_sfc + (-g / (c_p * theta_sfc)) * z[-1]
        # Walk upward from z[-1] (pi_lowest) to z[0] (top).
        # d_pi_up[i] = -integrand_avg[i] * (-dz_vals[i]) = increment
        # going UP (NEGATIVE since pi decreases upward).
        d_pi_up = integrand_avg * dz_vals  # POSITIVE = going DOWN
        d_pi_up = -d_pi_up  # NEGATIVE = going UP
        # Reverse cumsum: pi[k] = pi_lowest + sum_{j=k}^{n-2} d_pi_up[j].
        pi_increments_reversed = jnp.cumsum(d_pi_up[::-1])  # (n-1,)
        pi_increments = pi_increments_reversed[::-1]  # (n-1,)
        exner_0 = jnp.concatenate(
            [pi_lowest + pi_increments, jnp.array([pi_lowest])]
        )

    # Fail fast on a non-physical reference Exner. A constant-θ
    # (isentropic) atmosphere reaches exner = 0 (p = T = 0) at
    # z = c_p·θ/g (≈ 30.7 km for θ = 300 K), so a tall model top with the
    # default constant reference silently yields exner_0 ≤ 0 → rho_0 = NaN
    # → a step-1 NaN far from the real cause. Raise at construction
    # instead with an actionable message. Concrete-only (skipped if z /
    # theta_ref_fn are traced — values aren't available then).
    try:
        exner_host = np.asarray(exner_0)
        z_host = np.asarray(z)
    except jax.errors.TracerArrayConversionError:
        exner_host = None
    if exner_host is not None and (
        not np.all(np.isfinite(exner_host))
        or float(np.min(exner_host)) <= 0.0
    ):
        k_bad = int(np.nanargmin(exner_host))
        raise ValueError(
            "compute_reference_state produced a non-physical reference "
            f"Exner (min={float(np.nanmin(exner_host)):.4g} at level index "
            f"{k_bad}, z={float(z_host[k_bad]):.0f} m). The theta_ref "
            "profile cannot hydrostatically support this column: a "
            "constant potential temperature is ISENTROPIC and reaches "
            "exner=0 at z = c_p·theta/g (~30.7 km for theta=300 K). For "
            "deep model tops (>~30 km) pass a theta_ref_fn whose theta "
            "increases aloft (tropopause + stratosphere sounding), e.g. "
            "the RCEMIP/Wing-2018 profile, instead of the constant-300 K "
            "default."
        )

    # Recover density from equation of state:
    # p = p_0 * pi^(c_p/R_d)
    # rho = p / (R_d * T) = p / (R_d * theta * pi)
    pressure = p_0 * exner_0 ** (c_p / R_d)
    rho_0 = pressure / (R_d * theta_0 * exner_0)

    return rho_0, theta_0, exner_0


def create_height_coordinate(
    n_levels: int,
    H: float,
    theta_ref_fn: Callable[[jax.Array], jax.Array] | None = None,
    p_sfc: float | None = None,
) -> HeightCoordinate:
    """Create a uniformly-spaced height-based (z-star) vertical coordinate.

    Parameters
    ----------
    n_levels : int
        Number of full vertical levels.
    H : float
        Model top height [m].
    theta_ref_fn : callable, optional
        Function theta_0(z) -> potential temperature [K].
        Default: constant 300 K (isothermal reference).
    p_sfc : float, optional
        Surface pressure [Pa]. When provided, switches the
        hydrostatic integration to bottom-up mode with surface
        BC ``pi_sfc=(p_sfc/p_ref)^kappa`` (iter-95 fix). For
        Wing 2018 RCE cases pass 101480.0 Pa. Default (None) uses
        the legacy top-down BC for backward-compat. See
        ``compute_reference_state`` docstring.

    Returns
    -------
    HeightCoordinate
        The vertical coordinate with precomputed reference state.
    """
    # z* grid: top-to-bottom (z_half[0] = H, z_half[-1] = 0)
    # Use JAX default dtype (float64 when x64 is enabled, float32 otherwise)
    z_half = jnp.linspace(H, 0.0, n_levels + 1)
    return create_height_coordinate_from_z_half(
        z_half, theta_ref_fn=theta_ref_fn, p_sfc=p_sfc,
    )


def create_height_coordinate_from_z_half(
    z_half: jax.Array,
    theta_ref_fn: Callable[[jax.Array], jax.Array] | None = None,
    p_sfc: float | None = None,
) -> HeightCoordinate:
    """HeightCoordinate from an EXPLICIT ``z_half`` interface array.

    For CUSTOM vertical grids that are neither uniform
    (:func:`create_height_coordinate`) nor geometrically stretched
    (:func:`create_stretched_height_coordinate`) — e.g. SAM's ``grd``-file
    levels (``read_sam_grd``), which are dz=50 m uniform in the boundary layer,
    ~100 m uniform through the deep-convection layer, then stretched aloft.

    ``z_half`` MUST be top-to-bottom (``z_half[0]`` = model top H,
    ``z_half[-1]`` = 0 surface, strictly decreasing) — the same convention as
    the other builders. The shared ``z_half → (z_full, dz, dz_half, reference
    state)`` core lives here; the uniform builder above just supplies a
    ``linspace`` ``z_half``.
    """
    if theta_ref_fn is None:
        theta_ref_fn = _default_theta_ref
    z_half = jnp.asarray(z_half)
    if z_half.ndim != 1 or z_half.shape[0] < 3:
        raise ValueError(
            f"z_half must be 1-D with >=3 interfaces, got shape "
            f"{tuple(z_half.shape)}.")
    if not bool(jnp.all(jnp.diff(z_half) < 0.0)):
        raise ValueError(
            "z_half must be STRICTLY DECREASING top-to-bottom "
            "(z_half[0]=H top, z_half[-1]=0 surface).")
    z_full = 0.5 * (z_half[:-1] + z_half[1:])  # (nlev,)
    dz = z_half[:-1] - z_half[1:]  # (nlev,) positive
    dz_half = z_full[:-1] - z_full[1:]  # (nlev-1,) positive

    # Compute reference state at full and half levels
    rho_ref, theta_ref, exner_ref = compute_reference_state(
        z_full, theta_ref_fn, p_sfc=p_sfc,
    )
    rho_ref_half, _, exner_ref_half = compute_reference_state(
        z_half, theta_ref_fn, p_sfc=p_sfc,
    )

    return HeightCoordinate(
        n_levels=int(z_full.shape[0]),
        H=float(z_half[0]),
        z_full=z_full,
        z_half=z_half,
        dz=dz,
        dz_half=dz_half,
        rho_ref=rho_ref,
        theta_ref=theta_ref,
        exner_ref=exner_ref,
        exner_ref_half=exner_ref_half,
        rho_ref_half=rho_ref_half,
    )


def create_stretched_height_coordinate(
    n_levels: int,
    H: float,
    dz_sfc: float = 50.0,
    stretching: float | None = None,
    theta_ref_fn: Callable[[jax.Array], jax.Array] | None = None,
    p_sfc: float | None = None,
) -> HeightCoordinate:
    """Geometrically-stretched height-based (z-star) vertical coordinate.

    Layer thicknesses grow geometrically from ``dz_sfc`` at the
    surface to whatever value the constraint
    ``sum_k dz_sfc · r^k == H`` requires::

        dz[k] = dz_sfc · r^k  (k = 0 .. n_levels - 1, k=0 = surface)

    The stretching ratio ``r`` is either supplied by the caller or
    solved for given ``(n_levels, H, dz_sfc)``. The closed-form
    constraint is ``dz_sfc · (1 - r^n) / (1 - r) == H``; we solve via
    Newton iteration on host arithmetic (the ratio is a static scalar,
    no per-step cost).

    Layout convention matches :func:`create_height_coordinate`:
    top-to-bottom storage so ``z_half[0] = H`` (model top) and
    ``z_half[-1] = 0`` (surface). The geometric stretching means
    ``dz[-1] = dz_sfc`` (thinnest at the surface, the last index) and
    ``dz[0]`` is the thickest layer at the model top.

    Parameters
    ----------
    n_levels : int
        Number of full vertical levels.
    H : float
        Model top height [m].
    dz_sfc : float
        Lowest-layer thickness [m]. RCEMIP1 spec uses ~50 m.
    stretching : float or None
        Geometric ratio ``dz[k+1] / dz[k]``. If None (default), solved
        from ``(n_levels, H, dz_sfc)`` via Newton iteration so the
        column sum hits ``H`` exactly. Typical values 1.05–1.15 for
        atmospheric CRM grids; values > 1.2 risk poorly-resolved
        upper troposphere.
    theta_ref_fn : callable, optional
        Function ``theta_0(z) -> potential temperature [K]``. Default
        is :func:`_default_theta_ref` (isothermal 300 K) — fine for
        a smoke/test reference state, but UNREALISTIC for an RCEMIP
        or other deep CRM column because the actual atmosphere is
        strongly stratified (potential temperature increases by
        ~100 K between the surface and the tropopause). Pass an
        explicit Wing 2018-style sounding for production RCE runs.

    Returns
    -------
    HeightCoordinate
    """
    if theta_ref_fn is None:
        theta_ref_fn = _default_theta_ref
    if dz_sfc <= 0.0:
        raise ValueError(f"dz_sfc={dz_sfc} must be positive.")
    if n_levels < 2:
        raise ValueError(
            f"n_levels={n_levels} must be >= 2 for a stretched grid."
        )
    if dz_sfc * n_levels >= H:
        raise ValueError(
            f"dz_sfc={dz_sfc} too large for H={H}, n_levels={n_levels}: "
            f"uniform layers would exceed H. Reduce dz_sfc or increase "
            f"n_levels."
        )
    # Codex review: callers that pass BOTH stretching and a non-default
    # dz_sfc are mixing two over-determined constraints (the geometric
    # sum must equal H). Reject explicitly rather than silently
    # overriding dz_sfc — earlier behaviour was easy to misuse.
    if stretching is not None and dz_sfc != 50.0:
        raise ValueError(
            f"Pass either `stretching` (and let `dz_sfc` derive from "
            f"H, n_levels, r) OR `dz_sfc` (and let `stretching` be "
            f"Newton-solved). Got both stretching={stretching} and "
            f"non-default dz_sfc={dz_sfc}; the (stretching, dz_sfc, "
            f"n_levels, H) tuple is over-determined for a geometric "
            f"column."
        )

    import numpy as _np

    n = n_levels
    if stretching is None:
        # Solve dz_sfc · (r^n - 1) / (r - 1) = H for r > 1 via
        # BRACKETED BISECTION. Newton fails here because r=1 is a
        # spurious fixed point of f(r) = dz_sfc·(r^n-1) - H·(r-1)
        # and the iteration drifts back to that root for many start
        # points (Codex iter-1). Bisection on (1+eps, r_max) is
        # robust and the bracket is monotone — f(1+eps) < 0,
        # f(r_max) > 0 by construction for any r_max with
        # dz_sfc·r_max^(n-1) > H.
        def _g(r):
            return dz_sfc * (r ** n - 1.0) / (r - 1.0) - H

        r_lo = 1.0 + 1.0e-9
        r_hi = 10.0
        # Ensure r_hi brackets the root (g(r_hi) > 0). For any
        # reasonable (n, H, dz_sfc), r=10 produces a huge sum; if
        # not, double until it does.
        for _ in range(20):
            if _g(r_hi) > 0.0:
                break
            r_hi *= 2.0
        if _g(r_lo) >= 0.0 or _g(r_hi) <= 0.0:
            raise ValueError(
                f"No stretched-grid solution found in r ∈ ({r_lo}, "
                f"{r_hi}) for n_levels={n}, H={H}, dz_sfc={dz_sfc}. "
                f"Inputs may be unphysical."
            )
        for _ in range(200):
            r_mid = 0.5 * (r_lo + r_hi)
            g_mid = _g(r_mid)
            if abs(g_mid) < 1.0e-12 * H or (r_hi - r_lo) < 1.0e-14:
                r_lo = r_hi = r_mid
                break
            if g_mid > 0.0:
                r_hi = r_mid
            else:
                r_lo = r_mid
        r = 0.5 * (r_lo + r_hi)
        final_sum = dz_sfc * (r ** n - 1.0) / (r - 1.0)
        rel_residual = abs(final_sum - H) / H
        if rel_residual > 1.0e-6:
            raise ValueError(
                f"Stretching solve did not converge: r={r}, "
                f"sum(dz)={final_sum:.6f} vs H={H} "
                f"(rel residual={rel_residual:.3e})."
            )
        stretching = float(r)
    else:
        # Caller supplied an explicit stretching ratio: derive the
        # consistent ``dz_sfc`` from it so the column sums to H exactly
        # and the layer ratios stay geometric throughout.
        dz_sfc = float(
            H * (stretching - 1.0) / (stretching ** n - 1.0)
        )
    if stretching <= 1.0:
        raise ValueError(
            f"stretching={stretching} must be > 1.0 (geometric ratio)."
        )

    # Layer thicknesses surface→top: dz_sfc · r^k. With the consistent
    # (dz_sfc, r) pair the sum is H to ~1e-12; the trailing residual
    # adjustment hides only floating-point round-off without
    # measurably perturbing the top layer.
    k_sfc_to_top = _np.arange(n_levels)
    dz_sfc_to_top = dz_sfc * stretching ** k_sfc_to_top
    residual = H - dz_sfc_to_top.sum()
    dz_sfc_to_top[-1] += residual
    # Storage is top-to-bottom, so reverse.
    dz_np = dz_sfc_to_top[::-1]
    dz = jnp.asarray(dz_np)
    z_half_np = _np.concatenate([
        _np.array([H]),
        H - _np.cumsum(dz_np),
    ])
    z_half = jnp.asarray(z_half_np)
    z_full = 0.5 * (z_half[:-1] + z_half[1:])
    dz_half = z_full[:-1] - z_full[1:]

    rho_ref, theta_ref, exner_ref = compute_reference_state(
        z_full, theta_ref_fn, p_sfc=p_sfc,
    )
    rho_ref_half, _, exner_ref_half = compute_reference_state(
        z_half, theta_ref_fn, p_sfc=p_sfc,
    )

    return HeightCoordinate(
        n_levels=n_levels,
        H=H,
        z_full=z_full,
        z_half=z_half,
        dz=dz,
        dz_half=dz_half,
        rho_ref=rho_ref,
        theta_ref=theta_ref,
        exner_ref=exner_ref,
        exner_ref_half=exner_ref_half,
        rho_ref_half=rho_ref_half,
    )


def compute_terrain_metric(
    z_s: jax.Array,
    height_coord: HeightCoordinate,
) -> TerrainMetric:
    """Compute 3D terrain-following coordinate metrics.

    The z-star to physical z transformation is:

        z(z*, x, y) = z_s(x, y) + z* · (H - z_s(x, y)) / H

    The Jacobian is:

        J = dz/dz* = (H - z_s) / H

    Parameters
    ----------
    z_s : jax.Array
        Surface elevation [m], shape (6, n, n).
    height_coord : HeightCoordinate
        Vertical coordinate.

    Returns
    -------
    TerrainMetric
        3D coordinate metric terms.
    """
    H = height_coord.H

    # Jacobian: J = (H - z_s) / H, shape same as z_s
    jacobian = (H - z_s) / H

    # Physical z at full levels: z = z_s + z* * (H - z_s) / H
    # z_full is (nlev,).  z_s can be (6,n,n) or (nCells,) — broadcast generically.
    n_spatial = z_s.ndim  # 3 for cubed-sphere, 1 for MPAS
    z_full_bc = height_coord.z_full.reshape((1,) * n_spatial + (-1,))
    z_half_bc = height_coord.z_half.reshape((1,) * n_spatial + (-1,))
    z_full_3d = z_s[..., None] + z_full_bc * jacobian[..., None]
    z_half_3d = z_s[..., None] + z_half_bc * jacobian[..., None]

    return TerrainMetric(
        z_s=z_s,
        jacobian=jacobian,
        z_full_3d=z_full_3d,
        z_half_3d=z_half_3d,
    )
