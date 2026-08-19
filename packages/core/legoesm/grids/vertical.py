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
    """Warn when the level count was MEASURED to be an expensive one.

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

    ``where`` names the coordinate being built so the message points at the
    call the user can change.
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
        f"{where}: {n_levels} vertical levels was MEASURED expensive on this "
        f"model — {cost} picoseconds per cell per level against {best} for "
        f"the cheapest counts measured, a factor of {cost / best:.1f}. At "
        f"163,842 cells the step is 22.6 ms at 30 levels and 8.5 ms at 32: "
        f"more work, a third of the time. Level counts measured cheap: "
        f"{', '.join(str(n) for n in _LEVELS_MEASURED_FAST)}. The cause is "
        f"NOT established — byte alignment of the column stride and cache "
        f"capacity were both tested and both refuted — so this is a table of "
        f"measurements, not a rule, and counts absent from it are unmeasured "
        f"rather than cheap. If {n_levels} is a physics requirement, keep it "
        f"and expect the cost. Re-measure on different hardware or a "
        f"different compiler before trusting any of it.",
        stacklevel=3,
    )


def create_sigma_coordinate(
    n_levels: int,
    sigma_top: float = 0.01,
    dtype=None,
    tropopause_refine: float = 1.0,
    sigma_refine: float = 0.12,
    refine_width: float = 0.45,
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
    # None: resolved from the precision policy, because the stride that
    # matters is the STATE arrays' and this coordinate may be higher
    # precision than they are.
    warn_if_unaligned_levels(n_levels, where="create_sigma_coordinate")
    if tropopause_refine == 1.0:
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


VERTICAL_ADVECTION_SCHEMES = ("upwind", "van_leer")


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
    from legoesm.core.flux_limiters import (
        grad_safe_ratio, ratio_grad_floor, van_leer_limiter,
    )

    nlev = field.shape[-1]
    if nlev < 4:
        raise ValueError(
            f"the van-Leer sigma reconstruction needs at least 4 vertical "
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
    dcp = jnp.pad(sigma_coord.dsigma_full, (2, 2), mode="edge")  # (nlev+3,)
    dc_up, dc_loc, dc_dn = dcp[0:nlev + 1], dcp[1:nlev + 2], dcp[2:nlev + 3]
    # MUSCL face weights: donor half-thickness / centre-to-centre distance.
    # Exactly 0.5 each on a uniform grid.
    dsp = jnp.pad(sigma_coord.dsigma, (1, 1), mode="edge")  # (nlev+2,)
    d_above, d_below = dsp[:-1], dsp[1:]                    # (nlev+1,)
    d_sum = d_above + d_below

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
            f"unknown vertical advection scheme {scheme!r}; expected one of "
            f"{VERTICAL_ADVECTION_SCHEMES}"
        )
    return _vertical_advection_upwind_sigma(field, sigma_dot, sigma_coord)


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
    warn_if_unaligned_levels(n_levels, where="create_hybrid_coordinate")
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


def standard_hybrid_levels(
    n_levels: int = 40,
    p_ref: float = constants.p_ref,
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

    Returns
    -------
    HybridSigmaPressureCoordinate
    """
    if n_levels <= 25:
        # Moderate resolution: 10 hPa top, mild stretching
        p_top = 1000.0
        stretching = 1.5
        transition_exponent = 2
    elif n_levels <= 45:
        # Standard resolution: 2 hPa top, good BL resolution
        p_top = 200.0
        stretching = 2.5
        transition_exponent = 3
    else:
        # High resolution: 0.1 hPa top, strong BL + strat resolution
        p_top = 10.0
        stretching = 2.5
        transition_exponent = 3

    return make_hybrid_levels(
        n_levels, p_top_Pa=p_top, p_ref=p_ref,
        transition_exponent=transition_exponent, stretching=stretching,
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


def get_eta_level(
    ak: jax.Array,
    bk: jax.Array,
    p_s: jax.Array,
    pscale: float | None = None,
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 634: FV3 hybrid → (pf, ph) log-mean full-level pressure.

    Faithful JAX port of FV3 ``get_eta_level``
    (tools/fv_eta.F90:1859-1890).  Computes:

        ph[k]  = ak[k] + bk[k]·p_s            # half-level pressure
        pf[k]  = (ph[k+1] - ph[k]) / log(ph[k+1]/ph[k])  # log-mean full

    At the top edge (k=0) FV3 distinguishes:
        - ak[0] > 1e-8 → standard log-mean (avoids log(0))
        - ak[0] ≤ 1e-8 → use kappa-based limit:
          pf[0] = (ph[1] - ph[0]) · kappa/(kappa+1)

    The FV3 ``kappa`` is R_d / c_p (here ``constants.kappa``).

    Differs from legoESM's ``pressure_from_hybrid(full=True)``
    which uses pre-computed ``A_full``/``B_full`` (linear midpoint
    or scheme-dependent); FV3 uses the logarithmic mean.  Both are
    valid full-level definitions; this helper makes FV3-faithful
    available standalone.

    Parameters
    ----------
    ak : jax.Array, shape ``(npz+1,)``
        Hybrid A coefficient at half levels.
    bk : jax.Array, shape ``(npz+1,)``
        Hybrid B coefficient at half levels.
    p_s : jax.Array, shape ``(...,)``
        Surface pressure (Pa).
    pscale : float, optional
        Multiplier applied to ph (FV3 lines 1874-1878).  Default
        None = no scaling.

    Returns
    -------
    pf : jax.Array, shape ``(..., npz)``
        Full-level pressure (log-mean).
    ph : jax.Array, shape ``(..., npz+1)``
        Half-level pressure.
    """
    # Broadcast p_s to a trailing level axis
    ps = p_s[..., None]                          # (..., 1)
    # ph[k] = ak[k] + bk[k]·p_s
    # FV3 line 1869: ph(1) = ak(1) (no p_s contribution at top edge)
    # FV3 lines 1870-1872: ph(k) = ak(k) + bk(k)·p_s for k=2..npz+1
    # In 0-indexed JAX: ph[0] = ak[0]; ph[k] = ak[k] + bk[k]·p_s for k>=1
    # We use the vectorized form ak + bk·p_s — equivalent if bk[0] = 0
    # (FV3 convention).  Add an explicit override for ph[0] to match the
    # FV3 special case for safety.
    ph = ak + bk * ps                            # (..., npz+1)
    # Override top edge to exactly ak[0] (FV3 line 1869)
    ph = ph.at[..., 0].set(ak[0])
    if pscale is not None:
        ph = pscale * ph

    # pf[k] = (ph[k+1] - ph[k]) / log(ph[k+1]/ph[k])
    dph = ph[..., 1:] - ph[..., :-1]             # (..., npz)
    # Top-edge special branch (FV3 lines 1880-1884)
    log_ratio = jnp.log(
        jnp.where(ph[..., 1:] > 0.0, ph[..., 1:], 1.0)
        / jnp.where(ph[..., :-1] > 0.0, ph[..., :-1], 1.0)
    )
    safe_log = jnp.where(jnp.abs(log_ratio) > 1e-30, log_ratio, 1.0)
    pf_general = dph / safe_log
    # Top branch: if ak[0] <= 1e-8, replace pf[0] with kappa-limit
    kappa = constants.kappa
    pf_top_kappa = dph[..., 0] * (kappa / (kappa + 1.0))
    use_kappa = ak[0] <= 1e-8
    pf_top = jnp.where(use_kappa, pf_top_kappa, pf_general[..., 0])
    pf = pf_general.at[..., 0].set(pf_top)
    return pf, ph


def compute_dz_fv3(
    km: int, ztop: float,
) -> jax.Array:
    """FV3_3D iter 635: FV3 initial uniform-with-stretched-edges dz.

    Faithful JAX port of FV3 ``compute_dz``
    (tools/fv_eta.F90:1894-1928).  Builds an initial layer-
    thickness array used as a starting point for FV3's hybrid-z
    setup (FV3 then iterates to satisfy ztop and other
    constraints).

    Algorithm:
        dz_uniform = ztop / km
        dz[0]   = 2·dz_uniform     # top (stretched)
        dz[km-1] = 0.5·dz_uniform  # bottom (compressed)
        dz[1..km-2] = dz_uniform   # interior

    Note: total height = (km + 0.5)·ztop/km > ztop by design
    (this is an initial guess; FV3 later iterates).

    Parameters
    ----------
    km : int
        Number of levels.
    ztop : float
        Approximate top height (m).

    Returns
    -------
    dz : jax.Array, shape ``(km,)``
        Layer thicknesses (top→bottom indexing, FV3 convention).
    """
    dz_uniform = ztop / km
    dz = jnp.full((km,), dz_uniform)
    dz = dz.at[0].set(2.0 * dz_uniform)
    dz = dz.at[km - 1].set(0.5 * dz_uniform)
    return dz


def zflip(q: jax.Array, axis: int = -1) -> jax.Array:
    """FV3_3D iter 635: flip array along vertical axis.

    Faithful JAX port of FV3 ``zflip`` (tools/fv_eta.F90:2482-2497).
    Reverses level ordering of ``q`` along ``axis``.  Useful to
    convert between FV3 top-down (k=1 at model top) and bottom-up
    conventions.

    Parameters
    ----------
    q : jax.Array
        Field to flip.
    axis : int, default -1
        Vertical axis to flip.

    Returns
    -------
    jax.Array
        ``q`` flipped along ``axis``.  Same shape as input.
    """
    return jnp.flip(q, axis=axis)


def set_external_eta(
    ak: jax.Array, bk: jax.Array, eps: float = 1.0e-7,
) -> tuple[jax.Array, int]:
    """FV3_3D iter 637: derive (ptop, ks) from external ak/bk arrays.

    Faithful JAX port of FV3 ``set_external_eta``
    (tools/fv_eta.F90:788-807).  Given hybrid coefficients
    ``ak`` (Pa) and ``bk`` (dimensionless), returns::

        ptop = ak[0]                              # model top pressure (Pa)
        ks   = max k where bk[k] < eps  -  1     # # pure-pressure layers

    The "-1" converts FV3's level count to layer count (FV3 stores
    levels at edges; layers are between edges).

    Parameters
    ----------
    ak : jax.Array, shape ``(km+1,)``
        Hybrid A coefficient at half levels (Pa).
    bk : jax.Array, shape ``(km+1,)``
        Hybrid B coefficient at half levels.
    eps : float, default 1e-7
        Threshold to classify a level as "pure pressure" (bk < eps).

    Returns
    -------
    ptop : jax.Array (scalar)
        Top-of-model pressure (Pa).
    ks : int
        Number of pure-pressure LAYERS.
    """
    ptop = ak[0]
    # ks (level count) = max k with bk[k] < eps; in 0-indexed:
    #   ks = (count of consecutive bk < eps from k=0) - 1
    # but FV3 also counts ks even if subsequent bk increase; we take the
    # largest k.  Use jnp.argmax over the reverse-sorted boolean array.
    is_pure = bk < eps
    # Cumulative AND backward: only valid as long as all preceding were pure
    # Actually FV3 sets ks = k whenever bk[k] < eps; so ks ends up being
    # the LAST index where bk[k] < eps (using a sweep from low to high).
    # In JAX: ks_level = argmax(reverse[bk < eps]) interpreted as last True
    # index.  Simpler: use jnp.where + max.
    idx = jnp.arange(bk.shape[0])
    # Last index where is_pure is True
    masked_idx = jnp.where(is_pure, idx, -1)
    ks_level = int(jnp.max(masked_idx))
    # FV3: 1-indexed levels, ks = max k where bk(k) < eps; then ks = ks-1
    # 0-indexed: ks_level + 1 (1-indexed) → minus 1 → ks_level (0-indexed)
    ks = ks_level
    return ptop, ks


def compute_dz_L32() -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 638: FV3 L32 layer thicknesses + ztop.

    Faithful JAX port of FV3 ``compute_dz_L32``
    (tools/fv_eta.F90:2000-2067).  Builds the FV3-canonical
    32-layer vertical structure with ztop ≈ 60 km:

    Three blocks (FV3 1-indexed, see ze/dz arrays):
      - k=1, 2 (special bottom): dz[0]=75, dz[1]=112.5 m
      - k=3..23 (middle, k1=21): linear stretching to z1=10 km
        dz[k] = dz0 + (k-k0)·dz1
      - k=24..31 (upper, k2=8): linear stretching to z2=30 km
        dz[k] = dz0_new + (k-k0-k1)·dz2
      - k=32 (top): dz[31] = 2·dz[30]

    Then ``zflip`` reverses to top-down indexing (FV3 final
    convention).

    Returns
    -------
    dz : jax.Array, shape ``(32,)``
        Layer thicknesses (top→bottom indexing).
    ztop : jax.Array (scalar)
        Total height = sum(dz).
    """
    km = 32
    _k0, k1, k2 = 2, 21, 8
    z1, z2 = 10.0e3, 30.0e3
    dz0_init = 75.0

    # Build bottom-up dz in 1-indexed style then zflip at end
    dz = jnp.zeros((km,))
    # dz[0] = dz0; dz[1] = 1.5*dz0   (FV3 1-indexed k=1, 2)
    dz = dz.at[0].set(dz0_init)
    dz_special = 1.5 * dz0_init                       # 112.5
    dz = dz.at[1].set(dz_special)
    # ze[2] (1-indexed) = ze[3] in FV3 = dz[0] + dz[1] = 187.5
    ze3 = dz_special + dz0_init                        # 187.5

    # Middle block (FV3 k = k0+1..k0+k1 = 3..23 → 0-indexed [2, 22])
    dz0_mid = dz_special                               # 112.5
    dz1 = 2.0 * (z1 - ze3 - k1 * dz0_mid) / (k1 * (k1 - 1))
    # FV3 loop: do k = k0+1, k0+k1: dz[k] = dz0 + (k-k0)*dz1
    # 0-indexed k_python = k_fortran - 1
    # For k_fortran = 3..23 → k_python = 2..22; (k - k0) = (k_fortran - 2)
    k_arr = jnp.arange(2, 23)                          # 0-indexed
    k_minus_k0 = k_arr - 1                             # k_fortran - k0 in 1-indexed
    # k_fortran = k_python + 1; (k_fortran - k0) = (k_python - 1)
    dz_mid = dz0_mid + k_minus_k0 * dz1
    dz = dz.at[2:23].set(dz_mid)

    # After middle, ze[k0+k1+1 (1-indexed) = ze[24] = 0-indexed ze[23]]
    ze_after_mid = ze3 + float(jnp.sum(dz_mid))
    # Upper block (FV3 k = k0+k1+1..k0+k1+k2 = 24..31 → 0-indexed [23, 30])
    dz0_upper = float(dz[22])                          # dz[k1+k0] 1-indexed = dz[23]
    dz2 = 2.0 * (z2 - ze_after_mid - k2 * dz0_upper) / (k2 * (k2 - 1))
    k_arr_upper = jnp.arange(23, 31)
    k_minus_k0_k1 = k_arr_upper - 22                   # (k_fortran - k0 - k1) = (k_python - 22) when k_python = k_fortran - 1, k_fortran = k_python + 1, (k_python + 1 - 2 - 21) = k_python - 22
    dz_upper = dz0_upper + k_minus_k0_k1 * dz2
    dz = dz.at[23:31].set(dz_upper)

    # Top (FV3 k=km): dz[km-1] (0-indexed) = 2·dz[km-2]
    dz = dz.at[km - 1].set(2.0 * dz[km - 2])

    # zflip: FV3 dz was built bottom-up; flip to top-down
    dz_flipped = jnp.flip(dz)
    ztop = jnp.sum(dz_flipped)
    return dz_flipped, ztop


# iter-93: stored as numpy (not jnp) at module-top. jnp.asarray at
# import time eagerly dispatches to the default JAX backend (Metal
# on macOS), which currently rejects convert_element_type with
# "UNIMPLEMENTED: default_memory_space is not supported". That
# bricks `import legoesm` on Apple Silicon even for pure-Python
# unit tests. Defer jnp conversion to inside `set_eta_L60()` so
# only callers that actually need the FV3 L60 hybrid coord pay the
# JAX device-init cost.
_A60 = np.asarray([
    300.0000, 430.00000, 558.00000, 700.00000, 863.05803,
    1051.07995, 1265.75194, 1510.71101, 1790.05098, 2108.36604,
    2470.78817, 2883.03811, 3351.46002, 3883.05187, 4485.49315,
    5167.14603, 5937.04991, 6804.87379, 7780.84698, 8875.64338,
    10100.20534, 11264.35673, 12190.64366, 12905.42546, 13430.87867,
    13785.88765, 13986.77987, 14047.96335, 13982.46770, 13802.40331,
    13519.33841, 13144.59486, 12689.45608, 12165.28766, 11583.57006,
    10955.84778, 10293.60402, 9608.08306, 8910.07678, 8209.70131,
    7516.18560, 6837.69250, 6181.19473, 5552.39653, 4955.72632,
    4394.37629, 3870.38682, 3384.76586, 2937.63489, 2528.37666,
    2155.78385, 1818.20722, 1513.68173, 1240.03585, 994.99144,
    776.23591, 581.48797, 408.53400, 255.26520, 119.70243,
    0.0,
])
_B60 = np.asarray([
    0.00000, 0.00000, 0.00000, 0.00000, 0.00000,
    0.00000, 0.00000, 0.00000, 0.00000, 0.00000,
    0.00000, 0.00000, 0.00000, 0.00000, 0.00000,
    0.00000, 0.00000, 0.00000, 0.00000, 0.00000,
    0.00000, 0.00201, 0.00792, 0.01755, 0.03079,
    0.04751, 0.06761, 0.09097, 0.11746, 0.14690,
    0.17911, 0.21382, 0.25076, 0.28960, 0.32994,
    0.37140, 0.41353, 0.45589, 0.49806, 0.53961,
    0.58015, 0.61935, 0.65692, 0.69261, 0.72625,
    0.75773, 0.78698, 0.81398, 0.83876, 0.86138,
    0.88192, 0.90050, 0.91722, 0.93223, 0.94565,
    0.95762, 0.96827, 0.97771, 0.98608, 0.99347,
    1.0,
])


def set_eta_L60() -> tuple[jax.Array, jax.Array, jax.Array, int]:
    """FV3_3D iter 647: FV3 L60 hardcoded hybrid-coord ak/bk table.

    Faithful JAX port of the L60 ``a60`` / ``b60`` data tables in
    FV3 ``set_eta`` (tools/fv_eta.F90:45-85).

    The FV3 docstring notes: "The following L63 setting is the
    same as NCEP GFS's L64 except the top 3 layers".  Used as
    the FV3 reference for 60-layer baroclinic-instability and
    GFS-comparison tests.

    Returns
    -------
    ak : jax.Array, shape (61,)
        Hybrid A coefficient (Pa).
    bk : jax.Array, shape (61,)
        Hybrid B coefficient (dimensionless sigma).
    ptop : jax.Array (scalar)
        Top-of-model pressure = ak[0] = 300 Pa.
    ks : int
        Number of pure-pressure LAYERS = max index where bk < eps
        (from iter-637 set_external_eta).
    """
    ak = jnp.asarray(_A60)
    bk = jnp.asarray(_B60)
    ptop = ak[0]
    # ks = last index where bk < 1e-7
    eps = 1.0e-7
    idx = jnp.arange(bk.shape[0])
    masked = jnp.where(bk < eps, idx, -1)
    ks = int(jnp.max(masked))
    return ak, bk, ptop, ks


def hydro_eq(
    ak: jax.Array, bk: jax.Array,
    hs: jax.Array,
    drym: float = 1000.0e2,
    mountain: bool = False,
    area: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """FV3_3D iter 646: hydrostatic-equilibrium IC builder.

    Faithful JAX port of FV3 ``hydro_eq``
    (tools/init_hydro.F90:277-456), hybrid sigma-p branch
    (``hybrid_z=False``, hydrostatic-only).

    Reference profile:
        p1 = 250 hPa, z1 = 10 km · g (tropopause; geopotential)
        T1 = 200 K (isothermal above tropopause)
        T0 = 300 K (sea-level)
        a0 = 0.5·(T1 - T0)/z1
        c0 = T0/a0

    Algorithm:
        Surface pressure:
          if mountain: ps = mslp·exp(-1/(a0·R)·hs/(hs + c0))
                        (with global dps correction)
          else:        ps = drym (uniform)
        ph[k] = ak[k] + bk[k]·ps
        Build gz top-down:
          if ph[k] ≤ p1: gz[k] = gz[k+1] + R·T1·log(ph[k+1]/ph[k])
                                              (isothermal stratosphere)
          else:          gz[k] = c0/(1 + a0·R·log(ph[k]/ps)) + hs - c0
                                              (lapse-rate troposphere)
        pt[k] = (gz[k] - gz[k+1]) / (R·log(ph[k+1]/ph[k]))
        pt[k] = max(T1, pt[k])
        delp[k] = ph[k+1] - ph[k]

    Parameters
    ----------
    ak, bk : jax.Array, shape (km+1,)
        Hybrid coordinates (FV3 convention).
    hs : jax.Array, shape (...,)
        Surface geopotential (m²/s²).
    drym : float, default 100000 Pa
        Mean dry-mass surface pressure (used as mslp when mountain).
    mountain : bool, default False
        If True, ``ps`` follows topography via ``hs``.  Else uniform.
    area : jax.Array, shape (...,), optional
        Cell areas (used for dps mass correction if mountain).

    Returns
    -------
    ps : jax.Array, shape (...,)
        Surface pressure (Pa).
    delp : jax.Array, shape (..., km)
        Layer pressure thicknesses.
    pt : jax.Array, shape (..., km)
        Layer-mean temperature (K).
    """
    g = constants.g
    rdgas = constants.R_d

    p1 = 25000.0
    z1 = 10.0e3 * g
    t1 = 200.0
    t0 = 300.0
    a0 = (t1 - t0) / z1 * 0.5
    c0 = t0 / a0
    float(ak[0])

    # Surface pressure
    if mountain:
        mslp = 100917.4
        ps_init = mslp * jnp.exp(
            -1.0 / (a0 * rdgas) * hs / (hs + c0)
        )
        if area is not None:
            psm = jnp.sum(ps_init * area) / jnp.sum(area)
            dps = drym - psm
        else:
            dps = 0.0
        ps = ps_init + dps
    else:
        ps = jnp.full(hs.shape, drym)

    # ph[..., k] = ak[k] + bk[k]·ps
    km = ak.shape[0] - 1
    ph = ak + bk * ps[..., None]                # shape (..., km+1)

    # Build gz top-down from surface (gz[km] = hs)
    # Use a Python loop over k (km is static, ~32-101)
    gz = jnp.zeros(ps.shape + (km + 1,))
    gz = gz.at[..., km].set(hs)
    for k in range(km - 1, 0, -1):
        # Branch on ph[k] ≤ p1 (tropopause)
        ph_k = ph[..., k]
        ph_kp1 = ph[..., k + 1]
        gz_kp1 = gz[..., k + 1]
        # Stratosphere branch
        gz_strat = gz_kp1 + rdgas * t1 * jnp.log(
            jnp.maximum(ph_kp1, 1.0) / jnp.maximum(ph_k, 1.0)
        )
        # Troposphere branch
        ratio = ph_k / ps
        safe_log = jnp.log(jnp.maximum(ratio, 1e-30))
        denom = 1.0 + a0 * rdgas * safe_log
        gz_trop = c0 / denom + hs - c0
        gz_new = jnp.where(ph_k <= p1, gz_strat, gz_trop)
        gz = gz.at[..., k].set(gz_new)

    # k=0 (model top): same branch logic; ph[0]=ptop = ak[0]
    ph_0 = ph[..., 0]
    ph_1 = ph[..., 1]
    gz_strat_top = gz[..., 1] + rdgas * t1 * jnp.log(
        jnp.maximum(ph_1, 1.0) / jnp.maximum(ph_0, 1.0)
    )
    ratio_top = ph_0 / ps
    safe_log_top = jnp.log(jnp.maximum(ratio_top, 1e-30))
    denom_top = 1.0 + a0 * rdgas * safe_log_top
    gz_trop_top = c0 / denom_top + hs - c0
    gz_top = jnp.where(ph_0 <= p1, gz_strat_top, gz_trop_top)
    gz = gz.at[..., 0].set(gz_top)

    # pt and delp
    log_ph_ratio = jnp.log(
        jnp.maximum(ph[..., 1:], 1.0) / jnp.maximum(ph[..., :-1], 1.0)
    )
    safe_log_ph = jnp.where(jnp.abs(log_ph_ratio) > 1e-30, log_ph_ratio, 1.0)
    pt = (gz[..., :-1] - gz[..., 1:]) / (rdgas * safe_log_ph)
    pt = jnp.maximum(pt, t1)
    delp = ph[..., 1:] - ph[..., :-1]

    return ps, delp, pt


def drymadj(
    delp: jax.Array,
    q: jax.Array | None,
    area: jax.Array,
    ptop: float,
    dry_mass: float,
    adjust_dry_mass: bool = True,
    nwat: int = 0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """FV3_3D iter 645: dry-mass surface pressure + adjustment.

    Faithful JAX port of FV3 ``drymadj`` (tools/init_hydro.F90:
    195-275), serial branch (no MPI).

    Algorithm:
        ps[i,j]  = ptop + Σ_k delp[i,j,k]
        psd[i,j] = ptop + Σ_k delp[i,j,k] · (1 - Σ_n q[i,j,k,n])
                                       # dry surface pressure
        psdry    = area-weighted global mean of psd
        dpd      = dry_mass - psdry  (if adjust_dry_mass; else 0)

    Parameters
    ----------
    delp : jax.Array, shape (..., km)
        Layer pressure thicknesses (Pa).
    q : jax.Array, shape (..., km, nwat), optional
        Water-substance tracers (mass mixing ratios).  If None or
        ``nwat==0``, psd defaults to ps.
    area : jax.Array, shape (...,)
        Cell areas (matching delp leading axes).
    ptop : float
        Top-of-model pressure (Pa).
    dry_mass : float
        Target global mean dry surface pressure (Pa).
    adjust_dry_mass : bool, default True
        If True, return ``dpd = dry_mass - psdry``; else dpd = 0.
    nwat : int, default 0
        Number of water-substance tracers in ``q[..., :nwat]``.

    Returns
    -------
    ps : jax.Array, shape (...,)
        Total surface pressure.
    psd : jax.Array, shape (...,)
        Dry surface pressure.
    dpd : jax.Array (scalar)
        Mass adjustment (dry_mass - psdry), or 0 if disabled.
    """
    ps = ptop + jnp.sum(delp, axis=-1)
    if q is not None and nwat >= 1:
        # Sum of nwat water tracers per cell (last axis runs over species)
        q_sum = jnp.sum(q[..., :nwat], axis=-1)        # shape (..., km)
        psd = ptop + jnp.sum(delp * (1.0 - q_sum), axis=-1)
    else:
        psd = ps
    # Area-weighted mean of psd (iter-623 g_sum mode=1 analog)
    total_area = jnp.sum(area)
    safe_area = jnp.where(total_area > 0.0, total_area, 1.0)
    psdry = jnp.sum(psd * area) / safe_area
    if adjust_dry_mass:
        dpd = dry_mass - psdry
    else:
        dpd = jnp.asarray(0.0)
    return ps, psd, dpd


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


def compute_dz_L101(
    stretch_f: float = 1.16,
    dz0: float = 40.0,
    k0: int = 24,  # FV3 1-indexed k0=25 → 0-indexed 24
    k1: int = 1,   # FV3 1-indexed k1=2  → 0-indexed 1
) -> tuple[jax.Array, jax.Array]:
    """FV3_3D iter 637: FV3 L101 layer thicknesses + ztop.

    Faithful JAX port of FV3 ``compute_dz_L101``
    (tools/fv_eta.F90:2069-2108).  Builds the FV3-canonical
    101-layer vertical structure with ztop ≈ 20.3 km:

        - Bottom (k = k0..km-1, 0-indexed):  uniform dz = dz0 = 40 m
        - Middle (k = k1..k0):  geometric, dz[k] = stretch_f · dz[k+1]
        - Top (k = 0):  dz[0] = 4 · dz[1]

    With defaults (FV3 reference): k1=1, k0=24, dz0=40 m,
    stretch_f=1.16 → 25 geometric layers from 46.4 m to 1656 m,
    77 uniform 40-m bottom layers, single 6.6 km top layer.
    Total ztop ≈ 20.3 km.

    Returns
    -------
    dz : jax.Array, shape ``(101,)``
        Layer thicknesses (top→bottom indexing).
    ztop : jax.Array (scalar)
        Total height = sum(dz).
    """
    km = 101
    dz = jnp.full((km,), dz0)
    # Geometric middle: dz[k] = stretch_f^(k0+1-k) · dz0 for k in [k1, k0]
    # (k0 = 24, k1 = 1)  →  exponents (k0+1-k) for k=1..24:  exponents 24..1
    k_idx = jnp.arange(k1, k0 + 1)              # 0-indexed [1, 24]
    exponents = (k0 + 1) - k_idx                # 24, 23, ..., 1
    geo_dz = (stretch_f ** exponents) * dz0
    dz = dz.at[k1:k0 + 1].set(geo_dz)
    # Top: dz[0] = 4 · dz[1]
    dz = dz.at[0].set(4.0 * dz[1])
    ztop = jnp.sum(dz)
    return dz, ztop


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


def vertical_advection_theta_hybrid(
    T: jax.Array,
    mass_flux: jax.Array,
    p_s: jax.Array,
    coord: HybridSigmaPressureCoordinate,
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

    # Advect θ with the SAME upwind operator, then convert back: -exner·F·∂θ/∂p
    return exner * vertical_advection_hybrid(theta, mass_flux, p_s, coord)


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
    warn_if_unaligned_levels(n_levels, where="create_height_coordinate")
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
