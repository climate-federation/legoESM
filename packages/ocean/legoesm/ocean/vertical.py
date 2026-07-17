"""Ocean z-star vertical coordinate.

z* = H_max * (z + H) / (eta + H)

where H is the local ocean depth (bathymetry) and eta is the
time-varying sea surface height.

Unlike the atmosphere's z-star (static terrain Jacobian), the ocean
z-star has a DYNAMIC Jacobian J = (eta + H) / H that is recomputed
at every timestep as eta evolves.

Level convention: k=0 is surface, k=nlev-1 is deepest.
Reference z values are negative (below sea level).
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from legoesm.core.precision import get_policy
from legoesm.timestepping.tridiagonal import thomas_solve

# Shchepetkin (2015) adaptive-implicit vertical-advection Courant
# thresholds (NEMO ``ln_zad_Aimp`` PARAMETERs).  Below ``CU_MIN`` the
# vertical advection is fully explicit; above ``CU_CUT = 2*CU_MAX - CU_MIN``
# it is fully implicit; in between a smooth ramp blends the two.  These
# are scheme constants (not physical constants), so they live here as
# documented module defaults rather than in ``constants.py``; expose as
# kwargs on the wrapper so a single edit retunes the scheme.
_AIMP_CU_MIN = 0.15
_AIMP_CU_MAX = 0.30
# Layer-thickness floor [m] for advective-tendency / Courant denominators
# (matches ``flux_form_vertical_momentum_advection``).
_H_FLOOR = 1.0e-10


class OceanZStarCoordinate(NamedTuple):
    """Static vertical grid definition (independent of eta).

    Levels indexed surface-to-bottom: k=0 is surface, k=nlev-1 is deepest.
    Reference layer thicknesses assume eta=0 and flat bottom H_max.

    Fields
    ------
    n_levels : int
        Number of vertical levels.
    H_max : float
        Maximum ocean depth [m] (positive).
    z_full_ref : array
        Reference z* at full (cell center) levels [m], shape (nlev,).
        Negative values (below sea level). z_full_ref[0] is shallowest.
    z_half_ref : array
        Reference z* at half (interface) levels [m], shape (nlev+1,).
        z_half_ref[0] = 0 (surface), z_half_ref[-1] = -H_max (bottom).
    dz_ref : array
        Reference layer thickness [m], shape (nlev,). Positive.
    dz_half_ref : array
        Distance between adjacent full levels [m], shape (nlev-1,).
    linear_free_surface : bool
        NEMO ``key_linssh``: thicknesses frozen at the eta=0 reference
        (J eta-independent), diagnosed w without the z-star sigma
        correction. Default False (full z*).
    t_depth_ref : array or None
        Optional EXACT positive T-point reference depths [m], shape
        (nlev,).  ``None`` (default) means ``|z_full_ref|`` (the
        cell-centre midpoint) is the T-point depth ladder — correct for
        legoESM's own z* grid.  A fidelity bridge that must reproduce an
        external model whose T-points are NOT the interface midpoints
        (e.g. NEMO's analytic MI96 ``gdept_1d`` ≠ midpoint of
        ``gdepw_1d``) supplies that model's exact T-depths here so the
        ``nemo_trapezoid`` PGF quadrature reconstructs the identical
        ``e3w`` (W-spacing) recurrence.  Read ONLY by the hydrostatic
        pressure quadrature; ``dz_half_ref`` and every other operator
        keep using the midpoint ``z_full_ref``.
    """
    n_levels: int
    H_max: float
    z_full_ref: jnp.ndarray
    z_half_ref: jnp.ndarray
    dz_ref: jnp.ndarray
    dz_half_ref: jnp.ndarray
    t_depth_ref: jnp.ndarray | None = None
    # NEMO key_linssh (LINEAR free surface): freeze the geometry at eta=0 —
    # layer thicknesses NEVER stretch (J = H_bathy/H_max, eta-independent) and
    # the diagnosed w skips the z-star sigma redistribution of deta/dt (NEMO
    # sshwzv.F90:190-193 fixed-e3t continuity; w[0]=deta/dt, w[bottom]=0).
    # eta still evolves via the barotropic solver and drives g*grad(eta).
    # STATIC Python bool — gates are `if` branches (never jnp.where); the
    # coordinate is constructor-captured, not traced.
    linear_free_surface: bool = False


def create_ocean_z_star(
    n_levels: int = 50,
    H_max: float = 5500.0,
    dz_surface: float = 10.0,
    dz_deep: float = 200.0,
) -> OceanZStarCoordinate:
    """Create a stretched ocean z-star coordinate.

    Uses hyperbolic tangent stretching: fine resolution near surface
    (~dz_surface m), coarse at depth (~dz_deep m).

    Parameters
    ----------
    n_levels : int
        Number of vertical levels.
    H_max : float
        Maximum ocean depth [m].
    dz_surface : float
        Target layer thickness near surface [m].
    dz_deep : float
        Target layer thickness at depth [m].

    Returns
    -------
    OceanZStarCoordinate : The vertical coordinate.
    """
    if n_levels < 1:
        raise ValueError(
            f"n_levels must be >= 1, got {n_levels!r}",
        )
    if H_max <= 0.0:
        raise ValueError(f"H_max must be > 0, got {H_max!r}")
    if dz_surface <= 0.0:
        raise ValueError(f"dz_surface must be > 0, got {dz_surface!r}")
    if dz_deep <= 0.0:
        raise ValueError(f"dz_deep must be > 0, got {dz_deep!r}")

    # Stretched grid: dz grows smoothly from dz_surface to dz_deep.
    # Use a normalized distribution then scale to match H_max.
    k = jnp.arange(n_levels, dtype=get_policy().control)

    # Layer thickness profile: linear growth from dz_surface to dz_deep
    dz_raw = dz_surface + k * (dz_deep - dz_surface) / jnp.maximum(n_levels - 1.0, 1.0)

    # Normalize so total thickness matches H_max
    scale = H_max / jnp.sum(dz_raw)
    dz_ref = dz_raw * scale

    # Snap the last layer so ``sum(dz_ref) == H_max`` to bit-precision.
    # Without this, the cumsum round-trip below leaves a float-drift
    # residue of order ``H_max * eps_dtype`` (~2e-4 m for H_max=4000m
    # in fp32) that breaks the Hallberg-Adcroft 2009 column-sum
    # identity in partial-cell models — ``sum_k(h_partial)`` would
    # differ from the user-supplied ``H_bathy`` by that residue.
    dz_ref = dz_ref.at[-1].set(H_max - jnp.sum(dz_ref[:-1]))

    # Interface depths from cumulative sum (surface=0, bottom=-H_max)
    z_half_ref = jnp.concatenate([
        jnp.array([0.0], dtype=dz_ref.dtype),
        -jnp.cumsum(dz_ref),
    ])
    # Snap the bottom interface to exactly -H_max (kills cumsum drift).
    z_half_ref = z_half_ref.at[-1].set(-H_max)

    # Full level depths (cell centers)
    z_full_ref = 0.5 * (z_half_ref[:-1] + z_half_ref[1:])

    # Layer thicknesses (positive). Recover from the snapped interfaces.
    dz_ref = z_half_ref[:-1] - z_half_ref[1:]

    # Distance between full levels
    dz_half_ref = z_full_ref[:-1] - z_full_ref[1:]  # positive

    return OceanZStarCoordinate(
        n_levels=n_levels,
        H_max=H_max,
        z_full_ref=z_full_ref,
        z_half_ref=z_half_ref,
        dz_ref=dz_ref,
        dz_half_ref=dz_half_ref,
    )


def create_z_star_from_thicknesses(
    dz_ref_m, t_depth_ref_m=None,
) -> OceanZStarCoordinate:
    """Build a z* coordinate from EXPLICIT reference layer thicknesses.

    Reproduces an external model's vertical grid EXACTLY -- pass another model's
    1-D reference thicknesses (e.g. NEMO ``e3t_1d`` [m], surface ~1 m growing to
    ~200 m for ORCA L75) and get back the identical level interfaces / centres,
    with no stretching-parameter guessing.  Used by the OMIP runner's
    ``--nemo-vertical`` to match NEMO ORCA1's 75-level grid so vertical gradients
    (thermocline, mixed layer) are resolved comparably.

    ``n_levels`` and ``H_max`` are inferred from the input (``len(dz)`` and
    ``sum(dz)``).  Construction mirrors :func:`create_ocean_z_star` after its
    thickness profile is fixed -- the same snap-to-``H_max`` and interface
    recovery so the Hallberg-Adcroft column-sum identity holds to bit precision.

    Parameters
    ----------
    dz_ref_m : 1-D array-like
        Reference layer thicknesses [m], top -> bottom, all > 0.
    t_depth_ref_m : 1-D array-like or None
        Optional EXACT positive T-point depths [m], shape (nlev,), stored
        on the coordinate's ``t_depth_ref`` field for the fidelity PGF
        quadrature (see :class:`OceanZStarCoordinate`).  ``None`` (default)
        leaves ``t_depth_ref=None`` → the midpoint ``z_full_ref`` is used.
        Pass an external model's true T-depths (e.g. NEMO ``gdept_1d``)
        when they differ from the interface midpoint.

    Returns
    -------
    OceanZStarCoordinate
    """
    # Check ndim on the ORIGINAL array BEFORE any ravel -- a 2-D array would
    # otherwise be silently flattened and accepted as 1-D (codex HIGH).
    dz_np = np.asarray(dz_ref_m, dtype=np.float64)
    if dz_np.ndim != 1 or dz_np.size < 2:
        raise ValueError(
            f"dz_ref_m must be a 1-D array of >= 2 thicknesses, got shape "
            f"{dz_np.shape}")
    if not np.all(dz_np > 0.0):
        raise ValueError("dz_ref_m thicknesses must all be > 0")
    n_levels = int(dz_np.size)
    H_max = float(dz_np.sum())

    dz_ref = jnp.asarray(dz_np, dtype=get_policy().control)
    # Interfaces from cumulative sum (surface=0, bottom=-H_max); snap the bottom
    # to exactly -H_max to kill cumsum drift, then recover dz from the snapped
    # interfaces (identical pattern to create_ocean_z_star).
    z_half_ref = jnp.concatenate([
        jnp.array([0.0], dtype=dz_ref.dtype),
        -jnp.cumsum(dz_ref),
    ])
    z_half_ref = z_half_ref.at[-1].set(-H_max)
    z_full_ref = 0.5 * (z_half_ref[:-1] + z_half_ref[1:])
    dz_ref = z_half_ref[:-1] - z_half_ref[1:]
    dz_half_ref = z_full_ref[:-1] - z_full_ref[1:]

    t_depth_ref = None
    if t_depth_ref_m is not None:
        t_np = np.asarray(t_depth_ref_m, dtype=np.float64)
        if t_np.ndim != 1 or t_np.size != n_levels:
            raise ValueError(
                f"t_depth_ref_m must be a 1-D array of length n_levels="
                f"{n_levels}, got shape {t_np.shape}")
        if not np.all(t_np > 0.0):
            raise ValueError("t_depth_ref_m depths must all be > 0")
        # Monotone-increasing: the PGF e3w recurrence uses gdept(k)-gdept(k-1) as
        # a positive W-spacing; a non-monotone ladder would give a negative e3w
        # and a silently nonphysical pressure gradient.
        if not np.all(np.diff(t_np) > 0.0):
            raise ValueError("t_depth_ref_m depths must be strictly increasing")
        t_depth_ref = jnp.asarray(t_np, dtype=get_policy().control)

    return OceanZStarCoordinate(
        n_levels=n_levels,
        H_max=H_max,
        z_full_ref=z_full_ref,
        z_half_ref=z_half_ref,
        dz_ref=dz_ref,
        dz_half_ref=dz_half_ref,
        t_depth_ref=t_depth_ref,
    )


def _levy_stretching_coefficients(
    K_formula: int,
    H: float,
    dz_min: float,
    k_th: float,
    a_cr: float,
) -> tuple[float, float, float]:
    """Compute (a₀, a₁, a₂) for the Lévy (2010) tanh+ln(cosh) vertical
    stretching used by NEMO's mi96_1d routine (Madec-Imbard 1996).

    The stretching function is::

        z(k) = a₂ + a₁·k + a₀·a_cr·ln(cosh((k - k_th)/a_cr))

    Coefficients are determined by three constraints:

      z(k=1)         = 0     (surface interface)
      z(k=K_formula) = H     (bottom interface)
      dz/dk at k=1   = dz_min  (derivative-based top-layer scale)

    Parameters
    ----------
    K_formula : int
        Index of the bottom interface in the formula's k-coordinate.
        In NEMO terminology this is ``jpk`` (interface count). For
        ``n_levels`` cells the value is ``n_levels + 1``.
    H : float
        Total ocean depth [m] (positive).
    dz_min : float
        Target top-layer derivative ``dz/dk`` at k=1 [m].
    k_th : float
        Inflection-level index. Layers ``k > k_th`` are thicker than
        ``k < k_th``. Typically ``k_th = n_levels - 1``.
    a_cr : float
        Stretching width parameter. Smaller = sharper transition near
        ``k_th``. Typically 5-15.

    Returns
    -------
    (a0, a1, a2) : tuple of float
    """
    Km1 = K_formula - 1
    th = math.tanh((1 - k_th) / a_cr)
    log_cosh_K = math.log(math.cosh((K_formula - k_th) / a_cr))
    log_cosh_1 = math.log(math.cosh((1 - k_th) / a_cr))
    denom = th - (a_cr / Km1) * (log_cosh_K - log_cosh_1)
    a0 = (dz_min - H / Km1) / denom
    a1 = dz_min - a0 * th
    a2 = -a1 - a0 * a_cr * log_cosh_1
    return a0, a1, a2


def _levy_depth_at_k(k, a0, a1, a2, k_th, a_cr) -> float:
    """Evaluate the Lévy stretching formula at arbitrary k (positive)."""
    return a2 + a1 * k + a0 * a_cr * math.log(math.cosh((k - k_th) / a_cr))


def create_levy_stretched_z_star(
    n_levels: int,
    H_max: float,
    dz_min: float,
    k_th: float,
    a_cr: float,
    analytic_t_depths: bool = False,
) -> OceanZStarCoordinate:
    """Construct a Lévy (2010) / Madec-Imbard (1996) stretched z* grid.

    Used by NEMO's ``mi96_1d`` routine and many idealized NEMO configs
    (Neverworld 2, DINO, Munday-Marshall-Johnson). Top-layer derivative
    is ``dz_min``; layer thickness grows smoothly to ~H/K_formula·tanh
    in the deep abyss.

    Returns
    -------
    OceanZStarCoordinate
        With ``n_levels`` cells, surface interface at 0, bottom
        interface snapped to exactly ``-H_max``.

    Notes
    -----
    NEMO note on indexing: the formula's "K" in the literature is
    the interface count (= n_levels + 1), NOT the cell count.
    See Madec & Imbard 1996 / Lévy et al. 2010.

    ``analytic_t_depths=False`` (default, bit-identical legacy): cell
    centres are interface midpoints.  ``True``: cell centres are the
    ANALYTIC stretching formula at k+0.5 — exactly NEMO's ``mi96_1d``
    ``pdept_1d`` (zgr_lib.F90: ``zt = jk + 0.5``), which on a stretched
    grid is NOT the interface midpoint (up to ~4.6 m difference on the
    DINO 36-interface ladder).  NEMO jpk convention: NEMO's ``jpk``
    counts INTERFACE indices (its level jpk is a permanently-masked
    dummy), so a NEMO config with jpk=36 maps to ``n_levels=35`` here.
    """
    if n_levels < 2:
        raise ValueError(f"n_levels must be >= 2, got {n_levels!r}")
    if H_max <= 0.0:
        raise ValueError(f"H_max must be > 0, got {H_max!r}")
    if dz_min <= 0.0:
        raise ValueError(f"dz_min must be > 0, got {dz_min!r}")

    K_formula = n_levels + 1  # interface count (NEMO jpk convention)

    a0, a1, a2 = _levy_stretching_coefficients(
        K_formula=K_formula,
        H=H_max,
        dz_min=dz_min,
        k_th=float(k_th),
        a_cr=a_cr,
    )

    # Interfaces at integer k = 1, 2, ..., K_formula → n_levels+1 interfaces
    z_half_pos = [
        _levy_depth_at_k(float(k), a0, a1, a2, float(k_th), a_cr)
        for k in range(1, K_formula + 1)
    ]

    # legoESM convention: z negative below surface
    z_half_list = [-z for z in z_half_pos]
    z_half_list[0] = 0.0     # snap surface
    z_half_list[-1] = -H_max # snap bottom (kills sub-meter formula residue)
    z_half_ref = jnp.asarray(z_half_list)

    dz_ref = z_half_ref[:-1] - z_half_ref[1:]
    if analytic_t_depths:
        # NEMO mi96_1d pdept_1d: the SAME stretching formula at k+0.5
        # (one centre per cell, k = 1..n_levels; NEMO's dummy jpk-th
        # centre below the last interface is not represented).
        t_pos = [
            _levy_depth_at_k(k + 0.5, a0, a1, a2, float(k_th), a_cr)
            for k in range(1, n_levels + 1)
        ]
        z_full_ref = jnp.asarray([-t for t in t_pos])
    else:
        z_full_ref = 0.5 * (z_half_ref[:-1] + z_half_ref[1:])
    dz_half_ref = z_full_ref[:-1] - z_full_ref[1:]

    return OceanZStarCoordinate(
        n_levels=n_levels,
        H_max=H_max,
        z_full_ref=z_full_ref,
        z_half_ref=z_half_ref,
        dz_ref=dz_ref,
        dz_half_ref=dz_half_ref,
    )


class OceanPartialCellCoordinate(NamedTuple):
    """z* + partial bottom cells (Adcroft, Hill, Marshall 1997;
    Adcroft & Campin 2004).

    Same z*-style reference levels as ``OceanZStarCoordinate``, but
    augmented with per-cell layer thicknesses ``h_partial(..., k)``
    that account for the seafloor cutting through the deepest active
    level.

    For each column with bathymetry depth ``H_bathy(i,j)``:
      - Levels k < bottom_level(i,j): full cells, h_partial[k] = dz_ref[k]
      - Level k = bottom_level(i,j): partial cell,
                                       h_partial[k] = H_bathy - |z_half_ref[k]|
      - Levels k > bottom_level(i,j): below seafloor, h_partial[k] = 0

    The reference fields (``z_full_ref``, ``z_half_ref``, ``dz_ref``,
    ``dz_half_ref``) match ``OceanZStarCoordinate`` exactly so callers
    that need only the reference grid can treat both coords
    uniformly.

    Fields specific to partial cells
    --------------------------------
    h_partial : array, shape (..., nlev)
        Per-cell at-rest layer thickness [m], with H_bathy folded in.
        Sum over k of h_partial[..., k] equals H_bathy(i, j) per column.
    bottom_level : array of int32, shape (...)
        Index of the deepest active level for each column.  ``-1`` for
        dry columns (H_bathy <= 0).
    is_active : array of bool, shape (..., nlev)
        True for cells at or above bottom_level.  Cells below the
        seafloor are False.
    """
    n_levels: int
    H_max: float
    z_full_ref: jnp.ndarray
    z_half_ref: jnp.ndarray
    dz_ref: jnp.ndarray
    dz_half_ref: jnp.ndarray
    h_partial: jnp.ndarray
    bottom_level: jnp.ndarray
    is_active: jnp.ndarray
    # Exact reference T-level depths (NEMO ``gdept_1d``; z*-only fidelity
    # field) propagated from the wrapped z* coordinate so non-midpoint
    # reference ladders keep their true centre geometry under partial
    # cells (MLE nla10 + gate-N2 consumers; codex MLE-rhop r2).  ``None``
    # on the model's own midpoint grids.
    t_depth_ref: jnp.ndarray | None = None


def create_partial_cell_coordinate(
    z_coord: OceanZStarCoordinate,
    H_bathy: jnp.ndarray,
) -> OceanPartialCellCoordinate:
    """Build an ``OceanPartialCellCoordinate`` from a z* coord + bathymetry.

    Parameters
    ----------
    z_coord : OceanZStarCoordinate
        Reference vertical grid (sets H_max, dz_ref, etc.).  Used to
        derive partial-cell thicknesses.
    H_bathy : array
        Per-column bathymetry depth [m], positive downward.  Land
        cells should have H_bathy <= 0; they're flagged as
        ``bottom_level = -1`` and ``is_active = False`` everywhere.

    Returns
    -------
    OceanPartialCellCoordinate

    Notes
    -----
    The factory is differentiable w.r.t. continuous ``H_bathy`` only
    while ``bottom_level`` does not change — i.e., piecewise smooth
    with discontinuities at every reference-level interface.  This is
    the documented limitation of partial-cell schemes (see
    ``docs/ocean/experiments/partial_cells_plan.md`` Differentiability
    Contract); not specific to this implementation.
    """
    H = jnp.asarray(H_bathy)
    nlev = z_coord.n_levels
    abs_z_half = jnp.abs(z_coord.z_half_ref)        # (nlev+1,) positive depths

    # Number of half-interfaces strictly shallower than H_bathy.
    # E.g. abs_z_half = [0, 10, 300, 1500, 4000], H=2350 → count=4 → bottom_level=3.
    n_lead = H.ndim
    H_exp = H[..., jnp.newaxis]                     # (..., 1)
    interfaces_above = jnp.sum(
        abs_z_half[(jnp.newaxis,) * n_lead + (slice(None),)] < H_exp,
        axis=-1,
    )                                                # (...) integer
    bottom_level = interfaces_above.astype(jnp.int32) - 1
    # Dry columns (H <= 0): mark bottom_level = -1 (no active cells).
    bottom_level = jnp.where(H > 0.0, bottom_level, -1)
    # Cap at the deepest possible level (when H exceeds H_max).
    bottom_level = jnp.minimum(bottom_level, nlev - 1)

    # Per-cell active mask.
    k_idx = jnp.arange(nlev, dtype=jnp.int32)
    k_view = k_idx.reshape((1,) * n_lead + (nlev,))
    bottom_view = bottom_level[..., jnp.newaxis]    # (..., 1)
    is_active = (k_view <= bottom_view) & (bottom_view >= 0)

    # Per-cell layer thickness.
    # Start with dz_ref broadcast to (..., nlev).
    dz_ref_view = z_coord.dz_ref.reshape((1,) * n_lead + (nlev,))
    h_full = jnp.broadcast_to(dz_ref_view, H.shape + (nlev,))

    # Partial thickness at bottom level: H - |z_half_ref[bottom_level]|,
    # capped above by the full-cell reference thickness ``dz_ref[bottom]``,
    # AND snapped to ``dz_ref[bottom]`` when within float32 precision.
    #
    # The cap matters when ``H_bathy >= H_max`` (caller passes a column
    # at or beyond reference depth — full cells, no extra thickness).
    #
    # The snap matters because ``abs_z_half`` (cumsum-built in float32)
    # has ~1e-7 relative error → for a column at exactly the reference
    # depth, ``H - abs_z_half[bottom]`` differs from ``dz_ref[bottom]``
    # by sub-millimetre.  Without the snap, ``use_partial_cells=False``
    # backwards-compat regression in Phase 6 breaks — the partial path
    # gives a slightly different thickness than the legacy path.
    # We snap when the values agree to ~1e-5 relative, well below any
    # physically meaningful column-thickness variation.
    #
    # For dry columns, bottom_level = -1 and we use 0 (the value gets
    # masked out by is_active anyway).
    safe_bottom = jnp.maximum(bottom_level, 0)
    abs_z_at_bottom = abs_z_half[safe_bottom]       # (...)
    dz_at_bottom = z_coord.dz_ref[safe_bottom]      # (...)
    raw_partial = H - abs_z_at_bottom
    capped = jnp.minimum(raw_partial, dz_at_bottom)
    near_full = jnp.abs(capped - dz_at_bottom) < dz_at_bottom * 1e-5
    partial_thickness = jnp.where(near_full, dz_at_bottom, capped)

    is_bottom = (k_view == bottom_view) & (bottom_view >= 0)
    h_partial = jnp.where(is_bottom, partial_thickness[..., jnp.newaxis], h_full)
    h_partial = jnp.where(is_active, h_partial, 0.0)

    return OceanPartialCellCoordinate(
        n_levels=nlev,
        H_max=z_coord.H_max,
        z_full_ref=z_coord.z_full_ref,
        z_half_ref=z_coord.z_half_ref,
        dz_ref=z_coord.dz_ref,
        dz_half_ref=z_coord.dz_half_ref,
        h_partial=h_partial,
        bottom_level=bottom_level,
        is_active=is_active,
        # Propagate the z*-only exact NEMO gdept so partial-cell wraps of a
        # NEMO reference ladder keep true centre depths (nla10 tolerance,
        # gate-N2 pressure geometry, and any future partial-cell PGF
        # fidelity).  ``getattr``: plain midpoint z* coords carry None.
        t_depth_ref=getattr(z_coord, "t_depth_ref", None),
    )


def extrapolate_below_seafloor(
    field: jnp.ndarray,
    z_coord: OceanPartialCellCoordinate,
) -> jnp.ndarray:
    """Fill below-seafloor (inactive) cells of a per-column field with the
    deepest ACTIVE value of that column (constant downward extrapolation).

    Active cells are a surface-down prefix (``k <= bottom_level``), so the
    inactive cells are the suffix below the seafloor.  Filling them with the
    deepest-active value gives the horizontal stencils (FC spectral or C-D
    Arakawa-Lamb) a smooth, physically-defined value at the seafloor step, so
    an active cell adjacent to a shallower column cannot import a stale/poison
    rock-cell value (a tracer or, via the EOS, density).  This is the
    per-level analogue of the cd-grid ``fill_land_cells`` horizontal rock
    fill, and is grid-neutral (touches only the trailing level axis).

    Dry columns (``bottom_level == -1``) become a constant column; they are
    removed by the 2D land mask downstream.

    Parameters
    ----------
    field : array, shape (..., nlev)
        Per-column field (e.g. T or S) to extrapolate below the seafloor.
    z_coord : OceanPartialCellCoordinate
        Provides ``bottom_level`` (deepest active level) and ``is_active``.

    Returns
    -------
    array, same shape as ``field``, with below-seafloor cells filled by the
    deepest active value of their column.
    """
    bl = jnp.maximum(z_coord.bottom_level, 0)[..., jnp.newaxis]   # (..., 1)
    deepest = jnp.take_along_axis(field, bl, axis=-1)             # (..., 1)
    return jnp.where(z_coord.is_active, field, deepest)


def compute_centroid_depth(
    eta: jnp.ndarray,
    H_bathy: jnp.ndarray,
    z_coord,
    min_water_column_m: float | None = None,
) -> jnp.ndarray:
    """Compute the geometric centroid depth (positive downward) of each
    cell, accounting for eta and partial cells.

    For each (column, level k):
      centroid_depth[..., k] = sum_{j<k} h_actual[..., j] + 0.5 * h_actual[..., k]

    For pure z\\* coord (full cells everywhere): centroid is at
    ``|z_full_ref[k]| * (eta + H_bathy) / H_max`` — uniform across columns.
    For partial-cell coord: centroid varies per column at the partial
    bottom.  Cells below the seafloor have h_actual=0 and inherit the
    seafloor depth from above (no further increment).

    Used by the Adcroft-Campin face PGF correction (Phase 3b): the
    horizontal pressure gradient between two cells with different
    centroid depths is corrected by shifting each cell's pressure to a
    common face-reference depth.

    Parameters
    ----------
    eta : array
        Sea surface height [m], shape (...).
    H_bathy : array
        Local bathymetry depth [m], shape (...).  Positive.
    z_coord : OceanZStarCoordinate or OceanPartialCellCoordinate
        Vertical coordinate.
    min_water_column_m : float or None
        Optional water-column floor (passed to ``compute_layer_thickness``).

    Returns
    -------
    array : Centroid depth [m], shape (..., nlev).  Positive downward.
    """
    h = compute_layer_thickness(
        eta, H_bathy, z_coord, min_water_column_m=min_water_column_m,
    )
    # cumsum gives interface depths at the *bottom* of each layer.
    # Centroid is half a layer above the bottom interface.
    cum = jnp.cumsum(h, axis=-1)
    return cum - 0.5 * h


def compute_layer_thickness(
    eta: jnp.ndarray,
    H_bathy: jnp.ndarray,
    z_coord,
    min_water_column_m: float | None = None,
) -> jnp.ndarray:
    """Compute actual layer thickness incorporating eta and bathymetry.

    Dispatches on the coordinate type:

    - ``OceanZStarCoordinate``: pure z\\*.
      ``h_k = dz_ref[k] * (eta + H_bathy) / H_max``.
      All layers compressed uniformly by the column Jacobian.
    - ``OceanPartialCellCoordinate``: z\\* + partial bottom cell.
      ``h_k = h_partial[..., k] * (eta + H_bathy) / H_bathy``.
      Same uniform Jacobian, but applied to the per-cell partial-cell
      thicknesses.  Cells below the seafloor stay zero (h_partial = 0).

    For the flat-bottom case (``H_bathy = H_max`` everywhere),
    both formulas yield identical layer thicknesses — the partial-cell
    coord has ``h_partial = dz_ref`` for every column, and the
    Jacobian becomes ``(eta + H_max) / H_max`` either way.  This
    backwards-compat property is guaranteed by the snap-to-dz_ref logic
    in ``create_partial_cell_coordinate``.

    Parameters
    ----------
    eta : array
        Sea surface height [m], shape (...).
    H_bathy : array
        Local bathymetry depth [m], shape (...). Positive.
    z_coord : OceanZStarCoordinate or OceanPartialCellCoordinate
        Vertical coordinate.
    min_water_column_m : float or None
        Optional lower bound for local water-column thickness
        ``eta + H_bathy`` [m]. When set, Jacobian/thickness values are
        clipped to avoid dry or negative columns.

    Returns
    -------
    array : Layer thickness [m], shape (..., nlev). Positive.
    """
    if isinstance(z_coord, OceanPartialCellCoordinate):
        wc = eta + H_bathy
        if min_water_column_m is not None:
            wc = jnp.maximum(wc, min_water_column_m)
        # Avoid division-by-zero in dry columns; h_partial is already
        # zero there, so the result is zero regardless of the divisor.
        H_safe = jnp.maximum(H_bathy, 1.0e-10)
        return z_coord.h_partial * (wc / H_safe)[..., jnp.newaxis]
    # Pure z\\* path (legacy, unchanged).
    J = compute_ocean_jacobian(
        eta, H_bathy, z_coord, min_water_column_m=min_water_column_m,
    )
    return z_coord.dz_ref * J[..., jnp.newaxis]


def compute_ocean_jacobian(
    eta: jnp.ndarray,
    H_bathy: jnp.ndarray,
    z_coord,
    min_water_column_m: float | None = None,
) -> jnp.ndarray:
    """Compute the dynamic vertical-coordinate Jacobian.

    Dispatches on coord type:

    - ``OceanZStarCoordinate``: ``J = (eta + H_bathy) / H_max``.  Used
      with ``dz_ref`` to get per-cell thickness.
    - ``OceanPartialCellCoordinate``: ``J = (eta + H_bathy) / H_bathy``.
      Used with ``h_partial`` to get per-cell thickness (the partial
      cell, full cells, and below-seafloor zero cells all scale with
      the same Jacobian).

    For backwards-compat on flat-bottom (H_bathy = H_max everywhere),
    both formulas give the same Jacobian.

    Parameters
    ----------
    eta : array
        Sea surface height [m], shape (...).
    H_bathy : array
        Local bathymetry depth [m], shape (...). Positive.
    z_coord : OceanZStarCoordinate or OceanPartialCellCoordinate
        Vertical coordinate.
    min_water_column_m : float or None
        Optional lower bound for local water-column thickness
        ``eta + H_bathy`` [m].

    Returns
    -------
    array : Jacobian, shape (...).
    """
    if getattr(z_coord, "linear_free_surface", False):
        # NEMO key_linssh: the column NEVER stretches — J is the eta=0
        # reference (H_bathy/H_max; ==1 on a flat bottom where H_bathy==H_max).
        # No min-column clip: the fixed column is positive by construction.
        water_col = jnp.broadcast_to(
            jnp.asarray(H_bathy, dtype=jnp.asarray(eta).dtype), jnp.shape(eta))
    else:
        water_col = eta + H_bathy
        if min_water_column_m is not None:
            min_col = jnp.asarray(min_water_column_m, dtype=water_col.dtype)
            water_col = jnp.maximum(water_col, min_col)
    if isinstance(z_coord, OceanPartialCellCoordinate):
        H_safe = jnp.maximum(H_bathy, 1.0e-10)
        return water_col / H_safe
    return water_col / z_coord.H_max


def upwind_vertical_gradient(
    field: jnp.ndarray,
    dz_half: jnp.ndarray,
    w: jnp.ndarray,
    *,
    eps: float = 1.0e-12,
) -> jnp.ndarray:
    """Compute first-order upwind d(field)/dz at full levels.

    Assumes levels are indexed surface-to-bottom (k=0 at surface).
    Caller must use consistent coordinates: both ``dz_half`` and ``w``
    should be in the same vertical coordinate (physical z or z*).

    Parameters
    ----------
    field : array
        Field at full levels, shape (..., nlev).
    dz_half : array
        Full-level spacing [m], shape (..., nlev-1). Positive.
    w : array
        Vertical velocity [m/s], shape (..., nlev).
        Only its sign is used (upwind direction). Positive = upward.
    eps : float
        Small denominator guard for spacing.

    Returns
    -------
    array : Upwind vertical gradient d(field)/dz, shape (..., nlev).
    """
    inv_dz_half = 1.0 / jnp.maximum(dz_half, eps)
    df = (field[..., :-1] - field[..., 1:]) * inv_dz_half

    # Pad along trailing axis instead of allocating a fresh ``zeros``
    # buffer + concatenate.  Single Pad HLO op each.  This helper
    # fires once per scan step inside ``vertical_advection_ocean`` for
    # u, v, T, S, and every tracer — so 4-6 zero-broadcast concats
    # per RHS evaluation in the hot loop.
    pad_axes = ((0, 0),) * (df.ndim - 1)
    # Upward flow (w>0): donor is deeper cell -> (f[k] - f[k+1]) / dz.
    grad_up = jnp.pad(df, (*pad_axes, (0, 1)))
    # Downward flow (w<0): donor is shallower cell -> (f[k-1] - f[k]) / dz.
    grad_down = jnp.pad(df, (*pad_axes, (1, 0)))

    return jnp.where(w > 0.0, grad_up, grad_down)


# ---------------------------------------------------------------------------
# Vertical velocity diagnosis and advection (shared across ocean dycores)
# ---------------------------------------------------------------------------

def diagnose_w_from_flux_div(flux_div_k, z_coord=None,
                              thickness_weighted=False):
    """Diagnose z-star transport velocity from flux divergence.

    Performs a bottom-up cumulative sum of the horizontal flux divergence
    and optionally applies the z-star sigma correction so that
    ẇ = 0 at both surface and bottom.

    Parameters
    ----------
    flux_div_k : array, shape (..., nlev)
        Horizontal flux divergence at each layer.
        If ``thickness_weighted=False`` (legacy), this is ``div(u)`` and
        will be multiplied by ``dz_ref`` before integration.
        If ``thickness_weighted=True``, this is ``div(h*u)`` [m/s] and
        already has layer thickness folded in; no dz multiplication.
    z_coord : OceanZStarCoordinate or None
        When provided, applies the z-star correction.
    thickness_weighted : bool
        If True, ``flux_div_k`` already includes layer thickness
        (i.e. it was computed from thickness-weighted velocity).
        Default False for backward compatibility.

    Returns
    -------
    w : array, shape (..., nlev+1)
        Vertical velocity on half levels (surface first, bottom last = 0).
    """
    # From continuity: w(k) = w(k+1) + div_h(h_k * u_k)
    # If flux_div_k already includes layer thickness (thickness_weighted=True),
    # we cumsum directly. Otherwise, multiply by dz_ref first.
    if thickness_weighted:
        fd_integrated = flux_div_k
    elif z_coord is not None:
        dz_ref = z_coord.dz_ref  # Layer thicknesses
        fd_integrated = flux_div_k * dz_ref[jnp.newaxis, jnp.newaxis, :]
    else:
        # Fallback for testing (assume unit thickness)
        fd_integrated = flux_div_k

    fd_rev = fd_integrated[..., ::-1]
    cumsum_rev = jnp.cumsum(fd_rev, axis=-1)
    w_inner = -cumsum_rev[..., ::-1]
    # Pad along the trailing axis instead of allocating a fresh
    # ``(..., 1)`` zero buffer and concatenating.
    pad_axes_w = ((0, 0),) * (w_inner.ndim - 1)
    w_euler = jnp.pad(w_inner, (*pad_axes_w, (0, 1)))

    if z_coord is None:
        return w_euler
    if getattr(z_coord, "linear_free_surface", False):
        # NEMO key_linssh w (sshwzv.F90:190-193): fixed-e3t continuity —
        # w[..., 0] = deta/dt at the fixed z=0 surface, w[..., -1] = 0, NO
        # sigma redistribution of deta/dt through the column (that z-star
        # term is what pumps the surface tendency into the abyss).
        return w_euler

    deta_dt = w_euler[..., 0:1]
    if isinstance(z_coord, OceanPartialCellCoordinate):
        # Partial cells: sigma must use each column's actual seafloor
        # depth, not the reference H_max.  Otherwise w at the partial
        # seafloor (k = bottom_level + 1, not k = nlev) is non-zero by
        # ``sigma_zstar - sigma_partial`` * deta_dt — a spurious vertical
        # mass flux at the seafloor that breaks tracer mass conservation.
        # z_half_actual[k] = -cumsum(h_partial[0..k-1]) from surface;
        # H_bathy_per_column = sum(h_partial).  sigma_per_cell[k] =
        # (z_half_actual + H_bathy)/H_bathy is 1 at surface, 0 at the
        # column's own seafloor (where h_partial = 0 below).
        h_p = z_coord.h_partial                                  # (..., nlev)
        z_half_actual_inner = -jnp.cumsum(h_p, axis=-1)          # (..., nlev)
        pad_axes = ((0, 0),) * (z_half_actual_inner.ndim - 1)
        z_half_actual = jnp.pad(
            z_half_actual_inner, (*pad_axes, (1, 0)),
        )                                                          # (..., nlev+1)
        H_col = jnp.sum(h_p, axis=-1, keepdims=True)             # (..., 1)
        H_col_safe = jnp.maximum(H_col, 1e-10)
        sigma = (z_half_actual + H_col) / H_col_safe
    else:
        sigma = (z_coord.z_half_ref + z_coord.H_max) / z_coord.H_max
    return w_euler - sigma * deta_dt


def vertical_advection_ocean(field, w_half, z_coord, jacobian):
    """Vertical advection ``-w * d(field)/dz`` with upwind scheme.

    Parameters
    ----------
    field : array, shape (..., nlev)
        Quantity being advected.
    w_half : array, shape (..., nlev+1)
        Vertical velocity on half levels.
    z_coord : OceanZStarCoordinate
        Vertical coordinate (provides ``dz_half_ref``).
    jacobian : array, shape (...)
        Dynamic z-star Jacobian.

    Returns
    -------
    tendency : array, shape (..., nlev)
    """
    w_full = 0.5 * (w_half[..., :-1] + w_half[..., 1:])
    jac_safe = jnp.maximum(jacobian[..., jnp.newaxis], 1.0e-10)
    dz_half = z_coord.dz_half_ref * jac_safe
    grad = upwind_vertical_gradient(field, dz_half, w_full)

    # Vertical advection calculation

    return -w_full * grad


def flux_form_vertical_momentum_advection(
    u: jnp.ndarray,
    w_half: jnp.ndarray,
    h_u: jnp.ndarray,
    face_active: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Flux-form vertical momentum advection as a per-thickness tendency.

    Computes the interface-upwind vertical momentum flux
    ``F[k] = w_half[k] * u_face[k]`` with

    - ``F[0] = F[nlev] = 0``  (rigid-lid / no-flux boundary)
    - ``u_face[k] = u[k]``     when ``w_half[k] > 0``  (upward, from below)
    - ``u_face[k] = u[k-1]``   when ``w_half[k] <= 0`` (downward, from above)

    and returns ``-(F_top - F_bot) / h_u`` at each level.  This is the
    tracer-path pattern (``flux_form_vertical_tracer_advection``)
    converted back to a per-thickness advective tendency so that
    callers can add it directly to ``du/dt``.

    Properties
    ----------
    1. Interior interface-upwind (consistent with the tracer path).
    2. Rigid-lid boundary by construction: ``F[0] = F[nlev] = 0``.
       No artificial momentum injection from the boundary via the
       "hard zero at k=0 and k=nlev-1" pathology that the old
       ``vertical_advection_ocean`` cell-upwind gradient has.
    3. Column momentum flux identity: for any ``w_half`` with
       ``w_half[0] = w_half[nlev] = 0`` (closed column), the sum of
       ``(tendency * h_u)`` over the column is exactly zero.

    Partial-fix status (issue #171)
    -------------------------------
    This is a **Level-1** fix.  Dividing by ``h_u_old`` instead of
    doing a full ``(h·u)_new = (h·u)_old - dt * flux_div`` / ``u_new =
    (h·u)_new / h_u_new`` update leaves a residual
    ``O(dt · u · dh_u/dt / h_u)`` error under dynamic z-star.  A full
    flux-form momentum update requires restructuring the model step
    function (Level 2 in the #171 discussion) and is still open.

    Parameters
    ----------
    u : array, shape (..., nlev)
        Velocity at full levels at the momentum points (u-face, v-face,
        or edge — caller's choice, as long as ``w_half`` and ``h_u``
        are interpolated to the same points).
    w_half : array, shape (..., nlev+1)
        Vertical velocity on half (interface) levels, at the same
        momentum points as ``u``.  Positive = upward.  Must be zero
        at the surface and bottom interfaces.
    h_u : array, shape (..., nlev)
        Layer thickness at the momentum points.  Used only as the
        advective-form denominator.
    face_active : array | None, shape (..., nlev)
        Optional per-level face-activity mask (1 = wet face, 0 = closed
        face below the partial seafloor).  When provided, the vertical
        flux at any interface bordering an inactive face level is
        gated to exactly zero — same purpose as the ``cell_active``
        argument of the tracer helper, applied here to momentum.

    Returns
    -------
    tendency : array, shape (..., nlev)
        ``-(F_top - F_bot) / h_u`` — a per-thickness momentum tendency
        ready to add to ``du/dt``.
    """
    vert_flux_div = flux_form_vertical_tracer_advection(
        u, w_half, cell_active=face_active,
    )
    h_u_safe = jnp.maximum(h_u, 1.0e-10)
    return -vert_flux_div / h_u_safe


def flux_form_vertical_momentum_advection_centered(
    u: jnp.ndarray,
    w_half: jnp.ndarray,
    h_u: jnp.ndarray,
    face_active: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Veros-faithful 2nd-order CENTERED vertical momentum advection.

    Same per-thickness advective-tendency interface as
    :func:`flux_form_vertical_momentum_advection` (the 1st-order upwind
    version), but the interface velocity is the UNLIMITED 2-cell average

        ``F[k] = w_half[k] * 0.5*(u[k-1] + u[k])``   (1 <= k <= nlev-1)
        ``F[0] = F[nlev] = 0``                        (rigid-lid / no-flux)

    and the tendency is ``-(F_top - F_bot) / h_u`` per level.  This is the
    vertical part of Veros ``core/momentum.py`` ``momentum_advection``
    (``flux_top = 0.25*(u[k+1]+u[k])*(wtr+wtr_east)``: the ``0.25`` is
    ``0.5`` for the 2-cell ``u`` average times ``0.5`` for the
    interpolation of ``w`` to the momentum point — the latter is handled
    by the caller, which passes ``w_half`` already interpolated to the
    u/v face).  Veros leaves ``flux_top`` zero at both the surface and the
    bottom interface (``flux_top[..., :-1]`` set; bottom skipped in the
    ``du_adv[1:] += flux_top[:-1]`` update), matching the zero-pad at both
    ends here.

    Energy property
    ---------------
    The centered (skew-symmetric) flux conserves the vertical-advection
    contribution to column kinetic energy to machine precision for a
    non-divergent column ``w`` (``w_half[0]=w_half[nlev]=0``): the discrete
    ``sum_k u[k] * (F[k]-F[k+1])`` telescopes to a boundary term that
    vanishes.  The 1st-order upwind flux is strictly KE-dissipative
    (implicit vertical viscosity ``~|w|*dz/2``), which damps baroclinic
    shear; the centered scheme removes that damping.

    DISPERSION / STABILITY
    ----------------------
    The centered face value is UNLIMITED, so this scheme is dispersive (no
    monotonicity, no implicit viscosity).  Stability rests on the same
    ingredients Veros relies on: a short momentum time step (``dt_mom``)
    and explicit/implicit vertical friction (background ``A_v`` + TKE/KPP
    ``kappaM``).  Use only with those in place (the ACC recipe).

    Caller convention
    ------------------
    To reproduce Veros, pass the FULL face velocity ``u`` (barotropic +
    baroclinic).  This restores the depth-integral-zero redistribution
    term ``-d/dz(w * U_bar)`` that advecting the perturbation ``u' =
    u - U_bar`` alone omits — the term that vertically redistributes
    barotropic momentum into shear.

    Parameters
    ----------
    u : array, shape (..., nlev)
        Velocity at full levels at the momentum point.  Pass the FULL
        velocity for the Veros-faithful behaviour.
    w_half : array, shape (..., nlev+1)
        Vertical velocity on half (interface) levels at the SAME momentum
        point as ``u``.  Positive = upward; zero at surface and bottom.
    h_u : array, shape (..., nlev)
        Layer thickness at the momentum point (advective-form denominator).
    face_active : array | None, shape (..., nlev)
        Optional per-level face-activity mask (1 = wet, 0 = closed below
        the partial seafloor).  Gates the flux at any interface bordering
        an inactive face to exactly zero (same role as in the upwind /
        tracer helpers).

    Returns
    -------
    tendency : array, shape (..., nlev)
        ``-(F_top - F_bot) / h_u`` — per-thickness momentum tendency.
    """
    vert_flux_div = flux_form_vertical_tracer_advection_centered(
        u, w_half, cell_active=face_active,
    )
    h_u_safe = jnp.maximum(h_u, 1.0e-10)
    return -vert_flux_div / h_u_safe


# ---------------------------------------------------------------------------
# Adaptive-implicit vertical momentum advection
# (Shchepetkin 2015 / NEMO ``ln_zad_Aimp``)
# ---------------------------------------------------------------------------
#
# Explicit first-order-upwind vertical momentum advection (above) is only
# stable while the vertical Courant number ``Cw = |w| dt / h`` stays below
# 1.  In an OMIP cold-start the spurious equatorial pressure-gradient seed
# drives a transient convergence -> spurious ``w`` -> ``Cw > 1`` in thin
# cells, and the explicit scheme then amplifies it super-exponentially (the
# documented "vertadv is the residual amplifier" runaway).  NEMO removes
# this CFL limit with Shchepetkin's adaptive-implicit scheme: at each
# interface the vertical velocity is split ``w = w_exp + w_imp`` by a
# Courant-dependent fraction; ``w_exp`` (Courant-capped) goes through the
# normal explicit flux-form scheme, and ``w_imp`` is handled by a
# backward-Euler first-order-upwind solve that is unconditionally stable,
# monotone, and conservative (an M-matrix tridiagonal).
#
# Reference: A.F. Shchepetkin (2015), "An adaptive, Courant-number-dependent
# implicit scheme for vertical advection in oceanic modeling", Ocean
# Modelling 91, 38-69.  NEMO impl: sshwzv.F90 (split), trazdf.F90 /
# dynzdf.F90 (implicit solve).


def shchepetkin_implicit_fraction(
    cu: jnp.ndarray,
    cu_min: float = _AIMP_CU_MIN,
    cu_max: float = _AIMP_CU_MAX,
) -> jnp.ndarray:
    """Implicit fraction ``zcff(Cu)`` of Shchepetkin (2015) / NEMO wAimp.

    Maps a (non-negative) vertical Courant number ``cu`` to the fraction
    of the vertical velocity that is treated implicitly:

    - ``cu <= cu_min``                : ``0``  (fully explicit, high order)
    - ``cu_min < cu < cu_cut``        : ``d² / (Fcu + d²)``, ``d = cu-cu_min``
    - ``cu >= cu_cut``                : ``(cu - cu_max) / cu``
    - then clipped to ``<= 1``

    with ``cu_cut = 2 cu_max - cu_min`` and ``Fcu = 4 cu_max (cu_max-cu_min)``.
    The two interior branches join continuously at ``cu_cut`` (both give
    ``(cu_max-cu_min)/(2 cu_max-cu_min)``) and the ramp is monotone
    increasing from 0 to 1.  Pure ``jnp.where`` (no Python control flow) so
    it is safe on traced Courant numbers and differentiable.
    """
    cu_cut = 2.0 * cu_max - cu_min
    fcu = 4.0 * cu_max * (cu_max - cu_min)
    d = cu - cu_min
    mid = (d * d) / (fcu + d * d)
    # Guard the division in the high branch; cu >= cu_cut > 0 there.
    high = (cu - cu_max) / jnp.maximum(cu, 1.0e-30)
    zcff = jnp.where(
        cu <= cu_min,
        jnp.zeros_like(cu),
        jnp.where(cu < cu_cut, mid, high),
    )
    return jnp.minimum(zcff, 1.0)


def implicit_vertical_advection_ocean(
    field: jnp.ndarray,
    w_imp_half: jnp.ndarray,
    h: jnp.ndarray,
    dt: float,
    face_active: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Backward-Euler first-order-upwind vertical advection (one solve).

    Solves, per column, the unconditionally-stable implicit update

        field_new[k] + (dt/h[k]) * (F[k] - F[k+1]) = field[k]

    with the interface-upwind flux ``F[k] = max(w[k],0) field_new[k] +
    min(w[k],0) field_new[k-1]`` (``w`` positive = upward, interface ``k``
    sits above level ``k``; ``w[0] = w[nlev] = 0``).  Grouping by unknown
    gives the tridiagonal system

        a[k] =  dt * min(w_top[k], 0) / h[k]            (sub-diagonal)
        b[k] =  1 + dt*(max(w_top[k],0) - min(w_bot[k],0)) / h[k]  (diag)
        c[k] = -dt * max(w_bot[k], 0) / h[k]            (super-diagonal)
        d[k] =  field[k]                                (rhs)

    where ``w_top = w_imp_half[..., :-1]`` and ``w_bot = w_imp_half[..., 1:]``.
    This is an M-matrix (``b >= 1``, off-diagonals ``<= 0``) so the
    backward-Euler step is unconditionally stable and monotone, and the
    telescoping flux form conserves the column integral ``sum_k h[k]*field[k]``
    to machine precision (with the zero-flux top/bottom boundaries).

    Solver-conditioning note
    ------------------------
    ``thomas_solve`` adds a ``finfo(float32).tiny`` (~1.18e-38) guard to
    its pivots.  Because ``b >= 1`` here, ``b + tiny`` underflows back to
    ``b`` at both float32 and float64 (``tiny`` is below the ULP of any
    number ``>= 1``), and the forward-elimination pivots of a
    diagonally-dominant M-matrix stay ``O(1)`` so the ``|denom| < tiny``
    fallback is never selected.  The guard is therefore inert for this
    matrix class: the solve is *exactly* the conservative finite-volume
    solve, a ``w = 0`` system is bit-exact identity, and an inactive
    (``a = c = 0, b = 1``) row reproduces its input to round-off.
    (Verified: ``test_w_zero_is_exact_identity``,
    ``test_conservation_machine_precision_high_courant``.)

    Parameters mirror :func:`flux_form_vertical_momentum_advection`.
    ``face_active`` (1 = wet, 0 = below the partial seafloor) gates the
    implicit flux at interfaces bordering any inactive cell to exactly
    zero, so rock cells decouple (``a = c = 0``, ``b = 1`` -> identity) and
    cannot mix spurious values up the column.
    """
    nlev = field.shape[-1]
    if nlev < 2:
        return field

    w = w_imp_half
    if face_active is not None:
        # Interior interface i (1..nlev-1) is between cells i-1 and i;
        # gate it to zero unless both are active.  Surface/bottom
        # interfaces are zero by construction (w_half[0]=w_half[nlev]=0).
        active_above = face_active[..., :-1]   # cells 0..nlev-2
        active_below = face_active[..., 1:]    # cells 1..nlev-1
        face_int = active_above * active_below
        pad_axes = ((0, 0),) * (face_int.ndim - 1)
        gate = jnp.pad(face_int, (*pad_axes, (1, 1)))   # (..., nlev+1)
        w = w * gate

    w_top = w[..., :-1]    # interface above each cell, (..., nlev); w_top[0]=0
    w_bot = w[..., 1:]     # interface below each cell, (..., nlev); w_bot[-1]=0
    inv_h = 1.0 / jnp.maximum(h, _H_FLOOR)

    a = dt * jnp.minimum(w_top, 0.0) * inv_h
    c = -dt * jnp.maximum(w_bot, 0.0) * inv_h
    b = 1.0 + dt * (jnp.maximum(w_top, 0.0) - jnp.minimum(w_bot, 0.0)) * inv_h

    return thomas_solve(a, b, c, field)


def adaptive_implicit_vertical_momentum_advection(
    u: jnp.ndarray,
    w_half: jnp.ndarray,
    h: jnp.ndarray,
    dt: float,
    *,
    cu_min: float = _AIMP_CU_MIN,
    cu_max: float = _AIMP_CU_MAX,
    face_active: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Adaptive-implicit vertical momentum advection (Shchepetkin 2015).

    Returns the velocity field ``u`` after one step of vertical advection
    ``-w du/dz``, split into a Courant-capped explicit part and an
    unconditionally-stable backward-Euler implicit part so the explicit
    flux-form scheme can never violate the vertical CFL limit (NEMO
    ``ln_zad_Aimp``).  Reduces *exactly* to the explicit
    :func:`flux_form_vertical_momentum_advection` wherever the vertical
    Courant number stays below ``cu_min`` (``w_imp = 0`` there), so it is a
    drop-in robustness upgrade that only changes the answer in the
    high-Courant cells where the explicit scheme is unstable anyway.

    Parameters
    ----------
    u : array, shape ``(..., nlev)``
        Velocity at the momentum point (u-face or v-face).
    w_half : array, shape ``(..., nlev+1)``
        Vertical velocity on interfaces at the same momentum point,
        positive up, zero at surface and bottom.
    h : array, shape ``(..., nlev)``
        Layer thickness at the momentum point.
    dt : float
        Time step [s].
    cu_min, cu_max : float
        Shchepetkin Courant thresholds (see
        :func:`shchepetkin_implicit_fraction`).
    face_active : array | None, shape ``(..., nlev)``
        Per-level face-activity mask (1 = wet, 0 = below seafloor).
    """
    nlev = u.shape[-1]
    if nlev < 2:
        return u

    inv_h = 1.0 / jnp.maximum(h, _H_FLOOR)
    # Per-cell vertical (outflow) Courant number: the upward outflow
    # through the top interface plus the downward outflow through the
    # bottom interface, normalised by the cell thickness.  This is the
    # quantity the vertical-advection CFL limit constrains.  (NEMO also
    # folds in the horizontal flux divergence to form a single combined
    # Courant number; we use the vertical-only Courant because only the
    # vertical advection is being made implicit here -- the horizontal
    # CFL is governed separately by dt and the grid spacing.)
    w_top = w_half[..., :-1]
    w_bot = w_half[..., 1:]
    cu_cell = dt * (jnp.maximum(w_top, 0.0) - jnp.minimum(w_bot, 0.0)) * inv_h

    # Interface Courant number = max over the two adjacent cells
    # (interior interfaces 1..nlev-1); pad the surface/bottom interfaces
    # with zero (w_half is zero there anyway).
    cu_iface = jnp.maximum(cu_cell[..., :-1], cu_cell[..., 1:])  # (..., nlev-1)
    zcff_int = shchepetkin_implicit_fraction(cu_iface, cu_min, cu_max)
    pad_axes = ((0, 0),) * (zcff_int.ndim - 1)
    zcff = jnp.pad(zcff_int, (*pad_axes, (1, 1)))                # (..., nlev+1)

    w_imp = zcff * w_half
    w_exp = (1.0 - zcff) * w_half

    # Explicit (Courant-capped) part through the existing flux-form scheme.
    tend_exp = flux_form_vertical_momentum_advection(
        u, w_exp, h, face_active=face_active,
    )
    u_exp = u + dt * tend_exp

    # Implicit part: unconditionally-stable backward-Euler upwind solve.
    return implicit_vertical_advection_ocean(
        u_exp, w_imp, h, dt, face_active=face_active,
    )


def flux_form_vertical_tracer_advection(
    field: jnp.ndarray,
    w_half: jnp.ndarray,
    cell_active: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Flux-form vertical tracer advection with first-order upwind.

    Computes the vertical flux divergence  F_top[k] - F_bot[k]  for each
    level k, where F = w * T_face is the upward tracer flux on interfaces.

    Level convention
    ----------------
    k = 0 is surface, k = nlev-1 is bottom.
    Interface k sits ABOVE level k:
      - interface 0  = sea surface  (top of level 0)
      - interface k  = between level k-1 (above) and level k (below), k=1..nlev-1
      - interface nlev = ocean bottom (below level nlev-1)
    w positive = upward.

    Upwind at interior interface k (k = 1 .. nlev-1):
      - w[k] > 0  (upward):  fluid from level k  (below) → T_face = field[k]
      - w[k] <= 0 (downward): fluid from level k-1 (above) → T_face = field[k-1]

    Surface and bottom fluxes are zero (w[0] = w[nlev] = 0 by construction).

    Parameters
    ----------
    field : array, shape (..., nlev)
        Tracer at full levels (e.g. temperature [degC]).
    w_half : array, shape (..., nlev+1)
        Vertical velocity on half (interface) levels [m/s].
    cell_active : array | None, shape (..., nlev)
        Optional per-cell activity mask (1 = wet, 0 = below seafloor).
        When provided, the flux at any interface bordering an inactive
        cell is gated to exactly zero — needed on partial-cell grids
        where ``w_half`` may carry float-precision noise (~1e-10 m/s)
        at inactive interfaces.  Without the gate, that noise produces
        a tiny spurious ``vert_flux_div`` at inactive cells; combined
        with the ``max(h_k_new, 1e-10)`` floor at the caller, this can
        amplify into ~1e3 spurious tracer values inside the rock, then
        propagate into the EOS as huge density and break the model.
        Also makes adjoint sensitivities through inactive cells exactly
        zero (under ``jax.grad``), instead of poorly-conditioned values
        depending on the float noise.

    Returns
    -------
    vert_flux_div : array, shape (..., nlev)
        Vertical flux divergence  F_top[k] - F_bot[k]  for each level.
        Units are [tracer] * [m/s]  (NOT divided by layer thickness).
        The caller uses:  h_new*T_new = h_old*T_old - dt*vert_flux_div - dt*horiz_flux_div
    """
    nlev = field.shape[-1]

    # --- Compute upwind tracer flux at each interface ---
    # F has shape (..., nlev+1).  F[..., 0] = 0, F[..., nlev] = 0.
    # For interior interface k (1 <= k <= nlev-1):
    #   F[k] = w[k] * T_face[k]
    #   where T_face[k] = field[k]   if w[k] > 0   (upward, from below)
    #                    = field[k-1] if w[k] <= 0  (downward, from above)

    # Interior w values: w_half[..., 1:nlev] has shape (..., nlev-1)
    w_interior = w_half[..., 1:nlev]  # (..., nlev-1)

    # Upwind selection at interior interfaces
    # Interface k (1-indexed) is between level k-1 (above) and level k (below)
    T_below = field[..., 1:]    # field[k]   for k=1..nlev-1 → (..., nlev-1)
    T_above = field[..., :-1]   # field[k-1] for k=1..nlev-1 → (..., nlev-1)

    T_face_interior = jnp.where(w_interior > 0.0, T_below, T_above)
    F_interior = w_interior * T_face_interior  # (..., nlev-1)

    # Mask the flux at interfaces that border any inactive cell.  An
    # interior interface k (k=1..nlev-1) is between cells k-1 and k —
    # both must be active for the flux there to be physical.  The
    # surface (k=0) and bottom (k=nlev) interfaces are already zero by
    # the pad below.
    if cell_active is not None:
        active_above = cell_active[..., :-1]   # cells k-1 for k=1..nlev-1
        active_below = cell_active[..., 1:]    # cells k   for k=1..nlev-1
        face_active_interior = active_above * active_below
        F_interior = F_interior * face_active_interior

    # Full flux array with zero boundaries — single Pad HLO op vs
    # alloc fresh ``(..., 1)`` zero buffer and 3-array concatenate.
    pad_axes_f = ((0, 0),) * (F_interior.ndim - 1)
    F = jnp.pad(F_interior, (*pad_axes_f, (1, 1)))  # (..., nlev+1)

    # Flux divergence: F_top[k] - F_bot[k] = F[k] - F[k+1]
    vert_flux_div = F[..., :-1] - F[..., 1:]  # (..., nlev)

    return vert_flux_div


def flux_form_vertical_tracer_advection_centered(
    field: jnp.ndarray,
    w_half: jnp.ndarray,
    cell_active: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Flux-form vertical tracer advection with UNLIMITED centered 2nd order.

    Veros ``adv_flux_2nd`` vertical flux (``veros/core/advection.py``):
    the interface tracer flux uses the plain 2-cell average
    ``T_face[k] = 0.5*(field[k-1] + field[k])`` rather than the upwind /
    TVD-limited value.  This is the Veros ACC tracer scheme
    (``enable_superbee_advection=False``).

    Same level convention and output semantics as
    :func:`flux_form_vertical_tracer_advection` (the 1st-order upwind
    version): k=0 surface, interface k sits ABOVE level k, surface and
    bottom interface fluxes are zero, and the returned ``vert_flux_div``
    is ``F_top[k] - F_bot[k]`` in units ``[tracer]*[m/s]`` (NOT divided
    by layer thickness).

    DISPERSION
    ----------
    The centered face value is UNLIMITED, so this scheme is dispersive:
    it can produce over/undershoots (new local extrema, locally negative
    tracer) near sharp gradients.  It carries zero implicit diapycnal
    diffusion in smooth regions (unlike upwind/TVD).  Used for the
    Veros-faithful ACC comparison (``tracer_advection="centered"``);
    legoESM's production default stays TVD (Van Leer), which is monotone.

    Parameters
    ----------
    field : array, shape (..., nlev)
        Tracer at full levels.
    w_half : array, shape (..., nlev+1)
        Vertical velocity on half (interface) levels [m/s]; positive =
        upward; zero at surface and bottom.
    cell_active : array | None, shape (..., nlev)
        Optional per-cell activity mask (1 = wet, 0 = below seafloor).
        When provided, the flux at any interface bordering an inactive
        cell is gated to exactly zero (same role as in
        :func:`flux_form_vertical_tracer_advection`).

    Returns
    -------
    vert_flux_div : array, shape (..., nlev)
        Vertical flux divergence ``F_top[k] - F_bot[k]`` for each level.
    """
    nlev = field.shape[-1]

    # Interior interface k (1 <= k <= nlev-1) is between level k-1 (above)
    # and level k (below).  Centered face value = unlimited 2-cell average.
    w_interior = w_half[..., 1:nlev]    # (..., nlev-1)
    T_below = field[..., 1:]            # field[k]   for k=1..nlev-1
    T_above = field[..., :-1]           # field[k-1] for k=1..nlev-1
    T_face_interior = 0.5 * (T_above + T_below)
    F_interior = w_interior * T_face_interior  # (..., nlev-1)

    # Gate the flux at interfaces bordering any inactive cell (matches the
    # upwind version: both bordering cells must be active to be physical).
    if cell_active is not None:
        active_above = cell_active[..., :-1]   # cells k-1 for k=1..nlev-1
        active_below = cell_active[..., 1:]    # cells k   for k=1..nlev-1
        F_interior = F_interior * (active_above * active_below)

    # Zero surface/bottom interface fluxes via a single Pad HLO op.
    pad_axes_f = ((0, 0),) * (F_interior.ndim - 1)
    F = jnp.pad(F_interior, (*pad_axes_f, (1, 1)))  # (..., nlev+1)

    # Flux divergence: F_top[k] - F_bot[k] = F[k] - F[k+1]
    vert_flux_div = F[..., :-1] - F[..., 1:]
    return vert_flux_div


# Canonical Van Leer limiter from core (redundancy audit), aliased to the local
# vertical-advection private name so call sites are unchanged.
from legoesm.core.flux_limiters import (
    grad_safe_ratio as _grad_safe_ratio,
    ratio_grad_floor as _ratio_grad_floor,
    van_leer_limiter as _van_leer_limiter_vert,
)


def flux_form_vertical_tracer_advection_tvd(
    field: jnp.ndarray,
    w_half: jnp.ndarray,
    h_k: jnp.ndarray,
    dt: float,
    cell_active: jnp.ndarray | None = None,
    limiter_fn=_van_leer_limiter_vert,
) -> jnp.ndarray:
    """Flux-form vertical tracer advection with TVD scheme.

    ``limiter_fn`` selects the flux limiter family; defaults to Van Leer
    (used by ``tracer_advection="tvd"``). Pass
    :func:`legoesm.ocean.dynamics._flux_limiters.sweby_limiter` for
    Veros-compatible superbee (``tracer_advection="superbee"``).

    Second-order accurate in smooth regions, falls back to first-order
    upwind at discontinuities.  Monotone (no new extrema).  The implicit
    numerical diffusivity is dramatically reduced compared to first-order
    upwind: K_num ~ 0 in smooth regions vs K_num ~ |w|*dz/2 for upwind.

    Same output semantics as flux_form_vertical_tracer_advection.

    Parameters
    ----------
    field : array, shape (..., nlev)
        Tracer at full levels.
    w_half : array, shape (..., nlev+1)
        Vertical velocity on half (interface) levels [m/s].
        Positive = upward. Zero at surface and bottom.
    h_k : array, shape (..., nlev)
        Layer thickness [m] at full levels (z-star actual thickness).
    dt : float
        Time step [s], for CFL computation.
    cell_active : array, shape (..., nlev), optional
        Per-level active mask (1=ocean, 0=sub-seafloor).  When provided,
        sub-seafloor ghost values in the upwind-of-upwind stencil are
        replaced with the boundary active value, preventing the TVD
        limiter from seeing T=0/S=0 below the seafloor.

    Returns
    -------
    vert_flux_div : array, shape (..., nlev)
        Vertical flux divergence F_top[k] - F_bot[k] for each level.
        Units: [tracer]*[m/s] (NOT divided by layer thickness).
    """
    eps = 1e-30
    nlev = field.shape[-1]

    # On partial cells, replace sub-seafloor values with the nearest
    # active value above.  This prevents the TVD upwind-of-upwind
    # stencil from seeing T=0/S=0 below the seafloor.
    if cell_active is not None:
        # Propagate bottom active value downward through inactive levels.
        # Scan from top to bottom: if level k is inactive, copy from k-1.
        def _fill_down(carry, k):
            prev = carry
            cur = field[..., k]
            active_k = cell_active[..., k] > 0.5
            filled = jnp.where(active_k, cur, prev)
            return filled, filled
        import jax.lax
        _, filled_cols = jax.lax.scan(
            _fill_down, field[..., 0], jnp.arange(nlev))
        # filled_cols is (nlev, ...) — transpose back to (..., nlev)
        field_safe = jnp.moveaxis(filled_cols, 0, -1)
    else:
        field_safe = field

    # Interior interface values: k = 1..nlev-1
    w_interior = w_half[..., 1:nlev]   # (..., nlev-1)
    T_below = field_safe[..., 1:]      # field[k]   for k=1..nlev-1
    T_above = field_safe[..., :-1]     # field[k-1] for k=1..nlev-1

    # --- First-order upwind flux ---
    T_upwind = jnp.where(w_interior > 0.0, T_below, T_above)
    F_upwind = w_interior * T_upwind

    # --- CFL number at each interface ---
    h_below = h_k[..., 1:]            # h[k]   for k=1..nlev-1
    h_above = h_k[..., :-1]           # h[k-1] for k=1..nlev-1
    h_donor = jnp.where(w_interior > 0.0, h_below, h_above)
    t_grad = _ratio_grad_floor(field.dtype)
    CFL = _grad_safe_ratio(
        jnp.abs(w_interior) * dt, jnp.maximum(h_donor, eps), h_donor > t_grad)
    CFL = jnp.minimum(CFL, 1.0)

    # --- Smoothness ratio r ---
    # Local gradient across interface k:
    delta = T_above - T_below          # field[k-1] - field[k]

    # Upwind-of-upwind gradient:
    # For upward flow (w>0), donor=k(below): need field[k]-field[k+1]
    # For downward flow (w<=0), donor=k-1(above): need field[k-2]-field[k-1]
    # Ghost cells at boundaries copy boundary value → delta=0 → r=0 → upwind.
    # Using field_safe ensures sub-seafloor ghost = bottom active value.
    field_bot_ghost = jnp.concatenate(
        [field_safe, field_safe[..., -1:]], axis=-1)     # ghost at bottom
    field_top_ghost = jnp.concatenate(
        [field_safe[..., :1], field_safe], axis=-1)      # ghost at top

    # Upwind gradient for upward flow: field[k] - field[k+1]
    delta_upwind_up = field_bot_ghost[..., 1:nlev] - field_bot_ghost[..., 2:nlev + 1]
    # Upwind gradient for downward flow: field[k-2] - field[k-1]
    delta_upwind_down = field_top_ghost[..., :nlev - 1] - field_top_ghost[..., 1:nlev]

    delta_upwind = jnp.where(w_interior > 0.0, delta_upwind_up, delta_upwind_down)

    # r = upwind_gradient / local_gradient
    r = _grad_safe_ratio(
        delta_upwind,
        jnp.where(jnp.abs(delta) > eps, delta, eps),
        jnp.abs(delta) > t_grad,
    )

    # --- Flux limiter and TVD correction ---
    phi = limiter_fn(r)
    F_interior = F_upwind + 0.5 * jnp.abs(w_interior) * (1.0 - CFL) * phi * delta

    # Full flux array with zero boundaries — single Pad HLO op.
    pad_axes_t = ((0, 0),) * (F_interior.ndim - 1)
    F = jnp.pad(F_interior, (*pad_axes_t, (1, 1)))

    # Flux divergence: F_top[k] - F_bot[k] = F[k] - F[k+1]
    vert_flux_div = F[..., :-1] - F[..., 1:]

    return vert_flux_div
