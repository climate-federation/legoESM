"""Grid-agnostic baroclinic tendency helpers shared by ocean_pe_*.py files.

This module factors logic that was previously duplicated across the
A-grid (``ocean_pe_cdgrid.py``), lat-lon C-grid
(``ocean_pe_latlon_cgrid.py``), and MPAS Voronoi (``ocean_pe_mpas.py``)
baroclinic tendency entry points.  Grid-specific operators (gradient,
divergence, vorticity, fill, ...) are passed in as callables so this
module never imports from a particular grid package.

Closes #214 (Phase 1).

Functions
---------
``iterate_eos_and_pressure_anomaly``
    EOS iteration (T_filled, S_filled, p) → ρ, ρ', p' using a
    reference-thickness hydrostatic integral.  The cubed-sphere caller
    can request high-precision arithmetic for the cumsum so that the
    halo-exchanged p' gradient stays clean.

``apply_sponge_tracer_relaxation``
    Linear restoring of T, S towards reference fields with rate
    ``γ``.  Used identically by the lat-lon C-grid and MPAS callers.

``apply_freshwater_virtual_salt_top``
    Top-layer salinity tendency from the freshwater volume flux.

``implicit_bottom_drag_factor``
    Returns ``1 - dt * r / max(H, eps)`` — the per-substep multiplicative
    factor used by both C-grid lat-lon and MPAS barotropic substeps.

These helpers are pure and pytree-friendly: they accept and return
``jax.Array`` values and never mutate inputs.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import jax.numpy as jnp

from legoesm.ocean.eos import compute_hydrostatic_pressure
from legoesm.ocean.freshwater import virtual_salt_flux


def iterate_eos_and_pressure_anomaly(
    T: jnp.ndarray,
    S: jnp.ndarray,
    mask: jnp.ndarray,
    fill_fn: Callable[[jnp.ndarray], jnp.ndarray],
    eos_fn: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    dz_ref: jnp.ndarray,
    rho_0: float,
    g: float,
    *,
    n_iter: int = 2,
    hi_precision_pressure: bool = False,
    h_actual: jnp.ndarray | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run the standard 2-pass EOS iteration and form ``p_prime``.

    Replicates the identical iteration that previously lived inline in
    every ``ocean_pe_*.py`` file:

    1. Fill land-cell ``T``, ``S`` with ocean-neighbour values via
       ``fill_fn`` so the EOS does not produce spurious ``ρ'`` values
       on land that contaminate the gradient at coastlines.
    2. Iterate ``ρ ← EOS(T, S, p_hydro(ρ))`` ``n_iter`` times against
       the **reference** thickness profile ``dz_ref`` (i.e. ``J=1``,
       ``η=0``).  Using the actual Jacobian here would double-count
       the ``-g·∇η`` forcing handled by the barotropic solver.
    3. Build the layer-centred baroclinic pressure anomaly

       ``p'(k) = g · Σ_{j<k} ρ'(j) · dz_ref(j) + 0.5 · g · ρ'(k) · dz_ref(k)``

       which equals the half-trapezoidal cumulative integral of
       ``g·ρ'`` from the surface to the layer mid-point.

    Parameters
    ----------
    T, S : jax.Array
        Tracer fields with a trailing vertical axis (``..., nlev``).  All
        upstream callers store T and S with identical shapes.
    mask : jax.Array
        Land mask with the same horizontal shape as ``T[..., 0]`` (used
        only by ``fill_fn``; passed back to the caller for any post-
        processing it needs).
    fill_fn : Callable[[jax.Array], jax.Array]
        Grid-specific land-cell filler.  Must accept and return arrays
        with the same shape as ``T``.  Typical implementations:
        ``jax.vmap(fill_land_cells_cubed_sphere, in_axes=-1)``,
        ``_neumann_fill_cgrid`` (lat-lon C-grid), or
        ``fill_land_cells_mpas`` (MPAS).
    eos_fn : Callable
        Equation of state ``(T, S, p) → ρ``.  Same signature used by
        every grid (built via ``make_eos_fn``).
    dz_ref : jax.Array
        Reference layer thickness (``z_coord.dz_ref``), shape
        ``(nlev,)``.
    rho_0, g : float
    n_iter : int, default 2
        Number of EOS iterations *before* the final pressure update.
        All current callers use 2.
    hi_precision_pressure : bool, default False
        If True, perform the pressure cumsum in float64 so that
        halo-exchange interpolation errors do not contaminate the
        downstream gradient.  Used by the cubed-sphere C-D path; on
        lat-lon and MPAS the compact 2-cell stencils are well-behaved
        enough that the working precision is sufficient.

    Returns
    -------
    rho : jax.Array
        In-situ density (same shape as ``T``).
    rho_prime : jax.Array
        ``rho - rho_0`` (same shape as ``T``).
    p_prime : jax.Array
        Baroclinic pressure anomaly (same shape as ``T``).  Returned in
        whatever precision was used for the cumulative sum.
    """
    del mask  # currently unused (passed to fill_fn by the caller); kept
              # in signature for clarity at call sites.

    T_filled = fill_fn(T)
    S_filled = fill_fn(S)

    # Reference Jacobian (J=1, η=0).  Both have the horizontal shape of
    # T (i.e. no vertical axis).  Match dtype to the working state so we
    # never accidentally promote the EOS iteration to float64.
    horiz_shape = T.shape[:-1]
    J_ref = jnp.ones(horiz_shape, dtype=T.dtype)
    eta_ref = jnp.zeros(horiz_shape, dtype=T.dtype)

    rho = eos_fn(T_filled, S_filled, jnp.zeros_like(T))
    for _ in range(n_iter):
        p_hydro = compute_hydrostatic_pressure(
            rho, eta_ref, dz_ref, J_ref, rho_0, g,
            h_actual=h_actual,
        )
        rho = eos_fn(T_filled, S_filled, p_hydro)

    rho_prime = rho - rho_0

    # Layer-thickness array used in the baroclinic-anomaly cumsum.
    # When ``h_actual`` is None, use the reference ``dz_ref`` (legacy z*
    # path).  When provided (partial-cell path), use per-cell thickness
    # so the anomaly integrates to each cell's actual centroid depth.
    # Cells below the seafloor have h_actual=0 and contribute zero.
    if h_actual is None:
        h_for_cumsum = dz_ref
    else:
        h_for_cumsum = h_actual

    if hi_precision_pressure:
        rho_prime_hi = rho_prime.astype(jnp.float64)
        h_hi = jnp.asarray(h_for_cumsum, dtype=jnp.float64)
        dp_layer = rho_prime_hi * g * h_hi
    else:
        dp_layer = rho_prime * g * h_for_cumsum

    p_prime = jnp.cumsum(dp_layer, axis=-1) - dp_layer
    p_prime = p_prime + 0.5 * dp_layer

    return rho, rho_prime, p_prime


def apply_sponge_tracer_relaxation(
    dT_dt: jnp.ndarray,
    dS_dt: jnp.ndarray,
    T: jnp.ndarray,
    S: jnp.ndarray,
    sponge,
    mask: Optional[jnp.ndarray] = None,
    *,
    expand_gamma_axis: int = -1,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply tracer sponge relaxation ``+γ·(ref - q)``.

    Casts ``sponge`` arrays to ``T.dtype`` so the precision policy that
    holds the state in float32 is not silently promoted to float64
    (which previously crashed the barotropic scan; see latlon C-grid).

    Parameters
    ----------
    dT_dt, dS_dt : jax.Array
        Existing tendency arrays (modified by addition).
    T, S : jax.Array
        Current tracer state.
    sponge : SpongeForcing
        Must expose ``gamma``, ``T_ref``, ``S_ref``.  ``gamma`` is the
        relaxation rate per cell (1 / s), broadcast along the vertical
        axis via ``expand_gamma_axis``.
    mask : jax.Array, optional
        Ocean mask.  When provided, the relaxation tendency is
        multiplied by ``mask`` (with the same axis expansion as
        ``gamma``) so land cells stay quiescent.  When ``None`` no
        masking is applied (the caller masks downstream).
    expand_gamma_axis : int, default -1
        Axis on which to insert a singleton in ``sponge.gamma`` so that
        it broadcasts against the (..., nlev) tracer arrays.  Use ``-1``
        for both lat-lon C-grid (axis after lat/lon) and MPAS (axis
        after nCells).

    Returns
    -------
    (dT_dt_new, dS_dt_new) : tuple of jax.Array
    """
    dtype = T.dtype
    gamma = sponge.gamma.astype(dtype)
    gamma_b = jnp.expand_dims(gamma, expand_gamma_axis)
    dT = gamma_b * (sponge.T_ref.astype(dtype) - T)
    dS = gamma_b * (sponge.S_ref.astype(dtype) - S)
    if mask is not None:
        mask_b = jnp.expand_dims(mask, expand_gamma_axis)
        dT = dT * mask_b
        dS = dS * mask_b
    return dT_dt + dT, dS_dt + dS


def apply_freshwater_virtual_salt_top(
    dS_dt: jnp.ndarray,
    freshwater,
    S_ref: float,
    h_top: jnp.ndarray,
    rho_0: float,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """Add the surface virtual-salt tendency to the top tracer level.

    Wraps ``freshwater.virtual_salt_flux`` and assigns the result to
    ``dS_dt[..., 0]`` (or the equivalent leading-axis slice for MPAS).
    All callers operate on cell-centred salinity, so the trailing
    ``nlev`` axis is the vertical axis.

    Parameters
    ----------
    dS_dt : jax.Array
        Salinity tendency, modified at index ``[..., 0]``.
    freshwater : FreshwaterForcing
    S_ref : float
        Reference salinity used for the virtual-flux closure.
    h_top : jax.Array
        Top-layer thickness with the same horizontal shape as ``mask``.
    rho_0 : float
    mask : jax.Array
        Ocean mask (1 = ocean) with the same horizontal shape as the
        leading axes of ``dS_dt``.

    Returns
    -------
    jax.Array
        ``dS_dt`` with the virtual-salt flux added to the top layer.
    """
    dS_top = virtual_salt_flux(freshwater, S_ref, h_top, rho_0)
    # Cast the freshwater contribution to dS_dt's dtype so the scatter
    # add does not silently widen on x64 mode (the freshwater struct
    # is built at JAX-default precision in init helpers, which can be
    # f64 while the salinity tendency runs at the storage policy's
    # f32).
    return dS_dt.at[..., 0].add((dS_top * mask).astype(dS_dt.dtype))


def implicit_bottom_drag_factor(
    dt: jnp.ndarray | float,
    drag_r: jnp.ndarray | float,
    H: jnp.ndarray,
    *,
    eps: float = 1e-10,
) -> jnp.ndarray:
    """Per-substep bottom-drag multiplier ``1 - dt · r / max(H, eps)``.

    Used by both the lat-lon C-grid and MPAS barotropic substeps to
    apply a linear bottom drag on the depth-averaged velocity.  The
    floor on ``H`` prevents the drag from blowing up over very thin
    water columns (≈ inundation).

    Parameters
    ----------
    dt : float or jax.Array
        Substep size [s].
    drag_r : float or jax.Array
        Linear drag coefficient [m/s].
    H : jax.Array
        Total water column depth at the velocity location [m].
    eps : float
        Floor on ``H`` for numerical safety.

    Returns
    -------
    jax.Array, same shape as ``H``.
    """
    return 1.0 - dt * drag_r / jnp.maximum(H, eps)
