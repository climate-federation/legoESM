"""CLUBB staggered-grid container and interpolation/derivative operators.

Part of the fuller CLUBB port (see ``PORT_CLUBB.md``). CLUBB closes its
higher-order moment equations on a *staggered* vertical grid with two level
sets (Golaz et al. 2002, JAS 59:3540-3551, Sec. 3c):

  * **momentum levels** ``zm`` (``nzm`` of them) — carry fluxes / w-moments
    (``wp2``, ``wp3``, ``wpthlp``, ``wprtp``, ``up2``, ``vp2``, ...);
  * **thermodynamic levels** ``zt`` (``nzt`` of them) — carry means and scalar
    variances (``thlm``, ``rtm``, ``um``, ``vm``, ``thlp2``, ``rtp2``, ...).

Both are **ascending** (index 0 = lowest level, nearest the surface), which is
the CLUBB convention and the OPPOSITE of legoESM's top-down full/half ordering.
The legoESM<->CLUBB orientation flip and grid construction from
``z_full``/``z_half`` lives in the ``clubb.py`` adapter, NOT here; this module
operates purely on an already-built ascending CLUBB grid.

This is a faithful port of the operators in
``../CLUBB-JAX/clubb_jax/src/CLUBB_core/grid_class.py`` (itself a port of
``grid_class.F90``), adapted to legoESM conventions: pure pytree functions, no
module-scope ``jit`` (the caller JITs the whole physics step), and a typed
``CLUBBGrid`` ``NamedTuple`` (a JAX pytree) instead of a mutable ``gr`` object.

Operator summary (array layout ``(ngrdcol, nz)``):
  * :func:`zm2zt` — momentum-level field -> thermodynamic-level field (linear).
  * :func:`zt2zm` — thermodynamic-level field -> momentum-level field (linear).
  * :func:`ddzm`  — d/dz of a momentum-level field, evaluated at zt levels.
  * :func:`ddzt`  — d/dz of a thermo-level field, evaluated at zm levels.
  * :func:`zm2zt2zm` / :func:`zt2zm2zt` — round-trip smoothers.

The two interpolation weights are computed DIRECTLY (``w_above`` and
``w_below`` each from the grid spacings) rather than as ``w_below = 1 -
w_above``. They are identical (0.5) on a uniform grid but differ by ~1 ULP on a
stretched grid; the direct form is the faithful one (``calc_zm2zt_weights`` /
``calc_zt2zm_weights`` in grid_class.F90).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class CLUBBGrid(NamedTuple):
    """Ascending staggered vertical grid for the CLUBB closure (a JAX pytree).

    All arrays are shape ``(ngrdcol, nz*)`` with ascending levels (index 0 is
    the lowest, nearest the surface). ``nzm == nzt + 1`` in the standard CLUBB
    layout (momentum levels bracket thermodynamic levels), but the operators
    here only require the shape relationships noted per function, so the
    container does not hard-enforce it.

    Attributes
    ----------
    zm : jax.Array
        Momentum-level heights [m], shape ``(ngrdcol, nzm)``.
    zt : jax.Array
        Thermodynamic-level heights [m], shape ``(ngrdcol, nzt)``.
    invrs_dzm : jax.Array
        ``1 / (zt[k] - zt[k-1])`` evaluated at momentum levels [1/m], shape
        ``(ngrdcol, nzm)``. Used by :func:`ddzt`.
    invrs_dzt : jax.Array
        ``1 / (zm[k+1] - zm[k])`` evaluated at thermodynamic levels [1/m],
        shape ``(ngrdcol, nzt)``. Used by :func:`ddzm`.
    """

    zm: jax.Array
    zt: jax.Array
    invrs_dzm: jax.Array
    invrs_dzt: jax.Array


def zm2zt(azm: jax.Array, gr: CLUBBGrid) -> jax.Array:
    """Interpolate a momentum-level field to thermodynamic levels (linear).

    Faithful port of Fortran ``linear_interpolated_azt_2D``. For each thermo
    level ``k`` (ascending grid)::

        azt[k] = w_above * azm[k+1] + w_below * azm[k]
        w_above = (zt[k] - zm[k])   / (zm[k+1] - zm[k])
        w_below = (zm[k+1] - zt[k]) / (zm[k+1] - zm[k])

    Parameters
    ----------
    azm : jax.Array
        Field on momentum levels, shape ``(ngrdcol, nzm)``.
    gr : CLUBBGrid
        Grid; uses ``gr.zm`` ``(ngrdcol, nzm)`` and ``gr.zt`` ``(ngrdcol,
        nzt)`` with ``nzt = nzm - 1``.

    Returns
    -------
    jax.Array
        Field on thermodynamic levels, shape ``(ngrdcol, nzt)``.
    """
    zm = gr.zm
    zt = gr.zt
    dzt = zm[:, 1:] - zm[:, :-1]          # (ngrdcol, nzt), = nzm-1
    w_above = (zt - zm[:, :-1]) / dzt
    w_below = (zm[:, 1:] - zt) / dzt
    return w_above * azm[:, 1:] + w_below * azm[:, :-1]


def zt2zm(azt: jax.Array, gr: CLUBBGrid, zm_min: float | None = None) -> jax.Array:
    """Interpolate a thermodynamic-level field to momentum levels (linear).

    Faithful port of Fortran ``linear_interpolated_azm_2D`` (ascending grid).

    Interior ``k = 1 .. nzm-2``::

        azm[k] = w_above * azt[k] + w_below * azt[k-1]
        w_above = (zm[k] - zt[k-1]) / (zt[k] - zt[k-1])
        w_below = (zt[k] - zm[k])   / (zt[k] - zt[k-1])

    Boundaries (ascending grid):
      * lower ``k=0``: ``azm[0] = azt[0]`` (Fortran sets it directly);
      * upper ``k=nzm-1``: linear extension from ``zt[-2], zt[-1]``.

    Parameters
    ----------
    azt : jax.Array
        Field on thermodynamic levels, shape ``(ngrdcol, nzt)``.
    gr : CLUBBGrid
        Grid; uses ``gr.zm`` ``(ngrdcol, nzm)`` and ``gr.zt`` ``(ngrdcol,
        nzt)`` with ``nzm = nzt + 1``.
    zm_min : float, optional
        Lower clamp applied after interpolation (e.g. positivity floor for
        variances). ``None`` leaves the field unclamped.

    Returns
    -------
    jax.Array
        Field on momentum levels, shape ``(ngrdcol, nzm)``.
    """
    zm = gr.zm
    zt = gr.zt

    # Interior k = 1 .. nzm-2 (zt[k-1] < zm[k] < zt[k] for ascending grids).
    denom_int = zt[:, 1:] - zt[:, :-1]       # (ngrdcol, nzt-1) = (ngrdcol, nzm-2)
    zm_int = zm[:, 1:-1]                       # (ngrdcol, nzm-2)
    w_above_int = (zm_int - zt[:, :-1]) / denom_int
    w_below_int = (zt[:, 1:] - zm_int) / denom_int
    azm_int = w_above_int * azt[:, 1:] + w_below_int * azt[:, :-1]

    # Lower boundary: azm[0] = azt[0].
    azm_bot = azt[:, :1]

    # Upper boundary: linear extension above zt[-1].
    denom_top = zt[:, -1:] - zt[:, -2:-1]
    w_above_top = (zm[:, -1:] - zt[:, -2:-1]) / denom_top
    w_below_top = (zt[:, -1:] - zm[:, -1:]) / denom_top
    azm_top = w_above_top * azt[:, -1:] + w_below_top * azt[:, -2:-1]

    azm = jnp.concatenate([azm_bot, azm_int, azm_top], axis=1)
    if zm_min is not None:
        azm = jnp.maximum(azm, zm_min)
    return azm


def ddzm(azm: jax.Array, gr: CLUBBGrid) -> jax.Array:
    """Vertical derivative of a momentum-level field, at thermodynamic levels.

    Faithful port of Fortran ``gradzm_2D``::

        dazm_dz[k] = (azm[k+1] - azm[k]) * invrs_dzt[k]   for k = 0 .. nzt-1

    Parameters
    ----------
    azm : jax.Array
        Field on momentum levels, shape ``(ngrdcol, nzm)``.
    gr : CLUBBGrid
        Grid; uses ``gr.invrs_dzt`` ``(ngrdcol, nzt)``.

    Returns
    -------
    jax.Array
        Derivative on thermodynamic levels, shape ``(ngrdcol, nzt)``.
    """
    return (azm[:, 1:] - azm[:, :-1]) * gr.invrs_dzt


def ddzt(azt: jax.Array, gr: CLUBBGrid) -> jax.Array:
    """Vertical derivative of a thermodynamic-level field, at momentum levels.

    Faithful port of Fortran ``gradzt_2D``::

        interior k = 1 .. nzm-2: dazt_dz[k] = (azt[k] - azt[k-1]) * invrs_dzm[k]
        boundaries: dazt_dz[0] = dazt_dz[1],  dazt_dz[nzm-1] = dazt_dz[nzm-2]

    Parameters
    ----------
    azt : jax.Array
        Field on thermodynamic levels, shape ``(ngrdcol, nzt)``.
    gr : CLUBBGrid
        Grid; uses ``gr.invrs_dzm`` ``(ngrdcol, nzm)``.

    Returns
    -------
    jax.Array
        Derivative on momentum levels, shape ``(ngrdcol, nzm)``.
    """
    interior = (azt[:, 1:] - azt[:, :-1]) * gr.invrs_dzm[:, 1:-1]  # (ngrdcol, nzm-2)
    bottom = interior[:, :1]
    top = interior[:, -1:]
    return jnp.concatenate([bottom, interior, top], axis=1)


def zm2zt2zm(azm: jax.Array, gr: CLUBBGrid, zm_min: float | None = None) -> jax.Array:
    """Round-trip smoother ``zm -> zt -> zm`` (Fortran ``zm2zt2zm``)."""
    return zt2zm(zm2zt(azm, gr), gr, zm_min=zm_min)


def zt2zm2zt(azt: jax.Array, gr: CLUBBGrid, zt_min: float | None = None) -> jax.Array:
    """Round-trip smoother ``zt -> zm -> zt`` (Fortran ``zt2zm2zt``)."""
    result = zm2zt(zt2zm(azt, gr), gr)
    if zt_min is not None:
        result = jnp.maximum(result, zt_min)
    return result


def _safe_invrs(dz: jax.Array, floor: float = 1.0e-30) -> jax.Array:
    """Reciprocal with the reference's zero-spacing guard (``setup_grid``).

    Faithful to ``_calc_grid_spacings``: ``invrs = 1/dz`` where ``|dz| >
    floor`` else ``0``. Degenerate (zero) spacings therefore yield ``0`` rather
    than ``inf`` — the same graceful failure mode as the Fortran/JAX reference,
    and JIT/autodiff-safe (no data-dependent host control flow).
    """
    return jnp.where(jnp.abs(dz) > floor, 1.0 / jnp.where(jnp.abs(dz) > floor, dz, 1.0), 0.0)


def make_clubb_grid(zm: jax.Array, zt: jax.Array) -> CLUBBGrid:
    """Build a :class:`CLUBBGrid` from ascending ``zm``/``zt`` height arrays.

    Faithful to the canonical CLUBB grid construction
    (``grid_class.F90:setup_grid_heights`` /
    ``derived_types/grid_class.py:_calc_grid_spacings``) for an **ascending**
    grid:

      * ``dzt[k]  = zm[k+1] - zm[k]``                       (thermo levels, nzt)
      * interior ``dzm[k] = zt[k] - zt[k-1]``, ``k = 1 .. nzm-2``
      * lower boundary ``dzm[0]    = 2 * (zt[0] - zm[0])``  (NOT a copy of the
        adjacent interior spacing — the host grid's lowest thermo level need
        not be the midpoint of the two lowest momentum levels)
      * upper boundary ``dzm[nzm-1] = dzm[nzm-2]``          (copy adjacent)

    Inverses use the reference zero-spacing guard (:func:`_safe_invrs`). Only
    the *structural* preconditions (2-D, ``nzt == nzm - 1``, ``nzt >= 2``) are
    enforced here — these are static (shape-level) so they remain JIT-safe and
    fail before tracing array values. Strict-ascending monotonicity is a
    documented precondition; a violated (e.g. duplicate) level yields a ``0``
    inverse spacing there rather than a spurious ``inf`` (matching the
    reference), instead of a data-dependent runtime exception.

    Parameters
    ----------
    zm : jax.Array
        Ascending momentum-level heights [m], shape ``(ngrdcol, nzm)``.
    zt : jax.Array
        Ascending thermodynamic-level heights [m], shape ``(ngrdcol, nzt)``,
        with ``nzt = nzm - 1`` and ``nzt >= 2``.

    Returns
    -------
    CLUBBGrid
    """
    if zm.ndim != 2 or zt.ndim != 2:
        raise ValueError(
            f"zm and zt must be 2-D (ngrdcol, nz); got zm.ndim={zm.ndim}, "
            f"zt.ndim={zt.ndim}."
        )
    ngrdcol_m, nzm = zm.shape
    ngrdcol_t, nzt = zt.shape
    if ngrdcol_m != ngrdcol_t:
        raise ValueError(
            f"zm and zt must share ngrdcol; got {ngrdcol_m} vs {ngrdcol_t}."
        )
    if nzt != nzm - 1:
        raise ValueError(
            f"CLUBB staggered grid requires nzt == nzm - 1; got nzm={nzm}, "
            f"nzt={nzt}."
        )
    if nzt < 2:
        raise ValueError(
            f"CLUBB grid needs nzt >= 2 (nzm >= 3) for the staggered "
            f"interpolation/derivative stencils; got nzt={nzt}."
        )

    dzt = zm[:, 1:] - zm[:, :-1]                          # (ngrdcol, nzt)
    dzm_int = zt[:, 1:] - zt[:, :-1]                       # (ngrdcol, nzt-1) = (nzm-2)
    dzm_lower = 2.0 * (zt[:, :1] - zm[:, :1])              # (ngrdcol, 1)
    dzm_upper = dzm_int[:, -1:]                            # (ngrdcol, 1), copy adjacent
    dzm = jnp.concatenate([dzm_lower, dzm_int, dzm_upper], axis=1)  # (ngrdcol, nzm)

    return CLUBBGrid(
        zm=zm,
        zt=zt,
        invrs_dzm=_safe_invrs(dzm),
        invrs_dzt=_safe_invrs(dzt),
    )


__all__ = [
    "CLUBBGrid",
    "make_clubb_grid",
    "zm2zt",
    "zt2zm",
    "ddzm",
    "ddzt",
    "zm2zt2zm",
    "zt2zm2zt",
]
