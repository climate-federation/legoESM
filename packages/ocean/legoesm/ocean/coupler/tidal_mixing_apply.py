"""Apply Jayne-StLaurent tidal mixing to ocean tracers per timestep.

Operates as a SEPARATE diffusion step layered on top of the dycore's
own vertical-mixing path, so existing schemes (KPP, Richardson,
constant κ) keep their own κ_v + diffusion solver while the tidal
contribution lives in this helper.  Implicit-Euler tridiagonal
solve along each column:

    h_k · (T_k^{n+1} − T_k^n) / Δt
        = F_{k+1/2} − F_{k-1/2}

with edge fluxes ``F_{k+1/2} = K_{k+1/2} · (T_{k+1} − T_k) / Δz_{k+1/2}``,
``K_{k+1/2} = 0.5·(K_k + K_{k+1})``,
``Δz_{k+1/2} = 0.5·(h_k + h_{k+1})``.  No-flux at the surface (k=−1/2)
and at the bottom (k=nlev−1/2).

Stable for any dt because the implicit step diagonalises to a
positive-definite tridiagonal system per column.  Solved via the
Thomas algorithm vectorised over the (lat, lon) cells.

The helper runs in NumPy and accepts JAX arrays via ``np.asarray``;
the returned state is rebuilt with ``jnp.asarray`` so downstream
JIT-compiled steps see the same dtype-promoted JAX arrays they
expect.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from legoesm.core.field import Field


def _thomas_solve_columns(a, b, c, d):
    """Vectorised Thomas algorithm.

    Solves ``a_k · x_{k-1} + b_k · x_k + c_k · x_{k+1} = d_k`` for each
    column over the trailing axis ``k = 0 … n−1``.

    Sub-diagonal ``a[..., 0]`` and super-diagonal ``c[..., -1]`` are
    ignored (top + bottom boundaries).

    Parameters
    ----------
    a, b, c, d : ndarray ``(..., n)``
        Tridiagonal coefficients + RHS.  Shapes must match.

    Returns
    -------
    x : ndarray ``(..., n)``
    """
    n = a.shape[-1]
    cp = np.zeros_like(a)
    dp = np.zeros_like(a)

    cp[..., 0] = c[..., 0] / b[..., 0]
    dp[..., 0] = d[..., 0] / b[..., 0]
    for k in range(1, n):
        denom = b[..., k] - a[..., k] * cp[..., k - 1]
        denom_safe = np.where(np.abs(denom) > 1e-30, denom, 1.0)
        cp[..., k] = c[..., k] / denom_safe
        dp[..., k] = (d[..., k] - a[..., k] * dp[..., k - 1]) / denom_safe

    x = np.zeros_like(a)
    x[..., -1] = dp[..., -1]
    for k in range(n - 2, -1, -1):
        x[..., k] = dp[..., k] - cp[..., k] * x[..., k + 1]
    return x


def _build_tridiag(
    h: np.ndarray,
    K_cell: np.ndarray,
    dt: float,
):
    """Build (a, b, c) for the implicit Euler tridiagonal mixing system.

    ``h`` and ``K_cell`` are cell-centred arrays with shape
    ``(..., nlev)``.  Returns coefficients with the same shape.

    Sub-diagonal ``a_k`` is the coefficient on ``x_{k-1}`` after
    rearranging the implicit equation.  ``b`` is the diagonal,
    ``c`` is the super-diagonal.  Sign convention follows the
    standard form ``-a x_{k-1} + b x_k - c x_{k+1} = x_k^n``.
    """
    nlev = h.shape[-1]
    # K at layer interfaces: K_{k+1/2} = 0.5 (K_k + K_{k+1}) for
    # interior interfaces; zero at the top + bottom (no-flux BC).
    K_int = np.zeros(h.shape[:-1] + (nlev + 1,), dtype=np.float64)
    K_int[..., 1:-1] = 0.5 * (K_cell[..., :-1] + K_cell[..., 1:])
    # K_int[..., 0] = 0  (no surface flux)
    # K_int[..., -1] = 0 (no bottom flux)

    # Centre-to-centre spacing Δz_{k+1/2} = 0.5 (h_k + h_{k+1}).
    dz_int = np.zeros_like(K_int)
    dz_int[..., 1:-1] = 0.5 * (h[..., :-1] + h[..., 1:])
    dz_int = np.where(dz_int > 1e-12, dz_int, 1.0)

    a = np.zeros_like(h)
    c = np.zeros_like(h)

    # a_k = dt · K_{k-1/2} / (h_k · Δz_{k-1/2}) — picks K_int[..., k]
    # c_k = dt · K_{k+1/2} / (h_k · Δz_{k+1/2}) — picks K_int[..., k+1]
    h_safe = np.where(h > 1e-12, h, 1.0)
    a = dt * K_int[..., :-1] / (h_safe * dz_int[..., :-1])
    c = dt * K_int[..., 1:] / (h_safe * dz_int[..., 1:])
    # k=0 has no left flux → a_0 should already be zero because
    # K_int[..., 0] = 0; same for c_{nlev-1}.
    a[..., 0] = 0.0
    c[..., -1] = 0.0
    b = 1.0 + a + c

    # The Thomas form ``a' x_{k-1} + b' x_k + c' x_{k+1} = d``
    # expects the SIGNED coefficients of the tridiagonal matrix.
    # Rewriting our ``-a x_{k-1} + b x_k - c x_{k+1} = d``:
    return (-a, b, -c)


def apply_tidal_mixing_step(
    state,
    *,
    K_tidal: np.ndarray | jnp.ndarray,
    h_partial: np.ndarray | jnp.ndarray,
    dt: float,
) -> object:
    """Apply one timestep of tidal vertical mixing to T + S.

    GRID-AGNOSTIC. The implicit vertical diffusion is a per-column Thomas solve
    over the trailing (vertical) axis, and every operation here uses ``...`` /
    ``axis=-1`` / ``[..., None]`` broadcasting -- so it runs unchanged on a
    lat-lon C-grid state (``state.T``/``state.S`` shape ``(n_lat, n_lon, nlev)``,
    ``state.land_mask`` ``(n_lat, n_lon)``) AND on an MPAS state (``(nCells,
    nlev)`` / ``(nCells,)``). The only requirement is that ``K_tidal``,
    ``h_partial`` and the state arrays share the same ``(*spatial, nlev)`` shape.
    Land cells (``land_mask=0``) are masked back to their pre-step values.

    Parameters
    ----------
    state : LatLonCGridOceanState or an MPAS ocean state
    K_tidal : array ``(*spatial, nlev)``
        Cell-centred tidal diffusivity [m²/s] from
        :func:`legoesm.ocean.physics.vertical_mixing.tidal.compute_tidal_diffusivity`
        (itself grid-agnostic).
    h_partial : array ``(*spatial, nlev)``
        Layer thickness [m] from
        :func:`legoesm.ocean.vertical.compute_layer_thickness`.
    dt : float
        Time step [s].

    Returns
    -------
    new_state
    """
    K = np.asarray(K_tidal, dtype=np.float64)
    h = np.asarray(h_partial, dtype=np.float64)
    if K.shape != h.shape:
        raise ValueError(
            f"apply_tidal_mixing_step: K_tidal shape {K.shape} must "
            f"equal h_partial shape {h.shape}"
        )

    a, b, c = _build_tridiag(h, K, dt)

    # Preserve the state's storage dtype: the host solve runs in float64 for
    # accuracy, but returning float64 under JAX_ENABLE_X64=1 would silently
    # promote the fp32 state arrays and force a retrace of the jitted
    # ``model.step`` that consumes ``state.T``/``state.S`` downstream.
    T_dtype = state.T.data.dtype
    S_dtype = state.S.data.dtype
    T_arr = np.asarray(state.T.data, dtype=np.float64)
    S_arr = np.asarray(state.S.data, dtype=np.float64)
    T_new = _thomas_solve_columns(a, b, c, T_arr)
    S_new = _thomas_solve_columns(a, b, c, S_arr)

    # Mask out land cells (land_mask=0 → keep pre-step values).
    land_mask = np.asarray(state.land_mask.data, dtype=np.float64)
    keep = (land_mask[..., None] == 0.0)
    T_new = np.where(keep, T_arr, T_new)
    S_new = np.where(keep, S_arr, S_new)

    return state._replace(
        T=Field(
            jnp.asarray(T_new, dtype=T_dtype),
            name=state.T.name,
            dims=state.T.dims,
            units=state.T.units,
        ),
        S=Field(
            jnp.asarray(S_new, dtype=S_dtype),
            name=state.S.name,
            dims=state.S.dims,
            units=state.S.units,
        ),
    )
