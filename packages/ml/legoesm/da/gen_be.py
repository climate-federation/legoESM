"""GEN_BE background error covariance for 4D-Var.

Implements the GEN_BE approach (Bannister 2008; Barker et al. 2004):

  B^{1/2} = U_wind ∘ U_sigma ∘ U_vert ∘ U_bal ∘ U_horiz

where each operator is learned from an ensemble of background error samples:

  U_horiz  : horizontal Gaussian correlation via area-weighted Laplacian diffusion
  U_bal    : balance operator — regress psi vertical modes onto all other channels
  U_vert   : vertical EOF rotation: V diag(√λ)
  U_sigma  : pointwise standard-deviation scaling (surface pressure only)
  U_wind   : stream function/velocity potential ↔ u/v

The control variable v lives in the whitened, debalanced, vertically-decorrelated,
horizontally-uncorrelated space.  B^{1/2} v maps it to a physically-consistent
atmospheric increment.

Grid-agnostic design
--------------------
The horizontal correlation uses spherical harmonics on Gaussian grids and
neighbor-graph Laplacian diffusion on unstructured MPAS/Voronoi grids.  Generic
grids without a spectral transform or neighbor connectivity use a conservative
global-mean fallback.

Variable ordering (internal GEN_BE channel indexing)
------------------------------------------------------
  channel 0          : p_s  (surface pressure, 1 channel)
  channels 1..nlev   : psi  (streamfunction on MPAS cell-vector winds)
  channels nlev+1..  : chi  (velocity potential on MPAS cell-vector winds)
  channels 2*nlev+1..: T
  channels 3*nlev+1..: tracer_0, tracer_1, ...

References
----------
Bannister (2008), QJRMS 134, 1961-1994.
Barker et al. (2004), MWR 132, 897-914.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)


_MPAS_HELMHOLTZ_POISSON_ITER = 240
_MPAS_HELMHOLTZ_RELAX = 0.70


# ---------------------------------------------------------------------------
# Parameters (NamedTuple — pytree with JAX leaves + static Python scalars)
# ---------------------------------------------------------------------------

class GenBEParams(NamedTuple):
    """Fitted GEN_BE background error covariance parameters.

    All JAX arrays are in float64 for offline fitting accuracy; they are cast
    to the runtime dtype inside GenBETransform.

    Attributes
    ----------
    vert_eig_vec : jax.Array, shape (n_3d_groups, nlev, nlev)
        Vertical EOF eigenvectors.  Column i is the i-th eigenvector.
        Variable groups: [psi, chi, T, tracer_0, tracer_1, ...].
    vert_eig_val : jax.Array, shape (n_3d_groups, nlev)
        Vertical EOF eigenvalues (positive; covariance units).
    std_ps : jax.Array, shape ()
        Standard deviation of surface pressure [Pa].
    reg_coeff : jax.Array, shape (n_total_channels, nlev)
        Balance regression coefficients: channel i ~ sum_j reg[i,j] * psi_mode[j].
        Psi channels (indices 1..nlev) have zero coefficients (no self-regression).
    len_scale : jax.Array, shape (n_total_channels,)
        Horizontal Gaussian correlation length scale [m] per channel.
    tracer_names : tuple of str
        Names of tracer variables in channel order (not a JAX leaf).
    n_levels : int
        Number of vertical levels (static, not a JAX leaf).
    wind_transform : str
        Wind control transform. ``"mpas_helmholtz"`` means the first two
        3D groups are streamfunction/velocity-potential and are converted to
        cell-centered u/v at runtime. ``"identity"`` means those groups are
        already physical u/v and no Helmholtz conversion is applied.
    """
    vert_eig_vec: jax.Array   # (n_3d_groups, nlev, nlev)
    vert_eig_val: jax.Array   # (n_3d_groups, nlev)
    std_ps: jax.Array         # ()
    reg_coeff: jax.Array      # (n_total_channels, nlev)
    len_scale: jax.Array      # (n_total_channels,) in metres
    tracer_names: tuple        # tuple of str — not a JAX leaf
    n_levels: int              # static
    wind_transform: str = "mpas_helmholtz"  # static


# ---------------------------------------------------------------------------
# Offline fitting helpers
# ---------------------------------------------------------------------------

def _build_vert_cov(errors_ncol_nlev: np.ndarray, corr_length: float) -> tuple:
    """Fit vertical covariance model and return (eig_vec, eig_val).

    The covariance model is:
      C[i, j] = sqrt(var[i] * var[j]) * exp(-|i-j| / corr_length)

    This ensures the eigenvalues absorb the per-level variances, so no
    separate standard-deviation scaling is needed for 3D variables after
    the EOF expansion.

    Parameters
    ----------
    errors_ncol_nlev : np.ndarray, shape (n_ens, ncol, nlev)
    corr_length : float
        Exponential vertical correlation length in levels.

    Returns
    -------
    eig_vec : np.ndarray, shape (nlev, nlev)
        Eigenvectors as columns, sorted descending by eigenvalue.
    eig_val : np.ndarray, shape (nlev,)
        Eigenvalues, sorted descending.
    """
    n_ens, ncol, nlev = errors_ncol_nlev.shape
    # Per-level variance averaged over ensemble and spatial points
    var_lev = np.var(errors_ncol_nlev, axis=(0, 1))  # (nlev,)

    idx = np.arange(nlev, dtype=float)
    # Symmetric exponential covariance matrix
    cov = np.sqrt(var_lev[:, None] * var_lev[None, :]) * np.exp(
        -np.abs(idx[:, None] - idx[None, :]) / corr_length
    )  # (nlev, nlev)

    # eigh is numerically stable for symmetric matrices; eigenvalues are real
    eig_val, eig_vec = np.linalg.eigh(cov)  # ascending order
    order = np.argsort(eig_val)[::-1]        # descending
    return eig_vec[:, order], eig_val[order]


def _to_eof_modes(
    errors_ncol_nlev: np.ndarray,
    eig_vec: np.ndarray,
    eig_val: np.ndarray,
) -> np.ndarray:
    """Transform (n_ens, ncol, nlev) errors into (n_ens*ncol, nlev) EOF modes.

    Forward inverse: z_mode = x @ V @ diag(1/sqrt(λ))

    Parameters
    ----------
    errors_ncol_nlev : np.ndarray, shape (n_ens, ncol, nlev)
    eig_vec : (nlev, nlev), eigenvectors as columns
    eig_val : (nlev,)

    Returns
    -------
    modes : np.ndarray, shape (n_ens * ncol, nlev)
    """
    n_ens, ncol, nlev = errors_ncol_nlev.shape
    flat = errors_ncol_nlev.reshape(n_ens * ncol, nlev)  # (N, nlev)
    sqrt_lam = np.sqrt(np.maximum(eig_val, 1e-30))
    # z_mode = flat @ V / sqrt(λ)  (divide each column of V by corresponding sqrt_λ)
    return flat @ (eig_vec / sqrt_lam[None, :])  # (N, nlev)


def _fit_balance(
    err_psi_modes: np.ndarray,
    err_all_modes: np.ndarray,
    psi_start: int,
    psi_end: int,
) -> np.ndarray:
    """Fit linear balance regression: channel_i ≈ sum_j reg[i,j] * psi_mode[j].

    Parameters
    ----------
    err_psi_modes : np.ndarray, shape (N, nlev)
        Psi EOF modes (psi group only), N = n_ens * ncol.
    err_all_modes : np.ndarray, shape (N, n_total_channels)
        All channels flattened: surface + 3D groups in EOF mode space.
    psi_start, psi_end : int
        Slice into all_modes where psi modes live.

    Returns
    -------
    reg_coeff : np.ndarray, shape (n_total_channels, nlev)
        Psi modes have reg_coeff[psi_start:psi_end] = 0.
    """
    N, nlev = err_psi_modes.shape

    # Auto-covariance of psi modes
    cov_psi = (err_psi_modes.T @ err_psi_modes) / N  # (nlev, nlev)
    # Regularise to avoid singular matrix (small ridge)
    ridge = 1e-6 * np.trace(cov_psi) / max(nlev, 1)
    cov_psi_inv = np.linalg.inv(cov_psi + ridge * np.eye(nlev))

    # Vectorised cross-covariance: (n_ch, nlev) = (n_ch, N) @ (N, nlev) / N
    cross_cov = (err_all_modes.T @ err_psi_modes) / N  # (n_ch, nlev)
    reg_coeff = cross_cov @ cov_psi_inv                 # (n_ch, nlev)

    # Psi channels must not regress against themselves (they are the source)
    reg_coeff[psi_start:psi_end] = 0.0

    return reg_coeff


def _fit_len_scale(
    err_static: np.ndarray,
    grid,
    default_m: float = 500_000.0,
) -> np.ndarray:
    """Estimate horizontal Gaussian correlation length scale per channel.

    Uses the Laplacian variance ratio method (Bannister 2008 eq. 2.3):
      L ≈ (8 * var(f) / var(∇²f))^{1/4}

    For GaussianGrid the spherical Laplacian is computed exactly via SH
    analysis (matching the original numpy GEN_BE implementation).
    For MPAS/Voronoi grids the Laplacian is approximated with the finite-volume
    cell-centred operator
    ``areaCell_i^-1 * sum_edges (dvEdge / dcEdge) * (f_neighbor - f_i)``,
    using the same geometry as the runtime diffusion smoother.  Generic grids
    without neighbor connectivity fall back to the default length scale instead
    of fabricating a grid-scale estimate from a global-mean relaxation.

    Parameters
    ----------
    err_static : np.ndarray, shape (n_ens, ncol, n_channels)
        Debalanced, vertically-decorrelated errors.
    grid : GridProtocol
    default_m : float
        Fallback length scale [m] when variance is negligible.

    Returns
    -------
    len_scale : np.ndarray, shape (n_channels,)
        Estimated length scales in metres.
    """
    from legoesm.grids.gaussian import GaussianGrid

    n_ens, ncol, n_ch = err_static.shape
    len_scale = np.full(n_ch, default_m)

    if isinstance(grid, GaussianGrid):
        from legoesm.grids.gaussian import sh_analysis, sh_synthesis
        lap_factors = np.array(grid.lap)  # (n_sh,) = -n(n+1)/a^2
        n_lat, n_lon = grid.n_lat, grid.n_lon
        a = float(grid.radius)

        for ch in range(n_ch):
            var_f_list, var_lap_list = [], []
            for e in range(n_ens):
                f_grid = err_static[e, :, ch].reshape(n_lat, n_lon)
                var_f_list.append(float(np.var(f_grid)))
                f_spec = np.array(sh_analysis(grid, jnp.array(f_grid, dtype=jnp.float64)))
                lap_grid = np.array(sh_synthesis(grid, jnp.array(f_spec * lap_factors)))
                var_lap_list.append(float(np.var(lap_grid)))

            var_f = float(np.mean(var_f_list))
            if var_f < 1e-30:
                continue
            var_lap = float(np.mean(var_lap_list))
            if var_lap < 1e-30:
                len_scale[ch] = a * np.pi  # very smooth: half circumference
            else:
                L = (8.0 * var_f / var_lap) ** 0.25
                len_scale[ch] = float(np.clip(L, a * 0.001, a * np.pi))
    else:
        dx = float(grid.grid_radius) * np.sqrt(4 * np.pi / ncol)

        has_mpas_laplacian = all(
            hasattr(grid, name)
            for name in (
                "cellsOnCell",
                "edgesOnCell",
                "nEdgesOnCell",
                "areaCell",
                "dcEdge",
                "dvEdge",
            )
        )

        if has_mpas_laplacian:
            neighbors = np.asarray(grid.cellsOnCell, dtype=np.int64)
            edges = np.asarray(grid.edgesOnCell, dtype=np.int64)
            n_edges = np.asarray(grid.nEdgesOnCell, dtype=np.int64)
            area = np.asarray(grid.areaCell, dtype=np.float64)
            dc_edge = np.asarray(grid.dcEdge, dtype=np.float64)
            dv_edge = np.asarray(grid.dvEdge, dtype=np.float64)
            if neighbors.ndim != 2 or neighbors.shape[1] != ncol:
                raise ValueError(
                    "grid.cellsOnCell must have shape (maxEdges, ncol) for "
                    "GEN_BE neighbor Laplacian length-scale fitting."
                )
            if edges.shape != neighbors.shape:
                raise ValueError(
                    "grid.edgesOnCell must match cellsOnCell shape for GEN_BE "
                    "MPAS Laplacian length-scale fitting."
                )
            if n_edges.shape != (ncol,):
                raise ValueError(
                    "grid.nEdgesOnCell must have shape (ncol,) for GEN_BE "
                    "neighbor Laplacian length-scale fitting."
                )
            mask = np.arange(neighbors.shape[0])[:, None] < n_edges[None, :]
            neighbors = np.clip(neighbors, 0, ncol - 1)
            edges = np.clip(edges, 0, dc_edge.shape[0] - 1)
            edge_weights = (
                dv_edge[edges]
                / np.maximum(dc_edge[edges], 1.0)
                / np.maximum(area[None, :], 1.0)
            )
            edge_weights = np.where(mask, edge_weights, 0.0)

            for ch in range(n_ch):
                f = err_static[:, :, ch]  # (n_ens, ncol)
                var_f = float(np.var(f))
                if var_f < 1e-30:
                    continue
                neighbor_vals = f[:, neighbors]  # (n_ens, maxEdges, ncol)
                lap_f = np.sum(
                    edge_weights[None, :, :] * (neighbor_vals - f[:, None, :]),
                    axis=1,
                )
                var_lap = float(np.var(lap_f))
                if var_lap < 1e-30:
                    len_scale[ch] = min(dx * 20.0, 1e7)
                else:
                    L = (8.0 * var_f / var_lap) ** 0.25
                    len_scale[ch] = float(np.clip(L, dx, 1e7))
        else:
            logger.warning(
                "GEN_BE length-scale fitting grid %s has no spectral transform "
                "or cell-neighbor connectivity; using default %.0f km for all "
                "channels.",
                type(grid).__name__,
                default_m / 1000.0,
            )

    return len_scale


def _has_mpas_cell_vector_wind(grid) -> bool:
    """Return True when grid has enough MPAS cell geometry for wind Helmholtz."""
    return all(
        hasattr(grid, name)
        for name in (
            "cellsOnCell",
            "nEdgesOnCell",
            "latCell",
            "lonCell",
            "grid_radius",
        )
    )


def _mpas_lsq_gradient_weights_np(grid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Least-squares cell-neighbor gradient weights on an MPAS/Voronoi mesh.

    The saved MPAS NMC wind samples used by this workflow are cell-centered
    east/north winds, not native TRiSK edge-normal winds.  These weights provide
    a local tangent-plane gradient operator for cell-centered scalars:

      grad_x(f)_i = sum_n wx[n,i] * (f_neighbor - f_i)
      grad_y(f)_i = sum_n wy[n,i] * (f_neighbor - f_i)
    """
    neighbors_raw = np.asarray(grid.cellsOnCell, dtype=np.int64)
    n_edges = np.asarray(grid.nEdgesOnCell, dtype=np.int64)
    ncol = int(grid.grid_n_columns)
    if neighbors_raw.ndim != 2 or neighbors_raw.shape[1] != ncol:
        raise ValueError("MPAS cellsOnCell must have shape (maxEdges, nCells)")

    max_edges = neighbors_raw.shape[0]
    mask = np.arange(max_edges)[:, None] < n_edges[None, :]
    neighbors = np.clip(neighbors_raw, 0, ncol - 1)

    lat = np.asarray(grid.latCell, dtype=np.float64)
    lon = np.asarray(grid.lonCell, dtype=np.float64)
    radius = float(grid.grid_radius)

    dlon = lon[neighbors] - lon[None, :]
    dlon = (dlon + np.pi) % (2.0 * np.pi) - np.pi
    dlat = lat[neighbors] - lat[None, :]
    dx = radius * np.cos(lat)[None, :] * dlon
    dy = radius * dlat
    dx = np.where(mask, dx, 0.0)
    dy = np.where(mask, dy, 0.0)

    sxx = np.sum(dx * dx, axis=0)
    sxy = np.sum(dx * dy, axis=0)
    syy = np.sum(dy * dy, axis=0)
    det = np.maximum(sxx * syy - sxy * sxy, 1.0)

    # [gx, gy] = inv(A^T A) A^T df.
    wx = (syy[None, :] * dx - sxy[None, :] * dy) / det[None, :]
    wy = (-sxy[None, :] * dx + sxx[None, :] * dy) / det[None, :]
    wx = np.where(mask, wx, 0.0)
    wy = np.where(mask, wy, 0.0)
    return neighbors, wx, wy


def _mpas_grad_np(
    field: np.ndarray,
    neighbors: np.ndarray,
    wx: np.ndarray,
    wy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply MPAS LSQ gradient weights to (..., nCells) scalar fields."""
    neighbor_vals = field[..., neighbors]  # (..., maxEdges, nCells)
    diff = neighbor_vals - field[..., None, :]
    gx = np.sum(wx[None, :, :] * diff, axis=-2)
    gy = np.sum(wy[None, :, :] * diff, axis=-2)
    return gx, gy


def _mpas_laplace_np(
    field: np.ndarray,
    neighbors: np.ndarray,
    wx: np.ndarray,
    wy: np.ndarray,
) -> np.ndarray:
    """Cell-neighbor Laplacian proxy: div(grad(field))."""
    gx, gy = _mpas_grad_np(field, neighbors, wx, wy)
    gxx, _ = _mpas_grad_np(gx, neighbors, wx, wy)
    _, gyy = _mpas_grad_np(gy, neighbors, wx, wy)
    return gxx + gyy


def _mpas_inverse_laplace_np(
    rhs: np.ndarray,
    neighbors: np.ndarray,
    wx: np.ndarray,
    wy: np.ndarray,
    n_iter: int = _MPAS_HELMHOLTZ_POISSON_ITER,
) -> np.ndarray:
    """Approximate mean-zero inverse Laplacian with fixed Richardson steps."""
    rhs = rhs - rhs.mean(axis=-1, keepdims=True)
    phi = np.zeros_like(rhs)
    radius2 = 1.0 / np.maximum(np.mean(wx * wx + wy * wy), 1e-20)
    step = _MPAS_HELMHOLTZ_RELAX * radius2
    for _ in range(n_iter):
        residual = rhs - _mpas_laplace_np(phi, neighbors, wx, wy)
        phi = phi - step * residual
        phi = phi - phi.mean(axis=-1, keepdims=True)
    return phi


def _mpas_cell_wind_to_helmholtz_np(
    u_cell: np.ndarray,
    v_cell: np.ndarray,
    grid,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert MPAS cell-centered east/north wind to streamfunction/potential.

    Returns ``psi, chi`` with shape ``(n_ens, nCells, nlev)``.  Internally the
    batched solve uses local tangent-plane div/vorticity and a mean-zero fixed
    inverse-Laplacian approximation.  This is the MPAS analogue of AI-VarDA's
    regular-grid ``uv2sfvp`` preprocessing for the current cell-vector sample
    files.
    """
    neighbors, wx, wy = _mpas_lsq_gradient_weights_np(grid)
    u_e_l_c = np.moveaxis(u_cell, 2, 1)  # (n_ens, nlev, nCells)
    v_e_l_c = np.moveaxis(v_cell, 2, 1)

    du_dx, du_dy = _mpas_grad_np(u_e_l_c, neighbors, wx, wy)
    dv_dx, dv_dy = _mpas_grad_np(v_e_l_c, neighbors, wx, wy)
    div = du_dx + dv_dy
    vort = dv_dx - du_dy

    chi = _mpas_inverse_laplace_np(div, neighbors, wx, wy)
    psi = _mpas_inverse_laplace_np(vort, neighbors, wx, wy)
    return np.moveaxis(psi, 1, 2), np.moveaxis(chi, 1, 2)


# ---------------------------------------------------------------------------
# Main fitting function
# ---------------------------------------------------------------------------

def fit_gen_be(
    ensemble_errors: Sequence,
    grid,
    sigma_coord=None,
    vert_corr_length: float = 2.0,
    default_len_scale_km: float = 500.0,
) -> GenBEParams:
    """Fit GEN_BE background error covariance parameters from ensemble errors.

    Parameters
    ----------
    ensemble_errors : sequence of HydrostaticState
        Background error samples (ensemble forecasts minus truth / analysis).
        Each state must have fields: u, T, p_s, and optionally v, tracers.
        On MPAS/Voronoi grids with cell-centered u/v samples, winds are
        transformed to streamfunction/velocity-potential control variables
        before fitting, analogous to AI-VarDA's uv2sfvp preprocessing.
    grid : GridProtocol
        Model grid.  Must support to_columns(), grid_n_columns, grid_area,
        grid_radius.
    sigma_coord : VerticalCoordProtocol, optional
        Vertical coordinate (currently unused; reserved for pressure weighting).
    vert_corr_length : float
        Exponential vertical correlation length in model levels.
    default_len_scale_km : float
        Fallback horizontal length scale [km] when variance is too small.

    Returns
    -------
    GenBEParams
    """
    if len(ensemble_errors) < 2:
        raise ValueError("Need at least 2 ensemble members to fit a covariance.")

    # --- Extract arrays from states ---
    def _to_np(field):
        """Extract numpy array from Field or plain array."""
        if hasattr(field, "data"):
            return np.array(field.data)
        return np.array(field)

    def _to_cols_np(arr):
        """Flatten spatial dimensions to columns using grid.to_columns."""
        return np.array(grid.to_columns(arr))

    # Collect per-variable errors as list of (ncol, nlev) or (ncol,) arrays
    sample_state = ensemble_errors[0]
    has_v = getattr(sample_state, "v", None) is not None
    tracers_dict = getattr(sample_state, "tracers", None) or {}
    tracer_names = tuple(sorted(tracers_dict.keys()))
    n_tracers = len(tracer_names)

    err_u_list, err_v_list, err_T_list, err_ps_list = [], [], [], []
    err_tracer_lists = [[] for _ in range(n_tracers)]

    for state in ensemble_errors:
        err_u_list.append(_to_cols_np(_to_np(state.u)))
        err_T_list.append(_to_cols_np(_to_np(state.T)))
        err_ps_list.append(_to_cols_np(_to_np(state.p_s)))
        if has_v:
            err_v_list.append(_to_cols_np(_to_np(state.v)))
        if state.tracers:
            for k, name in enumerate(tracer_names):
                err_tracer_lists[k].append(_to_cols_np(_to_np(state.tracers[name])))

    # Stack → (n_ens, ncol, nlev) for 3D, (n_ens, ncol) for p_s
    n_ens = len(ensemble_errors)
    ncol = grid.grid_n_columns

    err_u  = np.stack(err_u_list, axis=0)   # (n_ens, ncol, nlev)
    err_T  = np.stack(err_T_list, axis=0)
    err_ps = np.stack(err_ps_list, axis=0)  # (n_ens, ncol)
    if has_v:
        err_v = np.stack(err_v_list, axis=0)
    else:
        err_v = np.zeros_like(err_u)
    err_tracers = [np.stack(lst, axis=0) for lst in err_tracer_lists]  # each (n_ens, ncol, nlev)

    # Subtract ensemble mean from each
    for arr in [err_u, err_v, err_T, err_ps] + err_tracers:
        arr -= arr.mean(axis=0, keepdims=True)

    nlev = err_u.shape[2]
    use_mpas_helmholtz = (
        has_v
        and _has_mpas_cell_vector_wind(grid)
        and err_u.shape[1] == ncol
        and err_v.shape[1] == ncol
    )
    if use_mpas_helmholtz:
        logger.info("fit_gen_be: converting MPAS cell u/v errors to psi/chi")
        err_psi, err_chi = _mpas_cell_wind_to_helmholtz_np(err_u, err_v, grid)
    else:
        err_psi, err_chi = err_u, err_v

    n_3d_groups = 2 + 1 + n_tracers           # psi, chi, T, tracers
    n_sur_channels = 1                          # p_s
    n_total_channels = n_sur_channels + n_3d_groups * nlev
    psi_start = n_sur_channels                  # = 1
    psi_end   = psi_start + nlev                # = 1 + nlev

    logger.info(
        "fit_gen_be: n_ens=%d, ncol=%d, nlev=%d, n_tracers=%d, "
        "n_total_channels=%d",
        n_ens, ncol, nlev, n_tracers, n_total_channels,
    )

    # --- 1. Vertical covariance per 3D group ---
    groups_raw = [err_psi, err_chi, err_T] + err_tracers  # (n_3d_groups,) × (n_ens, ncol, nlev)
    eig_vecs, eig_vals = [], []
    for k, grp in enumerate(groups_raw):
        ev, el = _build_vert_cov(grp, vert_corr_length)
        eig_vecs.append(ev)
        eig_vals.append(el)
    eig_vec_arr = np.stack(eig_vecs, axis=0)  # (n_3d_groups, nlev, nlev)
    eig_val_arr = np.stack(eig_vals, axis=0)  # (n_3d_groups, nlev)

    # --- 2. Surface pressure standard deviation ---
    std_ps = float(np.sqrt(np.mean(np.var(err_ps, axis=0))))

    # --- 3. Project all errors into vertical EOF mode space ---
    # Shape: (n_ens*ncol, nlev) for each 3D group
    N = n_ens * ncol
    modes_groups = []
    for k, grp in enumerate(groups_raw):
        modes_groups.append(_to_eof_modes(grp, eig_vecs[k], eig_vals[k]))  # (N, nlev)

    # ps modes: just scale by 1/std_ps
    modes_ps = (err_ps.reshape(N, 1) / (std_ps + 1e-30))  # (N, 1)

    # Assemble all_modes in channel order: [ps(1), psi(nlev), chi(nlev), T(nlev), ...]
    all_modes = np.concatenate(
        [modes_ps] + modes_groups, axis=1
    )  # (N, n_total_channels)
    # Note: modes_ps has shape (N, 1), modes_groups[k] has shape (N, nlev)
    # → total = 1 + n_3d_groups*nlev = n_total_channels ✓

    # --- 4. Balance regression: psi modes → all channels ---
    err_psi_modes = all_modes[:, psi_start:psi_end]  # (N, nlev)
    reg_coeff = _fit_balance(err_psi_modes, all_modes, psi_start, psi_end)
    if n_tracers:
        tracer_start = n_sur_channels + 3 * nlev
        tracer_end = tracer_start + n_tracers * nlev
        # Moisture is treated as an unbalanced control variable by default.
        # A psi->moisture balance can create large, nonphysical q_v increments
        # from mass/wind observations unless it is explicitly designed/tuned.
        reg_coeff[tracer_start:tracer_end] = 0.0
    # reg_coeff: (n_total_channels, nlev)

    # --- 5. Debalance: subtract psi contribution from all channels ---
    balance_contrib = err_psi_modes @ reg_coeff.T   # (N, n_total_channels)
    all_modes_static = all_modes - balance_contrib

    # Reshape to (n_ens, ncol, n_total_channels) for len_scale fitting
    err_static_3d = all_modes_static.reshape(n_ens, ncol, n_total_channels)

    # --- 6. Horizontal length scales ---
    len_scale = _fit_len_scale(
        err_static_3d, grid, default_m=default_len_scale_km * 1000.0
    )

    logger.info(
        "fit_gen_be: ps std=%.2f Pa, len_scale min=%.0f km, max=%.0f km",
        std_ps, len_scale.min() / 1000, len_scale.max() / 1000,
    )

    return GenBEParams(
        vert_eig_vec=jnp.array(eig_vec_arr),
        vert_eig_val=jnp.array(eig_val_arr),
        std_ps=jnp.array(std_ps),
        reg_coeff=jnp.array(reg_coeff),
        len_scale=jnp.array(len_scale),
        tracer_names=tracer_names,
        n_levels=nlev,
        wind_transform="mpas_helmholtz" if use_mpas_helmholtz else "identity",
    )


# ---------------------------------------------------------------------------
# Runtime B matrix operator
# ---------------------------------------------------------------------------

class GenBETransform:
    """Runtime B = U U^T background error covariance using GEN_BE.

    Implements the interface expected by build_cost_fn:
      - sqrt_multiply(v)  →  B^{1/2} v
      - inv_multiply(x)   →  B^{-1} x  (exact via jax.vjp)

    Parameters
    ----------
    params : GenBEParams
    spec : ControlVectorSpec
        Control vector specification.  Must contain entries for "u", "T",
        "p_s".  "v" and "tracers.*" are included when present.
    grid : GridProtocol
    n_diffusion_iter : int
        Number of Laplacian diffusion iterations for horizontal correlation.
        More iterations → smoother kernel, closer to Gaussian.
    """

    def __init__(self, params: GenBEParams, spec, grid, n_diffusion_iter: int = 20):
        self.params = params
        self.spec = spec
        self.grid = grid
        self.n_iter = n_diffusion_iter

        nlev = params.n_levels
        ncol = grid.grid_n_columns
        n_3d_groups = 2 + 1 + len(params.tracer_names)
        n_total_ch = 1 + n_3d_groups * nlev

        self._nlev = nlev
        self._ncol = ncol
        self._n_3d_groups = n_3d_groups
        self._n_total_ch = n_total_ch
        self._psi_start = 1         # channel index where psi starts
        self._psi_end   = 1 + nlev  # exclusive

        # Build variable slices from ControlVectorSpec before operator setup.
        self._slices = {}  # name → (offset, size, shape)
        for entry in spec.entries:
            self._slices[entry.field_name] = (entry.offset, entry.size, entry.shape)

        # Precompute per-channel kappas from length scales
        # kappa ≈ L^2 / (2 * n_iter * dx^2); clipped to [0, 0.5]
        dx = grid.grid_radius * jnp.sqrt(4.0 * jnp.pi / ncol)
        raw_kappa = (params.len_scale / dx) ** 2 / (2.0 * n_diffusion_iter)
        self._kappa = jnp.clip(raw_kappa, 0.0, 0.49)  # (n_total_ch,)

        # Area weights for diffusion fallback (precomputed, shape (ncol,))
        area = grid.to_columns(grid.grid_area)
        self._area_norm = jnp.array(area) / jnp.sum(jnp.array(area))
        has_mpas_laplacian = all(
            hasattr(grid, name)
            for name in (
                "cellsOnCell",
                "edgesOnCell",
                "nEdgesOnCell",
                "areaCell",
                "dcEdge",
                "dvEdge",
            )
        )
        if has_mpas_laplacian:
            neighbors = jnp.asarray(grid.cellsOnCell, dtype=jnp.int32)
            self._neighbors = jnp.clip(neighbors, 0, ncol - 1)
            edges = jnp.asarray(grid.edgesOnCell, dtype=jnp.int32)
            edges = jnp.clip(edges, 0, grid.nEdges - 1)
            edge_index = jnp.arange(neighbors.shape[0], dtype=jnp.int32)[:, None]
            n_edges = jnp.asarray(grid.nEdgesOnCell, dtype=jnp.int32)[None, :]
            self._neighbor_mask = edge_index < n_edges
            area = jnp.asarray(grid.areaCell)
            dc_edge = jnp.asarray(grid.dcEdge)
            dv_edge = jnp.asarray(grid.dvEdge)
            edge_weights = (
                dv_edge[edges]
                / jnp.maximum(dc_edge[edges], 1.0)
                / jnp.maximum(area[None, :], 1.0)
            )
            self._laplacian_weights = jnp.where(self._neighbor_mask, edge_weights, 0.0)
            max_diag = jnp.max(jnp.sum(self._laplacian_weights, axis=0))
            raw_step_m2 = (params.len_scale ** 2) / (2.0 * n_diffusion_iter)
            self._diffusion_step_m2 = jnp.clip(
                raw_step_m2,
                0.0,
                0.49 / jnp.maximum(max_diag, 1e-30),
            )
        else:
            self._neighbors = None
            self._neighbor_mask = None
            self._laplacian_weights = None
            self._diffusion_step_m2 = None

        self._use_mpas_helmholtz = (
            getattr(params, "wind_transform", "mpas_helmholtz") == "mpas_helmholtz"
            and _has_mpas_cell_vector_wind(grid)
            and "u" in self._slices
            and "v" in self._slices
            and self._slices["u"][2][0] == ncol
            and self._slices["v"][2][0] == ncol
        )
        if self._use_mpas_helmholtz:
            neighbors_np, wx_np, wy_np = _mpas_lsq_gradient_weights_np(grid)
            self._helm_neighbors = jnp.asarray(neighbors_np, dtype=jnp.int32)
            self._helm_wx = jnp.asarray(wx_np)
            self._helm_wy = jnp.asarray(wy_np)
            radius2 = 1.0 / jnp.maximum(
                jnp.mean(self._helm_wx ** 2 + self._helm_wy ** 2),
                1e-20,
            )
            self._helm_poisson_step = _MPAS_HELMHOLTZ_RELAX * radius2
        else:
            self._helm_neighbors = None
            self._helm_wx = None
            self._helm_wy = None
            self._helm_poisson_step = None

        # Precompute SH Gaussian kernels for GaussianGrid
        # kernel[k, ch] = exp(grid.lap[k] * L[ch]^2 / 2)
        # grid.lap = -n(n+1)/a^2, so this decays with total wavenumber (smoothing).
        from legoesm.grids.gaussian import GaussianGrid
        if isinstance(grid, GaussianGrid):
            lap = jnp.array(grid.lap)  # (n_sh,)
            self._sh_kernels = jnp.exp(
                lap[:, None] * (params.len_scale[None, :] ** 2) / 2.0
            )  # (n_sh, n_total_ch), real
        else:
            self._sh_kernels = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_cols(self, flat_vec: jax.Array, name: str) -> jax.Array:
        """Extract and reshape a named field from the flat control vector.

        Returns shape (ncol, nlev) for 3D fields or (ncol,) for surface.
        Delegates to grid.to_columns so cubed-sphere / lat-lon / MPAS layouts
        are all handled correctly.
        """
        off, sz, shape = self._slices[name]
        arr = jax.lax.dynamic_slice(flat_vec, (off,), (sz,)).reshape(shape)
        return self.grid.to_columns(arr)

    def _from_cols(self, arr_col: jax.Array, name: str) -> jax.Array:
        """Reshape columnar array back to the shape expected by the control vector."""
        return self.grid.from_columns(arr_col)

    def _pack_to_flat(
        self,
        psi_col: jax.Array,   # (ncol, nlev)
        chi_col: jax.Array,   # (ncol, nlev)
        T_col: jax.Array,     # (ncol, nlev)
        ps_col: jax.Array,    # (ncol,)
        tracer_cols: list,    # list of (ncol, nlev)
    ) -> jax.Array:
        """Repack per-variable arrays into the flat control vector order."""
        parts = []
        for entry in self.spec.entries:
            name = entry.field_name
            if name == "u":
                parts.append(self._from_cols(psi_col, name).ravel())
            elif name == "v":
                parts.append(self._from_cols(chi_col, name).ravel())
            elif name == "T":
                parts.append(self._from_cols(T_col, name).ravel())
            elif name == "p_s":
                parts.append(self._from_cols(ps_col, name).ravel())
            elif name.startswith("tracers."):
                tname = name.split(".", 1)[1]
                k = self.params.tracer_names.index(tname)
                parts.append(self._from_cols(tracer_cols[k], name).ravel())
            else:
                # Field present in spec but not handled by GEN_BE — zero increment
                _, sz, _ = self._slices[name]
                parts.append(jnp.zeros(sz, dtype=psi_col.dtype))
        return jnp.concatenate(parts)

    def _mpas_grad(self, field_col: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Cell-centered MPAS LSQ gradient for (nCells, nlev) fields."""
        neighbors = self._helm_neighbors
        wx = self._helm_wx[..., None]
        wy = self._helm_wy[..., None]
        neighbor_vals = field_col[neighbors, :]
        diff = neighbor_vals - field_col[None, :, :]
        gx = jnp.sum(wx * diff, axis=0)
        gy = jnp.sum(wy * diff, axis=0)
        return gx, gy

    def _mpas_laplace(self, field_col: jax.Array) -> jax.Array:
        """Cell-centered MPAS LSQ Laplacian for (nCells, nlev) fields."""
        gx, gy = self._mpas_grad(field_col)
        gxx, _ = self._mpas_grad(gx)
        _, gyy = self._mpas_grad(gy)
        return gxx + gyy

    def _mpas_inverse_laplace(self, rhs_col: jax.Array) -> jax.Array:
        """Approximate mean-zero inverse Laplacian for (nCells, nlev) RHS."""
        rhs = rhs_col - jnp.mean(rhs_col, axis=0, keepdims=True)
        step_size = self._helm_poisson_step

        def step(phi, _):
            residual = rhs - self._mpas_laplace(phi)
            phi = phi - step_size * residual
            phi = phi - jnp.mean(phi, axis=0, keepdims=True)
            return phi, None

        result, _ = jax.lax.scan(
            step,
            jnp.zeros_like(rhs),
            None,
            length=_MPAS_HELMHOLTZ_POISSON_ITER,
        )
        return result

    def _wind_to_psi_chi(
        self,
        u_col: jax.Array,
        v_col: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Runtime U_wind^{-1}: physical u/v -> psi/chi."""
        if not self._use_mpas_helmholtz:
            return u_col, v_col
        du_dx, du_dy = self._mpas_grad(u_col)
        dv_dx, dv_dy = self._mpas_grad(v_col)
        div = du_dx + dv_dy
        vort = dv_dx - du_dy
        chi_col = self._mpas_inverse_laplace(div)
        psi_col = self._mpas_inverse_laplace(vort)
        return psi_col, chi_col

    def _psi_chi_to_wind(
        self,
        psi_col: jax.Array,
        chi_col: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Runtime U_wind: psi/chi -> physical cell-centered east/north u/v."""
        if not self._use_mpas_helmholtz:
            return psi_col, chi_col
        dpsi_dx, dpsi_dy = self._mpas_grad(psi_col)
        dchi_dx, dchi_dy = self._mpas_grad(chi_col)
        u_col = dchi_dx - dpsi_dy
        v_col = dchi_dy + dpsi_dx
        return u_col, v_col

    def _horiz_smooth(self, all_ch: jax.Array, inverse: bool = False) -> jax.Array:
        """Apply per-channel horizontal Gaussian smoothing.

        For GaussianGrid: SH Gaussian convolution (matching the original numpy
        GEN_BE implementation).  For other grids: area-weighted Laplacian
        diffusion fallback.

        Parameters
        ----------
        all_ch : jax.Array, shape (ncol, n_total_ch)
        inverse : bool
            If True, apply the inverse (deconvolution / reverse diffusion).

        Returns
        -------
        jax.Array, shape (ncol, n_total_ch)
        """
        if self._sh_kernels is not None:
            return self._sh_smooth(all_ch, inverse)
        return self._diffusion_smooth(all_ch, inverse)

    def _sh_smooth(self, all_ch: jax.Array, inverse: bool = False) -> jax.Array:
        """SH Gaussian convolution for GaussianGrid."""
        from legoesm.grids.gaussian import sh_analysis_3d, sh_synthesis_3d
        grid = self.grid
        field_3d = grid.from_columns(all_ch)           # (n_lat, n_lon, n_ch)
        coeffs = sh_analysis_3d(grid, field_3d)         # (n_sh, n_ch)
        kernels = self._sh_kernels                      # (n_sh, n_ch)
        if inverse:
            kernels = 1.0 / jnp.maximum(kernels, 1e-30)
        out_3d = sh_synthesis_3d(grid, coeffs * kernels)  # (n_lat, n_lon, n_ch)
        return grid.to_columns(out_3d)                  # (ncol, n_ch)

    def _diffusion_smooth(self, all_ch: jax.Array, inverse: bool = False) -> jax.Array:
        """Grid-neighbor Laplacian diffusion for non-Gaussian grids.

        MPAS/Voronoi grids expose cell/edge geometry.  Use the finite-volume
        cell-centred Laplacian
        areaCell_i^-1 * sum_edges (dvEdge / dcEdge) * (x_neighbor - x_i).
        Generic grids without geometry fall back to a weak global mean
        relaxation so legacy behavior remains available.
        """
        kappa = self._kappa          # (n_total_ch,)
        sign = -1.0 if inverse else 1.0
        if (
            self._neighbors is not None
            and self._neighbor_mask is not None
            and self._laplacian_weights is not None
            and self._diffusion_step_m2 is not None
        ):
            neighbors = self._neighbors
            weights = self._laplacian_weights[..., None]
            step_m2 = self._diffusion_step_m2

            def step(x, _):
                neighbor_vals = x[neighbors, :]
                lap = jnp.sum(weights * (neighbor_vals - x[None, :, :]), axis=0)
                return x + sign * step_m2[None, :] * lap, None

            result, _ = jax.lax.scan(step, all_ch, None, length=self.n_iter)
            return result

        area_norm = self._area_norm  # (ncol,)

        def smooth_one_channel(channel_col, kap):
            """Apply n_iter diffusion steps to a single (ncol,) column."""
            def step(x, _):
                # Area-weighted global mean
                mean_x = jnp.dot(x, area_norm)
                return x + sign * kap * (mean_x - x), None

            result, _ = jax.lax.scan(step, channel_col, None, length=self.n_iter)
            return result

        # vmap over channels: each channel gets its own kappa
        smooth_vmapped = jax.vmap(smooth_one_channel, in_axes=(1, 0), out_axes=1)
        return smooth_vmapped(all_ch, kappa)

    def _assemble_channels(
        self,
        ps_col: jax.Array,       # (ncol,)
        psi_col: jax.Array,      # (ncol, nlev)
        chi_col: jax.Array,      # (ncol, nlev)
        T_col: jax.Array,        # (ncol, nlev)
        tracer_cols: list,       # list of (ncol, nlev)
    ) -> jax.Array:
        """Stack all variables into (ncol, n_total_channels) in GEN_BE order."""
        parts = [ps_col[:, None]]   # (ncol, 1)
        parts.append(psi_col)       # (ncol, nlev)
        parts.append(chi_col)
        parts.append(T_col)
        for tc in tracer_cols:
            parts.append(tc)
        return jnp.concatenate(parts, axis=1)  # (ncol, n_total_ch)

    def _disassemble_channels(
        self, all_ch: jax.Array
    ) -> tuple:
        """Split (ncol, n_total_channels) back into per-variable arrays."""
        nlev = self._nlev
        ps_col = all_ch[:, 0]
        cursor = 1
        psi_col = all_ch[:, cursor:cursor + nlev]; cursor += nlev
        chi_col = all_ch[:, cursor:cursor + nlev]; cursor += nlev
        T_col   = all_ch[:, cursor:cursor + nlev]; cursor += nlev
        tracer_cols = []
        for _ in self.params.tracer_names:
            tracer_cols.append(all_ch[:, cursor:cursor + nlev])
            cursor += nlev
        return ps_col, psi_col, chi_col, T_col, tracer_cols

    # ------------------------------------------------------------------
    # Forward transform: B^{1/2} v = U v
    # ------------------------------------------------------------------

    def _forward(self, v: jax.Array) -> jax.Array:
        """Apply U = U_wind ∘ U_sigma ∘ U_vert ∘ U_bal ∘ U_horiz to v."""
        params = self.params

        # --- Unpack control vector into per-variable columnar arrays ---
        psi_in = self._to_cols(v, "u")      # (ncol, nlev)  — psi modes
        chi_in = self._to_cols(v, "v") if "v" in self._slices else jnp.zeros_like(psi_in)
        T_in   = self._to_cols(v, "T")      # (ncol, nlev)
        ps_in  = self._to_cols(v, "p_s")    # (ncol,)
        tracer_ins = [
            self._to_cols(v, f"tracers.{n}") for n in params.tracer_names
        ]

        # --- Step 1: U_horiz — horizontal Gaussian correlation ---
        all_ch = self._assemble_channels(ps_in, psi_in, chi_in, T_in, tracer_ins)
        all_ch = self._horiz_smooth(all_ch, inverse=False)

        # --- Step 2: U_bal — add balance from psi modes ---
        # balance_contrib[i] = sum_j all_ch[psi_j] * reg_coeff[i, j]
        psi_smooth = all_ch[:, self._psi_start:self._psi_end]  # (ncol, nlev)
        # (ncol, nlev) @ (nlev, n_total_ch) = (ncol, n_total_ch)
        balance_contrib = psi_smooth @ params.reg_coeff.T
        all_ch = all_ch + balance_contrib

        # --- Step 3: U_vert — vertical EOF expansion per 3D group ---
        ps_ch, psi_ch, chi_ch, T_ch, tracer_chs = self._disassemble_channels(all_ch)

        def vert_expand(modes_col, group_idx):
            """Apply V @ diag(sqrt(λ)) to (ncol, nlev) mode array."""
            sqrt_lam = jnp.sqrt(jnp.maximum(params.vert_eig_val[group_idx], 0.0))
            V = params.vert_eig_vec[group_idx]  # (nlev, nlev)
            # phys = modes_col @ (D @ V^T) = modes_col @ (V.T * sqrt_lam[:, None])
            return modes_col @ (V.T * sqrt_lam[:, None])

        psi_phys = vert_expand(psi_ch, 0)
        chi_phys = vert_expand(chi_ch, 1)
        T_phys   = vert_expand(T_ch,   2)
        tracer_phys = [vert_expand(tc, 3 + k) for k, tc in enumerate(tracer_chs)]

        # --- Step 4: U_sigma — scale surface pressure ---
        ps_phys = ps_ch * params.std_ps

        # --- Step 5: U_wind — psi,chi → u,v ---
        u_phys, v_phys = self._psi_chi_to_wind(psi_phys, chi_phys)

        # --- Pack back to flat control vector ---
        return self._pack_to_flat(u_phys, v_phys, T_phys, ps_phys, tracer_phys)

    # ------------------------------------------------------------------
    # Inverse transform: U^{-1} x
    # ------------------------------------------------------------------

    def _inverse(self, x: jax.Array) -> jax.Array:
        """Apply U^{-1} to a physical-space increment x.

        Applies the operators in reverse order:
          U_wind^{-1} → U_sigma^{-1} → U_vert^{-1} → U_bal^{-1} → U_horiz^{-1}
        """
        params = self.params

        # --- Unpack physical increment ---
        u_col  = self._to_cols(x, "u")
        v_col  = self._to_cols(x, "v") if "v" in self._slices else jnp.zeros(
            (self._ncol, self._nlev), dtype=x.dtype
        )
        T_col  = self._to_cols(x, "T")
        ps_col = self._to_cols(x, "p_s")
        tracer_cols = [self._to_cols(x, f"tracers.{n}") for n in params.tracer_names]

        # --- Inverse U_wind: u,v → psi,chi ---
        psi_col, chi_col = self._wind_to_psi_chi(u_col, v_col)

        # --- Inverse U_sigma: unscale surface pressure ---
        ps_col = ps_col / (params.std_ps + 1e-30)

        # --- Inverse U_vert: diag(1/sqrt(λ)) @ V^T ---
        def vert_inverse(phys_col, group_idx):
            """Apply V @ diag(1/sqrt(λ)) to (ncol, nlev)."""
            sqrt_lam = jnp.sqrt(jnp.maximum(params.vert_eig_val[group_idx], 0.0))
            V = params.vert_eig_vec[group_idx]
            # modes = phys_col @ V @ diag(1/sqrt(λ)) = phys_col @ (V / sqrt_lam[None, :])
            return phys_col @ (V / jnp.maximum(sqrt_lam[None, :], 1e-30))

        psi_modes = vert_inverse(psi_col, 0)
        chi_modes = vert_inverse(chi_col, 1)
        T_modes   = vert_inverse(T_col,   2)
        tracer_modes = [vert_inverse(tc, 3 + k) for k, tc in enumerate(tracer_cols)]

        # --- Assemble all channels in GEN_BE order ---
        all_ch = self._assemble_channels(
            ps_col, psi_modes, chi_modes, T_modes, tracer_modes
        )

        # --- Inverse U_bal: subtract balance contribution ---
        # Forward balance: y = x + R x_psi  (exact inverse: x = y - R y_psi)
        # Since psi channels are unchanged by R (reg_coeff[psi] = 0), y_psi = x_psi,
        # so the exact inverse is: x = y - R y_psi
        psi_ch = all_ch[:, self._psi_start:self._psi_end]   # (ncol, nlev)
        balance_contrib = psi_ch @ params.reg_coeff.T        # (ncol, n_total_ch)
        all_ch = all_ch - balance_contrib

        # --- Inverse U_horiz: reverse diffusion ---
        all_ch = self._horiz_smooth(all_ch, inverse=True)

        # --- Disassemble and pack back ---
        ps_out, psi_out, chi_out, T_out, tracer_outs = self._disassemble_channels(all_ch)
        return self._pack_to_flat(psi_out, chi_out, T_out, ps_out, tracer_outs)

    # ------------------------------------------------------------------
    # Public interface (matches DiagonalB / DiffusionB)
    # ------------------------------------------------------------------

    def sqrt_multiply(self, v: jax.Array) -> jax.Array:
        """Apply B^{1/2} v. Pure, differentiable, JIT-compatible."""
        return self._forward(v)

    def inv_multiply(self, x: jax.Array) -> jax.Array:
        """Apply B^{-1} x = U^{-T} U^{-1} x.

        Uses jax.vjp to compute U^{-T} exactly (U^{-1} is linear, so its
        VJP is the matrix transpose applied to the cotangent).

        Notes
        -----
        The jax.vjp call builds a linearisation graph of _inverse.  For large
        states this is memory-proportional to the number of scalar operations in
        _inverse (dominated by the diffusion scan, O(n_iter * ncol * n_ch)).
        For the preconditioned formulation (use_preconditioning=True in
        IncrementalConfig) this method is never called; prefer that path for
        production runs.
        """
        y = self._inverse(x)            # U^{-1} x
        _, vjp_fn = jax.vjp(self._inverse, x)
        return vjp_fn(y)[0]             # U^{-T} y  (exact for linear U^{-1})


# ---------------------------------------------------------------------------
# Persistence: save / load GenBEParams
# ---------------------------------------------------------------------------

def save_gen_be_params(params: GenBEParams, path: str | os.PathLike) -> None:
    """Save GenBEParams to disk.

    Writes two files:
      <path>.npz   — JAX arrays as float64 numpy arrays
      <path>.json  — static metadata (tracer_names, n_levels, wind_transform)

    Parameters
    ----------
    params : GenBEParams
    path : str or Path
        Base path without extension (e.g. ``"experiments/run1/gen_be"``).
        The parent directory must already exist.

    Examples
    --------
    >>> save_gen_be_params(params, "outputs/gen_be")
    # writes outputs/gen_be.npz and outputs/gen_be.json
    """
    path = Path(path)
    np.savez(
        str(path) + ".npz",
        vert_eig_vec=np.array(params.vert_eig_vec),
        vert_eig_val=np.array(params.vert_eig_val),
        std_ps=np.array(params.std_ps),
        reg_coeff=np.array(params.reg_coeff),
        len_scale=np.array(params.len_scale),
    )
    meta = {
        "tracer_names": list(params.tracer_names),
        "n_levels": params.n_levels,
        "wind_transform": getattr(params, "wind_transform", "mpas_helmholtz"),
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    logger.info("Saved GenBEParams to %s.{npz,json}", path)


def load_gen_be_params(path: str | os.PathLike) -> GenBEParams:
    """Load GenBEParams from disk.

    Parameters
    ----------
    path : str or Path
        Base path without extension (e.g. ``"experiments/run1/gen_be"``).
        Reads ``<path>.npz`` and ``<path>.json``.

    Returns
    -------
    GenBEParams
    """
    path = Path(path)
    arrays = np.load(str(path) + ".npz")
    meta = json.loads(path.with_suffix(".json").read_text())
    wind_transform = meta.get("wind_transform", "mpas_helmholtz")
    return GenBEParams(
        vert_eig_vec=jnp.array(arrays["vert_eig_vec"]),
        vert_eig_val=jnp.array(arrays["vert_eig_val"]),
        std_ps=jnp.array(arrays["std_ps"]),
        reg_coeff=jnp.array(arrays["reg_coeff"]),
        len_scale=jnp.array(arrays["len_scale"]),
        tracer_names=tuple(meta["tracer_names"]),
        n_levels=int(meta["n_levels"]),
        wind_transform=str(wind_transform),
    )
