"""Semi-implicit time integration for the spectral primitive equations.

Implements the Hoskins & Simmons (1975) scheme that treats the fast
gravity-wave terms implicitly in the divergence equation while keeping
all other terms explicit. This removes the gravity-wave CFL constraint,
enabling much larger time steps and higher spectral resolutions (T42+).

The key idea: linearize the gravity-wave terms around a reference
temperature T_ref. The divergence equation becomes:

    D' = D_explicit - alpha^2 * dt^2 * lap * Gamma * D'

where Gamma is the vertical coupling matrix encoding how divergence at
one level affects the pressure gradient at other levels through the
hydrostatic relation. For each spectral mode (n, m), we solve:

    (I + alpha^2 * dt^2 * n(n+1)/a^2 * Gamma) * D' = D_explicit

Since n(n+1)/a^2 are eigenvalues of -nabla^2 on the sphere, each mode
requires solving an L x L system (L = number of levels). We precompute
LU factorizations for each unique n value.

After correcting the divergence, the temperature and lnps corrections
maintain consistency:

    T_corrected  = T_explicit  - alpha * dt * T_ref * (D_corr - D_expl)
    lnps_corrected = lnps_explicit - alpha * dt * sum_k((D_corr - D_expl)_k * dsigma_k) / sigma_range

Combined with SSP-RK3, each RK stage applies the explicit tendencies
first, then applies the implicit correction.

References
----------
- Hoskins, B. J. & Simmons, A. J. (1975). A multi-layer spectral model
  and the semi-implicit method. Q. J. R. Met. Soc., 101, 637-655.
- Simmons, A. J. et al. (1978). Stability of the semi-implicit method
  of time integration. Mon. Wea. Rev., 106, 405-412.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.numpy.linalg import solve

from legoesm.grids.gaussian import GaussianGrid
from legoesm.grids.vertical import SigmaCoordinate, HybridSigmaPressureCoordinate
from legoesm import constants


class SemiImplicitData(NamedTuple):
    """Precomputed matrices for the semi-implicit scheme.

    Attributes
    ----------
    Gamma : jax.Array, shape (nlev, nlev)
        Vertical coupling matrix (Hoskins & Simmons 1975).
        Gamma[k, k'] is the contribution of D at level k' to the
        geopotential + R_d*T_ref*lnps tendency at level k.
    eigenvalues : jax.Array, shape (n_max+1,)
        n(n+1)/a^2 for each total wavenumber n.
    si_matrices : jax.Array, shape (n_max+1, nlev, nlev)
        (I + alpha^2 * dt^2 * eigenvalue[n] * Gamma) for each n.
        Precomputed for direct solve.
    T_ref : float
        Reference temperature for linearization [K].
    alpha : float
        Implicitness parameter (0.5 = Crank-Nicolson).
    dt : float
        Time step used for precomputation.
    dsigma : jax.Array, shape (nlev,)
        Layer thicknesses in sigma (for lnps correction).
    sigma_range : float
        1 - sigma_top (for lnps correction).
    """
    Gamma: jax.Array
    eigenvalues: jax.Array
    si_matrices: jax.Array
    T_ref: float
    alpha: float
    dt: float
    dsigma: jax.Array
    sigma_range: jax.Array


def compute_Gamma_matrix(
    sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
    T_ref: float,
) -> jax.Array:
    """Compute the Hoskins-Simmons vertical coupling matrix Gamma.

    Gamma encodes the vertical coupling of divergence to pressure
    gradient through the hydrostatic relation and the equation of state.

    The coupling has two pathways:

    1. **Geopotential (hydrostatic integral)**: D at level k' changes T
       at k' by -alpha*dt*T_ref*D_{k'} (compression heating). This T
       change propagates into the geopotential Phi at level k through
       the Simmons-Burridge hydrostatic integral:

           Phi_k = phis + sum_{k''>k} R_d*T_{k''}*ln_ratio[k'']
                   + R_d*T_k*alpha_SB[k]

       So dPhi_k/dT_{k'} = R_d*ln_ratio[k'] for k'>k (below level k),
                          = R_d*alpha_SB[k]   for k'=k (self-coupling),
                          = 0                  for k'<k (above level k).

    2. **Surface pressure**: D at all levels changes lnps via the
       continuity equation: d(lnps)/dt = -sum_k D_k*dsigma_k/sigma_range.
       This enters the PGF as R_d*T_ref*lnps.

    The combined matrix is:

        Gamma[k,k'] = R_d*T_ref * (S[k,k'] + dsigma[k']/sigma_range)

    where S is an upper-triangular geopotential coupling matrix.

    For hybrid coordinates, the reference ln_ratio and alpha are
    precomputed at p_s = p_ref (standard linearization). The effective
    dsigma = dA + dB and B_range replace dsigma and sigma_range.

    Parameters
    ----------
    sigma_coord : SigmaCoordinate or HybridSigmaPressureCoordinate
        Vertical coordinate.
    T_ref : float
        Reference temperature [K].

    Returns
    -------
    Gamma : jax.Array, shape (nlev, nlev)
    """
    R_d = constants.R_d
    nlev = sigma_coord.n_levels

    _hybrid = isinstance(sigma_coord, HybridSigmaPressureCoordinate)

    if _hybrid:
        dsigma = jnp.asarray(sigma_coord.dsigma_eff, dtype=jnp.float64)
        ln_ratio = jnp.asarray(sigma_coord.ln_ratio_ref, dtype=jnp.float64)
        alpha_sb = jnp.asarray(sigma_coord.alpha_ref, dtype=jnp.float64)
        sigma_range = jnp.asarray(sigma_coord.B_range, dtype=jnp.float64)
    else:
        dsigma = jnp.asarray(sigma_coord.dsigma, dtype=jnp.float64)
        ln_ratio = jnp.asarray(sigma_coord.ln_ratio, dtype=jnp.float64)
        alpha_sb = jnp.asarray(sigma_coord.alpha, dtype=jnp.float64)
        sigma_top = jnp.asarray(sigma_coord.sigma_half[0], dtype=jnp.float64)
        sigma_range = 1.0 - sigma_top

    # --- Geopotential coupling matrix S[k, k'] ---
    # S[k, k'] = ln_ratio[k']  if k' > k  (k' is below level k in the
    #                                        hydrostatic integral)
    #          = alpha_sb[k]    if k' == k  (Simmons-Burridge self-coupling)
    #          = 0              if k' < k   (k' is above level k)
    k_idx = jnp.arange(nlev)
    kk = k_idx[:, None]    # row indices (nlev, 1)
    kkp = k_idx[None, :]   # col indices (1, nlev)

    S = jnp.where(
        kkp > kk,
        ln_ratio[None, :],              # below: ln_ratio[k']
        jnp.where(
            kkp == kk,
            alpha_sb[:, None],           # diagonal: alpha_sb[k]
            0.0,                         # above: zero
        ),
    )

    # --- Surface pressure coupling ---
    # Each k' contributes dsigma[k']/sigma_range to all rows k
    lnps_coupling = dsigma[None, :] / sigma_range  # (1, nlev) -> broadcast

    # --- Combined Gamma ---
    # NOTE: No E-variable diagonal (lnps_ref * I) — the correct PGF form
    # does not include the R_d*lnps_0*∇²(T') same-level coupling.
    Gamma = R_d * T_ref * (S + lnps_coupling)

    return Gamma


def precompute_si_matrices(
    grid: GaussianGrid,
    sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
    T_ref: float = 300.0,
    alpha: float = 0.5,
    dt: float = 1200.0,
) -> SemiImplicitData:
    """Precompute semi-implicit matrices for all wavenumbers.

    Parameters
    ----------
    grid : GaussianGrid
        Gaussian grid (for radius and max wavenumber).
    sigma_coord : SigmaCoordinate or HybridSigmaPressureCoordinate
        Vertical coordinate.
    T_ref : float
        Reference temperature for linearization [K].
    alpha : float
        Implicitness parameter (0.5 = Crank-Nicolson, 1.0 = fully implicit).
    dt : float
        Time step [seconds]. Must match the dt used in integration.

    Returns
    -------
    SemiImplicitData
        Precomputed matrices for use in si_correction.
    """
    _hybrid = isinstance(sigma_coord, HybridSigmaPressureCoordinate)

    a = grid.radius
    n_max = grid.n_max
    nlev = sigma_coord.n_levels

    # Compute vertical coupling matrix
    Gamma = compute_Gamma_matrix(sigma_coord, T_ref)

    # Eigenvalues of -nabla^2: n(n+1)/a^2
    ns = jnp.arange(n_max + 1, dtype=jnp.float64)
    eigenvalues = ns * (ns + 1) / a**2

    # Build (I + alpha^2 * dt^2 * eigenvalue[n] * Gamma) for each n
    I = jnp.eye(nlev, dtype=jnp.float64)
    coeff = alpha**2 * dt**2

    # Vectorized construction
    si_matrices = I[None, :, :] + coeff * eigenvalues[:, None, None] * Gamma[None, :, :]

    # Store dsigma and sigma_range for lnps correction
    if _hybrid:
        dsigma = jnp.asarray(sigma_coord.dsigma_eff, dtype=jnp.float64)
        sigma_range = jnp.asarray(sigma_coord.B_range, dtype=jnp.float64)
    else:
        dsigma = jnp.asarray(sigma_coord.dsigma, dtype=jnp.float64)
        sigma_top = jnp.asarray(sigma_coord.sigma_half[0], dtype=jnp.float64)
        sigma_range = 1.0 - sigma_top

    return SemiImplicitData(
        Gamma=Gamma,
        eigenvalues=eigenvalues,
        si_matrices=si_matrices,
        T_ref=T_ref,
        alpha=alpha,
        dt=dt,
        dsigma=dsigma,
        sigma_range=sigma_range,
    )


def si_correction(
    state_explicit,
    state_old,
    si_data: SemiImplicitData,
    grid: GaussianGrid,
    dt: float,
):
    """Apply semi-implicit correction to divergence, temperature, and lnps.

    Given the explicitly advanced state, correct the divergence field
    by solving the implicit system, then correct temperature and surface
    pressure to maintain consistency.

    Parameters
    ----------
    state_explicit : SpectralHydrostaticState
        State after explicit RK stage.
    state_old : SpectralHydrostaticState
        State at the beginning of the RK stage (before tendency).
    si_data : SemiImplicitData
        Precomputed matrices.
    grid : GaussianGrid
        Gaussian grid.
    dt : float
        Time step.

    Returns
    -------
    state_corrected : SpectralHydrostaticState
        State with implicit correction applied.
    """
    alpha = si_data.alpha
    T_ref = si_data.T_ref

    # Explicit divergence (what RK produced) and old divergence (start of stage)
    div_hat_explicit = state_explicit.div_hat.data  # (n_sh, nlev)
    div_hat_old = state_old.div_hat.data            # (n_sh, nlev)

    # The Hoskins-Simmons SI correction solves:
    # (I + alpha^2*dt^2*eigenvalue[n]*Gamma) * D_new = D_explicit + alpha^2*dt^2*eigenvalue[n]*Gamma * D_old
    #
    # This ensures only the TENDENCY is implicitly modified, not the full state:
    # D_new = D_old + (I + M)^{-1} * dt * F(X_old)
    # where M = alpha^2 * dt^2 * eigenvalue * Gamma.

    # Map each SH coefficient to its total wavenumber n
    ns = grid.ls  # (n_sh,) -- total wavenumber for each coefficient

    # Gather si_matrices for each coefficient's n value
    # si_matrices: (n_max+1, nlev, nlev), ns: (n_sh,)
    matrices = si_data.si_matrices[ns]  # (n_sh, nlev, nlev)

    # Build RHS: D_explicit + M * D_old
    # M * D_old = (matrices - I) * D_old
    I_nlev = jnp.eye(matrices.shape[-1], dtype=matrices.dtype)
    M_times_D_old = jnp.einsum('...ij,...j->...i', matrices - I_nlev, div_hat_old)
    rhs = div_hat_explicit + M_times_D_old

    # Solve: matrices @ div_corrected = rhs (per coefficient)
    div_hat_corrected = solve(matrices, rhs[..., None]).squeeze(-1)

    # Divergence correction
    delta_div = div_hat_corrected - div_hat_explicit  # (n_sh, nlev)

    # Temperature correction:
    # T_corrected = T_explicit - alpha*dt*T_ref*(D_corrected - D_explicit)
    T_hat_corrected = state_explicit.T_hat.data - alpha * dt * T_ref * delta_div

    # Surface pressure (lnps) correction:
    # lnps_corrected = lnps_explicit - alpha*dt * sum_k(delta_D_k * dsigma_k) / sigma_range
    dsigma = si_data.dsigma  # (nlev,)
    sigma_range = si_data.sigma_range
    delta_div_weighted = delta_div * dsigma[None, :]  # (n_sh, nlev)
    lnps_correction = -alpha * dt * jnp.sum(delta_div_weighted, axis=-1) / sigma_range
    lnps_hat_corrected = state_explicit.lnps_hat.data + lnps_correction

    return state_explicit._replace(
        div_hat=state_explicit.div_hat.replace(data=div_hat_corrected),
        T_hat=state_explicit.T_hat.replace(data=T_hat_corrected),
        lnps_hat=state_explicit.lnps_hat.replace(data=lnps_hat_corrected),
    )


def ssp_rk3_step_si(
    state,
    tendency_fn,
    dt: float,
    si_data: SemiImplicitData,
    grid: GaussianGrid,
):
    """SSP-RK3 step with semi-implicit gravity wave correction.

    Each RK stage:
    1. Compute explicit tendencies
    2. Apply explicit update (standard SSP-RK3)
    3. Apply implicit correction to divergence, temperature, and lnps

    Parameters
    ----------
    state : SpectralHydrostaticState
        Current state.
    tendency_fn : callable
        Explicit tendency function.
    dt : float
        Time step.
    si_data : SemiImplicitData
        Precomputed matrices.
    grid : GaussianGrid
        Gaussian grid.

    Returns
    -------
    SpectralHydrostaticState
        State after one time step.
    """
    # Stage 1: k1 = state + dt * F(state), then SI correction
    tend_0 = tendency_fn(state)
    k1_explicit = _pytree_axpy(state, tend_0, dt)
    k1 = si_correction(k1_explicit, state, si_data, grid, dt)

    # Stage 2: k2 = 3/4 * state + 1/4 * (k1 + dt * F(k1) + SI correction)
    tend_1 = tendency_fn(k1)
    k1_step_explicit = _pytree_axpy(k1, tend_1, dt)
    k1_step = si_correction(k1_step_explicit, k1, si_data, grid, dt)
    k2 = _pytree_linear_combination(state, k1_step, 0.75, 0.25)

    # Stage 3: k3 = 1/3 * state + 2/3 * (k2 + dt * F(k2) + SI correction)
    tend_2 = tendency_fn(k2)
    k2_step_explicit = _pytree_axpy(k2, tend_2, dt)
    k2_step = si_correction(k2_step_explicit, k2, si_data, grid, dt)
    k3 = _pytree_linear_combination(state, k2_step, 1.0 / 3.0, 2.0 / 3.0)

    return k3


def euler_si_step(
    state,
    tendency_fn,
    dt: float,
    si_data: SemiImplicitData,
    grid: GaussianGrid,
):
    """Forward Euler step with semi-implicit correction.

    Used for the first time step of leapfrog integration (startup).
    SI data should be precomputed with the same dt.

    Parameters
    ----------
    state : SpectralHydrostaticState
        Current state.
    tendency_fn : callable
        Explicit tendency function.
    dt : float
        Time step.
    si_data : SemiImplicitData
        Precomputed matrices (for dt).
    grid : GaussianGrid
        Gaussian grid.

    Returns
    -------
    SpectralHydrostaticState
        State after one forward Euler + SI step.
    """
    tend = tendency_fn(state)
    state_explicit = _pytree_axpy(state, tend, dt)
    return si_correction(state_explicit, state, si_data, grid, dt)


def leapfrog_si_step(
    state_n,
    state_nm1,
    tendency_fn,
    dt: float,
    si_data: SemiImplicitData,
    grid: GaussianGrid,
):
    """Leapfrog step with semi-implicit gravity wave correction.

    Computes X^{n+1} from X^n and X^{n-1} using:
    1. Explicit leapfrog: X* = X^{n-1} + 2*dt * F(X^n)
    2. SI correction: solves the implicit gravity-wave system

    The SI data should be precomputed with dt_eff = 2*dt (the full
    leapfrog step size), using the same alpha.

    Leapfrog is neutral for oscillatory modes (|amplification factor| = 1),
    so unlike SSP-RK3, the SI correction only needs to handle the implicit
    gravity-wave coupling without fighting Euler amplification.

    Parameters
    ----------
    state_n : SpectralHydrostaticState
        State at time level n.
    state_nm1 : SpectralHydrostaticState
        State at time level n-1.
    tendency_fn : callable
        Explicit tendency function.
    dt : float
        Base time step. The leapfrog step is 2*dt.
    si_data : SemiImplicitData
        Precomputed matrices (for dt_eff = 2*dt).
    grid : GaussianGrid
        Gaussian grid.

    Returns
    -------
    SpectralHydrostaticState
        State at time level n+1.

    References
    ----------
    - Hoskins, B. J. & Simmons, A. J. (1975). A multi-layer spectral model
      and the semi-implicit method. Q. J. R. Met. Soc., 101, 637-655.
    """
    tend = tendency_fn(state_n)
    dt2 = 2.0 * dt
    state_explicit = _pytree_axpy(state_nm1, tend, dt2)
    return si_correction(state_explicit, state_n, si_data, grid, dt2)


def robert_asselin_filter(state_nm1, state_n, state_np1, gamma, alpha=0.5):
    """Robert-Asselin-Williams (RAW) time filter for leapfrog.

    The standard Robert-Asselin filter damps the computational mode
    (2*dt oscillation) that leapfrog permits, but introduces a first-order
    phase error.  The Williams (2009) modification splits the correction
    between the current and next time levels, restoring second-order
    accuracy while preserving the damping:

        d_n = (gamma/2) * (X^{n-1} - 2*X^n + X^{n+1})
        X^n_filtered   = X^n   + (1 - alpha) * d_n
        X^{n+1}_filtered = X^{n+1} + alpha * d_n

    With alpha=0.5 this is the RAW filter; alpha=0 recovers the original
    Robert-Asselin filter.

    Parameters
    ----------
    state_nm1 : pytree
        State at time n-1 (already filtered from previous step).
    state_n : pytree
        State at time n (unfiltered).
    state_np1 : pytree
        State at time n+1 (just computed, unfiltered).
    gamma : float
        Filter coefficient (typically 0.05-0.1).
    alpha : float
        Williams parameter. 0.5 = RAW (default, recommended for long
        climate runs). 0.0 = original Robert-Asselin.

    Returns
    -------
    state_n_filtered, state_np1_filtered : tuple of pytrees
        Filtered states at time n and n+1.

    References
    ----------
    - Robert, A. J. (1966). The integration of a low order spectral form of
      the primitive meteorological equations. J. Met. Soc. Japan, 44, 237-245.
    - Asselin, R. (1972). Frequency filter for time integrations.
      Mon. Wea. Rev., 100, 487-490.
    - Williams, P. D. (2009). A proposed modification to the Robert-Asselin
      time filter. Mon. Wea. Rev., 137, 2538-2546.
    """
    coeff = gamma / 2.0
    d_n = jax.tree.map(
        lambda xm, xn, xp: coeff * (xm - 2.0 * xn + xp),
        state_nm1, state_n, state_np1,
    )
    state_n_filtered = jax.tree.map(
        lambda xn, dn: xn + (1.0 - alpha) * dn,
        state_n, d_n,
    )
    state_np1_filtered = jax.tree.map(
        lambda xp, dn: xp + alpha * dn,
        state_np1, d_n,
    )
    return state_n_filtered, state_np1_filtered


def _pytree_axpy(x, y, alpha):
    """Compute x + alpha * y for two pytrees."""
    return jax.tree.map(lambda xi, yi: xi + alpha * yi, x, y)


def _pytree_linear_combination(x, y, a, b):
    """Compute a * x + b * y for two pytrees."""
    return jax.tree.map(lambda xi, yi: a * xi + b * yi, x, y)
