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
    tau : jax.Array, shape (nlev, nlev)
        Reference thermodynamic coupling matrix: dT'/dt = -tau @ D
        (linearized adiabatic heating, see compute_tau_matrix).  Used for
        the temperature correction; consistent with Gamma = R_d*(S @ tau +
        T_ref*1*b^T).
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
    tau: jax.Array
    eigenvalues: jax.Array
    si_matrices: jax.Array
    T_ref: float
    alpha: float
    dt: float
    dsigma: jax.Array
    sigma_range: jax.Array


def compute_tau_matrix(
    sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
    T_ref: float,
) -> jax.Array:
    """Reference thermodynamic coupling matrix ``tau``: ``dT'/dt = -tau @ D``.

    ``tau`` is the linearization of the dycore's adiabatic heating term
    ``kappa*T*omega/p`` about the isothermal (``T = T_ref``) resting
    reference state.  For an isothermal reference the vertical advection of
    ``T`` vanishes (``dT_ref/dsigma = 0``), so the ENTIRE temperature-from-
    divergence coupling is the adiabatic term:

        dT'_k/dt = kappa * T_ref * (omega/p)_k,
        (omega/p)_k = d(ln p_s)/dt + sigma_dot_k / sigma_k,

    where ``d(ln p_s)/dt`` and ``sigma_dot`` are diagnosed from the
    divergence ``D`` by the SAME shared operators the model integrates with
    (continuity + :func:`compute_sigma_dot` + :func:`compute_pressure_velocity`).
    Note the surface pressure ``p_s`` cancels in ``omega/p``, so ``tau`` is
    independent of the reference pressure.  Because this ``tau`` is the exact
    linearization of the model's own thermodynamics, the gravity-wave
    structure matrix ``Gamma = R_d*(S @ tau + T_ref*1*b^T)`` is symmetrizable
    with REAL, POSITIVE eigenvalues — the vertical normal-mode "equivalent
    depths" ``H = lambda/g`` (external/Lamb mode ~10 km down to tiny internal
    modes).

    This replaces the earlier diagonal approximation ``tau = T_ref*I``, which
    dropped both the factor ``kappa`` and the vertical non-locality of
    ``omega/p`` and therefore made ``Gamma`` non-normal, with complex
    eigenvalues at ``nlev >= 4`` (issue #960 — the 2nd contributor to the
    T85 NaN).

    Parameters
    ----------
    sigma_coord : SigmaCoordinate or HybridSigmaPressureCoordinate
        Vertical coordinate.  Hybrid coordinates are linearized at
        ``p_s = p_ref`` via their sigma-compatibility views (``sigma_full``,
        ``dsigma``, ``fractional_sigma``).
    T_ref : float
        Reference temperature [K].

    Returns
    -------
    tau : jax.Array, shape (nlev, nlev)
    """
    # Reuse the model's shared vertical operators — never re-derive sigma_dot
    # or omega here (CLAUDE.md: reuse shared numerics).  Function-scope import
    # avoids a core->grids top-level cycle.
    from legoesm.grids.vertical import compute_sigma_dot, compute_pressure_velocity

    kappa = constants.kappa
    nlev = sigma_coord.n_levels
    _hybrid = isinstance(sigma_coord, HybridSigmaPressureCoordinate)

    if _hybrid:
        dsigma = jnp.asarray(sigma_coord.dsigma_eff, dtype=jnp.float64)
        sigma_range = jnp.asarray(sigma_coord.B_range, dtype=jnp.float64)
    else:
        dsigma = jnp.asarray(sigma_coord.dsigma, dtype=jnp.float64)
        sigma_top = jnp.asarray(sigma_coord.sigma_half[0], dtype=jnp.float64)
        sigma_range = 1.0 - sigma_top

    sigma_full = jnp.asarray(sigma_coord.sigma_full, dtype=jnp.float64)
    p_ref = jnp.asarray(constants.p_ref, dtype=jnp.float64)
    p_full = sigma_full * p_ref  # reference full-level pressure (p_ref cancels)

    def adiabatic_of_D(D: jax.Array) -> jax.Array:
        # Continuity: d(ln p_s)/dt = -D_total / sigma_range.
        D_total = jnp.sum(D * dsigma)
        dlnps_dt = -D_total / sigma_range
        dp_s_dt = p_ref * dlnps_dt
        # sigma_dot and omega from the shared discrete operators.
        sigma_dot = compute_sigma_dot(D, sigma_coord)
        omega = compute_pressure_velocity(sigma_dot, p_ref, dp_s_dt, sigma_coord)
        return kappa * T_ref * omega / p_full  # = adiabatic dT'/dt

    # ``adiabatic_of_D`` is EXACTLY linear in D, so its Jacobian is the
    # operator matrix; dT'/dt = -tau @ D  =>  tau = -d(adiabatic)/dD.
    dadiab_dD = jax.jacfwd(adiabatic_of_D)(jnp.zeros(nlev, dtype=jnp.float64))
    return -dadiab_dD


def compute_Gamma_matrix(
    sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
    T_ref: float,
    tau: jax.Array | None = None,
) -> jax.Array:
    """Compute the Hoskins-Simmons vertical coupling matrix Gamma.

    Gamma encodes the vertical coupling of divergence to pressure
    gradient through the hydrostatic relation and the equation of state,
    substituting the linearized thermodynamic and continuity equations:

        d^2 D/dt^2 = nabla^2 * Gamma * D,
        Gamma = R_d * (S @ tau + T_ref * 1 * b^T).

    The coupling has two pathways:

    1. **Geopotential (hydrostatic integral)**: divergence heats/cools each
       level adiabatically via ``tau`` (``dT'/dt = -tau @ D``, see
       :func:`compute_tau_matrix`); the temperature change propagates into
       the geopotential ``Phi`` at level k through the Simmons-Burridge
       hydrostatic integral:

           Phi_k = phis + sum_{k''>k} R_d*T_{k''}*ln_ratio[k'']
                   + R_d*T_k*alpha_SB[k]

       So ``dPhi_k/dT_{k'} = R_d * S[k, k']`` with
           S[k, k'] = ln_ratio[k']  for k'>k (below level k),
                    = alpha_SB[k]    for k'=k (Simmons-Burridge self-coupling),
                    = 0              for k'<k (above level k).

    2. **Surface pressure**: D at all levels changes lnps via the
       continuity equation: d(lnps)/dt = -sum_k D_k*dsigma_k/sigma_range.
       This enters the PGF as R_d*T_ref*lnps, giving the rank-1 term
       ``T_ref * 1 * b^T`` with ``b[k'] = dsigma[k']/sigma_range``.

    Using the CONSISTENT ``tau`` (the linearization of the model's own
    adiabatic term) — rather than the diagonal ``tau = T_ref*I`` — is what
    makes Gamma symmetrizable with real, positive eigenvalues at all nlev
    (issue #960).

    For hybrid coordinates, the reference ln_ratio and alpha are
    precomputed at p_s = p_ref (standard linearization). The effective
    dsigma = dA + dB and B_range replace dsigma and sigma_range.

    Parameters
    ----------
    sigma_coord : SigmaCoordinate or HybridSigmaPressureCoordinate
        Vertical coordinate.
    T_ref : float
        Reference temperature [K].
    tau : jax.Array, optional
        Precomputed thermodynamic coupling matrix (shape (nlev, nlev)).
        If None, computed via :func:`compute_tau_matrix`.

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

    if tau is None:
        tau = compute_tau_matrix(sigma_coord, T_ref)

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

    # --- Surface pressure coupling (rank-1: 1 * b^T) ---
    # Each k' contributes dsigma[k']/sigma_range to all rows k
    lnps_coupling = dsigma[None, :] / sigma_range  # (1, nlev) -> broadcast

    # --- Combined Gamma = R_d * (S @ tau + T_ref * 1 * b^T) ---
    # Geopotential response to the adiabatic heating (S @ tau) plus the
    # surface-pressure PGF (rank-1).  Consistent ``tau`` => real, positive
    # eigenvalues (equivalent depths).  NOTE: No E-variable diagonal
    # (lnps_ref * I) — the correct PGF form does not include the
    # R_d*lnps_0*∇²(T') same-level coupling.
    Gamma = R_d * (S @ tau + T_ref * lnps_coupling)

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
    # alpha is the theta-method implicitness weight. It MUST lie in (0, 1]:
    # the forward ``si_correction`` RHS carries a ``((1-alpha)/alpha)`` factor
    # (division by alpha), and alpha<=0 is not semi-implicit at all (fully
    # explicit -> no gravity-wave stabilization). Reject it loudly rather than
    # silently producing a no-op / NaN correction.
    if not (0.0 < alpha <= 1.0):
        raise ValueError(
            f"semi-implicit alpha must be in (0, 1] (0.5=Crank-Nicolson, "
            f"1.0=fully implicit); got {alpha!r}"
        )

    _hybrid = isinstance(sigma_coord, HybridSigmaPressureCoordinate)

    a = grid.radius
    n_max = grid.n_max
    nlev = sigma_coord.n_levels

    # Compute vertical coupling matrices.  ``tau`` is the linearized adiabatic
    # heating (dT'/dt = -tau @ D); ``Gamma`` reuses it so the reference matrix
    # and the temperature correction stay consistent (issue #960).
    tau = compute_tau_matrix(sigma_coord, T_ref)
    Gamma = compute_Gamma_matrix(sigma_coord, T_ref, tau=tau)

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
        tau=tau,
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
    predictor: str = "forward",
):
    """Apply semi-implicit correction to divergence, temperature, and lnps.

    Given the explicitly advanced state, correct the divergence field
    by solving the implicit system, then correct temperature and surface
    pressure to maintain consistency.

    The exact implicit centering depends on how the *explicit predictor*
    was built, so it MUST match the outer integrator (``predictor``):

    - ``"forward"`` (Euler / SSP-RK3 stage): the predictor is a forward step
      ``X* = X_old + dt*F(X_old)``, so ``state_old`` is the level the step
      advances FROM.  The trapezoidal (theta-method) Helmholtz solve is

          (I + M) * D_new = D_explicit - ((1-alpha)/alpha) * M * D_old
          delta_D         = D_new - D_old

      with ``M = alpha^2*dt^2*eigenvalue[n]*Gamma``.  alpha=0.5 is
      Crank-Nicolson (neutral, |lambda|=1); alpha>0.5 damps.  (The earlier
      ``+ M*D_old`` RHS with ``delta_D = D_new - D_explicit`` was a
      forward-Euler-amplifying increment, unconditionally unstable for the
      gravity wave -- issue #920.)

    - ``"leapfrog"``: the predictor is the centered ``2*dt`` leapfrog
      ``X* = X^{n-1} + 2*dt*F(X^n)`` and ``state_old = X^n`` is the CENTER
      time level, NOT the level the step advances from.  The forward
      trapezoidal derivation does NOT apply (its ``D_explicit = D_old +
      dt*F(D_old)`` identity is false here); using it drops leapfrog from
      2nd- to ~1st-order and makes it weakly amplifying (|lambda|>1).  The
      correct centered leapfrog-SI keeps the increment form

          (I + M) * D_new = D_explicit + M * D^n
          delta_D         = D_new - D_explicit

      which is 2nd-order and neutral to O((omega*dt)^4) (verified against the
      textbook centered SI-leapfrog).  This is the pre-#920 behaviour, now
      scoped to the leapfrog predictor only.

    Parameters
    ----------
    state_explicit : SpectralHydrostaticState
        State after the explicit predictor.
    state_old : SpectralHydrostaticState
        For ``"forward"``: the stage-start state (level advanced FROM).
        For ``"leapfrog"``: the CENTER time level X^n.
    si_data : SemiImplicitData
        Precomputed matrices.
    grid : GaussianGrid
        Gaussian grid.
    dt : float
        Time step (the leapfrog caller passes the full 2*dt step).
    predictor : str
        ``"forward"`` (Euler/RK, default) or ``"leapfrog"``.  Selects the
        implicit centering.  Raises on any other value.

    Returns
    -------
    state_corrected : SpectralHydrostaticState
        State with implicit correction applied.
    """
    if predictor not in ("forward", "leapfrog"):
        raise ValueError(
            f"si_correction predictor must be 'forward' or 'leapfrog', "
            f"got {predictor!r}"
        )

    alpha = si_data.alpha
    tau = si_data.tau  # (nlev, nlev) reference thermodynamic coupling

    # Explicit divergence (predictor output) and the reference-level divergence
    # (forward: stage-start X_old; leapfrog: center X^n).
    div_hat_explicit = state_explicit.div_hat.data  # (n_sh, nlev)
    div_hat_old = state_old.div_hat.data            # (n_sh, nlev)

    # Map each SH coefficient to its total wavenumber n
    ns = grid.ls  # (n_sh,) -- total wavenumber for each coefficient

    # Gather si_matrices for each coefficient's n value
    # si_matrices: (n_max+1, nlev, nlev), ns: (n_sh,)
    matrices = si_data.si_matrices[ns]  # (n_sh, nlev, nlev)

    # M * D_old = (matrices - I) * D_old  (M = alpha^2*dt^2*eigenvalue*Gamma)
    I_nlev = jnp.eye(matrices.shape[-1], dtype=matrices.dtype)
    M_times_D_old = jnp.einsum('...ij,...j->...i', matrices - I_nlev, div_hat_old)

    # Build RHS with the centering that matches the predictor (see docstring).
    # ``predictor`` is a static Python str (feature gate) -> a plain ``if`` is
    # correct here; this is not a traced/data-dependent branch.
    if predictor == "forward":
        rhs = div_hat_explicit - ((1.0 - alpha) / alpha) * M_times_D_old
    else:  # "leapfrog": centered 2*dt increment off the explicit predictor
        rhs = div_hat_explicit + M_times_D_old

    # Solve: matrices @ div_corrected = rhs (per coefficient)
    div_hat_corrected = solve(matrices, rhs[..., None]).squeeze(-1)

    # Divergence correction. Forward: full implicit increment off the OLD stage
    # state (D_new - D_old); measuring off D_explicit would double-count the
    # explicit gravity-wave increment already in D_explicit (issue #920).
    # Leapfrog: increment off the explicit predictor (D_new - D_explicit), the
    # centered-leapfrog form that stays 2nd-order and neutral.
    if predictor == "forward":
        delta_div = div_hat_corrected - div_hat_old        # (n_sh, nlev)
    else:  # "leapfrog"
        delta_div = div_hat_corrected - div_hat_explicit   # (n_sh, nlev)

    # Temperature correction (consistent with Gamma's linearized adiabatic
    # coupling): dT'/dt = -tau @ D  =>
    #   T_corrected = T_explicit - alpha*dt * (tau @ delta_div)
    # delta_div is (n_sh, nlev); (tau @ delta_div)[n,k] = sum_j tau[k,j]*delta_div[n,j]
    # = (delta_div @ tau^T)[n,k].  Reduces to the old -alpha*dt*T_ref*delta_div
    # only when tau = T_ref*I (the previous diagonal approximation, #960).
    tau_delta_div = delta_div @ tau.T  # (n_sh, nlev)
    T_hat_corrected = state_explicit.T_hat.data - alpha * dt * tau_delta_div

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
    # Leapfrog centering: state_n is the CENTER level X^n, not the level the
    # 2*dt step advances from -- use the leapfrog-scoped SI correction (the
    # forward/#920 trapezoidal centering would drop leapfrog to ~1st order and
    # make it weakly unstable).
    return si_correction(
        state_explicit, state_n, si_data, grid, dt2, predictor="leapfrog",
    )


def robert_asselin_filter(state_nm1, state_n, state_np1, gamma, alpha=0.53):
    """Robert-Asselin-Williams (RAW) time filter for leapfrog.

    The standard Robert-Asselin filter damps the computational mode
    (2·dt oscillation) that leapfrog permits, but introduces a
    first-order phase error.  The Williams (2009) modification splits
    the correction between the current and next time levels with
    OPPOSITE signs, restoring second-order accuracy while preserving
    the damping AND (at α = 0.5) conserving the three-time-level mean:

        d_n = (γ/2) · (X^{n-1} − 2·X^n + X^{n+1})
        X^n_filtered    = X^n    + α     · d_n
        X^{n+1}_filtered = X^{n+1} − (1 − α) · d_n

    With α = 0.5 the corrections cancel in the n + (n+1) sum (the
    "neutral" property), but Williams shows this choice is
    unconditionally unstable for the leapfrog amplitude factor;
    α ≈ 0.53 is the practical conditionally-stable choice and is
    used as the default here.  With α = 1 only X^n is modified and
    the original Robert-Asselin filter (3rd-order phase error) is
    recovered.

    NOTE — iter-54 fix: a prior implementation had both filter
    increments with the SAME sign and with α/(1−α) swapped:
        X^n_filtered    = X^n + (1 − α)·d_n
        X^{n+1}_filtered = X^{n+1} + α·d_n
    This does NOT preserve the three-time-level sum (sum changes by
    +d_n every step) and produces a slow climate-relevant drift
    toward the centered value at γ = 0.05.

    Parameters
    ----------
    state_nm1 : pytree
        State at time n-1 (already filtered from previous step).
    state_n : pytree
        State at time n (unfiltered).
    state_np1 : pytree
        State at time n+1 (just computed, unfiltered).
    gamma : float
        Filter coefficient (typically 0.05-0.2 — Williams 2009 uses
        γ ≈ 0.1 for most tests).
    alpha : float
        Williams parameter.  0.53 (default) is conditionally stable
        and gives near-optimal RAW behaviour for typical climate
        runs.  0.5 conserves the three-time-level mean exactly but is
        unconditionally unstable per Williams 2009 §3b.  1.0 recovers
        the original Robert-Asselin filter (no modification on n+1).

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
    - Williams, P. D. (2009). A proposed modification to the
      Robert-Asselin time filter.  Mon. Wea. Rev., 137, 2538-2546
      (eqs. 8-9 of the published paper give the canonical RAW form).
    """
    coeff = gamma / 2.0
    d_n = jax.tree.map(
        lambda xm, xn, xp: coeff * (xm - 2.0 * xn + xp),
        state_nm1, state_n, state_np1,
    )
    state_n_filtered = jax.tree.map(
        lambda xn, dn: xn + alpha * dn,
        state_n, d_n,
    )
    state_np1_filtered = jax.tree.map(
        lambda xp, dn: xp - (1.0 - alpha) * dn,
        state_np1, d_n,
    )
    return state_n_filtered, state_np1_filtered


# Pytree arithmetic shared with every other integrator (ssp_rk*, split_explicit):
# import the canonical helpers instead of re-defining them locally (redundancy
# audit).  Aliased to the existing private names so call sites are unchanged.
from legoesm.timestepping.pytree_ops import (
    pytree_axpy as _pytree_axpy,
    pytree_linear_combination as _pytree_linear_combination,
)
